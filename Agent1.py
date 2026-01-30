"""
PASSLLM-III: 基于 PII 和姊妹密码的定向密码猜测系统
使用 DeepSeek 大模型进行个性化密码生成
"""

from openai import OpenAI
from typing import List, Dict, Optional
import json
import os
from tqdm import tqdm


class PASSLLMIII:
    """
    PASSLLM-III 实现类
    利用用户个人信息(PII)和姊妹密码进行定向密码猜测
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", seed_file: Optional[str] = "dedup/clean.txt"):
        """
        初始化 PASSLLM-III

        Args:
            api_key: DeepSeek API key，如果不提供则使用默认值
            model: 使用的模型名称
        """
        # 默认 API Key
        self.api_key = api_key or "xxxxxxxxxxxxxxxxxxxxxxxxxx"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        # 可选：加载风格样例密码，以便模型“参考风格但不直接复用具体字符串”
        self.seed_examples: List[str] = self._load_seed_examples(seed_file)

    def _load_seed_examples(self, path: Optional[str]) -> List[str]:
        examples: List[str] = []
        if not path:
            return examples
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            examples.append(s)
            except Exception as e:
                print(f"读取样例文件失败: {e}")
        return examples

    def construct_prompt(self,
                        pii: Dict[str, str],
                        sibling_passwords: List[str],
                        num_candidates: int = 20,
                        style_examples: Optional[List[str]] = None) -> str:
        """
        构造输入提示词

        Args:
            pii: 用户个人信息字典，例如 {"name": "zhangsan", "birth": "1998"}
            sibling_passwords: 姊妹密码列表
            num_candidates: 需要生成的候选密码数量

        Returns:
            格式化的提示词
        """
        # 格式化 PII 信息
        pii_str = ", ".join([f"{k}={v}" for k, v in pii.items()])

        # 格式化姊妹密码
        sibling_str = ", ".join(sibling_passwords) if sibling_passwords else "无"

        extra_examples_block = ""
        if style_examples:
            show = style_examples[:min(40, len(style_examples))]
            extra_examples_block = "\n更多样例（仅参考风格，勿重复这些具体字符串）:\n" + "\n".join([f"- {x}" for x in show]) + "\n"

        prompt = f"""你是一个密码分析专家，请根据用户信息与历史密码生成候选口令。

用户信息: {pii_str}
历史密码: {sibling_str}

输出要求（严格遵守）:
- 只输出 Markdown 列表，共 {num_candidates} 行，每行仅一个最可能的密码
- 每行示例:
- hong11@lvtimeshow.com
- lady@zdress.com
- gatwickairport2020
- Hypojyfenc2e&
- 密码必须为单行且不含空格
{extra_examples_block}
"""

        return prompt

    def generate_passwords(self,
                          pii: Dict[str, str],
                          sibling_passwords: List[str] = None,
                          num_candidates: int = 20,
                          temperature: float = 0.8) -> List[Dict]:
        """
        生成密码候选

        Args:
            pii: 用户个人信息
            sibling_passwords: 姊妹密码列表
            num_candidates: 生成的候选数量
            temperature: 生成温度(0-1)，越高越随机

        Returns:
            密码候选列表，每项包含 password, confidence, reasoning
        """
        if sibling_passwords is None:
            sibling_passwords = []

        # 构造提示词
        prompt = self.construct_prompt(pii, sibling_passwords, num_candidates, style_examples=self.seed_examples)

        try:
            # 调用 DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个密码安全分析专家，擅长分析用户密码习惯和生成密码候选。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                stream=False
            )

            # 解析响应
            content = response.choices[0].message.content

            # 尝试提取 JSON
            candidates = self._extract_json(content)

            # 验证和清理结果
            validated_candidates = self._validate_candidates(candidates, num_candidates)

            # 若为空，尝试从 Markdown 抓取多行列表
            if not validated_candidates:
                md_candidates = []
                for line in content.splitlines():
                    s = line.strip()
                    if s.startswith('- ') or s.startswith('* '):
                        pwd = s[2:].strip().strip('`')
                        if pwd and (' ' not in pwd) and ('\t' not in pwd):
                            md_candidates.append({"password": pwd, "confidence": 0.6, "reasoning": "markdown-line"})
                            if len(md_candidates) >= num_candidates:
                                break
                if md_candidates:
                    validated_candidates = self._validate_candidates(md_candidates, num_candidates)

            return validated_candidates

        except Exception as e:
            print(f"生成密码时出错: {str(e)}")
            return []



    def _extract_json(self, content: str) -> List[Dict]:
        """从响应中提取 JSON 数据"""
        try:
            # 尝试直接解析
            if content.strip().startswith('['):
                return json.loads(content.strip())

            # 尝试提取 ```json ``` 代码块
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_str = content[start:end].strip()
                return json.loads(json_str)

            # 尝试提取 ``` ``` 代码块
            if '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                json_str = content[start:end].strip()
                return json.loads(json_str)

            # 如果都失败，尝试查找 [ 和 ]
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)

            return []

        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {str(e)}")
            print(f"原始内容: {content[:500]}")
            return []

    def _validate_candidates(self, candidates: List[Dict], expected_num: int) -> List[Dict]:
        """验证和标准化候选密码"""
        validated = []

        for item in candidates:
            if not isinstance(item, dict):
                continue

            # 确保必需字段存在
            if 'password' not in item:
                continue

            validated_item = {
                'password': str(item['password']),
                'confidence': float(item.get('confidence', 0.5)),
                'reasoning': str(item.get('reasoning', ''))
            }

            validated.append(validated_item)

        # 按置信度排序
        validated.sort(key=lambda x: x['confidence'], reverse=True)

        return validated[:expected_num]

    # ===== 新增：在线评测格式的 TXT 解析与生成 =====
    def _parse_online_txt_line(self, line: str) -> Dict[str, str]:
        """解析一行在线评测 TXT，返回 PII 字典。
        期望格式（\t 分隔）：
        email:...\tname:片段1|片段2\taccount:...\tphone:...\tbirth:...
        """
        fields = {"email": "", "name": "", "account": "", "phone": "", "birth": ""}
        parts = [p for p in line.strip().split('\t') if p.strip()]
        if len(parts) == 1:
            parts = [p for p in line.strip().split() if p.strip()]
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip().lower()
                v = v.strip()
                if k in fields:
                    fields[k] = v
        # 映射为 PII 键
        pii: Dict[str, str] = {}
        if fields["name"]:
            pii["name"] = fields["name"]
        if fields["birth"]:
            pii["birth"] = fields["birth"]
        if fields["account"]:
            pii["username"] = fields["account"]
        if fields["email"]:
            pii["email"] = fields["email"]
        if fields["phone"]:
            pii["phone"] = fields["phone"]
        return pii

    def write_answer_from_online(self, txt_file: str, output_answer: str, num_per_user: int = 1) -> None:
        """读取 online.txt，为每个用户生成 num_per_user 个密码（默认 1 个，Markdown 行），写入 answer.txt，并用 <END> 分隔。"""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"输入文件不存在: {txt_file}")

        lines: List[str] = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    lines.append(line.rstrip('\n'))

        with open(output_answer, 'w', encoding='utf-8') as out_f:
            for idx, raw in enumerate(tqdm(lines, desc="生成答案", unit="行")):
                pii = self._parse_online_txt_line(raw)

                collected: List[str] = []
                collected_set = set()
                attempts = 0
                max_attempts = 5
                while len(collected) < num_per_user and attempts < max_attempts:
                    need = num_per_user - len(collected)
                    # 将已收集的作为“避免重复的历史密码”提示，帮助模型多样化
                    sibling_for_diversity = collected[-50:]  # 限制长度避免提示过长
                    candidates = self.generate_passwords(
                        pii=pii,
                        sibling_passwords=sibling_for_diversity,
                        num_candidates=need,
                        temperature=0.85 if attempts == 0 else 0.95,
                    )
                    for item in candidates:
                        pwd = item.get('password', '')
                        if pwd and (pwd not in collected_set):
                            collected.append(pwd)
                            collected_set.add(pwd)
                            if len(collected) >= num_per_user:
                                break
                    attempts += 1

                print(f"\n目标 {idx + 1}: 共生成 {len(collected)} 条密码（尝试 {attempts} 轮）")
                for pwd in collected:
                    print(pwd)
                    out_f.write(pwd + "\n")
                out_f.write("<END>\n")
                # 立刻刷新并落盘，确保边生成边写入
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except Exception:
                    pass
        print(f"已写出答案文件: {output_answer}")


if __name__ == "__main__":
    in_file = "online.txt"
    out_file = "answer.txt"
    num_per_user = 100

    try:
        passllm = PASSLLMIII(seed_file="./dedup/clean.txt")
    except ValueError as e:
        print(f"错误: {e}")
        raise SystemExit(1)

    passllm.write_answer_from_online(in_file, out_file, num_per_user=num_per_user)