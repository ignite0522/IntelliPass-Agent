import sys
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from typing import List, Dict, Optional, Tuple, Set

from process_answerpro import generate_common_pass_variations

"""
基于大模型的遗传算法密码生成器

用法: python genetic_algorithm_llm.py <输入文件> <输出文件> <抽样比例> [选项]

示例:
  python genetic_algorithm_llm.py answer33.4.txt answer.txt 0.01 --log mutation.log --seed 55 --num-gen 50
  python genetic_algorithm_llm.py answer34.4.txt answer.txt 0.01 --log mutation.log --seed 55 --num-gen 50



注意:会自动加载当前目录下correct_passwords.txt文件作为示例
选项:
  --log <文件>     将详细日志保存到指定文件
  --seed <数字>    设置随机种子
  --num-gen <数字> 每次生成密码数量（默认20）
"""

"""
# 处理第401到500个用户
python genetic_algorithm_llm.py answer34.4.txt answer.txt 0.05 --users 400-500 --num-gen 50 --log mutation.log 

# 处理单个用户(第401个用户)
python genetic_algorithm_llm.py answer33.4.txt answer.txt 0.01 --users 401 --num-gen 50

# 处理所有用户（默认）
python genetic_algorithm_llm.py answer33.4.txt answer.txt 0.01 --num-gen 50
"""

# 大模型配置
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1afe57bab4a5434aae81b03c20ab9a34")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def parse_user_info(line: str) -> Dict[str, str]:
    """
    解析用户信息行
    格式: email:xxx name:xxx account:xxx phone:xxx birth:xxx
    """
    info = {}
    parts = line.strip().split('\t')
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            info[key.strip()] = value.strip()
    return info


def load_user_info(filepath: str) -> List[Dict[str, str]]:
    """加载用户信息"""
    users = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    users.append(parse_user_info(line))
        return users
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        sys.exit(1)


def load_correct_passwords(filepath: str = "correct_passwords.txt") -> List[str]:
    """
    加载正确密码示例
    如果文件不存在，返回空列表（将使用内置示例）
    """
    passwords = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    passwords.append(line)
        return passwords
    except FileNotFoundError:
        # 如果文件不存在，返回一些常见密码模式作为示例
        return []


def get_builtin_examples() -> List[str]:
    """获取内置的密码示例（常见密码模式）"""
    return [
        # 基础密码 + 数字
        "password123", "Password123", "PASSWORD123", "password1", "Password1",
        "admin123", "Admin123", "ADMIN123", "admin1", "Admin1",
        "user123", "User123", "USER123", "user1", "User1",
        "test123", "Test123", "TEST123", "test1", "Test1",
        "welcome123", "Welcome123", "WELCOME123", "welcome1",
        "hello123", "Hello123", "HELLO123", "hello1",
        "qwerty123", "Qwerty123", "QWERTY123", "qwerty1",
        # 纯数字
        "123456", "12345678", "123456789", "1234567890", "1234",
        # 基础密码 + 特殊字符
        "password!", "Password!", "PASSWORD!", "admin!", "Admin!",
        "user!", "User!", "test!", "Test!",
        # 组合密码
        "abc123", "Abc123", "ABC123", "abc!", "letmein", "Letmein", "LETMEIN",
        # 年份相关
        "password2024", "Password2024", "admin2024", "user2024",
        "password2023", "Password2023", "admin2023",
        # 常见变体
        "pass123", "Pass123", "PASS123", "pass1", "Pass1",
        "admin1234", "Admin1234", "user1234", "User1234",
    ]


def extract_user_tokens(user_info: Dict[str, str]) -> Set[str]:
    """
    从用户信息中提取可用于构造密码的关键词集合。
    包含：姓名片段、邮箱局部、域名、账号、电话片段、生日片段等。
    """
    tokens: Set[str] = set()

    def add_token(token: str):
        if not token:
            return
        token = token.strip()
        if not token:
            return
        tokens.add(token)
        tokens.add(token.lower())
        tokens.add(token.upper())
        tokens.add(token.capitalize())

    name = user_info.get("name", "")
    if name:
        for part in name.replace("|", " ").split():
            add_token(part)

    email = user_info.get("email", "")
    if email and "@" in email:
        local, _, domain = email.partition("@")
        add_token(local)
        if domain:
            for part in domain.replace(".", " ").replace("-", " ").split():
                add_token(part)

    account = user_info.get("account", "")
    if account:
        add_token(account)

    phone = user_info.get("phone", "")
    if phone:
        digits_only = re.sub(r"\D", "", phone)
        if digits_only:
            add_token(digits_only)
            if len(digits_only) >= 4:
                add_token(digits_only[-4:])
            if len(digits_only) >= 3:
                add_token(digits_only[:3])

    birth = user_info.get("birth", "")
    if birth:
        digits_only = re.sub(r"\D", "", birth)
        if digits_only:
            add_token(digits_only)
            if len(digits_only) >= 4:
                add_token(digits_only[:4])
            if len(digits_only) >= 6:
                add_token(digits_only[:6])
            if len(digits_only) >= 8:
                add_token(digits_only[:8])
            if len(digits_only) >= 2:
                add_token(digits_only[-2:])

    return {t for t in tokens if t}


def generate_fallback_variants(original: str,
                               user_tokens: Set[str],
                               example_pool: List[str],
                               max_variants: int = 120) -> List[str]:
    """
    针对某个原始密码生成一批候选备选变体，用于 LLM 输出不足或不理想时的补充。
    返回去重后的列表，保持一定顺序。
    """
    variants: List[str] = []

    def add_variant(candidate: str):
        if not candidate:
            return
        if candidate not in variants:
            variants.append(candidate)

    # 基础变体（利用现有的变体生成器）
    try:
        base_variants = generate_common_pass_variations(original)
    except Exception:
        base_variants = []
    for var in base_variants:
        if var != original:
            add_variant(var)

    # 原始密码的字母和数字片段
    alpha_part = re.sub(r"[^A-Za-z]", "", original)
    digit_parts = re.findall(r"\d+", original)

    # 基于用户 tokens 的组合
    suffixes = ["", "123", "2024", "2025", "!", "@", "#", "$"]
    for token in sorted(user_tokens, key=len, reverse=True):
        base = token
        add_variant(base)
        for suf in suffixes:
            add_variant(base + suf)
            add_variant(suf + base)
        if digit_parts:
            for d in digit_parts:
                add_variant(base + d)
                add_variant(d + base)
        if alpha_part:
            add_variant(base + alpha_part)
            add_variant(alpha_part + base)

    # 结合正确密码示例的尾部/头部片段
    for example in example_pool[:10]:
        ex = example.strip()
        if not ex:
            continue
        add_variant(ex)
        if alpha_part:
            add_variant(alpha_part + ex[-2:])
            add_variant(ex[:2] + alpha_part)
        for token in list(user_tokens)[:5]:
            add_variant(token + ex)
            add_variant(ex + token)

    # 若数字部分缺失，补充常见数字组合
    common_digits = ["123", "321", "007", "000", "789", "369", "520", "1314"]
    for token in list(user_tokens)[:8]:
        for digits in common_digits:
            add_variant(f"{token}{digits}")
            add_variant(f"{digits}{token}")
    if alpha_part:
        for digits in common_digits:
            add_variant(f"{alpha_part}{digits}")
            add_variant(f"{digits}{alpha_part}")

    if len(variants) > max_variants:
        variants = variants[:max_variants]
    return variants


def is_valid_candidate(candidate: str,
                       original: str,
                       existing_new: Set[str],
                       existing_originals: Set[str]) -> bool:
    """判断候选密码是否符合要求。"""
    if not candidate:
        return False
    candidate = candidate.strip()
    if candidate == original:
        return False
    if candidate in existing_new:
        return False
    if candidate in existing_originals:
        return False
    if len(candidate) < 4 or len(candidate) > 50:
        return False
    if not re.match(r"^[A-Za-z0-9!@#$%^&*._\-]+$", candidate):
        return False
    return True


def choose_candidate(original: str,
                     llm_candidate: Optional[str],
                     user_tokens: Set[str],
                     example_pool: List[str],
                     existing_new: Set[str],
                     existing_originals: Set[str],
                     fallback_cache: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    为原始密码选择一个最终候选。
    优先使用 LLM 返回的候选；若无效，则从 fallback 生成。
    返回 (candidate, source) source in {"llm", "fallback", "original"}
    """
    options: List[str] = []
    if llm_candidate:
        options.append(llm_candidate.strip())

    if original not in fallback_cache:
        fallback_cache[original] = generate_fallback_variants(original, user_tokens, example_pool)
    options.extend(fallback_cache[original])

    for cand in options:
        if is_valid_candidate(cand, original, existing_new, existing_originals):
            existing_new.add(cand)
            if llm_candidate and cand == llm_candidate.strip():
                return cand, "llm"
            else:
                return cand, "fallback"
    return original, "original"


def build_prompt(user_info: Dict[str, str], current_passwords: List[str],
                 correct_examples: List[str], selected_passwords: List[str],
                 num_generate: int = 20) -> str:
    """
    构建大模型提示词

    Args:
        user_info: 用户信息
        current_passwords: 当前用户的所有密码列表
        correct_examples: 正确密码示例
        selected_passwords: 当前批次选中的密码（作为信息源）
        num_generate: 需要生成的密码数量
    """
    # 提取用户信息的关键部分
    name_parts = user_info.get('name', '').split('|') if user_info.get('name') else []
    email = user_info.get('email', '')
    account = user_info.get('account', '')
    phone = user_info.get('phone', '')
    birth = user_info.get('birth', '')

    # 构建用户信息描述
    user_context = []
    if name_parts:
        user_context.append(f"姓名: {' '.join(name_parts)}")
    if email:
        user_context.append(f"邮箱: {email}")
    if account:
        user_context.append(f"账号: {account}")
    if phone:
        user_context.append(f"电话: {phone}")
    if birth:
        user_context.append(f"生日: {birth}")

    # 当前密码列表（显示前20个作为参考）
    password_context = "\n".join([f"  - {pwd}" for pwd in current_passwords[:20]])
    if len(current_passwords) > 20:
        password_context += f"\n  ... 还有 {len(current_passwords) - 20} 个密码"

    # 正确密码示例（显示前30个）
    example_context = "\n".join(correct_examples[:30])
    if len(correct_examples) > 30:
        example_context += f"\n... 还有更多示例"

    prompt = f"""你是一个密码生成专家。请根据用户信息和现有密码列表，生成 {num_generate} 个高质量且多样化的密码变体（每个新密码都应当与其对应的输入密码存在实质差异，而不是仅仅在末尾加一个数字）。

用户信息：
{chr(10).join(user_context) if user_context else '  无额外信息'}

当前用户的密码列表（部分）：
{password_context}

正确密码示例（参考这些模式）：
{example_context}

生成要求：
1. 信息来源不变：仅基于
   - 该用户信息（姓名/账号/邮箱/电话/生日等的片段与组合）
   - 当前用户已有密码模式（上面“当前用户的密码列表（部分）”）
   - 正确密码示例中的常见结构
2. 提升多样性与实用性，覆盖不同“模式族”的混合分布（不要集中在同一种改法）：
   - 大小写混排/CamelCase/全大写/全小写（如 anderson -> AndErSoN / ANDERSON）
   - 数字前后缀与插入（如 1anderson / anderson1 / ande123rson）
   - 年份/日期/手机号片段/生日片段（如 anderson1985 / anderson0601）
   - 特殊符号的合理使用与分隔（如 anderson! / anderson@ / anderson_1985 / anderson-85）
   - Leet/近形替换（如 a->4, e->3, o->0, s->5, i->1）
   - 组合变体（姓名片段+账号/邮箱本地部分/域名片段/公司或邮箱后缀片段）
   - 轻微键盘序列/重复字符的克制使用（如 anderson!!, anderson@@, anderson##, 但不要全部都用）
3. 每个输出必须与对应输入密码存在“实质差异”（至少一种以上的变换组合），避免仅加单一后缀或只改一个字符。
4. 长度范围建议在 4-30 字符之间；整体输出在短/中/长长度上有分布（不要全部长度相近）。
5. 严格避免输出完全重复；同一批次中尽量不产生相同形态（如仅大小写不同也视为低差异，少量即可）。
6. 允许包含以下字符集：字母、数字、常见符号（!@#$%^&*._-），不允许空格或不可见字符。
7. 优先贴合用户信息（姓名片段、邮箱本地部分、账号、生日等），但避免泄露完整隐私（如完整电话号码可只取局部片段）。

重要：请严格输出 {num_generate} 行，每行一个新密码。输出顺序必须稳定一致（我们会按内部索引一一对应进行替换），不要输出任何解释或编号。

请直接输出 {num_generate} 个密码，每行一个，不要编号，不要解释，只输出密码本身。
"""

    return prompt


def generate_passwords_with_llm(prompt: str, num_generate: int = 20, user_idx: int = 0, batch_idx: int = 0) -> Tuple[
    List[str], float]:
    """
    使用大模型生成密码（线程安全）

    Returns:
        (生成的密码列表, 耗时)
    """
    llm_start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个密码生成专家，擅长根据用户信息生成可能的密码变体。请直接输出密码，每行一个，不要任何解释、编号或标记。请根据你能获得的信息尽量生成相对复杂的密码"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_tokens=2000
        )

        # 解析响应
        content = response.choices[0].message.content.strip()

        # 提取密码（每行一个）
        passwords = []
        for line in content.split('\n'):
            line = line.strip()
            # 移除可能的编号（如 "1. ", "1) ", "(1) " 等）
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^\(\d+\)\s*', '', line)
            # 移除可能的标记符号
            line = re.sub(r'^[-*•]\s*', '', line)
            # 移除可能的引号
            line = line.strip('"\'')
            # 只保留非空且合理的密码（3-50字符）
            if line and len(line) >= 3 and len(line) <= 50:
                # 去重
                if line not in passwords:
                    passwords.append(line)

        # 如果生成的密码不够，尝试补充（可以添加一些变体）
        if len(passwords) < num_generate:
            # 对现有密码进行简单变体生成
            additional = []
            for pwd in passwords[:min(10, len(passwords))]:
                # 添加大小写变体
                if pwd.lower() not in passwords and pwd.lower() not in additional:
                    additional.append(pwd.lower())
                if pwd.upper() not in passwords and pwd.upper() not in additional:
                    additional.append(pwd.upper())
                if pwd.capitalize() not in passwords and pwd.capitalize() not in additional:
                    additional.append(pwd.capitalize())

            passwords.extend(additional[:num_generate - len(passwords)])

        llm_time = time.time() - llm_start_time
        return (passwords[:num_generate], llm_time)  # 限制返回数量

    except Exception as e:
        llm_time = time.time() - llm_start_time
        print(f"警告: 用户{user_idx}批次{batch_idx + 1}大模型生成失败: {e}")
        import traceback
        traceback.print_exc()
        return ([], llm_time)


def sample_passwords(passwords: List[str], sample_ratio: float) -> set:
    """
    按比例抽样密码（保留原有抽样逻辑）
    """
    if sample_ratio == 0.0:
        return set()

    sample_count = max(1, min(len(passwords), int(len(passwords) * sample_ratio)))
    return set(random.sample(range(len(passwords)), sample_count))


def parse_user_range(user_range_str: str) -> Optional[tuple]:
    """
    解析用户范围字符串，格式: "start-end" 或 "start"

    Args:
        user_range_str: 用户范围字符串，如 "401-500" 或 "401"

    Returns:
        (start, end) 元组，如果解析失败返回None
    """
    try:
        if '-' in user_range_str:
            parts = user_range_str.split('-')
            if len(parts) != 2:
                return None
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if start < 1 or end < 1 or start > end:
                return None
            return (start, end)
        else:
            # 单个用户编号
            user_num = int(user_range_str.strip())
            if user_num < 1:
                return None
            return (user_num, user_num)
    except ValueError:
        return None


def process_file(input_file: str, output_file: str, sample_ratio: float,
                 log_file: Optional[str] = None, num_generate: int = 20,
                 seed: Optional[int] = None, user_range: Optional[tuple] = None,
                 concurrency: int = 10):
    """
    处理文件，使用大模型生成替换密码

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_ratio: 抽样比例
        log_file: 日志文件路径
        num_generate: 每次生成密码数量
        seed: 随机种子
        user_range: 用户范围 (start, end)，None表示处理所有用户
        concurrency: 并发数量（默认10）
    """
    if seed is not None:
        random.seed(seed)
        print(f"随机种子: {seed}")

    # 加载用户信息
    print("正在加载用户信息...")
    users = load_user_info("online.txt")
    print(f"已加载 {len(users)} 个用户信息")

    # 加载正确密码示例
    print("正在加载正确密码示例...")
    correct_passwords = load_correct_passwords()
    if not correct_passwords:
        correct_passwords = get_builtin_examples()
        print("使用内置密码示例")
    else:
        print(f"已加载 {len(correct_passwords)} 个正确密码示例")

    # 读取输入文件
    print(f"正在读取输入文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)

    # 按<END>分割成每个用户的密码列表
    target_lists = content.split('<END>')
    total_users = len(target_lists)

    # 确定需要处理的用户范围
    if user_range:
        start_user, end_user = user_range
        # 验证用户范围是否在有效范围内
        if start_user > total_users:
            print(f"错误: 起始用户编号 {start_user} 超过总用户数 {total_users}")
            sys.exit(1)
        if end_user > total_users:
            print(f"警告: 结束用户编号 {end_user} 超过总用户数 {total_users}，将调整为 {total_users}")
            end_user = total_users

        # 转换为0-based索引，并确保在有效范围内
        start_idx = max(0, start_user - 1)
        end_idx = min(total_users, end_user)
        process_range = (start_idx, end_idx)
        range_size = end_user - start_user + 1
        print(f"用户范围: 第 {start_user} 到 {end_user} 个用户 (共 {range_size} 个用户)")
        # 显示其他用户范围
        other_ranges = []
        if start_user > 1:
            other_ranges.append(f"1-{start_user - 1}")
        if end_user < total_users:
            other_ranges.append(f"{end_user + 1}-{total_users}")
        if other_ranges:
            print(f"其他用户 ({', '.join(other_ranges)}) 将保持不变")
        else:
            print(f"所有用户都在处理范围内")
    else:
        process_range = (0, total_users)
        start_user = 1
        end_user = total_users
        print(f"处理所有用户")

    print(f"开始处理 {total_users} 个用户的密码列表...")
    print(f"抽样比例: {sample_ratio * 100:.1f}%")
    print(f"每次生成密码数: {num_generate}")
    print(f"并发数量: {concurrency}")
    print()

    total_mutated_count = 0
    mutation_logs = []
    processed_users = 0  # 实际处理的用户数

    # 进度跟踪
    start_time = time.time()

    # 初始化输出文件（清空或创建）
    print(f"初始化输出文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # 创建空文件

    # 用于跟踪是否是第一个写入的用户
    is_first_user = True

    for user_idx, target_content in enumerate(target_lists, 1):
        # 检查是否需要处理这个用户（user_idx是1-based）
        if user_range and (user_idx < start_user or user_idx > end_user):
            # 不在处理范围内，直接复制原密码并立即写入
            lines = target_content.strip().split('\n')
            passwords = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#')[0].strip()
                if line:
                    passwords.append(line)

            # 立即写入到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                if not is_first_user:
                    f.write('<END>\n')
                for pwd in passwords:
                    f.write(pwd + '\n')
                is_first_user = False

            # 显示跳过信息（每50个用户显示一次）
            if user_idx % 50 == 0:
                print(f"跳过用户 {user_idx}/{total_users} (不在处理范围内，已保存)")
            continue

        # 在处理范围内的用户，进行正常处理
        processed_users += 1
        lines = target_content.strip().split('\n')

        # 解析每一行，去除注释
        passwords = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                passwords.append(line)

        if len(passwords) == 0:
            # 空密码列表，写入空内容
            with open(output_file, 'a', encoding='utf-8') as f:
                if not is_first_user:
                    f.write('<END>\n')
                is_first_user = False
            continue

        # 获取当前用户信息
        user_info = users[user_idx - 1] if user_idx <= len(users) else {}

        # 抽样
        sampled_indices = sample_passwords(passwords, sample_ratio)
        total_mutated_count += len(sampled_indices)

        if len(sampled_indices) == 0:
            # 没有抽中密码，直接写入原密码
            with open(output_file, 'a', encoding='utf-8') as f:
                if not is_first_user:
                    f.write('<END>\n')
                for pwd in passwords:
                    f.write(pwd + '\n')
                is_first_user = False
            continue

        # 为抽中的密码生成替换密码
        # 显示总体进度（每个用户都显示）
        elapsed_time = time.time() - start_time

        if user_range:
            # 如果指定了用户范围，显示范围内的进度
            range_size = end_user - start_user + 1
            range_progress = (processed_users / range_size * 100) if range_size > 0 else 0

            if processed_users > 1:
                avg_time_per_user = elapsed_time / (processed_users - 1)
                remaining_in_range = range_size - processed_users
                estimated_remaining = avg_time_per_user * remaining_in_range
                print(f"\n{'=' * 60}")
                print(
                    f"[总体进度] 用户 {user_idx}/{total_users} | 范围内进度: {processed_users}/{range_size} ({range_progress:.1f}%) | "
                    f"已替换: {len(mutation_logs)} 个密码 | "
                    f"已用时间: {elapsed_time:.1f}s | "
                    f"预计剩余: {estimated_remaining:.1f}s ({estimated_remaining / 60:.1f}分钟)")
                print(f"{'=' * 60}\n")
            else:
                print(f"\n{'=' * 60}")
                print(
                    f"[总体进度] 用户 {user_idx}/{total_users} | 范围内进度: {processed_users}/{range_size} ({range_progress:.1f}%) | "
                    f"已替换: {len(mutation_logs)} 个密码 | "
                    f"已用时间: {elapsed_time:.1f}s")
                print(f"{'=' * 60}\n")
        else:
            # 处理所有用户
            progress_percent = (user_idx / total_users) * 100

            if user_idx > 1:
                avg_time_per_user = elapsed_time / (user_idx - 1)
                remaining_users = total_users - user_idx
                estimated_remaining = avg_time_per_user * remaining_users
                print(f"\n{'=' * 60}")
                print(f"[总体进度] {user_idx}/{total_users} ({progress_percent:.1f}%) | "
                      f"已替换: {len(mutation_logs)} 个密码 | "
                      f"已用时间: {elapsed_time:.1f}s | "
                      f"预计剩余: {estimated_remaining:.1f}s ({estimated_remaining / 60:.1f}分钟)")
                print(f"{'=' * 60}\n")
            else:
                print(f"\n{'=' * 60}")
                print(f"[总体进度] {user_idx}/{total_users} ({progress_percent:.1f}%) | "
                      f"已替换: {len(mutation_logs)} 个密码 | "
                      f"已用时间: {elapsed_time:.1f}s")
                print(f"{'=' * 60}\n")

        print(f"用户 {user_idx}/{total_users}: 抽中 {len(sampled_indices)} 个密码，将分批次处理...")
        user_tokens = extract_user_tokens(user_info)
        example_pool = correct_passwords[:50] if correct_passwords else []

        # 将选中的密码索引转换为列表并排序，以便按顺序处理
        sampled_indices_list = sorted(list(sampled_indices))

        # 获取选中的密码列表
        selected_passwords_list = [passwords[idx] for idx in sampled_indices_list]

        # 按 num_generate 分批处理
        num_batches = (len(sampled_indices_list) + num_generate - 1) // num_generate  # 向上取整
        print(f"  将分 {num_batches} 个批次处理，每批最多 {num_generate} 个密码")

        # 创建替换映射：索引 -> 新密码
        replacement_map = {}

        # 准备批次任务（用于并发处理）
        batch_tasks = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * num_generate
            end_idx = min(start_idx + num_generate, len(sampled_indices_list))
            batch_indices = sampled_indices_list[start_idx:end_idx]
            batch_selected_passwords = selected_passwords_list[start_idx:end_idx]
            batch_size = len(batch_indices)

            # 构建提示词（传入当前批次的选中密码）
            prompt = build_prompt(user_info, passwords, correct_passwords,
                                  batch_selected_passwords, batch_size)

            batch_tasks.append({
                'batch_idx': batch_idx,
                'batch_indices': batch_indices,
                'batch_selected_passwords': batch_selected_passwords,
                'batch_size': batch_size,
                'prompt': prompt
            })

        print(f"  启动 {num_batches} 个批次，并发数: {concurrency}")

        # 使用线程池并发处理批次
        batch_results = {}  # batch_idx -> (generated_passwords, llm_time)
        log_lock = Lock()  # 用于保护日志写入

        def process_batch(task):
            """处理单个批次的任务函数"""
            batch_idx = task['batch_idx']
            batch_indices = task['batch_indices']
            batch_selected_passwords = task['batch_selected_passwords']
            batch_size = task['batch_size']
            prompt = task['prompt']

            print(f"  [并发] 批次 {batch_idx + 1}/{num_batches}: 开始处理 {batch_size} 个密码...")

            # 调用大模型生成密码
            generated_passwords, llm_time = generate_passwords_with_llm(
                prompt, batch_size, user_idx, batch_idx)

            with log_lock:
                print(
                    f"  [并发] 批次 {batch_idx + 1}/{num_batches}: 完成，耗时 {llm_time:.2f}s，生成 {len(generated_passwords)} 个密码")

            return batch_idx, generated_passwords, llm_time, batch_indices, batch_selected_passwords

        # 并发执行批次任务
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(process_batch, task): task['batch_idx']
                for task in batch_tasks
            }

            # 收集结果（按批次索引顺序）
            completed_batches = {}
            for future in as_completed(future_to_batch):
                batch_idx, generated_passwords, llm_time, batch_indices, batch_selected_passwords = future.result()
                completed_batches[batch_idx] = (generated_passwords, llm_time, batch_indices, batch_selected_passwords)

        # 按批次索引顺序处理结果（确保顺序正确）
        fallback_cache: Dict[str, List[str]] = {}
        existing_originals = set(passwords)
        existing_new_candidates: Set[str] = set()

        for batch_idx in sorted(completed_batches.keys()):
            generated_passwords, llm_time, batch_indices, batch_selected_passwords = completed_batches[batch_idx]
            batch_size = len(batch_indices)

            print(f"  处理批次 {batch_idx + 1}/{num_batches} 的结果...")

            if not generated_passwords:
                print(f"    警告: 批次 {batch_idx + 1} 生成失败，保持原密码")
                # 如果生成失败，保持原密码
                for idx in batch_indices:
                    replacement_map[idx] = passwords[idx]
                continue

            # 为当前批次的每个密码分配生成的新密码（严格按照顺序对应），不足则使用 fallback
            for i, idx in enumerate(batch_indices):
                original_password = batch_selected_passwords[i]  # 使用批次中的密码，确保顺序一致
                llm_candidate = generated_passwords[i].strip() if i < len(generated_passwords) else None

                assert passwords[idx] == original_password, f"密码索引不匹配: idx={idx}, batch_idx={i}"

                candidate, source = choose_candidate(
                    original_password,
                    llm_candidate,
                    user_tokens,
                    example_pool,
                    existing_new_candidates,
                    existing_originals,
                    fallback_cache
                )

                batch_position = i + 1
                if candidate != original_password:
                    replacement_map[idx] = candidate
                    log_entry = (f"用户{user_idx}的第{idx + 1}个密码(批次{batch_idx + 1}第{batch_position}个): "
                                 f"'{original_password}' -> '{candidate}' (来源: {source})")
                    with log_lock:
                        mutation_logs.append(log_entry)
                        print(
                            f"    [{batch_idx + 1}-{batch_position}] 第{idx + 1}个密码: '{original_password}' -> '{candidate}' ({source})")
                else:
                    replacement_map[idx] = original_password
                    with log_lock:
                        print(
                            f"    [{batch_idx + 1}-{batch_position}] 第{idx + 1}个密码: '{original_password}' (未生成有效变体，保持原密码)")

        # 构建最终的密码列表
        mutated_passwords = []
        for idx, password in enumerate(passwords):
            if idx in sampled_indices:
                # 使用替换映射中的新密码
                mutated_passwords.append(replacement_map.get(idx, password))
            else:
                # 保持原密码不变
                mutated_passwords.append(password)

        # 对该用户内部的密码进行去重（保持顺序）
        original_count = len(mutated_passwords)
        seen = set()
        deduplicated_passwords = []
        duplicates_removed = []

        for pwd in mutated_passwords:
            if pwd not in seen:
                seen.add(pwd)
                deduplicated_passwords.append(pwd)
            else:
                duplicates_removed.append(pwd)

        # 更新为去重后的密码列表
        mutated_passwords = deduplicated_passwords
        removed_count = original_count - len(mutated_passwords)

        if removed_count > 0:
            print(
                f"  用户 {user_idx} 内部去重: 移除 {removed_count} 个重复密码 (原 {original_count} -> 现 {len(mutated_passwords)})")
            if duplicates_removed:
                # 显示部分被移除的重复密码（最多显示5个）
                sample_duplicates = duplicates_removed[:5]
                sample_str = ', '.join([f"'{d}'" for d in sample_duplicates])
                print(f"    移除的重复密码示例: {sample_str}")
                if len(duplicates_removed) > 5:
                    print(f"    ... 还有 {len(duplicates_removed) - 5} 个重复密码")

        # 立即写入当前用户的密码到文件
        with open(output_file, 'a', encoding='utf-8') as f:
            if not is_first_user:
                f.write('<END>\n')
            for pwd in mutated_passwords:
                f.write(pwd + '\n')
            is_first_user = False

        print(f"  用户 {user_idx} 已保存到文件")

        # 输出详细进度（每10个处理的用户或最后一个用户）
        if processed_users % 10 == 0 or user_idx == end_user:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if user_range:
                # 计算在处理范围内的进度
                range_size = end_user - start_user + 1
                range_progress = processed_users / range_size * 100 if range_size > 0 else 0
                remaining_in_range = range_size - processed_users

                print(f"\n{'=' * 60}")
                print(f"[详细进度报告]")
                print(f"  用户范围进度: {processed_users}/{range_size} ({range_progress:.1f}%)")
                print(f"  当前用户: {user_idx}/{total_users}")
                print(f"  已替换密码数: {len(mutation_logs)}")
                print(f"  已用时间: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")
                if processed_users > 0:
                    avg_time_per_user = elapsed_time / processed_users
                    users_per_second = processed_users / elapsed_time if elapsed_time > 0 else 0
                    print(f"  处理速度: {users_per_second:.2f} 用户/秒")
                    if remaining_in_range > 0:
                        estimated_remaining = avg_time_per_user * remaining_in_range
                        print(f"  预计剩余时间: {estimated_remaining:.1f}秒 ({estimated_remaining / 60:.1f}分钟)")
                print(f"{'=' * 60}\n")
            else:
                # 处理所有用户
                progress_percent = (user_idx / total_users) * 100
                if user_idx > 0:
                    avg_time_per_user = elapsed_time / user_idx
                    remaining_users = total_users - user_idx
                    estimated_remaining = avg_time_per_user * remaining_users
                    users_per_second = user_idx / elapsed_time if elapsed_time > 0 else 0

                    print(f"\n{'=' * 60}")
                    print(f"[详细进度报告]")
                    print(f"  总体进度: {user_idx}/{total_users} ({progress_percent:.1f}%)")
                    print(f"  已替换密码数: {len(mutation_logs)}")
                    print(f"  已用时间: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")
                    print(f"  处理速度: {users_per_second:.2f} 用户/秒")
                    if remaining_users > 0:
                        print(f"  预计剩余时间: {estimated_remaining:.1f}秒 ({estimated_remaining / 60:.1f}分钟)")
                    print(f"{'=' * 60}\n")

    # 输出统计信息
    # 重新读取输出文件统计密码数量（因为已经实时写入了）
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_content = f.read()
        output_lists = output_content.split('<END>')
        total_passwords = sum(len([l for l in lst.strip().split('\n') if l.strip()]) for lst in output_lists)
    except:
        # 如果读取失败，使用处理过的用户数估算
        total_passwords = 0
    actual_mutated = len(mutation_logs)

    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"{'=' * 60}")
    print(f"总用户数: {total_users}")
    if user_range:
        print(f"处理用户范围: 第 {start_user} 到 {end_user} 个用户 (共 {processed_users} 个用户)")
        print(f"其他用户: 保持不变")
    else:
        print(f"处理用户数: {processed_users}")
    print(f"总密码数: {total_passwords}")
    print(f"抽样密码数: {total_mutated_count}")
    print(f"实际替换数: {actual_mutated}")
    if total_passwords > 0:
        print(f"抽样比例: {total_mutated_count / total_passwords * 100:.2f}%")
        print(f"实际替换比例: {actual_mutated / total_passwords * 100:.2f}%")
    print(f"输出文件: {output_file}")

    # 保存日志到文件
    if log_file:
        print(f"\n正在保存日志到: {log_file}")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"大模型遗传算法运行日志\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"输出文件: {output_file}\n")
            f.write(f"抽样比例: {sample_ratio * 100:.1f}%\n")
            f.write(f"每次生成密码数: {num_generate}\n")
            f.write(f"总用户数: {total_users}\n")
            if user_range:
                f.write(f"处理用户范围: 第 {start_user} 到 {end_user} 个用户 (共 {processed_users} 个用户)\n")
            else:
                f.write(f"处理用户数: {processed_users}\n")
            f.write(f"总密码数: {total_passwords}\n")
            f.write(f"抽样密码数: {total_mutated_count}\n")
            f.write(f"实际替换数: {actual_mutated}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"详细替换记录:\n")
            f.write(f"{'=' * 60}\n")
            for log_entry in mutation_logs:
                f.write(log_entry + '\n')
        print(f"日志已保存到: {log_file}")
    else:
        # 如果没有指定日志文件，也在控制台输出总结
        print(f"\n{'=' * 60}")
        print(f"替换记录总结 (共 {actual_mutated} 条):")
        print(f"{'=' * 60}")
        if actual_mutated > 0:
            print("前10条替换记录:")
            for log_entry in mutation_logs[:10]:
                print(f"  {log_entry}")
            if actual_mutated > 10:
                print(f"  ... 还有 {actual_mutated - 10} 条记录未显示")
                print(f"  使用 --log <文件名> 参数可保存完整日志到文件")
        else:
            print("  没有密码被替换")

    print(f"{'=' * 60}\n")


def main():
    if len(sys.argv) < 4:
        print("用法: python genetic_algorithm_llm.py <输入文件> <输出文件> <抽样比例> [选项]")
        print(
            "示例: python genetic_algorithm_llm.py answer.txt output.txt 0.005 --log mutation.log --seed 40 --num-gen 20")
        print("示例: python genetic_algorithm_llm.py answer.txt output.txt 0.01 --users 401-500 --num-gen 50")
        print("\n选项:")
        print("  --log <文件>     将详细日志保存到指定文件")
        print("  --seed <数字>    设置随机种子")
        print("  --num-gen <数字> 每次生成密码数量（默认20）")
        print("  --users <范围>   指定处理的用户范围，格式: start-end (如 401-500) 或单个用户 (如 401)")
        print("                   默认处理所有用户")
        print("  --concurrency <数字> 并发请求数量（默认10，范围1-50）")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        sample_ratio = float(sys.argv[3])
        if sample_ratio < 0.0 or sample_ratio > 1.0:
            print("错误: 抽样比例必须在 0.0 到 1.0 之间")
            sys.exit(1)
    except ValueError:
        print("错误: 抽样比例必须是数字")
        sys.exit(1)

    # 解析选项
    log_file = None
    seed = None
    num_generate = 20
    user_range = None
    concurrency = 10  # 默认并发数

    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == '--log' and i + 1 < len(sys.argv):
            log_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--seed' and i + 1 < len(sys.argv):
            try:
                seed = int(sys.argv[i + 1])
            except ValueError:
                print("错误: 种子必须是整数")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--num-gen' and i + 1 < len(sys.argv):
            try:
                num_generate = int(sys.argv[i + 1])
                if num_generate < 1 or num_generate > 100:
                    print("错误: 生成数量必须在 1 到 100 之间")
                    sys.exit(1)
            except ValueError:
                print("错误: 生成数量必须是整数")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--users' and i + 1 < len(sys.argv):
            user_range_str = sys.argv[i + 1]
            user_range = parse_user_range(user_range_str)
            if user_range is None:
                print(f"错误: 用户范围格式错误: {user_range_str}")
                print("  正确格式: start-end (如 401-500) 或单个用户 (如 401)")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--concurrency' and i + 1 < len(sys.argv):
            try:
                concurrency = int(sys.argv[i + 1])
                if concurrency < 1 or concurrency > 50:
                    print("错误: 并发数必须在 1 到 50 之间")
                    sys.exit(1)
            except ValueError:
                print("错误: 并发数必须是整数")
                sys.exit(1)
            i += 2
        else:
            i += 1

    process_file(input_file, output_file, sample_ratio, log_file, num_generate, seed, user_range, concurrency)


if __name__ == '__main__':
    main()

