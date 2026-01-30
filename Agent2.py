#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
密码爆破Agent
根据用户信息生成爆破密码列表
"""

import os
import sys
import re
import json
import traceback
import time
from typing import Dict, List, Set, Optional
from openai import OpenAI
from openai import APITimeoutError, APIError, APIConnectionError
from tqdm import tqdm
import httpx
from httpx import ReadTimeout as HttpxReadTimeout

# API配置
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")

# 目标数量
TARGET_PASSWORDS_PER_USER = 2000  # 每个用户生成2000个不重复密码
BATCH_SIZE = 1500  # 每次生成的数量（增加以提高效率）

# 超时和重试配置
API_TIMEOUT = 300  # API调用超时时间（秒）- 增加以应对大响应和流式处理
STREAM_READ_TIMEOUT = 60  # 流式处理中单个读取操作的超时时间（秒）
MAX_RETRIES = 5  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

# 输出文件
ANSWER_FILE = "answer.txt"

# 常用弱密码列表
COMMON_WEAK_PASSWORDS = [
    "Password", "Admin", "Welcome", "123456", "password", "admin",
    "123", "1234", "12345", "123456", "1234567", "12345678", "123456789",
    "000", "111", "222", "333", "444", "555", "666", "777", "888", "999"
]

# 特殊字符
SPECIAL_CHARS = "!@#$%&*"
SEPARATORS = "._-"

# 近年年份
RECENT_YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


def parse_user_info(line: str) -> Dict[str, str]:
    """解析用户信息"""
    user = {
        "email": "",
        "name": "",
        "account": "",
        "phone": "",
        "birth": ""
    }

    # 提取email
    email_match = re.search(r'email:([^\t]*)', line)
    if email_match:
        user["email"] = email_match.group(1).strip()

    # 提取name
    name_match = re.search(r'name:([^\t]*)', line)
    if name_match:
        user["name"] = name_match.group(1).strip()

    # 提取account
    account_match = re.search(r'account:([^\t]*)', line)
    if account_match:
        user["account"] = account_match.group(1).strip()

    # 提取phone
    phone_match = re.search(r'phone:([^\t]*)', line)
    if phone_match:
        user["phone"] = phone_match.group(1).strip()

    # 提取birth
    birth_match = re.search(r'birth:([^\t]*)', line)
    if birth_match:
        user["birth"] = birth_match.group(1).strip()

    return user


def extract_user_elements(user: Dict[str, str]) -> Dict[str, List[str]]:
    """提取用户信息的各个元素"""
    elements = {
        "account": [],
        "name_parts": [],
        "birth_parts": [],
        "phone_parts": [],
        "email_parts": []
    }

    # 账号信息
    if user["account"]:
        account = user["account"].lower()
        elements["account"].extend([account, account.upper(), account.capitalize()])

    # 姓名信息
    if user["name"]:
        name = user["name"].replace("|", "").replace("-", "").replace(".", "").strip()
        name_lower = name.lower()
        # 提取姓名前缀（如R）
        name_parts = re.split(r'[|\s]+', name)
        for part in name_parts:
            if part and len(part) <= 3:
                elements["name_parts"].append(part)
        # 提取主要用户名部分
        if name_lower:
            elements["name_parts"].extend([name_lower, name_lower.upper(), name_lower.capitalize()])

    # 生日信息
    if user["birth"]:
        birth = user["birth"]
        if len(birth) == 8:  # YYYYMMDD格式
            year = birth[:4]
            month = birth[4:6]
            day = birth[6:8]
            elements["birth_parts"].extend([
                birth,  # 完整生日
                year,  # 年份
                month + day,  # 月日
                day + month,  # 日月
                year[-2:],  # 后两位年份
                day + month + year[-2:],  # 日月+后两位年
                month + day + year[-2:],  # 月日+后两位年
                year[-2:] + month + day,  # 后两位年+月日
            ])

    # 电话信息
    if user["phone"]:
        phone = user["phone"]
        elements["phone_parts"].extend([
            phone,  # 完整电话
            phone[:4] if len(phone) >= 4 else phone,  # 前四位
            phone[-4:] if len(phone) >= 4 else phone,  # 后四位
        ])

    # 邮箱信息
    if user["email"]:
        email = user["email"]
        if "@" in email:
            email_parts = email.split("@")
            if len(email_parts) == 2:
                username_part = email_parts[0]
                domain = email_parts[1]
                domain_suffix = domain.split(".")[-1] if "." in domain else domain
                elements["email_parts"].extend([
                    username_part,
                    domain,
                    domain_suffix,
                    "@",
                ])

    return elements


def build_password_generation_prompt(user: Dict[str, str], elements: Dict[str, List[str]]) -> str:
    """构建密码生成提示词"""

    prompt = f"""你是一个密码生成专家。根据以下用户信息，生成尽可能多的密码变体。

用户信息：
- 账号(account): {user['account']}
- 姓名(name): {user['name']}
- 邮箱(email): {user['email']}
- 电话(phone): {user['phone']}
- 生日(birth): {user['birth']}

提取的元素：
- 账号变体: {', '.join(elements['account'][:5])}
- 姓名部分: {', '.join(elements['name_parts'][:5])}
- 生日片段: {', '.join(elements['birth_parts'][:8])}
- 电话片段: {', '.join(elements['phone_parts'])}
- 邮箱组件: {', '.join(elements['email_parts'])}

请严格按照以下14条规则生成密码，每次生成{BATCH_SIZE}个不重复的密码：

1. 基础信息提取
直接复用用户核心信息，包括账号、姓名、生日、电话、邮箱后缀。
截取信息片段，如电话取前四位/后四位，生日取年月日等关键片段。

2. 格式变换
大小写转换，如 "alghamdi" 与 "ALGHAMDI" 相互切换适配。
信息顺序调换，生日与用户名正反组合、生日字段反转（如 270689↔19890627）。
姓名结构调整，拆分姓名前缀与用户名，生成 "R.alghamdi"、"alghamdi.R"。

3. 组合拼接
核心信息两两组合，用户名 + 生日、用户名 + 电话、邮箱标识 + 用户名等。
核心信息 + 常用弱密码，如 "alghamdi" 搭配 "123"，"Password" 搭配生日片段等。

4. 辅助元素添加
加特殊字符（!@#$%&* 等），如 alghamdi123!、alghamdi1989@。
加分隔符（.、_等），如 alghamdi.1989、alghamdi_123。
加连续数字/年份，如 "000"、"111" 重复数字，"2020-2025" 近年年份。

5. 生日信息的多维度拆分与复用
生日格式多样化拆分，涵盖完整生日、年+月日、日月年、日月+后两位年等8种以上形式。
生日与其他信息深度绑定，与常用词、特殊字符、邮箱组件交叉搭配。

6. 电话信息的分层使用
电话全段与片段并用，完整电话、前四位、后四位分别作为独立元素组合。
电话与其他核心信息叠加，电话片段 + 用户名 + 特殊字符、完整电话 + 用户名正反组合。

7. 邮箱组件的拆分与重组
提取 @符号、域名后缀单独作为连接符或组合元素。
邮箱组件与多类信息融合，如 alghamdi.ws + 生日、ws@alghamdi + 数字等。

8. 姓名标识的灵活变形
姓名前缀与用户名组合，采用 "R.+用户名"、"用户名.+R"，支持大小写混合。
姓名标识与其他元素叠加，如 R.ALGHAMDI + 生日、alghamdi.R + 特殊字符等。

9. 常用词与弱密码的精准适配
常用弱密码固定搭配模式，首字母大写 + 数字（如 Password123、Admin1989）。
全面覆盖系统类、问候类、通用类常用词，每类均匹配核心数字信息。

10. 单一元素的极致延伸
核心用户名 + 单一数字，覆盖 0-9。
核心用户名 + 单一特殊字符，单独添加各类特殊字符，不额外搭配其他元素。

11. 多元素叠加的组合逻辑
三段式组合，用户名 + 生日 + 特殊字符、姓名标识 + 用户名 + 生日等。
延伸出四段式组合逻辑，如用户名 + 电话片段 + 特殊字符 + 数字。

12. 大小写混合的精细化处理
局部大小写切换，仅姓名标识大写、仅用户名大写、标识与用户名混合大小写。
全大写 + 多元素组合，如 ALGHAMDI + 生日 + 特殊字符、ALGHAMDI + 电话 + 特殊字符等。

13. 纯弱密码的全面覆盖
纯数字弱密码，包含连续数字、重复数字、纯生日数字、纯电话数字。
无核心信息的通用弱密码，仅依赖常用词 + 通用数字，不关联用户个人信息。

14. 年份信息的扩展使用
生日年份多形式复用，完整年份、后两位年份分别与各类元素组合。
近年年份批量适配，2020-2025 连续年份与用户名直接组合。

要求：
1. 每次必须生成{BATCH_SIZE}个不同的密码
2. 密码长度建议在4-30个字符之间
3. 确保密码的多样性和覆盖性
4. 直接输出密码列表，每行一个密码，不要添加序号、说明或其他文字
5. 只输出密码，不要输出其他内容

现在开始生成密码："""

    return prompt


def write_passwords_to_file(passwords: List[str], account: str):
    """实时写入密码到answer.txt（只写入纯密码，不包含账号前缀）"""
    try:
        with open(ANSWER_FILE, "a", encoding="utf-8") as f:
            for pwd in passwords:
                f.write(f"{pwd}\n")  # 只写入密码，不添加账号前缀
                f.flush()  # 立即刷新缓冲区
    except Exception as e:
        print(f"  [警告] 写入文件失败: {e}")


def generate_passwords_batch(client: OpenAI, prompt: str, existing_count: int, target_count: int, account: str,
                             retry_num: int = 0) -> List[str]:
    """调用API生成一批密码（使用流式处理）"""
    try:
        # 动态调整需要生成的数量
        remaining = target_count - existing_count
        batch_size = min(BATCH_SIZE, remaining)

        # 更新提示词中的数量要求
        updated_prompt = prompt.replace(f"{BATCH_SIZE}个", f"{batch_size}个")
        updated_prompt += f"\n\n注意：当前已生成 {existing_count} 个，还需要 {remaining} 个。请生成 {batch_size} 个新的、不重复的密码。"

        print(f"  [调试] 正在调用API (重试 {retry_num}/{MAX_RETRIES})...", flush=True)
        print(f"  [调试] 超时设置: {API_TIMEOUT}秒", flush=True)
        sys.stdout.flush()  # 强制刷新输出缓冲区
        start_time = time.time()

        # 使用流式处理实时接收响应
        print(f"  [调试] 开始流式接收响应...", flush=True)
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "你是一个密码生成专家，严格按照要求生成密码列表，只输出密码，每行一个，不要添加任何序号、说明或其他文字。"},
                {"role": "user", "content": updated_prompt}
            ],
            temperature=0.9,  # 提高温度以增加多样性
            max_tokens=8192,  # 增加token数以生成更多密码
            stream=True  # 启用流式处理
        )

        # 实时处理流式响应
        content_chunks = []
        received_chars = 0
        last_progress_time = start_time
        last_data_time = start_time  # 记录最后一次收到数据的时间

        print(f"  [调试] 开始接收数据流...", flush=True)
        print(f"  [调试] 流式读取超时: {STREAM_READ_TIMEOUT}秒", flush=True)
        sys.stdout.flush()

        try:
            for chunk in stream:
                last_data_time = time.time()  # 更新最后收到数据的时间

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content_chunks.append(delta.content)
                        received_chars += len(delta.content)

                        # 每接收1000字符显示一次进度
                        current_time = time.time()
                        if received_chars % 1000 == 0 or (current_time - last_progress_time) > 5:
                            elapsed = current_time - start_time
                            print(f"  [进度] 已接收 {received_chars} 字符 (耗时 {elapsed:.1f}秒)", flush=True)
                            last_progress_time = current_time
                            sys.stdout.flush()

                # 心跳检测：如果超过30秒没有新数据，认为接收完成
                if time.time() - last_data_time > 30:
                    print(f"  [警告] 超过30秒未收到新数据，认为流已结束", flush=True)
                    sys.stdout.flush()
                    break

        except (httpx.ReadTimeout, HttpxReadTimeout) as e:
            # 流式处理中的读取超时，如果已经接收到内容，就保存并继续
            elapsed = time.time() - start_time
            print(f"  [警告] 流式读取超时 (耗时 {elapsed:.1f}秒)，但已接收 {received_chars} 字符", flush=True)
            print(f"  [信息] 将使用已接收的内容继续处理", flush=True)
            sys.stdout.flush()
            # 继续处理已接收的内容
        except Exception as e:
            # 其他异常，如果已经接收到内容，也保存
            elapsed = time.time() - start_time
            if received_chars > 0:
                print(f"  [警告] 流式处理异常: {e}，但已接收 {received_chars} 字符，将使用已接收内容", flush=True)
                sys.stdout.flush()
            else:
                # 没有接收到任何内容，重新抛出异常
                raise

        # 合并所有内容
        content = ''.join(content_chunks)
        elapsed_time = time.time() - start_time
        print(f"  [调试] API调用完成，耗时 {elapsed_time:.2f} 秒，共接收 {len(content)} 字符", flush=True)
        sys.stdout.flush()

        if not content:
            print("  [错误] API返回的内容为空", flush=True)
            return []

        content = content.strip()
        print(f"  [调试] 收到响应内容，长度: {len(content)} 字符", flush=True)
        print(f"  [调试] 响应前100字符: {content[:100]}...", flush=True)
        sys.stdout.flush()

        # 解析密码列表
        passwords = []
        lines = content.split('\n')
        print(f"  [调试] 共 {len(lines)} 行内容，开始解析...", flush=True)
        sys.stdout.flush()

        for idx, line in enumerate(lines):
            original_line = line
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 移除可能的序号、标点等
            line = re.sub(r'^\d+[\.\)\-\s]*', '', line)
            # 移除可能的标记符号
            line = re.sub(r'^[-•\*\s]+', '', line)
            line = line.strip()
            # 移除引号
            line = line.strip('"\'')

            # 验证密码格式
            if line and len(line) >= 3 and len(line) <= 50:
                passwords.append(line)
            elif line:  # 如果过滤后还有内容但不符合条件，记录一下
                if idx < 5:  # 只记录前5个异常，避免输出太多
                    print(f"  [调试] 跳过无效密码: {original_line[:50]} (长度: {len(line)})", flush=True)

        print(f"  [调试] 成功解析 {len(passwords)} 个有效密码", flush=True)
        sys.stdout.flush()

        # 立即写入文件
        if passwords:
            print(f"  [写入] 正在写入 {len(passwords)} 个密码到 {ANSWER_FILE}...", flush=True)
            write_passwords_to_file(passwords, account)
            print(f"  [写入] 写入完成", flush=True)
            sys.stdout.flush()

        if len(passwords) == 0:
            print(f"  [警告] 未能解析出任何密码，原始内容:\n{content[:500]}", flush=True)
            sys.stdout.flush()

        return passwords

    except APITimeoutError as e:
        print(f"\n  [错误] API调用超时 (>{API_TIMEOUT}秒): {e}", flush=True)
        sys.stdout.flush()
        if retry_num < MAX_RETRIES:
            print(f"  [重试] {RETRY_DELAY}秒后重试...", flush=True)
            sys.stdout.flush()
            time.sleep(RETRY_DELAY)
            return generate_passwords_batch(client, prompt, existing_count, target_count, account, retry_num + 1)
        return []

    except APIConnectionError as e:
        print(f"\n  [错误] API连接错误: {e}", flush=True)
        sys.stdout.flush()
        if retry_num < MAX_RETRIES:
            print(f"  [重试] {RETRY_DELAY}秒后重试...", flush=True)
            sys.stdout.flush()
            time.sleep(RETRY_DELAY)
            return generate_passwords_batch(client, prompt, existing_count, target_count, account, retry_num + 1)
        return []

    except APIError as e:
        print(f"\n  [错误] API错误: {e}", flush=True)
        print(f"  [错误详情] 状态码: {e.status_code if hasattr(e, 'status_code') else 'N/A'}", flush=True)
        print(f"  [错误详情] 错误类型: {type(e).__name__}", flush=True)
        sys.stdout.flush()
        if retry_num < MAX_RETRIES and (not hasattr(e, 'status_code') or e.status_code not in [400, 401, 403]):
            print(f"  [重试] {RETRY_DELAY}秒后重试...", flush=True)
            sys.stdout.flush()
            time.sleep(RETRY_DELAY)
            return generate_passwords_batch(client, prompt, existing_count, target_count, account, retry_num + 1)
        return []

    except Exception as e:
        print(f"\n  [错误] 生成密码时出现未知错误: {e}", flush=True)
        print(f"  [错误类型] {type(e).__name__}", flush=True)
        print(f"  [错误堆栈] {traceback.format_exc()}", flush=True)
        sys.stdout.flush()
        if retry_num < MAX_RETRIES:
            print(f"  [重试] {RETRY_DELAY}秒后重试...", flush=True)
            sys.stdout.flush()
            time.sleep(RETRY_DELAY)
            return generate_passwords_batch(client, prompt, existing_count, target_count, account, retry_num + 1)
        return []


def generate_passwords_for_user(client: OpenAI, user: Dict[str, str], user_index: int, total_users: int) -> Set[str]:
    """为单个用户生成10000个不重复密码"""
    print(f"\n{'=' * 60}")
    print(f"处理用户 {user_index + 1}/{total_users}")
    print(f"账号: {user['account']}, 姓名: {user['name']}")
    print(f"{'=' * 60}")

    elements = extract_user_elements(user)
    prompt = build_password_generation_prompt(user, elements)

    all_passwords = set()
    batch_num = 0

    # 创建进度条
    pbar = tqdm(total=TARGET_PASSWORDS_PER_USER, desc=f"用户 {user_index + 1}/{total_users} - {user['account']}")

    consecutive_failures = 0
    max_consecutive_failures = 3

    while len(all_passwords) < TARGET_PASSWORDS_PER_USER:
        batch_num += 1
        current_count = len(all_passwords)
        remaining = TARGET_PASSWORDS_PER_USER - current_count

        # 检查是否超过最大批次限制（防止无限循环）
        if batch_num > 50:  # 最多50批，每批1000个应该足够
            print(f"\n[警告] 已达到最大批次限制 (50批)，当前生成 {current_count} 个密码")
            break

        print(f"\n第 {batch_num} 批生成中... (还需 {remaining} 个)", flush=True)
        sys.stdout.flush()
        batch_start_time = time.time()

        # 生成一批密码（函数内部已有重试机制）
        new_passwords = generate_passwords_batch(client, prompt, current_count, TARGET_PASSWORDS_PER_USER,
                                                 user['account'])

        batch_elapsed = time.time() - batch_start_time

        if not new_passwords:
            consecutive_failures += 1
            print(f"  [警告] 本次未生成密码 (耗时 {batch_elapsed:.2f}秒)")
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n[错误] 连续 {max_consecutive_failures} 次未生成密码，跳过该用户")
                print(f"  [信息] 当前已生成 {current_count} 个密码")
                break
            print(f"  [重试] {RETRY_DELAY}秒后重试... ({consecutive_failures}/{max_consecutive_failures})")
            time.sleep(RETRY_DELAY)
            continue

        consecutive_failures = 0  # 重置连续失败计数

        # 添加到集合（自动去重）
        before_count = len(all_passwords)
        all_passwords.update(new_passwords)
        after_count = len(all_passwords)
        added_count = after_count - before_count

        # 更新进度
        current_count = len(all_passwords)
        pbar.n = min(current_count, TARGET_PASSWORDS_PER_USER)
        pbar.refresh()

        print(
            f"  [成功] 本批生成: {len(new_passwords)} 个, 新增: {added_count} 个, 总计: {current_count}/{TARGET_PASSWORDS_PER_USER} (耗时 {batch_elapsed:.2f}秒)",
            flush=True)
        sys.stdout.flush()

        # 如果新增数量太少，给出警告
        if added_count < len(new_passwords) * 0.5:  # 如果新增少于生成的一半，说明重复率很高
            print(f"  [警告] 重复率较高: {len(new_passwords) - added_count} 个重复密码", flush=True)
            sys.stdout.flush()

        # 如果已经达到目标，退出
        if current_count >= TARGET_PASSWORDS_PER_USER:
            break

        # 避免请求过快
        time.sleep(0.3)

    pbar.close()

    # 取前10000个
    result = list(all_passwords)[:TARGET_PASSWORDS_PER_USER]
    print(f"\n✓ 用户 {user_index + 1} 完成: 生成 {len(result)} 个不重复密码")

    return set(result)


def main():
    """主函数"""
    print("=" * 60)
    print("密码爆破Agent启动")
    print("=" * 60)
    print(f"API配置: {BASE_URL}")
    print(f"模型: {MODEL}")
    print(f"目标: 每个用户生成 {TARGET_PASSWORDS_PER_USER} 个不重复密码")
    print("=" * 60)

    # 初始化OpenAI客户端
    print(f"\n初始化API客户端...")
    print(f"  API Key: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else '***'}")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Model: {MODEL}")
    print(f"  超时设置: {API_TIMEOUT}秒")

    try:
        # 初始化客户端时设置默认timeout
        # 设置详细的超时配置：连接超时10秒，读取超时API_TIMEOUT秒
        # 注意：流式处理中，read timeout是单个读取操作的超时，不是总超时
        timeout_config = httpx.Timeout(
            connect=10.0,  # 连接超时
            read=STREAM_READ_TIMEOUT,  # 流式处理中单个读取操作的超时（不是总超时）
            write=10.0,  # 写入超时
            pool=10.0  # 连接池超时
        )
        http_client = httpx.Client(timeout=timeout_config)
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=http_client
        )
        print("  ✓ API客户端初始化成功")
        print(f"  [配置] 连接超时: 10秒, 读取超时: {API_TIMEOUT}秒")
    except Exception as e:
        print(f"  ✗ API客户端初始化失败: {e}")
        print(f"  [错误堆栈] {traceback.format_exc()}")
        return

    # 读取用户信息
    print("\n读取用户信息...")
    users = []
    try:
        with open("online.txt", "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        user = parse_user_info(line)
                        if user["account"]:  # 至少要有账号
                            users.append(user)
                        else:
                            print(f"  [跳过] 第 {line_num} 行: 无账号信息")
                    except Exception as e:
                        print(f"  [错误] 解析第 {line_num} 行时出错: {e}")
                        continue

        print(f"  ✓ 共读取 {len(users)} 个有效用户")
        if len(users) == 0:
            print("  ✗ 未读取到任何有效用户，程序退出")
            return
    except FileNotFoundError:
        print(f"  ✗ 错误: 找不到文件 'online.txt'")
        return
    except Exception as e:
        print(f"  ✗ 读取文件时出错: {e}")
        print(f"  [错误堆栈] {traceback.format_exc()}")
        return

    # 清空或创建answer.txt文件
    print(f"\n初始化输出文件: {ANSWER_FILE}")
    try:
        with open(ANSWER_FILE, "w", encoding="utf-8") as f:
            f.write(f"# 密码爆破结果 - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 总用户数: {len(users)}\n")
            f.write(f"# 每个用户目标密码数: {TARGET_PASSWORDS_PER_USER}\n\n")
        print(f"  ✓ 已初始化 {ANSWER_FILE}")
    except Exception as e:
        print(f"  ✗ 初始化文件失败: {e}")

    # 为每个用户生成密码
    all_results = {}

    for i, user in enumerate(users):
        try:
            passwords = generate_passwords_for_user(client, user, i, len(users))
            all_results[user["account"]] = {
                "user_info": user,
                "passwords": list(passwords)
            }

            # 保存中间结果（可选）
            if (i + 1) % 10 == 0:
                print(f"\n已处理 {i + 1} 个用户，保存中间结果...")
                with open("passwords_results.json", "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\n处理用户 {i + 1} 时出错: {e}")
            continue

    # 保存最终结果
    print("\n保存最终结果...")
    with open("passwords_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 生成文本格式的密码列表（每个用户一个文件）
    print("\n生成文本格式密码列表...")
    os.makedirs("passwords", exist_ok=True)
    for account, data in all_results.items():
        filename = f"passwords/{account}_passwords.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for pwd in data["passwords"]:
                f.write(pwd + "\n")

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print(f"共处理 {len(all_results)} 个用户")
    print(f"结果保存在: passwords_results.json 和 passwords/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()

