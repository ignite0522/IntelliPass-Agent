#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：基于online.txt中的用户信息与passwords_merged.txt中的密码库进行相似度匹配,在passwords_merged.txt中只留下topk

使用方法:

  python3 password_matcher.py --online-file online.txt --password-file passwords.txt --output answer_xxxdataset.txt --top-k 10000
  python3 password_matcher.py --online-file online.txt --password-file passwords.txt --output answer.txt --top-k 1000
  python3 password_matcher.py -o online.txt -p passwords.txt -O answer.txt -k 500
  python3 password_matcher.py --online-file online.txt --password-file passwords.txt --output answer.txt --top-k 1000 --max-users 500
  python3 password_matcher.py --online-file online.txt --password-file seclist_all.txt --output answer.txt --top-k 10000 --max-users 500
"""

import re
import argparse
import os
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
from collections import defaultdict
import multiprocessing as mp
from functools import partial

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的进度显示
    def tqdm(iterable, desc=None, unit=None):
        if desc:
            print(desc)
        return iterable


def parse_online_line(line: str) -> Tuple[str, List[str], str, str, str]:
    """
    解析 online.txt 中的一行数据

    Returns:
        (email, name_parts, account, phone, birth)
    """
    line = line.strip()
    if not line:
        return "", [], "", "", ""

    # 使用制表符分割
    parts = line.split('\t')

    email = ""
    name_parts = []
    account = ""
    phone = ""
    birth = ""

    for part in parts:
        if not part or ':' not in part:
            continue

        key, value = part.split(':', 1)
        key = key.strip().lower()
        value = value.strip()

        if key == 'email':
            email = value
        elif key == 'name':
            # name 部分用 | 分隔
            name_parts = [n.strip() for n in value.split('|') if n.strip()]
        elif key == 'account':
            account = value
        elif key == 'phone':
            phone = value
        elif key == 'birth':
            birth = value

    return email, name_parts, account, phone, birth


def load_passwords(password_file: str) -> List[str]:
    """加载所有密码"""
    passwords = []
    print(f"正在加载密码文件: {password_file}")
    with open(password_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="加载密码"):
            pwd = line.strip()
            if pwd:
                passwords.append(pwd)
    print(f"共加载 {len(passwords)} 个密码")
    return passwords


def calculate_similarity(user_info: str, password: str) -> float:
    """
    计算用户信息与密码的相似度

    使用多种方法计算相似度，返回最高值：
    1. 序列相似度（SequenceMatcher）
    2. 包含关系（如果密码包含用户信息或反之）
    3. 编辑距离归一化
    """
    if not user_info or not password:
        return 0.0

    user_info_lower = user_info.lower()
    password_lower = password.lower()

    # 如果完全匹配
    if user_info_lower == password_lower:
        return 1.0

    # 方法1: 序列相似度
    seq_similarity = SequenceMatcher(None, user_info_lower, password_lower).ratio()

    # 方法2: 包含关系
    contain_score = 0.0
    if user_info_lower in password_lower:
        # 用户信息在密码中
        contain_score = len(user_info_lower) / len(password_lower)
    elif password_lower in user_info_lower:
        # 密码在用户信息中
        contain_score = len(password_lower) / len(user_info_lower)

    # 方法3: 字符重叠度（Jaccard相似度）
    user_chars = set(user_info_lower)
    pwd_chars = set(password_lower)
    if user_chars or pwd_chars:
        jaccard = len(user_chars & pwd_chars) / len(user_chars | pwd_chars)
    else:
        jaccard = 0.0

    # 方法4: 公共子串长度
    common_substr_score = 0.0
    if user_info_lower and password_lower:
        # 找出最长的公共子串
        max_common_len = 0
        for i in range(len(user_info_lower)):
            for j in range(len(password_lower)):
                k = 0
                while (i + k < len(user_info_lower) and
                       j + k < len(password_lower) and
                       user_info_lower[i + k] == password_lower[j + k]):
                    k += 1
                max_common_len = max(max_common_len, k)
        if max_common_len > 0:
            common_substr_score = max_common_len / max(len(user_info_lower), len(password_lower))

    # 方法5: 部分匹配（如果用户信息的部分字符在密码中按顺序出现）
    partial_match_score = 0.0
    if len(user_info_lower) >= 3:  # 只对长度>=3的信息计算
        user_chars_list = list(user_info_lower)
        pwd_chars_list = list(password_lower)
        matched_positions = []
        for uc in user_chars_list:
            for i, pc in enumerate(pwd_chars_list):
                if uc == pc and i not in matched_positions:
                    matched_positions.append(i)
                    break
        if matched_positions:
            # 检查是否按顺序
            if matched_positions == sorted(matched_positions):
                partial_match_score = len(matched_positions) / len(user_info_lower)

    # 返回各种相似度的最大值，但给予不同权重
    final_score = max(
        seq_similarity * 0.4,  # 序列相似度权重0.4
        contain_score * 0.8,  # 包含关系权重0.8（最重要）
        jaccard * 0.3,  # Jaccard相似度权重0.3
        common_substr_score * 0.5,  # 公共子串权重0.5
        partial_match_score * 0.4  # 部分匹配权重0.4
    )

    return final_score


def match_user_passwords(user_data: Tuple, passwords: List[str], top_k: int = 1000) -> List[str]:
    """
    为一个用户匹配最相似的密码

    Args:
        user_data: (email, name_parts, account, phone, birth)
        passwords: 所有密码列表
        top_k: 返回前k个最匹配的密码

    Returns:
        最匹配的密码列表
    """
    email, name_parts, account, phone, birth = user_data

    # 收集所有用户信息元素
    user_info_list = []
    user_info_lower_list = []  # 小写版本，用于快速匹配

    if email:
        user_info_list.append(email)
        user_info_lower_list.append(email.lower())
        # 提取email的前缀部分（@之前）
        if '@' in email:
            email_prefix = email.split('@')[0]
            if email_prefix:
                user_info_list.append(email_prefix)
                user_info_lower_list.append(email_prefix.lower())

    if name_parts:
        for name_part in name_parts:
            user_info_list.append(name_part)
            user_info_lower_list.append(name_part.lower())
            # 如果名字部分较长，也添加前几个字符
            if len(name_part) > 3:
                user_info_list.append(name_part[:3])
                user_info_lower_list.append(name_part[:3].lower())

        # 添加名字的组合（如果名字有多个部分）
        if len(name_parts) > 1:
            # 名字的前两个部分组合
            combined = ''.join(name_parts[:2])
            user_info_list.append(combined)
            user_info_lower_list.append(combined.lower())
            # 名字的第一个字母组合
            first_letters = ''.join([n[0] for n in name_parts if n])
            if first_letters:
                user_info_list.append(first_letters)
                user_info_lower_list.append(first_letters.lower())

    if account:
        user_info_list.append(account)
        user_info_lower_list.append(account.lower())
        # 如果account较长，也添加前几个字符
        if len(account) > 3:
            user_info_list.append(account[:3])
            user_info_lower_list.append(account[:3].lower())

    if phone:
        user_info_list.append(phone)
        user_info_lower_list.append(phone.lower())
        # 提取手机号的后4位
        if len(phone) >= 4:
            phone_suffix = phone[-4:]
            user_info_list.append(phone_suffix)
            user_info_lower_list.append(phone_suffix.lower())

    if birth:
        user_info_list.append(birth)
        user_info_lower_list.append(birth.lower())
        # 提取生日的年份
        if len(birth) >= 4:
            year = birth[:4]
            user_info_list.append(year)
            user_info_lower_list.append(year.lower())
        # 提取生日的月日
        if len(birth) >= 8:
            month_day = birth[4:8]
            user_info_list.append(month_day)
            user_info_lower_list.append(month_day.lower())
            # 提取年份的后两位
            if len(birth) >= 4:
                year_short = birth[2:4]
                user_info_list.append(year_short)
                user_info_lower_list.append(year_short.lower())

    # 去除空字符串和重复
    user_info_dict = {}
    for info, info_lower in zip(user_info_list, user_info_lower_list):
        if info and len(info) >= 2:  # 至少2个字符
            user_info_dict[info_lower] = info

    user_info_list = list(user_info_dict.values())
    user_info_lower_list = list(user_info_dict.keys())

    if not user_info_list:
        # 如果没有用户信息，返回空列表
        return []

    # 第一步：快速过滤 - 找出可能相关的密码
    # 只对包含用户信息关键词的密码进行详细计算
    # 优化：先对用户信息按长度排序，长的优先匹配（更精确）
    sorted_info_list = sorted(user_info_lower_list, key=len, reverse=True)

    candidate_passwords = []
    candidate_indices = set()  # 使用set加速查找

    # 优化：使用集合存储需要检查的信息片段，减少重复检查
    info_set_3plus = set([info for info in sorted_info_list if len(info) >= 3])
    info_set_2 = set([info for info in sorted_info_list if 2 <= len(info) < 3])

    # 快速过滤：遍历所有密码，找出候选密码
    # 使用批量处理，每处理一定数量输出一次进度（但不要太频繁影响性能）
    total_passwords = len(passwords)
    check_interval = max(100000, total_passwords // 10)  # 每10%或每10万个检查一次

    for idx, pwd in enumerate(passwords):
        pwd_lower = pwd.lower()
        # 快速检查：密码是否包含任何用户信息
        is_candidate = False

        # 优先检查长信息（>=3字符）
        for info_lower in info_set_3plus:
            if info_lower in pwd_lower:
                is_candidate = True
                break

        # 如果还没匹配，检查短信息（2字符）但只对短密码
        if not is_candidate and len(pwd_lower) <= 20:
            for info_lower in info_set_2:
                if info_lower in pwd_lower:
                    is_candidate = True
                    break

        if is_candidate:
            candidate_passwords.append(pwd)
            candidate_indices.add(idx)

    # 记录快速过滤结果（仅在调试时输出，避免日志过多）

    # 如果候选密码太少，扩大范围：对所有密码进行相似度计算
    # 但只计算前一部分（例如前10万个密码），以平衡性能和准确性
    if len(candidate_passwords) < top_k * 2:
        # 扩大候选范围：添加更多密码
        # 优先添加包含数字的密码（可能与生日、电话相关）
        additional_needed = min(top_k * 10 - len(candidate_passwords), len(passwords) - len(candidate_passwords))
        added = 0
        for idx, pwd in enumerate(passwords):
            if idx not in candidate_indices:
                # 优先选择包含数字的密码
                if any(c.isdigit() for c in pwd):
                    candidate_passwords.append(pwd)
                    candidate_indices.add(idx)
                    added += 1
                    if added >= additional_needed:
                        break

        # 如果还不够，继续添加（最多到10k个候选）
        if len(candidate_passwords) < top_k * 10:
            max_additional = min(top_k * 10 - len(candidate_passwords), len(passwords) - len(candidate_passwords))
            for idx, pwd in enumerate(passwords):
                if idx not in candidate_indices:
                    candidate_passwords.append(pwd)
                    candidate_indices.add(idx)
                    if len(candidate_passwords) >= top_k * 10:
                        break

    # 第二步：对候选密码进行详细的相似度计算
    password_scores = []

    for pwd in candidate_passwords:
        max_similarity = 0.0

        # 对每个用户信息元素，计算与密码的相似度，取最大值
        for user_info in user_info_list:
            similarity = calculate_similarity(user_info, pwd)
            max_similarity = max(max_similarity, similarity)

        # 即使相似度为0，也保留一些密码（以防候选密码不够）
        password_scores.append((pwd, max_similarity))

    # 如果候选密码还不够，添加更多密码（相似度为0）
    # 确保至少有 top_k * 2 个候选密码以供选择
    if len(password_scores) < top_k:
        added_count = 0
        needed = top_k - len(password_scores)
        for idx, pwd in enumerate(passwords):
            if idx not in candidate_indices:
                password_scores.append((pwd, 0.0))
                added_count += 1
                if added_count >= needed * 2:  # 多添加一些以确保有足够的候选
                    break

    # 按相似度排序，取前top_k个
    password_scores.sort(key=lambda x: x[1], reverse=True)

    # 返回密码列表（去重，保持顺序）
    result = []
    seen = set()
    for pwd, score in password_scores:
        if pwd not in seen:
            result.append(pwd)
            seen.add(pwd)
            if len(result) >= top_k:
                break

    # 如果结果还不够1000个，从原始密码库中补充
    if len(result) < top_k:
        remaining = top_k - len(result)
        for pwd in passwords:
            if pwd not in seen:
                result.append(pwd)
                seen.add(pwd)
                remaining -= 1
                if remaining <= 0:
                    break

    return result


def process_user_batch(args):
    """
    处理一批用户的密码匹配（用于多进程）
    在子进程中重新加载密码，避免序列化大量数据的开销
    """
    process_id, user_indices, user_data_list, password_file, top_k = args
    import logging
    import sys

    # 设置日志
    log_msg = f"[进程{process_id}] 开始处理 {len(user_indices)} 个用户"
    print(log_msg, flush=True)
    sys.stdout.flush()

    # 在子进程中加载密码（空间换时间）
    print(f"[进程{process_id}] 正在加载密码文件...", flush=True)
    sys.stdout.flush()
    passwords = []
    try:
        with open(password_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                pwd = line.strip()
                if pwd:
                    passwords.append(pwd)
        print(f"[进程{process_id}] 密码加载完成，共 {len(passwords)} 个密码", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[进程{process_id}] 加载密码文件失败: {e}", flush=True)
        sys.stdout.flush()
        return {}

    results = {}

    # 处理每个用户
    import time as time_module
    for local_idx, global_idx in enumerate(user_indices):
        try:
            user_start_time = time_module.time()
            user_data = user_data_list[global_idx]
            print(f"[进程{process_id}] 开始处理用户 {global_idx + 1} (批次内第 {local_idx + 1}/{len(user_indices)} 个)",
                  flush=True)
            sys.stdout.flush()

            matched_passwords = match_user_passwords(user_data, passwords, top_k)
            results[global_idx] = matched_passwords

            user_elapsed = time_module.time() - user_start_time
            print(
                f"[进程{process_id}] 用户 {global_idx + 1} 处理完成，匹配到 {len(matched_passwords)} 个密码，耗时 {user_elapsed:.2f} 秒",
                flush=True)
            sys.stdout.flush()

        except Exception as e:
            print(f"[进程{process_id}] 处理用户 {global_idx + 1} 时出错: {e}", flush=True)
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            # 如果出错，返回空列表
            results[global_idx] = []

    print(f"[进程{process_id}] 批次处理完成，共处理 {len(results)} 个用户", flush=True)
    sys.stdout.flush()
    return results


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='基于用户信息与密码库进行相似度匹配，为每个用户找出最相似的密码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python3 password_matcher.py --online-file online.txt --password-file passwords.txt --output answer.txt --top-k 1000

  # 使用短参数
  python3 password_matcher.py -o online.txt -p passwords.txt -O answer.txt -k 500

  # 限制处理用户数量
  python3 password_matcher.py -o online.txt -p passwords.txt -O answer.txt -k 1000 --max-users 500

  # 指定进程数
  python3 password_matcher.py -o online.txt -p passwords.txt -O answer.txt -k 1000 --processes 4
        """
    )

    parser.add_argument(
        '--online-file', '-o',
        type=str,
        required=True,
        help='用户信息文件路径（online.txt格式）'
    )
    parser.add_argument(
        '--password-file', '-p',
        type=str,
        required=True,
        help='密码库文件路径'
    )
    parser.add_argument(
        '--output', '-O',
        type=str,
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=1000,
        help='每个用户匹配的密码数量（默认: 1000）'
    )
    parser.add_argument(
        '--max-users',
        type=int,
        default=None,
        help='限制处理的最大用户数量（默认: 处理所有用户）'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help=f'使用的进程数（默认: {mp.cpu_count()}，即CPU核心数）'
    )

    args = parser.parse_args()

    # 获取参数值
    online_file = args.online_file
    password_file = args.password_file
    output_file = args.output
    top_k = args.top_k
    max_users = args.max_users
    num_processes = args.processes if args.processes else mp.cpu_count()

    # 验证参数
    if top_k <= 0:
        print("错误: --top-k 必须大于 0")
        return 1

    if num_processes <= 0:
        print("错误: --processes 必须大于 0")
        return 1

    if max_users is not None and max_users <= 0:
        print("错误: --max-users 必须大于 0")
        return 1

    print("=" * 60)
    print("密码匹配脚本")
    print("=" * 60)
    print(f"用户信息文件: {online_file}")
    print(f"密码库文件: {password_file}")
    print(f"输出文件: {output_file}")
    print(f"每个用户匹配密码数: {top_k}")
    print(f"使用进程数: {num_processes}")
    if max_users:
        print(f"最大用户数: {max_users}")
    print("=" * 60)

    # 验证输入文件是否存在
    if not os.path.exists(online_file):
        print(f"错误: 用户信息文件不存在: {online_file}")
        return 1

    if not os.path.exists(password_file):
        print(f"错误: 密码文件不存在: {password_file}")
        return 1

    # 1. 读取用户信息
    print(f"\n步骤1: 读取用户信息文件: {online_file}")
    user_data_list = []
    try:
        with open(online_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    user_data = parse_online_line(line)
                    user_data_list.append(user_data)
    except Exception as e:
        print(f"错误: 读取用户信息文件失败: {e}")
        return 1

    print(f"共读取 {len(user_data_list)} 个用户信息")

    # 限制用户数量（如果指定）
    if max_users and len(user_data_list) > max_users:
        user_data_list = user_data_list[:max_users]
        print(f"限制为前 {max_users} 个用户")

    # 2. 验证密码文件存在（不加载，让子进程加载）
    print(f"\n步骤2: 验证密码文件: {password_file}")
    if not os.path.exists(password_file):
        print(f"错误: 密码文件不存在: {password_file}")
        return 1

    # 快速检查文件是否为空
    file_size = os.path.getsize(password_file)
    if file_size == 0:
        print("错误: 密码文件为空！")
        return 1

    print(f"密码文件存在，大小: {file_size / 1024 / 1024:.2f} MB")
    print(f"注意: 密码将在子进程中加载（空间换时间）")

    # 转换为绝对路径，确保子进程能正确找到文件
    password_file = os.path.abspath(password_file)
    print(f"密码文件绝对路径: {password_file}")

    # 为了在写入结果时补充密码，我们需要一个密码列表
    # 但这会占用大量内存，所以我们只在需要时加载
    passwords = None  # 延迟加载

    # 3. 匹配密码（多进程处理）
    print(f"\n步骤3: 开始匹配密码（使用 {num_processes} 个进程）")
    print(f"每个用户将匹配前 {top_k} 个最相似的密码")
    print(f"共需要处理 {len(user_data_list)} 个用户")
    print(f"密码文件路径: {password_file}")
    print(f"注意：每个子进程会重新加载密码文件（空间换时间）\n")

    # 将用户索引分批，确保顺序正确
    batch_size = max(1, len(user_data_list) // num_processes)
    batches = []
    for i in range(0, len(user_data_list), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(user_data_list))))
        process_id = len(batches)  # 进程ID
        # 传递密码文件路径而不是密码列表，让子进程自己加载
        batches.append((process_id, batch_indices, user_data_list, password_file, top_k))
        print(
            f"  批次 {process_id}: 处理用户 {batch_indices[0] + 1} 到 {batch_indices[-1] + 1} (共 {len(batch_indices)} 个用户)")

    print(f"\n开始多进程处理...")
    import time
    start_time = time.time()

    # 使用多进程处理
    all_results = {}
    try:
        with mp.Pool(processes=num_processes) as pool:
            # 使用 map_async 以便可以显示进度
            async_result = pool.map_async(process_user_batch, batches)

            # 等待完成，定期检查状态
            while not async_result.ready():
                time.sleep(2)
                elapsed = time.time() - start_time
                print(f"[主进程] 已运行 {elapsed:.1f} 秒，等待子进程完成...", flush=True)

            # 获取结果
            batch_results = async_result.get(timeout=3600)  # 最多等待1小时
            print(f"\n[主进程] 所有批次处理完成，开始合并结果...", flush=True)

            # 按顺序合并结果
            for batch_result in batch_results:
                if batch_result:
                    all_results.update(batch_result)
                    print(f"[主进程] 已合并 {len(batch_result)} 个用户的结果，总计 {len(all_results)} 个用户",
                          flush=True)

    except Exception as e:
        print(f"\n[主进程] 多进程处理出错: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    elapsed_time = time.time() - start_time
    print(f"\n[主进程] 匹配完成！总耗时: {elapsed_time:.1f} 秒")
    print(f"[主进程] 共处理 {len(all_results)} 个用户的结果")

    # 4. 按顺序写入结果（确保顺序正确）
    print(f"\n步骤4: 写入结果到: {output_file}")
    print(f"确保按用户原始顺序写入结果...")

    missing_users = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in range(len(user_data_list)):
            matched_passwords = all_results.get(idx, [])

            if idx not in all_results:
                missing_users.append(idx + 1)
                print(f"  警告: 用户 {idx + 1} 没有结果，使用空列表")
                matched_passwords = []

            # 如果匹配的密码不足1000个，从密码库中补充
            if len(matched_passwords) < top_k:
                remaining = top_k - len(matched_passwords)
                seen_passwords = set(matched_passwords)

                # 延迟加载密码（如果需要补充）
                if passwords is None:
                    print(f"  加载密码文件用于补充...", flush=True)
                    passwords = []
                    with open(password_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            pwd = line.strip()
                            if pwd:
                                passwords.append(pwd)
                    print(f"  密码加载完成，共 {len(passwords)} 个", flush=True)

                # 从密码库中补充
                for pwd in passwords:
                    if pwd not in seen_passwords:
                        matched_passwords.append(pwd)
                        seen_passwords.add(pwd)
                        remaining -= 1
                        if remaining <= 0:
                            break

            # 写入匹配的密码（最多top_k个）
            for pwd in matched_passwords[:top_k]:
                f.write(pwd + '\n')

            # 写入分隔符
            f.write('<END>\n')

            # 每10个用户刷新一次
            if (idx + 1) % 10 == 0:
                f.flush()
                print(f"  已写入 {idx + 1}/{len(user_data_list)} 个用户的结果", flush=True)

    if missing_users:
        print(f"\n警告: 以下用户没有匹配结果: {missing_users}")
    else:
        print(f"\n所有用户都有匹配结果")

    print(f"\n完成！结果已保存到: {output_file}")
    print(f"共处理 {len(user_data_list)} 个用户")

    return 0


if __name__ == "__main__":
    # 设置多进程启动方法（macOS 需要 spawn）
    import sys

    if sys.platform == 'darwin':  # macOS
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 已经设置过了

    exit(main())

