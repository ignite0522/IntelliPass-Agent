# /lg_test/bot.py
import re
import json
import os
import yaml
import random
import time
from datetime import datetime
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage  # 定义基础工具调用需要的状态信息
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage  # 导入消息类型
from lg_test.tool import search_tools
from lg_test.tool import search_sub_domain_tool
# 导入自己写的爬虫
from tool.crawl import crawl_page_content

single_need_domains_num = 3  # 需要的关键词的域名数量

# llm_api
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "deepseek-chat")
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["DEEPSEEK_API_KEY"] = API_KEY
# tavily_api
client = TavilyClient(api_key="")
os.environ["TAVILY_API_KEY"] = ""  # llm搜索引擎api

with open('./lg_test/prompt_list.yaml', 'r') as f:
    prompt_list = yaml.safe_load(f)

llm_gpt4o = init_chat_model(
    model=MODEL_NAME,
    temperature=1.2,
).bind_tools(search_tools)

llm_gpt4o_no_tool = init_chat_model(
    model=MODEL_NAME,
    temperature=0.4,
)

llm_gpt4o_search_sub = init_chat_model(
    model=MODEL_NAME,
    temperature=0.5,
).bind_tools(search_sub_domain_tool)


def search_domains_bot(state: dict, need_domains_num: int = single_need_domains_num):
    """域名搜索bot"""
    # state是在执行过程中动态初始化的
    if "domains_list" not in state:
        state["domains_list"] = []
    if "domain_tag" not in state:
        state["domain_tag"] = False
    if "sub_domain_tag" not in state:
        state["sub_domain_tag"] = False

    # 判断域名数量是否足够
    if len(state["domains_list"]) >= need_domains_num:
        state["domain_tag"] = True
        state["messages"].append(
            AIMessage(content=f'已收集到{len(state["domains_list"])}个域名,完成任务')
        )
        return state

    # 动态生成prompt
    system_prompt = prompt_list["search_domains_bot"].format(
        need_domains_num=need_domains_num,
        len_domains_list=len(state["domains_list"]),
        domains_list="\n".join(state["domains_list"]),
    )

    # print(f"search_domains_bot-system_prompt:{system_prompt}")

    """生成查询的随机温度&种子"""
    random_seed = random.randint(1, 10000)
    rand_temp = random.uniform(0.2, 2.0)

    # 插入agent定义&调用llm
    user_inputs = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)]
    message_to_llm = [{"role": "system", "content": system_prompt}] + [
        {"role": "user", "content": f"{user_inputs[0]}{random_seed}"}]
    res = llm_gpt4o.invoke(
        message_to_llm,
        config={"temperature": rand_temp},
    )  # 返回的是Aimessage

    # 更新历史消息
    state["messages"].append(res)
    return state


def extract_domains_bot(state: dict):
    """提取域名bot"""
    # 初始化状态
    if "domains_list" not in state:
        state["domains_list"] = []

    # 1.找到最新的工具调用结果
    tool_messages = [
        idx for idx, msg in enumerate(state["messages"]) if isinstance(msg, ToolMessage)
    ]

    if not tool_messages:
        state["messages"].append(
            AIMessage(content="extract_domains_bot未找到工具调用,无法提取域名")
        )
        return state

    latest_tool_idx = tool_messages[-1]
    latest_tool_message = state["messages"][latest_tool_idx]
    tool_result_content = latest_tool_message.content
    # 清洗工具调用结果
    all_content_str = ""  # 初始化空字符串
    try:
        # 将工具结果从JSON字符串解析为字典
        tool_result = json.loads(tool_result_content)
        # 遍历results列表中的每个结果
        for item in tool_result.get("results", []):
            # 提取当前结果的url、title、content（若不存在则用空字符串代替）
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "")
            # 拼接当前结果的三个字段，用换行分隔；每个结果之间用两个换行分隔
            item_str = f"URL: {url}\n标题: {title}\n内容: {content}\n\n"
            all_content_str += item_str  # 追加到总字符串
        # 去除末尾多余的换行
        all_content_str = all_content_str.strip()
    except json.JSONDecodeError:
        # 若工具结果不是JSON格式，直接用原始内容
        all_content_str = tool_result_content
    # print(f"extract_domains_bot-latest_tool_message:{all_content_str}")

    # 2.调用llm处理消息
    system_prompt = prompt_list["extract_domains_bot"]
    user_inputs = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_inputs:
        state["messages"].append(
            AIMessage(content="extract_domains_bot未找到用户输入,无法提取域名")
        )
        print("extract_domains_bot未找到用户输入,无法提取域名")
        return state
    user_prompt = f"提取{user_inputs[0]}相关URI,从下面内容中提取:\n{all_content_str}"
    if "user_demand" not in state:
        state["user_demand"] = user_inputs[0]

    print(f"提取{user_inputs[0]}相关URI,从下面内容中提取:\n")

    add_user_demand = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        res = llm_gpt4o_no_tool.invoke(add_user_demand)
    except Exception as e:
        print("查询中可能存在非法语言:\n{e}")
        state["messages"].append(
            AIMessage(content=f"查询中可能存在非法语言:{e}")
        )
        return state

    # print(f"extract_domains_bot-llm结果:{res}")
    # print(f"extract_domains_bot-add_user_demand:{add_user_demand}")
    domains_list = res.content.strip().split("\n")

    # 3.更新消息
    new_domains = []
    for line in domains_list:
        domain = line.strip()
        if domain not in state["domains_list"] and domain not in new_domains:
            new_domains.append(domain)
            print(f"extract_domains_bot提取到新域名:{domain}")

    # 4.更新状态
    if new_domains:
        state["domains_list"].extend(new_domains)
        state["messages"].append(
            AIMessage(
                content=f'extract_domains_bot提取到{len(new_domains)}个新域名,当前共{len(state["domains_list"])}个域名')
        )
        print(f"extract_domains_bot提取到{len(new_domains)}个新域名,当前共{len(state['domains_list'])}个url")
    else:
        state["messages"].append(
            AIMessage(content="extract_domains_bot未提取到新域名")
        )
        print("extract_domains_bot未提取到新域名")

    # 5.清除工具调用
    # state["messages"].pop(latest_tool_idx)
    return state


def sub_domain_bot(state: dict):
    """子域名收集bot"""
    # 初始化状态
    if "sub_domains_list" not in state:
        state["sub_domains_list"] = []
    if "sub_domain_tag" not in state:
        state["sub_domain_tag"] = False
    user_demand = state["user_demand"]

    """处理工具调用结果,加入state"""
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage) and msg.name == "get_sub_domains"]
    if tool_messages:
        last_tool_messages = tool_messages[-1]
        tool_result = last_tool_messages.content
        try:
            res = json.loads(tool_result)
            sub_domains_list = res["results"]
            # 子域名去重添加
            for url in sub_domains_list:
                if url not in state["sub_domains_list"]:
                    state["sub_domains_list"].append(url)
                    print(f"sub_domain_bot提取到新子域名:{url}")
            # base_url去重添加
            if res["base_url"] not in state["sub_domains_list"]:
                state["sub_domains_list"].append(res["base_url"])
                print(f"sub_domain_bot添加主域名:{res['base_url']}")

        except Exception as e:
            print(f"工具调用结果解析失败:{e}")

        print(f"已提取到工具结果:{tool_result}")

    """处理完成返回判断"""
    if len(state["domains_list"]) == 0:
        state["messages"].append(
            AIMessage(content=f"domains_list的子域寻找完毕")
        )
        print("domains_list的子域寻找完毕")
        # sub_domains_list = state["sub_domains_list"]
        # for
        state["sub_domain_tag"] = True
        return state
    else:
        print(f'还剩{len(state["domains_list"])}个子域需要处理')
        # print(f'域名如下:{state["domains_list"]}')

    """调用llm生成查询子域的function_call"""
    url = state["domains_list"].pop(0)

    system_prompt = prompt_list["sub_domain_bot"]
    user_prompt = f"查找{url}中和{user_demand}相关页面"

    message_to_llm = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    res = llm_gpt4o_search_sub.invoke(
        message_to_llm,
    )

    state["messages"].append(res)  # 返回tool_call
    return state


def get_urls_content_node(state: dict):
    """从获取的所有子页面url中提取内容,存到/result/raw中"""
    GOVERNMENT_DOMAIN_PATTERNS = ['.gov.cn', '.gov']
    content_result = []
    file_name = state["user_demand"][:20]
    url_len = len(state["sub_domains_list"])
    count = 0
    while len(state["sub_domains_list"]) > 0:
        count += 1
        target_url = state["sub_domains_list"].pop(0)
        # 跳过gov,不想吃紫蛋呀，兄弟萌
        if any(pattern in target_url for pattern in GOVERNMENT_DOMAIN_PATTERNS):
            print(f"进度{count}/{url_len}——跳过政府网站: {target_url}")
            continue
        content_map = crawl_page_content(target_url)
        if content_map["content"]:
            content_result.append(content_map)
            print(f"进度{count}/{url_len}——get_urls_content_node-爬取{target_url}成功")
        else:
            print(f"进度{count}/{url_len}——get_urls_content_node-爬取{target_url}出现问题:\n{content_map['error']}")

    date = datetime.now().strftime("%Y%m%d")
    sub_name = f"{file_name}_{date}.json"
    save_path = f"./result/raw/{sub_name}"
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state["raw_data_sub_name"] = sub_name
    state["raw_data_path"] = save_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(content_result, f, ensure_ascii=False, indent=2)
    print(f"get_urls_content_node-爬取结束,结果保存在{save_path}")

    return state


def remove_irelevant_content_node(state: dict):
    """移除爬取的与关键词不相关的内容"""
    orin_demand = state["user_demand"]
    # 优化用户需求,提取最关键部分
    message_to_llm = [{"role": "system", "content": prompt_list["demand_optimization_bot"]},
                      {"role": "user", "content": orin_demand}]
    try:
        res = llm_gpt4o_no_tool.invoke(message_to_llm)
    except Exception as e:
        print(f"可能含有违规信息,调用llm_api出错:\n{e}")
        return state

    demand = res.content
    print(f"原需求:{orin_demand},优化后的需求:{demand}")

    raw_data_path = state["raw_data_path"]
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data_list = json.load(f)
    sub_name = state["raw_data_sub_name"]
    relevant_data_list = []

    all_count = len(raw_data_list)
    count = 0
    for raw_data in raw_data_list:
        count += 1

        # 信息提炼,需要去除所有无关信息
        useless_system_prompt = prompt_list["useless_content_filter_bot"]
        useless_user_prompt = f'内容:{raw_data["content"]}'
        useless_message_to_llm = [{"role": "system", "content": useless_system_prompt},
                                  {"role": "user", "content": useless_user_prompt}]
        try:
            useless_res = llm_gpt4o_no_tool.invoke(useless_message_to_llm)
        except Exception as e:
            print(f"可能含有违规信息,调用llm_api出错:\n{e}")
            continue
        raw_data["content"] = useless_res.content
        time.sleep(0.4)

        # 提炼内容主要在做什么
        summary_system_prompt = prompt_list["content_summary_gen_bot"]
        summary_user_prompt = f'内容:{raw_data["content"]}'
        summary_message_to_llm = [{"role": "system", "content": summary_system_prompt},
                                  {"role": "user", "content": summary_user_prompt}]
        try:
            summary_res = llm_gpt4o_no_tool.invoke(summary_message_to_llm)
        except Exception as e:
            print(f"可能含有违规信息,调用llm_api出错:\n{e}")
            continue
        summary = summary_res.content
        time.sleep(0.4)

        # 给内容进行相关性打分,删除无关信息
        user_prompt = f'key-word:{demand}\nsummary:{summary}'

        message_to_llm = [{"role": "system", "content": prompt_list["value_content_searcher_bot"]},
                          {"role": "user", "content": user_prompt}]
        try:
            res = llm_gpt4o_no_tool.invoke(message_to_llm)
        except Exception as e:
            print(f"可能含有违规信息,调用llm_api出错:\n{e}")
            continue
        score = int(res.content)
        print(f'url:{raw_data["url"]}打分结果:----{res.content}')
        if score >= 8 and score <= 10:
            print(f"{count}/{all_count}大于等于8分,加入结果")
            raw_data["score"] = score
            relevant_data_list.append(raw_data)
        else:
            print(f"{count}/{all_count}小于8分,不加入结果")
        time.sleep(0.4)  # 避免llm_api调用频繁&爬虫调用过频繁-触发反爬

    save_path = f"./result/relevant/{sub_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保./result/relevant存在
    state["relevant_data_path"] = save_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(relevant_data_list, f, ensure_ascii=False, indent=2)
    print(f"remove_irelevant_content_node-爬取结束,结果保存在{save_path}")

    return state


def content_optimize_node(state: dict):
    """对relevant内容进行优化处理"""
    sub_name = state["raw_data_sub_name"]
    relevant_data_path = state["relevant_data_path"]
    with open(relevant_data_path, "r", encoding="utf-8") as f:
        relevant_data_list = json.load(f)

    all_count = len(relevant_data_list)
    count = 0

    optimizied_data_list = []
    for data in relevant_data_list:
        count += 1

        user_prompt = f"优化以下内容:\n{data['content']}"
        system_prompt = prompt_list["content_optimizer_bot"]
        message_to_llm = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            res = llm_gpt4o_no_tool.invoke(message_to_llm)
            print(f"{count}/{all_count},content优化完毕")
        except Exception as e:
            print(f"{count}/{all_count},content优化出错:\n{e}")
            continue
        data['content'] = res.content.strip()
        optimizied_data_list.append(data)
        time.sleep(0.4)

    save_path = f"./result/optimized/{sub_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保./result/optimized存在
    state["optimized_data_path"] = save_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(optimizied_data_list, f, ensure_ascii=False, indent=2)

    return state


def content_summarize_node(state: dict):
    """对优化后的内容进行总结:加入summary&content_type&query"""

    sub_name = state["raw_data_sub_name"]
    optimized_data_path = state["optimized_data_path"]
    with open(optimized_data_path, "r", encoding="utf-8") as f:
        optimized_data_list = json.load(f)

    all_count = len(optimized_data_list)
    count = 0

    """定义信息处理的函数"""

    def add_extra_info_from_content(single_data: dict, system_prompt: str, user_prompt: str, new_key: str):
        """利用不同prompt针对content生成额外信息"""
        data = single_data.copy()
        message_to_llm = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            res = llm_gpt4o_no_tool.invoke(message_to_llm)
            time.sleep(0.4)
        except Exception as e:
            print(f"add_extra_info_from_content-{new_key}出错:\n{e}")
            data[f"{new_key}"] = f"add_extra_info_from_content-{new_key}调用llm中出错了"
            return data
        if new_key == "query":
            data[f"{new_key}"] = res.content.strip().split("\n")
        else:
            data[f"{new_key}"] = res.content.strip()
        return data

    """进行信息处理"""
    add_summary_data_list = []
    for data in optimized_data_list:
        count += 1
        user_prompt = data["content"]
        print(f"content_summarize_node信息处理进度:{count}/{all_count}")
        # 1.总结知识-summary
        system_prompt = prompt_list["content_summary_gen_bot"]
        summary_data = add_extra_info_from_content(data, system_prompt, user_prompt, "summary")
        # 2.content类型-content_type
        system_prompt = prompt_list["content_type_gen_bot"]
        type_data = add_extra_info_from_content(summary_data, system_prompt, user_prompt, "content_type")
        # 3.添加提问总结-query
        system_prompt = prompt_list["content_query_gen_bot"]
        all_data = add_extra_info_from_content(type_data, system_prompt, user_prompt, "query")
        add_summary_data_list.append(all_data)

    save_path = f"./result/final/{sub_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保./result/summary存在
    state["final_data_path"] = save_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(add_summary_data_list, f, ensure_ascii=False, indent=2)

    return state


def main():
    # res = client.extract(
    #         urls = ['https://godice.top'],
    #         include_images = True,
    #         extract_depth = "basic",  # 搜索模式basic/advanced
    #     )
    # print(res)
    response = client.extract("https://blog.csdn.net/u013172930/article/details/148343036")

    print(response)


if __name__ == "__main__":
    main()