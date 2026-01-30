# /lg_test/chat_bot.py
import os
import json
import yaml
from langgraph.graph import StateGraph, START, END  # State图状态、StateGraph注册状态为状态图
from typing_extensions import TypedDict  # 定义固定格式字典类型
from typing import Annotated  # 为类型[]添加上下文相关的元信息
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model  # 初始化llm
from langchain_core.messages import ToolMessage  # 定义基础工具调用需要的状态信息
from langgraph.prebuilt import ToolNode, tools_condition  # 工具节点、条件边
# 从工具导入
from lg_test.tool import search_tools
from lg_test.tool import search_sub_domain_tool
from lg_test.bot import sub_domain_bot
from lg_test.bot import search_domains_bot
from lg_test.bot import extract_domains_bot
from lg_test.bot import get_urls_content_node
from lg_test.bot import remove_irelevant_content_node
from lg_test.bot import content_optimize_node
from lg_test.bot import content_summarize_node

# python3 -m lg_test.chat_bot


# llm超级爬虫

# pip install -U langgraph langsmith langchain
# python -m lg_test.chat_bot
"""安全大模型"""
# langsmith监控
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "chat_bot"  # 项目名

with open('./lg_test/prompt_list.yaml', 'r') as f:
    prompt_list = yaml.safe_load(f)


# TypedDict:明确key和value类型
class State(TypedDict):
    # Annotated: 为类型[]添加上下文相关的元信息
    messages: Annotated[list, add_messages]
    user_demand: str  # 用户输入关键词
    tool_result: dict  # 返回工具调用结果
    # 域名查询模块
    domains_list: list  # 查询到的domain列表,需要查询满10个相关域名才停止
    domain_tag: bool  # 是否已经查询到满足数量的相关域名
    # 子域模块
    sub_domains_list: list  # 子域列表
    sub_domain_tag: bool  # 子域查询是否结束(将domains_lis当作消费列表pop)
    """注意:这里为了避免爬取内容过多造成内存泄漏,我用文件方式传递数据了"""
    # 相关内容提取
    raw_data_sub_name: str  # 数据文件名
    raw_data_path: str  # 爬取到的数据保存路径
    relevant_data_path: str  # 关键词相关数据保存路径
    # 内容优化
    optimized_data_path: str  # 优化后的数据保存路径
    final_data_path: str  # 增加对content的query&summary的最终数据（保存路径）


def stream_graph_updates(user_input: str):
    """初始化状态、启动工作流、流式输出过程"""
    # graph.stream启动工作流的流式运行——实时流式返回工作流的每一个事件
    # graph.stream(state)这里就传入了初始状态
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # print(f"event:\n{event}")
        # 获取单个事件对象中的值
        for node_name, node_value in event.items():
            # print(f"value:\n{value}")
            if node_value['messages']:  # 先判断消息列表非空
                # print(f"节点{node_name}的最新回复:\n", node_value['messages'][-1].content)
                print(f"节点{node_name}的最新回复了")
            else:
                print(f"节点{node_name}无回复")


def router(state: State):
    """路由节点"""
    last_message = state["messages"][-1] if state["messages"] else None
    if not state["domain_tag"]:  # 域名数量未满足
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "search_tools"
        return "domain_bot"
    elif not state["sub_domain_tag"]:  # 开始子域查询
        return "sub_domain_bot"


def router_sub_domain(state: State):
    """子域查询路由"""
    last_message = state["messages"][-1] if state["messages"] else None
    if not state["sub_domain_tag"]:  # 子域数量未满足
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(last_message.tool_calls)
            return "search_sub_domain_tool"
        return "sub_domain_bot"
    else:  # 结束
        return 'get_urls_content_node'


"""BEGIN图"""
# 绑定预定义的结构到图(实例化State的对象state是在graph.stream()中完成的)
graph_builder = StateGraph(State, recursion_limit=35)

# 添加节点
graph_builder.add_node('domain_bot', search_domains_bot)  # 寻找内容相关域名
graph_builder.add_node('search_tools', ToolNode(search_tools))  # 工具节点
graph_builder.add_node('extract_domains_bot', extract_domains_bot)  # 从工具结果中提取domains
graph_builder.add_node('sub_domain_bot', sub_domain_bot)  # 子域查询bot
graph_builder.add_node('search_sub_domain_tool', ToolNode(search_sub_domain_tool))  # 子域查询工具
graph_builder.add_node('get_urls_content_node', get_urls_content_node)  # 提取所有子域中的内容
graph_builder.add_node('remove_irelevant_content_node', remove_irelevant_content_node)  # 去除和关键词无关内容
graph_builder.add_node('content_optimize_node', content_optimize_node)  # content美化
graph_builder.add_node('content_summarize_node', content_summarize_node)  # 生成content的query&summary&conten_type

# 添加边
graph_builder.add_edge(START, 'domain_bot')  # 开始
graph_builder.add_conditional_edges(  # 条件边
    'domain_bot',
    router,
    {  # 路由映射
        "search_tools": 'search_tools',
        "domain_bot": "domain_bot",
        "sub_domain_bot": "sub_domain_bot",
    }
)
graph_builder.add_edge('search_tools', 'extract_domains_bot')  # 工具调用结果中提取出domains
graph_builder.add_edge('extract_domains_bot', 'domain_bot')  # 返回节点继续手机domains
# graph_builder.add_edge('sub_domain_bot', 'search_sub_domain_tool')  # 执行子域查询工具
graph_builder.add_edge('search_sub_domain_tool', 'sub_domain_bot')  # 调用工具后返回继续查询子域查询
graph_builder.add_conditional_edges(  # 控制子域查询节点的路由
    "sub_domain_bot",
    router_sub_domain,
    {
        "search_sub_domain_tool": "search_sub_domain_tool",
        "sub_domain_bot": "sub_domain_bot",
        "get_urls_content_node": "get_urls_content_node",
    }
)
graph_builder.add_edge('get_urls_content_node', 'remove_irelevant_content_node')  # 去除无关内容
graph_builder.add_edge('remove_irelevant_content_node', 'content_optimize_node')  # 进入content美化
graph_builder.add_edge('content_optimize_node', 'content_summarize_node')  # 进行content的额外信息生成
graph_builder.add_edge('content_summarize_node', END)  # 结束

# 编译图
graph = graph_builder.compile()  # 编译图

# print("保存workflow图中....")
# try:
#     workflow_image = (graph.get_graph().draw_mermaid_png())
#     with open('./lg_test/chat_bot.png', 'wb') as f:
#         f.write(workflow_image)
# except Exception as e:
#     print(e)
"""END图"""

"""TODO:
1.把参数输入全部换成从命令行获取
2.给crawl增加可以代理的命令
"""


def main():
    # while True:

    user_input = input("机器人很高兴为您服务,请输入您的问题: 886离开\n")
    if '886' in user_input.lower():
        print("再见")
    stream_graph_updates(user_input)  # 更新用户消息到图

    # pass
    # try:
    #     user_input = input("机器人很高兴为您服务,请输入您的问题: 886离开\n")
    #     if '886' in user_input.lower():
    #         print("再见")
    #     stream_graph_updates(user_input)  # 更新用户消息到图
    # except Exception as e:
    #     print(f'\n\n{e}\n\n')
    #     user_input = input("请输入您的问题")
    #     print(f"用户输入:{user_input}")
    #     stream_graph_updates(user_input)


if __name__ == "__main__":
    main()
