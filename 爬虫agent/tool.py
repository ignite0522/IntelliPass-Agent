# /lg_test/tool.py
import os
import json
from langchain_core.tools import tool  # 导入@tool装饰器
from langchain_tavily import TavilySearch  # 搜索工具
from tavily import TavilyClient

TavilySearch_max_result = 3  # 主域搜索单次返回结果数量
max_depth = 1  # 子域搜索递归深度

# tavily搜索引擎api
os.environ["TAVILY_API_KEY"] = ""  # llm搜索引擎api
client = TavilyClient(api_key="")  # 工具化调用


@tool
def get_agent_owner():
    """
    获取agent负责人
    参数:无
    返回值:谁创造了这个agent(str)
    """
    return "godice"


@tool
def get_sub_domains(url: str, instruction: str, max_depth: int = max_depth):
    """
    获取域名列表中和用户给定关键词相关的子域名
    参数:
        url(str):搜索的域名
        instruction(str):搜索关键词
        max_depth(int):搜索深度(默认使用1)
    返回值:查询结果list
    """

    res = client.map(
        url=url,
        max_depth=max_depth,
        instruction=instruction,
    )

    # print(f"res:\n{res}")

    return res

    # return res

    # state["sub_domains_list"] = res


search_tool = TavilySearch(
    max_results=TavilySearch_max_result,
    topic="general",  # general/news/finance
    # include_domains = ["cnvd.org.cn/webinfo/show/"],
    # time_range = "year",  # day/week/month/year
)

search_tools = [search_tool, get_agent_owner]
search_sub_domain_tool = [get_sub_domains]