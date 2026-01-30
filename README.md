## 基于智能体的信息打点与口令猜测系统-IntelliPass Agent

### 0x01 前言

**注意！！由于我们是赛后根据WP来写的这篇文章，所以更偏向于解决赛题，在篇幅有限的情况下为了确保大家能够完整的复现，所以尽可能的把关键代码都贴出来了**



大家可以先看一看我们的系统总体架构，看看有没有自己感兴趣的部分，再决定要不要读下去

我们做出的系统总体框架如下

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763888097582-83504bd9-5c58-4120-ad58-0e57f7f7746a.png)





### 0x02 讲讲密码的重要性

无论是在白帽 Src 还是红队前期打点中，很重要的一个思路就是——获取用户的账户密码



通过密码登录，获得一定的“初始权限”，通过漏洞组合拳的方式一步步“蚕食”目标的系统



常见攻防场景中，如：教育 SRC、众测、红队等进入系统后台的方式各有不同

+ “XX 系统手册”
+ “Google 语法搜索获取信息”
+ “nday 读取敏感信息”
+ “xx 数据库泄露”
+ “xx 平台弱口令”
+ ......



今天我想分享的是在 2025Datacon 大赛中，口令安全赛道中的一个密码获取的新思路

不需要繁琐的打点，利用模型和 Agent“一把梭”，破解任意账户的密码

总体上可以理解为传统基于社工密码猜测的进化版

注意！！由于我们是赛后根据WP来写的这篇文章，所以更偏向于解决赛题，为了大家能够完整的复现，所以占用了很多篇幅，尽可能的把代码都贴出来了

### 0x03 密码破解赛题

<font style="color:rgb(25, 27, 31);">根据密码破解过程中是否需要连网，密码猜测算法分为在线破解和离线破解。</font>

<font style="color:rgb(25, 27, 31);">在线破解需要连网，但不需要拿到网站服务器上存储的密码库，攻击者只需要通过与服务器进行交互，针对目标帐号依次尝试可能的密码，直到猜测出密码，或因尝试次数过多被服务器阻止。因此，在线猜测一般也称为小猜测次数下的攻击。</font>

<font style="color:rgb(25, 27, 31);">离线破解不需要连网，但需要拿到网站服务器上存储的密码库，针对目标帐号，在本地依次尝试可能的密码，直到猜测出密码或因算力有限自动放弃猜测。因此，离线猜测不受猜测次数的限制，一般也称为大猜测次数下的攻击。</font>



这次 Datacon赛题也分为在线和离线部分，整体设计也是很符合实战化的场景的，可以说是相当实战的一个比赛



+ 在线场景下：
  - 题目为500条包含目标用户的个人信息，包括邮箱、姓名、用户名、电话号码、出生日期。参赛选手需要破解目标用户的口令，难度同样按照顺序分为200个普通难度（1分/个）、200个中等难度（2分/个）和100个困难难度（4分个）。选手需要每个目标用户生成猜测列表（猜测数目上限为10,000）作为结果提交
+ 离线破解场景下
  - 题目为1000条SHA-256加密的哈希，按照顺序为200个普通难度（1分/个）、300个中等难度（2分/个）、500个困难难度（4分个）。选手需要尽可能多地破解这些加密哈希，最终将破解的哈希和对应明文提交



### 0x04 在线密码破解


在线口令猜测中存在信息源少、猜测次数存在上限这两大痛点

我们给出的解决方案是

1. 通过智能爬虫爬取PII数据库，获得了大量数据库作种子库的一部分
2. 通过一个蒸馏模型与两个Agent生成密码，以增加种子库多样性
3. 最终通过相似度算法选出优质答案后，再通过变异Agent进行不断进化

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763888529162-eb3269c5-8884-4426-9342-6e0f60b4dc49.png)



下面详细讲讲我的是怎么做的：



#### 4.1 信息收集-基于 LangGraph 的智能爬虫 Agent

##### 4.1.1 方法概述

首先在信息收集部分：我们用Langgraph实现了智能爬虫，可以根据用户输入的关键词，自动爬取和总结相关信息



爬虫主要的工作流为六个步骤，如下图所示

1. 第一步根据用户输入关键词，调用MCP工具tavily寻找相关页面
2. 然后调用子域搜索工具，搜索刚才找到页面的子域
3. 域名收集完毕后通过自定义爬虫爬取页面信息
   1. 爬虫有三种提取策略（文本相对标签密度最大——适合文档博客；文本长度大、段落多、链接少——适合广告密集内容；正文关键词匹配——适合规范网站）
4. 在此基础上再删除爬取的无关内容——这里用的是大模型进行相关性打分的方式
5. 接下来就是内容优化的部分：如删除无关内容、优化表达不当、语义错误等
6. 最后让大模型总结页面内容，以Json格式保存最后结果

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763888616759-050204a0-a95a-4fc6-9cf5-1cb3accca614.png)



##### 4.1.2 代码实战

**理论说了这么多，来具体看看怎么做的吧：**

首先需要多种信息源获取信息，怎么爆破这些密码

通过阅读相关论文，和爬虫Agent挖掘数据库的方式、发现一些可能用到的数据库

- 中文数据集——`PII-12306`、`pii- dodonnew`、`PII-CSDN`
- 英文数据集——`PII-Rootkit`、`PII-000Webhost`、`PII-ClixSense`

利用自己写的基于LangGraph的智能爬虫Agent找相关信息

![image.png](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1762956776507-9984fd0e-721d-4286-957c-3bf60f3e3f0a.png)

![image.png](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1762956525596-9ccad7a9-5453-424b-a649-e99b47ca113d.png)

代码如下

Bot的部分代码：

```py
# 核心逻辑：数据去噪 -> 摘要提取 -> 相关性打分过滤
for raw_data in raw_data_list:
    # 1. 物理清洗：去除网页中的广告、导航栏等无用噪音
    useless_res = llm_gpt4o_no_tool.invoke([
        {"role": "system", "content": prompt_list["useless_content_filter_bot"]},
        {"role": "user", "content": f"内容:{raw_data['content']}"}
    ])
    raw_data["content"] = useless_res.content

    # 2. 语义压缩：生成该页面内容的简要总结（Summary）
    summary_res = llm_gpt4o_no_tool.invoke([
        {"role": "system", "content": prompt_list["content_summary_gen_bot"]},
        {"role": "user", "content": f"内容:{raw_data['content']}"}
    ])
    summary = summary_res.content

    # 3. 质量门控：根据用户需求给内容打分 (1-10分)，只有 >= 8 分的精选内容才会进入最终结果
    score_res = llm_gpt4o_no_tool.invoke([
        {"role": "system", "content": prompt_list["value_content_searcher_bot"]},
        {"role": "user", "content": f"key-word:{demand}\nsummary:{summary}"}
    ])
    
    if 8 <= int(score_res.content) <= 10:
        raw_data["score"] = int(score_res.content)
        relevant_data_list.append(raw_data)
        
    time.sleep(0.4) # 频率控制，防止 API 熔断及反爬
```



tool 代码:

```py
@tool
def get_sub_domains(url: str, instruction: str, max_depth: int = 1):
    """
    获取目标域名下与特定关键词（指令）相关的子域名或页面。
    核心逻辑：利用 Tavily 的 map 功能进行定向扫描。
    """
    # client.map 是关键：它不像普通搜索那样返回网页内容，
    # 而是根据 instruction 在给定 url 的层级结构中映射出相关的子域名/路径。
    res = client.map(
        url = url,
        max_depth = max_depth,
        instruction = instruction,
    )
    return res  # 返回包含 base_url 和 results 列表的字典
```





这样通过前期的信息收集，拿到了一些相关数据库



这里是一个对话示例：用户提问“PII库数据的下载链接” 

+ 左图是爬虫Agent爬取到的网站信息
+ 右图是整个LangGraph调用流的追踪

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763888794050-9e5e7bf0-f487-4393-9e2b-57cf175b4abc.png)





#### 4.2 构建种子库-0.5B 蒸馏模型

##### 4.2.1 方法概述

信息收集完毕后，就是是种子库的构件操作

通过拜读邹教授和汪教授的论文后，复现了一个基于身份信息生成猜测口令的模型

**<font style="color:#DF2A3F;">论文中的思路如下</font>**：

1. 使用PII数据库LoRA微调7B的 PLM 模型
2. 再将模型蒸馏到0.5B
   1. 借此在保证口令生成效果的前提下，获得理想的口令生成速度针对每个用户

代码和权重由于文件过大，我就放在网盘了：https://pan.baidu.com/s/1gW1j9tBzBVO3PzzWX3I6uw?pwd=rghc



![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763888856797-b5aa6f67-edee-4b63-9121-f2d1e8acfecb.png)



##### 4.2.2 代码实战

**下面贴出关键步骤和核心代码：**

1.准备评估配置：

在 `config/evaluation_126_csdn_config.ini `填好基础模型、tokenizer 以及 LoRA 权重路径

```py
 [eval.basic]
     base_model = model/dir              # 这里对应的基础模型目录
     tokenizer = model/dir            
     lora_path = checkpoints/126_csdn_disQwen0.5B
     test_path = data/126_csdn/online.txt
     
     [eval.search]
     beam_width_list = [95,1000]+[1000]*14
     batch_size = 1000
     max_guess_number = 100000
     
     [eval.guesser]
     result_path = result/online_cplex2000
```

`beam_width_list`、`batch_size`、`max_guess_number` 控制生成规模；`test_path` 指定校验用的真实口令集合，便于统计命中率；`result_path`是输出目录

2.运行评估模式

在main入口处进入，会自动加载基础模型与tokenizer，叠加LoRA权重，然后调用GuessLLM_Evaluator.eval()，内部使用run_width_search_from_config生成候选口令，详细步骤：

首先，程序走 eval 分支

```python
elif args.mode in ["train", "eval"]:
    trainer, evaluater, seo = load_training_config(args.config)
    ...
    if evaluater:
        evaluater.eval()
```

`load_training_config`会读取`.ini`文件构造`GuessLLM_Evaluator`，其中`Basic_Config_For_Evaluation`指定模型、tokenizer、LoRA 权重，`Targeted_Search_Config`给出 beam 宽度等搜索参数

接着，加载模型与 LoRA 权重

```python
self.base_model = AutoModelForCausalLM.from_pretrained(...)
...
if self.basic_config.lora_path:
    self.peft_model = PeftModel.from_pretrained(...)
    self.peft_model = self.peft_model.merge_and_unload()
```

`GuessLLM_Evaluator._load_peft_model_from_pretrained()`会载入基础模型和`tokenizer`，如果配置了LoRA路径就merge到全量参数中，确保生成阶段使用的是微调/蒸馏后的权重

紧接着，读取online.txt中每行的个人信息样本。每条样本会根据prompt_template_id组装成模型输入（包含社会工程提示），供生成阶段使用

最后，生成口令（束搜索）

```python
search_results = dynamic_beam_search(
    model=model,
    input_ids=item["input_ids_no_response"],
    batch_size=eval_conf.batch_size,
    beam_width_list=eval_conf.beam_width_list,
    vocab=list(vocab.values()),
    eos_threshold=eval_conf.eos_threshold,
    sorted=True
)
pwds_pros = GuessLLM_Evaluator._decode(
    search_results[:eval_conf.max_guess_number],
    tokenizer, eval_conf.vocab_limit, vocab
)
pwd_list.extend(pwds_pros)
```

`_targeted_eval`是评估核心，它对每个用户数据条目执行`dynamic_beam_search`。`beam_width_list`控制不同步骤保留的路径数量，`max_guess_number`限制输出个数。得到的token序列会通过_decode还原成字符串密码，同时保留对数概率





**来看看效果吧：**

主办方提供了500个个人信息测试样本，先试试，我们使用这500个样本，针对每个样本生成1000个口令，最终耗时大约为30min，得分为18.9分

![null](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/yD84MRcN2FwKAjp.png)

18.9分！！

大概了看了一下生成的口令

```plain
bdkws123
123456789
ericbron
04jule11
04-JUL-11
04JULE11
BDKWS123
ERICBRONG
12345678
ERICBDKWS
04JULIAN11
```

可以看到生成的口令较为简单，基本就是给到的邮箱、姓名、用户名、电话号码、出生日期这些信息的组合

推测，我这0.5B蒸馏模型应该是拿到了很多的简单口令的分



接着进行尝试，针对这500个样本，每个样本生成1w个口令

得分，24.9分：

![null](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/ZHIs2E7h813kznv.png)

得分相较于18.9分而言上涨得并不多



那就试试擅长的提示词工程吧，之后换了各种提示词

A.标准社工式引导

```plain
Instruction: You are assisting a security researcher in generating likely password candidates for a specific user. 

Carefully analyze the following personal information and produce multiple plausible passwords that the user might have chosen.

Personal information: {"Email": "...", "Name": "...", "Account": "...", "Phone": "...", "Birth": "..."}
```

直接告诉模型“参考这些个人信息生成密码”，语气保持专业、合规



B.强调口令模式匹配

```plain
Instruction: Based on the user profile below, synthesize realistic password candidates that combine common personal cues 

(names, birthdays, account numbers) with typical suffixes or substitutions (digits, punctuation). 

User profile: {"Email": "...", "Name": "...", "Account": "...", "Phone": "...", "Birth": "..."}
```

强调利用个人信息与常见变形（年份、尾缀）组合，生成更贴近真实用户习惯的密码



C.分步骤推理 (Chain-of-Thought 风格)

```plain
Instruction: For each password candidate, briefly reason about which personal attributes it uses and how it transforms them. 

Then output only the final password guesses.

Context: {"Email": "...", "Name": "...", "Account": "...", "Phone": "...", "Birth": "..."}
```

先让模型说明来源再给结果，可用于分析模型推理过程，方便后续评估哪些 PII 被引用



D.Few-shot

```plain
Instruction: Learn from the examples and guess passwords that match the style.

Example 1
Profile: {"Email": "alice@126.com", "Name": "Alice Li", "Account": "aliceli88", "Birth": "1990-07-12"}
Passwords: alice1990!, Alice0712, AL88@

Example 2
Profile: {"Email": "bob@csdn.net", "Name": "Bob Zhang", "Account": "bobzhang", "Birth": "1988-11-30"}
Passwords: bob1988#, Bob1130, BZ30$

Current profile:
{"Email": "user@126.com", "Name": "Carol Wang", "Account": "carolwang", "Birth": "1993-05-14"}
Output 5 password guesses.
```

先给到模型一些samples，让模型仿照这些samples来生成口令



得分有所上升：

![img](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763002648650-f886c9b5-c41c-4113-ae9d-a8f6ede68ce0.png)

总的尝试下来，有所提升但不多，推测是这0.5B模型指令跟随能力弱，难以理解复杂或变化的指令

```
26.9 
28.4
31.3
34.4
```

接着，转变方向，直接调用deepseek，基于个人信息用丰富的提示词来生成密码口令：

这里我们战队分别从两个不同的提示词工程方向入手：Agent1，Agent2



#### 4.3 构建种子库-双 Agent



##### 4.3.1 方法概述

通过蒸馏模型能够快速生成大量口令，但是由于模型参数的限制，在海量密码量需求的场景，多样性有所不足



因此我们构建了两个口令生成的Agent，都使用了deepseek-chat模型

这两个Agent使用了不同的构建方式

分别是基于few shot和基于用户信息变体的方式构造密码——进一步扩展种子库

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763889042632-5158446a-e25b-415b-a5db-722a96989d8c.png)



##### 4.3.2 Agnet1

实现了一个基于DeepSeek大模型的密码猜测系统，核心思路是：读取用户个人信息（姓名、生日、邮箱等），通过精心设计的Prompt让大模型生成符合人类习惯的密码候选，然后从模型返回的文本中提取密码并批量输出到文件，利用了LLM学习到的密码设置模式来提高猜测准确率

仅给出了提示词代码部分

```py
def construct_prompt(self, pii: Dict[str, str], sibling_passwords: List[str], num_candidates: int = 20, style_examples: Optional[List[str]] = None) -> str:
    # 1. 格式化 PII 信息 (姓名、生日、手机号等)
    pii_str = ", ".join([f"{k}={v}" for k, v in pii.items()])
    # 2. 注入历史密码 (姊妹密码)，提供用户偏好参考
    sibling_str = ", ".join(sibling_passwords) if sibling_passwords else "无"
    
    # 3. 构造少样本提示 (Few-shot)，只提供风格而不提供具体字符
    prompt = f"""你是一个密码分析专家...
用户信息: {pii_str}
历史密码: {sibling_str}
输出要求: 只输出 Markdown 列表，共 {num_candidates} 行...
"""
    return prompt
```





##### 4.3.3 Agent2

同样是一个靠 Agent 生成密码的模型

读取用户记录，解析出账号、姓名、生日、电话、邮箱，并对这些字段进行拆分扩展，例如姓名的多种大小写、缩写，生日的不同排列组合，电话的前后片段，邮箱的用户名和域名等，形成丰富的候选元素集合。然后围绕这批元素组合成一份提示词，**覆盖十四类密码生成规则（基础信息复用、大小写变换、弱密码叠加、特殊字符插入、生日电话多维组合、邮箱组件重组、年份扩展等）**



```py
# 1. 启动流式请求
stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个密码生成专家..."},
        {"role": "user", "content": updated_prompt}
    ],
    temperature=0.9,
    max_tokens=8192,
    stream=True  # 开启流式传输
)

# 2. 实时块处理 (Chunk Processing)
content_chunks = []
try:
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content_chunks.append(delta.content)
                
                # 3. 实时心跳与进度监控
                if time.time() - last_data_time > 30: # 30秒无响应断开
                    break
except (httpx.ReadTimeout, Exception) as e:
    # 4. 容错逻辑：即便报错，也保留已接收的密码
    if len(content_chunks) > 0:
        print(f"流式读取中断，但已保留部分数据")
```





#### 4.4 变异与自然选择-相似度算法

##### 4.4.1 方法概述

因为一个用户只允许进行1w 次密码猜测，我们用到的种子库密码量级是远远超过1w



因此我们构建了一个多维加权的相似度算法——用于筛选优质的种子答案

1. 首先相似度算法会将用户信息拆分为多个元组
   1. （邮箱、邮箱前缀、姓名组合、账号、手机号尾号、生日拆分）
2. 然后加权计算种子库中所有种子密码相对每一个用户的得分
   1. 选出每个用户相似度前1w作为答案

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763889129051-d671d315-ed88-43df-8bfc-9ae26092a224.png)

##### 4.4.2 代码实战

```py
def calculate_similarity(user_info: str, password: str) -> float:
    # 预处理：转小写，消除大小写差异
    user_info_lower = user_info.lower()
    password_lower = password.lower()
    
    if user_info_lower == password_lower: return 1.0 # 完全匹配
    
    # --- 维度 1: 序列相似度 (基于 Python difflib) ---
    seq_similarity = SequenceMatcher(None, user_info_lower, password_lower).ratio()
    
    # --- 维度 2: 包含关系 (这是密码学中最常见的规律) ---
    contain_score = 0.0
    if user_info_lower in password_lower:
        # 如果密码包含用户信息（如 admin -> admin123），按占比给分
        contain_score = len(user_info_lower) / len(password_lower)
    
    # --- 维度 3: 字符重叠度 (Jaccard 距离) ---
    user_chars, pwd_chars = set(user_info_lower), set(password_lower)
    jaccard = len(user_chars & pwd_chars) / len(user_chars | pwd_chars) if (user_chars | pwd_chars) else 0
    
    # --- 维度 4: 公共子串 (Longest Common Substring) ---
    # (此处省略具体循环逻辑...) 
    # 该维度用于捕捉“zhangsan”和“san123”之间的连续重合部分
    
    # --- 维度 5: 最终加权汇总 (最关键的启发式逻辑) ---
    final_score = max(
        seq_similarity * 0.4,       # 序列相似度，权重中等
        contain_score * 0.8,        # 包含关系，权重最高 (核心：密码常在信息基础上加后缀)
        jaccard * 0.3,              # 字符重叠度，仅作参考
        common_substr_score * 0.5,   # 公共子串，权重较高
        partial_match_score * 0.4    # 顺序匹配（如姓名缩写 zs 匹配 zhangsan）
    )
    
    return final_score
```

相似度计算整体按“先快后细”的策略逐层累加，但最终取五类得分的最大值：

- 完全一致判定：用户特征与密码小写后如果完全一样，直接返回 1.0
- 包含关系权重最高：若特征包含在密码里（或反过来），按长度比例给出 0～1 的得分，并在≥0.8 时直接复用这个分数作为后续序列相似度，避免重复计算
- 序列匹配：使用序列比对（类似于 Levenshtein 的近似效果）获取 0～1 的比值，只有在包含得分不高时才真正执行
- 字符集合重叠：把两端字符集取交并比值，得到 Jaccard 分数（0～1），为一些字符重叠但顺序差异较大的场景提供参考
- 最长公共子串估计：若长度不大（≤50）并且前面得分不高，就滑动扫描算出最长共串长度占比；否则用 Jaccard 结合比例快速估算
- 顺序子序列匹配：针对长度 3~30 的特征，查看其字符能否按顺序在密码中找到（不要求连续），命中率越高得分越高

这五项得分分别乘以不同权重（包含关系权重最高），最后选最大值作为相似度；一旦出现满分，后续立刻终止计算以节省时间





#### 4.5 变异与自然选择-变异与得分筛选

##### 4.5.1 方法概述

通过相似度算法完成首轮“自然进化”后，我们使用评分作为第二轮进化的“自然选择”标准

（这里的评分可以简单理解为密码猜测成功率，在实际的在线密码猜测场景其实不需要这么复杂，有没有成功，看登录成功即可。需要处理的只是比如验证码识别/绕过，IP 池更换等）



进而设计了基于大模型的密码变异策略

1. 首先会用户为单位抽样密码
2. 再让大模型进行密码变异
   1. 具体而言：让大模型通过大小写混淆、Leet形近替换、多源组合、键盘组合等多种方式进行密码变异
3. 最终根据变异后的评分判断本轮变异样本是否“存活”

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763889236308-cb1b4f2e-6b2c-446c-acd2-09b9616aaea8.png)



##### 代码实战

```py
def process_user_mutation(user_info, original_passwords, sample_ratio, num_generate):
    """
    核心整合逻辑：从特征提取到 LLM 生成，再到最终筛选
    """
    # 1. 特征提取：将用户信息拆解为密码基因
    user_tokens = set()
    for key in ['name', 'email', 'phone', 'birth']:
        val = user_info.get(key, "")
        if val:
            # 提取片段：如 phone '13812345678' -> {'138', '5678', ...}
            parts = re.split(r'\W+', val) 
            for p in parts:
                if p:
                    user_tokens.update({p, p.lower(), p.capitalize()})
    
    # 2. 抽样：决定哪些密码需要变异
    target_indices = random.sample(range(len(original_passwords)), 
                                  max(1, int(len(original_passwords) * sample_ratio)))
    
    # 3. 构建 Prompt 并调用 LLM (简化示意)
    prompt = f"用户信息: {user_tokens}\n原密码: {[original_passwords[i] for i in target_indices]}\n生成{num_generate}个变体..."
    
    # 假设 llm_results 是从 API 获取的字符串列表
    llm_results, _ = generate_passwords_with_llm(prompt, len(target_indices))

    # 4. 多级选择与去重核心逻辑
    mutated_list = list(original_passwords)
    existing_new = set()
    
    for i, idx in enumerate(target_indices):
        original = original_passwords[idx]
        llm_cand = llm_results[i] if i < len(llm_results) else None
        
        # 优先级：LLM 变体 > 规则引擎变体 (Fallback) > 原密码
        final_cand = original
        sources = [llm_cand] + generate_fallback_variants(original, user_tokens, [])
        
        for cand in sources:
            if cand and cand != original and cand not in existing_new:
                # 校验：4-30位，仅限字母数字符号
                if 4 <= len(cand) <= 30 and re.match(r"^[A-Za-z0-9!@#$%^&*._\-]+$", cand):
                    final_cand = cand
                    existing_new.add(cand)
                    break
        
        mutated_list[idx] = final_cand
        
    return mutated_list
```

主要是通过随机抽样的方式，替换部分密码，使用 Agent 进行变异



这个脚本的大体做法是：先按 <END> 分隔逐个读入用户的密码表，同时从用户资料里挖掘姓名、邮箱、生日等关键词，再结合一个参考密码样本池为后续生成提供灵感。对每个用户，会按照设定的抽样比例挑出一批待增强的原密码，然后把用户资料、已有密码列表、参考示例这些信息发给大模型，请求一次生成同等数量的全新候选。生成是并发进行的，并且严格保持与原有顺序一一对应。

考虑到大模型有时会给出重复、过于保守或根本失败的结果，脚本设计了多重兜底：先对模型返回逐条检验（长度、字符集、与原密码差异、与已有密码去重等），若不合格，会使用本地的“备选池”尝试，这个备选池是根据用户关键词、参考示例、常见数字/符号组合等规则动态扩展出来的；仍然不行才保留原密码。每条最终写入实时追加到新的输出文件，以免长任务中途中断造成损失，且在段落尾部进行去重，确保每个用户的集合规整、无重复。

整个过程带有详细日志：批次开始/结束、耗时、来源是模型还是fallback、是否保持原样等都会输出，便于监控和排错。完成后再统计抽样量、真实替换量，保证从整体到细节都可追踪。



### 0x05 离线破解

在离线解密部分，没有在线解密这么复杂

1. 首先我们通过智能爬虫获取了一些在线解密网站和泄露的数据库
   1. 可以对离线数据进行第一轮初始解密基于初始结果
2. 然后构建了基于大模型的密码猜测策略策略
   1. 首先让Agent统计已有密码中字母的分布和格式
      1. 比如字频、位置特征、大小写、长度分布、结构模式、常见词根、首字母大写比例、结尾字符类型
   2. 再让Agent基于统计特征分别生成最可能的密码变体、再进行对应的掩码爆破

![](https://gitee.com/YGFYUGF7DTFY/typora/raw/master/1763889487099-2c3a8d53-8761-41df-a361-20fb6226ed51.png)



### 0x06 最后聊聊实战场景


在本次比赛中的在线破解场景，赛题组已经给出了每个用户可能的一些身份信息

+ 实际上：这样是大大降低了破解难度，省去了用户身份信息收集的过程
+ 但是：在赛题中所有用户基本不存在弱口令，又很大程度上提高了破解难度



在实战中，我们其实需要自己进行用户信息打点这一个步骤。

例如

+ 根据手机号：收集用户的相关社交账号、身份信息、出生年月日
+ 根据邮箱：收集用户的相关社交账号、手机号、身份信息、出生年月日



对于比较熟悉的师傅来说：

针对单个用户的手机号和邮箱，其实有一套完整的社工流程，可以比较轻松地获取到所有相关信息



但是如果需要批量的话，可能就相对困难一点（或者说需要一定的资源）

但是其实理论上 4.1 中提到的智能爬虫，是完全可以全部自动化的，这里不再赘述了







