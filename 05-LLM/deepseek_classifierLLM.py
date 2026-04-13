import os
import json

import pandas as pd
from tenacity import stop_after_attempt, wait_fixed, retry

from src.config.bert_config import Config
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

conf = Config()
# 定义分类标签隐射
with open(conf.class_chinese_path, "r", encoding="utf-8") as f:
    class_List = [line.strip() for line in f]
    LABEL_MAPPING = {i:e for i, e in enumerate(class_List)}
    REVERSE_LABEL_MAPPING = {e:i for i, e in enumerate(class_List)}
    print(LABEL_MAPPING)
    print(REVERSE_LABEL_MAPPING)

# 初始化deepseek模型
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)
print(llm)

# 定义带重试机制的LLM调用函数
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def invoke_llm(prompt):
    """
    调用LLM模型,若失败则重试最多 3 次，每次间隔 2 秒
    :param prompt:
    :return:
    """
    return llm.invoke(prompt)

# 读取文件用于分类
def read_file(file_path):
    """
    读取文件
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path, sep='\t', names=['title','lable'],header=None,encoding='utf-8')
    return df['title'].tolist(), df['lable'].tolist()

# deepseek分类逻辑实现
def deepseek_classifier(title:str):
    """
    deepseek分类逻辑实现
    :param title:
    :return:
    """
    # 定义提示词，包含类别说明和示例
    prompt = [
        {
            "role": "system",
            "content": """
    你是一名新闻分类审核员，任务是将新闻标题分类到以下类别之一：
    财政, 房地产, 股票, 教育, 科学, 社会, 政治, 体育, 游戏, 娱乐。
    请根据标题内容和以下关键词与示例，匹配最相关的类别。如果标题涉及教育机构但核心是社会贡献，优先归为 society。
    返回 JSON 格式：{"category": "类别", "reason": "分类原因"}

    类别关键词与示例：
    - 财政: 银行、信用卡、贷款、利率 (例: "各银行信用卡挂失费迥异")
    - 房地产: 房产、地价、楼盘 (例: "东5环海棠公社230平准现房")
    - 股票: 股市、股指、期货 (例: "金证顾问：过山车行情意味着什么")
    - 教育: 学校、考试、招生 (例: "中华女子学院仅1专业招男生")
    - 科学: 技术、网站、宇航 (例: "“手机钱包”亮相科博会")
    - 社会: 社会事件、犯罪、公益 (例: "82岁老太为学生做饭扫地44年")
    - 政治: 政策、国际关系 (例: "查韦斯称愿为俄罗斯提供空军基地")
    - 体育: 比赛、运动员、奥运 (例: "卡佩罗：德国脚生猛的原因")
    - 游戏: 电子游戏、网游、电竞 (例: "《赤壁OL》攻城战硝烟又起")
    - 娱乐: 明星、影视、综艺 (例: "冯德伦徐若瑄隔空传情")
    """
        },
        {
            "role": "user",
            "content": f"新闻标题：'{title}'，请分类并说明原因。"
        }
    ]
    # 调用LLM模型
    result = invoke_llm(prompt)
    # 从 AIMessage 对象中提取 JSON 字符串并解析
    # try:
    result_dict = json.loads(result.content)
    return result_dict
    #     result_dict.fore
    #     return {
    #         "category": result_dict.get("category", "society"),
    #         "reason": result_dict.get("reason", "未明确分类，归为社会类别")
    #     }
    # except (json.JSONDecodeError, AttributeError) as e:
    #     print(f"JSON解析失败: {e}")
    #     return {
    #         "category": "society",
    #         "reason": f"解析失败，使用默认分类: {str(e)}"
    #     }

if __name__ == '__main__':
    # 读取文件
    titles, labels = read_file(conf.test_path)
    # 批量处理,为了节省token，只预测前5个数据
    test_titles = titles[:20]
    # results = [deepseek_classifier(title) for title in titles]
    results = deepseek_classifier(test_titles)
    print(results)
    total_num = len(test_titles)
    acc = sum([1 for i in range(total_num) if LABEL_MAPPING[labels[i]] == results[i].get("category", "society")]) / total_num
    print(f"准确率：{acc:.2%}")
    # 输出结果
    for i, result in enumerate(results):
        print(f"标题：{test_titles[i]}")
        print(f"分类结果：{result.get('category', 'society')}")
        print(f"分类原因：{result.get('reason', '未明确分类，归为社会类别')}")
        print(f"真实标签：{LABEL_MAPPING[labels[i]]}")
        print("-" * 20)


