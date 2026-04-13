import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from tenacity import stop_after_attempt, wait_fixed, retry

from src.config.bert_config import Config
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量（如需本地代理或特定配置）
load_dotenv()

conf = Config()

# 1. 定义分类标签映射（基于你之前的 bert_config.py 路径逻辑）
with open(conf.class_chinese_path, "r", encoding="utf-8") as f:
    class_List = [line.strip() for line in f]
    LABEL_MAPPING = {i: e for i, e in enumerate(class_List)}
    REVERSE_LABEL_MAPPING = {e: i for i, e in enumerate(class_List)}
    print("标签映射表:", LABEL_MAPPING)

# 2. 初始化 omlx 模型 (替换原有的 DeepSeek 初始化)
# omlx 默认端口为 8000，且提供 OpenAI 兼容接口
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="Jl123456@@@",  # 本地服务不需要真实 Key
    # model 需对应你 omlx 模型目录下的子文件夹名称
    model="MLX-Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-6bit",
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)
print(f"已连接到本地 omlx 服务器，使用模型: {llm.model_name}")


# 定义带重试机制的 LLM 调用函数
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def invoke_llm(prompt):
    """
    调用本地 omlx 模型，若失败则重试
    """
    return llm.invoke(prompt)


# 读取文件用于分类
def read_file(file_path):
    """
    读取数据集
    """
    df = pd.read_csv(file_path, sep='\t', names=['title', 'label'], header=None, encoding='utf-8')
    return df['title'].tolist(), df['label'].tolist()


# 分类逻辑实现
def local_classifier(title: str):
    """
    通过 omlx 实现本地分类逻辑
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

    # 调用本地模型
    result = invoke_llm(prompt)

    try:
        # 从响应中解析 JSON
        result_dict = json.loads(result.content)
        return result_dict
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"JSON 解析失败: {e}")
        return {
            "category": "社会",
            "reason": f"本地解析失败，默认归为社会。错误: {str(e)}"
        }


if __name__ == '__main__':
    # 读取测试集
    titles, labels = read_file(conf.test_path)

    # 批量处理（先测试前 5 条以验证速度）
    test_titles = titles[:10]
    test_labels = labels[:10]

    print(f"开始本地分类推理，共 {len(test_titles)} 条...")
    results = [local_classifier(title) for title in test_titles]
    # results = local_classifier(test_labels)

    # 计算准确率
    total_num = len(test_titles)
    correct_count = 0

    # 输出结果对比
    print("\n" + "=" * 30)
    pre_labels = []
    for i, result in enumerate(results):
        pred_label = result.get("category", "社会")
        pre_labels.append(REVERSE_LABEL_MAPPING.get(pred_label,5))
        true_label = LABEL_MAPPING[test_labels[i]]


        if pred_label == true_label:
            correct_count += 1

        print(f"标题：{test_titles[i]}")
        print(f"预测：{pred_label} | 真实：{true_label}")
        print(f"原因：{result.get('reason', 'N/A')}")
        print("-" * 20)

    # 评估性能
    print("\n评估指标：")
    report = classification_report(
        pre_labels,
        test_labels,
        # target_names=[LABEL_MAPPING[i] for i in range(10)],
        digits=4
    )
    print(report)

    acc = correct_count / total_num
    print(f"\n测试集准确率：{acc:.2%}")