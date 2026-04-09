"""
将文本切词后保存到新的文件，切词的文本使用空格进行拼接字符串进行保存
"""

import pandas as pd
import jieba
from configs.config import Config
conf = Config()

# 第一步：读取数据
def load_data(path):
    data = pd.read_csv(path, sep="\t", names=["text", "label"])
    return data

# 第二步：进行分词预处理
def cut_sentence(text):
    """对输入文本进行结巴分词，获取前30个词并用空格连接"""
    return " ".join(jieba.lcut(text)[:30])
def data_add_words(data):
    data['words'] = data['text'].apply(cut_sentence)
    return data

# 第三步：保存数据
def save_data(data,path):
    data.to_csv(path, index=False)

def pipeline(input_data_path,output_data_path):
    data = load_data(input_data_path)
    data = data_add_words(data)
    save_data(data, output_data_path)

if __name__=="__main__":
    # 处理train
    pipeline(conf.train_path,conf.process_train_path)
    # 处理test
    pipeline(conf.test_path,conf.process_test_path)
    # 处理dev
    pipeline(conf.dev_path,conf.process_dev_path)