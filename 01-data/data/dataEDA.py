"""
数据分析
"""
from collections import Counter

import pandas as pd

from config import Config

# 导入配置类，并获取训练数据集的路径
config = Config()
train_datapath = config.train_datapath
# 读取文件，且查看基本信息
train_data = pd.read_csv(train_datapath, sep='\t', names=['text','lable'],header=None)
print(train_data.head(5))

# 统计标签分布
counter_lables = Counter(train_data['lable'])
for lable,count  in counter_lables.items():
    print(f'标签：{lable},次数： {count}')
# 计算标签比例
#计算总行数
total_count = len(train_data)
for lable,count  in counter_lables.items():
    proportion = count/total_count
    print(f'标签：{lable},比例： {proportion}')

#计算文本长度
train_data['text_len'] = train_data['text'].apply(lambda x:len(x))
print(train_data.head(5))
print(train_data[['text', 'text_len']].head(10))
print(f"平均长度：{train_data['text_len'].mean():.2f} 字符")  # 平均值
print(f"长度标准差：{train_data['text_len'].std():.2f} 字符")  # 标准差
print(f"最大长度：{train_data['text_len'].max()} 字符")  # 最大值
print(f"最小长度：{train_data['text_len'].min()} 字符")  # 最小值