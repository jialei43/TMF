import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from configs.config import Config
from tqdm import tqdm
import os

conf = Config()

# 第一步：读取数据
train = pd.read_csv(conf.process_train_path)[:20000]
words = train['words']
labels = train['label']

# 第二步：将文本转换为数值特征
# 1.读取停用词
stopwords = open(conf.stopwords_path,encoding='utf-8').read().split()

# 2.tfidf实例化
tfidf = TfidfVectorizer(stop_words=stopwords)

# 3.训练
features = tfidf.fit_transform(words)

# 查看特征，features是稀疏矩阵，稀疏矩阵的格式是(row,col) value
print(features)

# 查看特征维度 1000行 4687
"""
含义： 它返回一个元组 (行数, 列数)。
行数 (1000)： 代表你输入的文档（或句子）的总数。
列数 (4687)： 代表词袋中所有不重复词汇的总数（去重后的单词量）。
"""
print(features.shape)
# 查看特征名称  这是词表里的所有单词，按照字母顺序排列。
print(list(tfidf.get_feature_names_out()))
# 特征词的总个数  这个数字应该和 features.shape 的第二个数（列数）完全一致。
print(len(tfidf.get_feature_names_out()))
# 查看特征名称和索引
"""
输出内容： 特征词与索引的对应字典。
含义： 这是一个 Python 字典（dict），键（Key）是单词，值（Value）是该词在特征矩阵中对应的列索引（Index）。
注意： 这里的数字不是词频，而是该词在矩阵中排在第几列。
示例： {'python': 302, 'machine': 150} 表示“python”这个词的信息存储在矩阵的第 302 列。
"""
print(tfidf.vocabulary_)
# 查看特征名称的个数  词汇表字典的大小
print(len(tfidf.vocabulary_))

print("-"*34)

# 第三步：模型训练和评估
# 1、划分数据集
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=22)

# 2.rf训练
model = RandomForestClassifier()
print("训练模型。。。")
# 使用tqdm包装model.fit来显示进度条
for _ in tqdm(range(1),desc="RandomForest模型训练进度..."):
    model.fit(x_train,y_train)

# 3.模型预测并评估
print("模型预测评估...")
y_pred = model.predict(x_test)
print(f"预测结果：{y_pred}")
# 评估
# 准确率 (真正列+真反列)/样本总数
print(f"准确率：{accuracy_score(y_test,y_pred)}")
# 精确率 正样本预测正确的数量 / 预测为正类的数量
print("精确率 (micro):", precision_score(y_test, y_pred, average='micro'))
# 召回率 正类中实际被预测正确的数量 / 正类中的数量
print("召回率 (micro):", recall_score(y_test, y_pred, average='micro'))
# F1分数 2*(精确率*召回率)/(精确率+召回率)  P=\frac {2*Precision*Recall}{Precison+Recall}=\frac {2倍精确率*召回率}{精确率+召回率}
print("F1分数 (micro):", f1_score(y_test, y_pred, average='micro'))

# 第四步：模型保存
print("保存模型和向量化器...")
with open(os.path.join(conf.rf_model_save_path,"rf_model.pkl"),'wb') as f:
    pickle.dump(model,f)

with open(os.path.join(conf.rf_model_save_path,"tf_idf_vectorizer.pkl"),'wb') as f:
    pickle.dump(tfidf,f)

print("模型和向量化器，保存成功！")