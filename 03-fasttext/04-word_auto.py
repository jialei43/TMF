# 导入工具包
import fasttext
from configs.config import Config
import datetime
import os


# 获取当前日期
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 1. 导入配置文件
conf = Config()

# 2. 模型训练
model = fasttext.train_supervised(
    input=conf.train_ft_jieba_path,
    autotuneValidationFile=conf.dev_ft_jieba_path,
    autotuneDuration=300,  # 设置调参时间 5 分钟
    # autotuneModelSize='2M',  # 目标模型大小 2MB（触发量化）
    thread=1,  # 单线程，确保可复现性
    verbose=3  # 输出调参过程
)

# 3. 模型保存
model_save_path = os.path.join(conf.ft_model_save_path,f"fastText_jieba_auto_{str(current_time)}.bin")
model.save_model(model_save_path)

print(f"模型已保存至: {model_save_path}")

# 4. 模型预测
sentence = "俄 达吉斯坦 共和国 一名 区长 被 枪杀"
pred_label, pred_prob = model.predict(sentence)
print(f"预测结果: 标签={pred_label[0]}, 概率={pred_prob[0]:.4f}")

# 5. 查看模型子词
word = "你好"
subwords, subword_ids = model.get_subwords(word)
print(f"词 '{word}' 的子词: {subwords}")
print(f"子词 ID: {subword_ids}")

#获取词向量维度
print(model.get_dimension())

# 6. 模型测试
res = model.test(conf.dev_ft_jieba_path)
print(f"测试结果: 样本数={res[0]}, 精确率={res[1]:.4f}, 召回率={res[2]:.4f}")