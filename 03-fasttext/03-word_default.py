# 导入工具包
import fasttext
from configs.config import Config
import datetime
import os

#获取时间
current_time=datetime.datetime.now().date().today().strftime("%Y%m%d")
#1、导入配置文件
conf=Config()

# 2、模型训练
model = fasttext.train_supervised(
    #无监督情况下 wordNgrams等同于word2vec
    input=conf.train_ft_jieba_path
)

#3、模型保存
model.save_model(os.path.join(conf.ft_model_save_path,f"fastText_jieba_default_{str(current_time)}.bin"))

#4、模型预测
print(model.predict("名师 详解 考研 复试 英语听力 备考 策略"))

# 5、模型词表查看
# print(model.words)
# print(len(model.words))

# 6、查看模型子词，上述训练未开启子词，所以这里查到还是词本身
print(model.get_subwords('你好'))

# 7、模型测试
print("模型测试验证评估开始...")
res = model.test(conf.dev_ft_jieba_path)
print(res)