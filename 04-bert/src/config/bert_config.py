import datetime
import os

import torch
from transformers import BertTokenizer, BertConfig

current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
# 1. 确定当前文件 (bert_config.py) 的绝对路径
# abspath(__file__) 会得到 .../src/config/bert_config.py
# dirname 得到 .../src/config/
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 找到 src 目录 (即 config 的上一级)
src_dir = os.path.dirname(curr_dir)
class Config():

    def __init__(self):
        """
        配置类：包含模型和训练需要的各种参数
        """
        # 模型名称
        self.model_name = "bert"
        # 训练集
        self.train_path = os.path.join(src_dir, "data", "train.txt")
        #验证集
        self.dev_path = os.path.join(src_dir, "data", "dev.txt")
        # 测试集
        self.test_path = os.path.join(src_dir, "data", "test.txt")
        # 类别名单
        self.class_path = os.path.join(src_dir, "data", "class.txt")
        self.class_chinese_path = os.path.join(src_dir, "data", "class_chinese")
        # 类别名称
        self.class_List = []
        with open(self.class_chinese_path ,"r", encoding="utf-8") as f:
            self.class_List = [line.strip() for line in f]

        # 模型结果保存地址
        self.model_save_dir = os.path.join(src_dir, "saved_dic")
        # 如果模型结果保存地址不存在，就创建
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        # 模型保存地址
        self.model_save_path = os.path.join(self.model_save_dir, self.model_name+current_time+".pt")
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("使用设备：", self.device)

        #类别数
        self.num_classes = len(self.class_List)
        # epoch数
        self.num_epochs = 10
        # mini-batch 大小
        self.batch_size = 128
        # 每句话处理的长度（短填长切）
        self.padding_size = 32
        # 学习率
        self.learning_rate = 5e-5
        # 预训练bert的模型路径
        self.pretrain_bert_dir = os.path.join(src_dir, "bert-base-chinese")
        # bert模型的分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_bert_dir)
        # bert模型的配置文件
        self.bert_config = BertConfig.from_pretrained(os.path.join(self.pretrain_bert_dir,"config.json"))
        # bert模型隐藏层大小
        self.hidden_size = self.bert_config.hidden_size

if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    print(conf.class_List)
    input_size = conf.tokenizer.convert_tokens_to_ids((["你", "好", "中国"]))
    print(input_size)
    print(conf.train_path)
    print(conf.pretrain_bert_dir)


