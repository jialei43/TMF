import os

from src.config.bert_config import Config, current_time

# 1. 确定当前文件 (bert_config.py) 的绝对路径
# abspath(__file__) 会得到 .../src/config/bert_config.py
# dirname 得到 .../src/config/
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 找到 src 目录 (即 config 的上一级)
src_dir = os.path.dirname(curr_dir)
class Config(Config):
    def __init__(self):
        super(Config, self).__init__()
        # 嵌入维度 128
        self.embed_size = 128
        # LSTM隐藏状态维度 256
        self.lstm_hidden_size = 256
        # LSTM层数 2
        self.num_layers = 2
        # dropout=0.1
        self.dropout = 0.1
        # 蒸馏模型保存路径dir
        # self.distill_model_save_dir = os.path.join(src_dir, "saved_dic")
        self.distill_model_save_path = os.path.join(src_dir, "saved_dic")


if __name__ == '__main__':
    conf = Config()
    print(conf.device)
    print(conf.model_save_path)
