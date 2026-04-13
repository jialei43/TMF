import datetime
import os

from src.config.bert_config import Config

current_date = datetime.datetime.now().strftime("%Y%m%d")
# 1. 确定当前文件 (bert_config.py) 的绝对路径
# abspath(__file__) 会得到 .../src/config/bert_config.py
# dirname 得到 .../src/config/
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 找到 src 目录 (即 config 的上一级)
src_dir = os.path.dirname(curr_dir)
class Config_Quantization(Config):
    def __init__(self):
        super(Config_Quantization, self).__init__()
        # 量化模型结果保存路径
        self.quantize_model_save_path = os.path.join(src_dir,"model_quantize",
                                                     self.model_name + current_date + "_quantized.pt")

        # 模型量化的时候，放开下一行代码，在CPU运行
        self.device = "cpu"


if __name__ == '__main__':

    conf = Config_Quantization()
    print(conf.model_save_path)
    print(conf.quantize_model_save_path)
    print(conf.device)
    print(conf.bert_config)
    input_size = conf.tokenizer.convert_tokens_to_ids((["你", "好", "中国"]))
    print(input_size)

