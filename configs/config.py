
class Config():
    def __init__(self):
        # 原始数据路径
        self.train_path = r"../01-data/data/train.txt"
        self.test_path = r"../01-data/data/test.txt"
        self.dev_path = r"../01-data/data/dev.txt"
        self.class_path = r"../01-data/data/class.txt"
        self.class_chinese_path = r"../01-data/data/class_chinese"

        # 停用词路径
        self.stopwords_path = r"../01-data/data/stopwords.txt"

        # 处理后的数据路径
        self.process_train_path = r"../02-random_forest/data/train_process.csv"
        self.process_test_path = r"../02-random_forest/data/test_process.csv"
        self.process_dev_path = r"../02-random_forest/data/dev_process.csv"

        # 保存模型路径
        self.rf_model_save_path = r"../02-random_forest/save_models"

        self.rf_model_predict_result = r"../02-random_forest/result"



