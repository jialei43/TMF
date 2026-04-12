from tqdm import tqdm
from src.config.bert_config import Config




def load_row_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f,desc="loading data"):
            line = line.strip()
            if not line:
                continue
            text,lable = line.split("\t")
            data.append((text,int(lable)))
    return  data

if __name__ == '__main__':
    conf = Config()
    data = load_row_data(conf.train_path)
    print(data[:10])