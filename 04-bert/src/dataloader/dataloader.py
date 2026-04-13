import torch
from torch.utils.data import DataLoader

from src.config.bert_config import Config
from src.dataloader.dataset import TextDataset
from src.dataloader.reader import load_row_data


class DataLoaderCls():
    def __init__(self, conf= Config):
        self.conf = conf

    def collate_fn(self, batch):
        # 提取文本和标签
        texts = [line[0] for line in batch]
        lables = [line[1] for line in batch]

        # 批量分词，自动添加[CLS]和[SEP]
        text_tokens = self.conf.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.conf.padding_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt" # 返回张量
         )

        token_ids_list = text_tokens["input_ids"].to(self.conf.device)
        toekn_attention_mask_list = text_tokens["attention_mask"].to(self.conf.device)
        # 转换为Tensor并添加到设备
        # token_ids_list = torch.tensor(token_ids_list, device=self.conf.device)
        # toekn_attention_mask_list = torch.tensor(toekn_attention_mask_list, device=self.conf.device)
        lables = torch.tensor(lables, device=self.conf.device)

        return token_ids_list, toekn_attention_mask_list, lables

    def build_dataloader(self):
        """
        根据配置信息构建模型训练所需的数据集。
        参数：
        - config (object): 配置信息对象，包含有关数据集和模型的相关参数。
        返回：
        - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
        """
        train_data = load_row_data(self.conf.train_path)
        dev_data = load_row_data(self.conf.dev_path)
        test_data = load_row_data(self.conf.test_path)

        # 构建数据集dataset
        train_dataset = TextDataset(train_data)
        dev_dataset = TextDataset(dev_data)
        test_dataset = TextDataset(test_data)

        # 创建data_loader
        train_dataloader = DataLoader(train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=self.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.conf.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=self.conf.batch_size, shuffle=False, collate_fn=self.collate_fn)
        return train_dataloader, dev_dataloader, test_dataloader

if __name__ == '__main__':
    conf = Config()
    data_loader_cls = DataLoaderCls(conf)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_cls.build_dataloader()
    for batch in train_dataloader:
        print(batch)
        input_ids, attention_mask, labels = batch
        print(input_ids)
        print(input_ids.shape)
        print(attention_mask.shape)
        print(labels.shape)
        break
