from torch import nn
from transformers import BertModel
from src.config.bert_config import Config
from src.dataloader.dataloader import DataLoaderCls


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrain_bert_dir,config=config.bert_config)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        _,pooler = self.bert(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        logits = self.classifier(pooler)
        return logits


if __name__ == '__main__':
    conf = Config()
    model = BERT(conf)
    model.to(conf.device)
    data_loader_cls = DataLoaderCls(model.config)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_cls.build_dataloader()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids.to(conf.device)
        attention_mask.to(conf.device)
        labels.to(conf.device)
        logits = model(input_ids, attention_mask)
        print(logits)
        print(logits.shape)
        break