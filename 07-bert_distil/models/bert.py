from torch import nn
from transformers import BertModel
from src.config.bert_config import Config
from src.dataloader.dataloader import DataLoaderCls

conf = Config()


class BERT(nn.Module):
    def __init__(self, config=conf):
        super(BERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrain_bert_dir,config=config.bert_config)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask,return_hidden=False):
        """
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :return:
        """
        """
        在 Hugging Face 的 transformers 库中，当你调用 BertModel 并且设置 return_dict=False 时，它会返回一个 元组 (Tuple)。
        1、last_hidden_state (对应代码中的 _)
            形状 (Shape): [batch_size, sequence_length, hidden_size]
            含义: 这是 BERT 模型最后一层输出的每个 Token 的隐藏状态（特征向量）。
            用途: 通常用于序列标注任务（如 NER）或需要获取每个词语上下文表示的任务
        2、pooler_output (对应代码中的 pooler)
            形状 (Shape): [batch_size, hidden_size]
            含义: 这是对序列第一个 Token（即 [CLS] 标记）的隐藏状态进一步处理后的结果。
            处理过程:    1. 取出 last_hidden_state 中第一个位置的向量。
                        2. 通过一个全连接层（nn.Linear）。
                        3. 经过一个 Tanh 激活函数
            
        """
        _,pooler = self.bert(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        logits = self.classifier(pooler)
        if return_hidden:
            return logits, pooler
        return logits

if __name__ == '__main__':
    model = BERT(conf)
    model.to(conf.device)
    train_dataloader, test_dataloader, dev_dataloader = DataLoaderCls(conf).build_dataloader()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        outputs, hidden = model(input_ids, attention_mask, return_hidden=True)
        print(outputs.shape, hidden.shape)
        breakpoint()