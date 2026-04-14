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
        """
        在使用知识蒸馏（Knowledge Distillation）训练模型时，这几行代码起到了关键的作用。我们通常把大模型称为教师模型（Teacher），小模型称为学生模型（Student）。

        在蒸馏的过程中，学生模型不仅要学习教师模型最终输出的分类结果（Logits），有时还需要学习教师模型中间层的“思维过程”。

        这段代码的具体作用可以从以下两个维度来理解：

        1. 提取“软标签”与“中间特征” 🧩
        在 forward 函数中：

        out 是模型最终输出的 Logits（分类概率的前身）。
        
        pooled 是经过 BERT 压缩后的句子级特征向量（隐藏状态）。
        
        当 return_hidden=True 时，模型会同时把这两个东西交给你。
        
        2. 支撑不同的蒸馏损失计算 📐
        知识蒸馏通常涉及两种损失（Loss）：
        
        Logits 蒸馏：学生模型模仿教师模型的分类输出 out。这可以看作是让学生学习老师对不同类别的“打分倾向”。
        
        特征蒸馏 (Feature-based Distillation)：学生模型模仿教师模型的中间特征 pooled。这相当于让学生学习老师是如何理解句子语义的，而不仅仅是看最后的答案。
        
        这是一个非常深刻的见解！在知识蒸馏中，让学生模型模仿教师模型的 pooled 向量（即特征蒸馏）确实具有独特的优势。

        我们来分析一下为什么 pooled 往往能提供比单纯的分类输出 out（Logits）更丰富的信息：
        
        语义深度：pooled 向量包含了 BERT 对整个句子语义的深层理解。如果学生模型能学到这个向量的分布，它就不仅仅是在模仿“答案”，而是在模仿老师的“解题思路” 🧠。
        
        空间约束：out 通常只是一个维度很小的分类向量（比如 2 分类或 10 分类），而 pooled 通常是 768 维。在高维空间中进行对齐，能给学生模型提供更强的约束，减少信息流失 📉
        """
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