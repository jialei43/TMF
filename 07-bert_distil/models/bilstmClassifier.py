from torch import nn
from configs.LSTMConfig import Config
from src.dataloader.dataloader import DataLoaderCls

lstm_config = Config()
class BiLstmClassifier(nn.Module):
    def __init__(self, config= lstm_config):
        super(BiLstmClassifier, self).__init__()
        # 配置参数
        self.config = config
        # 词嵌入层
        self.embedding = nn.Embedding(config.bert_config.vocab_size, config.embed_size)
        # LSTM层 bidirectional=True 双向lstm
        self.lstm = nn.LSTM(config.embed_size, config.lstm_hidden_size, config.num_layers,
                            bidirectional=True, dropout=config.dropout, batch_first=True)
        # 隐藏层映射:映射到BERT隐藏状态维度(hidden_projection)
        self.hidden_projection = nn.Linear(config.lstm_hidden_size * 2, config.hidden_size)
        # 全连接层：映射到最终输出的类别
        self.fc = nn.Linear(config.lstm_hidden_size * 2, config.num_classes)
        # dropout层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask,return_hidden=False):
        """
        前向传播，仅在嵌入层使用attention_mask进行掩码处理
        参数：
            input_ids: 输入的token ID，形状为 [batch_size, seq_len]。
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]，1 表示有效 token，0 表示填充 token。
            return_hidden: 是否返回隐藏状态。
        返回：
            logits: 分类logits，形状为 [batch_size, num_classes]。
            hidden: 最后一时间步的隐藏状态（若 return_hidden=True），形状为 [batch_size, hidden_size*2]。
        """
        # 词嵌入
        # 嵌入层 -> [batch_size,seq_len,embed_size]
        embed = self.embedding(input_ids)

        # 使用attention_mask 掩码填充token的嵌入 -> [batch_size, seq_len, 1]
        embed = embed * attention_mask.unsqueeze(-1)

        # LSTM层:# lstm_out,(hidde,c) -> [batch_size, seq_len, hidden_size*2]
        lstm_out, (hidde, c) = self.lstm(embed)
        # 获取最后一时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        # 隐藏层映射 -> [batch_size, bert.hidden_size]
        if return_hidden:
            hidden = self.hidden_projection(last_hidden)
            return out, hidden
        # 输出层
        return  out


if __name__ == '__main__':
    print(lstm_config.bert_config.vocab_size)
    print(lstm_config.embed_size)
    model = BiLstmClassifier()
    model.to(lstm_config.device)
    print(model)
    dataloader= DataLoaderCls(lstm_config)
    train_dataloader, dev_dataloader, test_dataloader = dataloader.build_dataloader()
    for token_ids_list, toekn_attention_mask_list, lables in dev_dataloader:
        out = model(token_ids_list, toekn_attention_mask_list, return_hidden=True)
        print(out)
        print(out[0].shape,out[1].shape)
        break