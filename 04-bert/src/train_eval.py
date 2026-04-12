import time

import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.config.bert_config import Config
from src.dataloader.dataloader import DataLoaderCls
from src.models.bert import BERT
from src.utils import get_time_dif

conf = Config()
def train_model(model,train_loader,dev_loader,conf=conf):
    """
    模型训练函数。
    参数：
    - config: 配置信息对象。
    - model: 待训练的模型。
    - train_iter: 训练集的迭代器。
    - dev_iter: 验证集的迭代器。
    """

    # 初始化模型和优化器
    model = model.to(conf.device)
    model.to(conf.device)
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    # 初始化f1值
    best_dev_f1 = 0.0
    best_acc = 0.0
    for epoch in range(conf.num_epochs):
        # 将模型设置为训练模式
        model.train()
        total_batch = 0
        total_loss = 0
        train_preds,train_lables=[],[]
        for batch_iter,batch in enumerate(tqdm(train_loader,desc=f"Bert Classifier Training Epoch {epoch + 1} / {conf.num_epochs}...")):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算总损失
            total_loss += loss.item()
            total_batch += 1
            # 获取预测值
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_lables.extend(labels.cpu().numpy())

            # 每100个batch打印一次损失和准确率
            if batch_iter % 100 == 0 and batch_iter != 0:
                print(f"Epoch {epoch + 1} / {conf.num_epochs}")
                print(f"Train Loss:{total_loss / (total_batch + 1):.4f}")
                # 训练集指标
                f1 = f1_score(train_lables, train_preds, average='macro', zero_division=0)
                acc = accuracy_score(train_lables, train_preds)
                print(f'train f1:{f1:.4f}')
                print(f"train acc:{acc:.4f}")
                # 验证集
                # cls_report, acc, precision, recall, f1 = eval_model(model, dev_dataloader)
                acc, precision, recall, f1 = evaluate(model, dev_loader)
                print(f'dev f1:{f1:.4f}')
                print(f"dev acc:{acc:.4f}")
                # 验证集上acc有提升保存模型
                if acc > best_acc:
                    # 模型保存
                    torch.save(model.state_dict(), conf.model_save_path)
                    best_acc = acc
                    nonimprove = 0
                else:
                    # 无提升自增
                    nonimprove += 1
                # 连续5次不提升,提前停止
                if nonimprove > 5:
                    print(f"已连续5次不提升，提前停止训练")
                    return '提前停止'

def evaluate(model,data_loader,conf=conf):
    model = model.to(conf.device)
    model.eval()
    preds,true_labels = [],[]
    with torch.no_grad():
        # 遍历数据
        for batch in tqdm(data_loader,desc="Evaluating ..."):
            # 获取input_ids,attention_mask,labels
            input_ids,attention_mask,labels = batch
            # 映射到device上
            input_ids,attention_mask,labels = input_ids.to(conf.device),attention_mask.to(conf.device),labels.to(conf.device)
            # 计算logits
            logits = model(input_ids,attention_mask)
            # 获取概率最大的值
            batch_preds =torch.argmax(logits,dim=1)
            # 将预测值加入到模型预测列表preds中
            preds.extend(batch_preds.cpu().numpy())
            # 将标签加入到真值列表true_label中
            true_labels.extend(labels.cpu().numpy())
        # 计算分类报告
        report = classification_report(true_labels,preds,target_names=conf.class_List,output_dict=True,zero_division=1)
        # 计算f1
        f1score = f1_score(true_labels,preds,average="macro",zero_division=1)
        # 计算准确率
        accuracy = accuracy_score(true_labels,preds)
        # 计算precision
        precision = precision_score(true_labels,preds,average="macro",zero_division=1)
    return accuracy,precision,report,f1score


if __name__=="__main__":
    # 加载数据
    train_loader,test_loader,dev_loader = DataLoaderCls(conf).build_dataloader()
    print(f'train_loader:{len(train_loader)}')
    # 初始化模型
    model = BERT(conf)
    # 训练和验证
    # 训练开始时间
    start_time = time.time()
    # 开始训练模型
    train_model(model,train_loader,dev_loader,conf)
    print(f"Training completed in {get_time_dif(start_time)}")

    # 测试集评估
    report, f1score, accuracy, precision = evaluate(model,dev_loader,conf)
    print("Test Set Evaluation:")
    print(f"Test F1: {f1score:.4f}")
    # print("Test Classification Report:")





