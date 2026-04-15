import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
import sys, os

from models.bilstmClassifier import BiLstmClassifier
from src.dataloader.dataloader import DataLoaderCls
from src.models.bert import BERT

sys.path.append(os.path.dirname(os.getcwd()))
from configs.LSTMConfig import Config
from src.train_eval import evaluate
import time
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")
lstmConf = Config()


# print(vars(lstmConf))
def model_train(teacher_model, student_model, train_loader, dev_loader, num_epochs, learning_rate, device, save_path):
    """
    训练学生模型（BiLSTM）使用硬标签蒸馏，学习教师模型（BERT）的预测类别
    参数：
        teacher_model: 教师模型（BERT），提供硬标签。
        student_model: 学生模型（BiLSTM），需要学习教师模型的预测。
        train_loader: 训练数据加载器，提供训练数据批次。
        dev_loader: 验证数据加载器，提供验证数据批次。
        num_epochs: 训练的总轮数（epoch 数量）。
        learning_rate: 学习率，控制优化器更新步长。
        device: 训练设备（"cuda" 或 "cpu"）。
        save_path: 模型保存路径，保存最佳模型权重。
    """
    # 将模型移动到指定设备（GPU或者CPU）
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 初始化优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)

    # 交叉熵损失，用于硬标签损失
    criterion = nn.CrossEntropyLoss()

    # 温度参数T,用于软标签蒸馏
    T = 2.0

    # 软标签和硬标签损失的权重
    alpha = 0.7

    # 记录最佳验证f1分数
    best_dev_f1 = 0.0

    # 训练步数计数器
    step = 0
    # 早停耐心值patience 3
    patience = 5

    # 记录未提升的 epoch 数
    epochs_no_improve = 0

    # 打印训练参数
    print("训练参数 Training Parameters:")
    print(
        f"num_epochs: {num_epochs}, learning_rate: {learning_rate}, device: {device}, batch_size: {train_loader.batch_size}")

    # 遍历每一个epoch
    for epoch in range(num_epochs):
        # 设置学生模型为训练模式
        student_model.train()
        # 设置教师模型为评估模式（不更新权重）
        teacher_model.eval()
        # 记录当前epoch的总损失
        total_loss = 0
        # 记录训练预测和真实标签
        train_preds, train_labels = [], []
        # 记录epoch开始时间
        epoch_start_time = time.time()

        print(f"\n软标签蒸馏训练 Soft Label Distillation Epoch {epoch + 1} / {num_epochs}...")
        # 遍历训练批次
        for batch in tqdm(train_loader, desc=f"Soft Label Distillation Epoch {epoch + 1} / {num_epochs}"):
            # 记录当前step开始时间
            step_start_time = time.time()
            # 获取输入数据
            input_ids, attention_mask, labels = batch
            # 映射到device
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # 获取教师模型的预测（硬标签）
            # 关闭梯度运算
            with torch.no_grad():
                # 计算教师模型的概率分布
                teacher_logits = teacher_model(input_ids, attention_mask)
                # 获取教师模型的预测值
                teacher_preds = torch.argmax(teacher_logits, dim=1)

            # 获取学生模型的输出logits
            student_logits = student_model(input_ids, attention_mask)

            # 计算软标签损失（KL散度）
            # 教师模型的概率分布
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            # 学生模型的概率分布
            student_probs = F.log_softmax(student_logits / T, dim=1)
            # 软标签损失=KL散度损失
            soft_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (T * T)

            # 计算硬标签损失（交叉熵，使用教师模型的预测）
            hard_loss = criterion(student_logits, teacher_preds)
            # 总损失：软标签损失和硬标签损失的加权和
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            # 反向传播
            loss.backward()

            # 梯度更新
            optimizer.step()

            # 累加损失
            total_loss += loss.item()

            # 记录预测结果
            preds = torch.argmax(student_logits, dim=1)

            # 将结果添加到train_preds,train_labels中
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # 步数+1
            step += 1

            # 计算step耗时
            step_duration = time.time() - step_start_time

            # 每10个step验证一次
            if step % 200 == 0:
                # 切换到评估模式
                student_model.eval()
                # 平均loss
                avg_loss = total_loss / (len(train_preds) / train_loader.batch_size)
                # 使用评估函数获取评估值 accuracy,precision,report,f1score
                accuracy, precision,report, f1score  = evaluate(student_model, dev_loader)
                print(f"Step {step}, Epoch {epoch + 1}/{num_epochs}")
                print(f"Step Duration: {step_duration:.2f} seconds")
                print(f"Train Loss: {avg_loss:.4f}")
                print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
                print(f"Dev Precision: {precision:.4f}")
                print("Dev Classification Report:")
                print(report)
                # 学生模型切换回训练模式
                student_model.train()
                # 保存最佳模型并检查早停
                if f1score > best_dev_f1:
                    best_dev_f1 = f1score
                    torch.save(student_model.state_dict(), save_path)
                    print("模型保存！")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"dev f1未提升，当前未提升epoch数：{epochs_no_improve} / {patience}")
                    if epochs_no_improve >= patience and epoch > 5:
                        print(f"早停触发！ dev f1 在{patience} 个epoch内未提升，停止训练。")
                        break

        # 计算训练集指标
        train_report = classification_report(train_labels, train_preds, target_names=lstmConf.class_List,
                                             output_dict=True)
        train_f1 = train_report["weighted avg"]["f1-score"]

        # 验证（每个epoch结束时）
        student_model.eval()
        accuracy, precision,report, f1score = evaluate(student_model, dev_loader)
        print(f"Train Loss: {total_loss / len(train_loader):.4f}, Train F1: {train_f1:.4f}")
        print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
        print(f"Dev Precision: {precision:.4f}")
        print("Dev Classification Report:")
        print(report)

        # 计算epoch耗时
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")

        # 保存最佳模型并检查早停
        # if f1score > best_dev_f1:
        #     best_dev_f1 = f1score
        #     torch.save(student_model.state_dict(), save_path)
        #     print("模型保存！")
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     print(f"dev f1未提升，当前未提升epoch数：{epochs_no_improve} / {patience}")
        #     if epochs_no_improve >= patience:
        #         print(f"早停触发！ dev f1 在{patience} 个epoch内未提升，停止训练。")
        #         break

        student_model.train()


if __name__ == '__main__':

    strudent_model = BiLstmClassifier()
    teacher_model = BERT(config=lstmConf)
    teacher_model.load_state_dict(torch.load(lstmConf.model_save_path))
    dataloader = DataLoaderCls(lstmConf)
    train_dataloader, dev_dataloader, test_dataloader = dataloader.build_dataloader()
    model_train(
        strudent_model,
        teacher_model,
        train_dataloader,
        dev_dataloader,
        lstmConf.num_epochs,
        lstmConf.learning_rate,
        lstmConf.device,
        lstmConf.distill_model_save_dir,
    )

