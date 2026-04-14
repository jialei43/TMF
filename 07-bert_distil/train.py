# 忽略警告信息
import datetime
import os
import time
import warnings

import torch
from sklearn.metrics import classification_report

from configs.LSTMConfig import Config
from models.bilstmClassifier import BiLstmClassifier
from src.dataloader.dataloader import DataLoaderCls
from models.bert import BERT
from src.train_eval import evaluate
import hard_label_distillation
import soft_label_distillation
import intermediate_layer_distillation
from src.utils import get_time_dif

warnings.filterwarnings("ignore")
lstmConf = Config()
current_time=datetime.datetime.now().date().today().strftime("%Y%m%d")

def train(kd_type):
    """
    模型训练
    输入：
    kd_type:蒸馏方式，取值有3个，“hard"表示硬标签蒸馏，”soft“表示软标签蒸馏，”inter“表示中间层蒸馏
    """
    # 获取device
    device = lstmConf.device
    # 加载教师模型
    teacher_model = BERT(lstmConf)
    teacher_model.load_state_dict(torch.load(lstmConf.model_save_path))
    teacher_model = teacher_model.to(device)

    # 加载学生模型
    student_model = BiLstmClassifier()

    # 加载数据迭代器
    dloader = DataLoaderCls()
    train_dataloader, test_dataloader, dev_dataloader = dloader.build_dataloader()
    # 获取epochs
    num_epochs = lstmConf.num_epochs

    # 获取学习率
    learning_rate = lstmConf.learning_rate

    # 模型保存路径dir
    distill_model_save_dir_path = lstmConf.distill_model_save_path

    # 开始蒸馏
    if kd_type == 'hard':
        # 模型保存完整路径
        distill_model_save_path = os.path.join(distill_model_save_dir_path,"hard_distill_model" + current_time + ".pt")
        # 硬标签蒸馏训练
        hard_label_distillation.model_train(teacher_model,student_model,train_dataloader,dev_dataloader,num_epochs,learning_rate,device,distill_model_save_path)
    elif kd_type == "soft":
        # 模型保存完整路径
        distill_model_save_path = os.path.join(distill_model_save_dir_path,"soft_distill_model" + current_time + ".pt")
        soft_label_distillation.model_train(teacher_model,student_model,train_dataloader,dev_dataloader,num_epochs,learning_rate,device,distill_model_save_path)
        # 软标签蒸馏训练
    elif kd_type == "inter":
        # 模型保存完整路径
        distill_model_save_path = os.path.join(distill_model_save_dir_path,"intermediate_layer_distill_model" + current_time + ".pt")
        intermediate_layer_distillation.model_train(teacher_model,student_model,train_dataloader,dev_dataloader,num_epochs,learning_rate,device,distill_model_save_path)
        # 软标签蒸馏训练
    else:
        print("蒸馏方式不在既定策略中，请重新确认蒸馏方式kd_type")
        return
    student_model.load_state_dict(torch.load(distill_model_save_path))
    student_model = student_model.to(device)
    # 测试集评估模型
    accuracy, precision,report, f1score = evaluate(student_model, test_dataloader)
    print("Test Set Evaluation:")
    print(f"Test F1: {f1score:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Test Classification Report:")
    print(classification_report)
    return

if __name__ == "__main__":
    start_time = time.time()
    # 蒸馏类型：hard,soft,inter
    kd_type = "inter"
    train(kd_type)
    print(f"Training complete in {get_time_dif(start_time)}")
