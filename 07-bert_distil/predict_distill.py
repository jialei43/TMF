import torch
from transformers import BertTokenizer
from configs.LSTMConfig import Config
import numpy as np
from models.bilstmClassifier import BiLstmClassifier
import os

# 初始化配置
lstmConf = Config()

def predict_distilled(texts,model_path=None,device=None):
    """
    使用蒸馏后的BiLSTMClassifier模型对输入文本进行预测
    参数：
        texts (str or list): 输入文本，单个字符串或字符串列表。
        model_path (str, optional): 模型权重路径，默认为 conf.save_model_path3 + "\\bilstm_distilled.pth"。
        device (str, optional): 设备（"cuda"或"cpu"），默认为 conf.device。

    返回：
        list: 每个文本的预测结果，包含类别索引、类别名称和概率。
    """
    # 设置设备
    device = device if device else lstmConf.device

    # 设置模型路径
    model_path = os.path.join(lstmConf.distill_model_save_path,"intermediate_layer_distill_model20250612.pt")

    # 加载模型
    model = BiLstmClassifier()
    model.load_state_dict((torch.load(model_path)))
    model = model.to(device)
    model.eval()

    # 确保输入texts处理为列表
    if isinstance(texts,str):
        texts = [texts]

    # 初始化分词器
    tokenizer = lstmConf.tokenizer

    # 处理文本
    input_ids_list = []
    attention_mask_list = []

    # 分词并编码，添加[CLS]和[SEP]，不填充
    encoded = tokenizer.encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids_list.append(encoded['input_ids'])
    attention_mask_list.append(encoded['attention_mask'])

    # 如果没有有效输入，返回空结果
    if not input_ids_list:
        print("错误：没有有效输入")
        return []

    # 合并为批次张量
    input_ids = torch.cat(input_ids_list,dim=0).to(device)
    attention_mask = torch.cat(attention_mask_list,dim=0).to(device)

    # 模型推理
    with torch.no_grad():
        # 前向传播，返回维度：[batch_size,num_classes]
        logits = model(input_ids,attention_mask)
        # 转换为概率
        probs = torch.softmax(logits,dim=1)
        # 预测类别索引
        preds = torch.argmax(probs,dim=1)

    # 整理结果
    results = []
    for i,(pred,prob) in enumerate(zip(preds,probs)):
        # 获取类别id
        class_idx = pred.item()
        #映射到类别名称
        class_name = lstmConf.class_list[class_idx]
        # 预测类别的概率
        class_prob = prob[class_idx].item()
        results.append({
            "text":texts[i],
            "class_index":class_idx,
            "class_name":class_name,
            "prob":class_prob
        })
    return results

if __name__=="__main__":
    sample_text = "体验2D巅峰 倚天屠龙记十大创新概览"

    # 进行预测
    results = predict_distilled(sample_text)

    # 打印结果
    print("预测结果：")
    for result in results:
        print(f"文本: {result['text']}")
        print(f"预测类别索引: {result['class_index']}")
        print(f"预测类别名称: {result['class_name']}")
        print(f"预测概率: {result['prob']:.4f}")