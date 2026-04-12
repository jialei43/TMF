import torch

from src.config.bert_config import Config
from src.models.bert import BERT

conf = Config()
def model_predict(texts):
    # 模型初始化
    model = BERT(conf)

    # 模型加载
    model.load_state_dict(torch.load(conf.model_save_path,map_location=conf.device))
    # 模型映射待device上
    model.to(conf.device)
    # 模型转为验证模型
    model.eval()
    # 获取输入 确保输入texts为列表，为了兼容只输入一个文本的情况，不需要再针对单个样本进行逻辑编写
    if isinstance(texts,str):
        texts = [texts]

    # 初始化分词器
    tokenizer = conf.tokenizer

    input_ids_list = []
    attention_mask_list = []
    # 处理文本
    for text in texts:
        if not text.strip():
            print("输入文本为空")
            break
        encoder = tokenizer(text,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=conf.padding_size,
                                   return_attention_mask=True,
                                   return_tensors="pt")
        # 获取词向量 list里面存储的是多个二维张量  encoder["input_ids"]-》tensor
        input_ids_list.append(encoder["input_ids"])
        # 获取位置编码
        attention_mask_list.append(encoder["attention_mask"])

    # 如果没有有效返回，返回空结果集
    if not input_ids_list:
        print("没有有效输入文本")
        return []

    #合并批次张量  list中多个张亮拼接生成一个二张量
    input_ids = torch.cat(input_ids_list,dim=0).to(conf.device)
    attention_mask = torch.cat(attention_mask_list,dim=0).to(conf.device)

    # 模型推理
    with torch.no_grad():
        # 计算logits
        logits = model(input_ids,attention_mask)
        # 获取概率
        probs = torch.softmax(logits, dim=-1)
        # 获取概率最大的值的索引
        preds = torch.argmax(probs, dim=-1)

    # 整理结果
    result=[]
    for i, (pred,prob) in enumerate(zip(preds, probs)):
        # 获取预测的id索引
        class_idx = pred.item()
        # 获取预测的类别名称
        class_name = conf.class_List[class_idx]
        # 获取预测的概率
        class_prob = prob[class_idx].item()
        result.append({
            "text": texts[i],
            "class_name": class_name,
            "class_prob": class_prob
        })

    return result

if __name__ == '__main__':
    # 待分析的文本
    input_text = ['体验2D巅峰 倚天屠龙记十大创新概览',
                  '郭晶晶夺得金牌',
                  '成龙成功出道，饰演黑道大哥',
                  '中央一号文件正式下发关注三农问题',
                  '伊朗凌晨对以色列发起总攻',
                  '美伊战争，霍尔木兹海峡成为全世界的焦点']
    result = model_predict(input_text)
    for e in result:
        print(e)
