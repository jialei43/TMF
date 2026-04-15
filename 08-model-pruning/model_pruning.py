import torch
import torch.nn.utils.prune as prune

from models.bert import BERT
from src.config.bert_config import Config
from src.dataloader.dataloader import DataLoaderCls
from src.train_eval import evaluate

conf = Config()

# 计算所有encoder层query的稀疏度
def compute_sparsity(model):
    # 计算所有参数
    total_params = 0
    # 统计所有参数为0的参数
    zero_params = 0
    # 获取所有的encoder层
    layers = conf.bert_config.num_hidden_layers
    # 遍历所有层
    for i in range(layers):
        # 获取权重矩阵
        weight = model.bert.encoder.layer[i].attention.self.query.weight
        # 所有权重数
        total_params += weight.numel()
        # 获取权重为0的参数
        zero_params += (weight == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0

# 打印权重矩阵的前rows * cols 部分
def print_weights(weight,name,rows=5,cols=5):
    print(f"\n{name} (前{rows} * {cols})")
    print(weight[:rows,:cols])

def main():
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取数据迭代器
    train_loader,test_loader,dev_loader = DataLoaderCls(conf).build_dataloader()

    # 加载模型
    model = BERT(conf)
    state_dict = torch.load(conf.model_save_path)
    model.load_state_dict(state_dict,strict=False)
    model.to(device)

    # 剪枝前
    print("剪枝前模型：")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(model.bert.encoder.layer[0].attention.self.query.weight,"layer[0].attention.self.query.weight 剪枝前")
    accuracy, precision,report, f1score = evaluate(model,dev_loader,conf)
    print(f"\n剪枝前准确率：{accuracy:.4f},F1:{f1score:.4f}")

    # 全局非结构化剪枝：所有encoder层query权重30%
    # 待剪枝参数
    parameters_to_prune = [
        (model.bert.encoder.layer[i].attention.self.query,"weight") for i in range(12)
    ]
    # 整体剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3
    )
    for module,param in parameters_to_prune:
        prune.remove(module,param)
    # 剪枝后
    print("\n剪枝后模型：")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(model.bert.encoder.layer[0].attention.self.query.weight,"layer[0].attention.self.query.weight 剪枝后")
    accuracy, precision,report, f1score = evaluate(model,dev_loader,conf)
    sparsity = compute_sparsity(model)
    print(f"\n剪枝后准确率：{accuracy:.4f},F1:{f1score:.4f}\n稀疏度：{sparsity:.4f}")

if __name__=="__main__":
    main()