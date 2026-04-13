import torch

from configs.config import Config_Quantization
from src.dataloader.dataloader import DataLoaderCls
from src.models.bert import BERT
from src.train_eval import evaluate

if __name__ == '__main__':
    # --- 新增：为 Mac M1/M2 (ARM) 设置量化后端 ---
    torch.backends.quantized.engine = 'qnnpack'
    conf = Config_Quantization()
    # 加载数据
    data_loader_cls = DataLoaderCls(conf)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_cls.build_dataloader()
    # 加载模型
    model = BERT(conf)
    print(model)
    model.to(conf.device)
    weight = torch.load(conf.model_save_path)
    model.load_state_dict(weight)
    # 模型转换为验证模式
    model.eval()

    # 模型量化
    quantize_dynamic_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # 检查量化模型中各层的参数数据类型
    print(quantize_dynamic_model)

    # 测试量化后的模型
    report,f1score,accuracy,precision = evaluate(quantize_dynamic_model, test_dataloader, conf)
    print("Test Classification Report:", report)
    print("Test F1:", f1score)
    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)

    # 保存整个量化模型
    torch.save(quantize_dynamic_model, conf.quantize_model_save_path)
    print("保存量化模型成功！保存路径为：", conf.quantize_model_save_path)