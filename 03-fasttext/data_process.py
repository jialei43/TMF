import os

import jieba

from configs.config import Config
from pathlib import Path
conf = Config()

# 步骤 1：检查文件存在性
if not os.path.exists(conf.train_path) or not os.path.exists(conf.test_path) \
        or not os.path.exists(conf.dev_path) or not os.path.exists(conf.class_path):
    print("文件不存在，请检查文件是否存在")


id2name = {}
print(f"类别文件：{conf.class_chinese_path}")
with open(conf.class_chinese_path,'r',encoding='utf-8') as f:
    for idx,line in enumerate(f):
        id2name[idx] = line.strip()
print(f"类别映射：{id2name}")

def build_ft_data(path,use_jieba=False):
    datas = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 以制表符分割
            text,label = line.split("\t")
            # 转换标签，如 __label__education
            label_name = f"__label__{id2name[int(label)]}"
            # 部分文本含有冒号，这里移除冒号
            text = text.replace(":","")
            # use_jieba True，则是jieba ,是词级分词；为False，则是字级分词
            if use_jieba:
                words = jieba.lcut(text)
            else:
                words = list(text)
            # 拼接分词结果
            text_processed = " ".join(words)

            ft_line = f"{label_name} {text_processed}"
            datas.append(ft_line)
    return datas

def build_and_save_ft_data(input_path,output_path,use_jieba=False):
    data = build_ft_data(input_path, use_jieba=use_jieba)
    # file_path =Path(output_path)
    # touch() 方法：如果文件不存在则创建，如果存在则不执行任何操作
    # exist_ok=True 类似于 Linux 的 touch 命令
    # file_path.touch(exist_ok=True)
    # print(f"文件 {file_path} 已准备就绪")
    with open(output_path,'w',encoding='utf-8') as f:
        f.write('\n'.join(data))

if __name__=="__main__":
    # 字级别数据构造
    build_and_save_ft_data(conf.train_path,conf.train_ft_char_path)
    build_and_save_ft_data(conf.test_path, conf.test_ft_char_path)
    build_and_save_ft_data(conf.dev_path, conf.dev_ft_char_path)
    # 词级别数据构造
    build_and_save_ft_data(conf.train_path, conf.train_ft_jieba_path,use_jieba=True)
    build_and_save_ft_data(conf.test_path, conf.test_ft_jieba_path, use_jieba=True)
    build_and_save_ft_data(conf.dev_path, conf.dev_ft_jieba_path, use_jieba=True)
