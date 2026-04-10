import fasttext
from configs.config import Config
import datetime
import os
current_time=datetime.datetime.now().date().today().strftime("%Y%m%d")
conf=Config()
def train():
    # 获取时间


    #1、模型训练
    print(conf.train_ft_char_path)
    model = fasttext.train_supervised(
        input = conf.train_ft_char_path,
        minn=2,
        maxn=3,
        # epoch=100,
        # lr=0.1,
        # wordNgrams=2,
        # thread=4,
        # verbose=2,
        # loss="ova"
    )

    # 2、模型保存
    model.save_model(os.path.join(conf.ft_model_save_path,f"fasttext_char_default_{str(current_time)}.bin"))

    # 3、模型预测
    print(model.predict("《 赤 壁 O L 》 攻 城 战 诸 侯 战 硝 烟 又 起"))

    # 4、模型词表查看
    print("查看模型词表[:10]:")
    print(model.words[:10])

    # 5、查看模型子词，上述训练未开启子词，所以这里查到还是词本身
    print(model.get_subwords("你好"))

    # 6、模型测试
    res = model.test(conf.test_ft_char_path)
    print(res)

    print(model.get_subwords('中国北京'))

def predict():
    model = fasttext.load_model(os.path.join(conf.ft_model_save_path, f"fasttext_char_default_{str(current_time)}.bin"))
    # res = model.test(conf.test_ft_char_path)
    res = model.test(conf.dev_ft_char_path)
    print(res)
    print(model.predict("新 三国 正式 上线 好礼 送 不 断"))
    print(model.get_subwords('中国北京'))
    print(model.get_subwords('中华女子学院'))
if __name__ == '__main__':
    # train()
    predict()