import os

import fasttext
from flask import Flask, request, jsonify
import jieba
from configs.config import Config
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
conf = Config()


print("加载模型和向量化器...")
model = fasttext.load_model(os.path.join(conf.ft_model_save_path, f"fasttext_char_auto_{str(20260410)}.bin"))



# 定义路由
@app.route("/predict",methods=["POST"])
def main_server():
    try:
        data = request.get_json()
        sentence = data.get('text',"")
        if not sentence:
            return jsonify({"error":"请输入句子"}),400
        processed_sentence = " ".join(jieba.lcut((sentence)))
        res = model.predict(processed_sentence)
        # 获取预测结果
        # res = {tuple: 2} (('__label__股票',), [0.90697235])
        res = res[0][0][9:] #res[0][0].replace("__label__","")

        return jsonify({"predicted_label":res})

    except Exception as e:
        print(e)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8003,debug= True)