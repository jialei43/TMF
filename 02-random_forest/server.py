from flask import Flask, request, jsonify
import pickle
import jieba
from configs.config import Config
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
conf = Config()

print("加载模型和向量化器...")
with open(conf.rf_model_save_path + '/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(conf.rf_model_save_path + '/tf_idf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

print("加载标签映射...")
with open(conf.class_chinese_path, 'r') as f:
    label_names = [line.strip() for line in f if line.strip()]
label_map = {i: name for i, name in enumerate(label_names)}
print("标签映射：", label_map)


def cut_sentence(s):
    """对输入文本进行结巴分词，并限制前30个词"""
    return ' '.join(list(jieba.cut(s))[:30])


@app.route("/predict", methods={"POST"})
def predict():
    try:
        data = request.get_json()
        sentence = data.get('sentence', "")
        if not sentence:
            return jsonify({"error": "请输入句子"}), 400
        processed_sentence = cut_sentence((sentence))
        print("处理后的句子：", processed_sentence)
        if isinstance(processed_sentence, str):
            processed_sentence = [processed_sentence]
        features = tfidf.transform(processed_sentence)
        print("TF-IDF特征形状：", features.shape, "非零特征数：", features.nnz)
        if features.nnz == 0:
            return jsonify({"error": "输入句子无有效特征，请检查输入或停用词"}), 400
        prediction = model.predict(features)[0]
        print("模型预测值：", prediction)

        predicted_label = label_map.get(int(prediction), "未知标签")
        print("模型预测值的映射：", predicted_label)

        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2222)