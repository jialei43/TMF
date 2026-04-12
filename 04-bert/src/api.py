

from flask import Flask, request, jsonify
from predict import model_predict

# 创建flash应用
app = Flask(__name__)

# 创建路由
@app.route('/predict',methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or not data.get('text'):
        return jsonify({"error": "请求必须包含text字段，值为字符串或者字符串列表"}), 400
    texts = data.get('text')
    if not isinstance(texts, (list, str)):
        return jsonify({"error": "text字段的值必须是字符串或者字符串列表"}), 400
    for text in texts:
        if not isinstance(text, str):
            return jsonify({"error": "文本内容必须是字符串"}), 400

    result = model_predict(texts)
    if not result:
        return jsonify({"error": "预测结果为空"}), 500
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004)

