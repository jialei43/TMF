import pandas as pd
import pickle
import os
from configs.config import Config
import warnings
warnings.filterwarnings("ignore")

conf = Config()

# 第一步：加载模型及向量化器
print("加载模型及向量化器")
with open(os.path.join(conf.rf_model_save_path,"rf_model.pkl"),"rb") as f:
    model = pickle.load(f)
with open(os.path.join(conf.rf_model_save_path,"tf_idf_vectorizer.pkl"),"rb") as f:
    tfidf = pickle.load(f)

# 第二步：读取dev数据
print("读取dev数据")
dev_df = pd.read_csv(conf.process_dev_path)

# 第三步：通过tfidf向量器，转换为数值特征
print("转换dev数据为数值...")
dev_features = tfidf.transform(dev_df['words'])

# 第四步：进行模型预测与保存
print("进行预测")
dev_predicts = model.predict(dev_features)

# 保存预测结果
print("保存预测结果")
output_df = pd.DataFrame({"words":dev_df['words'],"predictions":dev_predicts})

# 结果保存到result中
output_path = os.path.join(conf.rf_model_predict_result,'dev_predictions.csv')
output_df.to_csv(output_path,index=False)
print(f"预测结果已保存到 {output_path}")
print("预测结果前5行：")
print(output_df.head(5))