import requests
import time
from utils import get_time_dif

# 定义请求url和传入的data
url = "http://127.0.0.1:8004/predict"
data = {"text": "体验2D巅峰 倚天屠龙记十大创新概览"}

start_time = time.time()
# 向服务发送post请求
res = requests.post(url, json=data)
cost_time = get_time_dif(start_time)
print(f'res:{res}')
# 打印返回的结果
print('文本类别: ', res.json())
print(f'单条样本耗时: {cost_time:.1f}ms')