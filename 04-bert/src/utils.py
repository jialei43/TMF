import time
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间，返回格式化的字符串"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif