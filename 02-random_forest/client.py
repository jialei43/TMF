import requests

if __name__ == "__main__":
    # 配置 API 地址
    url = "http://127.0.0.1:2222/predict"

    print("--- 文本分类预测系统 (输入 'quit' 退出) ---")

    while True:
        # 1. 获取键盘输入
        user_input = input("\n请输入要预测的句子: ").strip()

        # 2. 设置退出条件
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("程序已退出。")
            break

        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        try:
            # 3. 构造请求数据
            input_data = {"sentence": user_input}

            # 4. 调用 Flask API
            response = requests.post(
                url,
                json=input_data,
                timeout=5  # 设置超时保护
            )

            # 5. 解析结果
            response.raise_for_status()
            result = response.json()

            if 'predicted_label' in result:
                print(f"✅ 预测结果：{result['predicted_label']}")
            else:
                print("⚠️ 返回结果中未包含 predicted_label 字段")

        except requests.exceptions.ConnectionError:
            print("❌ 错误：无法连接到服务器，请检查 Flask 服务是否已启动 (127.0.0.1:2222)")
        except Exception as e:
            print(f"❌ 发生错误：{str(e)}")