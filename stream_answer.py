from ollama import chat


def getAnswer(question):
    stream = chat(
        model="deepseek-r1",
        messages=[
            {
                "role": "system",
                "content": "你是一个有用的AI助手"
            },
            {
                "role": "user",
                "content": question
            }
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


if __name__ == '__main__':
    print("开始查询")
    while True:
        try:
            user_input = input("\n请输入问题(输入'exit'退出:)")
            if user_input.lower() == 'exit':
                break
            getAnswer(user_input)
        except Exception as e:
            print(f"查询错误:{e}")
