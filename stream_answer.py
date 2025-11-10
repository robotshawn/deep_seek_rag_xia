# stream_answer.py — Ollama streaming demo (protect against proxy issues)
import os

# Ensure local Ollama calls don't go through a proxy (must be before importing ollama)
for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(var, None)
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

from ollama import chat


def getAnswer(question: str):
    stream = chat(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手"},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    print("开始查询")
    while True:
        try:
            user_input = input("\n请输入问题(输入'exit'退出:)")
            if user_input.strip().lower() == "exit":
                break
            getAnswer(user_input)
        except Exception as e:
            print(f"查询错误:{e}")
