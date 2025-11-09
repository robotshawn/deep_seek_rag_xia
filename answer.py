import requests
from ollama import chat,ChatResponse

# res = requests.post('http://localhost:11434/api/generate', json={
#     "model": "deepseek-r1:latest",
#     "prompt": "量子力学的基本原理是什么？",
#     "stream": False
# })
# print("=============")
# print(res)

response: ChatResponse=chat(model="deepseek-r1",messages=[
    {
        'role':'user',
        'content':'天空为什么是蓝色的',
    },
])

print(response['message']['content'])

"""
参考文档: https://github.com/ollama/ollama-python
"""