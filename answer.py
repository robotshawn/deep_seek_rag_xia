# answer.py — Ollama non-streaming demo (protect against proxy issues)
import os

# Ensure local Ollama calls don't go through a proxy (must be before importing ollama)
for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(var, None)
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

from ollama import chat

response = chat(
    model="deepseek-r1",
    messages=[{"role": "user", "content": "天空为什么是蓝色的"}],
)
print(response["message"]["content"])

# 参考文档: https://github.com/ollama/ollama-python
