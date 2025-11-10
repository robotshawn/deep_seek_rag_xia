# main.py — LangChain 0.2.16 + DeepSeek API + FAISS RAG
import os
from configparser import ConfigParser
from pathlib import Path

# ---------- proxy hardening (fix socks:// & bypass deepseek/localhost) ----------
def _patch_proxies_for_httpx_and_openai():
    # Normalize unsupported scheme "socks://" -> "socks5://"
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        v = os.environ.get(key)
        if v and v.strip().lower().startswith("socks://"):
            os.environ[key] = "socks5://" + v.strip()[len("socks://"):]
    # Always bypass these hosts
    must_bypass = {"api.deepseek.com", "127.0.0.1", "localhost"}
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged = {h.strip() for h in existing.split(",") if h.strip()}
    merged |= must_bypass
    new_no_proxy = ",".join(sorted(merged))
    os.environ["NO_PROXY"] = new_no_proxy
    os.environ["no_proxy"] = new_no_proxy

_patch_proxies_for_httpx_and_openai()
# -------------------------------------------------------------------------------

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # 0.2.x OK (shows deprecation warning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def _pick_device() -> str:
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


config = ConfigParser()
config.read("config.ini")

os.environ["OPENAI_API_KEY"] = config["deepseek"]["KEY"]
os.environ["OPENAI_API_BASE"] = config["deepseek"]["BASE_URL"]

MODEL_NAME = config.get("embeddings", "MODEL", fallback="GanymedeNil/text2vec-large-chinese")
INDEX_DIR = "faiss_index"
DOC_PATH = config.get("data", "DOC_PATH", fallback="deepseek_rag/doc/ai_knowledge.txt")


def get_vector_knowledge(path: str):
    docs = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""],
    )
    texts = splitter.split_documents(docs)
    print(f"知识库分割完成，{len(texts)} 个文本块")

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": _pick_device()})
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(INDEX_DIR)
    print(f"向量数据库已保存至 {INDEX_DIR} 目录")
    return vector_store


class DeepseekRagService:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": _pick_device()})
        self.vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

        self.llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, max_tokens=1024, streaming=False)

        prompt_tmpl = """基于以下上下文，请用中文回答问题；若不知道，请回答不知道。
上下文：
{context}
用户输入：
问题：{question}
回答："""
        qa_prompt = PromptTemplate(template=prompt_tmpl, input_variables=["context", "question"])

        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        return self.retrieval_chain.invoke({"query": question})


if __name__ == "__main__":
    print("初始化RAG服务……")
    if not Path(INDEX_DIR).exists():
        print("未发现向量索引，开始构建……")
        get_vector_knowledge(DOC_PATH)

    rag_service = DeepseekRagService()
    while True:
        try:
            user_input = input("\n请输入您的问题(退出则输入exit)：").strip()
            if user_input.lower() == "exit":
                break
            print("处理中……")
            result = rag_service.query(user_input)
            print("\n" + "=" * 50)
            print(f"问题: {user_input}")
            print(f"答案：{result['result']}")
            print("\n来源文档：")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"文档 {i}")
                print(f"来源: {doc.metadata.get('source')}")
                print(f"文档摘要: {doc.page_content[:150]}……")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"查询有误: {e}")
