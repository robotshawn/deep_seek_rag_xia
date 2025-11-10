# retriever.py — LangChain 0.2.16 + DeepSeek API + FAISS RAG
import os
from pathlib import Path
from configparser import ConfigParser

# ---------- proxy hardening (fix socks:// & bypass deepseek/localhost) ----------
def _patch_proxies_for_httpx_and_openai():
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        v = os.environ.get(key)
        if v and v.strip().lower().startswith("socks://"):
            os.environ[key] = "socks5://" + v.strip()[len("socks://"):]
    must_bypass = {"api.deepseek.com", "127.0.0.1", "localhost"}
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged = {h.strip() for h in existing.split(",") if h.strip()}
    merged |= must_bypass
    new_no_proxy = ",".join(sorted(merged))
    os.environ["NO_PROXY"] = new_no_proxy
    os.environ["no_proxy"] = new_no_proxy

_patch_proxies_for_httpx_and_openai()
# -------------------------------------------------------------------------------

cfg = ConfigParser()
if cfg.read("config.ini") and "deepseek" in cfg:
    os.environ.setdefault("OPENAI_API_KEY", cfg["deepseek"]["KEY"])
    os.environ.setdefault("OPENAI_API_BASE", cfg["deepseek"]["BASE_URL"])

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # 0.2.x OK
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def _pick_device() -> str:
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


MODEL_NAME = os.getenv("EMBEDDING_MODEL", cfg.get("embeddings", "MODEL", fallback="GanymedeNil/text2vec-large-chinese"))
INDEX_DIR = "faiss_index"
DOC_PATH = os.getenv("DOC_PATH", cfg.get("data", "DOC_PATH", fallback="deepseek_rag/doc/ai_knowledge.txt"))


def prepare_knowledge_base(path: str = DOC_PATH):
    documents = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""],
    )
    texts = splitter.split_documents(documents)
    print(f"知识库分割完成: {len(texts)} 个文本块")

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": _pick_device()})
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(INDEX_DIR)
    print(f"FAISS 向量索引已保存至 {INDEX_DIR}")
    return vector_store


class DeepSeekRAGService:
    def __init__(self, index_path: str = INDEX_DIR):
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": _pick_device()})
        self.vector_store = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, max_tokens=1024, streaming=False)

        prompt_tmpl = """基于以下上下文信息，请用中文回答问题。如果你不知道答案，请诚实回答不知道。
上下文:
{context}
问题: {question}
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
    print("初始化RAG服务...")
    if not Path(INDEX_DIR).exists():
        print("未找到索引，开始构建...")
        prepare_knowledge_base(DOC_PATH)

    rag = DeepSeekRAGService()
    while True:
        try:
            q = input("\n请输入问题 (输入'exit'退出): ").strip()
            if q.lower() == "exit":
                break
            print("处理中...")
            res = rag.query(q)
            print("\n" + "=" * 50)
            print(f"问题: {q}")
            print(f"答案: {res['result']}")
            print("\n来源文档:")
            for i, doc in enumerate(res["source_documents"], 1):
                print(f"文档 {i}:")
                print(f"  来源: {doc.metadata.get('source')}")
                print(f"  内容摘要: {doc.page_content[:150]}...")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"查询错误: {e}")
