import os
from configparser import ConfigParser
from pathlib import Path

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

config = ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config["deepseek"]["KEY"]
os.environ["OPENAI_API_BASE"] = config["deepseek"]["BASE_URL"]
print(os.environ["OPENAI_API_KEY"])
print(os.environ["OPENAI_API_BASE"])

MODEL_NAME = "GanymedeNil/text2vec-large-chinese"
LOCAL_MODEL_NAME = config["deepseek_model"]["MODEL"]
LOCAL_MODEL_PATH = config["deepseek_model"]["LOCAL_MODEL_PATH"]

faiss_index = "faiss_index"


def get_vector_knowledge(path, use_local_embeddings=True):
    # print(path)
    """1. 构建知识库"""
    document = TextLoader(path, "utf-8", True).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]
    )

    texts = text_splitter.split_documents(document)
    print(f"知识库分割完成，{len(texts)}个文本块")
    # print(texts[0])

    if use_local_embeddings and Path(LOCAL_MODEL_NAME).exists():
        print("使用本地嵌入模型")
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_NAME,
            cache_folder=LOCAL_MODEL_PATH,
        )
    else:
        print("下载嵌入模型")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            cache_folder=LOCAL_MODEL_PATH,
            model_kwargs={"device": "gpu"}
        )
    print("构建向量索引")
    vector_store = FAISS.from_documents(
        documents=texts,
        embedding=embeddings,
    )

    # 保存向量索引
    vector_store.save_local(faiss_index)
    print(f"向量数据库已保存至{faiss_index}目录")
    return vector_store


"""2.构建DeepseekRagService"""


class DeepseekRagService:
    def __init__(self, use_local_embeddings: bool = True):
        # 加载中文嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_NAME,
            cache_folder=LOCAL_MODEL_PATH,
        )

        # 加载faiss向量库
        self.vector_store = FAISS.load_local(
            folder_path=faiss_index,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 配置deepseek模型
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3, # 用于调节输出随机性和确定性的核心超参数,创意任务‌（如小说创作、诗歌生成）：建议使用较高值（1.0-1.5），激发模型发散性思维,精确任务‌（如代码生成、知识问答）：推荐较低值（0.3-0.7），保证输出的一致性和准确性
            max_tokens=1024,
            streaming=False
        )

        self.prompt_template = """基于以下上下文件，请用中文回答问题。如果不知道，请诚实回答不知道。
        上下文：
        {context}
        用户输入：
        问题：{question}
        """

        # 设置prompt模型
        self.qa_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # 创建检索链
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="mmr", # 指定检索类型为最大边际相关性，该策略在返回结果时不仅考虑文档与查询的相关性，还会平衡结果多样性，避免返回内容重复的文档
                search_kwargs={"k": 5, "fetch_k": 20} # fetch_k先获取与查询最相似的候选文档集，k,从中筛选中既相关又多样化的子集
            ),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        """
        执行查询
        Args:
            question: 用户问题
        Returns:
            包含答案和来源的字典
        """
        return self.retrieval_chain.invoke({"query": question})


if __name__ == "__main__":
    # r执行过一次，后面可以注释掉，此处可单独优化为一次调用
    # get_vector_knowledge(os.path.join(os.getcwd(), "deepseek_rag/doc/ai_knowledge.txt"), use_local_embeddings=True)
    # 创建知识库
    print("初始化RAG服务……")
    try:
        rag_service = DeepseekRagService(use_local_embeddings=True)
    except Exception as e:
        print(f"初始化失败{e}")
        print("尝试重建知识库")
        get_vector_knowledge(os.path.join(os.getcwd(), "deepseek_rag/doc/ai_knowledge.txt"), use_local_embeddings=True)
        rag_service = DeepseekRagService(use_local_embeddings=True)

    # 交互试查询
    while (True):
        try:
            user_input = input("\n请输入您的问题(退出则输入exit)：")
            if user_input == "exit":
                break
            print("处理中……")
            result = rag_service.query(user_input)
            print("\n" + "=" * 50)
            print(f"问题:{user_input}")
            print(f"答案：{result["result"]}")
            print("\n来源文档：")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"文档 {i}")
                print(f"来源{doc.matedate["source"]}")
                print(f"文档摘要{doc.page_content[:150]}……")
            print("=n" * 50 + "\n")
        except Exception as e:
            print(f"查询有误{e}")
