import encodings
import os
from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 配置环境变量
os.environ["OPENAI_API_KEY"] = "sk-44768b72523841f1b0b574963d91b35d"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/api/generate"

# 模型下载路径（本地缓存）
# MODEL_NAME = "GanymedeNil/text2vec-large-chinese"
MODEL_NAME = "D:/hugging_face/models--GanymedeNil--text2vec-large-chinese/snapshots/add4e02ec45da97442798f25093d9ab96e40c5ad"
LOCAL_MODEL_PATH = "D:/hugging_face"


# 1. 知识库准备
def prepare_knowledge_base(data_path: str = "D:/projects/python/ai_practice/deepseek_rag/doc",
                           use_local_embeddings: bool = True):
    """
    加载并处理知识库文档
    Returns:
        FAISS向量存储对象
    """

    # 加载文档
    # loader = DirectoryLoader(
    #     path=data_path,
    #     glob="**/*.txt",
    #     loader_cls=TextLoader,
    #     show_progress=True,
    # )
    # print("===========================")
    # loader.load().encoding="utf-8"
    # documents = loader.load(})
    path = os.path.join(os.getcwd(), "deepseek_rag/doc/ai_knowledge.txt")
    print(path)
    documents = TextLoader(path, "utf-8", True).load()
    print(documents)
    # 中文文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]
    )

    # 分割文档
    texts = text_splitter.split_documents(documents)

    # 中文嵌入模型
    if use_local_embeddings and Path(LOCAL_MODEL_PATH).exists():
        print("使用本地嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            cache_folder=LOCAL_MODEL_PATH
        )
    else:
        print("下载嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            cache_folder=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'gpu'}
        )

    print(f"知识库分割完成: {len(texts)}个文本块")
    # 创建FAISS向量存储
    print("构建向量索引...")
    vector_store = FAISS.from_documents(
        documents=texts,
        embedding=embeddings
    )

    # 保存向量索引
    vector_store.save_local("faiss_index")
    print("FAISS向量索引已保存至 faiss_index 目录")
    return vector_store


# 2. RAG服务构建
class DeepSeekRAGService:
    def __init__(self, index_path="faiss_index", use_local_embeddings: bool = True):
        """
        初始化RAG服务
        Args:
            index_path: FAISS索引目录路径
            use_local_embeddings: 是否使用本地嵌入模型
        """

        # 加载中文嵌入模型
        if use_local_embeddings and Path(LOCAL_MODEL_PATH).exists():
            print("加载本地嵌入模型...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                cache_folder=LOCAL_MODEL_PATH
            )
        else:
            print("使用远程嵌入模型...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="GanymedeNil/text2vec-large-chinese"
            )

        # 加载FAISS向量库
        print("加载FAISS向量库...")
        self.vector_store = FAISS.load_local(
            folder_path=index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 配置DeepSeek模型
        print("配置DeepSeek模型...")
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3, # 控制生成文本随机性，取值范围通常0.1（高度确定性）-1.5（高度随机性）;低温度（如0.2）锐化分布，偏向高概率词；高温度（如1.2）平滑分布，增加多样性
            max_tokens=1024,
            streaming=False
        )

        # 中文提示模板
        print("加载中文提示模板...")
        self.prompt_template = """基于以下上下文信息，请用中文回答问题。如果你不知道答案，请诚实回答不知道。
        上下文:
        {context}
        问题: {question}
        回答:"""

        self.qa_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # 创建检索链
        print("正在创建检索链...")
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever( # 该方法将向量存储（VectorStore）转换为检索器（Retriever）对象，使其能够执行文档检索操作。检索器的核心功能是接受查询文本输入，返回与查询语义最相关的文档列表。
                search_type="mmr", #指定检索类型为最大边际相关性（Maximal Marginal Relevance，MMR),该策略返回结果时不仅考虑文档与查询的相关性，还会考虑文件多样性，避免返回重复文档。
                search_kwargs={"k": 5, "fetch_k": 20} # 先获取与查询最相似的候选文档集（由fetch_k参数控制数量）。从中筛选出既相关又多样化的子集（由k参数控制最终返回数量）。
            ),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
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


# 3. 主程序
if __name__ == "__main__":
    # os.makedirs("./models", exist_ok=True)

    # 第一步：准备知识库 (首次运行需要执行,之后直接注释掉就可以)
    # prepare_knowledge_base(use_local_embeddings=True)

    # 第二步：初始化RAG服务
    print("初始化RAG服务...")
    try:
        rag_service = DeepSeekRAGService(use_local_embeddings=True)
        print("服务初始化成功!")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("尝试重建知识库...")
        prepare_knowledge_base(use_local_embeddings=True)
        rag_service = DeepSeekRAGService(use_local_embeddings=True)

    # 第三步：交互式查询
    while True:
        try:
            user_input = input("\n请输入问题 (输入'exit'退出): ")
            if user_input.lower() == 'exit':
                break

            print("处理中...")
            result = rag_service.query(user_input)

            # 打印结果
            print("\n" + "=" * 50)
            print(f"问题: {user_input}")
            print(f"答案: {result['result']}")
            print("\n来源文档:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"文档 {i}:")
                print(f"  来源: {doc.metadata['source']}")
                print(f"  内容摘要: {doc.page_content[:150]}...")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"查询错误: {e}")
