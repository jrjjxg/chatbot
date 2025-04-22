# rag_utils.py
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from typing import Optional
import requests # 需要导入 requests 来检查异常类型
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception # 导入 tenacity

from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
# --- 配置常量 (可以考虑从 main.py 或 .env 导入) ---
VECTORSTORE_BASE_PATH = "./db/vectorstores" # 与 main.py 保持一致
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # 与 main.py 保持一致
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ALLOWED_EXTENSIONS = {'.txt', '.pdf'} # 允许上传的文件类型

# --- 初始化 Embedding 模型 (单例模式) ---
_embedding_model = None

# 日志记录器
import logging
logger = logging.getLogger(__name__)

def get_embedding_model():
    """获取或初始化Jina Embedding模型"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("[RAG Utils] 初始化Jina Embedding模型...")
        try:
            # 从环境变量获取API Key
            api_key = os.environ.get("JINA_API_KEY")
            if not api_key:
                raise ValueError("JINA_API_KEY环境变量未设置")

            _embedding_model = JinaEmbeddings(
                jina_api_key=api_key,
                model_name="jina-embeddings-v2-base-zh"  # 使用中文模型
            )
            logger.info("[RAG Utils] Jina Embedding模型初始化成功")
        except Exception as e:
            logger.error(f"[RAG Utils] 初始化Jina embedding模型失败: {e}", exc_info=True)
            raise
    return _embedding_model

def get_vectorstore_path_for_thread_internal(thread_id: str) -> str:
    """内部函数，生成与 app.py 一致的路径"""
    safe_thread_id = "".join(c if c.isalnum() else "_" for c in thread_id)
    return os.path.join(VECTORSTORE_BASE_PATH, f"thread_{safe_thread_id}")

def index_document(file_path: str, thread_id: str) -> bool:
    """
    加载、分割、向量化并索引单个文档到与 thread_id 关联的 ChromaDB 集合中。
    如果已存在同名集合，则会向其中添加文档（注意：ChromaDB 默认行为可能需要调整以避免重复）。
    """
    print(f"[RAG Indexing] Starting indexing for file: {file_path}, thread_id: {thread_id}")
    try:
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            print(f"[RAG Indexing] Error: File type '{file_extension}' not allowed.")
            return False

        # 1. 加载文档
        if file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == '.pdf':
             # 注意: PyPDFLoader 需要 'pypdf' 库
            try:
                loader = PyPDFLoader(file_path)
            except ImportError:
                 print("[RAG Indexing] Error: 'pypdf' library not found. Please install it to handle PDF files.")
                 return False
            except Exception as pdf_err:
                print(f"[RAG Indexing] Error loading PDF {file_path}: {pdf_err}")
                return False
        else:
             # Should not happen due to earlier check, but as safeguard
             print(f"[RAG Indexing] Error: Unsupported file type '{file_extension}'.")
             return False

        documents = loader.load()
        if not documents:
            print(f"[RAG Indexing] Error: No content loaded from file {file_path}")
            return False
        print(f"[RAG Indexing] Loaded {len(documents)} document pages/parts from {file_path}")


        # 2. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        if not splits:
            print(f"[RAG Indexing] Error: Failed to split document {file_path} into chunks.")
            return False
        print(f"[RAG Indexing] Split document into {len(splits)} chunks.")

        # 添加来源元数据，方便后续引用或过滤
        for i, split in enumerate(splits):
            split.metadata["source"] = Path(file_path).name
            split.metadata["chunk_index"] = i # 添加块索引

        # 3. 获取 Embedding 模型
        embeddings = get_embedding_model()

        # 4. 初始化并存入 ChromaDB
        persist_directory = get_vectorstore_path_for_thread_internal(thread_id)
        print(f"[RAG Indexing] Ensuring vector store directory exists: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True) # 确保目录存在

        print(f"[RAG Indexing] Initializing ChromaDB for thread {thread_id} at {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
            # collection_name is implicitly managed by directory in latest Chroma versions if not specified
        )

        print(f"[RAG Indexing] Adding {len(splits)} document chunks to ChromaDB...")
        # 添加文档块。如果集合已存在，默认会添加。
        # 对于更新或去重，ChromaDB有更高级的用法（如 add_texts 指定 ids）
        vectorstore.add_documents(splits)

        # 确保数据持久化 (虽然 add_documents 通常会触发，显式调用更保险)
        # vectorstore.persist() # 在较新版本中，持久化是自动的或通过构造函数参数控制

        print(f"[RAG Indexing] Successfully indexed document {file_path} for thread {thread_id}")
        return True

    except Exception as e:
        print(f"[RAG Indexing] Error during indexing for {file_path} (thread {thread_id}): {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 可选：删除临时保存的上传文件
        # if os.path.exists(file_path):
        #     try:
        #         os.remove(file_path)
        #         print(f"[RAG Indexing] Cleaned up temporary file: {file_path}")
        #     except Exception as e:
        #         print(f"[RAG Indexing] Error cleaning up file {file_path}: {e}")
        pass


def get_retriever_for_thread(thread_id: str, search_kwargs: dict = {"k": 3}) -> Optional[VectorStoreRetriever]:
    """
    根据 thread_id 获取 ChromaDB 向量存储的检索器。
    如果对应的向量存储不存在，则返回 None。
    """
    persist_directory = get_vectorstore_path_for_thread_internal(thread_id)
    print(f"[RAG Retriever] Checking for vector store for thread {thread_id} at {persist_directory}")

    # 检查存储目录是否存在且包含 ChromaDB 文件（简单检查）
    # 更可靠的方式是尝试加载，如果失败则认为不存在
    # if not os.path.exists(persist_directory) or not os.path.exists(os.path.join(persist_directory, 'chroma.sqlite3')):
    #     print(f"[RAG Retriever] Vector store for thread {thread_id} not found.")
    #     return None

    try:
        embeddings = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"[RAG Retriever] Successfully loaded vector store for thread {thread_id}")
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    except Exception as e:
        # 捕捉加载错误，例如目录不存在或损坏
        print(f"[RAG Retriever] Error loading vector store for thread {thread_id}: {e}. Assuming it doesn't exist or is invalid.")
        # 检查是否是特定类型的错误，例如文件不存在
        # import chromadb.errors
        # if isinstance(e,FileNotFoundError) or isinstance(e,chromadb.errors.NotACollectionError):
        #     print(f"[RAG Retriever] Vector store for thread {thread_id} not found.")
        # else:
        #      print(f"[RAG Retriever] Unexpected error loading vector store for thread {thread_id}: {e}")
        return None

def load_document(file_path: str) -> list[Document]:
    """根据文件扩展名加载文档，如无扩展名尝试检测文件类型"""
    logger.info(f"[RAG Utils] Loading document from: {file_path}")
    try:
        # 获取文件扩展名
        file_extension = Path(file_path).suffix.lower()
        
        # 如果没有扩展名，尝试检测文件类型
        if not file_extension:
            # 尝试检测是否为PDF文件（检查文件头部）
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header == b'%PDF':
                        logger.info(f"[RAG Utils] 检测到PDF文件: {file_path}")
                        file_extension = '.pdf'
                    else:
                        # 如果不是PDF，假设为文本文件
                        logger.info(f"[RAG Utils] 无法确定文件类型，尝试作为文本文件加载: {file_path}")
                        file_extension = '.txt'
            except Exception as e:
                logger.error(f"[RAG Utils] 文件类型检测失败: {e}", exc_info=True)
                return []
        
        # 根据扩展名加载文件
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            # 加载文档
            documents = loader.load()
            # 手动确保PDF页码元数据正确设置
            for i, doc in enumerate(documents):
                # 确保page字段存在且为正确类型
                if 'page' not in doc.metadata or not isinstance(doc.metadata['page'], int):
                    # 如果page_number存在，使用它
                    if 'page_number' in doc.metadata:
                        doc.metadata['page'] = doc.metadata['page_number']
                    else:
                        # 否则用索引+1作为页码（从1开始）
                        doc.metadata['page'] = i + 1
                logger.info(f"[RAG Utils] PDF页面 {i+1} 元数据: {doc.metadata}")
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8') # 明确编码
            documents = loader.load()
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            documents = loader.load()
        elif file_extension == '.doc' or file_extension == '.docx':
             # 需要安装 python-docx 和 unstructured
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
            documents = loader.load()
        else:
            logger.warning(f"[RAG Utils] Unsupported file type: {file_path}. Returning empty list.")
            return []
            
        logger.info(f"[RAG Utils] Loaded {len(documents)} document pages/parts from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"[RAG Utils] Failed to load document {file_path}: {e}", exc_info=True)
        return [] # 加载失败返回空列表

def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    """分割文档为更小的块"""
    logger.info(f"[RAG Utils] Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # 添加起始索引元数据
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"[RAG Utils] Split document into {len(chunks)} chunks.")
    return chunks

# --- Tenacity 重试配置 ---
def is_retryable_http_error(exception):
    """检查异常是否是可重试的 HTTP 错误 (特别是 429)"""
    return (
        isinstance(exception, requests.exceptions.HTTPError)
        and exception.response is not None
        and exception.response.status_code == 429 # 只重试 429 Too Many Requests
        # 可以考虑添加 5xx 服务器错误: or exception.response.status_code >= 500
    )

def log_retry_attempt(retry_state):
    """在重试前记录日志"""
    logger.warning(
        f"[RAG Indexing] Rate limit (429) encountered. Retrying in "
        f"{retry_state.next_action.sleep:.2f} seconds... (Attempt {retry_state.attempt_number})"
    )

# --- 内部辅助函数：带重试逻辑的添加文档 --- 
@retry(
    stop=stop_after_attempt(5), # 增加到最多 5 次尝试
    wait=wait_exponential(multiplier=1, min=2, max=15), # 指数退避：2s, 4s, 8s, 15s(capped)... 最多等 15s
    retry=retry_if_exception(is_retryable_http_error), # 只在特定 HTTP 错误时重试
    before_sleep=log_retry_attempt # 重试前打印日志
)
def _add_documents_with_retry(vectorstore, chunks):
    """调用 vectorstore.add_documents 并应用重试逻辑"""
    logger.info(f"[RAG Indexing] Attempting to add {len(chunks)} chunks to vector store (may trigger embedding API calls)...")
    vectorstore.add_documents(chunks)
    logger.info(f"[RAG Indexing] Successfully added {len(chunks)} chunks.")

# --- 主要的索引函数 --- 
def index_document(file_path: str, thread_id: str):
    """加载、分割文档，并使用 FAISS 创建或更新索引 (包含重试逻辑)"""
    index_dir = get_vectorstore_path_for_thread_internal(thread_id)
    faiss_file_path = os.path.join(index_dir, "index.faiss")
    pkl_file_path = os.path.join(index_dir, "index.pkl")

    try:
        logger.info(f"[RAG Indexing] Starting indexing for file: {file_path}, thread_id: {thread_id}")
        documents = load_document(file_path)
        if not documents:
            logger.warning(f"[RAG Indexing] No documents loaded from {file_path}. Skipping indexing.")
            return False # 改为返回False，表示没有实际创建索引

        chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
        if not chunks:
             logger.warning(f"[RAG Indexing] No chunks created from {file_path}. Skipping indexing.")
             return False # 同样改为返回False

        embeddings = get_embedding_model()
        os.makedirs(index_dir, exist_ok=True)

        if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
            logger.info(f"[RAG Indexing] Loading existing FAISS index for thread {thread_id} from {index_dir}")
            vectorstore = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
            logger.info(f"[RAG Indexing] Adding {len(chunks)} new chunks to existing index (with retry)...")
            # vectorstore.add_documents(chunks) # <-- 原来的直接调用
            _add_documents_with_retry(vectorstore, chunks) # <-- 调用带重试的辅助函数
        else:
            logger.info(f"[RAG Indexing] Creating new FAISS index for thread {thread_id} (with retry)...")
            # FAISS.from_documents 也会调用 embed_documents，理论上也应该加重试
            # 但为简化起见，先只给 add_documents 加。如果创建新索引时也报 429，再考虑包装 from_documents
            # vectorstore = FAISS.from_documents(chunks, embeddings)
            # --- 简单包装 from_documents --- 
            @retry(
                stop=stop_after_attempt(5), # 增加到最多 5 次尝试
                wait=wait_exponential(multiplier=1, min=2, max=15), # 指数退避：2s, 4s, 8s, 15s(capped)... 最多等 15s
                retry=retry_if_exception(is_retryable_http_error),
                before_sleep=log_retry_attempt
            )
            def _create_vectorstore_with_retry(chunks_to_embed, embedding_func):
                 logger.info(f"[RAG Indexing] Attempting to create FAISS index from {len(chunks_to_embed)} chunks (may trigger embedding API calls)...")
                 vs = FAISS.from_documents(chunks_to_embed, embedding_func)
                 logger.info(f"[RAG Indexing] Successfully created FAISS index.")
                 return vs
            vectorstore = _create_vectorstore_with_retry(chunks, embeddings)
            # --- 结束包装 --- 

        logger.info(f"[RAG Indexing] Saving FAISS index for thread {thread_id} to {index_dir}")
        vectorstore.save_local(index_dir)

        # 添加验证步骤，确保向量存储确实被创建
        if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
            logger.info(f"[RAG Indexing] 成功验证向量存储创建: {index_dir}")
            logger.info(f"[RAG Indexing] Indexing completed successfully for {file_path} (thread {thread_id}).")
            return True
        else:
            logger.error(f"[RAG Indexing] 向量存储文件未成功创建: {index_dir}")
            return False

    except Exception as e:
        logger.error(f"[RAG Indexing] Error during indexing for {file_path} (thread {thread_id}): {e}", exc_info=True)
        return False # 发生异常，返回False而不是抛出异常，让调用方能更好地处理

def search_index(query: str, thread_id: str, k: int = 5) -> list[Document]:
    """在指定线程的 FAISS 索引中搜索相关文档，并返回带有相关性分数的文档列表"""
    index_dir = get_vectorstore_path_for_thread_internal(thread_id)
    faiss_file_path = os.path.join(index_dir, "index.faiss")
    pkl_file_path = os.path.join(index_dir, "index.pkl")

    logger.info(f"[RAG Search] Searching index for thread {thread_id} with query: '{query[:50]}...'")

    if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
        logger.warning(f"[RAG Search] No index found for thread {thread_id} at {index_dir}. Returning empty list.")
        return []

    try:
        embeddings = get_embedding_model()
        vectorstore = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        # 使用 similarity_search_with_score 获取文档和分数
        results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        # --- 开始：优化文档片段，去除内容相似的片段 ---
        processed_results = []
        seen_content_hashes = set()  # 用于跟踪已处理的内容
        
        # 按相关性分数排序（分数越低表示相关性越高）
        sorted_results = sorted(results_with_scores, key=lambda x: x[1])
        
        for doc, score in sorted_results:
            # 创建内容指纹
            # 提取前150个字符作为内容标识（可调整）
            content_hash = hash(doc.page_content[:150])
            
            # 如果内容已经存在（重复），则跳过
            if content_hash in seen_content_hashes:
                logger.info(f"[RAG Search] Skipping similar document with score {score}")
                continue
                
            # 添加内容指纹到已处理集合
            seen_content_hashes.add(content_hash)
            
            # 将分数添加到元数据中
            doc.metadata["score"] = score 
            processed_results.append(doc)
        # --- 结束：优化文档片段 ---
        
        logger.info(f"[RAG Search] Found {len(processed_results)} relevant documents (with scores) for thread {thread_id}.")
        # 返回包含分数的文档列表
        return processed_results 
    except Exception as e:
        logger.error(f"[RAG Search] Error searching index for thread {thread_id}: {e}", exc_info=True)
        return []



if __name__ == '__main__':
    # 添加一些基本日志配置以便直接运行时看到输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # test_baichuan_embedding()

    # 可以在这里添加更多的测试代码，例如索引一个示例文档和搜索
    # pass

    # 测试Jina Embeddings
    try:
        print("测试Jina Embeddings...")
        model = get_embedding_model()
        test_text = "这是一个测试文本"
        embedding = model.embed_query(test_text)
        print(f"成功生成嵌入向量，维度: {len(embedding)}")
    except Exception as e:
        print(f"测试失败: {e}")
