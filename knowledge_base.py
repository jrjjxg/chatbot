# knowledge_base.py
import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime

from langchain_core.documents import Document

# 导入现有RAG工具中的函数
from rag_utils import (
    get_embedding_model, 
    load_document, 
    split_documents, 
    ALLOWED_EXTENSIONS
)

# 配置日志
logger = logging.getLogger(__name__)

# 知识库根目录
KB_BASE_PATH = "./db/knowledge_bases"

# 确保知识库根目录存在
os.makedirs(KB_BASE_PATH, exist_ok=True)

class KnowledgeBase:
    """知识库类，封装知识库管理功能"""
    
    def __init__(self, kb_id: str = None):
        """初始化知识库对象
        
        Args:
            kb_id: 知识库ID，如果不提供，则创建新知识库
        """
        if kb_id:
            self.kb_id = kb_id
            self._load_metadata()
        else:
            self.kb_id = f"kb_{uuid.uuid4().hex[:8]}"
            self.metadata = {
                "id": self.kb_id,
                "name": "",
                "description": "",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "",
                "document_count": 0,
                "documents": []
            }
    
    @property
    def kb_path(self) -> str:
        """获取知识库路径"""
        return os.path.join(KB_BASE_PATH, self.kb_id)
    
    @property
    def docs_path(self) -> str:
        """获取知识库文档目录路径"""
        return os.path.join(self.kb_path, "docs")
    
    @property
    def index_path(self) -> str:
        """获取知识库索引目录路径"""
        return os.path.join(self.kb_path, "index")
    
    @property 
    def metadata_path(self) -> str:
        """获取知识库元数据文件路径"""
        return os.path.join(self.kb_path, "metadata.json")
    
    def _load_metadata(self):
        """从文件加载知识库元数据"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                logger.warning(f"知识库 {self.kb_id} 的元数据文件不存在")
                self.metadata = {
                    "id": self.kb_id,
                    "name": "未命名知识库",
                    "description": "",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "created_by": "",
                    "document_count": 0,
                    "documents": []
                }
        except Exception as e:
            logger.error(f"加载知识库 {self.kb_id} 元数据失败: {e}")
            raise
    
    def _save_metadata(self):
        """保存知识库元数据到文件"""
        try:
            os.makedirs(self.kb_path, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"知识库 {self.kb_id} 元数据保存成功")
        except Exception as e:
            logger.error(f"保存知识库 {self.kb_id} 元数据失败: {e}")
            raise
    
    def create(self, name: str, description: str, user_id: str) -> str:
        """创建知识库
        
        Args:
            name: 知识库名称
            description: 知识库描述
            user_id: 创建者ID
            
        Returns:
            知识库ID
        """
        try:
            # 更新元数据
            self.metadata["name"] = name
            self.metadata["description"] = description
            self.metadata["created_by"] = user_id
            
            # 创建目录结构
            os.makedirs(self.kb_path, exist_ok=True)
            os.makedirs(self.docs_path, exist_ok=True)
            os.makedirs(self.index_path, exist_ok=True)
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"创建知识库成功: {self.kb_id}")
            return self.kb_id
        except Exception as e:
            logger.error(f"创建知识库失败: {e}")
            raise
    
    def update(self, name: str = None, description: str = None) -> bool:
        """更新知识库信息
        
        Args:
            name: 知识库名称
            description: 知识库描述
            
        Returns:
            更新是否成功
        """
        try:
            # 更新元数据
            if name is not None:
                self.metadata["name"] = name
            if description is not None:
                self.metadata["description"] = description
            
            self.metadata["updated_at"] = datetime.now().isoformat()
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"更新知识库 {self.kb_id} 成功")
            return True
        except Exception as e:
            logger.error(f"更新知识库 {self.kb_id} 失败: {e}")
            return False
    
    def delete(self) -> bool:
        """删除知识库
        
        Returns:
            删除是否成功
        """
        try:
            if os.path.exists(self.kb_path):
                shutil.rmtree(self.kb_path)
            logger.info(f"删除知识库 {self.kb_id} 成功")
            return True
        except Exception as e:
            logger.error(f"删除知识库 {self.kb_id} 失败: {e}")
            return False
    
    def get_info(self) -> Dict:
        """获取知识库信息
        
        Returns:
            知识库信息字典
        """
        return self.metadata
    
    def add_document(self, file_path: str, file_name: str = None) -> bool:
        """添加文档到知识库并索引
        
        Args:
            file_path: 文档路径
            file_name: 文档名称，如果不提供则使用原文件名
            
        Returns:
            添加是否成功
        """
        try:
            # 检查文件类型
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                logger.error(f"不支持的文件类型: {file_extension}")
                return False
            
            # 使用原文件名或提供的名称
            if file_name is None:
                file_name = Path(file_path).name
            
            # 目标文件路径
            target_path = os.path.join(self.docs_path, file_name)
            
            # 复制文件到知识库文档目录
            os.makedirs(self.docs_path, exist_ok=True)
            shutil.copy2(file_path, target_path)
            
            # 更新文档列表
            doc_info = {
                "id": str(uuid.uuid4()),
                "name": file_name,
                "path": os.path.relpath(target_path, self.kb_path),
                "type": file_extension.lstrip('.'),
                "added_at": datetime.now().isoformat(),
                "size": os.path.getsize(target_path)
            }
            
            self.metadata["documents"].append(doc_info)
            self.metadata["document_count"] = len(self.metadata["documents"])
            self.metadata["updated_at"] = datetime.now().isoformat()
            
            # 保存元数据
            self._save_metadata()
            
            # 对文档进行索引
            self._index_document(target_path)
            
            logger.info(f"添加文档 {file_name} 到知识库 {self.kb_id} 成功")
            return True
        except Exception as e:
            logger.error(f"添加文档到知识库 {self.kb_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _index_document(self, file_path: str) -> bool:
        """索引文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            索引是否成功
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            # 加载文档
            documents = load_document(file_path)
            if not documents:
                logger.warning(f"从 {file_path} 加载的文档为空")
                return False
            
            # 分割文档
            chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
            if not chunks:
                logger.warning(f"分割文档 {file_path} 后没有块")
                return False
            
            # 为每个块添加来源元数据
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = Path(file_path).name
                chunk.metadata["chunk_index"] = i
                chunk.metadata["kb_id"] = self.kb_id
            
            # 获取嵌入模型
            embeddings = get_embedding_model()
            
            # 准备索引目录
            os.makedirs(self.index_path, exist_ok=True)
            
            # 检查是否已有索引
            faiss_file_path = os.path.join(self.index_path, "index.faiss")
            pkl_file_path = os.path.join(self.index_path, "index.pkl")
            
            if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
                # 加载现有索引并添加新文档
                vectorstore = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(chunks)
            else:
                # 创建新索引
                vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # 保存索引
            vectorstore.save_local(self.index_path)
            
            logger.info(f"索引文档 {file_path} 成功")
            return True
        except Exception as e:
            logger.error(f"索引文档 {file_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """从知识库中移除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            移除是否成功
        """
        try:
            # 查找文档
            doc_info = None
            for doc in self.metadata["documents"]:
                if doc["id"] == doc_id:
                    doc_info = doc
                    break
            
            if doc_info is None:
                logger.warning(f"在知识库 {self.kb_id} 中未找到文档 {doc_id}")
                return False
            
            # 删除文档文件
            doc_path = os.path.join(self.kb_path, doc_info["path"])
            if os.path.exists(doc_path):
                os.remove(doc_path)
            
            # 更新元数据
            self.metadata["documents"] = [doc for doc in self.metadata["documents"] if doc["id"] != doc_id]
            self.metadata["document_count"] = len(self.metadata["documents"])
            self.metadata["updated_at"] = datetime.now().isoformat()
            
            # 保存元数据
            self._save_metadata()
            
            # 需要重建索引
            self._rebuild_index()
            
            logger.info(f"从知识库 {self.kb_id} 中移除文档 {doc_id} 成功")
            return True
        except Exception as e:
            logger.error(f"从知识库 {self.kb_id} 中移除文档失败: {e}")
            return False
    
    def _rebuild_index(self) -> bool:
        """重建知识库索引
        
        Returns:
            重建是否成功
        """
        try:
            # 清理旧索引
            index_path = self.index_path
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            os.makedirs(index_path, exist_ok=True)
            
            # 对所有文档重新索引
            for doc_info in self.metadata["documents"]:
                doc_path = os.path.join(self.kb_path, doc_info["path"])
                if os.path.exists(doc_path):
                    self._index_document(doc_path)
            
            logger.info(f"重建知识库 {self.kb_id} 索引成功")
            return True
        except Exception as e:
            logger.error(f"重建知识库 {self.kb_id} 索引失败: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """搜索知识库
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            检索到的文档列表
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            # 检查索引是否存在
            faiss_file_path = os.path.join(self.index_path, "index.faiss")
            pkl_file_path = os.path.join(self.index_path, "index.pkl")
            
            if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
                logger.warning(f"知识库 {self.kb_id} 索引不存在")
                return []
            
            # 获取嵌入模型
            embeddings = get_embedding_model()
            
            # 加载索引
            vectorstore = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
            
            # 检索文档
            results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
            
            # 处理结果，避免内容重复
            processed_results = []
            seen_content_hashes = set()
            
            # 按相关性分数排序
            sorted_results = sorted(results_with_scores, key=lambda x: x[1])
            
            for doc, score in sorted_results:
                # 创建内容指纹
                content_hash = hash(doc.page_content[:150])
                
                # 如果内容已存在，跳过
                if content_hash in seen_content_hashes:
                    continue
                
                # 添加内容指纹
                seen_content_hashes.add(content_hash)
                
                # 将分数添加到元数据
                doc.metadata["score"] = score
                processed_results.append(doc)
            
            logger.info(f"在知识库 {self.kb_id} 中搜索 '{query[:30]}...'，找到 {len(processed_results)} 个结果")
            return processed_results
        except Exception as e:
            logger.error(f"在知识库 {self.kb_id} 中搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []


# --- 知识库管理函数 ---

def list_knowledge_bases(user_id=None, include_public=True) -> List[Dict]:
    """获取知识库列表，支持按用户ID过滤
    
    Args:
        user_id: 用户ID，如果提供，则只返回该用户创建的知识库和公共知识库
        include_public: 是否包含公共知识库，默认为True
        
    Returns:
        知识库信息列表
    """
    try:
        result = []
        
        # 遍历知识库目录
        for kb_dir in os.listdir(KB_BASE_PATH):
            kb_path = os.path.join(KB_BASE_PATH, kb_dir)
            
            # 检查是否是目录
            if not os.path.isdir(kb_path):
                continue
            
            # 检查是否有元数据文件
            metadata_path = os.path.join(kb_path, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"知识库 {kb_dir} 没有元数据文件")
                continue
            
            # 加载元数据
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 检查是否是系统知识库（公共）
                is_public = metadata.get("created_by") == "system"
                # 添加公私有标记到元数据中
                metadata["is_public"] = is_public
                
                # 根据用户ID过滤
                if user_id is None or metadata.get("created_by") == user_id or (include_public and is_public):
                    result.append(metadata)
            except Exception as e:
                logger.error(f"加载知识库 {kb_dir} 元数据失败: {e}")
        
        return result
    except Exception as e:
        logger.error(f"获取知识库列表失败: {e}")
        return []

def get_knowledge_base(kb_id: str) -> Optional[KnowledgeBase]:
    """获取知识库对象
    
    Args:
        kb_id: 知识库ID
        
    Returns:
        知识库对象，如果不存在则返回None
    """
    try:
        kb_path = os.path.join(KB_BASE_PATH, kb_id)
        
        # 检查知识库是否存在
        if not os.path.exists(kb_path) or not os.path.isdir(kb_path):
            logger.warning(f"知识库 {kb_id} 不存在")
            return None
        
        # 实例化知识库对象
        return KnowledgeBase(kb_id)
    except Exception as e:
        logger.error(f"获取知识库 {kb_id} 失败: {e}")
        return None

def create_knowledge_base(name: str, description: str, user_id: str) -> Optional[str]:
    """创建知识库
    
    Args:
        name: 知识库名称
        description: 知识库描述
        user_id: 创建者ID
        
    Returns:
        知识库ID，如果创建失败则返回None
    """
    try:
        kb = KnowledgeBase()
        kb_id = kb.create(name, description, user_id)
        return kb_id
    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        return None

def delete_knowledge_base(kb_id: str) -> bool:
    """删除知识库
    
    Args:
        kb_id: 知识库ID
        
    Returns:
        删除是否成功
    """
    try:
        kb = get_knowledge_base(kb_id)
        if kb is None:
            return False
        return kb.delete()
    except Exception as e:
        logger.error(f"删除知识库 {kb_id} 失败: {e}")
        return False 