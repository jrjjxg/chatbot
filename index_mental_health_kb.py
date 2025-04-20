# index_mental_health_kb.py
# 初始化预定义的心理健康知识库脚本

import os
import sys
import shutil
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入知识库管理模块
from knowledge_base import create_knowledge_base, get_knowledge_base

def init_mental_health_kb():
    """初始化预定义的心理健康知识库"""
    try:
        # 创建心理健康知识库
        kb_id = create_knowledge_base(
            name="心理健康知识库",
            description="包含心理健康、心理咨询和心理治疗相关的知识",
            user_id="system"
        )
        
        if not kb_id:
            logger.error("创建心理健康知识库失败")
            return False
        
        logger.info(f"成功创建心理健康知识库，ID: {kb_id}")
        
        # 获取知识库对象
        kb = get_knowledge_base(kb_id)
        if not kb:
            logger.error(f"无法获取知识库对象，ID: {kb_id}")
            return False
        
        # 使用kb对象的docs_path属性
        kb_docs_path = os.path.join(kb.docs_path)
        os.makedirs(kb_docs_path, exist_ok=True)

        # 然后不再寻找预定义目录的文件，而是提示用户添加文件
        print(f"心理健康知识库已创建，ID: {kb_id}")
        print(f"请手动将心理健康相关文档添加到: {kb_docs_path}")
        print("然后使用KB API上传文档: POST /api/knowledge-bases/{kb_id}/documents")
        
        return True
    
    except Exception as e:
        logger.error(f"初始化心理健康知识库时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("开始初始化心理健康知识库...")
    if init_mental_health_kb():
        logger.info("心理健康知识库初始化成功")
        sys.exit(0)
    else:
        logger.error("心理健康知识库初始化失败")
        sys.exit(1) 