# AI聊天助手系统技术文档

## 1. 项目概述

本项目是基于LangGraph和LangChain构建的智能聊天系统，集成了RAG（检索增强生成）功能、长对话管理、文件处理和情感监控等高级特性。后端采用Python实现，前端使用Vue框架开发，共同构成完整的知识型智能助手系统。

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────┐      ┌─────────────────────────────────┐
│             │      │           后端(LangGraph)        │
│  前端(Vue)  │ <──> │ ┌─────────┐ ┌─────────┐ ┌─────┐ │
│  (MyAI)     │      │ │App API  │ │RAG模块  │ │Agent│ │
│             │      │ └─────────┘ └─────────┘ └─────┘ │
└─────────────┘      └─────────────────────────────────┘
                              │          │
                     ┌────────┴──────────┴────────┐
                     │      PostgreSQL数据库       │
                     └───────────────────────────┘
```

### 2.2 后端架构

- **app.py**: Flask服务器，提供REST API接口
- **main.py**: LangGraph流程定义和Agent实现
- **thread_manager.py**: 对话线程管理和存储
- **rag_utils.py**: 文档处理、索引和检索实现
- **tools.py**: 工具函数集合
- **emotion_monitor.py**: 情感监控模块

### 2.3 前端架构

- **pages/chatbot/chat.vue**: 聊天界面
- **pages/chatbot/threads.vue**: 会话列表管理
- **components/**: 可复用组件
- **api/**: 后端API调用封装

## 3. 核心功能实现

### 3.1 消息管理系统

后端通过`thread_manager.py`实现线程和消息的管理：

- 基于PostgreSQL存储消息和线程信息
- 实现消息的去重和摘要功能
- 支持系统消息和用户消息分类处理
- 基于令牌计数的智能截断和摘要生成

优化亮点：
```python
# 系统消息去重逻辑
[Agent Node] System message deduplication: 12 -> 4
[Agent Node] Separated messages: 4 System, 25 Other
```

### 3.2 RAG检索增强生成

通过`rag_utils.py`实现文档的处理和检索：

- 支持PDF、TXT等多种文档格式处理
- 文档分块和嵌入向量化（使用Baichuan等模型）
- FAISS向量检索和相关性排序
- 基于阈值的相关文档过滤（分数≥0.85）

优化亮点：
```python
# 文档去重和相关性过滤
filtered_docs = [doc for doc in retrieved_docs_with_scores if doc.metadata.get('score', 0.0) >= threshold]
```

### 3.3 独立知识库管理

通过`knowledge_base.py`实现中心化的知识库管理：

- 知识库与聊天线程完全解耦，支持共享使用
- 支持创建、列表、更新、删除知识库
- 支持用户自定义知识库的创建和维护
- 文档可独立上传与管理，支持更新和索引刷新

知识库结构：
```
/db/
  /knowledge_bases/       # 独立知识库根目录
    /kb_[id]/             # 每个知识库的文件夹
      /docs/              # 存储文档文件
      /index/             # 存储向量索引
      metadata.json       # 知识库元数据
```

API接口：
```
GET  /api/knowledge-bases              # 获取知识库列表
POST /api/knowledge-bases              # 创建新知识库
GET  /api/knowledge-bases/<kb_id>      # 获取知识库详情
PUT  /api/knowledge-bases/<kb_id>      # 更新知识库信息
DELETE /api/knowledge-bases/<kb_id>    # 删除知识库
POST /api/knowledge-bases/<kb_id>/documents      # 上传文档
DELETE /api/knowledge-bases/<kb_id>/documents/<doc_id>  # 删除文档
POST /api/knowledge-bases/<kb_id>/search         # 搜索知识库
```

在聊天中使用知识库：
```javascript
// 前端传递知识库ID
const params = {
  message: userInput,
  threadId: this.threadId,
  userId: this.userId,
  kb_id: this.selectedKnowledgeBase // 指定要使用的知识库ID
};
```

### 3.4 LLM交互和流式响应

通过`main.py`实现LLM的交互：

- 支持流式响应的SSE实现
- 动态提示词构建
- 支持切换RAG模式
- 友好的文档引用格式

优化亮点：
```python
# 用户友好的RAG引用格式
rag_context = "\n\n".join([
    f"《{doc.metadata['source']}》(第{doc.metadata.get('page', 0)}页):\n{doc.page_content}\n\n" 
    for i, doc in enumerate(retrieved_docs)
])
```

## 4. 前端实现

### 4.1 聊天界面

`chat.vue`实现了丰富的聊天功能：

- 消息展示和Markdown渲染
- 流式接收和显示
- 文件上传功能
- RAG模式切换
- 日记分享和分析
- 消息保存到笔记本

```javascript
// RAG模式开关功能
toggleRagMode() {
    this.useRag = !this.useRag;
    uni.showToast({
        title: this.useRag ? 'RAG模式已开启' : 'RAG模式已关闭',
        icon: 'none',
        duration: 1500
    });
}
```

### 4.2 会话管理

`threads.vue`实现了会话管理功能：

- 会话列表展示
- 创建新会话
- 删除会话
- 系统提示词设置
- 助手类型选择
- 日记分享模式

```javascript
// 创建新会话并支持日记分享
async createNewThread() {
    // ...创建逻辑...
    if (this.isShareMode) {
        this.navigateToChatWithDiary(threadId, title);
    } else {
        uni.navigateTo({
            url: `/pages/chatbot/chat?threadId=${threadId}&title=${encodeURIComponent(title)}`
        });
    }
}
```

## 5. 系统集成与交互

### 5.1 文件上传流程

1. 前端选择文件并调用上传API
2. 后端接收文件并存储
3. `rag_utils.py`处理文件内容，生成向量索引
4. 用户可直接查询相关文档内容

### 5.2 日记分享与分析

1. 日记应用分享内容到聊天系统
2. 前端创建分享卡片并发送特定格式触发消息
3. 后端识别触发信息并获取日记内容
4. LLM分析日记内容并生成反馈

### 5.3 RAG查询流程

1. 用户发送查询（启用RAG模式）
2. 后端检索相关文档片段
3. 系统基于相关性过滤文档
4. 构建友好的RAG系统提示
5. LLM生成基于文档的响应

## 6. 性能优化

### 6.1 后端优化

- 系统消息去重机制减少冗余信息
- 相似文档片段合并降低token消耗
- 基于token计数的智能上下文管理
- 页码和来源信息格式化，提升用户体验

### 6.2 前端优化

- 流式接收减少等待时间
- 懒加载历史消息提升加载速度
- Markdown渲染优化阅读体验
- 移动端适配确保跨平台一致性

## 7. 总结

本系统结合了LangGraph的灵活图执行框架和Vue的前端开发能力，实现了一个功能完善的智能聊天助手系统。通过RAG技术增强了LLM的知识能力，通过会话管理提升了长对话能力，为用户提供了丰富、准确的交互体验。

项目亮点包括文档的智能处理、消息去重优化、友好的引用格式以及与日记系统的深度集成，体现了现代AI助手系统的核心优势。
