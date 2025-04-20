# 知识库独立化设计规划

## 设计目标

1. **知识库与线程完全独立**：摆脱当前知识库复制到线程的模式，实现中心化知识库管理
2. **支持知识库更新**：允许对已有知识库进行内容更新和索引刷新
3. **用户自定义知识库**：支持用户创建和维护个人或团队知识库
4. **简单易实现**：考虑到项目基础薄弱，采用直观的实现方案

## 当前项目架构

1. **存储结构**
   - 目前使用线程关联的知识库方式
   - 每应用一次知识库，就会复制一份到线程目录
   - 基于 FAISS 实现本地向量存储

2. **检索机制**
   - 使用 `search_index` 函数从线程特定路径检索信息
   - 有相关性分数过滤机制（阈值0.85）
   - 文档加载、分割和向量化功能已实现

3. **API接口**
   - 已有预定义知识库列表 API
   - 已有将知识库应用到线程的接口
   - 聊天流API支持RAG增强

## 需要实现的内容

1. **知识库存储重构**
   ```
   /db/
     /knowledge_bases/       # 新的独立知识库根目录
       /mental_health/       # 心理健康知识库
       /nutrition/           # 更多领域知识库
       /custom_kb_[id]/      # 用户自定义知识库
   ```

2. **核心功能模块**
   - 知识库管理：创建、列表、更新、删除
   - 文档处理：上传、处理、索引更新
   - 检索接口：直接从知识库检索，不依赖线程

3. **API 扩展**
   - `/api/knowledge-bases` GET/POST：获取列表/创建知识库
   - `/api/knowledge-bases/<kb_id>/documents` POST：上传文档
   - `/api/knowledge-bases/<kb_id>/search` POST：从特定知识库搜索

4. **聊天流程改造**
   - 修改聊天接口，支持指定知识库ID
   - 修改检索逻辑，从指定知识库检索而非线程目录

## 实施工作项

1. **核心模块开发** (估计工作量：2-3天)
   - [ ] 创建知识库存储结构和路径函数
   - [ ] 实现知识库管理基础功能
   - [ ] 改造文档上传和索引流程
   - [ ] 开发独立知识库检索功能

2. **API接口实现** (估计工作量：1-2天)
   - [ ] 开发知识库管理API
   - [ ] 实现文档上传接口
   - [ ] 创建知识库搜索专用接口
   - [ ] 编写接口文档

3. **聊天流程改造** (估计工作量：1天)
   - [ ] 修改聊天API支持知识库指定
   - [ ] 调整RAG增强逻辑使用指定知识库
   - [ ] 保持向后兼容性

4. **测试与优化** (估计工作量：1-2天)
   - [ ] 单独知识库测试
   - [ ] 聊天集成测试
   - [ ] 性能评估和优化
   - [ ] 边界情况处理

## 技术实现要点

1. **知识库创建与管理**
   ```python
   def create_knowledge_base(name, description, user_id):
       kb_id = f"kb_{uuid.uuid4().hex[:8]}"
       kb_path = os.path.join("db/knowledge_bases", kb_id)
       os.makedirs(os.path.join(kb_path, "docs"), exist_ok=True)
       
       # 保存元数据...
       return kb_id
   ```

2. **搜索功能改造**
   ```python
   def search_in_knowledge_base(query, kb_id, k=5):
       kb_path = os.path.join("db/knowledge_bases", kb_id)
       # 加载向量存储并检索...
       return filtered_docs
   ```

3. **聊天接口调整**
   ```python
   # 聊天接口参数增加
   kb_id = request.json.get('kb_id')
   
   # 检索逻辑变更
   if kb_id:
       docs = search_in_knowledge_base(query, kb_id)
   else:
       # 兼容原有线程检索...
   ```

通过以上工作，可以实现知识库与线程的完全解耦，让知识库成为独立的资源，支持创建、更新和共享使用，同时保持系统的简洁性和可维护性。
