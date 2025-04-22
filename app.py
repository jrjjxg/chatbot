import os
import time
import uuid
import threading
from flask import Flask, request, Response, jsonify, g, stream_with_context
from flask_cors import CORS
import json
import psycopg # 确保导入psycopg
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from werkzeug.utils import secure_filename # 用于安全地处理文件名

from emotion_monitor_agent import start_agent_monitoring_in_background
from rag_utils import index_document, ALLOWED_EXTENSIONS, search_index # 添加 search_index
from knowledge_base import (  # 导入知识库功能
    list_knowledge_bases,
    get_knowledge_base,
    create_knowledge_base,
    delete_knowledge_base
)
import requests
import datetime
from pathlib import Path # 确保导入 Path
import logging
import asyncio
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
# --- 在所有其他代码之前配置日志 --- 
# ... (日志配置)



# --- 文件上传配置 (定义常量) ---
UPLOAD_FOLDER = 'temp_uploads'
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
VECTORSTORE_BASE_PATH = "./db/vectorstores" # 将定义移到前面



# --- 定义 get_vectorstore_path_for_thread 函数 --- 
def get_vectorstore_path_for_thread(thread_id: str) -> str:
    """根据 thread_id 获取向量数据库的持久化路径"""
    safe_thread_id = "".join(c if c.isalnum() else "_" for c in thread_id)
    return os.path.join(VECTORSTORE_BASE_PATH, f"thread_{safe_thread_id}")

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True) # 确保基础向量存储目录也存在

# 导入需要的函数，避免循环导入
from main import (
    create_graph,
    process_graph_stream,
    get_default_system_message,
    get_llm
)

# 导入线程管理模块
from thread_manager import (
    save_thread,
    get_user_threads,
    remove_thread,
    save_message,
    get_chat_history,
    save_file_upload_message,
    initialize_db_pool
)

# 添加一个获取日记详情的函数
def get_journal_detail(journal_id, user_id=None):
    """获取日记详情 (使用正确的 /get/{id} 路径)"""
    try:
        # 配置Java后端API基础URL
        SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "http://localhost:9000")

        api_url = f"{SPRING_BOOT_BASE_URL}/api/journal/get/{journal_id}" 
        
        print(f"获取日记详情 (修正后的URL): {api_url}")

        headers = {} 

        response = requests.get(api_url, headers=headers, timeout=10) 
        
        if response.status_code == 200:
            data = response.json()
            # 检查Java后端返回的Result对象的结构
            if data.get("code") == 200 and data.get("data"):
                return data.get("data")
            else:
                print(f"获取日记详情失败 (业务错误): code={data.get('code')}, message={data.get('message')}")
                return None
        else:
            print(f"获取日记详情失败 (HTTP错误): {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"获取日记详情异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 创建日记提示的辅助函数
def create_diary_prompt(journal):
    """根据日记内容创建提示词"""
    title = journal.get("title", "无标题")
    content = journal.get("content", "")
    create_time = journal.get("createTime", "")
    word_count = journal.get("wordCount", 0)
    is_private = journal.get("isPrivate", 0)
    
    # 格式化日期
    date_str = ""
    if create_time:
        try:
            # 尝试解析ISO格式的日期
            if isinstance(create_time, str):
                # 移除可能的Z后缀并添加时区
                date = datetime.datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                date_str = date.strftime("%Y年%m月%d日 %H:%M")
            else:
                date_str = create_time
        except Exception as e:
            print(f"格式化日期失败: {e}")
            date_str = str(create_time)
    
    # 构建提示词
    prompt = f"我想讨论我的这篇日记：\n\n标题：{title}\n"
    prompt += f"日期：{date_str}\n"
    
    if word_count:
        prompt += f"字数：{word_count}字\n"
    
    if is_private == 1:
        prompt += f"私密状态：是\n"
    
    # 添加日记内容
    prompt += f"\n日记内容：\n{content}\n\n"
    
    # 添加请求
    prompt += "请帮我分析这篇日记，给我一些建议或者反馈。"
    
    return prompt

# 初始化 Flask 应用
logging.basicConfig(level=logging.INFO, # 或 logging.DEBUG 获取更详细信息
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- 在 app 初始化后进行配置 ---
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES # 使用常量

# 配置 CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"], "supports_credentials": True}})

# 创建 LangGraph 图构建器，全局共享
try:
    graph_builder = create_graph()
    print("LangGraph 图构建器创建成功")
except Exception as e:
    print(f"创建 LangGraph 图构建器时出错: {e}")
    graph_builder = None

# 数据库连接参数
DB_URI = "postgresql://chatbot_user:123456@localhost:5433/langgraph_db"

# 最大重试次数
MAX_RETRY = 3
retry_count = 0

# --- 数据库Setup阶段 ---
while retry_count < MAX_RETRY:
    try:
        print(f"正在连接数据库以执行Setup (尝试 {retry_count+1}/{MAX_RETRY})...")
        # 创建一个自动提交的连接
        with psycopg.connect(DB_URI, autocommit=True) as setup_conn:
            print("数据库连接成功，正在执行Setup...")
            # 创建临时实例来调用setup，将连接作为第一个位置参数传递
            print("正在创建临时Saver实例以执行Setup...")
            temp_saver = PostgresSaver(setup_conn)
            print("正在调用Saver的Setup...")
            temp_saver.setup() # 在实例上调用setup

            print("正在创建临时Store实例以执行Setup...")
            temp_store = PostgresStore(setup_conn)
            print("正在调用Store的Setup...")
            temp_store.setup() # 在实例上调用setup
            print("数据库Setup成功完成。")
            break  # 成功连接和设置后退出循环
    except Exception as e:
        retry_count += 1
        print(f"数据库Setup失败 (尝试 {retry_count}/{MAX_RETRY}): {e}")
        if retry_count >= MAX_RETRY:
            print("达到最大重试次数，应用将退出")
            raise  # 重新抛出异常以停止执行
        import time
        time.sleep(2)  # 等待2秒后重试

# --- 应用运行时阶段 ---
# 为应用的常规操作创建连接池
print("正在创建数据库连接池...")
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    min_size=2,
    timeout=30,
    max_lifetime=60*5,
    max_idle=60*2
)
print("数据库连接池创建成功。")

# --- 初始化 ThreadManager 的数据库连接池 ---
initialize_db_pool(pool)
print("ThreadManager 数据库连接池已初始化。")

# 使用连接池初始化Checkpointer和Store
print("正在使用连接池初始化Checkpointer和Store...")
app.checkpointer = PostgresSaver(pool)
print("Checkpointer和Store初始化完成。")

# 全局编译好的图实例
print("正在编译LangGraph图...")
app.runnable = graph_builder.compile(
    checkpointer=app.checkpointer
)
print("LangGraph图编译成功。")




@app.route('/api/chat/stream', methods=['GET', 'POST', 'OPTIONS'])
def chat_stream_api():
    """为前端提供流式输出的API接口 (入口不变，但调用异步生成器)"""
    if request.method == 'OPTIONS':
        # 处理 OPTIONS 请求以支持 CORS
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # --- 获取请求参数 ---
    is_post = (request.method == 'POST')
    req_data = request.json if is_post else request.args
    
    # 从请求中提取参数
    thread_id = req_data.get('threadId')
    request_user_id = req_data.get('userId') # 前端传入的用户ID
    message = req_data.get('message')
    no_streaming = req_data.get('noStreaming', False)
    no_search = req_data.get('noSearch', False)  # 前端可以选择是否禁用搜索
    # 新增：知识库ID参数
    kb_id = req_data.get('kb_id')
    
    # 验证必需参数
    if not all([thread_id, request_user_id, message]):
        missing = []
        if not thread_id: missing.append('threadId')
        if not request_user_id: missing.append('userId')
        if not message: missing.append('message')
        return jsonify({'error': f'缺少必要参数: {", ".join(missing)}'}), 400
    
    # --- 验证用户与线程关系 (安全检查) ---
    # 为简化处理，我们假设线程ID格式为 "user_id_uuid"，
    # 其中前面的部分应该与请求中的用户ID匹配
    actual_user_id = thread_id.split('_')[0] if '_' in thread_id else None
    
    if not actual_user_id or actual_user_id != request_user_id:
        # 或者允许，但记录一个警告
        print(f"警告: 用户 {request_user_id} 尝试访问可能不属于他的线程 {thread_id}")
        # 严格模式下直接拒绝
        # return jsonify({'error': '无权访问该线程'}), 403



    # --- 准备聊天处理 ---
    try:
        # 检查用户是否已有历史消息记录
        history = get_chat_history(request_user_id, thread_id)
        
        # 获取默认系统提示词
        default_system_prompt = get_default_system_message().content
        
        # 自定义系统提示词 (如果有)
        custom_system_prompt = None

        # 从数据库获取线程信息 ---
        try:
            user_threads_list = get_user_threads(actual_user_id) # 预期返回 List[Dict]

            # 使用生成器表达式和 next() 查找匹配的线程信息
            # 如果找不到，next() 的第二个参数 None 会被返回
            thread_info = next(
                (t for t in user_threads_list if isinstance(t, dict) and t.get('id') == thread_id),
                None
            )

            if thread_info:
                # 找到了线程信息，尝试获取 system_prompt
                custom_system_prompt = thread_info.get('system_prompt')
                if custom_system_prompt:
                    print(f"从数据库找到线程 '{thread_id}' 的自定义系统提示词。") # 日志简洁化
                else:
                    print(f"线程 '{thread_id}' 在数据库中存在，但没有设置自定义系统提示词。")
            else:
                # 如果 thread_info 为 None，说明未找到
                thread_count = len(user_threads_list) if isinstance(user_threads_list, list) else '未知数量'
                print(f"在从数据库获取的 {thread_count} 个线程中未找到线程 '{thread_id}'。")

        except Exception as e:
            print(f"从数据库获取或处理线程信息时发生异常: {e}")
            # 异常情况下，custom_system_prompt 保持为 None，将使用默认提示词

        # 确定使用哪个系统提示词
        system_prompt = custom_system_prompt if custom_system_prompt else default_system_prompt
        print(f"最终决定使用的系统提示词: {system_prompt}")




        # ---- RAG检索和动态提示词逻辑 ----
        # 变量初始化
        has_rag_context = False
        retrieved_docs = []
        
        # 记录是否尝试RAG检索
        print(f"检查是否使用RAG增强")
        
        # 确定是否需要使用RAG
        need_rag = not no_search  # 如果前端设置了不搜索，则不使用RAG

        # --- 知识库RAG检索逻辑 ---
        if kb_id and need_rag:
            print(f"使用知识库 {kb_id} 进行RAG检索")
            kb = get_knowledge_base(kb_id)
            if kb:
                # 设置相关性阈值
                threshold = 0.85  # 可以根据需要调整
                
                # 在知识库中搜索
                results = kb.search(message, k=5)
                
                # 添加详细日志输出每个结果的内容和相关性分数
                print(f"\n==== 知识库 {kb_id} 搜索结果 (共 {len(results)} 个) ====")
                for i, doc in enumerate(results):
                    score = doc.metadata.get('score', 0.0)
                    source = doc.metadata.get('source', '未知来源')
                    page = doc.metadata.get('page', '未知页码')
                    content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    print(f"结果 #{i+1}:")
                    print(f"  来源: {source}, 页码: {page}")
                    print(f"  相关性分数: {score}")
                    print(f"  内容预览: {content_preview}")
                    print(f"  通过过滤条件: {'是' if score < threshold else '否'}")
                    print("----------")
                
                # 过滤低相关性结果
                filtered_results = [doc for doc in results if doc.metadata.get('score', 0.0) < threshold]  # 分数越低越相关（欧氏距离）
                
                # 优化：当所有结果都被过滤掉时，取最相关的前2个结果
                if not filtered_results and results:
                    # 按相关性分数排序（升序，因为分数越低越相关）
                    sorted_results = sorted(results, key=lambda x: x.metadata.get('score', float('inf')))
                    # 取前2个最相关的结果
                    filtered_results = sorted_results[:2]
                    print(f"所有结果分数都 >= {threshold}，保留最相关的 {len(filtered_results)} 个结果用于RAG")
                
                if filtered_results:
                    has_rag_context = True
                    retrieved_docs = filtered_results
                    print(f"从知识库 {kb_id} 成功检索到 {len(retrieved_docs)} 个相关片段")
                else:
                    print(f"从知识库 {kb_id} 未找到相关内容")
            else:
                print(f"知识库 {kb_id} 不存在")



        # --- 原有线程向量库检索逻辑 ---
        elif need_rag:
            print(f"使用线程向量库进行RAG检索")
            vectorstore_path = get_vectorstore_path_for_thread(thread_id)
            
            # 设置相关性阈值 (与原来保持一致)
            threshold = 0.85

            if os.path.exists(vectorstore_path):
                print(f"发现向量库：{vectorstore_path}，尝试检索相关文档")
                
                # 尝试检索相关文档 (现在会返回带分数的文档)
                retrieved_docs_with_scores = search_index(message, thread_id, k=3) 
                
                # --- 关键强化点：基于分数过滤 ---
                filtered_docs = [doc for doc in retrieved_docs_with_scores if doc.metadata.get('score', 0.0) < threshold]  # 分数越低越相关（欧氏距离）

                # 优化：当所有结果都被过滤掉时，取最相关的前2个结果
                if not filtered_docs and retrieved_docs_with_scores:
                    # 按相关性分数排序（升序，因为分数越低越相关）
                    sorted_docs = sorted(retrieved_docs_with_scores, key=lambda x: x.metadata.get('score', 0))
                    # 取前2个最相关的结果
                    filtered_docs = sorted_docs[:2]
                    print(f"所有结果分数都 >= {threshold}，保留最相关的 {len(filtered_docs)} 个结果用于RAG")

                if filtered_docs:
                    # 只有在过滤后仍有文档时，才设置 has_rag_context 并使用过滤后的文档
                    has_rag_context = True 
                    retrieved_docs = filtered_docs # 使用过滤后的文档列表
                    print(f"成功检索到 {len(retrieved_docs)} 个相关文档片段")
                else:
                    # 如果没有文档通过分数过滤，则不使用RAG
                    print(f"未找到相关的文档片段")
                    # has_rag_context 保持 False
            else:
                print(f"未找到向量库：{vectorstore_path}，跳过RAG检索")
        else:
            print("基于规则或前端设置，跳过RAG检索")



        # --- 构建动态提示词 ---
        if has_rag_context:
            # 调试: 输出检索文档的元数据，查看页码信息是否存在
            print("\n==== RAG检索文档元数据详情 ====")
            for i, doc in enumerate(retrieved_docs):
                print(f"文档 #{i+1} 元数据:")
                for key, value in doc.metadata.items():
                    print(f"  {key}: {value}")
                print(f"  页面内容前50字符: {doc.page_content[:50]}...")
                print("----------")
                
            # 构建文档内容上下文 (使用过滤后的 retrieved_docs)
            rag_context = "\n\n".join([
                f"《{doc.metadata['source']}》(第{doc.metadata.get('page', 0)}页):\n{doc.page_content}\n\n" 
                for i, doc in enumerate(retrieved_docs) # 使用过滤后的列表
            ])
            
            # --- 改进：创建更明确的RAG系统提示，而不是AI消息 ---
            rag_system_prompt = f"""你是一个有用的AI助手。请根据以下提供的文档内容回答用户问题：

{rag_context}

引用文档时，请使用"根据《文档名》第X页"或简单地说"文档中提到..."，不要使用"文档片段#X"或"要点X"这种用户看不懂的引用方式。

【重要指引】
1. 你必须只基于上述内容回答问题，不要添加自己的知识或推测
2. 如果文档中包含直接回答问题的内容，请直接引用该内容
3. 如果文档内容不足以回答问题，直接说明"基于提供的文档内容，我无法完全回答这个问题"
4. 在回答中直接反映文档原文的观点和结论，不要进行过度解释

原始系统提示: {system_prompt}
"""
            # 打印RAG上下文预览
            rag_context_preview = rag_context[:300] + "..." if len(rag_context) > 300 else rag_context
            print(f"[Chat API] 创建RAG系统消息, RAG上下文预览: {rag_context_preview}")
            
            # 使用RAG系统提示
            messages = []  # 完全清空历史消息
            # 只添加新的RAG系统消息和当前用户问题，不保留任何历史消息
            messages.append(SystemMessage(content=rag_system_prompt))
            # 直接添加当前用户查询作为最新消息，确保RAG只处理当前问题
            messages.append(HumanMessage(content=message))
            
            print(f"[Chat API] 创建了新的消息列表，包含 {len(messages)} 条消息")
            
            # --- 修改点3：设置标志，表示这是RAG增强的消息序列 ---
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": actual_user_id,  # 使用确认后的用户ID
                    "enable_memory": True,
                    "rag_enhanced": True,
                    "skip_history": True  # 新增标志：跳过历史消息
                }
            }
            
            print("使用RAG系统提示增强对话")
        else:
            # 使用普通系统提示
            messages = [SystemMessage(content=system_prompt)]
            # 设置普通配置
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": actual_user_id,  # 使用确认后的用户ID
                    "enable_memory": True
                }
            }
            print("使用普通系统提示（无RAG增强）")

        # 检查是否为日记分享触发消息
        is_share_trigger = message.startswith('//share_diary:')
        shared_diary_id = None
        diary_details = None
        input_for_llm = message  # 默认使用原始消息
        
        if is_share_trigger:
            try:
                # 提取日记ID
                shared_diary_id = message.split(':', 1)[1].strip()
                print(f"检测到日记分享触发: {shared_diary_id}")
                
                # 获取日记详情
                diary_details = get_journal_detail(shared_diary_id, actual_user_id)
                if diary_details:
                    print(f"成功获取日记详情: {diary_details.get('title', '无标题')}")
                    # 创建日记提示词
                    input_for_llm = create_diary_prompt(diary_details)
                else:
                    print("获取日记详情失败")
                    input_for_llm = f"我想分享我的日记(ID:{shared_diary_id})，但获取详情失败了，请给我提供一些通用的日记写作建议。"
            except Exception as e:
                print(f"处理日记分享触发失败: {e}")
                input_for_llm = f"我想分享我的日记，但处理过程出错: {str(e)}"


        response_id = str(uuid.uuid4())

        def generate():
            """SSE 事件生成器 (同步版本)"""
            full_response = "" # 累积响应
            yield f"event: start\ndata: {json.dumps({'response_id': response_id})}\n\n"
            
            # 如果是日记分享，先保存卡片消息 (使用统一的 save_message)
            if is_share_trigger and diary_details:
                try:
                    card_id = f"diary-{shared_diary_id}-{int(time.time())}" # ID可以简化
                    card_metadata = {
                        "diaryId": shared_diary_id,
                        "diaryTitle": diary_details.get("title", "无标题日记"),
                        "diaryDate": diary_details.get("createTime") 
                    }
                    
                    # --- 修改点: 调用 thread_manager 的 save_message ---
                    save_message( 
                        actual_user_id,
                        thread_id,
                        "system", # 卡片用 system 角色
                        f"分享了日记: {card_metadata['diaryTitle']}", 
                        msg_type='diary_share', 
                        metadata=card_metadata
                    )
                    print(f"已请求保存日记分享卡片到数据库: {thread_id}")
                except Exception as e:
                    print(f"尝试保存日记分享卡片时出错: {e}")

            try:
                # 使用同步process_graph_stream
                for chunk in process_graph_stream(
                        app.runnable,
                        input_for_llm,
                        history=messages, # 传递初始系统消息
                        config=config
                ):
                    if chunk: # 确保chunk不为空
                        full_response += chunk
                        yield f"event: chunk\ndata: {json.dumps({'chunk': chunk, 'response_id': response_id})}\n\n"

                # 流完成后发送完成事件
                yield f"event: complete\ndata: {json.dumps({'full_response': full_response, 'response_id': response_id})}\n\n"

                # 保存消息到数据库
                print(f"准备保存消息到数据库 (Thread: {thread_id}). User: '{message[:50]}...', AI: '{full_response[:50]}...'")
                try:
                    save_message(actual_user_id, thread_id, "user", message) # 保存用户消息
                    save_message(actual_user_id, thread_id, "assistant", full_response) # 保存累积的AI回复
                    print(f"消息成功保存到数据库 for thread {thread_id}")
                except Exception as db_err:
                    print(f"保存消息到数据库时出错 for thread {thread_id}: {db_err}")
            except Exception as e:
                print(f"流式处理错误: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        # 返回响应
        return Response(
            stream_with_context(generate()), # stream_with_context适用于同步生成器
            mimetype='text/event-stream',
            headers={ # 保留必要的头信息
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'Access-Control-Allow-Origin': 'http://localhost:5173',
                'Access-Control-Allow-Credentials': 'true',
            }
        )
    except Exception as e:
        print(f"处理流式消息时出错: {e}")
        import traceback
        traceback.print_exc()
        # Ensure error response has CORS headers if needed by frontend
        error_response = jsonify({'error': f'处理消息时出错: {str(e)}'})
        error_response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        error_response.headers.add('Access-Control-Allow-Credentials', 'true')
        return error_response, 500


@app.route('/api/thread', methods=['POST'])
def create_thread():
    """创建新的聊天线程"""
    try:
        data = request.json
        user_id = data.get('userId')
        title = data.get('title', f"新对话 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        system_prompt = data.get('systemPrompt')  # 可选的系统提示词

        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400

        # 生成线程ID
        thread_id = f"{user_id}_{str(uuid.uuid4())}"

        # 保存线程信息
        save_thread(user_id, thread_id, title, system_prompt)

        return jsonify(thread_id)
    except Exception as e:
        return jsonify({'error': f'创建线程时出错: {str(e)}'}), 500


@app.route('/api/threads', methods=['GET'])
def get_threads():
    """获取用户的所有聊天线程"""
    try:
        user_id = request.args.get('userId')
        group_by_date = request.args.get('groupByDate', 'false').lower() == 'true'

        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400

        # 获取用户的所有线程，可选按日期分组
        threads = get_user_threads(user_id, group_by_date)

        return jsonify(threads)
    except Exception as e:
        return jsonify({'error': f'获取线程列表时出错: {str(e)}'}), 500


@app.route('/api/threads/by-date', methods=['GET'])
def get_threads_by_date():
    """按日期获取用户的所有聊天线程"""
    try:
        user_id = request.args.get('userId')

        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400

        # 获取按日期分组的线程
        threads = get_user_threads(user_id, group_by_date=True)

        return jsonify(threads)
    except Exception as e:
        return jsonify({'error': f'获取线程列表时出错: {str(e)}'}), 500


@app.route('/api/history/<thread_id>', methods=['GET'])
def get_history(thread_id):
    """获取聊天历史记录 (自动从 thread_id 推断 userId)"""
    print(f"--- History Request ---") 
    print(f"Received thread_id (from path): {thread_id}")

    # --- 新增: 尝试从 thread_id 提取 user_id ---
    actual_user_id = None
    if '_' in thread_id:
        try:
            # 假设格式为 "userid_uuid"
            actual_user_id = thread_id.split('_')[0] 
            print(f"Extracted user_id from thread_id: {actual_user_id}")
        except Exception as e:
            print(f"从 thread_id 提取 user_id 时出错: {e}")
            # 提取失败，保持 actual_user_id 为 None

    if not actual_user_id:
        
        print(f"错误: 无法从 thread_id '{thread_id}' 提取有效的用户ID。")
        return jsonify({'error': f"无法从线程ID '{thread_id}' 确定用户"}), 400
     
    # --- 结束新增 ---
        
    # --- 修改点: 使用推断出的 actual_user_id 调用 get_chat_history ---
    if not thread_id: # 理论上不会发生，因为 thread_id 是路径参数
         return jsonify({'error': '缺少必要参数 threadId'}), 400

    print(f"Calling get_chat_history with inferred user_id: '{actual_user_id}' and thread_id: '{thread_id}'")
    try:
        # 调用 thread_manager 中的数据库版本 get_chat_history
        history = get_chat_history(actual_user_id, thread_id) 
        
        # 检查 get_chat_history 是否返回了错误信息
        if history.get("error"):
             print(f"get_chat_history 返回错误: {history.get('error')}")
             # 可以选择将数据库错误暴露给前端，或返回通用错误
             return jsonify({'error': f"获取历史记录失败: {history.get('error')}"}), 500

        print(f"History data retrieved successfully for thread {thread_id}")
        return jsonify(history) # 返回 {"data": [...]} 或 {"data": [], "error": ...}
        
    except Exception as e:
        print(f"调用 get_chat_history 或处理结果时出错: {e}")
        return jsonify({'error': f'获取聊天历史时发生内部错误: {str(e)}'}), 500


@app.route('/api/thread/<thread_id>', methods=['DELETE'])
def delete_thread(thread_id):
    """删除聊天线程"""
    try:
        user_id = request.args.get('userId')

        if not user_id or not thread_id:
            return jsonify({'error': '缺少必要参数'}), 400

        # 删除线程
        remove_thread(user_id, thread_id)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'删除线程时出错: {str(e)}'}), 500


@app.route('/api/journal/save-ai-response', methods=['POST'])
def save_ai_response_to_journal():
    """将 AI 回复保存到日记"""
    try:
        data = request.json
        journal_id = data.get('journalId')
        ai_response = data.get('aiResponse')
        user_id = data.get('userId')

        if not journal_id or not ai_response or not user_id:
            return jsonify({'error': '缺少必要参数 journalId, aiResponse 或 userId'}), 400

        # 调用 Java 后端 API 将 AI 回复保存到日记
        SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "http://localhost:9000")
        JOURNAL_SAVE_AI_RESPONSE_ENDPOINT = os.getenv(
            "JOURNAL_SAVE_AI_RESPONSE_ENDPOINT", 
            "/api/journal/"
        )
        
        import requests
        
        # 构建 API URL
        java_api_url = f"{SPRING_BOOT_BASE_URL}{JOURNAL_SAVE_AI_RESPONSE_ENDPOINT}{journal_id}/save-ai-response"
        headers = {
            "Content-Type": "application/json"
        }
        
        # 准备请求体
        payload = {
            "aiResponse": ai_response
        }
        
        print(f"正在调用 Java API 保存 AI 回复: {java_api_url}")
        response = requests.post(
            java_api_url, 
            headers=headers, 
            json=payload, 
            params={"userId": user_id},
            timeout=10
        )
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get('code') != 200:
            print(f"保存 AI 回复失败: {response_data.get('message', '未知错误')}")
            return jsonify({'error': f"保存失败: {response_data.get('message', '未知错误')}"}), 500
        
        return jsonify({'status': 'success', 'message': 'AI 回复已成功保存到日记'})
        
    except requests.exceptions.RequestException as e:
        print(f"调用 Java API 时出错: {e}")
        return jsonify({'error': f'保存 AI 回复失败 (API 错误): {str(e)}'}), 500
    except Exception as e:
        print(f"保存 AI 回复时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'保存 AI 回复时出错: {str(e)}'}), 500


@app.route('/api/chatbot/prompt', methods=['POST', 'OPTIONS'])
def update_prompt():
    """更新系统提示词 - 使用线程ID中的用户信息"""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    try:
        data = request.json
        thread_id = data.get('thread_id')
        frontend_user_id = data.get('user_id')  # 前端传入的用户ID
        system_prompt = data.get('system_prompt', '')

        if not thread_id:
            return jsonify({'error': '缺少必要参数 thread_id'}), 400

        if not system_prompt.strip():
            # 如果提供的提示词为空，则使用默认系统提示词
            system_prompt = get_default_system_message().content

        # 从线程ID中提取用户名
        user_id = thread_id.split('_')[0] if '_' in thread_id else None

        if not user_id:
            return jsonify({'error': '无效的线程ID格式'}), 400

        if frontend_user_id and frontend_user_id != user_id:
            return jsonify({'error': '用户ID不匹配'}), 403

        # 更新系统提示词
        save_thread(user_id, thread_id, system_prompt=system_prompt)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'更新系统提示词时出错: {str(e)}'}), 500

# --- 文件上传路由 (现在可以调用 get_vectorstore_path_for_thread) ---
@app.route('/api/upload/<thread_id>', methods=['POST'])
def upload_file_api(thread_id):
    """接收用户上传的文件, 保存消息记录, 并为其建立索引"""
    if not thread_id:
        return jsonify({'error': '缺少 thread_id'}), 400

    # --- 新增：从请求参数获取 user_id --- 
    #  假设 user_id 通过查询参数传递，与 chat_stream_api 类似
    request_user_id = request.args.get('userId')
    if not request_user_id:
         # 如果没传 userId，尝试从 thread_id 推断 (如果格式是 user_id_uuid)
         parts = thread_id.split('_')
         if len(parts) > 1:
              request_user_id = parts[0]
              print(f"[Upload API] Warning: userId not in query params, inferred from thread_id: {request_user_id}")
         else:
              print(f"[Upload API] Error: Missing 'userId' in query parameters and cannot infer from thread_id.")
              return jsonify({'error': '缺少必要参数 userId (Missing required parameter userId)'}), 400
    # --- 结束新增 --- 

    # 1. 检查文件是否存在于请求中
    if 'file' not in request.files:
        return jsonify({'error': '请求中没有文件部分 (No file part in the request)'}), 400

    file = request.files['file']

    # 2. 如果用户没有选择文件，浏览器可能会提交一个空文件名
    if file.filename == '':
        return jsonify({'error': '没有选择文件 (No selected file)'}), 400

    # 3. 检查文件类型是否允许
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'不允许的文件类型: {file_extension}. 支持的类型: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # 4. 文件名安全处理和保存
    filename = secure_filename(file.filename) # 获取安全的文件名
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[Upload API] Saving temporary file to: {temp_file_path}")
    try:
        file.save(temp_file_path) # 保存到临时目录
    except Exception as e:
         print(f"[Upload API] Error saving temporary file {temp_file_path}: {e}")
         return jsonify({'error': '保存上传文件时出错'}), 500

    # --- 新增：在索引前保存文件上传消息记录 --- 
    file_type = file_extension.lstrip('.') # 获取 'txt', 'pdf' 等
    print(f"[Upload API] Saving file upload message to history for user {request_user_id}, thread {thread_id}")
    message_saved = save_file_upload_message(request_user_id, thread_id, filename, file_type)
    if not message_saved:
         # 如果保存消息失败，也算作一个问题，但可能仍继续索引？或者直接返回错误？
         # 这里选择记录警告并继续，但你可以根据需要决定是否中止
         print(f"[Upload API] Warning: Failed to save file upload message to history for {filename}")
         # return jsonify({'error': '无法记录文件上传历史'}), 500 # 或者返回错误
    # --- 结束新增 ---

    # 调用 RAG 索引函数
    print(f"[Upload API] Calling index_document for file: {temp_file_path}, thread_id: {thread_id}")
    success = index_document(temp_file_path, thread_id)

    # 可选：索引后删除临时文件 (如果 rag_utils 中没有处理)
    try:
        os.remove(temp_file_path)
        print(f"[Upload API] Removed temporary file: {temp_file_path}")
    except Exception as e:
        print(f"[Upload API] Warning: Could not remove temporary file {temp_file_path}: {e}")

    # 返回结果
    if success:
        vs_path = get_vectorstore_path_for_thread(thread_id)
        app.logger.info(f"[Upload API] Checking existence of vector store at: {vs_path}") # 使用 app.logger
        if os.path.exists(vs_path):
             app.logger.info(f"[Upload API] Vector store confirmed. Preparing successful response for {filename}.") # 使用 app.logger
             try:
                 response = jsonify({'message': f'文件 "{filename}" 上传并索引成功'}), 200
                 app.logger.info("[Upload API] Successfully created 200 OK response object.") # 使用 app.logger
                 return response
             except Exception as json_err:
                 app.logger.error(f"[Upload API] Error during jsonify or returning 200 OK: {json_err}", exc_info=True) # 使用 app.logger
                 return jsonify({'error': '创建成功响应时出错'}), 500
        else:
             app.logger.error(f"[Upload API] Error: Indexing reported success but vector store directory not found at {vs_path}") # 使用 app.logger
             return jsonify({'error': '索引过程似乎成功，但未能验证向量存储'}), 500
    else:
        # 这个分支理论上不应该执行，因为 index_document 在失败时会 raise Exception
        app.logger.error(f"[Upload API] index_document unexpectedly returned False for file {filename}, thread {thread_id}") # 使用 app.logger
        return jsonify({'error': f'文件 "{filename}" 索引失败'}), 500


@app.route('/api/threads/dates-by-month', methods=['GET'])
def get_threads_dates_by_month():
    """获取某月内有聊天记录的所有日期列表"""
    try:
        user_id = request.args.get('userId')
        year = request.args.get('year')
        month = request.args.get('month')
        
        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400
        
        if not year or not month:
            return jsonify({'error': '缺少年份或月份参数'}), 400
            
        # 验证参数格式
        try:
            year = int(year)
            month = int(month)
            if month < 1 or month > 12:
                return jsonify({'error': '月份必须在1-12之间'}), 400
        except ValueError:
            return jsonify({'error': '年份和月份必须是整数'}), 400
            
        from thread_manager import get_threads_dates_by_month
        dates_with_threads = get_threads_dates_by_month(user_id, year, month)
        
        # 修改返回结构以匹配前端预期 { data: { dates: [...] } }
        return jsonify({"data": dates_with_threads})
    except Exception as e:
        import traceback
        print(f"获取月份聊天记录日期失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'获取月份聊天记录日期失败: {str(e)}'}), 500

# --- 知识库API路由 ---
@app.route('/api/knowledge-bases', methods=['GET'])
def list_kb_api():
    """获取所有知识库列表"""
    try:
        # 获取用户ID参数
        user_id = request.args.get('user_id')
        include_public = request.args.get('include_public', 'true').lower() == 'true'
        
        # 调用支持过滤的函数
        kb_list = list_knowledge_bases(user_id=user_id, include_public=include_public)
        return jsonify({"data": kb_list})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取知识库列表失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases', methods=['POST'])
def create_kb_api():
    """创建新知识库"""
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        user_id = data.get('user_id')
        is_public = data.get('is_public', False)  # 新增公私有选项，默认为私有
        
        if not name:
            return jsonify({"error": "知识库名称不能为空"}), 400
        if not user_id:
            return jsonify({"error": "用户ID不能为空"}), 400
            
        # 如果设置为公共知识库，将创建者设为system
        if is_public:
            user_id = "system"
            
        kb_id = create_knowledge_base(name, description, user_id)
        if kb_id:
            return jsonify({"message": "创建知识库成功", "data": {"kb_id": kb_id, "is_public": is_public}})
        else:
            return jsonify({"error": "创建知识库失败"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"创建知识库失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>', methods=['GET'])
def get_kb_api(kb_id):
    """获取知识库详情"""
    try:
        kb = get_knowledge_base(kb_id)
        if kb:
            return jsonify({"data": kb.get_info()})
        else:
            return jsonify({"error": "知识库不存在"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取知识库详情失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>', methods=['PUT'])
def update_kb_api(kb_id):
    """更新知识库信息"""
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description')
        
        kb = get_knowledge_base(kb_id)
        if not kb:
            return jsonify({"error": "知识库不存在"}), 404
            
        success = kb.update(name, description)
        if success:
            return jsonify({"message": "更新知识库成功"})
        else:
            return jsonify({"error": "更新知识库失败"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"更新知识库失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>', methods=['DELETE'])
def delete_kb_api(kb_id):
    """删除知识库"""
    try:
        success = delete_knowledge_base(kb_id)
        if success:
            return jsonify({"message": "删除知识库成功"})
        else:
            return jsonify({"error": "删除知识库失败，可能知识库不存在"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"删除知识库失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/documents', methods=['POST'])
def upload_kb_document_api(kb_id):
    """上传文档到知识库"""
    try:
        # 检查知识库是否存在
        kb = get_knowledge_base(kb_id)
        if not kb:
            return jsonify({"error": "知识库不存在"}), 404
            
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({"error": "请求中没有文件"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
            
        # 检查文件类型
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"不支持的文件类型: {file_extension}，支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
            
        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(temp_file_path)
        except Exception as e:
            return jsonify({"error": "保存上传文件时出错"}), 500
            
        # 添加文档到知识库
        success = kb.add_document(temp_file_path, filename)
        
        # 删除临时文件
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"删除临时文件失败: {e}")  # 只记录警告，继续处理

        # 确保正确缩进
        if success:
            return jsonify({"message": f"文件 '{filename}' 上传到知识库成功"})
        else:
            return jsonify({"error": "文件上传到知识库失败"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"上传文档到知识库失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/documents/<doc_id>', methods=['DELETE'])
def delete_kb_document_api(kb_id, doc_id):
    """从知识库中删除文档"""
    try:
        kb = get_knowledge_base(kb_id)
        if not kb:
            return jsonify({"error": "知识库不存在"}), 404
            
        success = kb.remove_document(doc_id)
        if success:
            return jsonify({"message": "从知识库中删除文档成功"})
        else:
            return jsonify({"error": "从知识库中删除文档失败，可能文档不存在"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"从知识库中删除文档失败: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/search', methods=['POST'])
def search_kb_api(kb_id):
    """在知识库中搜索"""
    try:
        data = request.json
        query = data.get('query')
        k = data.get('k', 5)  # 默认返回5个结果
        
        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400
            
        kb = get_knowledge_base(kb_id)
        if not kb:
            return jsonify({"error": "知识库不存在"}), 404
            
        results = kb.search(query, k)
        
        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0)
            })
            
        return jsonify({"data": formatted_results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"知识库搜索失败: {str(e)}"}), 500

if __name__ == '__main__':
    # 启动情绪监控
    # start_agent_monitoring_in_background()

    # 注册退出处理，确保应用退出时关闭连接
    import atexit
    @atexit.register
    def cleanup():
        """在应用退出时清理资源"""
        # 关闭PostgreSQL连接池
        # 检查pool是否存在并且有close方法
        if 'pool' in globals() and hasattr(pool, 'close'):
            print("正在关闭PostgreSQL连接池...")
            pool.close()
            print("PostgreSQL连接池已关闭。")

    # 启动 Flask 应用
    print("正在启动Flask应用...")
    app.run(host='0.0.0.0', port=5000, debug=True)    # 启动情绪监控
    start_agent_monitoring_in_background()

    # 注册退出处理，确保应用退出时关闭连接
    import atexit
    @atexit.register
    def cleanup():
        """在应用退出时清理资源"""
        # 关闭PostgreSQL连接池
        # 检查pool是否存在并且有close方法
        if 'pool' in globals() and hasattr(pool, 'close'):
            print("正在关闭PostgreSQL连接池...")
            pool.close()
            print("PostgreSQL连接池已关闭。")

    # 启动 Flask 应用
    print("正在启动Flask应用...")
    app.run(host='0.0.0.0', port=5000, debug=True)
