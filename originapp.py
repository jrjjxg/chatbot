import os
import time
import uuid
import threading
from flask import Flask, request, Response, jsonify, g
from flask_cors import CORS
import json
import psycopg # 确保导入psycopg
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
import requests
import datetime

# 导入需要的函数，避免循环导入
from main import (
    create_graph,
    process_graph_stream,
    get_default_system_message,
    get_llm,
    get_thread_messages
)

# 导入线程管理模块
from thread_manager import (
    save_thread,
    get_user_threads,
    remove_thread,
    save_message,
    get_chat_history,
    get_thread_messages
)

# 添加一个获取日记详情的函数
def get_journal_detail(journal_id, user_id=None):
    """获取日记详情 (使用正确的 /get/{id} 路径)"""
    try:
        # 配置Java后端API基础URL
        SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "http://localhost:9000")
        
        # --- 修改点: 构建正确的API URL，使用路径参数 ---
        # 注意: Java端 /get/{id} 接口似乎不直接接收 userId 作为路径或查询参数
        # 它通过 UserUtil.getCurrentUserId() 获取用户ID，这通常依赖于请求头中的认证信息（如JWT）
        api_url = f"{SPRING_BOOT_BASE_URL}/api/journal/get/{journal_id}" 
        
        print(f"获取日记详情 (修正后的URL): {api_url}")
        
        # --- 修改点: 移除不必要的 params={'userId': user_id}，因为接口路径已包含ID ---
        # 但仍可能需要传递认证信息，例如通过headers
        headers = {} 
        # 如果需要认证，可能需要类似:
        # if user_id: 
        #    # 假设需要某种形式的Token或Session ID传递用户身份
        #    # headers['Authorization'] = f'Bearer {get_user_token(user_id)}' 
        #    # 或者 headers['Cookie'] = f'JSESSIONID={get_user_session(user_id)}'
        #    pass 

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

# 修改 thread_manager.py 中的 save_message 函数（如果不能直接修改，这里提供一个增强版本）
def save_message_enhanced(user_id, thread_id, role, content, msg_type='text', metadata=None):
    """保存消息到会话历史，支持特殊类型的消息"""
    try:
        messages_file = os.path.join("data", "messages.json")
        
        try:
            if os.path.exists(messages_file):
                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)
            else:
                messages_data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            messages_data = {}
        
        # 创建消息对象
        timestamp = datetime.datetime.now().isoformat()
        message_id = f"{role}-{int(time.time()*1000)}"
        
        # 基本消息结构
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        # 处理特殊类型消息
        if msg_type != 'text':
            message["type"] = msg_type
            
            # 如果是日记分享卡片，添加额外字段
            if msg_type == 'diary_share' and metadata:
                for key, value in metadata.items():
                    if key not in message:  # 避免覆盖基本字段
                        message[key] = value
        
        # 确保线程ID存在
        if thread_id not in messages_data:
            messages_data[thread_id] = []
        
        # 添加消息
        messages_data[thread_id].append(message)
        
        # 写入文件
        with open(messages_file, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
            
        print(f"消息已保存 - 线程: {thread_id}, 角色: {role}, 类型: {msg_type}")
        return True
    except Exception as e:
        print(f"保存消息失败: {str(e)}")
        return False

# 初始化 Flask 应用
app = Flask(__name__)

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
# 为应用的常规操作创建连接池 (连接池中的连接默认不需要是autocommit)
print("正在创建数据库连接池...")
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    min_size=2,  # 确保始终有一些连接可用
    timeout=30,  # 获取连接的超时时间
    max_lifetime=60*5,  # 连接最大寿命5分钟
    max_idle=60*2  # 空闲连接最多保留2分钟
)
print("数据库连接池创建成功。")

# 使用连接池初始化Checkpointer和Store
print("正在使用连接池初始化Checkpointer和Store...")
# 将连接池作为第一个位置参数传递
app.checkpointer = PostgresSaver(pool)
print("Checkpointer和Store初始化完成。")

# 全局编译好的图实例 (包含 checkpointer 和 store)
print("正在编译LangGraph图...")
app.runnable = graph_builder.compile(
    checkpointer=app.checkpointer
)
print("LangGraph图编译成功。")


@app.route('/api/chat/stream', methods=['GET', 'POST', 'OPTIONS'])
def chat_stream_api():
    """为前端提供流式输出的API接口"""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    try:
        # 支持GET和POST两种请求方式
        if request.method == 'POST':
            data = request.json
            user_message = data.get('message', '')
            request_user_id = request.args.get('userId')  # 请求参数中的 userId
            thread_id = request.args.get('threadId')
        else:  # GET
            user_message = request.args.get('message', '')
            request_user_id = request.args.get('userId')  # 请求参数中的 userId
            thread_id = request.args.get('threadId')

        if not thread_id:
            return jsonify({'error': '缺少必要参数 threadId'}), 400

        # 尝试从thread_id中提取user_id
        thread_user_id = None
        if '_' in thread_id:
            thread_user_id = thread_id.split('_')[0]

        # 如果无法从thread_id提取，则使用请求参数中的userId，并打印警告
        if not thread_user_id:
            print(f"警告: 无法从 thread_id '{thread_id}' 提取 user_id, 将使用请求参数中的 user_id: {request_user_id}")
            actual_user_id = request_user_id
        else:
            actual_user_id = thread_user_id
            # 如果提取出的ID和参数ID不一致，打印警告
            if request_user_id and request_user_id != actual_user_id:
                print(
                    f"警告: 请求参数 userId '{request_user_id}' 与 thread_id 中的 userId '{actual_user_id}' 不一致，将优先使用 thread_id 中的用户ID")

        if not actual_user_id:
            return jsonify({'error': '无法确定有效的用户ID'}), 400

        if not user_message.strip():
            return jsonify({'error': '请输入有效的消息'}), 400

        # 验证并规范化 thread_id 格式 (如果需要创建新线程，要用actual_user_id)
        if '_' not in thread_id or thread_id.startswith('temp_'):
            print(f"检测到不符合标准的 thread_id: {thread_id}，将创建标准格式的 thread_id")
            new_thread_id = f"{actual_user_id}_{str(uuid.uuid4())}"
            print(f"创建新的 thread_id: {new_thread_id}")

            title = f"聊天 {time.strftime('%Y-%m-%d %H:%M:%S')}"
            # 使用 actual_user_id 保存线程
            save_thread(actual_user_id, new_thread_id, title)
            thread_id = new_thread_id  # 更新 thread_id 为新创建的

        print(f"使用规范化后的 thread_id: {thread_id}，关联的用户ID: {actual_user_id}")

        # 直接从线程存储中获取自定义系统提示词
        default_system_prompt = "你是一个有用的AI助手。"
        custom_system_prompt = None

        # 读取存储的线程数据
        threads_file = os.path.join("data", "threads.json")
        if os.path.exists(threads_file):
            try:
                with open(threads_file, 'r', encoding='utf-8') as f:
                    thread_data = json.load(f)
                    # 使用从thread_id提取或确认后的 actual_user_id 进行查找
                    if actual_user_id in thread_data and thread_id in thread_data[actual_user_id]:
                        # 读取自定义system_prompt
                        custom_system_prompt = thread_data[actual_user_id][thread_id].get('system_prompt')
                        print(
                            f"找到用户 '{actual_user_id}' 的线程 '{thread_id}' 的自定义系统提示词: {custom_system_prompt}")
                    else:
                        print(f"在用户 '{actual_user_id}' 下未找到线程 '{thread_id}' 或其自定义提示词")
            except Exception as e:
                print(f"读取线程文件出错: {e}")

        # 确定使用哪个系统提示词
        system_prompt = custom_system_prompt if custom_system_prompt else default_system_prompt
        print(f"最终决定使用的系统提示词: {system_prompt}")

        # 获取历史消息
        thread_key = thread_id
        history_messages = []

        # 读取消息文件
        messages_file = os.path.join("data", "messages.json")
        if os.path.exists(messages_file):
            try:
                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)
                    if thread_key in messages_data:
                        history_messages = messages_data[thread_key]
            except Exception as e:
                print(f"读取消息历史出错: {e}")

        # 构建消息列表
        messages = [SystemMessage(content=system_prompt)]
        for msg in history_messages:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

        print(f"构建的对话历史长度: {len(messages)}，第一条是系统提示词: {messages[0].content}")

        # 检查是否为日记分享触发消息
        is_share_trigger = user_message.startswith('//share_diary:')
        shared_diary_id = None
        diary_details = None
        input_for_llm = user_message  # 默认使用原始消息
        
        if is_share_trigger:
            try:
                # 提取日记ID
                shared_diary_id = user_message.split(':', 1)[1].strip()
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

        # 配置线程ID
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": actual_user_id,  # 使用确认后的用户ID
                "enable_memory": True
            }
        }

        response_id = str(uuid.uuid4())

        def generate():
            """SSE 事件生成器"""
            full_response = ""
            yield f"event: start\ndata: {json.dumps({'response_id': response_id})}\n\n"
            
            # 如果是日记分享，先保存卡片消息
            if is_share_trigger and diary_details:
                try:
                    # 创建卡片消息
                    card_id = f"diary-{shared_diary_id}-{int(time.time())}"
                    card_metadata = {
                        "diaryId": shared_diary_id,
                        "diaryTitle": diary_details.get("title", "无标题日记"),
                        "diaryDate": diary_details.get("createTime")
                    }
                    
                    # 保存卡片消息
                    save_message_enhanced(
                        actual_user_id,
                        thread_id,
                        "system",  # 用system角色
                        f"分享了日记: {card_metadata['diaryTitle']}",  # 显示内容
                        msg_type='diary_share',  # 特殊类型
                        metadata=card_metadata  # 额外数据
                    )
                    print(f"已保存日记分享卡片: {card_id}")
                except Exception as e:
                    print(f"保存日记分享卡片失败: {e}")

            for chunk in process_graph_stream(
                    app.runnable,
                    input_for_llm,  # 使用可能经过处理的消息
                    history=messages,
                    config=config
            ):
                full_response += chunk
                yield f"event: chunk\ndata: {json.dumps({'chunk': chunk, 'response_id': response_id})}\n\n"

            yield f"event: complete\ndata: {json.dumps({'full_response': full_response, 'response_id': response_id})}\n\n"

            # 保存消息到历史记录
            # 对于分享触发消息，保存触发消息本身（可选）
            if not is_share_trigger:
                # 只保存非触发的普通用户消息
                save_message(actual_user_id, thread_id, "user", user_message)
            else:
                # 触发消息可以选择保存或不保存
                # 这里选择保存，方便调试，如果不需要可以去掉
                save_message(actual_user_id, thread_id, "user", user_message)
            
            # 保存AI回复
            save_message(actual_user_id, thread_id, "assistant", full_response)

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'Transfer-Encoding': 'chunked',
                'Access-Control-Allow-Origin': 'http://localhost:5173',
                'Access-Control-Allow-Credentials': 'true',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            }
        )
    except Exception as e:
        print(f"处理流式消息时出错: {e}")
        return jsonify({'error': f'处理消息时出错: {str(e)}'}), 500


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

        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400

        # 获取用户的所有线程
        threads = get_user_threads(user_id)

        return jsonify(threads)
    except Exception as e:
        return jsonify({'error': f'获取线程列表时出错: {str(e)}'}), 500


@app.route('/api/history/<thread_id>', methods=['GET'])
def get_history(thread_id):
    """获取聊天历史记录"""
    try:
        user_id = request.args.get('userId')

        if not user_id or not thread_id:
            return jsonify({'error': '缺少必要参数 userId 或 threadId'}), 400

        # 获取聊天历史
        history = get_chat_history(user_id, thread_id)

        return jsonify(history)
    except Exception as e:
        print(f"获取聊天历史时出错: {e}")
        return jsonify({'error': f'获取聊天历史时出错: {str(e)}'}), 500


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


@app.route('/api/chat', methods=['POST'])
async def chat():
    """处理聊天请求"""
    try:
        data = request.json
        message = data.get('message')
        history = data.get('history', [])

        if not message:
            return jsonify({'error': '消息不能为空'}), 400

        # 调用LLM处理消息
        llm = get_llm()
        response = await llm.ainvoke(message)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'处理消息时出错: {str(e)}'}), 500


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
    app.run(host='0.0.0.0', port=5000, debug=True)
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
    app.run(host='0.0.0.0', port=5000, debug=True)