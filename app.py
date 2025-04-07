import os
import time
import uuid
import threading
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

# 导入情绪监控启动函数
from emotion_monitor import start_monitoring_in_background

# 初始化 Flask 应用
app = Flask(__name__)

# 配置 CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"], "supports_credentials": True}})

# 创建 LangGraph 图，全局共享
try:
    graph = create_graph()
    print("LangGraph 图创建成功")
except Exception as e:
    print(f"创建 LangGraph 图时出错: {e}")
    graph = None

# API 流式输出接口
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
            user_id = request.args.get('userId') # 直接获取
            thread_id = request.args.get('threadId') # 直接获取
        else:  # GET
            user_message = request.args.get('message', '')
            user_id = request.args.get('userId') # 直接获取
            thread_id = request.args.get('threadId') # 直接获取
            
        if not thread_id or not user_id:
            return jsonify({'error': '缺少必要参数 threadId 或 userId'}), 400
            
        if not user_message.strip():
            return jsonify({'error': '请输入有效的消息'}), 400

        # 验证并规范化 thread_id 格式
        # 检查传入的 thread_id 是否符合 "username_UUID" 格式
        # 假设正确格式是: username_uuid，至少有一个下划线
        if '_' not in thread_id or thread_id.startswith('temp_'):
            print(f"检测到不符合标准的 thread_id: {thread_id}，将创建标准格式的 thread_id")
            # 创建新的符合格式的 thread_id
            new_thread_id = f"{user_id}_{str(uuid.uuid4())}"
            print(f"创建新的 thread_id: {new_thread_id}")
            
            # 保存新线程（如果不存在）
            title = f"聊天 {time.strftime('%Y-%m-%d %H:%M:%S')}"
            save_thread(user_id, new_thread_id, title)
            
            # 使用新的 thread_id
            thread_id = new_thread_id
        
        print(f"使用规范化后的 thread_id: {thread_id}")
        
        # 准备消息历史 (使用规范化后的 ID)
        system_prompt = "你是一个有用的AI助手。"
        messages = get_thread_messages(user_id, thread_id, system_prompt)
        # 配置线程ID (使用规范化后的 ID)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "enable_memory": True
            }
        }
        
        response_id = str(uuid.uuid4())
        
        
        def generate():
            """SSE 事件生成器"""
            # 初始化响应内容
            full_response = ""
            
            # SSE 格式的前导数据 - 每个事件必须以\n\n结尾
            yield f"event: start\ndata: {json.dumps({'response_id': response_id})}\n\n"
            
            # 使用 process_graph_stream 流式获取响应
            for chunk in process_graph_stream(
                graph, 
                user_message, 
                history=messages, 
                config=config
            ):
                # 累积完整响应
                full_response += chunk
                
                # 发送当前块作为 SSE 事件
                yield f"event: chunk\ndata: {json.dumps({'chunk': chunk, 'response_id': response_id})}\n\n"
            
            # 发送完成事件
            yield f"event: complete\ndata: {json.dumps({'full_response': full_response, 'response_id': response_id})}\n\n"
            
            # 保存对话历史 (使用规范化后的 ID)
            save_message(user_id, thread_id, "user", user_message)
            save_message(user_id, thread_id, "assistant", full_response)
        
        # 返回 SSE 流
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # 对 Nginx 禁用缓冲
                'Connection': 'keep-alive',  # 保持连接
                'Content-Type': 'text/event-stream',  # 明确指定 SSE 内容类型
                'Transfer-Encoding': 'chunked',  # 分块传输
                'Access-Control-Allow-Origin': 'http://localhost:5173',  # 明确指定允许的源
                'Access-Control-Allow-Credentials': 'true',  # 允许凭证
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
        system_prompt = data.get('systemPrompt') # 可选的系统提示词
        
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




# 系统提示词设置
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
        thread_user_id = None
        if '_' in thread_id:
            parts = thread_id.split('_')
            thread_user_id = parts[0]
            print(f"从线程ID中提取的用户名: {thread_user_id}")
        
        # 优先使用线程ID中的用户名，如果没有则使用前端传入的
        user_id = thread_user_id if thread_user_id else frontend_user_id
        print(f"使用用户ID: {user_id}, 前端传入: {frontend_user_id}, 线程ID: {thread_id}")
        
        # 获取线程信息
        import os
        import json
        threads_file_path = os.path.join(os.path.dirname(__file__), 'data', 'threads.json')
        
        # 读取所有线程信息
        all_users_threads = {}
        if os.path.exists(threads_file_path):
            try:
                with open(threads_file_path, 'r', encoding='utf-8') as f:
                    all_users_threads = json.load(f)
            except Exception as e:
                print(f"加载线程数据出错: {e}")
                return jsonify({'error': f'加载线程数据出错: {str(e)}'}), 500
        
        # 检查线程所有者
        actual_owner = None
        for uid, threads in all_users_threads.items():
            if thread_id in threads:
                actual_owner = uid
                break
                
        print(f"线程实际所有者: {actual_owner}")
        
        # 如果找到实际所有者，优先使用实际所有者ID
        # 否则使用从线程ID提取的用户名或前端传入的用户ID
        target_user_id = actual_owner if actual_owner else user_id
        
        # 确保用户存在于线程存储中
        if target_user_id not in all_users_threads:
            all_users_threads[target_user_id] = {}
            
        # 更新或创建线程
        if actual_owner and thread_id in all_users_threads[actual_owner]:
            # 更新已有线程
            all_users_threads[actual_owner][thread_id]['system_prompt'] = system_prompt
            print(f"更新用户 {actual_owner} 的线程 {thread_id} 的系统提示词")
            
            # 保存更新后的线程数据
            with open(threads_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_users_threads, f, ensure_ascii=False, indent=2)
                
            return jsonify({'status': 'success', 'message': '系统提示词已更新'})
        else:
            # 创建新线程
            # 通常不应该到这一步，因为线程应该在聊天前已经创建
            # 但为了健壮性，我们仍保留这部分逻辑
            all_users_threads[target_user_id][thread_id] = {
                'id': thread_id,
                'title': f"聊天 {time.strftime('%Y-%m-%d %H:%M:%S')}",
                'createdAt': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lastMessagePreview': '',
                'system_prompt': system_prompt
            }
            print(f"为用户 {target_user_id} 创建线程 {thread_id} 并设置系统提示词")
            
            # 保存更新后的线程数据
            with open(threads_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_users_threads, f, ensure_ascii=False, indent=2)
                
            return jsonify({'status': 'success', 'message': '已创建线程并设置系统提示词'})
                
    except Exception as e:
        print(f"更新系统提示词时出错: {e}")
        return jsonify({'error': f'更新系统提示词时出错: {str(e)}'}), 500


# --- 应用启动入口 --- 
if __name__ == '__main__':
    print("启动 Flask 应用...")
    
    # 在启动 Flask 应用前，先启动后台情绪监控线程
    # 确保只在主进程中启动一次 (如果使用多进程模式如 Gunicorn 需要注意)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        try:
            start_monitoring_in_background()
        except Exception as e:
            print(f"启动情绪监控线程时出错: {e}")

    # 启动 Flask 开发服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
    # 生产环境建议使用 Gunicorn 或 uWSGI: gunicorn -w 4 -b 0.0.0.0:5000 app:app