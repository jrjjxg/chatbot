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
from emotion_monitor_agent import start_agent_monitoring_in_background, test_emotion_monitor_agent, Command, EnhancedEmotionMonitorAgent

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
                 print(f"警告: 请求参数 userId '{request_user_id}' 与 thread_id 中的 userId '{actual_user_id}' 不一致，将优先使用 thread_id 中的用户ID")

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
            thread_id = new_thread_id # 更新 thread_id 为新创建的
        
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
                        print(f"找到用户 '{actual_user_id}' 的线程 '{thread_id}' 的自定义系统提示词: {custom_system_prompt}")
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
        
        # 配置线程ID
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": actual_user_id, # 使用确认后的用户ID
                "enable_memory": True
            }
        }
        
        response_id = str(uuid.uuid4())
        
        def generate():
            """SSE 事件生成器"""
            full_response = ""
            yield f"event: start\ndata: {json.dumps({'response_id': response_id})}\n\n"
            
            for chunk in process_graph_stream(
                graph, 
                user_message, 
                history=messages, 
                config=config
            ):
                full_response += chunk
                yield f"event: chunk\ndata: {json.dumps({'chunk': chunk, 'response_id': response_id})}\n\n"
            
            yield f"event: complete\ndata: {json.dumps({'full_response': full_response, 'response_id': response_id})}\n\n"
            
            # 使用 actual_user_id 保存消息
            save_message(actual_user_id, thread_id, "user", user_message)
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
    start_agent_monitoring_in_background()
    
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=True)