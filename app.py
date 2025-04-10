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
            user_id = request.args.get('userId')  # 直接获取
            thread_id = request.args.get('threadId')  # 直接获取
        else:  # GET
            user_message = request.args.get('message', '')
            user_id = request.args.get('userId')  # 直接获取
            thread_id = request.args.get('threadId')  # 直接获取
            
        if not thread_id or not user_id:
            return jsonify({'error': '缺少必要参数 threadId 或 userId'}), 400
        
        if not user_message.strip():
            return jsonify({'error': '请输入有效的消息'}), 400
        
        # 验证并规范化 thread_id 格式
        if '_' not in thread_id or thread_id.startswith('temp_'):
            print(f"检测到不符合标准的 thread_id: {thread_id}，将创建标准格式的 thread_id")
            new_thread_id = f"{user_id}_{str(uuid.uuid4())}"
            print(f"创建新的 thread_id: {new_thread_id}")
            
            title = f"聊天 {time.strftime('%Y-%m-%d %H:%M:%S')}"
            save_thread(user_id, new_thread_id, title)
            thread_id = new_thread_id
        
        print(f"使用规范化后的 thread_id: {thread_id}")
        
        # 准备消息历史
        system_prompt = "你是一个有用的AI助手。"
        messages = get_thread_messages(user_id, thread_id, system_prompt)
        
        # 配置线程ID
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
            
            save_message(user_id, thread_id, "user", user_message)
            save_message(user_id, thread_id, "assistant", full_response)
        
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

@app.route('/api/test/emotion-monitor', methods=['POST'])
def test_emotion_monitor():
    """测试情绪监控功能"""
    try:
        data = request.json
        user_id = data.get('userId')
        
        if not user_id:
            return jsonify({'error': '缺少用户ID'}), 400
            
        # 执行测试
        result = test_emotion_monitor_agent(user_id)
        
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'测试情绪监控时出错: {str(e)}'}), 500

@app.route('/api/emotion-monitor/confirm', methods=['POST'])
def confirm_emotion_notification():
    """确认情绪监控通知"""
    try:
        data = request.json
        user_id = data.get('userId')
        thread_id = data.get('threadId')
        confirmed = data.get('confirmed', False)
        
        if not user_id or not thread_id:
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 这里可以添加确认逻辑
        # 例如：更新数据库中的通知状态
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'确认通知时出错: {str(e)}'}), 500

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

@app.route('/api/confirm', methods=['POST'])
async def confirm():
    """处理用户确认"""
    try:
        data = request.json
        confirmed = data.get('confirmed', False)
        
        # 这里可以添加确认处理逻辑
        
        return jsonify({'status': 'success', 'confirmed': confirmed})
    except Exception as e:
        return jsonify({'error': f'处理确认时出错: {str(e)}'}), 500

if __name__ == '__main__':
    # 启动情绪监控
    start_agent_monitoring_in_background()
    
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=True)