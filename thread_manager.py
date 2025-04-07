import time
import uuid
import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 内存存储（开发阶段使用）
threads_store = {}  # 用户线程信息
messages_store = {}  # 线程消息历史

# 数据文件路径
DATA_DIR = "data"
THREADS_FILE = os.path.join(DATA_DIR, "threads.json")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 加载存储数据
def load_data():
    """从文件加载线程和消息数据"""
    global threads_store, messages_store
    
    # 加载线程数据
    if os.path.exists(THREADS_FILE):
        try:
            with open(THREADS_FILE, 'r', encoding='utf-8') as f:
                threads_store = json.load(f)
        except Exception as e:
            print(f"加载线程数据出错: {e}")
    
    # 加载消息数据
    if os.path.exists(MESSAGES_FILE):
        try:
            with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
                messages_store = json.load(f)
        except Exception as e:
            print(f"加载消息数据出错: {e}")

# 保存存储数据
def save_data():
    """将线程和消息数据保存到文件"""
    # 保存线程数据
    try:
        with open(THREADS_FILE, 'w', encoding='utf-8') as f:
            json.dump(threads_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存线程数据出错: {e}")
    
    # 保存消息数据
    try:
        with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存消息数据出错: {e}")

# 线程管理功能
def save_thread(user_id: str, thread_id: str, title: str, system_prompt: str = None) -> None:
    """保存线程信息
    
    Args:
        user_id: 用户ID
        thread_id: 线程ID
        title: 线程标题
        system_prompt: 可选的自定义系统提示词
    """
    if user_id not in threads_store:
        threads_store[user_id] = {}
    
    threads_store[user_id][thread_id] = {
        'id': thread_id,
        'title': title,
        'createdAt': time.strftime('%Y-%m-%d %H:%M:%S'),
        'lastMessagePreview': ''
    }
    
    # 如果提供了自定义系统提示词，保存它
    if system_prompt:
        threads_store[user_id][thread_id]['system_prompt'] = system_prompt
    
    # 保存到文件
    save_data()

def get_user_threads(user_id: str) -> List[Dict[str, str]]:
    """获取用户的所有线程"""
    if user_id not in threads_store:
        return []
    
    # 按创建时间降序排序，最新的在前面
    threads = list(threads_store[user_id].values())
    threads.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
    
    return threads

def remove_thread(user_id: str, thread_id: str) -> None:
    """删除线程"""
    if user_id in threads_store and thread_id in threads_store[user_id]:
        del threads_store[user_id][thread_id]
    
    # 同时删除相关消息 - 直接使用 thread_id 作为键
    if thread_id in messages_store:
        del messages_store[thread_id]
    
    # 保存到文件
    save_data()

# 消息管理功能
def save_message(user_id: str, thread_id: str, role: str, content: str) -> None:
    """保存消息"""
    # 直接使用 thread_id 作为键
    thread_key = thread_id 
    
    if thread_key not in messages_store:
        messages_store[thread_key] = []
    
    # 添加新消息
    message = {
        'id': str(uuid.uuid4()),
        'role': role,
        'content': content,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    messages_store[thread_key].append(message)
    
    # 更新线程的最新消息预览
    if user_id in threads_store and thread_id in threads_store[user_id]:
        preview = content[:30] + "..." if len(content) > 30 else content
        threads_store[user_id][thread_id]['lastMessagePreview'] = preview
    
    # 保存到文件
    save_data()

def get_chat_history(user_id: str, thread_id: str) -> List[Dict[str, str]]:
    """获取聊天历史"""
    # 直接使用 thread_id 作为键
    thread_key = thread_id
    print(f"Attempting to fetch history with key: {thread_key}")

    # 尝试使用正确的键
    if thread_key in messages_store:
        print("Found history with key.")
        return messages_store[thread_key]

    # 如果找不到，返回空列表 (移除兼容逻辑)
    print("No matching history found.")
    return []

def get_thread_messages(user_id: str, thread_id: str, system_prompt: str) -> List:
    """获取线程的消息历史，转换为LangGraph格式"""
    # 直接使用 thread_id 作为键
    thread_key = thread_id
    
    # 检查线程中是否存在自定义系统提示词
    custom_system_prompt = None
    if user_id in threads_store and thread_id in threads_store[user_id]:
        custom_system_prompt = threads_store[user_id][thread_id].get('system_prompt')
    
    # 如果线程中存在自定义提示词，使用它；否则使用默认提示词
    final_system_prompt = custom_system_prompt if custom_system_prompt else system_prompt
    
    # 准备消息列表，始终以系统提示开始
    messages = [SystemMessage(content=final_system_prompt)]
    
    # 如果没有历史消息，只返回系统提示
    if thread_key not in messages_store:
        return messages
    
    # 添加历史消息
    for msg in messages_store[thread_key]:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content']))
    
    return messages

# 初始加载数据
load_data() 