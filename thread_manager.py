import time
import uuid
import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ThreadManager:
    """线程管理器类，封装所有线程和消息管理功能"""
    def __init__(self):
        # 内存存储
        self.threads_store = {}  # 用户线程信息
        self.messages_store = {}  # 线程消息历史
        
        # 数据文件路径
        self.DATA_DIR = "data"
        self.THREADS_FILE = os.path.join(self.DATA_DIR, "threads.json")
        self.MESSAGES_FILE = os.path.join(self.DATA_DIR, "messages.json")
        
        # 确保数据目录存在
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # 加载存储数据
        self.load_data()
    
    def load_data(self):
        """从文件加载线程和消息数据"""
        # 加载线程数据
        if os.path.exists(self.THREADS_FILE):
            try:
                with open(self.THREADS_FILE, 'r', encoding='utf-8') as f:
                    self.threads_store = json.load(f)
            except Exception as e:
                print(f"加载线程数据出错: {e}")
        
        # 加载消息数据
        if os.path.exists(self.MESSAGES_FILE):
            try:
                with open(self.MESSAGES_FILE, 'r', encoding='utf-8') as f:
                    self.messages_store = json.load(f)
            except Exception as e:
                print(f"加载消息数据出错: {e}")
    
    def save_data(self):
        """将线程和消息数据保存到文件"""
        # 保存线程数据
        try:
            with open(self.THREADS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.threads_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存线程数据出错: {e}")
        
        # 保存消息数据
        try:
            with open(self.MESSAGES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.messages_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存消息数据出错: {e}")
    
    def save_thread(self, user_id: str, thread_id: str, title: str, system_prompt: str = None) -> None:
        """保存线程信息"""
        if user_id not in self.threads_store:
            self.threads_store[user_id] = {}
        
        self.threads_store[user_id][thread_id] = {
            'id': thread_id,
            'title': title,
            'createdAt': time.strftime('%Y-%m-%d %H:%M:%S'),
            'lastMessagePreview': ''
        }
        
        if system_prompt:
            self.threads_store[user_id][thread_id]['system_prompt'] = system_prompt
        
        self.save_data()
    
    def get_user_threads(self, user_id: str) -> List[Dict[str, str]]:
        """获取用户的所有线程"""
        if user_id not in self.threads_store:
            return []
        
        threads = list(self.threads_store[user_id].values())
        threads.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        
        return threads
    
    def remove_thread(self, user_id: str, thread_id: str) -> None:
        """删除线程"""
        if user_id in self.threads_store and thread_id in self.threads_store[user_id]:
            del self.threads_store[user_id][thread_id]
        
        if thread_id in self.messages_store:
            del self.messages_store[thread_id]
        
        self.save_data()
    
    def save_message(self, user_id: str, thread_id: str, role: str, content: str) -> None:
        """保存消息"""
        thread_key = thread_id
        
        if thread_key not in self.messages_store:
            self.messages_store[thread_key] = []
        
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.messages_store[thread_key].append(message)
        
        if user_id in self.threads_store and thread_id in self.threads_store[user_id]:
            preview = content[:30] + "..." if len(content) > 30 else content
            self.threads_store[user_id][thread_id]['lastMessagePreview'] = preview
        
        self.save_data()
    
    def get_chat_history(self, user_id: str, thread_id: str) -> List[Dict[str, str]]:
        """获取聊天历史"""
        thread_key = thread_id
        print(f"Attempting to fetch history with key: {thread_key}")

        if thread_key in self.messages_store:
            print("Found history with key.")
            return self.messages_store[thread_key]

        print("No matching history found.")
        return []
    
    def get_thread_messages(self, user_id: str, thread_id: str, system_prompt: str) -> List:
        """获取线程的消息历史，转换为LangGraph格式"""
        thread_key = thread_id
        
        custom_system_prompt = None
        if user_id in self.threads_store and thread_id in self.threads_store[user_id]:
            custom_system_prompt = self.threads_store[user_id][thread_id].get('system_prompt')
        
        final_system_prompt = custom_system_prompt if custom_system_prompt else system_prompt
        
        messages = [SystemMessage(content=final_system_prompt)]
        
        if thread_key not in self.messages_store:
            return messages
        
        for msg in self.messages_store[thread_key]:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        return messages

# 创建全局实例
thread_manager = ThreadManager()

# 导出函数接口以保持向后兼容性
def save_thread(user_id: str, thread_id: str, title: str, system_prompt: str = None) -> None:
    thread_manager.save_thread(user_id, thread_id, title, system_prompt)

def get_user_threads(user_id: str) -> List[Dict[str, str]]:
    return thread_manager.get_user_threads(user_id)

def remove_thread(user_id: str, thread_id: str) -> None:
    thread_manager.remove_thread(user_id, thread_id)

def save_message(user_id: str, thread_id: str, role: str, content: str) -> None:
    thread_manager.save_message(user_id, thread_id, role, content)

def get_chat_history(user_id: str, thread_id: str) -> List[Dict[str, str]]:
    return thread_manager.get_chat_history(user_id, thread_id)

def get_thread_messages(user_id: str, thread_id: str, system_prompt: str) -> List:
    return thread_manager.get_thread_messages(user_id, thread_id, system_prompt) 