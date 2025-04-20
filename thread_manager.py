import time
import uuid
import json
import os
import datetime
import psycopg
import calendar # 导入 calendar 模块
from psycopg.rows import dict_row # 用于将查询结果转为字典
from psycopg_pool import ConnectionPool
import traceback
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from collections import defaultdict # 导入 defaultdict

# 全局变量，用于存储从 app.py 传入的连接池
db_pool: ConnectionPool = None

def initialize_db_pool(pool: ConnectionPool):
    """初始化数据库连接池"""
    global db_pool
    if db_pool is None:
        db_pool = pool
        print("ThreadManager: 数据库连接池已初始化。")
    else:
        print("ThreadManager: 数据库连接池已存在。")

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

    def save_thread(self, user_id: str, thread_id: str, title: str, system_prompt: str = None) -> bool:
        """保存线程信息到数据库"""
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            return False
        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    # 使用UPSERT逻辑：如果thread_id已存在，则更新title和system_prompt；否则插入新记录
                    sql = """
                    INSERT INTO app_threads (user_id, thread_id, title, system_prompt, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        system_prompt = EXCLUDED.system_prompt,
                        updated_at = EXCLUDED.updated_at;
                    """
                    timestamp = datetime.datetime.now(datetime.timezone.utc)
                    cur.execute(sql, (user_id, thread_id, title, system_prompt, timestamp))
                    conn.commit()
                    print(f"线程信息已保存/更新到数据库: {thread_id}")
                    return True
        except Exception as e:
            print(f"保存线程信息到数据库失败: {e}")
            traceback.print_exc()
            return False

    def get_user_threads(self, user_id: str, group_by_date: bool = False) -> list:
        """从数据库获取用户的所有线程信息"""
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            return []
        threads = []
        try:
            with db_pool.connection() as conn:
                # 使用dict_row将结果行映射为字典
                with conn.cursor(row_factory=dict_row) as cur:
                    sql = """
                    SELECT 
                        thread_id as id, 
                        title, 
                        system_prompt, 
                        created_at, 
                        updated_at,
                        -- 获取最后一条消息的预览和时间 (需要优化SQL)
                        (SELECT content FROM app_messages m WHERE m.thread_id = t.thread_id ORDER BY m.timestamp DESC LIMIT 1) as "lastMessagePreview",
                        (SELECT timestamp FROM app_messages m WHERE m.thread_id = t.thread_id ORDER BY m.timestamp DESC LIMIT 1) as "lastMessageTime"
                    FROM app_threads t
                    WHERE user_id = %s
                    ORDER BY updated_at DESC; 
                    """
                    cur.execute(sql, (user_id,))
                    threads = cur.fetchall()
                    # 将时间戳转换为ISO格式字符串
                    for thread in threads:
                        if thread.get("createdAt"):
                           thread["createdAt"] = thread["createdAt"].isoformat()
                        if thread.get("updatedAt"):
                           thread["updatedAt"] = thread["updatedAt"].isoformat()
                        if thread.get("lastMessageTime"):
                           thread["lastMessageTime"] = thread["lastMessageTime"].isoformat()

                    print(f"从数据库获取到用户 {user_id} 的 {len(threads)} 个线程")
        except Exception as e:
            print(f"从数据库获取线程列表失败: {e}")
            traceback.print_exc()
        return threads if threads else []

    def remove_thread(self, user_id: str, thread_id: str) -> bool:
        """从数据库删除线程及其相关消息"""
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            return False
        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    # 使用事务确保原子性
                    # 1. 删除消息
                    sql_delete_messages = "DELETE FROM app_messages WHERE thread_id = %s AND user_id = %s;"
                    cur.execute(sql_delete_messages, (thread_id, user_id))
                    deleted_messages_count = cur.rowcount
                    
                    # 2. 删除线程
                    sql_delete_thread = "DELETE FROM app_threads WHERE thread_id = %s AND user_id = %s;"
                    cur.execute(sql_delete_thread, (thread_id, user_id))
                    deleted_threads_count = cur.rowcount

                    conn.commit()
                    print(f"从数据库删除线程 {thread_id} (及 {deleted_messages_count} 条消息) 成功: {deleted_threads_count > 0}")
                    return deleted_threads_count > 0
        except Exception as e:
            print(f"从数据库删除线程失败: {e}")
            traceback.print_exc()
            return False

    def save_message(self, user_id: str, thread_id: str, role: str, content: str, msg_type: str = 'text', metadata: dict = None) -> bool:
        """
        保存消息到PostgreSQL数据库，支持特殊类型的消息 (替代 save_message_enhanced)
        """
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            # 尝试文件备份
            try:
                backup_file = os.path.join("data", "messages_backup_pool_error.jsonl")
                # ... (省略备份逻辑，同之前) ...
            except Exception as backup_error:
                print(f"数据库未初始化，消息备份也失败了: {backup_error}")
            return False
        
        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    sql = """
                    INSERT INTO app_messages 
                    (thread_id, user_id, role, content, msg_type, metadata, timestamp) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """
                    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                    timestamp = datetime.datetime.now(datetime.timezone.utc)
                    
                    cur.execute(
                        sql, 
                        (thread_id, user_id, role, content, msg_type, metadata_json, timestamp)
                    )
                    message_id = cur.fetchone()[0]
                    conn.commit()
                    print(f"消息已保存到数据库 - ID: {message_id}, 线程: {thread_id}, 类型: {msg_type}")
                    return True
                
        except Exception as e:
            print(f"保存消息到数据库失败: {str(e)}")
            traceback.print_exc()
            # 尝试文件备份 (同之前)
            try:
                backup_file = os.path.join("data", "messages_backup_db_error.jsonl")
                # ... (省略备份逻辑) ...
            except Exception as backup_error:
                print(f"数据库错误，消息备份也失败了: {str(backup_error)}")
            return False

    # --- 新增：保存文件上传消息 ---
    def save_file_upload_message(self, user_id: str, thread_id: str, filename: str, file_type: str) -> bool:
        """
        将文件上传事件作为一个特殊消息保存到数据库，以便在聊天记录中显示。
        """
        if not db_pool:
            print("错误: 数据库连接池未初始化，无法保存文件上传消息。")
            # 可以考虑添加备份逻辑，但对于这种系统消息可能不是必须的
            return False

        print(f"[ThreadManager] Saving file upload message for thread {thread_id}: {filename} ({file_type})")

        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    sql = """
                    INSERT INTO app_messages
                    (thread_id, user_id, role, content, msg_type, metadata, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """
                    # 使用 'system' 或 'file_upload' 作为角色，'file_info' 作为消息类型
                    role = 'system' # 或者可以定义一个更具体的角色
                    msg_type = 'file_info'
                    # Content 可以是描述性的，或者就是文件名
                    content = f"用户上传了文件: {filename}"
                    # 将文件名和类型存入 metadata
                    metadata_dict = {"filename": filename, "file_type": file_type}
                    metadata_json = json.dumps(metadata_dict, ensure_ascii=False)
                    timestamp = datetime.datetime.now(datetime.timezone.utc)

                    cur.execute(
                        sql,
                        (thread_id, user_id, role, content, msg_type, metadata_json, timestamp)
                    )
                    message_id = cur.fetchone()[0]
                    conn.commit()
                    print(f"[ThreadManager] 文件上传消息已保存到数据库 - ID: {message_id}, 线程: {thread_id}")
                    return True

        except Exception as e:
            print(f"[ThreadManager] 保存文件上传消息到数据库失败: {str(e)}")
            traceback.print_exc()
            return False
    # --- 结束新增 ---

    def get_chat_history(self, user_id: str, thread_id: str) -> dict:
        """从PostgreSQL数据库获取聊天历史记录"""
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            return {"data": [], "error": "数据库连接池未初始化"}
        
        messages = []
        try:
            with db_pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    sql = """
                    SELECT id, role, content, msg_type, metadata, timestamp 
                    FROM app_messages 
                    WHERE thread_id = %s AND user_id = %s
                    ORDER BY timestamp ASC
                    """
                    print(f"Executing history query for thread_id: '{thread_id}', user_id: '{user_id}'") # 添加日志
                    cur.execute(sql, (thread_id, user_id))
                    
                    rows = cur.fetchall() # 获取所有匹配行
                    
                    for row in rows:
                        message = {
                            "id": f"msg-{row['id']}", # 使用数据库ID构造唯一前端ID
                            "role": row['role'],
                            "timestamp": row['timestamp'].isoformat() if row['timestamp'] else None
                        }
                        
                        msg_type = row.get('msg_type')
                        # 处理消息类型
                        if msg_type and msg_type != 'text':
                            message["type"] = msg_type
                        
                        metadata_json = row.get('metadata')
                        print(f"metadata_json类型: {type(metadata_json)}, 值: {metadata_json}")
                        
                        # 处理消息内容
                        if msg_type == 'file_info':
                            # 对于file_info类型，拼接文件信息作为字符串
                            if metadata_json:
                                # 直接将metadata添加到message对象中
                                for key, value in metadata_json.items():
                                    message[key] = value
                                
                                # 将content作为message字段
                                message["message"] = f"用户上传了文件: {metadata_json.get('filename', '未知文件')}"
                            else:
                                # 如果没有metadata，使用content作为后备
                                message["message"] = row['content']
                        else:
                            # 对于其他类型的消息，使用content字段
                            message["message"] = row['content']
                        
                        # 处理其他类型消息的metadata
                        if metadata_json and msg_type != 'file_info':
                            # 确保所有消息类型都有metadata相关字段
                            for key, value in metadata_json.items():
                                if key not in message:
                                    message[key] = value
                                    
                        messages.append(message)
                        
                    print(f"从数据库获取到线程 {thread_id} 的 {len(messages)} 条消息")
                    return {"data": messages} # 返回符合前端期望的结构
                
        except Exception as e:
            print(f"从数据库获取聊天历史失败: {str(e)}")
            traceback.print_exc()
            return {"data": [], "error": str(e)}

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

    def get_threads_dates_by_month(self, user_id: str, year: int, month: int) -> dict:
        """获取特定用户在特定月份的所有线程，并按日期分组。
        
        Args:
            user_id: 用户ID
            year: 年份
            month: 月份 (1-12)
            
        Returns:
            包含日期到线程列表映射的字典: 
            { "YYYY-MM-DD": [thread1, thread2], ... }
            或包含错误的字典: { "error": "..." }
        """
        if not db_pool:
            print("错误: 数据库连接池未初始化。")
            return {"error": "数据库连接池未初始化"}
        
        # 使用 defaultdict 更方便地按日期分组
        threads_by_date = defaultdict(list)
        try:
            # 计算月份的起始和结束日期
            start_date = datetime.date(year, month, 1)
            if month == 12:
                next_month_start_date = datetime.date(year + 1, 1, 1)
            else:
                next_month_start_date = datetime.date(year, month + 1, 1)

            print(f"查询日期范围: >= {start_date} and < {next_month_start_date}")
            
            with db_pool.connection() as conn:
                # 使用 dict_row 获取字典形式的线程数据
                with conn.cursor(row_factory=dict_row) as cur:
                    # 查询指定用户和月份范围内的所有线程，包含最后消息预览
                    # 按 updated_at 排序，确保每个日期的线程列表内部有序
                    sql = """
                    SELECT 
                        t.thread_id as id, 
                        t.title, 
                        t.system_prompt, 
                        t.created_at, 
                        t.updated_at,
                        (SELECT content FROM app_messages m WHERE m.thread_id = t.thread_id ORDER BY m.timestamp DESC LIMIT 1) as "lastMessagePreview",
                        (SELECT timestamp FROM app_messages m WHERE m.thread_id = t.thread_id ORDER BY m.timestamp DESC LIMIT 1) as "lastMessageTime"
                    FROM app_threads t
                    WHERE t.user_id = %s 
                    AND t.created_at >= %s -- 使用 created_at 作为日期依据
                    AND t.created_at < %s
                    ORDER BY t.created_at DESC; -- 按创建时间排序，新的在前
                    """
                    cur.execute(sql, (user_id, start_date, next_month_start_date))
                    threads = cur.fetchall()
                    
                    # 按日期分组
                    for thread in threads:
                        # 确保 created_at 是 datetime 对象
                        created_at_dt = thread.get("created_at")
                        if isinstance(created_at_dt, datetime.datetime):
                            date_str = created_at_dt.strftime("%Y-%m-%d")
                            # 转换其他日期为 ISO 格式字符串
                            if thread.get("updated_at"):
                                thread["updated_at"] = thread["updated_at"].isoformat()
                            if thread.get("lastMessageTime"):
                                thread["lastMessageTime"] = thread["lastMessageTime"].isoformat()
                            thread["createdAt"] = created_at_dt.isoformat() # 保持前端使用的字段名
                            
                            threads_by_date[date_str].append(thread)
                        else:
                            print(f"警告: 线程 {thread.get('id')} 的 created_at 不是有效的 datetime 对象: {created_at_dt}")

                    print(f"从数据库获取到用户 {user_id} 在 {year}-{month} 月份的线程，按日期分组完成。共 {len(threads_by_date)} 天有记录。")
        except Exception as e:
            print(f"从数据库获取并分组线程失败: {e}")
            traceback.print_exc()
            return {"error": str(e)} # 返回带错误的字典
            
        return dict(threads_by_date) # 将 defaultdict 转为普通 dict 返回


# 创建全局实例
thread_manager = ThreadManager()


# 导出函数接口以保持向后兼容性
def save_thread(user_id: str, thread_id: str, title: str, system_prompt: str = None) -> bool:
    return thread_manager.save_thread(user_id, thread_id, title, system_prompt)


def get_user_threads(user_id: str, group_by_date: bool = False) -> list:
    return thread_manager.get_user_threads(user_id, group_by_date)


def remove_thread(user_id: str, thread_id: str) -> bool:
    return thread_manager.remove_thread(user_id, thread_id)


def save_message(user_id: str, thread_id: str, role: str, content: str, msg_type: str = 'text', metadata: dict = None) -> bool:
    return thread_manager.save_message(user_id, thread_id, role, content, msg_type, metadata)


def save_file_upload_message(user_id: str, thread_id: str, filename: str, file_type: str) -> bool:
    """模块级函数，调用 ThreadManager 实例的方法"""
    return thread_manager.save_file_upload_message(user_id, thread_id, filename, file_type)


def get_chat_history(user_id: str, thread_id: str) -> dict:
    return thread_manager.get_chat_history(user_id, thread_id)


def get_thread_messages(user_id: str, thread_id: str, system_prompt: str) -> List:
    return thread_manager.get_thread_messages(user_id, thread_id, system_prompt)


def get_threads_dates_by_month(user_id: str, year: int, month: int) -> dict:
    """获取特定用户在特定月份的所有线程，并按日期分组 (模块级接口)"""
    return thread_manager.get_threads_dates_by_month(user_id, year, month)

