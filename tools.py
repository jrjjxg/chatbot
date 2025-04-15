# tools.py
from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# 发送方邮箱配置 (从环境变量获取)
SENDER_EMAIL = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # QQ邮箱授权码
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

@tool
def send_email(recipient_email: str, subject: str, content: str):
    """向指定邮箱发送邮件通知。
    
    Args:
        recipient_email: 接收者的邮箱地址。
        subject: 邮件主题。
        content: 邮件正文内容。
    """
    if not all([SENDER_EMAIL, EMAIL_PASSWORD, recipient_email]):
        error_msg = "邮件发送失败：发件人邮箱、授权码或收件人邮箱未配置。"
        print(error_msg)
        return error_msg
        
    print(f"准备发送邮件至 {recipient_email} 主题: {subject}") # 调试信息
    
    try:
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = recipient_email
        message['Subject'] = Header(subject, 'utf-8')
        message.attach(MIMEText(content, 'plain', 'utf-8'))
        
        # 添加调试信息
        print(f"邮件头信息设置完成:")
        print(f"  From: {SENDER_EMAIL}")
        print(f"  To: {recipient_email}")
        print(f"  Subject: {subject}")
        print(f"  使用SMTP服务器: {SMTP_SERVER}:{SMTP_PORT}")
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        print("已连接到SMTP服务器")
        server.ehlo() # Or server.helo() if needed
        print("EHLO完成")
        server.starttls() # 启用TLS加密
        print("STARTTLS完成")
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        print(f"登录成功: {SENDER_EMAIL}")
        server.sendmail(SENDER_EMAIL, [recipient_email], message.as_string()) # 收件人是列表
        print("邮件已发送")
        server.quit()
        print("SMTP连接已关闭")
        
        success_msg = f"邮件已成功发送至 {recipient_email}"
        print(success_msg)
        return success_msg
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"发送邮件失败：SMTP认证错误（请检查邮箱和授权码） - {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"发送邮件失败: {e}"
        print(error_msg)
        return error_msg

# 优化的记忆管理工具
def create_optimize_manage_memory_tool(store):
    """创建优化的记忆管理工具"""
    
    @tool
    def manage_memory(information: str):
        """将重要信息存储到长期记忆中。
        
        使用此工具记住重要的用户信息，如姓名、偏好、关键事实等。
        
        Args:
            information: 要存储的信息内容（应为简明扼要的陈述句）
        """
        try:
            # 获取用户ID (从调用上下文中获取)
            from inspect import currentframe
            frame = currentframe()
            while frame:
                if 'config' in frame.f_locals:
                    config = frame.f_locals['config']
                    if isinstance(config, dict) and 'configurable' in config:
                        user_id = config.get('configurable', {}).get('user_id', 'default_user')
                        break
                frame = frame.f_back
            else:
                user_id = 'default_user'
                print(f"警告: 无法获取用户ID，使用默认值: {user_id}")

            # 创建记忆文档
            now = datetime.datetime.now().isoformat()
            doc = Document(
                page_content=information,
                metadata={
                    "timestamp": now, 
                    "type": "memory",
                    "user_id": user_id,  # 显式添加用户ID
                    "source": f"user_{user_id}"  # 添加更多标识
                }
            )
            
            # 存储到记忆库 - 不使用namespace
            print(f"存储记忆，用户ID: {user_id}, 内容: {information[:50]}...")
            try:
                # 尝试直接添加，不使用namespace
                store.add([doc])
                print(f"记忆已存储: {information[:50]}...")
            except Exception as e:
                print(f"存储记忆出错: {e}")
                import traceback
                traceback.print_exc()
            
            return f"已将信息「{information}」保存到长期记忆中。"
        except Exception as e:
            error_msg = f"保存记忆失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    return manage_memory

def create_optimize_search_memory_tool(store):
    """创建优化的记忆搜索工具"""
    
    @tool
    def search_memory(query: str):
        """搜索之前存储的记忆信息。
        
        使用此工具查找之前保存的有关用户的信息，如偏好、要求等。
        
        Args:
            query: 搜索查询，描述你想查找的信息
        """
        try:
            # 获取用户ID (从调用上下文中获取)
            from inspect import currentframe
            frame = currentframe()
            while frame:
                if 'config' in frame.f_locals:
                    config = frame.f_locals['config']
                    if isinstance(config, dict) and 'configurable' in config:
                        user_id = config.get('configurable', {}).get('user_id', 'default_user')
                        break
                frame = frame.f_back
            else:
                user_id = 'default_user'
                print(f"警告: 无法获取用户ID，使用默认值: {user_id}")

            # 搜索记忆 - 不使用namespace参数
            print(f"搜索记忆，用户ID: {user_id}, 查询: {query}")
            
            # 不使用namespace参数进行搜索
            results = []
            try:
                results = store.search(query, limit=10)  # 增大结果数量
            except Exception as e:
                print(f"记忆搜索出错: {e}")
                import traceback
                traceback.print_exc()
            
            if not results:
                return "没有找到相关记忆。"
            
            # 过滤与当前用户相关的记忆
            user_memories = []
            for mem in results:
                # 检查metadata中的user_id字段
                if hasattr(mem, 'metadata') and mem.metadata.get("user_id") == user_id:
                    user_memories.append(mem)
                # 如果metadata包含用户ID的字符串也算
                elif hasattr(mem, 'metadata') and str(user_id) in str(mem.metadata):
                    user_memories.append(mem)
                # 检查内容中是否包含用户ID (不太精确但可作为后备)
                elif hasattr(mem, 'page_content') and user_id in mem.page_content:
                    user_memories.append(mem)
            
            # 使用过滤后的记忆
            memories = [doc.page_content for doc in user_memories if hasattr(doc, 'page_content')]
            if not memories:
                return "没有找到与您相关的记忆。"
                
            result_text = "\n".join([f"- {memory}" for memory in memories])
            return f"找到以下相关记忆：\n{result_text}"
            
        except Exception as e:
            error_msg = f"搜索记忆失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    return search_memory

# DuckDuckGo Web Search Tool
from langchain_community.tools import DuckDuckGoSearchRun

dd_search = DuckDuckGoSearchRun()
dd_search.name = "web_search"
dd_search.description = "Use this tool to search the internet for current events, real-time information, or topics outside your internal knowledge base. Input should be a search query."

# 导出所有可用的工具 (方便在main.py中导入)
all_tools = [send_email]
# 注意：记忆管理和搜索工具需要 store 实例才能创建，
# 所以它们不能直接在这里导出，需要在 main.py 中创建 store 后再创建。

