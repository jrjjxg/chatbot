import os
import json
import time
import threading
import schedule
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Optional, TypedDict, Sequence

# LangChain 和 LangGraph 导入
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, create_react_agent  # 添加 create_react_agent 导入
from langgraph.types import Command, interrupt

# 导入本地模块
from emotion_monitor import EmotionAnalyzer # Keep this if needed elsewhere, maybe remove if only for the old agent class
from thread_manager import thread_manager
from tools import send_email

# 加载环境变量
load_dotenv()

# 配置
SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL") 
MOOD_ANALYTICS_API_ENDPOINT = os.getenv("MOOD_ANALYTICS_API_ENDPOINT", "/api/moods/analytics") 
EMERGENCY_CONTACT_API_ENDPOINT = os.getenv("EMERGENCY_CONTACT_API_ENDPOINT", "/api/alert/emergency-contacts") 
EMOTION_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", 0.3))
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", 3600))
NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", 86400))
ANALYTICS_HOURS_RANGE = int(os.getenv("ANALYTICS_HOURS_RANGE", 2400))
THREADS_FILE_PATH = os.getenv("THREADS_FILE_PATH", os.path.join(os.path.dirname(__file__), 'data', 'threads.json'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 记录上次为某个用户通知的时间 {userId: {contact_email: timestamp}}
last_notified_contact = {}

# 定义辅助函数
def get_api_headers(user_id=None):
    """构造API请求头"""
    import requests
    headers = {"Content-Type": "application/json"}
    if user_id:
        headers["userId"] = str(user_id)
    return headers

def get_users_to_monitor():
    """获取需要监控的用户ID列表"""
    if not os.path.exists(THREADS_FILE_PATH):
        print(f"警告: 线程文件未找到: {THREADS_FILE_PATH}")
        return []
        
    try:
        with open(THREADS_FILE_PATH, 'r', encoding='utf-8') as f:
            all_threads_data = json.load(f)
            user_ids = list(all_threads_data.keys())
            print(f"从线程文件中找到 {len(user_ids)} 个用户ID进行监控。")
            return user_ids
    except Exception as e:
        print(f"错误: 读取线程文件时出错: {e}")
        return []

def get_user_id_by_username(username):
    """通过用户名获取用户ID"""
    import requests
    api_url = f"{SPRING_BOOT_BASE_URL}/api/user/get-user-id-by-username"
    params = {'username': username}
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response_data = response.json()
        
        if 'data' in response_data:
            user_id = response_data['data']
            print(f"用户 {username} 的ID是: {user_id}")
            return user_id
    except Exception as e:
        print(f"获取用户ID失败: {e}")
    return None

@tool
def get_user_mood_analytics(user_id_or_name: str) -> float:
    """获取用户情绪分析数据
    
    Args:
        user_id_or_name: 用户ID或用户名
        
    Returns:
        用户的情绪分数，范围从1到5，数值越低表示情绪越低落
    """
    import requests
    # 尝试获取用户ID
    actual_user_id = get_user_id_by_username(user_id_or_name) or user_id_or_name
        
    api_url = f"{SPRING_BOOT_BASE_URL}{MOOD_ANALYTICS_API_ENDPOINT}"
    headers = get_api_headers(user_id=actual_user_id)
    
    # 计算日期范围
    end_date = datetime.now().date()
    start_date = end_date - timedelta(hours=ANALYTICS_HOURS_RANGE) 
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        print(f"请求用户 {actual_user_id} 的情绪分析: {api_url} params: {params}")
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        analytics_data = response.json()
        
        if 'data' in analytics_data and isinstance(analytics_data['data'], dict):
            data = analytics_data['data']
            mood_score = data.get('averageIntensity')
            print(f"用户 {actual_user_id} 的情绪分析结果: {mood_score}")
            return float(mood_score) if mood_score is not None else None
    except Exception as e:
        print(f"获取情绪分析失败: {e}")
    return None

@tool
def get_emergency_contacts(user_id: str) -> list:
    """获取用户的紧急联系人列表
    
    Args:
        user_id: 用户ID或用户名
        
    Returns:
        紧急联系人列表，每个联系人包含email、name和relationship等信息
    """
    import requests
    
    # 尝试获取用户ID
    actual_user_id = get_user_id_by_username(user_id) or user_id
    
    api_url = f"{SPRING_BOOT_BASE_URL}{EMERGENCY_CONTACT_API_ENDPOINT}"
    headers = get_api_headers(user_id=actual_user_id)
    
    print(f"\n[DEBUG] 开始获取用户 {user_id} (ID: {actual_user_id}) 的紧急联系人")
    print(f"[DEBUG] API URL: {api_url}")
    print(f"[DEBUG] 请求头: {headers}")
    
    try:
        print("[DEBUG] 发送 GET 请求...")
        response = requests.get(api_url, headers=headers, timeout=10)
        print(f"[DEBUG] 响应状态码: {response.status_code}")
        print(f"[DEBUG] 响应头: {response.headers}")
        
        response_data = response.json()
        print(f"[DEBUG] 响应数据: {response_data}")
        
        if 'data' in response_data and isinstance(response_data['data'], list):
            contacts = response_data['data']
            print(f"[DEBUG] 原始联系人列表: {contacts}")
            
            valid_contacts = [c for c in contacts if c.get("email")]
            print(f"[DEBUG] 有效联系人数量: {len(valid_contacts)}")
            print(f"[DEBUG] 有效联系人列表: {valid_contacts}")
            
            return valid_contacts
        else:
            print(f"[DEBUG] 响应数据格式不正确: 缺少 'data' 字段或 'data' 不是列表")
            print(f"[DEBUG] 响应数据结构: {type(response_data)}")
            if 'data' in response_data:
                print(f"[DEBUG] data 字段类型: {type(response_data['data'])}")
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] 请求异常: {str(e)}")
        print(f"[DEBUG] 异常类型: {type(e)}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON 解析错误: {str(e)}")
        print(f"[DEBUG] 响应内容: {response.text}")
    except Exception as e:
        print(f"[DEBUG] 其他错误: {str(e)}")
        print(f"[DEBUG] 错误类型: {type(e)}")
    
    print("[DEBUG] 返回空联系人列表")
    return []

# 定义Agent状态类型
class EmotionMonitorState(TypedDict):
    messages: Sequence[BaseMessage]
    user_id: str
    username: str
    emotion_score: Optional[float]
    emergency_contacts: List[dict]

# 创建LLM实例
def create_llm():
    """创建LLM实例"""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    
    if not deepseek_api_key or not openai_api_base:
        raise ValueError("缺少DEEPSEEK_API_KEY或OPENAI_API_BASE")
        
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=deepseek_api_key,
        base_url=openai_api_base
    )

llm = create_llm()
tools = [get_user_mood_analytics, get_emergency_contacts, send_email]
tool_node = ToolNode(tools)  # 使用 ToolNode 替代 ToolExecutor

# 构建Agent系统提示
AGENT_SYSTEM_PROMPT_CONTENT = f"""你是一个情绪监控助手。你的任务是检查用户的情绪状态并根据可用数据采取行动。
如果用户的情绪分数（通过get_user_mood_analytics工具获取的averageIntensity）低于阈值 {EMOTION_THRESHOLD}，你必须：
1. 调用get_emergency_contacts工具获取用户的紧急联系人。
2. 对每个有效的紧急联系人调用send_email工具发送邮件。邮件内容要表达对用户情绪状态的关心，并提及情绪分数。

重要提示：你必须用中文写邮件，不要使用英文。

根据情绪分值的不同，调整邮件内容的紧急程度：
- 如果情绪分值在3-4之间：表达温和的关心，建议联系用户了解情况
- 如果情绪分值在2-3之间：表达较强的担忧，建议尽快联系用户并提供支持
- 如果情绪分值低于2：表达高度担忧，建议立即联系用户并考虑寻求专业帮助

邮件风格要求：
- 使用自然、亲切的语气，像朋友之间的关心
- 避免过于正式的格式，如"此致敬礼"等
- 可以适当使用表情符号增加亲和力
- 内容简洁明了，重点突出

使用提供的工具。只返回工具调用或成功发送邮件后的最终确认消息。在调用工具前不要添加对话填充内容。优先调用get_user_mood_analytics工具。

用户信息将在state对象中提供。你将可以访问user_id和username用于工具调用。
"""

# 创建情绪监控Agent
def create_emotion_monitor_agent():
    """创建情绪监控Agent"""
    # 获取LLM
    llm = create_llm()

    # 定义可用工具
    tools = [get_user_mood_analytics, get_emergency_contacts, send_email]

    # 定义包含系统提示的 Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT_CONTENT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 创建基于ReAct的agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt
    )
    
    # 创建工具节点
    tool_executor = ToolNode(tools)
    
    # 创建状态图
    workflow = StateGraph(EmotionMonitorState)
    
    # 添加节点
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_executor)
    
    # 添加边
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    
    # 设置入口和出口
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")
    
    # 编译工作流
    app = workflow.compile()
    print("LangGraph 图创建成功")
    
    return app

# 修改 tool_node 函数
def tool_node(state: EmotionMonitorState):
    """Executes tools based on the last AI message."""
    print("--- Calling Tool Node ---")
    messages = state['messages']
    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        print("No tool calls found in the last message.")
        return {} # No changes to state if no tools called

    tool_calls = last_message.tool_calls
    print(f"Executing tool calls: {tool_calls}")

    try:
        # 使用工具执行器而不是递归调用
        tool_responses = []
        for call in tool_calls:
            tool_name = call['name']
            tool_args = call.get('arguments', {})
            
            # 查找对应的工具
            for tool in tools:
                if tool.__name__ == tool_name:
                    # 如果参数是字符串，尝试解析为JSON
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            # 如果解析失败，使用原始字符串
                            pass
                    
                    # 调用工具
                    result = tool(**tool_args)
                    tool_responses.append(result)
                    break
        
        print(f"Tool Responses: {tool_responses}")

        # Format responses as ToolMessage objects
        tool_messages = []
        for call, response in zip(tool_calls, tool_responses):
            # 确保响应是字符串格式
            if isinstance(response, (dict, list)):
                response = json.dumps(response, ensure_ascii=False)
            elif not isinstance(response, str):
                response = str(response)
                
            tool_messages.append(
                ToolMessage(content=response, name=call['name'], tool_call_id=call['id'])
            )
        return {"messages": tool_messages}
    except Exception as e:
        print(f"Error executing tools: {e}")
        # Return error messages as ToolMessages
        error_messages = [
             ToolMessage(content=f"Error executing tool {call['name']}: {e}", name=call['name'], tool_call_id=call['id'])
             for call in tool_calls
        ]
        return {"messages": error_messages}

# 运行单个用户的情绪监控
def monitor_user_emotion(user_id, username=None):
    """运行单个用户的情绪监控

    Args:
        user_id: 用户ID
        username: 用户名（可选）
    """
    global last_notified_contact

    print(f"开始监控用户 {username or user_id} 的情绪状态...")

    # 创建Agent
    app = create_emotion_monitor_agent()

    # 准备初始用户消息
    user_content = f"""请检查用户 {username or user_id} 的情绪状态，ID是 {user_id}。
如果情绪分数低于 {EMOTION_THRESHOLD}，请获取用户的紧急联系人信息并发送提醒邮件。
请使用提供的工具来完成这些任务。"""
    
    # 使用正确的消息格式
    initial_message = HumanMessage(content=user_content)
    
    # 初始化状态
    state = {
        "messages": [initial_message],
        "user_id": user_id,
        "username": username or user_id,
        "emotion_score": None,
        "emergency_contacts": []
    }

    try:
        # 运行工作流
        result = app.invoke(state)
        print(f"工作流执行结果: {result}")

        # 处理结果
        if result and "messages" in result:
            final_message = result["messages"][-1]
            
            # 确保消息内容是字符串
            if isinstance(final_message, (dict, list)):
                message_content = json.dumps(final_message)
            elif isinstance(final_message, AIMessage):
                message_content = final_message.content
            else:
                message_content = str(final_message)
                
            print(f"工作流结果: {message_content}")

            # 检查是否发送了通知
            if isinstance(final_message, AIMessage):
                # 更新通知时间
                if user_id not in last_notified_contact:
                    last_notified_contact[user_id] = {}
                
                # 从最终消息中提取联系人信息
                for contact_email in [word for word in message_content.split() if "@" in word]:
                    last_notified_contact[user_id][contact_email] = time.time()

            return message_content
        else:
            print("工作流执行未返回有效结果")
            return "工作流执行未返回有效结果"
            
    except Exception as e:
        print(f"监控用户 {username or user_id} 时发生错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return f"错误: {str(e)}"

# 主监控函数
def run_emotion_monitoring():
    """运行情绪监控任务"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始执行基于Agent的情绪监控...")
    
    # 获取用户列表
    users_to_check = get_users_to_monitor()
    if not users_to_check:
        print("没有找到需要监控的用户，本次检查结束。")
        return
    
    # 检查冷却期
    current_time = time.time()
    users_to_monitor = []
    
    for user_id in users_to_check:
        # 检查上次通知时间
        if user_id in last_notified_contact:
            contacts = last_notified_contact[user_id]
            if contacts and all(current_time - notify_time < NOTIFICATION_COOLDOWN_SECONDS 
                              for notify_time in contacts.values()):
                print(f"用户 {user_id} 的所有联系人仍在冷却期内，跳过。")
                continue
        
        users_to_monitor.append(user_id)
    
    # 监控每个用户
    for user_id in users_to_monitor:
        monitor_user_emotion(user_id)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基于Agent的情绪监控完成")

# 运行监控调度
def run_monitoring_schedule():
    """运行监控调度"""
    print(f"基于Agent的情绪监控任务已设置，每 {MONITOR_INTERVAL_SECONDS} 秒执行一次")
    run_emotion_monitoring()  # 启动时立即执行一次
    schedule.every(MONITOR_INTERVAL_SECONDS).seconds.do(run_emotion_monitoring)
    while True:
        schedule.run_pending()
        time.sleep(1)

# 在后台线程启动监控
def start_agent_monitoring_in_background():
    """在后台线程启动基于Agent的监控"""
    try:
        # 测试创建一个LLM实例，确保API密钥有效
        create_llm()
        
        monitor_thread = threading.Thread(target=run_monitoring_schedule, daemon=True)
        monitor_thread.start()
        print("基于Agent的情绪监控后台线程已启动。")
    except Exception as e:
        print(f"警告: 无法启动基于Agent的情绪监控: {e}")
        print("请确保设置了正确的OPENAI_API_KEY环境变量或替代的API配置。")

# 测试函数
def test_emotion_monitor_agent(user_id):
    """测试情绪监控Agent
    
    Args:
        user_id: 要测试的用户ID
    """
    result = monitor_user_emotion(user_id)
    print(f"测试结果: {result}")
    return result

class ChatState(TypedDict):
    messages: List[dict]
    user_id: str
    thread_id: str
    user_confirmation: Optional[bool]

def create_chat_agent():
    """创建对话agent"""
    # 1. 创建状态图
    workflow = StateGraph(ChatState)
    
    # 2. 定义节点
    def process_message(state):
        """处理用户消息"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # 调用LLM生成回复
        response = llm.invoke(last_message)
        return {"messages": [response]}
    
    def check_confirmation(state):
        """检查是否需要用户确认"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if "需要确认" in last_message.content:
            return "ask_confirmation"
        return END
    
    def ask_confirmation(state):
        """请求用户确认"""
        question = "您是否确认执行此操作？"
        response = interrupt(question)
        return {"user_confirmation": response.lower() in ["是", "yes", "y"]}
    
    # 3. 添加节点
    workflow.add_node("process", process_message)
    workflow.add_node("ask_confirmation", ask_confirmation)
    
    # 4. 添加边
    workflow.add_edge(START, "process")
    workflow.add_conditional_edges("process", check_confirmation)
    workflow.add_edge("ask_confirmation", "process")
    
    return workflow.compile()

from emotion_monitor import EmotionAnalyzer  # 添加这行导入
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional, TypedDict

class EnhancedEmotionMonitorAgent:
    """增强版情绪监控agent"""
    def __init__(self):
        self.chat_agent = create_chat_agent()
        self.memory = MemorySaver()
        self.emotion_analyzer = EmotionAnalyzer()
        self.thread_manager = thread_manager  # 使用 thread_manager 实例
    
    async def process_message(self, user_id, thread_id, message):
        """处理用户消息"""
        # 1. 初始化状态
        state = {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "thread_id": thread_id
        }
        
        # 2. 分析情绪
        emotion_analysis = self.emotion_analyzer.analyze(message)
        
        # 3. 处理消息
        for event in self.chat_agent.stream(state, {"configurable": {"thread_id": thread_id}}):
            if "__interrupt__" in event:
                # 需要用户确认
                question = event["__interrupt__"][0].value
                # 等待用户输入
                user_response = await self.wait_for_user_input(question)
                # 继续处理
                for event in self.chat_agent.stream(
                    Command(resume=user_response),
                    {"configurable": {"thread_id": thread_id}}
                ):
                    yield event
            else:
                yield event
        
        # 4. 检查是否需要预警
        if self._needs_alert(emotion_analysis):
            self._send_alert(user_id, emotion_analysis)
    
    async def wait_for_user_input(self, question):
        """等待用户输入"""
        # 实现等待用户输入的逻辑
        # 可以通过WebSocket或其他方式实现
        pass

if __name__ == "__main__":
    # 如果作为主程序运行，则执行测试
    test_user = "test_user_id"  # 替换为实际测试用户ID
    test_emotion_monitor_agent(test_user) 