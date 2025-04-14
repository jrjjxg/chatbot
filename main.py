import os
from typing import Annotated, Union, Dict, Any, List, Optional, Sequence
import re
import time
import json
import pickle
import uuid
from pathlib import Path
import sqlite3
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # 用于 OpenAI 兼容 API
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage  # 导入所需消息类型
from langchain_core.messages.utils import trim_messages  # 导入消息修剪工具
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver  # 导入 SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode, tools_condition

# 导入线程和消息管理函数
from thread_manager import (
    save_thread, get_user_threads, remove_thread,
    save_message, get_chat_history, get_thread_messages,
    thread_manager  # 使用 thread_manager 实例
)

# 从 .env 文件加载环境变量
load_dotenv()

# 定义图的状态结构
def message_list_manager(existing: list, updates: Union[list, Dict]) -> List:
    """自定义消息列表管理器"""
    if isinstance(updates, list):
        # 普通消息追加
        return existing + updates
    elif isinstance(updates, dict) and updates.get("type") == "keep":
        # 保留指定范围的消息
        start = updates.get("from", 0)
        end = updates.get("to", None)
        return existing[start:end]
    elif isinstance(updates, dict) and updates.get("type") == "summary":
        # 将旧消息替换为摘要
        summary_message = updates.get("summary")
        # 保留系统消息和最近的N条消息
        recent_count = updates.get("keep_recent", 4)
        system_messages = [msg for msg in existing if isinstance(msg, SystemMessage)]
        non_system = [msg for msg in existing if not isinstance(msg, SystemMessage)]
        recent_messages = non_system[-recent_count:] if len(non_system) > recent_count else non_system
        return system_messages + [summary_message] + recent_messages
    elif isinstance(updates, dict) and updates.get("type") == "remove":
        # 删除特定ID的消息
        ids_to_remove = updates.get("ids", [])
        if not ids_to_remove:
            return existing
        return [msg for msg in existing if getattr(msg, "id", None) not in ids_to_remove]
    else:
        # 默认行为
        return existing + (updates if isinstance(updates, list) else [updates])


class State(TypedDict):
    messages: Annotated[list, message_list_manager]
    retrieved_memories: Annotated[list[str], lambda x, y: x + y] = []


def get_llm():
    """获取配置好的 LLM 实例"""
    return ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),  # 允许通过环境变量指定模型
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),  # 允许通过环境变量配置温度
        streaming=True  # 启用流式输出
    )


# 简单的消息长度估算函数（使用字符数，而不是依赖特定模型的token计数）
def estimate_message_length(message: BaseMessage) -> int:
    """估算消息的长度（字符数）"""
    if hasattr(message, 'content'):
        if isinstance(message.content, str):
            return len(message.content)
        # 处理复杂内容类型
        return 100  # 默认值
    elif isinstance(message, tuple) and len(message) == 2:
        if isinstance(message[1], str):
            return len(message[1])
    return 100  # 默认值


def smart_trim_messages(messages: List[BaseMessage], max_length: int = 4000) -> List[BaseMessage]:
    """智能修剪消息历史

    - 始终保留系统消息
    - 保留最近的用户问题和AI回答
    - 对较早的对话进行摘要或选择性删除
    - 保留重要的上下文信息
    """
    # 实现智能修剪逻辑
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    # 如果消息数量少，直接返回
    if len(non_system_messages) <= 4:  # 保留至少2轮对话
        return system_messages + non_system_messages

    # 总是保留最新的2轮对话（4条消息）
    recent_messages = non_system_messages[-4:]
    older_messages = non_system_messages[:-4]

    # 创建一个RemoveMessage列表，用于标记要删除的消息
    # 这里可以实现更复杂的选择逻辑
    to_remove = []
    current_length = sum(estimate_message_length(msg) for msg in system_messages + recent_messages)

    # 从较早的消息中，优先保留重要的信息
    for msg in reversed(older_messages):
        msg_length = estimate_message_length(msg)
        if current_length + msg_length <= max_length:
            current_length += msg_length
        else:
            to_remove.append(msg)

    # 过滤掉要删除的消息
    preserved_older = [msg for msg in older_messages if msg not in to_remove]

    return system_messages + preserved_older + recent_messages


def format_messages_for_summary(messages: List) -> str:
    """格式化消息用于摘要生成"""
    formatted_text = ""
    for i, msg in enumerate(messages):
        role = ""
        content = ""

        if isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
        elif hasattr(msg, "type"):
            role = msg.type
            content = msg.content

        if role and content:
            formatted_text += f"{role.capitalize()}: {content}\n\n"

    return formatted_text


def summarize_messages(messages: List[BaseMessage], llm) -> SystemMessage:
    """将历史消息摘要化为一条系统消息"""
    if len(messages) <= 3:  # 消息太少不需要摘要
        return None

    # 筛选出用户和AI的对话，忽略系统消息
    dialog_messages = [msg for msg in messages
                       if not isinstance(msg, SystemMessage)]

    # 构建摘要提示
    prompt = f"""
    请总结以下对话的要点，简明扼要地提取关键信息：

    {format_messages_for_summary(dialog_messages)}

    总结:
    """

    # 使用LLM生成摘要
    summary = llm.invoke(prompt)

    # 创建系统消息
    return SystemMessage(content=f"之前对话的摘要: {summary}")


# 1. 定义图的状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    memory: Optional[List[str]]
    user_id: str


# 2. 辅助函数：提取关键信息
def extract_important_info(messages: Sequence[BaseMessage]) -> List[str]:
    """
    从对话消息中提取需要长期记住的关键信息。
    返回一个字符串列表，每个字符串是一条要记住的信息。
    """
    info_to_remember = []
    # 只关注最后一条用户消息
    if messages and isinstance(messages[-1], HumanMessage):
        last_user_message_content = messages[-1].content.lower()
        
        # 提取姓名 (非常基础的示例)
        name_match = re.search(r"(?:我叫|我(?:的)?名字是)\s?([^\s,，。!！?？]+)", last_user_message_content)
        if name_match and len(name_match.group(1)) < 10: # 简单长度限制
             info_to_remember.append(f"用户提到其名字可能是: {name_match.group(1)}")

        # 提取生日 (非常基础的示例)
        # 匹配 YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, YYYY年MM月DD日 等
        birthday_match = re.search(r"生日.*?(\d{4}[-/年.]?\d{1,2}[-/月.]?\d{1,2})", last_user_message_content)
        if birthday_match:
            info_to_remember.append(f"用户提到生日日期: {birthday_match.group(1)}")
            
        # 你可以添加更多规则，例如提取偏好、地址、重要事件等
        
    print(f"[Memory Extraction] 提取到的待记忆信息：{info_to_remember}")
    return info_to_remember


# 4. 创建图
def create_graph():
    """创建简化的LangGraph图"""
    builder = StateGraph(AgentState)

    # 定义一个agent节点，该节点将处理所有消息
    def agent(state: AgentState):
        """LLM Agent节点"""
        llm = get_llm()
        messages_from_state = state["messages"]

        # --- 检查和修复消息类型 ---
        valid_messages = []
        print(f"[Agent Node] Raw messages from state: {messages_from_state}") # 调试输出
        for msg in messages_from_state:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                valid_messages.append(msg)
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                 # 尝试从类似消息的对象恢复 (可能是反序列化不完全的对象)
                 print(f"[Agent Node] Attempting to reconstruct message: {msg}") # 调试输出
                 try:
                     if msg.type == 'human':
                         valid_messages.append(HumanMessage(content=str(msg.content)))
                     elif msg.type == 'ai':
                         valid_messages.append(AIMessage(content=str(msg.content)))
                     elif msg.type == 'system':
                          valid_messages.append(SystemMessage(content=str(msg.content)))
                     else:
                          print(f"[Agent Node] Skipping message with unknown type attribute: {msg.type}")
                 except Exception as recon_err:
                     print(f"[Agent Node] Error reconstructing message: {recon_err}, skipping message: {msg}")

            elif isinstance(msg, dict) and 'type' in msg and 'content' in msg:
                 # 尝试从字典恢复
                 print(f"[Agent Node] Attempting to reconstruct message from dict: {msg}") # 调试输出
                 try:
                     msg_type = msg.get("type")
                     content = msg.get("content", "")
                     if msg_type == "human":
                         valid_messages.append(HumanMessage(content=str(content)))
                     elif msg_type == "ai":
                          valid_messages.append(AIMessage(content=str(content)))
                     elif msg_type == "system":
                          valid_messages.append(SystemMessage(content=str(content)))
                     else:
                          print(f"[Agent Node] Skipping dict message with unknown type: {msg_type}")
                 except Exception as recon_err:
                     print(f"[Agent Node] Error reconstructing message from dict: {recon_err}, skipping message: {msg}")
            else:
                # 记录并跳过无法识别的消息类型
                print(f"[Agent Node] Warning: Skipping unknown message type {type(msg)} in state: {msg}")
        
        print(f"[Agent Node] Validated messages passed to LLM: {valid_messages}") # 调试输出
        if not valid_messages:
             # 如果过滤后没有有效消息，可能需要返回错误或默认响应
             print("[Agent Node] Error: No valid messages found after validation.")
             return {"messages": [AIMessage(content="Error processing message history.")]}
        # --- 结束检查和修复 ---

        try:
            # 使用验证和修复后的消息列表
            response = llm.invoke(valid_messages)
            print(f"[Agent Node] LLM Response: {response}")
            # 确保返回的是 AIMessage
            if not isinstance(response, AIMessage):
                 print(f"[Agent Node] Warning: LLM response is not AIMessage: {type(response)}. Attempting to wrap.")
                 # 尝试包装，如果失败则返回错误消息
                 try:
                     response_content = getattr(response, 'content', str(response))
                     response = AIMessage(content=str(response_content))
                 except Exception as wrap_err:
                     print(f"[Agent Node] Error wrapping LLM response: {wrap_err}")
                     response = AIMessage(content="Error: Could not process LLM response.")

        except Exception as llm_error:
            print(f"[Agent Node] Error during LLM invocation: {llm_error}")
            import traceback
            traceback.print_exc()
            response = AIMessage(content=f"Sorry, an error occurred while processing your request.")

        # 更新状态并返回，确保是列表形式
        return {"messages": [response]}
    
    # 添加agent节点
    builder.add_node("agent", agent)
    
    # 定义简化的图流程 - 删除记忆相关节点
    builder.add_edge(START, "agent")  # 直接开始agent节点
    builder.add_edge("agent", END)    # agent执行完直接结束

    # 编译图 (Checkpointer在app.py中编译时传入)
    # 这里只返回builder，编译在app.py中进行
    print("Graph definition complete.")
    return builder

# 获取默认系统提示
def get_default_system_message() -> SystemMessage:
    """返回默认的系统消息"""
    content = os.environ.get("DEFAULT_SYSTEM_PROMPT", "你是一个乐于助人的AI助手。请利用提供的背景记忆（如果有的话）来更好地与用户交流。")
    return SystemMessage(content=content)


def process_graph_stream(graph, user_input: str, history=None, config=None):
    """处理用户输入并返回生成的响应流"""
    # 准备消息输入
    messages_input = []

    # 如果提供了历史，使用它，否则只使用当前输入
    if history:
        messages_input = history

    # 添加当前用户消息
    if isinstance(user_input, BaseMessage):
        messages_input.append(user_input)
    else:
        # 假设这是一个字符串文本
        messages_input.append(HumanMessage(content=user_input))

    # 如果没有提供配置，创建一个默认配置
    if config is None:
        config = {"configurable": {"thread_id": "chat_session_1"}}
    
    # 记录当前请求的线程ID和会话状态
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"[LANGGRAPH] 正在处理 thread_id={thread_id} 的请求，消息历史长度: {len(messages_input)}")
    print(f"[LANGGRAPH] 当前请求的配置: {config}")

    # 使用 "messages" 模式流式传输 LLM 生成的内容
    try:
        # 使用 RunnableConfig 确保配置正确传递
        langgraph_config = RunnableConfig(configurable=config.get("configurable", {}))
        
        # 使用 "messages" 模式，专门用于流式传输 LLM 令牌
        for event in graph.stream(
                {"messages": messages_input},
                config=langgraph_config,
                stream_mode="messages"  # 使用 messages 模式
        ):
            # 更详细的调试信息
            print(f"[LANGGRAPH STREAM] 收到事件: {type(event)}")
            
            # event 会包含 LLM 生成的令牌和元数据
            if isinstance(event, tuple) and len(event) >= 1:
                token = event[0]
                # 尝试提取内容部分
                if hasattr(token, 'content') and token.content:
                    print(f"[LANGGRAPH STREAM] 发送token: '{token.content}'")
                    yield token.content
                elif isinstance(token, str):
                    print(f"[LANGGRAPH STREAM] 发送字符串token: '{token}'")
                    yield token
            # 直接处理消息对象
            elif hasattr(event, 'content') and event.content:
                print(f"[LANGGRAPH STREAM] 发送内容: '{event.content}'")
                yield event.content
            elif isinstance(event, str):
                print(f"[LANGGRAPH STREAM] 发送字符串: '{event}'")
                yield event
        
        print(f"[LANGGRAPH] thread_id={thread_id} 的请求处理完成")
    except Exception as e:
        print(f"[LANGGRAPH] thread_id={thread_id} 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"流式传输过程中发生错误: {str(e)}"


# 交互式命令行界面（保留原有功能）
def start_cli():
    """启动命令行界面"""
    graph_builder = create_graph()
    
    # 在应用启动时初始化 checkpointer，而不是在每次请求中
    sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(sqlite_conn)  # 直接使用连接创建 SqliteSaver

    # 编译图，并添加持久化
    runnable = graph_builder.compile(checkpointer=checkpointer)

# 如果直接运行此脚本，启动命令行界面
if __name__ == "__main__":
    start_cli()
