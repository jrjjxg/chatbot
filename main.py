import os
from typing import Annotated, Union, Dict, Any, List
import re
import time
import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # 用于 OpenAI 兼容 API
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage  # 导入所需消息类型
from langchain_core.messages.utils import trim_messages  # 导入消息修剪工具

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

# 导入线程和消息管理函数
from thread_manager import (
    save_thread, get_user_threads, remove_thread,
    save_message, get_chat_history, get_thread_messages,
    threads_store, messages_store
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


# 简单的文件存储实现，不依赖于OpenAI Embeddings API
class SimpleMemoryStore:
    def __init__(self, file_path="memory_store.pkl"):
        self.file_path = file_path
        self.memories = {}
        self.load()

    def load(self):
        """从文件加载记忆"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    self.memories = pickle.load(f)
        except Exception as e:
            print(f"加载记忆文件出错: {e}")
            self.memories = {}

    def save(self):
        """保存记忆到文件"""
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.memories, f)
        except Exception as e:
            print(f"保存记忆文件出错: {e}")

    def add_memory(self, user_id, content, metadata=None):
        """添加新记忆"""
        metadata = metadata or {}
        timestamp = time.time()

        if user_id not in self.memories:
            self.memories[user_id] = []

        self.memories[user_id].append({
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata
        })

        self.save()

    def search_memories(self, user_id, query, limit=3):
        """简单搜索记忆 (基于关键词匹配)"""
        if user_id not in self.memories:
            return []

        # 将查询分解为关键词
        keywords = query.lower().split()
        results = []

        for memory in self.memories[user_id]:
            content = memory["content"].lower()
            # 计算匹配的关键词数量
            match_score = sum(1 for keyword in keywords if keyword in content)
            if match_score > 0:
                results.append((memory, match_score))

        # 按相关性排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回最相关的记忆
        return [item[0]["content"] for item in results[:limit]]


# 初始化全局记忆存储
memory_store = SimpleMemoryStore()


def save_to_long_term_memory(user_id: str, content: str, metadata: Dict = None):
    """保存重要信息到长期记忆"""
    global memory_store
    metadata = metadata or {}
    memory_store.add_memory(user_id, content, metadata)


def retrieve_from_long_term_memory(user_id: str, query: str, k: int = 3):
    """从长期记忆中检索相关信息"""
    global memory_store
    return memory_store.search_memories(user_id, query, limit=k)


def create_graph():
    """创建并编译 LangGraph 图实例"""
    # 初始化 StateGraph
    graph_builder = StateGraph(State)

    # 获取 LLM 实例
    llm = get_llm()

    # 定义聊天机器人节点的核心逻辑
    def chatbot_node(state: State):
        """
        使用当前对话历史调用 LLM。

        Args:
            state: 当前图状态，包含消息列表。

        Returns:
            包含 LLM 回复的更新后消息列表的字典。
        """
        try:
            # 获取配置的最大消息长度（默认4000字符，约相当于1000个token）
            max_length = int(os.environ.get("MAX_CHARS", "4000"))

            # 使用自定义函数修剪消息历史
            trimmed_messages = smart_trim_messages(
                state["messages"],
                max_length=max_length
            )

            # 添加额外的系统指令，确保模型不重复用户的输入
            # 查找现有的系统消息
            has_system_message = any(isinstance(msg, SystemMessage) for msg in trimmed_messages)

            # 如果没有系统消息，添加一个
            if not has_system_message:
                system_content = get_default_system_message().content
                system_content += " 请直接回答用户问题，不要在回复中重复用户的输入。"
                trimmed_messages.insert(0, SystemMessage(content=system_content))

            # 记录修剪前后的消息数量
            print(f"消息历史: 修剪前 {len(state['messages'])} 条，修剪后 {len(trimmed_messages)} 条")

            # 使用修剪后的消息调用 LLM
            return {"messages": [llm.invoke(trimmed_messages)]}
        except Exception as e:
            # 捕获并打印错误
            print(f"调用 LLM 时出错: {e}")
            # 返回错误消息
            return {"messages": [(
                "assistant",
                f"很抱歉，处理您的请求时出现了错误: {e}"
            )]}

    # 定义记忆检索节点
    def memory_retrieval_node(state: State, config: Dict = None):
        """检索长期记忆并添加到当前上下文"""
        if not config:
            config = {}

        # 检查是否启用长期记忆
        enable_memory = config.get("configurable", {}).get("enable_memory", True)
        if not enable_memory:
            print("长期记忆已禁用，跳过记忆检索")
            return {}

        # 从config中获取user_id
        user_id = config.get("configurable", {}).get("user_id", "default_user")

        # 获取当前用户问题
        messages = state["messages"]
        last_user_msg = None
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human" or (isinstance(msg, tuple) and msg[0] == "user"):
                if isinstance(msg, tuple):
                    last_user_msg = msg[1]
                else:
                    last_user_msg = msg.content
                break

        if not last_user_msg:
            return {}  # 没有用户消息，不做任何改变

        # 检索相关记忆
        memories = retrieve_from_long_term_memory(user_id, last_user_msg)

        if not memories:
            return {}  # 没有找到相关记忆

        # 添加记忆作为系统消息
        memories_text = "\n".join([f"- {mem}" for mem in memories])
        memory_message = SystemMessage(content=f"以下是与当前问题相关的历史信息:\n{memories_text}")

        return {"messages": [memory_message]}

    # 定义记忆写入节点
    def memory_writer_node(state: State, config: Dict = None):
        """将重要信息写入长期记忆"""
        if not config:
            config = {}

        # 检查是否启用长期记忆
        enable_memory = config.get("configurable", {}).get("enable_memory", True)
        if not enable_memory:
            print("长期记忆已禁用，跳过记忆写入")
            return {}

        user_id = config.get("configurable", {}).get("user_id", "default_user")
        messages = state["messages"]

        # 分析最近的对话，判断是否包含重要信息
        recent_messages = messages[-4:] if len(messages) >= 4 else messages

        # 提取用户问题和AI回答
        user_question = None
        ai_answer = None

        for msg in recent_messages:
            if getattr(msg, "type", "") == "human" or (isinstance(msg, tuple) and msg[0] == "user"):
                if isinstance(msg, tuple):
                    user_question = msg[1]
                else:
                    user_question = msg.content
            elif getattr(msg, "type", "") == "ai" or (isinstance(msg, tuple) and msg[0] == "assistant"):
                if isinstance(msg, tuple):
                    ai_answer = msg[1]
                else:
                    ai_answer = msg.content

        if user_question and ai_answer:
            # 使用启发式规则判断是否是重要信息
            # 这里可以实现更复杂的逻辑或使用LLM判断
            important_keywords = ["记住", "保存", "记录", "重要", "不要忘记"]
            if any(keyword in user_question.lower() for keyword in important_keywords):
                # 保存到长期记忆
                memory_content = f"问题: {user_question}\n回答: {ai_answer}"
                save_to_long_term_memory(user_id, memory_content)

        # 不修改状态，只写入记忆
        return {}

    # 添加 chatbot 节点到图中
    graph_builder.add_node("chatbot", chatbot_node)

    # 添加记忆检索节点
    graph_builder.add_node("memory_retrieval", memory_retrieval_node)

    # 添加记忆写入节点
    graph_builder.add_node("memory_writer", memory_writer_node)

    # 定义图的流程
    graph_builder.add_edge(START, "memory_retrieval")
    graph_builder.add_edge("memory_retrieval", "chatbot")
    graph_builder.add_edge("chatbot", "memory_writer")
    graph_builder.add_edge("memory_writer", END)

    # 编译图，使其可执行
    return graph_builder.compile()


# 获取默认系统提示
def get_default_system_message() -> SystemMessage:
    """返回默认的系统消息"""
    content = os.environ.get("DEFAULT_SYSTEM_PROMPT", "你是一个乐于助人的AI助手。")
    return SystemMessage(content=content)


# 处理聊天更新的辅助函数（用于命令行和 Web 界面）
def process_graph_stream(graph, user_input: str, history=None, config=None):
    """
    处理用户输入并返回流式 AI 响应生成器

    Args:
        graph: 已编译的 LangGraph 图实例
        user_input: 用户输入的文本
        history: 可选的历史消息列表（如果为 None，只包含用户当前输入）
        config: 可选的配置信息（如 thread_id）

    Returns:
        生成器，产生 AI 响应的片段
    """
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
        messages_input.append(("user", user_input))

    # 如果没有提供配置，创建一个默认配置
    if config is None:
        config = {"configurable": {"thread_id": "chat_session_1"}}

    # 使用 messages 模式流式传输 LLM 生成的内容
    try:
        # 使用 "messages" 模式，专门用于流式传输 LLM 令牌
        for event in graph.stream(
                {"messages": messages_input},
                config=config,
                stream_mode="messages"  # 使用 messages 模式
        ):
            # event 会包含 LLM 生成的令牌和元数据
            if isinstance(event, tuple) and len(event) >= 1:
                token = event[0]
                if hasattr(token, 'content') and token.content:
                    yield token.content
            elif hasattr(event, 'content') and event.content:
                yield event.content
    except Exception as e:
        yield f"流式传输过程中发生错误: {e}"


# 交互式命令行界面（保留原有功能）
def start_cli():
    """启动命令行交互界面"""
    # 创建 LangGraph 图
    graph = create_graph()

    print("聊天机器人已初始化。输入 'quit', 'exit', 或 'q' 结束。")
    # 为流配置基础信息
    config = {"configurable": {"thread_id": "chat_session_1"}}

    while True:
        try:
            user_input = input("用户: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见!")
                break
            if user_input.strip():  # 确保输入不为空
                # 使用流式处理函数并直接打印结果
                for response_chunk in process_graph_stream(graph, user_input, config=config):
                    print("助手:", response_chunk, end="", flush=True)
                print()  # 换行
            else:
                print("请输入消息。")
        except EOFError:  # 处理 Ctrl+D
            print("\n再见!")
            break
        except KeyboardInterrupt:  # 处理 Ctrl+C
            print("\n收到中断请求，正在退出...")
            break
        except Exception as e:  # 捕获其他错误
            print(f"发生意外错误: {e}")
            break


# 如果直接运行此脚本，启动命令行界面
if __name__ == "__main__":
    start_cli()
