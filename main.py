import os
from typing import Annotated, Union, Dict, Any, List, Optional, Sequence
import re
import time
import json
import pickle
import uuid
from pathlib import Path
import operator
import tiktoken
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # 用于 OpenAI 兼容 API
from langchain.docstore.document import Document # To represent documents


from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, filter_messages # 导入 trim_messages 和 filter_messages
from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from tools import dd_search, send_email


# 从 .env 文件加载环境变量
load_dotenv()

# --- RAG Configuration ---
VECTORSTORE_BASE_PATH = "./db/vectorstores" # Base path for ChromaDB collections
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # Or configure OpenAIEmbeddings()
# --- End RAG Configuration ---

# --- Tiktoken计数器 ---
_tokenizer = None

def get_tokenizer():
    """获取 tiktoken tokenizer 实例 (单例模式)"""
    global _tokenizer
    if _tokenizer is None:
        try:
            # 尝试获取 cl100k_base 编码器，这是 OpenAI GPT-3.5/4 常用的
            _tokenizer = tiktoken.get_encoding("cl100k_base")
            print("[Token Counter] Initialized tiktoken tokenizer (cl100k_base).")
        except Exception as e:
            print(f"[Token Counter] Warning: Failed to get cl100k_base tokenizer: {e}. Falling back to character count.")
            _tokenizer = "char_count" # 回退标记
    return _tokenizer

def count_message_tokens(messages: list[BaseMessage]) -> int:
    """使用 tiktoken 估算消息列表的总 token 数"""
    tokenizer = get_tokenizer()
    if tokenizer == "char_count" or not hasattr(tokenizer, "encode"):
        # 如果 tokenizer 初始化失败，回退到字符计数
        total_chars = sum(len(str(msg.content)) if hasattr(msg, 'content') else 0 for msg in messages)
        print(f"[Token Counter Fallback] Counting characters: {total_chars}")
        return total_chars // 3 # 非常粗略的估计：3个字符约等于1个token

    tokens_per_message = 3  # 每个消息都会添加 {role/name}\n{content}\n\n
    tokens_per_name = 1 # 如果有 name
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        # LangChain 消息对象的 content 属性通常是字符串
        content = str(message.content) if hasattr(message, 'content') else ""
        role = message.type if hasattr(message, 'type') else "unknown" # 获取角色

        num_tokens += len(tokenizer.encode(content))
        num_tokens += len(tokenizer.encode(role))

        # 检查是否有 name 属性 (例如 ToolMessage)
        if hasattr(message, 'name') and message.name:
             num_tokens += tokens_per_name
             num_tokens += len(tokenizer.encode(message.name))

    num_tokens += 3  # 每个回复都以 <|im_start|>assistant<|im_sep|> 开始
    print(f"[Token Counter] Estimated tokens for {len(messages)} messages: {num_tokens}")
    return num_tokens
# --- 结束 Tiktoken计数器 ---

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
        # 只包含 Human 和 AI 消息
        elif isinstance(msg, HumanMessage):
            role = "Human"
            content = msg.content
        elif isinstance(msg, AIMessage) and msg.content: # 确保AI消息有内容
             role = "AI"
             content = msg.content

        if role in ["Human", "AI"] and content:
            formatted_text += f"{role}: {content}\n\n"

    return formatted_text


def summarize_conversation(messages: List[BaseMessage], llm) -> Optional[SystemMessage]:
    """将历史消息摘要化为一条系统消息，优化提示词"""
    # 筛选出用户和AI的对话，忽略系统消息和工具消息
    dialog_messages = filter_messages(
        messages,
        include_types=[HumanMessage, AIMessage],
        exclude_types=[SystemMessage, ToolMessage]
    )

    # 如果有效对话消息少于指定轮数（例如2轮=4条），则不生成摘要
    if len(dialog_messages) <= 4:
        print("[Summarize] 对话消息不足 (<=4)，跳过摘要生成。")
        return None

    print(f"[Summarize] 准备为 {len(dialog_messages)} 条对话消息生成摘要...")
    
    # 获取最后一条用户消息，用于相关性判断
    last_user_message = None
    for msg in reversed(dialog_messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    
    # 构建摘要提示，加入与最新查询相关性的指导
    prompt = f"""
    请总结以下对话的核心内容和关键信息。重点关注:
    1. 用户提供的重要事实、意图和偏好
    2. AI提供的关键答案或解决方案
    3. 与用户最新问题相关的历史上下文
    
    请忽略日常问候和无关紧要的闲聊。

    用户最新问题: {last_user_message if last_user_message else "未提供"}
    
    对话历史:
    {format_messages_for_summary(dialog_messages)}

    简洁总结 (务必保留与最新问题相关的历史信息):
    """

    try:
        # 使用LLM生成摘要
        summary_content = llm.invoke(prompt).content # 获取 AIMessage 的 content
        print(f"[Summarize] 生成的摘要内容: {summary_content}")
        # 创建系统消息
        return SystemMessage(content=f"之前对话的摘要: {summary_content}")
    except Exception as e:
        print(f"[Summarize] 生成摘要时出错: {e}")
        return None


# 1. 定义图的状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str # The current user question being processed
    documents: Optional[List[Document]] # Retrieved documents for the current question
    # You might add other fields like 'context' if needed later
    # memory: Optional[List[str]] # Keep memory if used, otherwise remove
    user_id: str # Keep user_id if needed for other logic


# 2. 辅助函数：提取关键信息 (保留，与记忆相关)
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
    """创建带有工具调用能力的 LangGraph 图"""
    llm = get_llm() # 获取 LLM 实例
    llm_with_tools = llm.bind_tools([dd_search, send_email])

    # 定义 Agent 节点
    def agent(state: AgentState):
        """LLM Agent 节点，决定是回复还是调用工具"""
        messages_from_state = state["messages"]

        # --- 消息验证和修复逻辑 (保留) ---
        valid_messages = []
        print(f"[Agent Node] Raw messages from state (before validation): {len(messages_from_state)} messages")
        # 打印原始消息用于调试 (如果消息过多，可以只打印类型或部分内容)
        # for i, msg in enumerate(messages_from_state):
        #     print(f"[Agent Node] Raw msg {i}: type={type(msg)}, content={str(msg)[:100]}...")
        
        for msg in messages_from_state:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                valid_messages.append(msg)
            # ... (其他消息重建逻辑保持不变, 增加了 AIMessage 和 ToolMessage 的重建细节) ...
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                 # print(f"[Agent Node] Attempting to reconstruct message: {msg}") # 日志可能过于冗长
                 try:
                     if msg.type == 'human':
                         valid_messages.append(HumanMessage(content=str(msg.content)))
                     elif msg.type == 'ai':
                         tool_calls = getattr(msg, 'tool_calls', None)
                         kwargs = {}
                         if tool_calls: kwargs['tool_calls'] = tool_calls
                         valid_messages.append(AIMessage(content=str(msg.content), **kwargs))
                     elif msg.type == 'system':
                          valid_messages.append(SystemMessage(content=str(msg.content)))
                     elif msg.type == 'tool':
                         tool_call_id = getattr(msg, 'tool_call_id', None)
                         if tool_call_id:
                            valid_messages.append(ToolMessage(content=str(msg.content), tool_call_id=tool_call_id))
                         else:
                             print(f"[Agent Node] Warning: Skipping ToolMessage reconstruction due to missing tool_call_id: {msg}")
                     else:
                          print(f"[Agent Node] Warning: Skipping message with unknown type attribute: {msg.type}")
                 except Exception as recon_err:
                     print(f"[Agent Node] Error reconstructing message: {recon_err}, skipping message: {msg}")
            elif isinstance(msg, dict) and 'type' in msg and 'content' in msg:
                 # print(f"[Agent Node] Attempting to reconstruct message from dict: {msg}") # 日志可能过于冗长
                 try:
                     msg_type = msg.get("type")
                     content = msg.get("content", "")
                     if msg_type == "human":
                         valid_messages.append(HumanMessage(content=str(content)))
                     elif msg_type == "ai":
                          tool_calls = msg.get('additional_kwargs', {}).get('tool_calls')
                          kwargs = {}
                          if tool_calls: kwargs['additional_kwargs'] = {'tool_calls': tool_calls}
                          valid_messages.append(AIMessage(content=str(content), **kwargs))
                     elif msg_type == "system":
                          valid_messages.append(SystemMessage(content=str(content)))
                     elif msg_type == 'tool':
                         tool_call_id = msg.get('tool_call_id')
                         if tool_call_id:
                             valid_messages.append(ToolMessage(content=str(content), tool_call_id=tool_call_id))
                         else:
                             print(f"[Agent Node] Warning: Skipping dict ToolMessage reconstruction due to missing tool_call_id: {msg}")
                     else:
                          print(f"[Agent Node] Warning: Skipping dict message with unknown type: {msg_type}")
                 except Exception as recon_err:
                     print(f"[Agent Node] Error reconstructing message from dict: {recon_err}, skipping message: {msg}")
            else:
                print(f"[Agent Node] Warning: Skipping unknown message type {type(msg)} in state: {msg}")
        # --- 结束消息验证和修复 ---

        print(f"[Agent Node] Validated messages before trim/summary: {len(valid_messages)} messages")

        # --- 新增：系统消息去重逻辑 ---
        def deduplicate_system_messages(messages):
            """去除重复的系统消息，合并相似内容"""
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            
            if len(system_messages) <= 1:
                return messages  # 不需要去重
            
            # 检测基本系统提示
            base_prompts = []
            rag_prompts = []
            other_system_prompts = []
            
            for msg in system_messages:
                content = msg.content
                # 判断是否是基本系统提示（你是一个有用的AI助手）
                if content.startswith("你是一个有用的AI助手") and len(content) < 50:
                    base_prompts.append(msg)
                # 判断是否是RAG增强提示
                elif "请根据以下提供的文档内容回答用户问题" in content:
                    rag_prompts.append(msg)
                else:
                    other_system_prompts.append(msg)
            
            # 处理基本系统提示：只保留最后一个
            final_base_prompt = base_prompts[-1] if base_prompts else None
            
            # 处理RAG提示：合并RAG文档内容
            final_rag_prompt = None
            if rag_prompts:
                # 找到相关性最高的RAG提示
                best_rag_prompt = None
                highest_score = -1
                
                for prompt in rag_prompts:
                    # 查找文档相关性分数
                    score_matches = re.findall(r"相关性: (0\.\d+)", prompt.content)
                    scores = [float(score) for score in score_matches]
                    max_score = max(scores) if scores else 0
                    
                    if max_score > highest_score:
                        highest_score = max_score
                        best_rag_prompt = prompt
                
                final_rag_prompt = best_rag_prompt
            
            # 构建最终的消息列表
            deduplicated_messages = []
            if final_base_prompt:
                deduplicated_messages.append(final_base_prompt)
            if final_rag_prompt:
                deduplicated_messages.append(final_rag_prompt)
            deduplicated_messages.extend(other_system_prompts)
            deduplicated_messages.extend(other_messages)
            
            print(f"[Agent Node] System message deduplication: {len(system_messages)} -> {len(deduplicated_messages) - len(other_messages)}")
            return deduplicated_messages

        # 应用系统消息去重
        valid_messages = deduplicate_system_messages(valid_messages)
        # --- 结束系统消息去重 ---

        # --- 应用 LangChain 消息管理 ---
        MAX_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", 3000)) # 从环境变量获取或默认
        SUMMARIZE_THRESHOLD = int(os.environ.get("SUMMARIZE_MSG_COUNT", 10)) # 超过多少条非系统消息时考虑摘要
        SUMMARIZE_TOKEN_THRESHOLD = int(os.environ.get("SUMMARIZE_TOKEN_THRESHOLD", MAX_TOKENS * 0.6)) # Token阈值，默认为最大上下文的60%

        # --- 新增：检查是否使用RAG增强，且需要跳过历史消息 ---
        config_dict = {}
        if state.get("config") and state["config"].get("configurable"):
            config_dict = state["config"]["configurable"]
        
        rag_enhanced = config_dict.get("rag_enhanced", False)
        skip_history = config_dict.get("skip_history", False)
        
        if rag_enhanced and skip_history:
            print("[Agent Node] RAG增强模式启用，且跳过历史消息")
            # 在RAG模式下，我们只保留系统消息和最新的消息（这些已经包含RAG检索内容）
            # 找到最后几条消息（通常是RAG上下文和新问题）
            # 假设最后3条消息为：系统提示、RAG上下文、用户问题
            last_messages = valid_messages[-3:] if len(valid_messages) >= 3 else valid_messages
            
            # 确保保留所有系统消息
            system_msgs = [msg for msg in valid_messages if isinstance(msg, SystemMessage)]
            
            # 去重合并
            final_messages_for_llm = []
            for msg in system_msgs:
                if not any(isinstance(m, SystemMessage) and m.content == msg.content for m in final_messages_for_llm):
                    final_messages_for_llm.append(msg)
            
            # 添加非系统消息（通常是RAG上下文和用户问题）
            non_system_last_msgs = [msg for msg in last_messages if not isinstance(msg, SystemMessage)]
            final_messages_for_llm.extend(non_system_last_msgs)
            
            print(f"[Agent Node] RAG模式：使用 {len(final_messages_for_llm)} 条消息 ({len(system_msgs)} 系统 + {len(non_system_last_msgs)} 最新)")
            
            # 跳过后续的修剪和摘要逻辑
            print(f"[Agent Node] Final messages passed to LLM: {len(final_messages_for_llm)} messages")
            print(f"[Agent Node] Final messages detail:\n{final_messages_for_llm}")
            
            print(f"[Agent Node] Calling LLM with {len(final_messages_for_llm)} messages...")
            try:
                # 调用绑定了工具的 LLM
                response = llm_with_tools.invoke(final_messages_for_llm)
                print(f"[Agent Node] LLM Response received: type={type(response)}, content={str(response.content)[:100]}...")
                # Agent 节点直接返回 LLM 的响应 (可能包含工具调用)
                return {"messages": [response]}
            except Exception as llm_error:
                print(f"[Agent Node] Error during LLM invocation: {llm_error}")
                import traceback
                traceback.print_exc()
                # 返回错误信息时，也应该是一个列表
                response = AIMessage(content=f"抱歉，处理您的请求时发生错误。")
                return {"messages": [response]}
        
        # 如果不是RAG模式或不需要跳过历史，继续原有逻辑
        # --- 结束RAG模式检查 ---

        # 手动保留 SystemMessage
        system_messages = filter_messages(valid_messages, include_types=[SystemMessage])
        other_messages = filter_messages(valid_messages, exclude_types=[SystemMessage])
        print(f"[Agent Node] Separated messages: {len(system_messages)} System, {len(other_messages)} Other")

        # 1. 对非系统消息进行初始修剪
        print(f"[Agent Node] Attempting initial trim on {len(other_messages)} messages with max_tokens={MAX_TOKENS}...")
        try:
            trimmed_other_messages = trim_messages(
                other_messages,
                max_tokens=MAX_TOKENS - 50, # 预留一点空间给系统消息和可能的摘要
                strategy="last", # 保留最新的消息
                token_counter=count_message_tokens, # 使用 tiktoken 计数器
                # allow_partial=True # 允许截断部分消息，避免因单条消息过长导致失败
            )
            print(f"[Agent Node] Messages after initial trim (excluding system): {len(trimmed_other_messages)} messages")
        except Exception as trim_error:
             print(f"[Agent Node] Error during initial trim: {trim_error}. Falling back to keeping last ~10 messages.")
             # 回退策略：保留最近的 N 条消息
             trimmed_other_messages = other_messages[-10:]


        # 2. 判断是否需要摘要 (基于修剪后的非系统消息)
        final_messages_for_llm = system_messages + trimmed_other_messages # 先组合起来
        non_system_messages_count = len(trimmed_other_messages)

        # 计算当前消息的token数，判断是否接近阈值
        current_tokens = count_message_tokens(final_messages_for_llm)
        print(f"[Agent Node] Current token count: {current_tokens}/{MAX_TOKENS} (threshold: {SUMMARIZE_TOKEN_THRESHOLD})")

        # 提前触发摘要条件：1) 消息数量超过阈值 或 2) token数超过阈值的60%
        should_summarize = (non_system_messages_count > SUMMARIZE_THRESHOLD) or (current_tokens > SUMMARIZE_TOKEN_THRESHOLD)

        if should_summarize:
            print(f"[Agent Node] Summarization triggered: message count={non_system_messages_count}, tokens={current_tokens}")
            # 保留最近的几条消息，其余进行摘要
            keep_recent = 3  # 保留最近3条非系统消息
            messages_to_summarize = trimmed_other_messages[:-keep_recent] if len(trimmed_other_messages) > keep_recent else []
            recent_messages = trimmed_other_messages[-keep_recent:] if len(trimmed_other_messages) > keep_recent else trimmed_other_messages

            # 生成摘要 (如果有足够消息)
            if len(messages_to_summarize) >= 2:  # 至少需要2条消息才值得摘要
                summary_message = summarize_conversation(messages_to_summarize, llm)

                if summary_message:
                    # 构建最终消息列表：系统消息 + 摘要 + 最近消息
                    final_messages_for_llm = system_messages + [summary_message] + recent_messages
                    print(f"[Agent Node] Summarization successful. Final messages count: {len(final_messages_for_llm)} (System + Summary + Recent)")
                else:
                    # 如果摘要失败，则仍然使用初始修剪后的结果
                    print("[Agent Node] Summarization failed, using initially trimmed messages.")
                    final_messages_for_llm = system_messages + trimmed_other_messages # 重新组合
            else:
                print(f"[Agent Node] Not enough messages to summarize ({len(messages_to_summarize)}), using initially trimmed messages.")
                final_messages_for_llm = system_messages + trimmed_other_messages
        else:
             print(f"[Agent Node] Message count ({non_system_messages_count}) and token count ({current_tokens}) within threshold, using initially trimmed messages.")
             final_messages_for_llm = system_messages + trimmed_other_messages # 重新组合

        # --- 结束消息管理 ---

        print(f"[Agent Node] Final messages passed to LLM: {len(final_messages_for_llm)} messages")
        # 打印最终消息用于调试 (可选，可能很长)
        print(f"[Agent Node] Final messages detail:\n{final_messages_for_llm}")

        if not final_messages_for_llm:
             print("[Agent Node] Error: No valid messages found after processing.")
             final_messages_for_llm = system_messages + valid_messages[-2:] # 至少保留系统和最近一轮
             if len(final_messages_for_llm) <= len(system_messages): # 极端情况
                 print("[Agent Node] Error: Could not recover any messages.")
                 return {"messages": [AIMessage(content="抱歉，处理消息历史时出现错误。")]}

        print(f"[Agent Node] Calling LLM with {len(final_messages_for_llm)} messages...")
        try:
            # 调用绑定了工具的 LLM
            response = llm_with_tools.invoke(final_messages_for_llm)
            print(f"[Agent Node] LLM Response received: type={type(response)}, content={str(response.content)[:100]}...")
            # Agent 节点直接返回 LLM 的响应 (可能包含工具调用)
            return {"messages": [response]}
        except Exception as llm_error:
            print(f"[Agent Node] Error during LLM invocation: {llm_error}")
            import traceback
            traceback.print_exc()
            # 返回错误信息时，也应该是一个列表
            response = AIMessage(content=f"抱歉，处理您的请求时发生错误。")
            return {"messages": [response]}

    # 定义图构建器
    builder = StateGraph(AgentState)

    # 添加 Agent 节点
    builder.add_node("agent", agent)

    # 添加 ToolNode
    tool_node = ToolNode([dd_search, send_email])
    builder.add_node("action", tool_node)

    # 设置入口点
    builder.add_edge(START, "agent")

    # 添加条件边
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "action",
            END: END
        }
    )

    # 添加从工具执行节点返回 Agent 的边
    builder.add_edge("action", "agent")

    print("Graph definition with tool handling complete.")
    return builder

# 获取默认系统提示
def get_default_system_message() -> SystemMessage:
    """返回默认的系统消息"""
    content = os.environ.get("DEFAULT_SYSTEM_PROMPT", "你是一个乐于助人的AI助手。你可以使用 web_search 工具来查找实时信息或你不知道的事情。请利用提供的背景记忆（如果有的话）来更好地与用户交流。")
    return SystemMessage(content=content)


def process_graph_stream(graph, user_input: str, history=None, config=None):
    """处理用户输入并返回每个节点的输出流"""
    messages_input = []
    if history:
        messages_input = history  # history should already be BaseMessage objects if passed from app.py

    # Ensure user_input is added correctly
    if isinstance(user_input, BaseMessage):
        messages_input.append(user_input)
    elif isinstance(user_input, str):
        messages_input.append(HumanMessage(content=user_input))
    else:
        # Handle potential error or unexpected type
        print(f"[process_graph_stream] Warning: Unexpected user_input type: {type(user_input)}. Converting to HumanMessage.")
        messages_input.append(HumanMessage(content=str(user_input)))

    if config is None:
        default_thread_id = f"cli_session_{str(uuid.uuid4())}"
        config = {"configurable": {"thread_id": default_thread_id}}

    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"[LANGGRAPH] 正在处理 thread_id={thread_id} 的请求，初始消息数: {len(messages_input)}")
    print(f"[LANGGRAPH] 当前请求的配置: {config}")

    try:
        langgraph_config = RunnableConfig(configurable=config.get("configurable", {}))
        
        # 跟踪已经生成的内容，用于检测新增内容
        last_content = ""

        # 使用stream模式流式获取LLM的输出
        for event in graph.stream(
            {"messages": messages_input},  # Pass the combined list
            config=langgraph_config,
            stream_mode="values"  # 使用values模式获取全部状态
        ):
            # 获取event中的messages
            if "messages" in event:
                msgs = event.get("messages", [])
                if msgs and len(msgs) > 0:
                    # 获取最后一条消息（AI回复）
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        current_content = last_msg.content
                        # 如果有新内容，只返回新增部分
                        if current_content != last_content:
                            new_content = current_content[len(last_content):]
                            if new_content:  # 确保有新内容才yield
                                yield new_content
                                last_content = current_content

        print(f"[LANGGRAPH] thread_id={thread_id} 的流处理完成")

    except Exception as e:
        print(f"[LANGGRAPH] thread_id={thread_id} 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"流式传输过程中发生错误: {str(e)}"  # Yield error message


# 交互式命令行界面（保留原有功能）
def start_cli():
    """启动命令行界面"""
    graph_builder = create_graph()
    graph = graph_builder.compile() # 不带检查点编译

    print("欢迎来到 LangGraph 聊天机器人 CLI！输入 'quit' 退出。")
    config = {"configurable": {"thread_id": f"cli_chat_{uuid.uuid4()}"}} # 每次启动新的 CLI 会话

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            continue

        print("AI: ", end="", flush=True)
        full_response = ""
        # CLI 每次都是新对话，传递空历史
        for chunk in process_graph_stream(graph, user_input, history=[], config=config):
            print(chunk, end="", flush=True)
            full_response += chunk
        print() # 换行


# 如果直接运行此脚本，启动命令行界面
if __name__ == "__main__":
    start_cli()
