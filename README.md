# LangGraph 聊天机器人

一个使用 LangGraph 和 Flask 构建的简单聊天机器人 Web 应用，支持 DeepSeek API。

## 功能特性

- 基于 LangGraph 构建的聊天机器人
- 使用 Flask 提供 Web 界面
- 支持 DeepSeek API (使用 OpenAI 兼容接口)
- 消息历史管理与修剪
- 流式响应
- 暗色模式支持
- 移动设备适配

## 安装

1. 克隆仓库或下载代码

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 创建 `.env` 文件并添加 API 密钥

```
DEEPSEEK_API_KEY=你的DeepSeek_API密钥
OPENAI_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat  # 或其他模型
LLM_TEMPERATURE=0.7  # 可选，控制生成文本的创造性
MAX_TOKENS=4000  # 可选，控制消息历史中保留的最大token数
DEFAULT_SYSTEM_PROMPT="你是一个乐于助人的AI助手。"  # 可选，系统提示词
```

## 运行

运行 Flask 应用:

```bash
python app.py
```

然后在浏览器中访问 http://localhost:5000/

## 项目结构

- `main.py` - LangGraph 聊天图的核心逻辑
- `app.py` - Flask Web 应用
- `templates/index.html` - 聊天界面模板
- `.env` - 环境变量配置文件
- `requirements.txt` - 依赖包列表

## 自定义

### 修改系统提示词

在 `.env` 文件中更新 `DEFAULT_SYSTEM_PROMPT` 变量。

### 调整温度

在 `.env` 文件中更新 `LLM_TEMPERATURE` 变量（0-1之间，越高越有创造性，越低越确定性）。

### 修改 Token 限制

在 `.env` 文件中更新 `MAX_TOKENS` 变量。 