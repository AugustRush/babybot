# BabyBot

轻量级多通道对话 Agent 框架，无 AgentScope 依赖。支持工具调用、MCP 服务、技能路由、长期记忆、多模态图片，以及飞书等即时通讯平台接入。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置
cp config.json.example config.json
# 编辑 config.json，填入模型 API key 等

# 交互式 CLI（本地调试）
uv run babybot

# 网关模式（启动所有 enabled 通道，用于生产部署）
uv run gateway
```

## 架构概览

```
                        Channel (Feishu, ...)
                              │
                    InboundMessage (text + media_paths)
                              │
                         MessageBus
                    (Queue + Semaphore 并发控制)
                              │
                      OrchestratorAgent
                    (Tape 记忆 + 上下文压缩)
                              │
                      ResourceManager
                    (工具注册 + 技能路由)
                              │
                    SingleAgentExecutor
                  (model ↔ tool_calls 循环)
                              │
                  OpenAICompatibleGateway
                    (OpenAI API 调用)
```

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **MessageBus** | `message_bus.py` | 异步消息队列，全局 + 单聊并发信号量 |
| **Heartbeat** | `heartbeat.py` | 空闲超时检测 + 硬超时保护 |
| **OrchestratorAgent** | `orchestrator.py` | 接收用户输入，管理 Tape 记忆，调度子任务 |
| **ResourceManager** | `resource.py` | 工具/MCP/技能的注册、发现与路由 |
| **SingleAgentExecutor** | `agent_kernel/executor.py` | model → tool_calls → tool_results 执行循环 |
| **OpenAICompatibleGateway** | `model_gateway.py` | OpenAI 兼容 API 封装，支持图片 vision 格式 |
| **Tape / TapeStore** | `context.py` | 长期记忆：SQLite 持久化 + LRU 缓存 + BM25 跨锚点召回 |

## 运行模式

### 交互式 CLI

```bash
uv run babybot
```

本地 REPL，用于调试。支持 `status`、`reset`、`quit` 命令。

### 网关模式

```bash
uv run gateway
```

生产模式。自动发现并启动 `config.json` 中所有 `enabled: true` 的通道，通过 MessageBus 调度消息。

## 配置说明

所有配置集中在 `config.json`，支持 `${VAR_NAME}` 环境变量语法。完整模板见 `config.json.example`。

### 模型配置

```json
{
  "model": {
    "model_name": "deepseek-ai/DeepSeek-V3.2",
    "api_key": "${MODEL_API_KEY}",
    "api_base": "https://api-inference.modelscope.cn/v1",
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

兼容任何 OpenAI 格式的 API 端点。

### 通道配置

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "app_id": "cli_xxx",
      "app_secret": "xxx",
      "encrypt_key": "",
      "verification_token": "",
      "group_policy": "mention",
      "reply_mode": "chat",
      "react_emoji": "THUMBSUP",
      "media_dir": ""
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `group_policy` | `mention` — 群聊中仅 @bot 时响应；`open` — 所有消息都响应 |
| `reply_mode` | `chat` — 回复到当前会话；`p2p` — 按发送者 open_id 私聊回复 |
| `media_dir` | 图片等媒体文件下载目录，为空则使用临时目录 |

### 系统参数

```json
{
  "system": {
    "timeout": 600,
    "idle_timeout": 60,
    "max_concurrency": 8,
    "max_per_chat": 1,
    "send_ack": true,
    "context_history_tokens": 2000,
    "context_compact_threshold": 3000,
    "context_max_chats": 500
  }
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | 600 | 单任务硬超时（秒） |
| `idle_timeout` | 60 | 空闲超时，无心跳则终止任务 |
| `max_concurrency` | 8 | 全局最大并行消息数 |
| `max_per_chat` | 1 | 单个会话最大并行数 |
| `send_ack` | true | 是否发送"收到，正在处理..."回执 |
| `context_history_tokens` | 2000 | 历史上下文 token 预算 |
| `context_compact_threshold` | 3000 | 触发锚点压缩的 token 阈值 |
| `context_max_chats` | 500 | 内存中 LRU 缓存的最大会话数 |

### MCP 服务

```json
{
  "mcp_servers": {
    "playwright": {
      "type": "stdio",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "group_name": "browser",
      "active": false
    },
    "gaode_map": {
      "type": "http",
      "url": "https://mcp.amap.com/mcp?key=YOUR_KEY",
      "transport": "streamable_http",
      "group_name": "map_services",
      "active": false
    }
  }
}
```

支持 `stdio`（子进程）和 `http`（HTTP/SSE）两种 MCP 传输方式。

### 工具组

```json
{
  "tool_groups": {
    "search": {
      "active": true,
      "description": "搜索工具",
      "notes": "用于信息检索"
    }
  }
}
```

### 自定义工具

```json
{
  "custom_tools": {
    "web_search": {
      "module": "tools.search",
      "function": "web_search",
      "group_name": "search",
      "preset_kwargs": {
        "api_key": "${SEARCH_API_KEY}"
      }
    }
  }
}
```

将 Python 函数注册为可调用工具。`preset_kwargs` 中的环境变量会在加载时展开。

### Agent 技能

```json
{
  "agent_skills": {
    "data_analysis": {
      "directory": "skills/data_analysis",
      "description": "数据分析技能"
    }
  }
}
```

技能目录需包含 `SKILL.md`（frontmatter 定义元数据 + 正文作为 system prompt），可选 `scripts/` 子目录放置技能专属工具脚本。

## 长期记忆 (TAPE)

基于 **Tape** 模式的长期记忆系统，SQLite 持久化：

1. **Tape** — 单个会话的 append-only 时间线，记录消息、工具调用、锚点等
2. **Anchor（锚点）** — 当累积 token 超过阈值时，LLM 自动生成结构化摘要（summary / entities / user_intent / pending），创建新锚点并压缩旧条目
3. **跨锚点召回** — 基于关键词提取（CJK bigram + Latin word）和 BM25 排序，从历史锚点前的条目中召回相关上下文
4. **话题漂移检测** — 对比当前段与前一锚点的关键词重叠度，低于 15% 标记为 topic_shift

## 多模态图片

通道（如飞书）下载的图片通过 `media_paths` 在管线中全程以文件路径形式流动，仅在 Gateway 最后一步才进行 base64 编码，发送为 OpenAI vision 格式：

```
Channel 下载图片 → InboundMessage.media_paths
  → MessageBus → Orchestrator → ResourceManager → Executor
    → ModelMessage(images=(...))
      → Gateway: base64 编码 → OpenAI content list 格式
```

Tape 中只记录 `[附带 N 张图片]` 文本标记，不存储图片二进制数据。

## 项目结构

```
babybot/
├── babybot/
│   ├── __init__.py              # 入口：main() / gateway()
│   ├── cli.py                   # CLI / Gateway 运行器
│   ├── config.py                # 统一配置管理
│   ├── orchestrator.py          # 编排 Agent
│   ├── resource.py              # 资源管理器（工具/MCP/技能）
│   ├── worker.py                # Worker 执行器工厂
│   ├── model_gateway.py         # OpenAI 兼容网关
│   ├── message_bus.py           # 异步消息总线
│   ├── heartbeat.py             # 心跳 / 超时监控
│   ├── context.py               # Tape 长期记忆 + TapeStore
│   ├── mcp_runtime.py           # MCP stdio/HTTP 客户端
│   ├── scheduler.py             # DAG 任务调度器
│   ├── agent_kernel/            # 轻量执行内核
│   │   ├── executor.py          # SingleAgentExecutor
│   │   ├── model.py             # ModelMessage / ModelRequest / ModelResponse
│   │   ├── tools.py             # Tool / ToolRegistry / ToolLease
│   │   ├── types.py             # TaskContract / ExecutionContext
│   │   ├── skills.py            # SkillPack 合并逻辑
│   │   └── mcp.py               # MCP 工具适配器
│   └── channels/                # 通道集成
│       ├── base.py              # BaseChannel / InboundMessage
│       ├── manager.py           # ChannelManager（自动发现）
│       ├── registry.py          # 通道类自动注册
│       ├── tools.py             # ChannelToolContext / ChannelCapabilities
│       └── feishu.py            # 飞书通道实现
├── skills/                      # 技能目录
├── tools/                       # 自定义工具目录
├── tests/                       # 测试
├── config.json                  # 配置文件
└── config.json.example          # 配置模板
```

## 特性

- **统一 JSON 配置** — 所有配置集中在 `config.json`，支持 `${VAR}` 环境变量
- **双运行模式** — CLI 交互调试 + Gateway 多通道生产部署
- **轻量内核** — 无 AgentScope 依赖，最小抽象
- **长期记忆** — Tape + SQLite + 自动锚点压缩 + BM25 跨锚点召回
- **多模态** — 图片从通道到 LLM 全链路支持（延迟 base64 编码）
- **MCP 支持** — stdio / HTTP 两种传输，动态注册工具
- **技能路由** — 关键词匹配自动选择 SkillPack，注入专属 prompt 和工具
- **工具租约** — ToolLease 最小权限策略，技能间 UNION 语义合并
- **并发控制** — 全局 + 单聊信号量，心跳空闲超时检测
- **通道扩展** — 实现 `BaseChannel` 即可接入新平台

## License

MIT
