# BabyBot

轻量级多通道对话 Agent 框架。支持可插拔执行后端、多 Agent 团队协作、工具调用、MCP 服务、技能路由、分层长期记忆、多模态图片，以及飞书 / 微信等即时通讯平台接入。

当前实现已经形成比较清晰的分层：

- `ResourceManager` 对外仍是统一 facade
- 技能加载、工具注册、资源作用域、宿主 Python 执行、workspace 工具、subagent runtime 等内部职责已拆到独立模块
- `ExecutorRegistry` 管理可插拔执行后端（Gateway / Claude Code），`DynamicOrchestrator` 负责子任务编排和团队协作调度（通过 `skill_id` 引用 prompt-only skill 定义角色）
- 只有主 agent 负责最终向通道发送消息；子 agent 只执行任务并返回文本/文件路径

## 快速开始

```bash
# 安装依赖
uv sync

# 准备默认目录
mkdir -p ~/.babybot ~/.babybot/workspace

# 配置主文件
cp config.json.example ~/.babybot/config.json
# 编辑 ~/.babybot/config.json，填入模型 API key 等

# 初始化工作区定时任务文件
cp scheduled_tasks.json.example ~/.babybot/workspace/scheduled_tasks.json

# 交互式 CLI（本地调试）
uv run babybot

# 网关模式（启动所有 enabled 通道，用于生产部署）
uv run gateway
```

## 架构概览

```
                        Channel (Feishu, Weixin, ...)
                              │
                    InboundMessage (text + media_paths)
                              │
                         MessageBus
                    (Queue + Semaphore 并发控制)
                              │
                      OrchestratorAgent
                    (Tape 记忆 + 上下文压缩)
                              │
                  DynamicOrchestrator ── dispatch_team ──► TeamRunner
                  (动态子任务编排)        (多 Agent 辩论/协作)
                              │
                      ExecutorRegistry
                    (可插拔执行后端路由)
                       ╱           ╲
            ResourceBridge     ClaudeCodeExecutor
          (内置 Gateway 链路)   (claude CLI 子进程)
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
| **OrchestratorAgent** | `orchestrator.py` | 接收用户输入，管理 Tape + Hybrid Memory，调度子任务 |
| **DynamicOrchestrator** | `agent_kernel/dynamic_orchestrator.py` | 动态子任务编排；支持 `dispatch_task` 和 `dispatch_team` |
| **ExecutorRegistry** | `agent_kernel/executors/__init__.py` | 可插拔执行后端注册与路由（按 backend 名称分发） |
| **ClaudeCodeExecutor** | `agent_kernel/executors/claude_code.py` | 通过 `claude` CLI 子进程驱动 Claude Code |
| **TeamRunner** | `agent_kernel/team.py` | 结构化多 Agent 交互（辩论模式），支持 per-agent executor |
| **ResourceManager** | `resource.py` | 对外资源 facade；聚合工具/MCP/技能/子任务运行时 |
| **SingleAgentExecutor** | `agent_kernel/executor.py` | model → tool_calls → tool_results 执行循环 |
| **OpenAICompatibleGateway** | `model_gateway.py` | OpenAI 兼容 API 封装，支持图片 vision 格式 |
| **Tape / TapeStore** | `context.py` | 长期记忆：SQLite 持久化 + LRU 缓存 + BM25 跨锚点召回 |
| **HybridMemoryStore** | `memory_store.py` | 软/硬/临时记忆、纠正覆盖、衰减与过期维护 |
| **ContextView** | `context_views.py` | Hot / Warm / Cold 三层上下文视图与相关性排序 |

### `resource.py` 内部拆分

当前 `ResourceManager` 相关逻辑已拆分到以下模块：

| 模块 | 职责 |
|------|------|
| `resource_models.py` | 资源/技能数据模型 |
| `resource_python_runner.py` | 宿主 Python 选择、探测、fallback、外部技能脚本执行 |
| `resource_workspace_tools.py` | workspace 文件读写、代码执行、shell 执行 |
| `resource_skill_loader.py` | `SKILL.md`、脚本 AST/CLI 解析、技能工具注册 |
| `resource_scope.py` | 资源 brief、scope 解析、lease 组装 |
| `resource_tool_loader.py` | workspace tools 的注册/加载 |
| `resource_subagent_runtime.py` | 子 agent 运行时 orchestration |
| `resource_skill_runtime.py` | skill pack 选择、worker prompt 与技能目录格式化 |
| `builtin_tools/` | 内置 basic/code/observability 工具定义与注册清单 |

## 运行模式

### 交互式 CLI

```bash
uv run babybot
```

本地 REPL，用于调试。支持 `status`、`reset`、`quit` 命令。

也支持按会话进入交互式 CLI backend 模式：

```text
@session start claude
@session status
@session stop
```

- `@session start claude`：在当前 `chat_key` 下启动 Claude 交互会话
- `@session status`：查看当前会话是否已启动及 session_id
- `@session stop`：关闭当前交互会话，后续消息恢复走默认 DAG 编排路径

当前实现说明：

- 当前 Claude 交互会话是“会话恢复式交互”，不是“常驻进程式交互”
- `@session start claude` 会先创建一个按 `chat_key` 隔离的 Claude 会话上下文
- 后续每条消息都会重新启动一次 `claude` CLI，并通过 `--resume <session_id>` 继续该会话
- 会话状态当前保存在 BabyBot 管理的隔离目录中，因此用户在默认终端里直接执行 `claude --resume <session_id>` 未必能接上

交互会话后续计划：

1. 保持现有 `@session start/status/stop` 控制面不变，避免影响当前 CLI 与网关路由。
2. 将 `InteractiveSessionManager` 继续作为按 `chat_key` 的会话注册表，不重写 MessageBus、Orchestrator 和渠道层。
3. 把 Claude backend 从“每条消息重新执行 `claude --resume`”逐步演进为“每个会话一个常驻 Claude 进程”。
4. 将 BabyBot 自己的 session 标识与 Claude 内部 resume id 解耦，避免状态展示误导用户。
5. `@session status` 后续改为优先展示 backend 模式、进程存活状态、最近活跃时间，而不是只展示 `session_id`。
6. 保留当前隔离环境策略，后续在常驻进程模式下继续沿用隔离目录管理 runtime、临时文件和状态文件。
7. 在 backend 层补齐 reader/writer 生命周期、超时、异常退出、reset/stop 清理，不把这些复杂度扩散到渠道层。
8. 等常驻进程模式稳定后，再评估是否追加流式输出回传到飞书/微信，以及是否支持手动接管会话。

## 编排策略学习

当前实现增加了一层保守型 orchestration policy learning，用来优化“任务怎么拆、什么时候并行、什么时候不要再开 worker”，而不是去微调底层模型。

默认行为：

- 默认自动开启 policy learning，不需要手工打开
- `policy_learning_enabled` 现在更适合当作开发/回滚开关；只有你想强制关闭时才需要配置
- 可选 override：
  - `policy_learning_min_samples`：`0` 表示自动模式；只有你想手工覆盖动作最小样本阈值时才需要配置
  - `policy_learning_explore_ratio`：`-1.0` 表示自动模式；只有你想手工覆盖探索预算时才需要配置

自动采集的信号：

- 拆解策略决策：如 `analyze_then_execute`
- 调度决策：如串行/并行 dispatch、wait barrier
- worker gate 决策：允许还是拒绝继续创建 worker
- 最终 outcome：成功/失败、reward，以及重试、dead letter、stalled 等惩罚信号
- action 聚合统计：`mean_reward`、`success_rate`、`failure_rate`
- 风险统计：`retry_rate`、`dead_letter_rate`、`stalled_rate`
- 衰减统计：`effective_samples`
- 人工反馈统计：`feedback_good_count`、`feedback_bad_count`、`feedback_score`
- 反馈可信度统计：`effective_feedback_samples`、`feedback_confidence`

当前保守选择规则：

- 样本不足时，拆解默认 `analyze_then_execute`
- 调度默认更偏向串行，只有历史收益更好且风险更低时才偏向受限并行
- worker 使用默认更保守；启用策略学习后，如果历史数据不足或风险偏高，会拒绝继续创建/分发 worker
- 不是全局平均直接排序，而是先按 bucket 查局部历史，没命中再回落到全局
- 当前 bucket 只用稳定特征：`task_shape`、`has_media`、`independent_subtasks`
- bucket 不是只查一个最具体值，而是按“具体 → 一般”自动回退，例如先查 `task_shape + has_media + subtasks`，再查更一般的组合
- 排序不是只看 `mean_reward`，还会对 `failure_rate`、`retry_rate`、`dead_letter_rate`、`stalled_rate` 做惩罚
- 历史 outcome 会做时间衰减，越旧的经验影响越小，不会长期主导当前策略
- `good/bad` 反馈不会直接覆盖 reward，而是先做时间衰减，再按 `feedback_confidence` 进行有限幅度 shaping
- 最终选择器是保守版 contextual bandit：对每个 action 用经验分减去和样本量相关的置信惩罚，优先选择更稳而不是更激进的动作
- 自动模式下，最小样本阈值和探索预算由系统内部护栏决定，不要求人工调参

人工纠偏命令：

```text
@policy feedback good 拆分合理
@policy feedback bad 并行导致多次重试
```

- 反馈会绑定到当前 `chat_key` 最近一次执行 flow
- 如果当前会话还没有最近任务，会直接返回明确错误，不会进入正常对话编排

查看策略数据：

- SQLite 文件：`~/.babybot/memory/policy.db`
- 快速查看最近反馈：

```bash
sqlite3 ~/.babybot/memory/policy.db \
  "select flow_id, chat_key, rating, reason, created_at from policy_feedback order by id desc limit 20;"
```

- 查看动作统计：

```bash
sqlite3 ~/.babybot/memory/policy.db \
  "select decision_kind, action_name, count(*) from policy_decisions group by decision_kind, action_name order by count(*) desc;"
```

### 网关模式

```bash
uv run gateway
```

生产模式。自动发现并启动 `config.json` 中所有 `enabled: true` 的通道，通过 MessageBus 调度消息。

## 配置说明

主配置位于 `config.json`，定时任务位于工作区独立文件 `scheduled_tasks.json`。默认路径：

- 主配置：`~/.babybot/config.json`（可由 `BABYBOT_CONFIG` 覆盖）
- 工作区：`~/.babybot/workspace`（可由 `BABYBOT_WORKSPACE` 覆盖）
- 定时任务：`~/.babybot/workspace/scheduled_tasks.json`
- workspace 技能目录：`~/.babybot/workspace/skills`

完整模板见 `config.json.example` 和 `scheduled_tasks.json.example`。

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
      "media_dir": "",
      "stream_reply": false
    },
    "weixin": {
      "enabled": true,
      "base_url": "https://ilinkai.weixin.qq.com",
      "cdn_base_url": "https://novac2c.cdn.weixin.qq.com/c2c",
      "token": "",
      "state_dir": "~/.babybot/weixin",
      "media_dir": "~/.babybot/weixin/media",
      "poll_timeout": 35,
      "allow_from": []
    }
  }
}
```

**Feishu 字段**

| 字段 | 说明 |
|------|------|
| `group_policy` | `mention` — 群聊中仅 @bot 时响应；`open` — 所有消息都响应 |
| `reply_mode` | `chat` — 回复到当前会话；`p2p` — 按发送者 open_id 私聊回复 |
| `media_dir` | 图片等媒体文件下载目录，为空则使用默认目录 |
| `stream_reply` | `true` 启用模型生成期实时流式推送（边生成边 patch 卡片） |

**Weixin 字段**

| 字段 | 说明 |
|------|------|
| `base_url` | 微信 long-poll API 地址，通常保持默认即可 |
| `cdn_base_url` | 微信媒体上传/下载 CDN 地址，通常保持默认即可 |
| `token` | 已登录后的 bot token；为空时启动会自动进入二维码登录 |
| `state_dir` | 保存二维码 PNG 和 `account.json` 登录状态 |
| `media_dir` | 微信图片/文件下载目录，为空则默认到 `~/.babybot/media/weixin` |
| `poll_timeout` | 长轮询超时时间（秒） |
| `allow_from` | 允许接入的微信用户 ID 白名单，空数组表示不限制 |

### 微信测试步骤

1. 安装依赖：`uv sync`
2. 编辑 `~/.babybot/config.json`，填好模型配置，并设置 `channels.weixin.enabled=true`
3. 若希望每次重新扫码，可先删除 `state_dir/account.json`
4. 启动网关：`uv run gateway`
5. 终端会打印二维码或登录 URL，并在 `state_dir/weixin-login-qr.png` 保存二维码图片
6. 扫码确认后，会在 `state_dir/account.json` 写入登录状态；之后重启会优先复用该状态
7. 发送文本、图片、文件到该微信账号，BabyBot 会按当前模型与工具能力回复

> 说明：若运行环境未安装 `qrcode` 相关依赖，终端仍会输出登录 URL，但二维码 PNG 可能不会生成。

### 系统参数

```json
{
  "system": {
    "timeout": 600,
    "idle_timeout": 60,
    "max_concurrency": 8,
    "scheduled_max_concurrency": 2,
    "message_queue_maxsize": 1000,
    "max_per_chat": 1,
    "send_ack": true,
    "python_executable": "",
    "python_fallback_executables": [],
    "worker_max_steps": 14,
    "orchestrator_max_steps": 30,
    "context_history_tokens": 2000,
    "context_compact_threshold": 3000,
    "context_max_chats": 500,
  }
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | 600 | 单任务硬超时（秒） |
| `idle_timeout` | 60 | 空闲超时，无心跳则终止任务 |
| `max_concurrency` | 8 | 全局最大并行消息数 |
| `scheduled_max_concurrency` | 2 | 定时任务并行上限（与用户消息并发池隔离） |
| `message_queue_maxsize` | 1000 | MessageBus 单队列最大积压，超过后入队会背压等待 |
| `max_per_chat` | 1 | 单个会话最大并行数 |
| `send_ack` | true | 是否发送"收到，正在处理..."回执 |
| `python_executable` | `""` | 技能脚本优先使用的宿主 Python |
| `python_fallback_executables` | `[]` | 技能脚本执行失败时可回退的 Python 列表 |
| `worker_max_steps` | 14 | 子 agent 最大执行轮数 |
| `orchestrator_max_steps` | 30 | 主 orchestrator 最大执行轮数 |
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

### 内置工具

当前项目中的工具入口以 `builtin_tools/` 为主，主要分为：

- `basic`：worker 调度、定时任务、观测工具、技能热加载（`reload_skill`）
- `code`：workspace 文件读写、Python / shell 执行
- `channel_*`：由通道在运行时注册的发送类工具

其中新的只读观测工具包括：

- `inspect_runtime_flow`：查看当前 flow 的子任务状态、最近 runtime events、progress
- `inspect_chat_context`：查看当前 chat 的 Hot / Warm / Cold 视图、memory records、tape 摘要

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

### Prompt-only Skill（团队角色技能）

Skill 同时承担两种用途：**工具路由**（带 `scripts/` 的常规技能）和**团队角色定义**（prompt-only 技能）。一个不含 `scripts/` 目录、仅有 SKILL.md 的技能就是 prompt-only skill，可在 `dispatch_team` 中通过 `skill_id` 引用，为团队辩论 / 协作 Agent 提供预定义的角色和 system prompt。

**SKILL.md 示例（prompt-only 团队角色）：**

```markdown
---
name: code-reviewer
role: 代码评审专家
description: 擅长发现代码质量问题和潜在 bug
---

你是一个严谨的代码评审专家。在评审时请关注：
1. 逻辑正确性
2. 边界条件处理
3. 错误处理完整性
4. 代码可读性
```

**Frontmatter 字段**

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | 是 | 技能唯一标识 |
| `description` | 是 | 技能用途描述，同时用于路由匹配 |
| `role` | 否 | 团队角色标注，用于 `dispatch_team` 中自动填充 agent 的 `role` |

Markdown body 作为 `prompt`（即 system_prompt），在 TeamRunner 执行时传递给 executor。

`dispatch_team` 的 agent 定义中使用 `skill_id` 引用已加载的 skill，skill 提供 `role` / `description` / `prompt` 的默认值，agent dict 中的显式字段会覆盖。

### 多 Agent 团队协作

BabyBot 支持通过 `dispatch_team` 组织多个 Agent 进行结构化多轮协作（辩论、评审、头脑风暴）。该能力已完全接入 Channel 链路，用户通过飞书、微信或 CLI 发送消息即可触发。

**触发方式：**

编排 Agent（DynamicOrchestrator）会根据用户请求自动判断是否需要多 Agent 协作。以下类型的请求会触发 `dispatch_team`：

- "请从正反两面分析这个方案的优缺点"
- "组织一场关于 Python vs Rust 的辩论"
- "让代码评审专家和安全专家一起审查这段代码"
- "请几个不同角色的专家讨论这个架构设计"

**工作流程：**

```
用户消息 (Channel)
  → MessageBus → OrchestratorAgent → DynamicOrchestrator
    → LLM 决策：调用 dispatch_team
      → TeamRunner.run_debate()
        → Agent A 发言 (Round 1) → on_turn 回调 → 独立消息卡片 + heartbeat
        → Agent B 发言 (Round 1) → on_turn 回调 → 独立消息卡片 + heartbeat
        → Agent A 发言 (Round 2)
        → ...（最多 max_rounds 轮，或 judge 判定收敛后提前结束）
      → 返回讨论记录和总结
    → LLM 调用 reply_to_user 汇总回复
  → Channel 发送最终结果给用户
```

每轮辩论发言通过 `on_turn` 回调实时推送为独立飞书消息卡片（而非覆盖同一张卡片），同时保持 heartbeat 活跃，避免长时间辩论因空闲超时被终止。

**高级用法 — 带工具能力的 Agent：**

如果 agent 指定了 `resource_id`，该 agent 在辩论中可以调用对应资源的工具（如搜索、代码执行、数据分析），而不仅仅是纯文本讨论。这通过 `ResourceBridgeExecutor` 实现，agent 的每次发言都会经过完整的 executor 链路。

**高级用法 — 引用预定义 Skill：**

在 `skills/` 目录下创建 prompt-only 的 SKILL.md 后，LLM 在调用 `dispatch_team` 时可以通过 `skill_id` 引用这些角色定义，无需每次在 tool call 参数中重复写完整的 role / description / system_prompt。

**快速体验步骤：**

1. 确保已配置好模型 API（`~/.babybot/config.json`）
2. （可选）创建 prompt-only 角色技能：
   ```bash
   mkdir -p ~/.babybot/workspace/skills/pro-analyst
   cat > ~/.babybot/workspace/skills/pro-analyst/SKILL.md << 'EOF'
   ---
   name: pro-analyst
   role: 正方分析师
   description: 从积极角度分析问题
   ---

   你是一个乐观的分析师，擅长发现方案的优势和机会。
   EOF
   ```
3. 启动 CLI 或 Gateway：`uv run babybot` 或 `uv run gateway`
4. 发送消息："请从多个角度分析远程办公的利弊"
5. BabyBot 会自动组织多 Agent 辩论，最终汇总回复

技能脚本执行说明：

- 技能脚本优先在“宿主 Python”环境执行，而不是项目当前 venv
- Python 选择顺序支持：
  1. 技能级 `python_executable`
  2. 技能级 `python_fallback_executables`
  3. 系统级 `system.python_executable`
  4. 系统级 `system.python_fallback_executables`
  5. 自动探测的本机 Python
- 遇到环境类失败（如缺包、动态库缺失、解释器不可用）会自动尝试 fallback
- 遇到业务类失败（脚本自身报错）不会盲目重试

示例：

```json
{
  "system": {
    "python_executable": "/Users/you/miniconda3/bin/python3",
    "python_fallback_executables": [
      "/usr/bin/python3",
      "/opt/homebrew/bin/python3"
    ]
  },
  "agent_skills": {
    "mlx_audio": {
      "directory": "skills/mlx-audio",
      "python_executable": "/Users/you/miniconda3/bin/python3",
      "python_fallback_executables": ["/usr/bin/python3"]
    }
  }
}
```

### 定时任务

工作区中的 `scheduled_tasks.json` 保存所有定时任务定义，不再和主配置混在一起。任务定义示例：

```json
[
  {
    "name": "每日新闻摘要",
    "prompt": "搜索并总结今天的科技新闻，列出5条最重要的",
    "schedule": "0 9 * * *",
    "target": {
      "channel": "feishu",
      "chat_id": "oc_xxx"
    },
    "enabled": false
  }
]
```

要求：

- 每个任务名必须唯一。
- `schedule` 必须是三选一：合法 cron 字符串、`{ "interval": 7200 }`（循环间隔秒）或 `{ "run_at": "2026-03-16T17:10:00+08:00" }`（一次性触发）。
- 旧版 `config.json` 中的 `scheduled_tasks` 会在首次加载时自动迁移到工作区文件。
- 通过自然语言创建任务时可以不提供 `name`；系统会根据 `prompt + target + schedule` 自动生成稳定名字，并尽量复用已有同类任务而不是重复创建。
- 默认要求当前进程有活跃调度器（通常是 `uv run gateway`）；否则会提示“仅落盘、不保证执行”，避免误报创建成功。

## 主 / 子 Agent 职责边界

- 主 agent 负责：
  - 接收用户输入
  - 路由技能与工具
  - 汇总子任务结果
  - 最终向飞书等通道回复
- 子 agent 负责：
  - 在受限 `ToolLease` 下执行任务
  - 返回文本结果和生成文件路径
  - 不直接向用户发送消息

这意味着即使某个技能能生成音频/图片，子 agent 也只返回产物路径，最终发送动作仍由主 agent 完成。

## 记忆与上下文

当前上下文系统由 **Tape + Hybrid Memory + Context View** 组成：

1. **Tape** — 单个会话的 append-only 时间线，记录消息、工具调用、runtime event、锚点等
2. **Anchor（锚点）** — 当累积 token 超过阈值时，LLM 自动生成结构化摘要（summary / entities / user_intent / pending / next_steps / artifacts / decisions），创建新锚点并压缩旧条目
3. **Hybrid Memory** — 额外维护硬记忆、软记忆、临时记忆：
   - 硬记忆：文件保存的稳定规则，如主/子 agent 边界
   - 软记忆：用户偏好、用户画像、助手角色、任务决策
   - 临时记忆：最近失败、最近成功、当前待办、当前产物
4. **Hot / Warm / Cold 视图** — query-aware 的三层上下文整理：
   - Hot：当前任务强相关的临时状态 + 硬约束
   - Warm：活跃的偏好/画像/决策
   - Cold：已衰减但仍可参考的较旧记忆
5. **纠正与衰减** — 新证据会覆盖旧偏好；单次软记忆会经历 `candidate -> decaying -> expired`
6. **跨锚点召回** — 基于关键词提取（CJK bigram + Latin word）和 BM25 排序，从历史锚点前的条目中召回相关上下文
7. **话题漂移检测** — 对比当前段与前一锚点的关键词重叠度，低于 15% 标记为 `topic_shift`

### 可观测性与调试

当前推荐使用以下只读工具进行排障：

- `inspect_runtime_flow(flow_id?, chat_key?)`
  - 查看子任务 heartbeat 快照
  - 查看最近 runtime events
  - 用于判断任务是否真的卡住，还是仍在持续产生日志/进度
- `inspect_chat_context(chat_key?, query?)`
  - 查看 Hot / Warm / Cold 上下文分层
  - 查看 memory records 的 `tier/status/confidence`
  - 查看当前 tape anchor 摘要

输出采用稳定分段文本格式，固定包含如 `[Runtime Flow]`、`[Tasks]`、`[Recent Events]`、`[Chat Context]`、`[Memory Records]` 等 section，方便人读，也方便 agent 后续消费。

#### 排障示例

当用户反馈“语音生成很慢、而且阶段消息重复”时，推荐按下面顺序看：

1. 先调用 `inspect_runtime_flow(chat_key="feishu:oc_xxx")`
   - 看 `[Tasks]` 里是否还有 task 处于进行中
   - 看 `[Recent Events]` 里是否持续出现 `event=progress`
   - 如果 `status=下载模型`、`progress=0.25/0.75` 持续推进，说明不是卡死，而是在慢任务中
2. 再调用 `inspect_chat_context(chat_key="feishu:oc_xxx", query="继续语音任务")`
   - 看 Hot 是否还保留“最近失败 / 当前待办 / 当前产物”
   - 看 Warm 是否存在“默认中文 / 简洁回答 / 助手角色”之类的活跃偏好
   - 看 Cold 是否堆积了已经衰减的旧记忆，帮助判断上下文是否变脏

一个典型的 `inspect_runtime_flow` 片段会像这样：

```text
[Runtime Flow]
flow_id=orchestrator:abc123
chat_key=feishu:oc_xxx
[Tasks]
- task_id=task_1 status=下载模型 progress=0.75
[Recent Events]
- task_id=task_1 event=progress (下载模型, status=下载模型, progress=0.75)
```

一个典型的 `inspect_chat_context` 片段会像这样：

```text
[Chat Context]
chat_key=feishu:oc_xxx
query=继续语音任务
[Hot Context]
- 当前待办：继续处理语音失败
- 最近失败：生成语音（tts timeout）
[Warm Context]
- 用户偏好默认中文回复。
- 用户偏好简洁回答。
[Memory Records]
- memory_type=task_state key=last_failure tier=ephemeral status=active ...
```

### 运行时进度反馈

当前阶段反馈比之前更轻量：

- `progress` 事件会显示为 `处理中：... (xx%)`
- `succeeded` 事件会显示为 `阶段完成：...`
- MessageBus 会去重重复进度文案，减少刷屏
- 不再在阶段完成时把完整结果提前重复发一次，避免和最终回复重复

对于较慢的宿主 Python 技能脚本：

- 系统会流式读取 stdout / stderr
- 只要脚本仍持续输出，就会持续 heartbeat，不再单纯依赖“长时间无响应 -> 超时”
- 如果日志中包含百分比（如 `25%` / `75%`），会自动转成 progress

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
│   ├── memory_models.py         # 记忆数据模型
│   ├── memory_store.py          # Hybrid Memory 存储 / 衰减 / 覆盖
│   ├── context_views.py         # Hot / Warm / Cold 视图构建
│   ├── resource.py              # ResourceManager facade
│   ├── resource_models.py       # 资源/技能数据模型
│   ├── resource_python_runner.py # 宿主 Python 选择与技能脚本执行
│   ├── resource_scope.py        # 资源 brief / scope / lease
│   ├── resource_skill_loader.py # 技能发现、frontmatter、脚本解析
│   ├── resource_skill_runtime.py # skill pack 选择与 worker prompt
│   ├── resource_subagent_runtime.py # 子 agent 运行时 orchestration
│   ├── resource_tool_loader.py  # 内置 / 通道工具注册加载
│   ├── resource_workspace_tools.py # workspace 文件与代码工具
│   ├── builtin_tools/           # 内置工具定义（worker/scheduler/code/observability）
│   ├── worker.py                # Worker 执行器工厂
│   ├── model_gateway.py         # OpenAI 兼容网关
│   ├── message_bus.py           # 异步消息总线
│   ├── heartbeat.py             # 心跳 / 超时监控
│   ├── context.py               # Tape 长期记忆 + TapeStore
│   ├── mcp_runtime.py           # MCP stdio/HTTP 客户端
│   ├── cron.py                  # 定时任务调度
│   ├── agent_kernel/            # 轻量执行内核
│   │   ├── executor.py          # SingleAgentExecutor
│   │   ├── dynamic_orchestrator.py # DynamicOrchestrator（子任务编排 + team dispatch）
│   │   ├── team.py              # TeamRunner（多 Agent 辩论 / 协作）
│   │   ├── protocols.py         # ExecutorPort protocol
│   │   ├── types.py             # TaskContract / ExecutionContext / TaskResult
│   │   ├── executors/           # 可插拔执行后端
│   │   │   ├── __init__.py      # ExecutorRegistry
│   │   │   └── claude_code.py   # ClaudeCodeExecutor
│   │   ├── dag_ports.py         # ResourceBridgeExecutor
│   │   ├── model.py             # ModelMessage / ModelRequest / ModelResponse
│   │   ├── tools.py             # Tool / ToolRegistry / ToolLease
│   │   ├── skills.py            # SkillPack 合并逻辑
│   │   └── mcp.py               # MCP 工具适配器
│   └── channels/                # 通道集成
│       ├── base.py              # BaseChannel / InboundMessage
│       ├── manager.py           # ChannelManager（自动发现）
│       ├── registry.py          # 通道类自动注册
│       ├── tools.py             # ChannelToolContext / ChannelCapabilities
│       ├── feishu.py            # 飞书通道实现
│       └── weixin.py            # 微信通道实现
├── skills/                      # 技能目录
├── config.json                  # 配置文件
├── config.json.example          # 配置模板
├── scheduled_tasks.json.example # 定时任务模板
├── docs/                        # 设计/实现计划文档
└── tests/                       # 测试
```

## 特性

- **配置与工作区分离** — 主配置在 `config.json`，定时任务在 workspace 独立文件，支持 `${VAR}` 环境变量
- **双运行模式** — CLI 交互调试 + Gateway 多通道生产部署
- **轻量内核** — 最小抽象、低耦合执行内核
- **可插拔执行后端** — `ExecutorRegistry` 按 backend 名称路由，内置 `ResourceBridgeExecutor`（Gateway 链路）和 `ClaudeCodeExecutor`（claude CLI），实现 `ExecutorPort` 即可扩展
- **多 Agent 团队协作** — `dispatch_team` + `TeamRunner` 支持结构化辩论模式，agent 可携带独立 executor 和 resource_id
- **统一 Skill 模型** — SKILL.md 同时支持工具路由和团队角色定义，`dispatch_team` 通过 `skill_id` 引用 prompt-only skill
- **分层记忆** — Tape + Hybrid Memory + Hot/Warm/Cold 三层上下文
- **多模态** — 图片从通道到 LLM 全链路支持（延迟 base64 编码）
- **MCP 支持** — stdio / HTTP 两种传输，动态注册工具
- **技能路由** — 技能 prompt、工具脚本、lease 与 worker prompt 解耦
- **技能热加载** — 创建或更新技能后调用 `reload_skill` 即时生效，无需重启进程
- **宿主 Python fallback** — 技能脚本可在独立本地 Python 环境运行，并支持多级回退
- **运行时可观测性** — flow / chat context 只读观测工具，稳定分段输出
- **更轻量阶段反馈** — progress 去重、阶段完成摘要、stdout/stderr 驱动 heartbeat
- **工具租约** — ToolLease 最小权限策略，技能间 UNION 语义合并
- **主/子 Agent 边界清晰** — 只有主 agent 负责通道发送，子 agent 只返回结果与产物路径
- **可配置执行轮数** — orchestrator / worker 的最大步数可配置
- **并发控制** — 全局 + 单聊信号量，心跳空闲超时检测
- **通道扩展** — 实现 `BaseChannel` 即可接入新平台（已支持飞书、微信）

## License

MIT
