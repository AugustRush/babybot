# BabyBot

轻量级多通道对话 Agent 框架。支持可插拔执行后端、多 Agent 团队协作、工具调用、MCP 服务、技能路由、分层长期记忆、多模态图片，以及飞书 / 微信等即时通讯平台接入。

当前实现已经形成比较清晰的分层：

- `ResourceManager` 对外仍是统一 facade
- 技能加载、工具注册、资源作用域、宿主 Python 执行、workspace 工具、subagent runtime 等内部职责已拆到独立模块
- `ExecutorRegistry` 管理可插拔执行后端（Gateway / Claude Code），`DynamicOrchestrator` 负责子任务编排和团队协作调度（通过 `skill_id` 引用 prompt-only skill 定义角色）
- 只有主 agent 负责最终向通道发送消息；子 agent 只执行任务并返回文本/文件路径

## 运行时文档

运行时合同、长任务与反馈状态机现在有单独文档，README 只保留操作入口与实现概览：

- `docs/agent-runtime/interaction-contract.md`：`TaskContract` / `ExecutionPlan` 的权威字段与约束
- `docs/agent-runtime/long-running-jobs.md`：`RuntimeJob` 状态、`job_id` 语义与长任务查询约定
- `docs/agent-runtime/debate-and-round-budget.md`：辩论轮数预算、停止条件与执行映射
- `docs/agent-runtime/feedback-state-machine.md`：规范化 `RuntimeFeedbackEvent`、状态渲染与去重规则

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
| **TaskContract / ExecutionPlan** | `task_contract.py` / `execution_plan.py` | 用户输入合同化、计划展开、round budget / stop condition 单点约束 |
| **RuntimeJob / RuntimeJobStore** | `runtime_jobs.py` / `runtime_job_store.py` | 长任务状态持久化、`job_id` 查询与最近任务索引 |
| **RuntimeFeedbackEvent** | `feedback_events.py` | 运行时反馈规范化、状态渲染与去重 identity |
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
| `resource_tool_loader.py` | 内置工具、通道工具与 workspace 自定义工具的注册/加载 |
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
@job status latest
@job status <job_id>
@job resume latest
@job resume <job_id>
@job cleanup
```

- `@session start claude`：在当前 `chat_key` 下启动 Claude 交互会话
- `@session status`：查看当前会话是否活跃；过期会话会先被清理，因此会直接显示“当前没有活动中的交互会话”
- `@session stop`：关闭当前交互会话；后续消息恢复走默认 DAG 编排路径
- `@job status latest`：查看当前 `chat_key` 最近一次持久化长任务状态
- `@job status <job_id>`：按显式 `job_id` 查看长任务状态
- `@job resume latest` / `@job resume <job_id>`：按持久化的 `goal`、`media_paths` 重新发起一次恢复执行；新作业会记录 `resumed_from`
- `@job cleanup`：执行一次轻量 runtime 维护，清理无 `flow_id` 且长期停滞的 orphaned jobs，并报告 stale session / unmatched flow

当前实现说明：

- 当前 Claude 交互会话已经切到“常驻本地进程式交互”
- `@session start claude` 会先创建一个按 `chat_key` 隔离的 Claude 会话上下文
- Claude backend 会为每个会话启动一个常驻 `claude` 子进程，并通过 `stream-json` stdin/stdout 进行 turn 级通信
- resident session 的正文输出现在走独立的 interactive 增量事件流，不再混进 `RuntimeFeedbackEvent`
- 同一套 interactive 增量事件会被 CLI、飞书、微信复用：CLI 直接增量打印，飞书优先 patch 同一条 interactive 卡片，微信退化为轻量增量文本消息
- `InteractiveSessionManager` 会在 `status()` / `send()` 前先清理过期会话；如果发送时发现已过期，会回退到默认 DAG 编排
- `InteractiveSessionManager.cleanup()` 可被 runtime 维护入口复用，用于统一清理过期 session
- 交互会话请求会保留 `media_paths`，因此图片/文件输入不会在 session 路径里丢失
- 默认隔离模式为 `same-dir`：会话进程在当前工作区内运行，同时把 `HOME/TMPDIR/CLAUDE_CONFIG_DIR` 隔离到会话 runtime 目录
- `@session status` 和 CLI `status` 会显示更真实的运行态，例如 `mode`、`pid`、`alive`

交互会话后续计划：

1. 保持现有 `@session start/status/stop` 控制面不变，避免影响当前 CLI 与网关路由。
2. 将 `InteractiveSessionManager` 继续作为按 `chat_key` 的会话注册表，不重写 MessageBus、Orchestrator 和渠道层。
3. 继续在现有统一 interactive 事件流之上优化节流、错误恢复和更多 backend 兼容性，而不是再分叉新的流式协议。
4. 将 BabyBot 自己的 session 标识与 Claude 内部 session resume 语义进一步解耦，避免状态展示误导用户。
5. 评估在 resident session 基础上追加 `worktree` 隔离模式，而不是一开始就强推更重的会话隔离。
6. 继续完善 backend reader/writer 生命周期、异常退出恢复和 reset/stop 清理，不把这些复杂度扩散到渠道层。
8. 等常驻进程模式稳定后，再评估是否追加流式输出回传到飞书/微信，以及是否支持手动接管会话。

## 轻量路由与策略学习

当前实现把编排优化拆成三层，目标是提升智能和效率，但保持本地 CPU 友好、低延迟、不过度设计：

- 轻量 routing 层（`orchestration_router.py`）：非 interactive session 的 DAG 路径里，只做一次结构化小模型判定；超时或失败立即回退旧逻辑
- `ConservativePolicySelector`：继续负责局部动作选择，例如拆解、串并行和 worker gate
- `TaskEvaluator + ReflectionStore`：任务结束后异步总结失败模式/稳态经验，下次只回注少量 hint

默认行为：

- `routing_enabled` 默认开启，但只影响 `_answer_with_dag()` 路径，不会打断 `@session` 交互式会话
- `routing_model_name` 可单独配置；为空时回退到当前会话同一模型
- `routing_timeout` 默认 `3.0` 秒；只有进入小模型 router 时才使用，并会根据最近模型路由 telemetry 做保守 fail-fast 调整，优先不阻塞主会话流程
- router 的短超时属于软回退：命中上限时会直接退回默认合同，并按低噪音日志记录，而不是把整条会话记成模型调用错误
- `reflection_enabled` 默认开启；`reflection_max_hints` 默认最多注入 3 条历史反思
- `debug_runtime_feedback` 默认关闭；开启后，消息通道会额外发送开发态调试卡片，直接展示 decomposition / routing / scheduling / worker 的 explain 摘要，不影响正式最终回复
- 对稳定命中的成功 bucket，会优先复用带时间衰减的历史成功反思直达同类路由；这些成功经验也会回流到调度/worker gate 的保守选择，并单独统计 execution_style / parallelism / worker_gate 命中率
- 对规则和 reflection 都没命中的模糊请求，会先尝试一个极轻量的稳定意图桶缓存；只有缓存也没把握时，才进入一次小模型 router
- 意图桶缓存会对旧样本做时间衰减；如果历史成功样本不足或过旧，会自动回退到模型 router
- 当某一维反思长期低命中时，会自动降低该维 reflection 注入强度；当 parallelism / worker 维长期高命中时，会在“样本稀疏但可并行”的场景里温和放宽默认保守动作，并把 guardrail 实际触发率一并写入 runtime telemetry
- 学习与诊断现在基于真实执行结果、运行时 telemetry 和用户反馈，而不是绑定额外的影子路由探测
- 极短问候/寒暄（如 `hi`、`你好`）会直接跳过执行约束抽取和 router 结构化调用，避免简单消息被前置 LLM 链路拖慢
- 对明显的请求类型（如显式辩论、多角色讨论、显式查询/检索、显式执行型任务），会先走零模型规则路由；只有模糊请求才进入一次小模型 router
- Router 只决定宏观路由（`tool_workflow` / `debate`）和执行倾向，不直接接管整个编排

当前实现增加了一层保守型 orchestration policy learning，用来优化“任务怎么拆、什么时候并行、什么时候不要再开 worker”，而不是去微调底层模型。

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
- 最近窗口护栏统计：`recent_guard_samples`、`recent_failure_rate`、`recent_bad_feedback_rate`
- 漂移统计：`recent_mean_reward`、`drift_score`

当前保守选择规则：

- 样本不足时，拆解默认 `analyze_then_execute`
- 调度默认更偏向串行，只有历史收益更好且风险更低时才偏向受限并行
- worker 使用默认更保守；启用策略学习后，如果历史数据不足或风险偏高，会拒绝继续创建/分发 worker
- 不是全局平均直接排序，而是先按 bucket 查局部历史，没命中再回落到全局
- 当前 bucket 只用稳定特征：`task_shape`、`has_media`、`independent_subtasks`
- bucket 不是只查一个最具体值，而是按“具体 → 一般”自动回退，例如先查 `task_shape + has_media + subtasks`，再查更一般的组合
- bucket 模板会优先选择有效样本量更强的组合，而不是机械地迷信最具体模板
- bucket 模板选择不只看样本量，还会优先保留“动作区分度”更高的模板，减少学到过于平均化的策略
- 排序不是只看 `mean_reward`，还会对 `failure_rate`、`retry_rate`、`dead_letter_rate`、`stalled_rate` 做惩罚
- 如果某个 action 的近期收益相对历史均值明显下滑，会额外计算 `drift_score`，把“旧经验很好、最近突然变差”的动作降权
- 历史 outcome 会做时间衰减，越旧的经验影响越小，不会长期主导当前策略
- `good/bad` 反馈不会直接覆盖 reward，而是先做时间衰减，再按 `feedback_confidence` 进行有限幅度 shaping
- 如果某个 action 在最近窗口内出现明显失败或负反馈，会触发 safeguard，优先退回更保守的动作
- `inspect_policy` 除了原有路由/反思命中率外，也会输出 `execution_style_guardrail_reduce_rate`、`parallelism_guardrail_soften_rate`、`worker_guardrail_soften_rate`
- 当前调度/worker 策略选择会附带 explain 摘要，便于在日志、debug 和 `@policy inspect` 时直接看出命中 bucket、分数与风险项
- 最终选择器是保守版 contextual bandit：对每个 action 用经验分减去和样本量相关的置信惩罚，优先选择更稳而不是更激进的动作
- 自动模式下，最小样本阈值和探索预算由系统内部护栏决定，不要求人工调参
- Reflection hint 只按当前稳定 bucket 回注，不做向量检索、不做本地训练、不引入重 retrieval

## 编排层当前护栏

- `TaskContract.allowed_tools` / `ExecutionPlan.steps[*].payload.allowed_tools` 会约束编排模型真正可见的 orchestration tools
- 常规回答默认走 `tool_workflow`，辩论请求默认走 `debate`
- 轻量 Router 只在合同冻结前做一次判定；如果失败、超时或返回无效结构，会直接退回默认合同推断
- runtime telemetry 会区分 `rule` / `model` / `skipped:*` / `fallback` 四类 router 来源，并单独统计 skip reason，便于判断是被门控跳过还是实际 fallback
- `reply_to_user` 必须单独收尾；如果仍有未完成的非 scheduler 子任务，编排层会拒绝提前结束
- runtime feedback 到 `RuntimeJob.state` 的投影已集中到 `runtime_jobs.py`，避免状态映射散落在渠道层
- `dispatch_team` 的阶段进度现在也会走规范化 runtime event，再由通道层按统一状态机渲染

### 当前策略算法（简化）

可以把当前实现理解成一层“自动保守优化器”，核心流程如下：

1. 先根据任务特征构造 bucket：
   - `task_shape`
   - `has_media`
   - `independent_subtasks`
2. 查询顺序优先走局部历史，而不是先看全局平均：
   - `task_shape + has_media + subtasks`
   - `task_shape + has_media`
   - `task_shape + subtasks`
   - `task_shape`
   - 全局 `global`
3. 如果多个局部 bucket 都有足够样本，不是简单选择“最具体”的那个，而是优先选择：
   - 动作区分度更高的模板
   - 同时有效样本量也更扎实的模板
4. 对候选 action 的分数会综合：
   - `mean_reward`
   - `failure_rate`
   - `retry_rate`
   - `dead_letter_rate`
   - `stalled_rate`
   - `feedback_score * feedback_confidence`
   - `drift_score`
5. 最后再减去一个和样本量相关的保守置信惩罚，避免“小样本高收益”过早主导策略。
6. 如果最近窗口显示某个 action 明显变差（比如失败率高、坏反馈集中、近期收益相对历史均值明显下滑），会直接触发 safeguard，优先退回更稳的动作。

因此它不是“固定规则表”，也不是“手工调参系统”，而是一个带护栏的自动学习策略层：随着真实任务 outcome、反馈和风险信号积累，拆解、调度和 worker 使用策略会逐步自适应收敛。

### Explain 输出

当前策略选择器会生成 explain 摘要，主要用于两类场景：

- 调试为什么这次选择了某个 scheduling / worker action
- 观察系统是否因为最近漂移、失败或负反馈而主动转保守

explain 中会包含这类信息：

- 命中的 `bucket`
- 被选中的 `action`
- 当前综合 `score`
- `mean_reward`
- `recent_mean_reward`
- `drift_score`
- `failure_rate`
- `effective_samples`

这让 `@policy inspect` 和运行时日志不再只告诉你“选了什么”，而是能顺便解释“为什么现在这么选”。

人工纠偏命令：

```text
@policy feedback latest good 拆分合理
@policy feedback flow-abc bad 并行导致多次重试
@policy inspect
@policy inspect scheduling
@policy inspect flow-abc
```

- 推荐优先使用显式 `flow_id`；`latest` 仅作为兼容路径保留
- `@policy feedback latest ...` 只会在最近运行记录唯一时自动绑定；如果最近有多个 flow，会要求显式指定 `flow_id`
- `@policy inspect <flow_id>` 会直接返回该运行的 runtime flow 视图
- 如果当前会话还没有最近任务，会直接返回明确错误，不会进入正常对话编排
- `@policy inspect` 会返回当前 policy 聚合摘要，`@policy inspect scheduling` / `worker` / `decomposition` 可查看对应决策维度
- `@policy inspect` 现在还会附带 `Routing Telemetry`，包括 router 平均延迟、`fallback_rate`、`skipped_rate`、`model_route_rate`、`skip_breakdown`、reflection match/override rate，以及按 `route_mode` 聚合的 reward
- CLI 的 `status` 也会显示同一份 router telemetry 摘要，便于快速判断当前是“被门控跳过”还是“真正 fallback”

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

完整模板见 `config.json.example` 和 `scheduled_tasks.json.example`；其中 router 默认超时已与当前实现保持一致（`3.0` 秒）。

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
      "enabled": false,
      "app_id": "cli_xxx",
      "app_secret": "xxx",
      "encrypt_key": "",
      "verification_token": "",
      "group_policy": "mention",
      "reply_mode": "chat",
      "react_emoji": "THUMBSUP",
      "media_dir": "",
      "streaming": false
    },
    "weixin": {
      "enabled": false,
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
| `streaming` | `true` 启用模型生成期实时流式推送（边生成边 patch 同一张卡片） |

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
    "subtask_timeout": 60,
    "skill_route_timeout": 3.0,
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
    "interactive_session_max_age_seconds": 7200,
    "routing_enabled": true,
    "routing_model_name": "",
    "routing_timeout": 3.0,
    "reflection_enabled": true,
    "reflection_max_hints": 3,
    "debug_runtime_feedback": false,
    "policy_learning_enabled": true,
    "policy_learning_min_samples": 0,
    "policy_learning_explore_ratio": -1.0
  }
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | 600 | 单任务硬超时（秒） |
| `subtask_timeout` | 60 | 子任务（worker/subagent）硬超时（秒） |
| `skill_route_timeout` | 3.0 | 技能路由判定超时（秒） |
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
| `interactive_session_max_age_seconds` | 7200 | 交互式 session（如 Claude 常驻进程）的最大存活时间（秒） |
| `routing_enabled` | true | 是否启用轻量路由层（仅影响 DAG 编排路径） |
| `routing_model_name` | `""` | 路由使用的模型名；为空时回退到当前会话同一模型 |
| `routing_timeout` | 3.0 | 模型 router 超时（秒）；超时后立即回退默认合同 |
| `reflection_enabled` | true | 是否启用历史反思回注 |
| `reflection_max_hints` | 3 | 每次最多注入的历史反思条数 |
| `debug_runtime_feedback` | false | 是否额外发送开发态调试卡片，展示编排 explain 摘要 |
| `policy_learning_enabled` | true | 是否启用策略自动学习（可用作回滚开关） |
| `policy_learning_min_samples` | 0 | 动作最小样本阈值；`0` 表示自动模式 |
| `policy_learning_explore_ratio` | -1.0 | 探索预算；`-1.0` 表示自动模式 |

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
      "headers": {},
      "transport": "streamable_http",
      "group_name": "map_services",
      "active": false
    }
  }
}
```

支持 `stdio`（子进程）和 `http`（HTTP/SSE）两种 MCP 传输方式。

BabyBot 会为每个 MCP 统一分配本地 artifact 根目录：`~/.babybot/workspace/output/mcp/<server-name>`。

- `stdio` MCP：默认以该目录作为工作目录，并注入 `BABYBOT_MCP_SERVER_NAME`、`BABYBOT_MCP_WORKSPACE_ROOT`、`BABYBOT_MCP_ARTIFACT_ROOT`
- `http` MCP：会在请求头里附带 `X-Babybot-Mcp-Server`、`X-Babybot-Workspace-Root`、`X-Babybot-Artifact-Root`
- 如果你在配置中显式提供 `cwd`、`env` 或 `headers`，会在默认值基础上合并；同名键以你的配置为准

这样可以把大多数 MCP 的临时产物统一收敛到 BabyBot workspace 下，而不是散落到当前项目目录。

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
- `web`：网页抓取与搜索（默认激活）
- `channel_*`：由通道在运行时注册的发送类工具

**web 工具组**

| 工具 | 说明 |
|------|------|
| `web_fetch` | 抓取指定 URL 内容，支持 `markdown`（默认）/ `text` / `html` 三种输出格式，自动提取正文并截断 |
| `web_search` | 通过 Tavily API 搜索互联网，支持 `basic` / `advanced` 深度和 `general` / `news` 话题，返回带 AI 摘要的结构化结果 |

`web_fetch` 无需配置即可使用；`web_search` 需在配置中填写 Tavily API key：

```json
{
  "web": {
    "tavily_api_key": "tvly-xxxxxxxx"
  }
}
```

Tavily 免费账户每月提供 1000 次搜索额度，在 [tavily.com](https://tavily.com) 注册获取。

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
3. **Hybrid Memory** — 额外维护软记忆、临时记忆；静态助手画像单独放在 `~/.babybot/assistant_profile.md`：
   - 静态画像：`assistant_profile.md`，直接注入 system prompt，不参与 memory 检索
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

当前进度反馈已切到统一生命周期卡片设计，ACK、进度动画和最终回复全程复用同一张消息卡片：

**消息流（以飞书为例）：**

```
[卡片] ⠋ 收到，正在处理…              ← ACK 即建卡
[patch] ⠙ 正在调用 text-to-image…     ← 子任务开始，卡片内容更新
[patch] ⠹ 正在调用 text-to-image…     ← 每 3 秒轮换 spinner 帧
[patch] ⠸ 正在调用 text-to-image…     ← 动画持续直至任务结束
[patch] ✅ 子任务已完成               ← 终态事件
[patch] 最终回复内容                   ← reply_to_user 原地替换卡片
[新消] 📎 图片/文件（如有）           ← 媒体单独发送
```

关键设计：

- 支持卡片 patch 的通道（飞书）全程只有一张卡片，不产生多条消息
- 不支持卡片的通道（微信等）自动退化为纯文本 ACK + `send_response`，行为不受影响
- 进度文案不再暴露内部 task ID（如 `task task_2_1527df succeeded`），改为自然语言（如"正在调用 weather"）
- spinner 帧序列（Braille 10帧，视觉连续旋转）：⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
- `started` 事件加入转发，子任务开始即可见，避免长任务执行期间卡片静止
- `RuntimeFeedbackEvent` 的 message 字段不含 `description`（系统提示词），不泄露内部信息

**通道扩展接入流式：**

新 channel 只需实现三个方法即可获得完整统一卡片体验：

```python
class MyChannel(BaseChannel):
    @property
    def supports_streaming(self) -> bool:
        return bool(self.config.streaming)

    async def create_stream_message(self, chat_id, text, *, sender_id=None, metadata=None) -> str | None:
        # 创建可 patch 的消息，返回消息 ID
        ...

    async def patch_stream_message(self, message_id: str, text: str) -> bool:
        # 更新已有消息内容，成功返回 True
        ...
```

其他反馈特性：

- MessageBus 先按 runtime identity 去重；对同一 task 作用域内重复出现的相同展示文案，也会继续折叠
- 编排层新增「结果充分即收敛」规则：专业 Skill 已成功返回结果后，不再用通用 web 工具重复验证
- 资源目录按专业（skill/mcp）和通用（tool_group）两层展示，帮助模型做出更好的资源选择
- 如果消息通道先超时，但任务已经创建了持久化作业，超时提示会附带 `job_id`，方便后续用 `@job status ...` 查询

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
│   ├── orchestration_router.py  # 轻量路由判定与门控
│   ├── orchestration_policy.py  # 保守策略选择器
│   ├── orchestration_policy_store.py # 策略/路由 telemetry 持久化
│   ├── orchestration_policy_types.py # 策略决策/结果类型
│   ├── task_contract.py         # 任务合同模型与归一化入口
│   ├── execution_plan.py        # 合同展开后的执行计划
│   ├── runtime_jobs.py          # 长任务状态模型
│   ├── runtime_job_store.py     # 长任务 SQLite 存储
│   ├── feedback_events.py       # 运行时反馈规范化与渲染
│   ├── runtime_feedback_commands.py # @policy / @job / @session 命令解析
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
│   ├── builtin_tools/           # 内置工具定义（worker/scheduler/code/observability/web）
│   ├── worker.py                # Worker 执行器工厂
│   ├── model_gateway.py         # OpenAI 兼容网关
│   ├── message_bus.py           # 异步消息总线
│   ├── heartbeat.py             # 心跳 / 超时监控
│   ├── context.py               # Tape 长期记忆 + TapeStore
│   ├── mcp_runtime.py           # MCP stdio/HTTP 客户端
│   ├── sqlite_utils.py          # SQLite 通用工具（WAL、upsert 等）
│   ├── cron.py                  # 定时任务调度
│   ├── task_evaluator.py        # 任务评估与反思总结
│   ├── interactive_sessions/    # 交互式 session 管理与 backend
│   │   ├── manager.py           # InteractiveSessionManager
│   │   ├── protocols.py         # InteractiveBackend protocol
│   │   ├── types.py             # session 数据类型
│   │   └── backends/
│   │       └── claude.py        # ClaudeInteractiveBackend（常驻子进程）
│   ├── agent_kernel/            # 轻量执行内核
│   │   ├── executor.py          # SingleAgentExecutor
│   │   ├── dynamic_orchestrator.py # DynamicOrchestrator（子任务编排 + team dispatch）
│   │   ├── team.py              # TeamRunner（多 Agent 辩论 / 协作）
│   │   ├── protocols.py         # ExecutorPort protocol
│   │   ├── types.py             # TaskContract / ExecutionContext / TaskResult
│   │   ├── execution_constraints.py # 运行约束与团队执行策略
│   │   ├── loop_guard.py        # 循环/重复工具调用防护
│   │   ├── errors.py            # 运行时错误分类与恢复提示
│   │   ├── executors/           # 可插拔执行后端
│   │   │   ├── __init__.py      # ExecutorRegistry
│   │   │   └── claude_code.py   # ClaudeCodeExecutor
│   │   ├── dag_ports.py         # ResourceBridgeExecutor
│   │   ├── context.py           # ExecutionContext 实现
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
│       ├── feishu_tools.py      # 飞书发送/卡片工具
│       ├── weixin.py            # 微信通道实现
│       └── weixin_tools.py      # 微信发送工具
├── skills/                      # 技能目录
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
- **统一 Skill 模型** — SKILL.md 同时支持工具路由和���队角色定义，`dispatch_team` 通过 `skill_id` 引用 prompt-only skill
- **分层记忆** — Tape + Hybrid Memory + Hot/Warm/Cold 三层上下文
- **多模态** — 图片从通道到 LLM 全链路支持（延迟 base64 编码）
- **MCP 支持** — stdio / HTTP 两种传输，动态注册工具
- **内置 Web 工具** — `web_fetch`（URL 抓取）和 `web_search`（Tavily API 搜索）开箱即用，归属 `web` 工具组
- **技能路由** — 技能 prompt、工具脚本、lease 与 worker prompt 解耦
- **技能热加载** — 创建或更新技能后调用 `reload_skill` 即时生效，无需重启进程
- **宿主 Python fallback** — 技能脚本可在独立本地 Python 环境运行，并支持多级回退
- **运行时可观测性** — flow / chat context 只读观测工具，稳定分段输出
- **统一生命周期卡片** — ACK、进度动画（spinner 轮换）、最终回复全程复用同一张消息卡片；不支持卡片的通道自动退化为纯文本
- **编排收敛优化** — 专业 Skill 优先于通用工具组；结果充分即收敛，不重复执行同语义子任务
- **工具租约** — ToolLease 最小权限策略，技能间 UNION 语义合并
- **主/子 Agent 边界清晰** — 只有主 agent 负责通道发送，子 agent 只返回结果与产物路径
- **可配置执行轮数** — orchestrator / worker 的最大步数可配置
- **并发控制** — 全局 + 单聊信号量，心跳空闲超时检测
- **通道扩展** — 实现 `BaseChannel.supports_streaming` + `create/patch_stream_message` 即可接入新平台并获得完整卡片体验（已支持飞书、微信）

## License

MIT
