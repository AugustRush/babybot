# Lightweight Routing & Reflection Design

**Goal**

在不破坏现有 `Tape` / `HybridMemoryStore` / `ContextView` / runtime event 链路的前提下，为 BabyBot 增加一层轻量的上下文路由与异步反思闭环，提升任务智能、执行效率和自我修正能力。

**硬约束**

- 首要保证实时性
- 必须能在本地 CPU 环境稳定运行
- 不引入本地训练、向量库或常驻大模型
- 不影响现有 interactive session 消息流
- 模型可配置；未配置时回退到当前会话同模型
- 避免过度设计，只实现最小有效闭环

**Scope**

V1 只做三件事：

1. 在正常 DAG 编排前加一次轻量 `ContextRouter`
2. 在任务结束后后台运行 `TaskEvaluator`
3. 把结构化 reflection 写入轻量存储，并在下一次类似任务时回注入少量 hints

## Architecture

### 1. ContextRouter

`ContextRouter` 只在非 interactive-session 路径上执行一次结构化判定。

输入：
- 当前 `user_input`
- 最近 `Tape` 消息摘要
- 最近 anchor summary
- `build_context_view()` 的 hot / warm / cold 结果
- 最近 job / runtime 状态摘要

输出：
- `route_mode`: `answer` / `tool_workflow` / `debate` / `job`
- `need_clarification`: `bool`
- `execution_style`: `direct` / `retrieve_first` / `analyze_first` / `verify_before_finish`
- `parallelism_hint`: `serial` / `bounded_parallel`
- `worker_hint`: `allow` / `deny`
- `explain`: 短解释

规则：
- 严格单次模型判定
- 超时、异常、空输出时回退到现有合同/规则路径
- interactive session 不经过该层

### 2. Routing Context Snapshot

为了避免把整段历史喂给小模型，新增一个 context snapshot builder，把现有上下文层压缩为小而稳定的结构：

- recent user / assistant turns（少量）
- anchor summary
- hot / warm / cold summaries
- active job/runtime summary
- reflection hints（最多 3 条）

### 3. TaskEvaluator

每次 DAG 任务结束后异步执行，不阻塞首响。

输入：
- `TaskContract`
- `ExecutionPlan`
- runtime events
- final status / reward / retry / stalled / dead-letter 信息
- 是否触发用户纠正或失败收尾

输出：
- `TaskEvaluation`
  - `success`
  - `latency_ms`
  - `reward`
  - `retry_count`
  - `stalled_count`
  - `dead_letter_count`
  - `premature_finish_blocked`
  - `suggested_execution_style`
  - `suggested_parallelism`
  - `suggested_worker_gate`
  - `failure_pattern`

### 4. ReflectionStore

不引入向量检索，只做结构化、可过滤的 reflection 表。

每条 reflection 保存：
- `chat_key`
- `route_mode`
- `task_shape`
- `has_media`
- `independent_subtasks`
- `failure_pattern`
- `recommended_action`
- `confidence`
- `created_at`

读取时只按：
- 当前 bucket
- 最近时间窗口
- 最高 confidence

最多返回 3 条 hint。

### 5. Policy Layer

保留现有 `ConservativePolicySelector`，但把它降为第二层动作选择器：

- Router 决定宏观路线
- PolicySelector 决定局部动作（clarification / scheduling / worker / execution style）
- Reflection hints 作为 selector 的额外先验输入

## Integration Points

- `babybot/orchestrator.py`
  - 构建 routing snapshot
  - 调用 `ContextRouter`
  - 将 routing decision 注入 `TaskContract` / `ExecutionPlan` / `policy_hints`
  - 在任务结束后启动 evaluator 后台任务
- `babybot/task_contract.py`
  - 接收 routing decision 作为 override
- `babybot/execution_plan.py`
  - 记录 router 决策产物
- `babybot/orchestration_policy_store.py`
  - 新增 reflection 存储与查询
- `babybot/config.py`
  - 新增 routing model / timeout / reflection 开关配置

## Why Not Pure Multi-Armed Bandit

纯 MAB 不适合作为 V1 主内核：

- 当前问题是上下文相关决策，不是无上下文单臂选择
- 奖励是延迟到任务结束后的，不是单步即时奖励
- 还需要接现有上下文压缩与记忆体系

因此 V1 采用：
- 一次轻量小模型 routing
- 现有保守 selector 做局部动作选择
- 异步 reflection 做持续修正

这比纯 MAB 更贴合现有系统，也更省资源。

## Non-Goals

V1 不做：
- 本地模型训练
- 向量数据库
- 多轮 planner / critic 对话
- interactive session 路径重构
- 重写 Tape / Memory 系统
