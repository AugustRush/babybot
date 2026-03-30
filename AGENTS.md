# Babybot Agent Map

在修改本仓库前，按需先看这些文档：

- `README.md`：总览、启动方式、现有模块入口
- `docs/agent-runtime/interaction-contract.md`：`TaskContract` 与 `ExecutionPlan` 的权威字段
- `docs/agent-runtime/long-running-jobs.md`：`JobState`、作业持久化与恢复
- `docs/agent-runtime/debate-and-round-budget.md`：讨论轮数、预算、停止条件
- `docs/agent-runtime/feedback-state-machine.md`：通道允许渲染的 runtime 反馈事件

入口文件：

- `babybot/orchestrator.py`：主编排入口
- `babybot/message_bus.py`：通道消息调度与反馈渲染
- `babybot/agent_kernel/dynamic_orchestrator.py`：子任务/团队编排
- `babybot/interactive_sessions/`：交互式会话管理与 backend
- `babybot/orchestration_policy_store.py`：策略决策/反馈持久化

阅读规则：

- 改执行输入或停止条件前，先读 `interaction-contract.md`
- 改超时、恢复、轮询、状态展示前，先读 `long-running-jobs.md`
- 改多 Agent 讨论、轮次、budget 前，先读 `debate-and-round-budget.md`
- 改消息进度文案、状态去重、通道渲染前，先读 `feedback-state-machine.md`
