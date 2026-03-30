# Interaction Contract

`TaskContract` 是进入编排层前的唯一用户任务合同。权威字段：

- `chat_key`：会话标识
- `goal`：归一化后的任务目标
- `mode`：`answer` / `debate` / `interactive_session` / `job`
- `deliverable`：最终产物类型
- `round_budget`：允许的讨论轮数上限，未知时为 `null`
- `termination_rule`：如 `final_answer` / `single_round` / `round_budget`
- `allow_clarification`：是否允许继续追问用户
- `allowed_tools`：白名单工具
- `allowed_agents`：白名单 agent
- `metadata`：附加约束与 runtime 绑定信息

`ExecutionPlan` 是 `TaskContract` 的执行展开。权威字段：

- `plan_id`
- `contract`
- `steps`
- `round_budget`
- `stopping_condition`

规则：

1. `orchestrator.process_task()` 只构建一次 `TaskContract` 和 `ExecutionPlan`
2. 下游运行时只读取合同/计划，不再直接推断原始用户文案
3. 若 runtime 参数与 `TaskContract` 冲突，必须立即报错，不能静默放宽
