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

当前补充语义：

- `metadata.execution_constraints`：执行约束抽取后的归一化结果
- `metadata.routing_decision`：轻量 Router 的结构化判定快照（可选）
- interactive session 的正文增量输出不写入 `TaskContract` / `RuntimeFeedbackEvent`，而是作为独立 interactive 输出事件透传到 CLI / channel
- Router 只允许在 `TaskContract` 冻结前覆写 `answer/debate` 路由与澄清倾向；冻结后下游运行时不得再次放宽

当前观测补充：

- `inspect_policy` 会输出 runtime routing telemetry，包括 `reflection_route_rate`、`execution_style_reflection_rate`、`parallelism_reflection_rate`、`worker_reflection_rate`
- 同时会输出 guardrail 实际触发率：`execution_style_guardrail_reduce_rate`、`parallelism_guardrail_soften_rate`、`worker_guardrail_soften_rate`
- 对被规则 / reflection / intent cache 直接命中的请求，后台可异步记录 shadow routing agreement，不改变本次 `TaskContract`，只增加 `shadow_routing_eval_rate`、`shadow_routing_agreement_rate` 观测
- 当某个 `intent cache` 桶的样本过旧，或最近 shadow agreement 持续偏低时，该桶会自动失效并回退模型路由
- 对 shadow agreement 偏低但仍值得观察的桶，系统只会按每 chat 的轻量预算做周期性 probe，而不是每次都继续影子评估
- 这些字段只做轻量统计与观测，不会改变 `TaskContract` 对下游 runtime 的硬约束

当前约束：

- 普通编排回答默认只允许 `dispatch_task` / `wait_for_tasks` / `get_task_result` / `reply_to_user`
- 辩论模式默认只允许 `dispatch_team` / `reply_to_user`
- `reply_to_user` 必须作为单独收尾动作，不能与其他 orchestration tool 混用

`ExecutionPlan` 是 `TaskContract` 的执行展开。权威字段：

- `plan_id`
- `contract`
- `steps`
- `round_budget`
- `stopping_condition`

当前步骤类型：

- `tool_workflow`：常规工具编排路径
- `debate`：多 Agent 讨论路径

规则：

1. `orchestrator.process_task()` 只构建一次 `TaskContract` 和 `ExecutionPlan`
2. 下游运行时只读取合同/计划，不再直接推断原始用户文案
3. 轻量 Router 若返回无效结构、超时或失败，必须回退默认合同推断，不能阻塞主流程
4. 若 runtime 参数与 `TaskContract` 冲突，必须立即报错，不能静默放宽
5. `DynamicOrchestrator` 发给模型的 orchestration tool 集，必须受 `TaskContract.allowed_tools` / `ExecutionPlan.steps[*].payload.allowed_tools` 约束
