# Debate And Round Budget

多 Agent 协作的轮数与停止条件只从 `TaskContract` / `ExecutionPlan` 读取。

映射规则：

- 用户出现“一轮定胜负” → `round_budget=1`，`termination_rule=single_round`
- `ExecutionPlan.round_budget` 是团队执行时的直接输入
- `dispatch_team.max_rounds` 只能收紧预算，不能放宽合同预算
- 若 runtime 实际轮数与合同预算不一致，立即报错

预算字段：

- `round_budget`
- `max_agents`
- `max_total_seconds`
- `max_turn_seconds`
- `stopping_condition`

降级规则：

- 预算耗尽且策略为 `summarize_partial` 时，返回部分总结
- 预算耗尽且策略为 `raise_timeout` 时，抛出超时
