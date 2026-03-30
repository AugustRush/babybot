# Long Running Jobs

长任务统一映射为持久化 `JobRuntime`。权威状态只有：

- `queued`
- `planning`
- `running`
- `waiting_tool`
- `waiting_user`
- `repairing`
- `completed`
- `failed`
- `cancelled`

权威字段：

- `job_id`
- `chat_key`
- `goal`
- `plan_id`
- `state`
- `progress_message`
- `created_at`
- `updated_at`
- `result_text`
- `error`
- `metadata`

规则：

1. 进入长任务前必须先创建 `job_id`
2. 通道超时提示必须带上已持久化的 `job_id`
3. 后续状态查询、恢复、反馈优先使用显式 `job_id`，不要依赖“最近一次”
4. runtime feedback 会回写 `RuntimeJob.state`、`metadata.flow_id`、最近 stage/task 信息，避免作业状态只停留在外层开始/结束
5. 维护入口可以清理“缺少 `flow_id` 且超过保留期仍停在 active state”的 orphaned jobs，并报告 stale session / unmatched flow
