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
