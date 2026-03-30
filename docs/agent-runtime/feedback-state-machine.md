# Feedback State Machine

通道层只渲染规范化后的 `RuntimeFeedbackEvent`，不直接解释业务文案。

权威状态：

- `queued`
- `planning`
- `running`
- `waiting_tool`
- `waiting_user`
- `repairing`
- `completed`
- `failed`
- `cancelled`

事件最小字段：

- `job_id`
- `flow_id`
- `task_id`
- `state`
- `stage`
- `message`
- `error`
- `progress`

常见 stage：

- `job`
- `task`
- `interactive_session`
- `debate`

渲染规则：

1. 去重键必须基于身份字段，不只看 `message`
2. `failed` 即使没有 `error` 文本，也必须按失败渲染
3. 最终回复与中间进度是两类不同输出，不能共用字符串比较逻辑
4. 团队讨论/辩论的阶段反馈也必须先规范化为 `RuntimeFeedbackEvent`，不能绕过状态机直接拼渠道文案
