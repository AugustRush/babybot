# Interactive Session Streaming Events Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 为 BabyBot 的 resident interactive session 增加统一的增量输出事件流，并让 CLI、飞书、微信复用同一套事件语义与退化策略。

**Architecture:** 保持 `RuntimeFeedbackEvent` 只负责阶段状态，新增独立的 interactive 输出事件模型，Claude resident backend 只产出标准化增量事件，`OrchestratorAgent` / `MessageBus` / CLI / channel 按同一回调链消费。飞书优先 patch 同一条 interactive 卡片，微信按能力做节流聚合退化，CLI 直接增量打印，不额外引入重型 pubsub。

**Tech Stack:** Python 3.11、`asyncio`、现有 `InteractiveSessionManager` / `OrchestratorAgent` / `MessageBus` / Feishu / Weixin / `pytest`

---

### Task 1: 定义统一 interactive 输出事件模型

**Files:**
- Modify: `babybot/interactive_sessions/types.py`
- Test: `tests/test_interactive_claude_backend.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [x] **Step 1: 写失败测试，锁定最小事件集合**

在测试中先声明 interactive 输出事件的最小语义，至少覆盖：
- `message_start`
- `message_delta`
- `message_complete`
- `tool_status`
- `session_status`
- `error`

并断言事件对象具备稳定字段（如 `event`, `text`, `delta`, `metadata`）。

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_interactive_claude_backend.py tests/test_orchestrator_interactive_sessions.py -k interactive_output`
Expected: FAIL，因当前尚无统一 interactive 输出事件模型。

- [x] **Step 3: 写最小实现**

在 `babybot/interactive_sessions/types.py` 中新增轻量 `InteractiveOutputEvent`，不与 `RuntimeFeedbackEvent` 混用，也不引入新的复杂总线抽象。

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_interactive_claude_backend.py tests/test_orchestrator_interactive_sessions.py -k interactive_output`
Expected: PASS

### Task 2: 让 Claude resident backend 产出增量事件

**Files:**
- Modify: `babybot/interactive_sessions/backends/claude.py`
- Test: `tests/test_interactive_claude_backend.py`

- [x] **Step 1: 写失败测试，锁定 stream-json 到事件的映射**

补测试覆盖：
- 一轮输出开始时发送 `message_start`
- partial / assistant message 文本变化只发送增量 `message_delta`
- 完成时发送 `message_complete`
- Claude 返回错误时发送 `error`
- 最终仍返回完整 `InteractiveReply`

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_interactive_claude_backend.py`
Expected: FAIL，因当前 backend 只聚合最终文本，不透传增量事件。

- [x] **Step 3: 写最小实现**

给 `ClaudeInteractiveBackend.send()` 增加可选 output event callback，在 `_read_turn_output()` 里解析 `stream-json` 输出并边读边发事件；同时维持现有 resident process、超时、报错和最终 reply 行为。

- [x] **Step 4: 运行 backend 测试**

Run: `pytest -q tests/test_interactive_claude_backend.py`
Expected: PASS

### Task 3: 透传 interactive 输出事件到编排层

**Files:**
- Modify: `babybot/interactive_sessions/manager.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_interactive_session_manager.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [x] **Step 1: 写失败测试，锁定 orchestrator 可接收 interactive 输出回调**

补测试覆盖：
- active session 消息会透传 interactive 输出事件
- session 失败仍会 stop 并回退 DAG
- runtime 阶段反馈与 interactive 文本 delta 不混流

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: FAIL，因当前 session 路径还没有 output event callback。

- [x] **Step 3: 写最小实现**

在 `InteractiveSessionManager.send()`、`OrchestratorAgent.process_task()` 与 interactive session 分支中增加可选 `interactive_output_callback`，只透传事件，不改变现有 `TaskResponse` 与 runtime job 反馈合同。

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: PASS

### Task 4: 用同一事件流驱动 CLI / 飞书 / 微信

**Files:**
- Modify: `babybot/message_bus.py`
- Modify: `babybot/cli.py`
- Modify: `babybot/channels/weixin.py`
- Test: `tests/test_message_bus_streaming.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_weixin_channel.py`

- [x] **Step 1: 写失败测试，锁定三种消费策略**

补测试覆盖：
- CLI interactive session 过程中按 delta 增量打印，而不是只等最终结果
- 飞书 interactive 文本复用现有 interactive stream card patch
- 飞书 runtime progress 仍然走 `post`
- 微信接收同一事件流，但退化为节流聚合文本发送，不要求 patch 卡片

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_message_bus_streaming.py tests/test_cli.py tests/test_weixin_channel.py`
Expected: FAIL，因当前 interactive session 还未接入统一增量事件流。

- [x] **Step 3: 写最小实现**

在 `MessageBus` 中新增 interactive 输出事件回调链：
- CLI：delta 直接 stdout，完成时收尾换行
- Feishu：优先复用 `create_stream_message` / `patch_stream_message`
- Weixin：按能力做轻量节流聚合后发送文本，不额外引入状态机分叉

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_message_bus_streaming.py tests/test_cli.py tests/test_weixin_channel.py`
Expected: PASS

### Task 5: 更新文档并完成回归验证

**Files:**
- Modify: `README.md`
- Modify: `docs/agent-runtime/feedback-state-machine.md`
- Modify: `docs/agent-runtime/interaction-contract.md`
- Verify: `tests/test_interactive_claude_backend.py`
- Verify: `tests/test_interactive_session_manager.py`
- Verify: `tests/test_orchestrator_interactive_sessions.py`
- Verify: `tests/test_message_bus_streaming.py`
- Verify: `tests/test_cli.py`
- Verify: `tests/test_weixin_channel.py`

- [x] **Step 1: 更新文档**

更新 README 与运行时文档，明确：
- `RuntimeFeedbackEvent` 只负责阶段反馈
- interactive session 正文增量走独立输出事件流
- CLI / 飞书 / 微信共享同一事件语义，但按通道能力降级渲染

- [x] **Step 2: 运行聚焦测试**

Run: `pytest -q tests/test_interactive_claude_backend.py tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py tests/test_message_bus_streaming.py tests/test_cli.py tests/test_weixin_channel.py`
Expected: PASS

- [x] **Step 3: 运行静态检查**

Run: `ruff check babybot/interactive_sessions/types.py babybot/interactive_sessions/backends/claude.py babybot/interactive_sessions/manager.py babybot/orchestrator.py babybot/message_bus.py babybot/cli.py babybot/channels/weixin.py tests/test_interactive_claude_backend.py tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py tests/test_message_bus_streaming.py tests/test_cli.py tests/test_weixin_channel.py`
Expected: PASS

- [x] **Step 4: 运行全量测试**

Run: `pytest -q`
Expected: PASS

- [x] **Step 5: 最终提交**

```bash
git add README.md docs/agent-runtime/feedback-state-machine.md docs/agent-runtime/interaction-contract.md docs/superpowers/plans/2026-03-31-interactive-session-streaming-events.md babybot tests
git commit -m "feat: unify interactive session streaming events"
```
