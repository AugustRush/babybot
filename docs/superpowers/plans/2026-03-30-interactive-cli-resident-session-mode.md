# Interactive CLI Resident Session Mode Implementation Plan

**Status:** Completed. Resident Claude session mode is the current interactive-session baseline.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将 BabyBot 的交互式 `@session` 路径从“每条消息重新 `claude --resume`”升级为“每个会话一个常驻本地 Claude 进程”。

**Architecture:** 保留 `InteractiveSessionManager` 作为按 `chat_key` 的会话注册表，不改 `OrchestratorAgent` 的外部入口。将 `ClaudeInteractiveBackend` 改为常驻子进程 backend，使用 stdin/stdout 进行 turn 级交互，并把真实运行状态暴露给 `status` / CLI 展示。

**Tech Stack:** Python 3.11、`asyncio` 子进程、现有 `InteractiveSessionManager` / `OrchestratorAgent` / `pytest`

---

### Task 1: 扩展交互会话数据模型

**Files:**
- Modify: `babybot/interactive_sessions/types.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [x] **Step 1: 写失败测试，锁定状态需要暴露真实运行态**

在 `tests/test_orchestrator_interactive_sessions.py` 增加断言：`@session status` 至少包含 `backend` 扩展状态，例如 `pid` / `alive` 或 `mode`。

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_orchestrator_interactive_sessions.py -k session_status`
Expected: FAIL，因当前状态文案只输出 `session_id`

- [x] **Step 3: 最小修改数据模型**

在 `InteractiveSession` / `InteractiveSessionStatus` 中补充常驻会话运行态需要的字段，至少支持：
- `mode`
- `runtime_root`
- `backend_status`

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_orchestrator_interactive_sessions.py -k session_status`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add babybot/interactive_sessions/types.py tests/test_orchestrator_interactive_sessions.py
git commit -m "feat: expose interactive session runtime status"
```

### Task 2: 用 TDD 把 Claude backend 改成常驻进程

**Files:**
- Modify: `babybot/interactive_sessions/backends/claude.py`
- Test: `tests/test_interactive_claude_backend.py`

- [x] **Step 1: 写失败测试，锁定常驻进程行为**

在 `tests/test_interactive_claude_backend.py` 增加测试：
- `start()` 只拉起一次子进程并保存 `pid`
- 连续两次 `send()` 不重复调用 `asyncio.create_subprocess_exec`
- `stop()` 会终止常驻进程

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_interactive_claude_backend.py`
Expected: FAIL，因当前 `send()` 每次都会重新拉起进程

- [x] **Step 3: 写最小实现**

在 `ClaudeInteractiveBackend` 中：
- `start()` 启动常驻 Claude 进程并创建 handle
- `send()` 改为向存活进程发送一轮输入并等待本轮输出
- `stop()` 负责回收进程和 runtime 目录
- `status()` 返回 `pid` / `alive` / `mode` / `runtime_root`

- [x] **Step 4: 运行 backend 测试**

Run: `pytest -q tests/test_interactive_claude_backend.py`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add babybot/interactive_sessions/backends/claude.py tests/test_interactive_claude_backend.py
git commit -m "feat: keep claude interactive sessions resident"
```

### Task 3: 接入会话管理与异常回退

**Files:**
- Modify: `babybot/interactive_sessions/manager.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_interactive_session_manager.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [x] **Step 1: 写失败测试，锁定真实进程失效后的回退**

补测试覆盖：
- 过期 session 停掉真实 backend
- backend 已死时，`process_task()` 回退 DAG
- `@session status` 能返回更丰富的 backend 状态

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: FAIL，因当前 manager/status 只处理轻量 handle

- [x] **Step 3: 最小实现**

在 manager / orchestrator 中补齐：
- 过期清理时调用真实 backend stop
- interactive send 失败时清理 session 并安全回退 DAG
- `@session status` 展示运行态而非只展示 `session_id`

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add babybot/interactive_sessions/manager.py babybot/orchestrator.py tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py
git commit -m "feat: harden resident interactive session lifecycle"
```

### Task 4: 更新 CLI 展示与最小操作体验

**Files:**
- Modify: `babybot/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: 写失败测试，锁定 CLI 状态展示**

在 `tests/test_cli.py` 增加断言：
- `status` 输出包含更具体的 interactive 状态摘要
- 输出不再只显示 chat_keys / active_count

- [x] **Step 2: 运行失败测试**

Run: `pytest -q tests/test_cli.py`
Expected: FAIL，因当前 CLI 没展示 backend 运行态

- [x] **Step 3: 最小实现**

在 `babybot/cli.py` 中增强 `status` 输出，使本地调试能看到：
- active session 数
- 当前 chat 的 backend
- 关键运行态摘要（例如 alive / mode）

- [x] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_cli.py`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add babybot/cli.py tests/test_cli.py
git commit -m "feat: surface resident session status in cli"
```

### Task 5: 文档、回归验证与最终提交

**Files:**
- Modify: `README.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_interactive_claude_backend.py`
- Test: `tests/test_interactive_session_manager.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [x] **Step 1: 更新 README**

更新 `README.md` 的 interactive session 章节，改写为：
- 当前实现已支持常驻本地会话
- 默认模式为 `same-dir`
- `@session status` 展示真实运行态

- [x] **Step 2: 运行聚焦测试**

Run: `pytest -q tests/test_cli.py tests/test_interactive_claude_backend.py tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: PASS

- [x] **Step 3: 运行全量测试**

Run: `pytest -q`
Expected: PASS（允许现有已知 warning）

- [x] **Step 4: 运行静态检查**

Run: `ruff check babybot/cli.py babybot/interactive_sessions/types.py babybot/interactive_sessions/manager.py babybot/interactive_sessions/backends/claude.py babybot/orchestrator.py tests/test_cli.py tests/test_interactive_claude_backend.py tests/test_interactive_session_manager.py tests/test_orchestrator_interactive_sessions.py`
Expected: PASS

- [x] **Step 5: 最终提交**

```bash
git add README.md babybot tests docs/superpowers/plans/2026-03-30-interactive-cli-resident-session-mode.md
git commit -m "feat: add resident interactive cli sessions"
```
