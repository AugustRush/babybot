# Interactive CLI Session Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lazily-started, chat-scoped interactive CLI sessions controlled by `@session` commands, with Claude as the first backend and a reusable path for future backends such as Codex.

**Architecture:** Route `@session` commands and active interactive-chat traffic through a new `InteractiveSessionManager` owned by `OrchestratorAgent`. Keep backend-specific process handling behind a small `InteractiveBackend` protocol so Claude is the first implementation, not a one-off special case.

**Tech Stack:** Python 3.10+, `asyncio`, dataclasses, existing `OrchestratorAgent`, subprocess-backed CLI adapters, `pytest`

---

### Task 1: Add Config Support For Interactive Sessions

**Files:**
- Modify: `babybot/config.py`
- Modify: `config.json.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_interactive_session_max_age_loaded_from_system(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({
            "model": {"api_key": "test"},
            "system": {"interactive_session_max_age_seconds": 1800},
        }),
        encoding="utf-8",
    )

    cfg = Config(str(config_path))

    assert cfg.system.interactive_session_max_age_seconds == 1800
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_config.py -k interactive_session_max_age -v`
Expected: FAIL because `SystemConfig` does not define `interactive_session_max_age_seconds`

- [ ] **Step 3: Add the minimal config field**

```python
@dataclass
class SystemConfig:
    ...
    interactive_session_max_age_seconds: int = 7200
```

Also thread the field through:

- `Config.__init__`
- `_bootstrap_config_file()`
- `to_dict()`
- `config.json.example`

- [ ] **Step 4: Run the focused config test**

Run: `pytest tests/test_config.py -k interactive_session_max_age -v`
Expected: PASS

- [ ] **Step 5: Run the full config suite**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/config.py config.json.example tests/test_config.py
git commit -m "feat: add interactive session config"
```

### Task 2: Introduce The Interactive Session Abstractions And Manager

**Files:**
- Create: `babybot/interactive_sessions/__init__.py`
- Create: `babybot/interactive_sessions/types.py`
- Create: `babybot/interactive_sessions/protocols.py`
- Create: `babybot/interactive_sessions/manager.py`
- Test: `tests/test_interactive_session_manager.py`

- [ ] **Step 1: Write the failing manager tests**

```python
@pytest.mark.asyncio
async def test_manager_reuses_existing_chat_session():
    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=7200,
    )

    first = await manager.start(chat_key="feishu:c1", backend_name="claude")
    second = await manager.start(chat_key="feishu:c1", backend_name="claude")

    assert first.session_id == second.session_id
    assert backend.start_calls == 1


@pytest.mark.asyncio
async def test_manager_stops_expired_session_before_send():
    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=1,
        time_fn=lambda: 100.0,
    )
    await manager.start(chat_key="feishu:c1", backend_name="claude")
    manager._time_fn = lambda: 102.0

    reply = await manager.send("feishu:c1", "hello")

    assert "超时关闭" in reply.text
    assert backend.stop_calls == 1
```

- [ ] **Step 2: Run the new manager tests to verify they fail**

Run: `pytest tests/test_interactive_session_manager.py -v`
Expected: FAIL because the interactive session package and manager do not exist yet

- [ ] **Step 3: Implement the minimal manager and data types**

```python
@dataclass
class InteractiveSession:
    chat_key: str
    backend_name: str
    started_at: float
    last_active_at: float
    handle: Any


class InteractiveSessionManager:
    async def start(self, chat_key: str, backend_name: str) -> InteractiveSession: ...
    async def send(self, chat_key: str, message: str) -> InteractiveReply: ...
    async def stop(self, chat_key: str, reason: str = "user_stop") -> bool: ...
    def status(self, chat_key: str) -> InteractiveSessionStatus | None: ...
```

Implementation notes:

- keep one session per `chat_key`
- reject switching backends without explicit stop
- update `last_active_at` on successful sends
- stop and delete expired sessions before forwarding
- guard `start/send/stop` with per-chat `asyncio.Lock`

- [ ] **Step 4: Run the manager test file**

Run: `pytest tests/test_interactive_session_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/interactive_sessions tests/test_interactive_session_manager.py
git commit -m "feat: add interactive session manager"
```

### Task 3: Implement The Claude Interactive Backend Behind The Protocol

**Files:**
- Create: `babybot/interactive_sessions/backends/__init__.py`
- Create: `babybot/interactive_sessions/backends/claude.py`
- Test: `tests/test_interactive_claude_backend.py`

- [ ] **Step 1: Write the failing Claude backend tests**

```python
@pytest.mark.asyncio
async def test_claude_backend_start_uses_isolated_environment(tmp_path):
    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )

    with patch("asyncio.create_subprocess_exec", return_value=fake_proc()) as mock_exec:
        await backend.start(chat_key="feishu:c1")

    env = mock_exec.call_args.kwargs["env"]
    assert env["HOME"].startswith(str(tmp_path))


@pytest.mark.asyncio
async def test_claude_backend_send_returns_backend_reply():
    backend = ClaudeInteractiveBackend(...)
    session = await backend.start(chat_key="feishu:c1")

    reply = await backend.send(session, "/models")

    assert reply.text
```

- [ ] **Step 2: Run the backend tests to verify they fail**

Run: `pytest tests/test_interactive_claude_backend.py -v`
Expected: FAIL because the backend implementation does not exist yet

- [ ] **Step 3: Implement the minimal Claude backend**

```python
class ClaudeInteractiveBackend(InteractiveBackend):
    async def start(self, chat_key: str) -> Any:
        env = self._build_isolated_env(chat_key)
        proc = await asyncio.create_subprocess_exec(
            self._claude_bin,
            ...,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self._workspace_root),
        )
        return ClaudeSessionHandle(...)
```

Implementation boundaries:

- isolate HOME/state/tmp directories under a backend-owned runtime root
- keep transport details private to the backend
- expose only `start/send/stop/status`
- return structured reply text to the manager
- make real Claude CLI integration opt-in, not part of default test flow

- [ ] **Step 4: Run the backend test file**

Run: `pytest tests/test_interactive_claude_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/interactive_sessions/backends tests/test_interactive_claude_backend.py
git commit -m "feat: add claude interactive backend"
```

### Task 4: Integrate Session Routing Into OrchestratorAgent

**Files:**
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`

- [ ] **Step 1: Write the failing orchestrator routing tests**

```python
@pytest.mark.asyncio
async def test_process_task_starts_interactive_session_on_command():
    agent = make_agent_with_session_manager()

    response = await agent.process_task("@session start claude", chat_key="feishu:c1")

    assert "Claude 会话已启动" in response.text


@pytest.mark.asyncio
async def test_process_task_routes_active_chat_messages_to_session_backend():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task("帮我看看 /models", chat_key="feishu:c1")

    assert response.text == "backend reply"
    assert agent.gateway_calls == 0
```

- [ ] **Step 2: Run the new orchestrator routing tests to verify they fail**

Run: `pytest tests/test_orchestrator_interactive_sessions.py -v`
Expected: FAIL because `OrchestratorAgent` does not parse `@session` commands or route to the manager

- [ ] **Step 3: Implement the routing hooks in `OrchestratorAgent`**

```python
async def process_task(...):
    if self._interactive_sessions is not None and chat_key:
        control = self._parse_interactive_session_command(user_input)
        if control is not None:
            return await self._handle_interactive_session_command(...)
        if self._interactive_sessions.has_active_session(chat_key):
            return await self._handle_interactive_session_message(...)
    ...
```

Integration requirements:

- initialize the session manager once in `__init__`
- expose `@session start/status/stop`
- keep non-session chats on the existing DAG path
- stop all active sessions from `reset()`
- include interactive session info in `get_status()`

- [ ] **Step 4: Run the new orchestrator routing tests**

Run: `pytest tests/test_orchestrator_interactive_sessions.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing orchestrator routing suite**

Run: `pytest tests/test_orchestrator_routing.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestrator.py tests/test_orchestrator_interactive_sessions.py
git commit -m "feat: route chats through interactive sessions"
```

### Task 5: Add Stop, Status, And Cleanup Coverage Across The End-To-End Path

**Files:**
- Modify: `tests/test_orchestrator_interactive_sessions.py`
- Modify: `tests/test_config.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing stop/status/cleanup tests**

```python
@pytest.mark.asyncio
async def test_session_stop_command_closes_active_session():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task("@session stop", chat_key="feishu:c1")

    assert "已关闭" in response.text
    assert agent.session_manager.stop_calls == 1


def test_get_status_includes_interactive_session_summary():
    agent = make_agent_with_session_manager(active_session=True)

    status = agent.get_status()

    assert "interactive_sessions" in status
```

- [ ] **Step 2: Run the stop/status tests to verify they fail**

Run: `pytest tests/test_orchestrator_interactive_sessions.py -k "stop or status or cleanup" -v`
Expected: FAIL until cleanup/status behavior is fully wired

- [ ] **Step 3: Implement the minimal cleanup and docs**

Update:

- `OrchestratorAgent.reset()` to close active sessions
- `OrchestratorAgent.get_status()` to include session summary
- `README.md` with `@session start claude`, `@session status`, `@session stop`

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_orchestrator_interactive_sessions.py -k "stop or status or cleanup" -v`
Expected: PASS

- [ ] **Step 5: Run the relevant combined suite**

Run: `pytest tests/test_config.py tests/test_interactive_session_manager.py tests/test_interactive_claude_backend.py tests/test_orchestrator_interactive_sessions.py tests/test_orchestrator_routing.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_config.py tests/test_orchestrator_interactive_sessions.py
git commit -m "docs: document interactive session commands"
```

### Task 6: Final Regression Verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run the targeted regression suite**

Run: `pytest tests/test_config.py tests/test_orchestrator_routing.py tests/test_interactive_session_manager.py tests/test_interactive_claude_backend.py tests/test_orchestrator_interactive_sessions.py -q`
Expected: PASS

- [ ] **Step 2: Run the full default test suite**

Run: `pytest -q`
Expected: PASS except for any pre-existing opt-in or environment-dependent Claude CLI integration cases that remain intentionally outside the default path

- [ ] **Step 3: Review the diff**

Run: `git diff --stat`
Expected: only the planned files changed

- [ ] **Step 4: Final commit if verification required follow-up edits**

```bash
git add babybot README.md config.json.example tests docs/superpowers/plans/2026-03-24-interactive-cli-session-mode.md
git commit -m "feat: add interactive cli session mode"
```
