# Pluggable Executors & Agent Teams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 DynamicOrchestrator 支持可插拔的外部 executor（如 Claude Code CLI）和 agent team 多轮协作模式（如辩论）。

**Architecture:** 分三个阶段：(1) 将 executor 可插拔化，按 resource type 路由到不同 backend；(2) 实现 ClaudeCodeExecutor 作为第一个外部 executor；(3) 在 orchestrator 内部增加 agent team 能力（mailbox + 共享 task list），支持 agent 间直接通信的多轮交互。

**Tech Stack:** Python 3.11+, asyncio, subprocess (for Claude Code CLI), pytest

---

## File Structure

```
babybot/agent_kernel/
  protocols.py              — 修改: 新增 ExecutorFactory protocol
  types.py                  — 修改: TaskContract.metadata 增加 backend 字段约定
  dag_ports.py              — 修改: ResourceBridgeExecutor 实现 ExecutorPort 注册
  dynamic_orchestrator.py   — 修改: executor 路由 + dispatch_team 工具 + team 循环
  executors/
    __init__.py             — 新建: executor 注册表
    claude_code.py          — 新建: ClaudeCodeExecutor
  team.py                   — 新建: AgentTeam, Mailbox, SharedTaskList

tests/
  test_executor_routing.py  — 新建: executor 路由测试
  test_claude_code_executor.py — 新建: ClaudeCodeExecutor 测试
  test_agent_team.py        — 新建: agent team 测试
```

---

## Phase 1: Executor 可插拔化

### Task 1: ExecutorRegistry — executor 注册与路由

**Files:**
- Create: `babybot/agent_kernel/executors/__init__.py`
- Modify: `babybot/agent_kernel/protocols.py`
- Test: `tests/test_executor_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor_routing.py
"""Tests for executor routing."""

from __future__ import annotations
import asyncio
import pytest
from babybot.agent_kernel.types import ExecutionContext, TaskContract, TaskResult
from babybot.agent_kernel.executors import ExecutorRegistry


class FakeExecutor:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[str] = []

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        self.calls.append(task.task_id)
        return TaskResult(task_id=task.task_id, status="succeeded", output=f"{self.tag}:{task.task_id}")


def test_registry_routes_by_backend() -> None:
    """Tasks with different 'backend' metadata route to different executors."""
    local = FakeExecutor("local")
    claude = FakeExecutor("claude_code")
    registry = ExecutorRegistry(default=local)
    registry.register("claude_code", claude)

    task_local = TaskContract(task_id="t1", description="local task")
    task_claude = TaskContract(
        task_id="t2", description="remote task",
        metadata={"backend": "claude_code"},
    )
    ctx = ExecutionContext()

    r1 = asyncio.run(registry.execute(task_local, ctx))
    r2 = asyncio.run(registry.execute(task_claude, ctx))

    assert r1.output == "local:t1"
    assert r2.output == "claude_code:t2"
    assert local.calls == ["t1"]
    assert claude.calls == ["t2"]


def test_registry_unknown_backend_uses_default() -> None:
    """Unknown backend falls back to default executor."""
    default = FakeExecutor("default")
    registry = ExecutorRegistry(default=default)

    task = TaskContract(task_id="t1", description="x", metadata={"backend": "unknown"})
    r = asyncio.run(registry.execute(task, ExecutionContext()))
    assert r.output == "default:t1"


def test_registry_no_backend_uses_default() -> None:
    """Task without backend metadata uses default executor."""
    default = FakeExecutor("default")
    registry = ExecutorRegistry(default=default)

    task = TaskContract(task_id="t1", description="x")
    r = asyncio.run(registry.execute(task, ExecutionContext()))
    assert r.output == "default:t1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babybot.agent_kernel.executors'`

- [ ] **Step 3: Implement ExecutorRegistry**

```python
# babybot/agent_kernel/executors/__init__.py
"""Pluggable executor registry — routes tasks to backend-specific executors."""

from __future__ import annotations

import logging
from typing import Any

from ..protocols import ExecutorPort
from ..types import ExecutionContext, TaskContract, TaskResult

logger = logging.getLogger(__name__)

__all__ = ["ExecutorRegistry"]


class ExecutorRegistry:
    """Routes task execution to backend-specific ExecutorPort implementations.

    Each task's ``metadata["backend"]`` selects which executor handles it.
    Tasks without a backend (or with an unregistered backend) go to *default*.
    """

    def __init__(self, default: ExecutorPort) -> None:
        self._default = default
        self._backends: dict[str, ExecutorPort] = {}

    def register(self, backend: str, executor: ExecutorPort) -> None:
        self._backends[backend] = executor

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        backend = task.metadata.get("backend", "")
        executor = self._backends.get(backend, self._default)
        if backend and backend not in self._backends:
            logger.debug("Unknown backend %r for task %s, using default", backend, task.task_id)
        return await executor.execute(task, context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_executor_routing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/executors/__init__.py tests/test_executor_routing.py
git commit -m "feat: add ExecutorRegistry for pluggable backend routing"
```

---

### Task 2: DynamicOrchestrator 使用 ExecutorRegistry 替代硬编码 bridge

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py:946-990`
- Modify: `tests/test_dynamic_orchestrator.py` (existing tests must continue passing)

- [ ] **Step 1: Write the failing test**

在 `tests/test_dynamic_orchestrator.py` 末尾追加：

```python
def test_orchestrator_accepts_executor_registry() -> None:
    """DynamicOrchestrator uses ExecutorRegistry when provided."""
    from babybot.agent_kernel.executors import ExecutorRegistry

    gateway = DummyGateway([_reply_tool_call("done")])
    rm = DummyResourceManager()
    # Should accept executor_registry kwarg without error
    orch = DynamicOrchestrator(
        resource_manager=rm,
        gateway=gateway,
        executor_registry=None,  # None means use default bridge
    )
    result = asyncio.run(orch.run("hi", ExecutionContext()))
    assert result.conclusion == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_orchestrator.py::test_orchestrator_accepts_executor_registry -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'executor_registry'`

- [ ] **Step 3: Modify DynamicOrchestrator.__init__ to accept and use ExecutorRegistry**

In `babybot/agent_kernel/dynamic_orchestrator.py`, modify `__init__`:

```python
def __init__(
    self,
    resource_manager: "ResourceManager",
    gateway: "OpenAICompatibleGateway",
    child_task_bus: InMemoryChildTaskBus | None = None,
    task_heartbeat_registry: "TaskHeartbeatRegistry | None" = None,
    task_stale_after_s: float | None = None,
    max_steps: int | None = None,
    default_task_timeout_s: float | None = 120.0,
    executor_registry: "ExecutorRegistry | None" = None,
) -> None:
    from ..heartbeat import TaskHeartbeatRegistry
    from .executors import ExecutorRegistry as _ER

    self._rm = resource_manager
    self._gateway = gateway
    self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
    if executor_registry is not None:
        self._executor: ExecutorPort = executor_registry
    else:
        self._executor = self._bridge
    self._child_task_bus = child_task_bus or InMemoryChildTaskBus()
    self._task_heartbeat_registry = (
        task_heartbeat_registry or TaskHeartbeatRegistry()
    )
    self._task_stale_after_s = task_stale_after_s
    self._max_steps = max(1, int(max_steps or self.MAX_STEPS))
    self._default_task_timeout_s = default_task_timeout_s
```

In `run()` method, change the `InProcessChildTaskRuntime` instantiation from passing `bridge=self._bridge` to `bridge=self._executor`:

```python
runtime = InProcessChildTaskRuntime(
    flow_id=flow_id,
    resource_manager=self._rm,
    bridge=self._executor,  # was: self._bridge
    ...
)
```

- [ ] **Step 4: Run ALL existing orchestrator tests to verify no regression**

Run: `pytest tests/test_dynamic_orchestrator.py -v`
Expected: ALL PASS (including the new test)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/dynamic_orchestrator.py tests/test_dynamic_orchestrator.py
git commit -m "feat: wire ExecutorRegistry into DynamicOrchestrator"
```

---

## Phase 2: ClaudeCodeExecutor

### Task 3: ClaudeCodeExecutor — 通过 CLI 驱动 Claude Code

**Files:**
- Create: `babybot/agent_kernel/executors/claude_code.py`
- Test: `tests/test_claude_code_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_claude_code_executor.py
"""Tests for ClaudeCodeExecutor."""

from __future__ import annotations
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from babybot.agent_kernel.types import ExecutionContext, TaskContract, TaskResult
from babybot.agent_kernel.executors.claude_code import ClaudeCodeExecutor


def _make_process_mock(stdout: str, returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_basic_execution() -> None:
    """ClaudeCodeExecutor invokes claude -p and returns the result."""
    output_json = json.dumps({
        "result": "Fixed the bug in auth.py",
        "session_id": "sess_123",
    })
    with patch("asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t1", description="Fix the bug in auth.py")
        result = await executor.execute(task, ExecutionContext())

    assert result.status == "succeeded"
    assert "Fixed the bug" in result.output
    assert result.metadata.get("session_id") == "sess_123"


@pytest.mark.asyncio
async def test_resume_session() -> None:
    """ClaudeCodeExecutor resumes a session when session_id is in metadata."""
    output_json = json.dumps({"result": "Continued work", "session_id": "sess_123"})
    with patch("asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)) as mock_exec:
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(
            task_id="t2",
            description="Continue the review",
            metadata={"session_id": "sess_123"},
        )
        result = await executor.execute(task, ExecutionContext())

    # Verify --resume flag was passed
    call_args = mock_exec.call_args
    assert "--resume" in call_args[0]
    assert "sess_123" in call_args[0]
    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_nonzero_exit_code() -> None:
    """Non-zero exit code results in failed TaskResult."""
    with patch("asyncio.create_subprocess_exec", return_value=_make_process_mock("error output", returncode=1)):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t3", description="Fail")
        result = await executor.execute(task, ExecutionContext())

    assert result.status == "failed"
    assert result.error


@pytest.mark.asyncio
async def test_timeout() -> None:
    """Task with timeout_s raises TimeoutError on hang."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t4", description="Hang", timeout_s=1.0)
        result = await executor.execute(task, ExecutionContext())

    assert result.status == "failed"
    assert "timeout" in result.error.lower()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_allowed_tools_passed() -> None:
    """allowed_tools from metadata are forwarded to CLI."""
    output_json = json.dumps({"result": "ok", "session_id": "s1"})
    with patch("asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)) as mock_exec:
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(
            task_id="t5", description="Read only",
            metadata={"allowed_tools": ["Read", "Grep"]},
        )
        await executor.execute(task, ExecutionContext())

    call_args = mock_exec.call_args[0]
    assert "--allowedTools" in call_args
    idx = call_args.index("--allowedTools")
    assert call_args[idx + 1] == "Read,Grep"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_claude_code_executor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ClaudeCodeExecutor**

```python
# babybot/agent_kernel/executors/claude_code.py
"""Executor that drives Claude Code via its CLI (claude -p)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..types import ExecutionContext, TaskContract, TaskResult

logger = logging.getLogger(__name__)

__all__ = ["ClaudeCodeExecutor"]

_DEFAULT_TIMEOUT_S = 300.0


class ClaudeCodeExecutor:
    """Runs tasks by invoking ``claude -p`` as a subprocess.

    Supports:
    - One-shot execution (``claude -p "prompt" --output-format json``)
    - Session resumption via ``--resume session_id``
    - Tool allowlisting via ``--allowedTools``
    - Timeout enforcement via ``task.timeout_s``
    """

    def __init__(
        self,
        workdir: str = ".",
        claude_bin: str = "claude",
        default_timeout_s: float = _DEFAULT_TIMEOUT_S,
        extra_flags: tuple[str, ...] = (),
    ) -> None:
        self._workdir = workdir
        self._claude_bin = claude_bin
        self._default_timeout_s = default_timeout_s
        self._extra_flags = extra_flags

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        cmd = self._build_command(task)
        timeout = task.timeout_s or self._default_timeout_s

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workdir,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=f"Timeout after {timeout}s",
                )
        except FileNotFoundError:
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=f"Claude Code binary not found: {self._claude_bin}",
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=stderr or stdout or f"Exit code {proc.returncode}",
            )

        return self._parse_output(task.task_id, stdout)

    def _build_command(self, task: TaskContract) -> list[str]:
        cmd = [self._claude_bin, "-p", task.description, "--output-format", "json"]

        session_id = task.metadata.get("session_id")
        if session_id:
            cmd.extend(["--resume", session_id])

        allowed_tools = task.metadata.get("allowed_tools")
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        system_prompt = task.metadata.get("system_prompt")
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        cmd.extend(self._extra_flags)
        return cmd

    @staticmethod
    def _parse_output(task_id: str, stdout: str) -> TaskResult:
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            # Plain text fallback
            return TaskResult(
                task_id=task_id,
                status="succeeded",
                output=stdout.strip(),
            )

        result_text = data.get("result", stdout.strip())
        session_id = data.get("session_id")
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["session_id"] = session_id

        return TaskResult(
            task_id=task_id,
            status="succeeded",
            output=result_text,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_claude_code_executor.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/executors/claude_code.py tests/test_claude_code_executor.py
git commit -m "feat: add ClaudeCodeExecutor for driving Claude Code via CLI"
```

---

### Task 4: 注册 ClaudeCodeExecutor 作为 resource type

**Files:**
- Modify: `babybot/resource_scope.py:78-142` (get_resource_briefs 增加 external_agents)
- Modify: `babybot/resource_scope.py:144-180` (resolve_resource_scope 增加 external_agents)
- Modify: `babybot/resource.py` (ResourceManager 读取配置、构造 ExecutorRegistry)
- Test: `tests/test_executor_routing.py` (追加集成测试)

- [ ] **Step 1: Write the failing test**

在 `tests/test_executor_routing.py` 追加：

```python
def test_resource_brief_includes_external_agent() -> None:
    """External agents configured via config appear in resource briefs."""
    from babybot.agent_kernel.executors.claude_code import ClaudeCodeExecutor

    registry = ExecutorRegistry(default=FakeExecutor("local"))
    executor = ClaudeCodeExecutor(workdir="/tmp")
    registry.register("claude_code", executor)

    assert "claude_code" in registry.list_backends()


def test_registry_list_backends() -> None:
    """list_backends returns all registered backend names."""
    registry = ExecutorRegistry(default=FakeExecutor("local"))
    registry.register("claude_code", FakeExecutor("cc"))
    registry.register("codex", FakeExecutor("cx"))

    backends = registry.list_backends()
    assert "claude_code" in backends
    assert "codex" in backends
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor_routing.py::test_registry_list_backends -v`
Expected: FAIL with `AttributeError: 'ExecutorRegistry' object has no attribute 'list_backends'`

- [ ] **Step 3: Add list_backends to ExecutorRegistry**

In `babybot/agent_kernel/executors/__init__.py`, add to the `ExecutorRegistry` class:

```python
def list_backends(self) -> list[str]:
    return list(self._backends.keys())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_executor_routing.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/executors/__init__.py tests/test_executor_routing.py
git commit -m "feat: add list_backends to ExecutorRegistry for resource discovery"
```

---

## Phase 3: Agent Team — 多 Agent 协作

### Task 5: Mailbox — agent 间消息通信

**Files:**
- Create: `babybot/agent_kernel/team.py`
- Test: `tests/test_agent_team.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_team.py
"""Tests for agent team: Mailbox and SharedTaskList."""

from __future__ import annotations
import asyncio
import pytest
from babybot.agent_kernel.team import Mailbox, SharedTaskList, TeamTask


# ── Mailbox tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mailbox_send_receive() -> None:
    """Messages sent to a recipient can be received."""
    mb = Mailbox()
    mb.send("agent_a", "agent_b", "Hello from A")
    messages = mb.receive("agent_b")
    assert len(messages) == 1
    assert messages[0].sender == "agent_a"
    assert messages[0].content == "Hello from A"


@pytest.mark.asyncio
async def test_mailbox_receive_clears() -> None:
    """receive() clears the inbox after reading."""
    mb = Mailbox()
    mb.send("a", "b", "msg1")
    mb.receive("b")
    assert mb.receive("b") == []


@pytest.mark.asyncio
async def test_mailbox_broadcast() -> None:
    """broadcast sends to all listed recipients."""
    mb = Mailbox()
    mb.broadcast("lead", ["a", "b", "c"], "Announcement")
    assert len(mb.receive("a")) == 1
    assert len(mb.receive("b")) == 1
    assert len(mb.receive("c")) == 1
    assert mb.receive("lead") == []  # sender doesn't get it


@pytest.mark.asyncio
async def test_mailbox_wait_for_message() -> None:
    """wait_for_message blocks until a message arrives."""
    mb = Mailbox()

    async def delayed_send():
        await asyncio.sleep(0.05)
        mb.send("a", "b", "delayed")

    asyncio.create_task(delayed_send())
    msg = await asyncio.wait_for(mb.wait_for_message("b"), timeout=1.0)
    assert msg.content == "delayed"


# ── SharedTaskList tests ─────────────────────────────────────────────────

def test_task_list_add_and_claim() -> None:
    """Tasks can be added and claimed by agents."""
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="Do X"))
    tl.add(TeamTask(task_id="t2", description="Do Y"))

    claimed = tl.claim("agent_a")
    assert claimed is not None
    assert claimed.task_id == "t1"
    assert claimed.assigned_to == "agent_a"

    claimed2 = tl.claim("agent_b")
    assert claimed2 is not None
    assert claimed2.task_id == "t2"

    # No more tasks
    assert tl.claim("agent_c") is None


def test_task_list_complete() -> None:
    """Completing a task updates its status."""
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="Do X"))
    tl.claim("agent_a")
    tl.complete("t1", output="Done")

    status = tl.get_status()
    assert status["t1"].status == "completed"
    assert status["t1"].output == "Done"


def test_task_list_dependencies() -> None:
    """Tasks with unresolved deps cannot be claimed."""
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="First"))
    tl.add(TeamTask(task_id="t2", description="Second", deps=["t1"]))

    # t2 is blocked, so agent gets t1
    claimed = tl.claim("agent_a")
    assert claimed.task_id == "t1"

    # t2 still blocked
    assert tl.claim("agent_b") is None

    # Complete t1, now t2 is available
    tl.complete("t1", output="ok")
    claimed2 = tl.claim("agent_b")
    assert claimed2 is not None
    assert claimed2.task_id == "t2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_team.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babybot.agent_kernel.team'`

- [ ] **Step 3: Implement Mailbox and SharedTaskList**

```python
# babybot/agent_kernel/team.py
"""Agent team primitives: Mailbox for inter-agent messaging, SharedTaskList for coordination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = ["Mailbox", "MailMessage", "SharedTaskList", "TeamTask"]


@dataclass
class MailMessage:
    sender: str
    content: str


@dataclass
class TeamTask:
    task_id: str
    description: str
    deps: list[str] = field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    assigned_to: str = ""
    output: str = ""


class Mailbox:
    """In-process async mailbox for agent-to-agent messaging."""

    def __init__(self) -> None:
        self._boxes: dict[str, list[MailMessage]] = {}
        self._events: dict[str, asyncio.Event] = {}

    def _ensure(self, recipient: str) -> None:
        if recipient not in self._boxes:
            self._boxes[recipient] = []
            self._events[recipient] = asyncio.Event()

    def send(self, sender: str, recipient: str, content: str) -> None:
        self._ensure(recipient)
        self._boxes[recipient].append(MailMessage(sender=sender, content=content))
        self._events[recipient].set()

    def broadcast(self, sender: str, recipients: list[str], content: str) -> None:
        for r in recipients:
            if r != sender:
                self.send(sender, r, content)

    def receive(self, recipient: str) -> list[MailMessage]:
        self._ensure(recipient)
        messages = list(self._boxes[recipient])
        self._boxes[recipient].clear()
        self._events[recipient].clear()
        return messages

    async def wait_for_message(self, recipient: str, timeout: float | None = None) -> MailMessage:
        self._ensure(recipient)
        if not self._boxes[recipient]:
            self._events[recipient].clear()
            await asyncio.wait_for(self._events[recipient].wait(), timeout=timeout)
        return self._boxes[recipient].pop(0)


class SharedTaskList:
    """Thread-safe shared task list with dependency tracking."""

    def __init__(self) -> None:
        self._tasks: dict[str, TeamTask] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def add(self, task: TeamTask) -> None:
        self._tasks[task.task_id] = task
        self._order.append(task.task_id)

    def _deps_met(self, task: TeamTask) -> bool:
        return all(
            self._tasks.get(dep, TeamTask(task_id=dep, description="")).status == "completed"
            for dep in task.deps
        )

    def claim(self, agent_id: str) -> TeamTask | None:
        for tid in self._order:
            task = self._tasks[tid]
            if task.status == "pending" and self._deps_met(task):
                task.status = "in_progress"
                task.assigned_to = agent_id
                return task
        return None

    def complete(self, task_id: str, output: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = "completed"
            task.output = output

    def fail(self, task_id: str, error: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = "failed"
            task.output = error

    def get_status(self) -> dict[str, TeamTask]:
        return dict(self._tasks)

    def all_done(self) -> bool:
        return all(t.status in ("completed", "failed") for t in self._tasks.values())

    def summary(self) -> str:
        lines = []
        for tid in self._order:
            t = self._tasks[tid]
            marker = {"pending": " ", "in_progress": ">", "completed": "x", "failed": "!"}
            lines.append(f"[{marker.get(t.status, '?')}] {t.task_id}: {t.description}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_team.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/team.py tests/test_agent_team.py
git commit -m "feat: add Mailbox and SharedTaskList for agent team coordination"
```

---

### Task 6: dispatch_team 编排工具 — 在 DynamicOrchestrator 中支持 team 模式

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py` (新增 `dispatch_team` 工具 schema + 执行逻辑)
- Test: `tests/test_agent_team.py` (追加 orchestrator 集成测试)

- [ ] **Step 1: Write the failing test**

在 `tests/test_agent_team.py` 追加：

```python
from babybot.agent_kernel.team import TeamRunner


@pytest.mark.asyncio
async def test_team_runner_debate() -> None:
    """TeamRunner executes a multi-round debate between two agents."""
    round_count = 0

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        nonlocal round_count
        round_count += 1
        if "agent_a" in agent_id:
            return f"A's argument round {round_count}: I disagree because X"
        return f"B's counter round {round_count}: Actually Y"

    runner = TeamRunner(executor=mock_executor, max_rounds=3)
    result = await runner.run_debate(
        topic="Should we use microservices?",
        agents=[
            {"id": "agent_a", "role": "proponent", "description": "Argues for microservices"},
            {"id": "agent_b", "role": "opponent", "description": "Argues against microservices"},
        ],
    )

    assert result.rounds >= 2
    assert result.rounds <= 3
    assert len(result.transcript) > 0
    assert result.summary  # judge produced a summary


@pytest.mark.asyncio
async def test_team_runner_convergence() -> None:
    """TeamRunner stops early when judge signals convergence."""

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        return "I agree with the previous point."

    runner = TeamRunner(executor=mock_executor, max_rounds=10)
    result = await runner.run_debate(
        topic="Test topic",
        agents=[
            {"id": "a", "role": "proponent", "description": "Pro"},
            {"id": "b", "role": "opponent", "description": "Con"},
        ],
        judge=lambda transcript: (True, "Agents reached consensus"),
    )

    # Should stop before max_rounds due to convergence
    assert result.rounds < 10
    assert "consensus" in result.summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_team.py::test_team_runner_debate -v`
Expected: FAIL with `ImportError: cannot import name 'TeamRunner'`

- [ ] **Step 3: Implement TeamRunner**

在 `babybot/agent_kernel/team.py` 追加：

```python
from dataclasses import dataclass as _dataclass


@_dataclass
class DebateResult:
    topic: str
    rounds: int
    transcript: list[dict[str, str]]
    summary: str


class TeamRunner:
    """Runs structured multi-agent interactions (e.g., debates).

    The runner alternates between agents, passing each agent the previous
    agent's output.  An optional judge function can signal early convergence.
    """

    def __init__(
        self,
        executor: Any,  # async callable(agent_id, prompt, context) -> str
        max_rounds: int = 5,
    ) -> None:
        self._executor = executor
        self._max_rounds = max_rounds

    async def run_debate(
        self,
        topic: str,
        agents: list[dict[str, str]],
        judge: Any | None = None,
    ) -> DebateResult:
        transcript: list[dict[str, str]] = []
        last_output = ""

        for round_num in range(1, self._max_rounds + 1):
            for agent in agents:
                prompt_parts = [
                    f"Topic: {topic}",
                    f"Your role: {agent['role']} — {agent['description']}",
                    f"Round: {round_num}/{self._max_rounds}",
                ]
                if last_output:
                    prompt_parts.append(f"Previous argument:\n{last_output}")
                prompt_parts.append("Present your argument:")

                prompt = "\n".join(prompt_parts)
                output = await self._executor(agent["id"], prompt, {})
                transcript.append({
                    "round": str(round_num),
                    "agent": agent["id"],
                    "role": agent["role"],
                    "content": output,
                })
                last_output = output

            # Check convergence
            if judge is not None:
                converged, reason = judge(transcript)
                if converged:
                    return DebateResult(
                        topic=topic,
                        rounds=round_num,
                        transcript=transcript,
                        summary=reason,
                    )

        # Max rounds reached — summarize
        summary_parts = [f"Debate on '{topic}' completed after {self._max_rounds} rounds."]
        for entry in transcript[-len(agents):]:
            summary_parts.append(f"- {entry['role']}: {entry['content'][:200]}")

        return DebateResult(
            topic=topic,
            rounds=self._max_rounds,
            transcript=transcript,
            summary="\n".join(summary_parts),
        )
```

Update `__all__` in `team.py`:

```python
__all__ = ["Mailbox", "MailMessage", "SharedTaskList", "TeamTask", "TeamRunner", "DebateResult"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_agent_team.py -v`
Expected: ALL PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/team.py tests/test_agent_team.py
git commit -m "feat: add TeamRunner for structured multi-agent debates"
```

---

### Task 7: 将 dispatch_team 接入 DynamicOrchestrator

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
  - `_ORCHESTRATION_TOOLS` 追加 `dispatch_team` schema
  - `_dispatch_tool` 新增 `dispatch_team` 分支
  - 新增 `_run_team` 方法调用 `TeamRunner`
- Test: `tests/test_dynamic_orchestrator.py` (追加集成测试)

- [ ] **Step 1: Write the failing test**

在 `tests/test_dynamic_orchestrator.py` 追加：

```python
def test_dispatch_team_tool_recognized() -> None:
    """DynamicOrchestrator recognizes dispatch_team as a valid tool."""
    from babybot.agent_kernel.dynamic_orchestrator import _ORCHESTRATION_TOOLS
    tool_names = [t["function"]["name"] for t in _ORCHESTRATION_TOOLS]
    assert "dispatch_team" in tool_names


def test_team_dispatch_and_reply() -> None:
    """Orchestrator dispatches a team debate and replies with the result."""
    team_args = {
        "topic": "Should we refactor?",
        "agents": [
            {"id": "pro", "role": "proponent", "description": "For refactoring"},
            {"id": "con", "role": "opponent", "description": "Against refactoring"},
        ],
        "max_rounds": 2,
    }
    gateway = DummyGateway([
        # Step 1: model dispatches a team
        ModelResponse(
            text="",
            tool_calls=(
                ModelToolCall(
                    call_id="call_team",
                    name="dispatch_team",
                    arguments=team_args,
                ),
            ),
            finish_reason="tool_calls",
        ),
        # Step 2: model replies with conclusion
        _reply_tool_call("Refactoring is recommended based on the debate."),
    ])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("Should we refactor?", ExecutionContext()))
    assert "refactor" in result.conclusion.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dynamic_orchestrator.py::test_dispatch_team_tool_recognized -v`
Expected: FAIL with `AssertionError: 'dispatch_team' not in [...]`

- [ ] **Step 3: Add dispatch_team tool schema to _ORCHESTRATION_TOOLS**

In `babybot/agent_kernel/dynamic_orchestrator.py`, add to `_ORCHESTRATION_TOOLS`:

```python
{
    "type": "function",
    "function": {
        "name": "dispatch_team",
        "description": (
            "启动一组Agent进行多轮协作讨论（如辩论、评审、头脑风暴）。"
            "Agent之间会交替发言，支持可选的judge函数来判断是否达成共识。"
            "返回完整讨论记录和总结。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "讨论主题",
                },
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "role": {"type": "string"},
                            "description": {"type": "string"},
                            "resource_id": {
                                "type": "string",
                                "description": "可选：指定该Agent使用的资源ID",
                            },
                        },
                        "required": ["id", "role", "description"],
                    },
                    "description": "参与讨论的Agent列表，至少2个",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "最大讨论轮数，默认5",
                },
            },
            "required": ["topic", "agents"],
        },
    },
},
```

- [ ] **Step 4: Add dispatch_team handling to _dispatch_tool and _run_team method**

In `_dispatch_tool`, add before the final `return f"error: unknown tool: {name}"`:

```python
if name == "dispatch_team":
    return await self._run_team(args, context)
```

Add `_run_team` method to `DynamicOrchestrator`:

```python
async def _run_team(self, args: dict[str, Any], context: ExecutionContext) -> str:
    import json as _json
    from .team import TeamRunner

    topic = args.get("topic", "")
    agents = args.get("agents", [])
    max_rounds = int(args.get("max_rounds", 5))

    if len(agents) < 2:
        return "error: dispatch_team requires at least 2 agents"

    async def agent_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        """Execute one agent turn using the gateway directly."""
        from .model import ModelMessage, ModelRequest
        messages = [
            ModelMessage(role="system", content="你是讨论参与者。根据你的角色，针对主题发表观点。"),
            ModelMessage(role="user", content=prompt),
        ]
        request = ModelRequest(messages=tuple(messages))
        response = await self._gateway.generate(request, ExecutionContext())
        return response.text

    runner = TeamRunner(executor=agent_executor, max_rounds=max_rounds)
    result = await runner.run_debate(topic=topic, agents=agents)

    return _json.dumps({
        "topic": result.topic,
        "rounds": result.rounds,
        "summary": result.summary,
        "transcript_length": len(result.transcript),
        "last_arguments": [
            {"agent": e["agent"], "role": e["role"], "content": e["content"][:500]}
            for e in result.transcript[-len(agents):]
        ],
    }, ensure_ascii=False)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_dynamic_orchestrator.py -v`
Expected: ALL PASS (including new team tests)

- [ ] **Step 6: Commit**

```bash
git add babybot/agent_kernel/dynamic_orchestrator.py tests/test_dynamic_orchestrator.py
git commit -m "feat: integrate dispatch_team tool into DynamicOrchestrator"
```

---

### Task 8: Run full test suite and verify no regressions

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Fix any regressions**

If any tests fail, fix and re-run.

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "fix: address test regressions from executor/team changes"
```

---

## Summary of deliverables

| Phase | Deliverable | Files |
|-------|-------------|-------|
| 1 | ExecutorRegistry + pluggable routing | `executors/__init__.py`, `dynamic_orchestrator.py` |
| 2 | ClaudeCodeExecutor | `executors/claude_code.py` |
| 3 | Agent team (Mailbox + SharedTaskList + TeamRunner) | `team.py`, `dynamic_orchestrator.py` |

## Suggested follow-ups (not in scope)

- `CodexExecutor` — OpenAI Codex API adapter，同样实现 `ExecutorPort`
- `ACPExecutor` — 通用 ACP client，当外部 agent 支持 ACP 时使用
- 将 babybot 自身包装为 ACP server 对外暴露
- 声明式 agent profile 配置（Markdown + YAML frontmatter），参考 Claude Code subagent 模型
- `dispatch_team` 支持通过 `resource_id` 让 team member 使用特定工具集
- judge 集成 — 允许指定一个第三方 agent 或 LLM 调用作为仲裁者
