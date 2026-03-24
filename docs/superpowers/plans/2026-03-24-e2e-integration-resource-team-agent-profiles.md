# E2E Integration, Resource-Aware Teams & Agent Profiles

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成三项增强：(1) ClaudeCodeExecutor 端到端集成测试；(2) dispatch_team 支持 per-agent resource_id，让 team member 具备工具能力；(3) 声明式 Agent Profile，让 team 配置可通过 AGENT.md 持久化和复用。

**Architecture:** 集成测试使用 skip-if-not-available 策略（`claude` CLI 不存在时自动 skip）。resource_id 支持复用现有 `ResourceBridgeExecutor.execute()` 作为 per-agent executor，仅修改 `_run_team` 的 executor 闭包。Agent Profile 采用与 SKILL.md 相同的 YAML frontmatter + Markdown body 模式，新建 `AgentProfileLoader` 解析 `AGENT.md`，让 `dispatch_team` 可以通过 `profile_id` 引用预定义的 agent。

**Tech Stack:** Python 3.11+, asyncio, pytest, subprocess (for claude CLI), YAML frontmatter

---

## File Structure

```
babybot/agent_kernel/
  dynamic_orchestrator.py   — 修改: _run_team 支持 resource_id + profile_id
  team.py                   — 修改: TeamRunner 支持 per-agent executor routing
  agent_profile.py          — 新建: AgentProfileLoader + AgentProfile dataclass
  executors/
    claude_code.py          — 不改动（集成测试只读）

tests/
  test_claude_code_executor.py  — 修改: 追加集成测试
  test_agent_team.py            — 修改: 追加 resource-aware team 测试
  test_agent_profile.py         — 新建: agent profile 加载/解析测试
  test_dynamic_orchestrator.py  — 修改: 追加 profile_id dispatch_team 测试
```

---

## Phase 1: ClaudeCodeExecutor 端到端集成测试

### Task 1: 添加 ClaudeCodeExecutor 集成测试

**Files:**
- Modify: `tests/test_claude_code_executor.py`

集成测试策略：用 `shutil.which("claude")` 检测 CLI 是否存在，不存在则 `pytest.skip`。测试发送一个简单 prompt（如 "Reply with exactly: HELLO"），验证 subprocess 通路、JSON 解析、TaskResult 构造完整链路。

- [ ] **Step 1: Write the integration test**

```python
# 追加到 tests/test_claude_code_executor.py 末尾

import shutil

_CLAUDE_AVAILABLE = shutil.which("claude") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_basic_prompt() -> None:
    """E2E: ClaudeCodeExecutor sends a real prompt to claude CLI."""
    executor = ClaudeCodeExecutor(workdir="/tmp", default_timeout_s=30.0)
    task = TaskContract(
        task_id="e2e_1",
        description='Reply with exactly the word HELLO and nothing else.',
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "succeeded", f"Expected success, got: {result.error}"
    assert result.task_id == "e2e_1"
    assert "HELLO" in result.output.upper()


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_session_id_returned() -> None:
    """E2E: ClaudeCodeExecutor returns a session_id in metadata."""
    executor = ClaudeCodeExecutor(workdir="/tmp", default_timeout_s=30.0)
    task = TaskContract(
        task_id="e2e_2",
        description='Say OK.',
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "succeeded", f"Expected success, got: {result.error}"
    # Claude Code JSON output should include session_id
    assert result.metadata.get("session_id"), "Expected session_id in metadata"


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_timeout_enforcement() -> None:
    """E2E: ClaudeCodeExecutor enforces timeout on real subprocess."""
    executor = ClaudeCodeExecutor(workdir="/tmp")
    task = TaskContract(
        task_id="e2e_3",
        description=(
            'Write a Python script that computes the first 10 million prime numbers '
            'and prints each one. Do not stop until all are printed.'
        ),
        timeout_s=3.0,  # Very short — should timeout
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "failed"
    assert "timeout" in result.error.lower()
```

- [ ] **Step 2: Run test to verify**

Run: `pytest tests/test_claude_code_executor.py -v -k "integration"`
Expected: PASS if `claude` is available, SKIP otherwise. Timeout test should reliably fail within ~5s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_claude_code_executor.py
git commit -m "test: add ClaudeCodeExecutor E2E integration tests (skip if CLI absent)"
```

---

## Phase 2: dispatch_team 支持 per-agent resource_id

### Task 2: TeamRunner 支持 per-agent executor 路由

**Files:**
- Modify: `babybot/agent_kernel/team.py:147-217` (TeamRunner)
- Test: `tests/test_agent_team.py`

当前 `TeamRunner` 接受一个全局 executor callable。改为支持 per-agent executor：如果 agent dict 中包含 `executor` key（一个 async callable），优先使用它；否则 fallback 到全局 executor。

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_agent_team.py

@pytest.mark.asyncio
async def test_team_runner_per_agent_executor() -> None:
    """TeamRunner routes each agent to its own executor when provided."""
    call_log: list[str] = []

    async def global_exec(agent_id: str, prompt: str, context: dict) -> str:
        call_log.append(f"global:{agent_id}")
        return f"global response from {agent_id}"

    async def special_exec(agent_id: str, prompt: str, context: dict) -> str:
        call_log.append(f"special:{agent_id}")
        return f"special response from {agent_id}"

    runner = TeamRunner(executor=global_exec, max_rounds=1)
    result = await runner.run_debate(
        topic="Test routing",
        agents=[
            {"id": "agent_a", "role": "pro", "description": "Uses global"},
            {"id": "agent_b", "role": "con", "description": "Uses special", "executor": special_exec},
        ],
    )

    assert "global:agent_a" in call_log
    assert "special:agent_b" in call_log
    assert "global:agent_b" not in call_log
    assert result.rounds == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_team.py::test_team_runner_per_agent_executor -v`
Expected: FAIL — current TeamRunner ignores `agent["executor"]`

- [ ] **Step 3: Modify TeamRunner to support per-agent executor**

In `babybot/agent_kernel/team.py`, inside `TeamRunner.run_debate()`, change the executor call:

```python
# Before:
output = await self._executor(agent["id"], prompt, {})

# After:
agent_exec = agent.get("executor", self._executor)
output = await agent_exec(agent["id"], prompt, {})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_team.py -v`
Expected: ALL PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/team.py tests/test_agent_team.py
git commit -m "feat: TeamRunner supports per-agent executor routing"
```

---

### Task 3: _run_team 为有 resource_id 的 agent 创建 resource-backed executor

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py:1296-1339` (_run_team)
- Test: `tests/test_dynamic_orchestrator.py`

当 agent dict 中有 `resource_id` 时，为该 agent 创建一个 executor 闭包：构造 `TaskContract`（携带 resource_id）并调用 `self._executor.execute()`（即通过 ExecutorRegistry → ResourceBridgeExecutor 完整链路）。没有 `resource_id` 的 agent 继续使用 gateway 直接生成。

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_dynamic_orchestrator.py

def test_team_dispatch_with_resource_id() -> None:
    """dispatch_team routes agents with resource_id through the bridge executor."""
    team_args = {
        "topic": "Code review",
        "agents": [
            {"id": "reviewer", "role": "reviewer", "description": "Reviews code",
             "resource_id": "skill.weather"},
            {"id": "author", "role": "author", "description": "Defends code"},
        ],
        "max_rounds": 1,
    }
    gateway = DummyGateway(
        [
            # Step 1: orchestrator dispatches team
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
            # Step 2: gateway call for agent with resource_id (run_subagent_task)
            # The reviewer agent goes through bridge → run_subagent_task
            # Step 3: gateway call for agent without resource_id (direct gateway)
            ModelResponse(text="Author's defense of the code"),
            # Step 4: orchestrator replies
            _reply_tool_call("Review complete."),
        ]
    )
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("Review this code", ExecutionContext()))
    assert result.conclusion == "Review complete."
    # Verify the reviewer agent went through run_subagent_task
    assert len(rm.calls) == 1
    assert "Reviews code" in rm.calls[0]["task_description"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_orchestrator.py::test_team_dispatch_with_resource_id -v`
Expected: FAIL — current `_run_team` ignores resource_id

- [ ] **Step 3: Modify _run_team to create resource-backed executors**

Replace `_run_team` in `dynamic_orchestrator.py`:

```python
async def _run_team(
    self, args: dict[str, Any], context: ExecutionContext
) -> str:
    from .team import TeamRunner

    topic = args.get("topic", "")
    agents = args.get("agents", [])
    max_rounds = int(args.get("max_rounds", 5))

    if len(agents) < 2:
        return "error: dispatch_team requires at least 2 agents"

    async def gateway_executor(
        agent_id: str, prompt: str, ctx: dict[str, Any]
    ) -> str:
        """Fallback executor: calls gateway directly (no tools)."""
        messages = [
            ModelMessage(
                role="system",
                content="你是讨论参与者。根据你的角色，针对主题发表观点。",
            ),
            ModelMessage(role="user", content=prompt),
        ]
        request = ModelRequest(messages=tuple(messages))
        response = await self._gateway.generate(request, ExecutionContext())
        return response.text

    async def resource_executor(
        resource_id: str, agent_id: str, prompt: str, ctx: dict[str, Any]
    ) -> str:
        """Resource-backed executor: runs through bridge with full tool access."""
        task = TaskContract(
            task_id=f"team_{agent_id}",
            description=prompt,
            metadata={"resource_id": resource_id},
        )
        result = await self._executor.execute(task, context)
        if result.status != "succeeded":
            return f"[error: {result.error}]"
        return result.output

    # Prepare per-agent executors
    enriched_agents: list[dict[str, Any]] = []
    for agent in agents:
        agent_copy = dict(agent)
        rid = agent.get("resource_id")
        if rid:
            # Validate resource exists
            scope = self._rm.resolve_resource_scope(rid, require_tools=True)
            if scope is not None:
                import functools
                agent_copy["executor"] = functools.partial(resource_executor, rid)
        enriched_agents.append(agent_copy)

    runner = TeamRunner(executor=gateway_executor, max_rounds=max_rounds)
    result = await runner.run_debate(topic=topic, agents=enriched_agents)

    return json.dumps(
        {
            "topic": result.topic,
            "rounds": result.rounds,
            "summary": result.summary,
            "transcript_length": len(result.transcript),
            "last_arguments": [
                {
                    "agent": e["agent"],
                    "role": e["role"],
                    "content": e["content"][:500],
                }
                for e in result.transcript[-len(agents):]
            ],
        },
        ensure_ascii=False,
    )
```

- [ ] **Step 4: Run ALL orchestrator tests**

Run: `pytest tests/test_dynamic_orchestrator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/dynamic_orchestrator.py tests/test_dynamic_orchestrator.py
git commit -m "feat: dispatch_team routes agents with resource_id through bridge executor"
```

---

## Phase 3: 声明式 Agent Profile

### Task 4: AgentProfile dataclass + AgentProfileLoader

**Files:**
- Create: `babybot/agent_kernel/agent_profile.py`
- Test: `tests/test_agent_profile.py`

Agent Profile 使用 AGENT.md 文件，格式与 SKILL.md 相同（YAML frontmatter + Markdown body）。Frontmatter 支持：`name`（必需）、`role`（必需）、`description`、`resource_id`、`system_prompt`。Markdown body 作为扩展 system_prompt 内容。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_profile.py
"""Tests for declarative agent profiles."""

from __future__ import annotations
import os
import tempfile
import pytest
from babybot.agent_kernel.agent_profile import AgentProfile, AgentProfileLoader


def test_parse_agent_profile_from_markdown() -> None:
    """Parse an AGENT.md file into an AgentProfile."""
    content = """\
---
name: code-reviewer
role: reviewer
description: Reviews code for quality and correctness
resource_id: skill.code
---

# Code Reviewer

You are an expert code reviewer. Focus on:
- Correctness
- Performance
- Readability
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(content)
        f.flush()
        profile = AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)

    assert profile.name == "code-reviewer"
    assert profile.role == "reviewer"
    assert profile.description == "Reviews code for quality and correctness"
    assert profile.resource_id == "skill.code"
    assert "expert code reviewer" in profile.system_prompt


def test_parse_agent_profile_minimal() -> None:
    """Minimal AGENT.md with only required fields."""
    content = """\
---
name: debater
role: proponent
---

Argue for the given position.
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(content)
        f.flush()
        profile = AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)

    assert profile.name == "debater"
    assert profile.role == "proponent"
    assert profile.description == ""
    assert profile.resource_id == ""
    assert "Argue for" in profile.system_prompt


def test_load_profiles_from_directory() -> None:
    """AgentProfileLoader.load_dir scans a directory for AGENT.md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two profile dirs
        for name, role in [("alice", "proponent"), ("bob", "opponent")]:
            d = os.path.join(tmpdir, name)
            os.makedirs(d)
            with open(os.path.join(d, "AGENT.md"), "w") as f:
                f.write(f"---\nname: {name}\nrole: {role}\n---\n\nPrompt for {name}.\n")

        profiles = AgentProfileLoader.load_dir(tmpdir)

    assert len(profiles) == 2
    names = {p.name for p in profiles}
    assert names == {"alice", "bob"}


def test_profile_to_agent_dict() -> None:
    """AgentProfile.to_agent_dict() produces a dict compatible with dispatch_team."""
    profile = AgentProfile(
        name="reviewer",
        role="reviewer",
        description="Reviews code",
        resource_id="skill.code",
        system_prompt="You are a reviewer.",
    )
    d = profile.to_agent_dict()
    assert d["id"] == "reviewer"
    assert d["role"] == "reviewer"
    assert d["description"] == "Reviews code"
    assert d["resource_id"] == "skill.code"
    assert d["system_prompt"] == "You are a reviewer."


def test_load_file_rejects_missing_name() -> None:
    """AGENT.md without 'name' in frontmatter raises ValueError."""
    content = "---\nrole: pro\n---\nHello\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(content)
        f.flush()
        with pytest.raises(ValueError, match="name"):
            AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)


def test_load_file_rejects_missing_role() -> None:
    """AGENT.md without 'role' in frontmatter raises ValueError."""
    content = "---\nname: foo\n---\nHello\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(content)
        f.flush()
        with pytest.raises(ValueError, match="role"):
            AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babybot.agent_kernel.agent_profile'`

- [ ] **Step 3: Implement AgentProfile and AgentProfileLoader**

```python
# babybot/agent_kernel/agent_profile.py
"""Declarative agent profiles loaded from AGENT.md files."""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["AgentProfile", "AgentProfileLoader"]

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


@dataclass
class AgentProfile:
    """A reusable agent identity for team interactions."""

    name: str
    role: str
    description: str = ""
    resource_id: str = ""
    system_prompt: str = ""

    def to_agent_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with dispatch_team agent schema."""
        d: dict[str, Any] = {
            "id": self.name,
            "role": self.role,
            "description": self.description,
        }
        if self.resource_id:
            d["resource_id"] = self.resource_id
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        return d


class AgentProfileLoader:
    """Loads AgentProfile instances from AGENT.md files."""

    @staticmethod
    def load_file(path: str) -> AgentProfile:
        """Parse a single AGENT.md file into an AgentProfile."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        match = _FRONTMATTER_RE.match(raw)
        if not match:
            raise ValueError(f"Invalid AGENT.md format (no frontmatter): {path}")

        frontmatter_text = match.group(1)
        body = match.group(2).strip()

        meta = _parse_yaml_frontmatter(frontmatter_text)

        name = meta.get("name", "").strip()
        if not name:
            raise ValueError(f"AGENT.md missing required field 'name': {path}")

        role = meta.get("role", "").strip()
        if not role:
            raise ValueError(f"AGENT.md missing required field 'role': {path}")

        return AgentProfile(
            name=name,
            role=role,
            description=meta.get("description", "").strip(),
            resource_id=meta.get("resource_id", "").strip(),
            system_prompt=body,
        )

    @classmethod
    def load_dir(cls, directory: str) -> list[AgentProfile]:
        """Scan a directory for subdirectories containing AGENT.md."""
        profiles: list[AgentProfile] = []
        if not os.path.isdir(directory):
            return profiles

        for entry in sorted(os.listdir(directory)):
            agent_md = os.path.join(directory, entry, "AGENT.md")
            if os.path.isfile(agent_md):
                try:
                    profiles.append(cls.load_file(agent_md))
                except (ValueError, OSError) as exc:
                    logger.warning("Skipping invalid agent profile %s: %s", agent_md, exc)

        return profiles


def _parse_yaml_frontmatter(text: str) -> dict[str, str]:
    """Minimal YAML-like key: value parser (no dependency on PyYAML)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_profile.py -v`
Expected: ALL PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add babybot/agent_kernel/agent_profile.py tests/test_agent_profile.py
git commit -m "feat: add AgentProfile and AgentProfileLoader for declarative agent identities"
```

---

### Task 5: dispatch_team 支持 profile_id 引用预定义 agent

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py` (_run_team + dispatch_team schema)
- Test: `tests/test_dynamic_orchestrator.py`

在 dispatch_team 的 agents schema 中新增可选的 `profile_id` 字段。当 agent 有 `profile_id` 时，`_run_team` 从配置好的 profiles 目录加载对应 AGENT.md 并合并其 role/description/resource_id/system_prompt。

同时在 `DynamicOrchestrator.__init__` 新增可选的 `agent_profiles_dir` 参数，预加载 profiles。

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_dynamic_orchestrator.py

import tempfile
import os


def test_team_dispatch_with_profile_id() -> None:
    """dispatch_team resolves profile_id to an agent profile."""
    with tempfile.TemporaryDirectory() as profiles_dir:
        # Create a profile
        alice_dir = os.path.join(profiles_dir, "alice")
        os.makedirs(alice_dir)
        with open(os.path.join(alice_dir, "AGENT.md"), "w") as f:
            f.write(
                "---\n"
                "name: alice\n"
                "role: proponent\n"
                "description: Argues for the proposal\n"
                "---\n\n"
                "You always argue in favor.\n"
            )

        team_args = {
            "topic": "Should we adopt TDD?",
            "agents": [
                {"id": "alice", "profile_id": "alice"},
                {"id": "bob", "role": "opponent", "description": "Against TDD"},
            ],
            "max_rounds": 1,
        }
        gateway = DummyGateway(
            [
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
                # Two agent turns (1 round x 2 agents)
                ModelResponse(text="TDD improves confidence"),
                ModelResponse(text="TDD slows initial dev"),
                _reply_tool_call("Debate concluded."),
            ]
        )
        rm = DummyResourceManager()
        orch = DynamicOrchestrator(
            resource_manager=rm,
            gateway=gateway,
            agent_profiles_dir=profiles_dir,
        )
        result = asyncio.run(orch.run("Debate TDD", ExecutionContext()))
        assert result.conclusion == "Debate concluded."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_orchestrator.py::test_team_dispatch_with_profile_id -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'agent_profiles_dir'`

- [ ] **Step 3: Add agent_profiles_dir to DynamicOrchestrator.__init__**

In `dynamic_orchestrator.py`, add to `__init__` signature:

```python
agent_profiles_dir: str | None = None,
```

And in the body:

```python
self._agent_profiles: dict[str, Any] = {}
if agent_profiles_dir:
    from .agent_profile import AgentProfileLoader
    for profile in AgentProfileLoader.load_dir(agent_profiles_dir):
        self._agent_profiles[profile.name] = profile
```

- [ ] **Step 4: Modify _run_team to resolve profile_id**

In `_run_team`, before preparing per-agent executors, add profile resolution:

```python
# Resolve profiles
resolved_agents: list[dict[str, Any]] = []
for agent in agents:
    agent_copy = dict(agent)
    profile_id = agent_copy.pop("profile_id", None)
    if profile_id and profile_id in self._agent_profiles:
        profile = self._agent_profiles[profile_id]
        # Profile provides defaults; explicit agent fields override
        if not agent_copy.get("role"):
            agent_copy["role"] = profile.role
        if not agent_copy.get("description"):
            agent_copy["description"] = profile.description
        if not agent_copy.get("resource_id") and profile.resource_id:
            agent_copy["resource_id"] = profile.resource_id
        if not agent_copy.get("system_prompt") and profile.system_prompt:
            agent_copy["system_prompt"] = profile.system_prompt
    resolved_agents.append(agent_copy)
agents = resolved_agents
```

Also update the `dispatch_team` tool schema to include `profile_id`:

```python
"profile_id": {
    "type": "string",
    "description": "可选：引用预定义的 AGENT.md profile name",
},
```

- [ ] **Step 5: Update gateway_executor to use system_prompt from agent**

In `_run_team`, modify the `gateway_executor` to accept and use a custom system prompt:

```python
async def gateway_executor(
    agent_id: str, prompt: str, ctx: dict[str, Any]
) -> str:
    sys_prompt = ctx.get("system_prompt", "你是讨论参与者。根据你的角色，针对主题发表观点。")
    messages = [
        ModelMessage(role="system", content=sys_prompt),
        ModelMessage(role="user", content=prompt),
    ]
    request = ModelRequest(messages=tuple(messages))
    response = await self._gateway.generate(request, ExecutionContext())
    return response.text
```

And update TeamRunner's prompt building to pass `system_prompt` through the context dict.

- [ ] **Step 6: Update TeamRunner to pass agent context to executor**

In `team.py` `TeamRunner.run_debate()`, change executor call to pass agent metadata:

```python
# Before:
agent_exec = agent.get("executor", self._executor)
output = await agent_exec(agent["id"], prompt, {})

# After:
agent_exec = agent.get("executor", self._executor)
exec_ctx = {}
if agent.get("system_prompt"):
    exec_ctx["system_prompt"] = agent["system_prompt"]
output = await agent_exec(agent["id"], prompt, exec_ctx)
```

- [ ] **Step 7: Run ALL tests**

Run: `pytest tests/test_dynamic_orchestrator.py tests/test_agent_team.py tests/test_agent_profile.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add babybot/agent_kernel/dynamic_orchestrator.py babybot/agent_kernel/team.py tests/test_dynamic_orchestrator.py
git commit -m "feat: dispatch_team supports profile_id for declarative agent identities"
```

---

### Task 6: Full test suite regression check

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Fix any regressions**

- [ ] **Step 3: Final commit if needed**

---

## Summary of deliverables

| Phase | Deliverable | Files |
|-------|-------------|-------|
| 1 | ClaudeCodeExecutor E2E integration tests | `tests/test_claude_code_executor.py` |
| 2 | Per-agent resource_id in dispatch_team | `team.py`, `dynamic_orchestrator.py` |
| 3 | AgentProfile + AgentProfileLoader + profile_id | `agent_profile.py`, `dynamic_orchestrator.py` |
