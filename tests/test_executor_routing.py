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

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
        self.calls.append(task.task_id)
        return TaskResult(
            task_id=task.task_id,
            status="succeeded",
            output=f"{self.tag}:{task.task_id}",
        )


def test_registry_routes_by_backend() -> None:
    local = FakeExecutor("local")
    claude = FakeExecutor("claude_code")
    registry = ExecutorRegistry(default=local)
    registry.register("claude_code", claude)

    task_local = TaskContract(task_id="t1", description="local task")
    task_claude = TaskContract(
        task_id="t2", description="remote task", metadata={"backend": "claude_code"}
    )
    ctx = ExecutionContext()

    r1 = asyncio.run(registry.execute(task_local, ctx))
    r2 = asyncio.run(registry.execute(task_claude, ctx))

    assert r1.output == "local:t1"
    assert r2.output == "claude_code:t2"
    assert local.calls == ["t1"]
    assert claude.calls == ["t2"]


def test_registry_unknown_backend_uses_default() -> None:
    default = FakeExecutor("default")
    registry = ExecutorRegistry(default=default)
    task = TaskContract(task_id="t1", description="x", metadata={"backend": "unknown"})
    r = asyncio.run(registry.execute(task, ExecutionContext()))
    assert r.output == "default:t1"


def test_registry_no_backend_uses_default() -> None:
    default = FakeExecutor("default")
    registry = ExecutorRegistry(default=default)
    task = TaskContract(task_id="t1", description="x")
    r = asyncio.run(registry.execute(task, ExecutionContext()))
    assert r.output == "default:t1"


def test_registry_list_backends() -> None:
    registry = ExecutorRegistry(default=FakeExecutor("local"))
    registry.register("claude_code", FakeExecutor("cc"))
    registry.register("codex", FakeExecutor("cx"))
    backends = registry.list_backends()
    assert "claude_code" in backends
    assert "codex" in backends
