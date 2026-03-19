"""Tests for internal child-task event emission in DynamicOrchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
from babybot.agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
    FileChildTaskStateStore,
    InMemoryChildTaskBus,
    InProcessChildTaskRuntime,
)
from babybot.agent_kernel.dag_ports import ResourceBridgeExecutor
from babybot.heartbeat import TaskHeartbeatRegistry
from babybot.agent_kernel.types import TaskResult


class _DummyGateway:
    def __init__(self) -> None:
        self._call_idx = 0
        self._task_id = ""

    async def generate(
        self, request: ModelRequest, context: ExecutionContext,
    ) -> ModelResponse:
        del context
        if self._call_idx == 0:
            self._call_idx += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c1",
                        name="dispatch_task",
                        arguments={
                            "resource_id": "skill.weather",
                            "description": "查询天气",
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        if self._call_idx == 1:
            for message in request.messages:
                if message.role == "tool" and message.tool_call_id == "c1":
                    self._task_id = message.content
            self._call_idx += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c2",
                        name="wait_for_tasks",
                        arguments={"task_ids": [self._task_id]},
                    ),
                ),
                finish_reason="tool_calls",
            )
        self._call_idx += 1
        return ModelResponse(
            text="",
            tool_calls=(
                ModelToolCall(
                    call_id="c3",
                    name="reply_to_user",
                    arguments={"text": "完成"},
                ),
            ),
            finish_reason="tool_calls",
        )


class _DummyResourceManager:
    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill.weather",
                "type": "skill",
                "name": "weather",
                "purpose": "天气查询",
                "group": "skill_weather",
                "tool_count": 1,
                "active": True,
            },
        ]

    def resolve_resource_scope(
        self, resource_id: str, require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        del require_tools
        if resource_id == "skill.weather":
            return {"include_groups": ["skill_weather"]}, ("weather",)
        return None

    async def run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: Any = None,
        tape_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        del lease, agent_name, tape, tape_store, heartbeat, media_paths, skill_ids
        return f"done: {task_description}", []


class _ScriptedBridge:
    def __init__(self, results: list[TaskResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def execute(self, task, context) -> TaskResult:  # type: ignore[no-untyped-def]
        del task, context
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        result = self._results[index]
        return TaskResult(
            task_id=result.task_id,
            status=result.status,
            output=result.output,
            error=result.error,
            attempts=result.attempts,
            metadata=dict(result.metadata),
        )


def test_dynamic_orchestrator_emits_child_task_lifecycle_events() -> None:
    bus = InMemoryChildTaskBus()
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
        child_task_bus=bus,
    )

    asyncio.run(orchestrator.run("查天气", ExecutionContext(session_id="flow-1")))

    events = bus.events_for("flow-1")
    assert [event.event for event in events] == ["queued", "started", "succeeded"]
    assert len({event.task_id for event in events}) == 1


def test_dynamic_orchestrator_persists_flow_snapshot(tmp_path) -> None:
    store = FileChildTaskStateStore(tmp_path)
    bus = InMemoryChildTaskBus()
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
        child_task_bus=bus,
        state_store=store,
    )

    asyncio.run(orchestrator.run("查天气", ExecutionContext(session_id="flow-1")))

    payload = json.loads((tmp_path / "flow-1.json").read_text(encoding="utf-8"))
    task_id = next(iter(payload["tasks"]))
    assert payload["flow_id"] == "flow-1"
    assert payload["tasks"][task_id]["status"] == "succeeded"


def test_runtime_restores_completed_results_from_snapshot(tmp_path) -> None:
    store = FileChildTaskStateStore(tmp_path)
    store.save_snapshot(
        "flow-restore",
        {
            "flow_id": "flow-restore",
            "tasks": {
                "task_1_done": {
                    "status": "succeeded",
                    "output": "cached result",
                    "error": "",
                    "attempts": 1,
                    "metadata": {},
                }
            },
        },
    )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-restore",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=ResourceBridgeExecutor(_DummyResourceManager(), _DummyGateway()),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        state_store=store,
    )

    assert runtime.get_task_result("task_1_done") == "succeeded: cached result"


def test_runtime_retries_retryable_failures_before_succeeding(tmp_path) -> None:
    store = FileChildTaskStateStore(tmp_path)
    bus = InMemoryChildTaskBus()
    bridge = _ScriptedBridge([
        TaskResult(task_id="ignored", status="failed", error="timeout while calling worker"),
        TaskResult(task_id="ignored", status="succeeded", output="done after retry"),
    ])
    runtime = InProcessChildTaskRuntime(
        flow_id="flow-retry-success",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
        child_task_bus=bus,
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        state_store=store,
        max_retries=2,
        retry_delay_seconds=lambda attempt: 0.0,
    )

    async def _run() -> tuple[str, str]:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "查询天气"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-retry-success"),
        )
        return task_id, await runtime.wait_for_tasks([task_id])

    task_id, wait_payload = asyncio.run(_run())
    payload = json.loads(wait_payload)
    snapshot = json.loads((tmp_path / "flow-retry-success.json").read_text(encoding="utf-8"))

    assert payload[task_id] == "succeeded: done after retry"
    assert runtime.results[task_id].attempts == 2
    assert [event.event for event in bus.events_for("flow-retry-success")] == [
        "queued",
        "started",
        "retrying",
        "started",
        "succeeded",
    ]
    assert snapshot["tasks"][task_id]["attempts"] == 2
    assert bridge.calls == 2


def test_runtime_dead_letters_after_retry_budget_exhausted(tmp_path) -> None:
    store = FileChildTaskStateStore(tmp_path)
    bus = InMemoryChildTaskBus()
    bridge = _ScriptedBridge([
        TaskResult(task_id="ignored", status="failed", error="timeout while calling worker"),
        TaskResult(task_id="ignored", status="failed", error="timeout while calling worker"),
    ])
    runtime = InProcessChildTaskRuntime(
        flow_id="flow-dead-letter",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
        child_task_bus=bus,
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        state_store=store,
        max_retries=1,
        retry_delay_seconds=lambda attempt: 0.0,
    )

    async def _run() -> tuple[str, str]:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "查询天气"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-dead-letter"),
        )
        return task_id, await runtime.wait_for_tasks([task_id])

    task_id, wait_payload = asyncio.run(_run())
    payload = json.loads(wait_payload)
    snapshot = json.loads((tmp_path / "flow-dead-letter.json").read_text(encoding="utf-8"))

    assert payload[task_id] == "failed: timeout while calling worker"
    assert runtime.results[task_id].attempts == 2
    assert [event.event for event in bus.events_for("flow-dead-letter")] == [
        "queued",
        "started",
        "retrying",
        "started",
        "dead_lettered",
    ]
    assert snapshot["dead_letters"][task_id]["attempts"] == 2
    assert snapshot["dead_letters"][task_id]["last_error"] == "timeout while calling worker"


def test_runtime_restores_unfinished_tasks_as_recoverable(tmp_path) -> None:
    store = FileChildTaskStateStore(tmp_path)
    store.save_snapshot(
        "flow-recoverable",
        {
            "flow_id": "flow-recoverable",
            "tasks": {
                "task_1_running": {
                    "status": "started",
                    "output": "",
                    "error": "",
                    "attempts": 1,
                    "metadata": {},
                }
            },
        },
    )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-recoverable",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=ResourceBridgeExecutor(_DummyResourceManager(), _DummyGateway()),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        state_store=store,
    )

    assert runtime.get_task_result("task_1_running") == "recoverable: previous run interrupted before completion"
