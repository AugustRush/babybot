"""Tests for internal child-task event emission in DynamicOrchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
from babybot.agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
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
    def __init__(self) -> None:
        self.config = type("C", (), {"home_dir": "/tmp/babybot-home"})()

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


def test_dynamic_orchestrator_uses_in_memory_state_only_by_default() -> None:
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
    )

    assert not hasattr(orchestrator, "_state_store")


def test_runtime_retries_retryable_failures_before_succeeding() -> None:
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

    assert payload[task_id] == "succeeded: done after retry"
    assert runtime.results[task_id].attempts == 2
    assert [event.event for event in bus.events_for("flow-retry-success")] == [
        "queued",
        "started",
        "retrying",
        "started",
        "succeeded",
    ]
    assert bridge.calls == 2


def test_runtime_dead_letters_after_retry_budget_exhausted() -> None:
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

    assert payload[task_id] == "failed: timeout while calling worker"
    assert runtime.results[task_id].attempts == 2
    assert [event.event for event in bus.events_for("flow-dead-letter")] == [
        "queued",
        "started",
        "retrying",
        "started",
        "dead_lettered",
    ]


def test_runtime_emits_progress_events_from_child_heartbeat() -> None:
    bus = InMemoryChildTaskBus()

    class _ProgressBridge:
        async def execute(self, task, context) -> TaskResult:  # type: ignore[no-untyped-def]
            del task
            heartbeat = context.state["heartbeat"]
            heartbeat.beat(status="下载模型", progress=0.25)
            await asyncio.sleep(0.08)
            heartbeat.beat(status="下载模型", progress=0.75)
            await asyncio.sleep(0.08)
            return TaskResult(task_id="ignored", status="succeeded", output="done")

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-progress",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_ProgressBridge(),  # type: ignore[arg-type]
        child_task_bus=bus,
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )

    async def _run() -> None:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "查询天气"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-progress"),
        )
        await runtime.wait_for_tasks([task_id])

    asyncio.run(_run())

    progress_events = [
        event for event in bus.events_for("flow-progress")
        if event.event == "progress"
    ]
    assert len(progress_events) >= 2
    assert progress_events[0].payload["status"] == "下载模型"
    assert progress_events[0].payload["progress"] == 0.25
    assert progress_events[-1].payload["progress"] == 0.75
