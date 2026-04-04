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


class _CapturingResourceManager(_DummyResourceManager):
    def __init__(self) -> None:
        super().__init__()
        self.last_task_description = ""

    async def run_subagent_task_result(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: Any = None,
        tape_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> TaskResult:
        del lease, agent_name, tape, tape_store, heartbeat, media_paths, skill_ids
        self.last_task_description = task_description
        return TaskResult(
            task_id=notebook_node_id or "worker-1",
            status="succeeded",
            output=f"done: {task_description}",
            metadata={
                "received_notebook_id": getattr(plan_notebook, "notebook_id", ""),
                "received_node_id": notebook_node_id,
            },
        )


def test_dynamic_orchestrator_emits_child_task_lifecycle_events() -> None:
    bus = InMemoryChildTaskBus()
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
        child_task_bus=bus,
    )

    async def _run() -> list[str]:
        seen: list[str] = []

        async def _collect() -> None:
            async for event in bus.subscribe("flow-1"):
                seen.append(event.event)
                if event.event == "succeeded":
                    break

        collector = asyncio.create_task(_collect())
        await orchestrator.run("查天气", ExecutionContext(session_id="flow-1"))
        await collector
        return seen

    events = asyncio.run(_run())
    assert events == ["queued", "started", "succeeded"]


def test_dynamic_orchestrator_uses_in_memory_state_only_by_default() -> None:
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
    )

    assert not hasattr(orchestrator, "_state_store")


def test_resource_bridge_executor_preserves_failed_subagent_status() -> None:
    class _DetailedRM(_DummyResourceManager):
        async def run_subagent_task_result(
            self,
            task_description: str,
            lease: dict[str, Any] | None = None,
            agent_name: str = "Worker",
            tape: Any = None,
            tape_store: Any = None,
            heartbeat: Any = None,
            media_paths: list[str] | None = None,
            skill_ids: list[str] | None = None,
        ) -> TaskResult:
            del (
                task_description,
                lease,
                agent_name,
                tape,
                tape_store,
                heartbeat,
                media_paths,
                skill_ids,
            )
            return TaskResult(
                task_id="worker-1",
                status="failed",
                error="No progress after 3 consecutive tool-only turns.",
                artifacts=("/tmp/demo.pdf",),
            )

    bridge = ResourceBridgeExecutor(
        resource_manager=_DetailedRM(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        bridge.execute(
            task=type(
                "T",
                (),
                {
                    "task_id": "task-1",
                    "description": "generate a pdf",
                    "lease": type(
                        "L",
                        (),
                        {
                            "include_groups": (),
                            "include_tools": (),
                            "exclude_tools": (),
                        },
                    )(),
                    "metadata": {"resource_id": "skill.weather", "skill_ids": []},
                    "deps": (),
                },
            )(),
            context=ExecutionContext(session_id="flow-1", state={}),
        )
    )

    assert result.status == "failed"
    assert "No progress" in result.error
    assert result.artifacts == ("/tmp/demo.pdf",)


def test_resource_bridge_executor_builds_worker_prompt_from_notebook_context() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    rm = _CapturingResourceManager()
    bridge = ResourceBridgeExecutor(
        resource_manager=rm,  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
    )
    notebook = create_root_notebook(goal="repair local pdf skill", flow_id="flow-notebook")
    dep = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Inspect reference",
        objective="collect upstream gaps",
        owner="worker",
    )
    notebook.transition_node(
        dep.node_id,
        "completed",
        summary="reference inspected",
        detail="Missing design/design.md and README examples.",
        metadata={"progress": True},
    )
    node = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Apply fixes",
        objective="patch local skill",
        owner="worker",
        deps=(dep.node_id,),
    )

    result = asyncio.run(
        bridge.execute(
            task=type(
                "T",
                (),
                {
                    "task_id": "task-1",
                    "description": "apply the local fixes",
                    "lease": type(
                        "L",
                        (),
                        {
                            "include_groups": (),
                            "include_tools": (),
                            "exclude_tools": (),
                        },
                    )(),
                    "metadata": {
                        "resource_id": "skill.weather",
                        "skill_ids": [],
                        "notebook_node_id": node.node_id,
                        "upstream_results": {"legacy-dep": "legacy blob that should not be injected"},
                    },
                    "deps": ("legacy-dep",),
                },
            )(),
            context=ExecutionContext(
                session_id="flow-1",
                state={
                    "plan_notebook": notebook,
                    "plan_notebook_id": notebook.notebook_id,
                    "current_notebook_node_id": node.node_id,
                    "original_goal": "参考仓库对本地 pdf 技能查漏补缺",
                },
            ),
        )
    )

    assert result.status == "succeeded"
    assert "[Current Step]" in rm.last_task_description
    assert "[Direct Dependencies]" in rm.last_task_description
    assert "Missing design/design.md" in rm.last_task_description
    assert "--- upstream_results ---" not in rm.last_task_description
    assert result.metadata["received_notebook_id"] == notebook.notebook_id
    assert result.metadata["received_node_id"] == node.node_id


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

    assert payload[task_id]["status"] == "succeeded"
    assert payload[task_id]["output"] == "done after retry"
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

    assert payload[task_id]["status"] == "failed"
    assert payload[task_id]["error"] == "timeout while calling worker"
    assert runtime.results[task_id].attempts == 2
    assert [event.event for event in bus.events_for("flow-dead-letter")] == [
        "queued",
        "started",
        "retrying",
        "started",
        "dead_lettered",
    ]


def test_runtime_caps_retry_budget_to_guard_rail() -> None:
    runtime = InProcessChildTaskRuntime(
        flow_id="flow-cap",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_ScriptedBridge([]),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        max_retries=999,
    )

    assert runtime._max_retries <= 8


def test_runtime_cancel_all_resets_cancelling_flag_when_gather_raises(monkeypatch) -> None:
    runtime = InProcessChildTaskRuntime(
        flow_id="flow-cancel",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_ScriptedBridge([]),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(10))
        runtime._in_flight["t1"] = task

        async def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "babybot.agent_kernel.dynamic_orchestrator.asyncio.gather",
            _boom,
        )

        async def _run() -> None:
            with pytest.raises(RuntimeError):
                await runtime.cancel_all()

        import pytest

        loop.run_until_complete(_run())
        assert runtime._cancelling is False
    finally:
        for pending in list(asyncio.all_tasks(loop)):
            pending.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


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
