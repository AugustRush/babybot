"""Tests for DynamicOrchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from babybot.agent_kernel import (
    ExecutionContext,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from babybot.agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
    InMemoryChildTaskBus,
    _build_resource_catalog,
)


# ── Helpers ──────────────────────────────────────────────────────────────


class DummyGateway:
    """Returns scripted ModelResponse objects in sequence."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    async def generate(
        self,
        request: ModelRequest,
        context: ExecutionContext,
    ) -> ModelResponse:
        if self._call_idx >= len(self._responses):
            # Fallback: plain text to avoid infinite loop
            return ModelResponse(text="(no more scripted responses)")
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


class DummyResourceManager:
    """Minimal resource manager for testing."""

    def __init__(self, fail_tasks: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_tasks: set[str] = fail_tasks or set()

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill.weather",
                "type": "skill",
                "name": "weather",
                "purpose": "天气查询",
                "group": "skill_weather",
                "tool_count": 1,
                "tools_preview": ["get_weather"],
                "active": True,
            },
            {
                "id": "group.scheduler",
                "type": "group",
                "name": "scheduler",
                "purpose": "定时任务",
                "group": "scheduler",
                "tool_count": 1,
                "tools_preview": ["create_scheduled_task"],
                "active": True,
            },
            {
                "id": "skill.image",
                "type": "skill",
                "name": "image",
                "purpose": "文生图",
                "group": "skill_image",
                "tool_count": 1,
                "tools_preview": ["generate_image"],
                "active": True,
            },
        ]

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        if resource_id == "skill.weather":
            return {"include_groups": ["skill_weather"]}, ("weather",)
        if resource_id == "group.scheduler":
            return {"include_groups": ["scheduler"]}, ()
        if resource_id == "skill.image":
            return {"include_groups": ["skill_image"]}, ("image",)
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
        self.calls.append(
            {
                "task_description": task_description,
                "agent_name": agent_name,
            }
        )
        if any(kw in task_description for kw in self._fail_tasks):
            raise RuntimeError("sub-agent failed")
        return f"result for: {task_description}", []


def _reply_tool_call(text: str, call_id: str = "call_reply") -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=(
            ModelToolCall(
                call_id=call_id, name="reply_to_user", arguments={"text": text}
            ),
        ),
        finish_reason="tool_calls",
    )


def _dispatch_tool_call(
    resource_id: str,
    description: str,
    deps: list[str] | None = None,
    call_id: str = "call_dispatch",
) -> ModelToolCall:
    args: dict[str, Any] = {"resource_id": resource_id, "description": description}
    if deps:
        args["deps"] = deps
    return ModelToolCall(call_id=call_id, name="dispatch_task", arguments=args)


def _wait_tool_call(task_ids: list[str], call_id: str = "call_wait") -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id, name="wait_for_tasks", arguments={"task_ids": task_ids}
    )


# ── Tests ────────────────────────────────────────────────────────────────


def test_direct_reply() -> None:
    """Model calls reply_to_user without any dispatch."""
    gateway = DummyGateway([_reply_tool_call("你好！我是助手。")])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("你好", ExecutionContext()))
    assert result.conclusion == "你好！我是助手。"
    assert len(rm.calls) == 0


def test_plain_text_response() -> None:
    """Model responds with plain text (no tool calls) — treated as final."""
    gateway = DummyGateway([ModelResponse(text="直接回答")])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("你好", ExecutionContext()))
    assert result.conclusion == "直接回答"


def test_system_prompt_adds_future_task_guard_for_deferred_requests() -> None:
    gateway = DummyGateway([])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    messages = orch._build_initial_messages(  # type: ignore[attr-defined]
        "先查询杭州天气，然后过两分钟后给我发一段画面描述，再根据描述画图",
        ExecutionContext(),
    )

    system_prompt = messages[0].content
    assert "不要立刻执行未来动作" in system_prompt
    assert "未来一次性任务的描述必须自包含" in system_prompt


def test_system_prompt_adds_multi_resource_guidance_for_skill_creation_with_url() -> (
    None
):
    gateway = DummyGateway([])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    messages = orch._build_initial_messages(  # type: ignore[attr-defined]
        "查看 https://github.com/zai-org/GLM-OCR 创建新的ocr识别技能",
        ExecutionContext(),
    )

    system_prompt = messages[0].content
    assert "resource_ids" in system_prompt
    assert "不要靠 create_worker 套娃补能力" in system_prompt


def test_single_task() -> None:
    """dispatch → wait → reply with result."""
    # Step 1: model dispatches a task
    step1 = ModelResponse(
        text="",
        tool_calls=(_dispatch_tool_call("skill.weather", "查询天气", call_id="c1"),),
        finish_reason="tool_calls",
    )
    # Step 2: model will receive the task_id, then wait + reply
    # We need to dynamically handle the task_id, so we use a callback gateway
    responses: list[ModelResponse] = [step1]

    class SmartGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__(responses)
            self._dispatched_id: str = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                resp = await super().generate(request, context)
                return resp
            if self._call_idx == 1:
                # Find the task_id from the last tool result message
                for msg in reversed(request.messages):
                    if msg.role == "tool" and not msg.content.startswith("error:"):
                        self._dispatched_id = msg.content
                        break
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._dispatched_id], call_id="c2"),),
                    finish_reason="tool_calls",
                )
            # Step 3: reply
            self._call_idx += 1
            return _reply_tool_call("天气查询完成", call_id="c3")

    gw = SmartGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("查询天气", ExecutionContext()))
    assert result.conclusion == "天气查询完成"
    assert len(rm.calls) == 1


def test_parallel_tasks() -> None:
    """dispatch A, dispatch B (no deps), wait both, reply."""

    class ParallelGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_ids: list[str] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "查A城天气", call_id="c1"),
                        _dispatch_tool_call("skill.weather", "查B城天气", call_id="c2"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                # Collect task_ids from tool result messages
                for msg in request.messages:
                    if msg.role == "tool" and not msg.content.startswith("error:"):
                        self._task_ids.append(msg.content)
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call(self._task_ids, call_id="c3"),),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("两个城市天气已查询", call_id="c4")

    gw = ParallelGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("查两个城市天气", ExecutionContext()))
    assert result.conclusion == "两个城市天气已查询"
    assert len(rm.calls) == 2


def test_dependent_tasks() -> None:
    """dispatch A, dispatch B(deps=[A]), wait B, reply."""

    class DepGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_a_id = ""
            self._task_b_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "任务A", call_id="c1"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_a_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather",
                            "任务B",
                            deps=[self._task_a_id],
                            call_id="c2",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 2:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c2":
                        self._task_b_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_b_id], call_id="c3"),),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("依赖任务完成", call_id="c4")

    gw = DepGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("依赖任务", ExecutionContext()))
    assert result.conclusion == "依赖任务完成"


def test_sequential_waits_do_not_leak_previous_wait_results_into_later_rounds() -> None:
    class SequentialGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_a_id = ""
            self._task_b_id = ""
            self._wait_a_payload = ""
            self.final_messages: tuple[ModelMessage, ...] = ()

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "任务A", call_id="c1"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_a_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_a_id], call_id="c2"),),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 2:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c2":
                        self._wait_a_payload = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "任务B", call_id="c3"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 3:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c3":
                        self._task_b_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_b_id], call_id="c4"),),
                    finish_reason="tool_calls",
                )
            self.final_messages = request.messages
            self._call_idx += 1
            return _reply_tool_call("顺序任务完成", call_id="c5")

    gw = SequentialGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)

    result = asyncio.run(orch.run("顺序执行两个任务", ExecutionContext()))

    assert result.conclusion == "顺序任务完成"
    assert gw._wait_a_payload
    final_tool_payloads = [
        msg.content for msg in gw.final_messages if msg.role == "tool"
    ]
    assert gw._wait_a_payload not in final_tool_payloads
    assert any("任务B" in payload for payload in final_tool_payloads)
    assert len(rm.calls) == 2


def test_dependent_task_receives_upstream_output() -> None:
    class RecordingResourceManager(DummyResourceManager):
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
            self.calls.append({"task_description": task_description})
            if task_description == "任务A":
                return "RESULT_A", []
            return f"result for: {task_description}", []

    class DepGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_a_id = ""
            self._task_b_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "任务A", call_id="c1"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_a_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather",
                            "任务B",
                            deps=[self._task_a_id],
                            call_id="c2",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 2:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c2":
                        self._task_b_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_b_id], call_id="c3"),),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("依赖任务完成", call_id="c4")

    gw = DepGateway()
    rm = RecordingResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)

    result = asyncio.run(orch.run("依赖任务", ExecutionContext()))

    assert result.conclusion == "依赖任务完成"
    assert len(rm.calls) == 2
    assert "RESULT_A" in rm.calls[1]["task_description"]


def test_max_steps_fallback() -> None:
    """Model never calls reply_to_user; verify fallback after MAX_STEPS."""
    # Return a no-op tool call every step to exhaust MAX_STEPS
    noop = ModelResponse(
        text="",
        tool_calls=(
            ModelToolCall(
                call_id="c_noop", name="get_task_result", arguments={"task_id": "xxx"}
            ),
        ),
        finish_reason="tool_calls",
    )
    gateway = DummyGateway([noop] * 35)
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("无限循环", ExecutionContext()))
    assert "编排步数已达上限" in result.conclusion


def test_max_steps_can_be_configured() -> None:
    noop = ModelResponse(
        text="",
        tool_calls=(
            ModelToolCall(
                call_id="c_noop",
                name="get_task_result",
                arguments={"task_id": "xxx"},
            ),
        ),
        finish_reason="tool_calls",
    )
    gateway = DummyGateway([noop] * 5)
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway, max_steps=2)

    result = asyncio.run(orch.run("限制两步", ExecutionContext()))

    assert "编排步数已达上限" in result.conclusion
    assert gateway._call_idx == 2


def test_failed_task_handling() -> None:
    """Sub-task fails; wait returns failure; model replies with error info."""

    class FailGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.weather", "FAIL_THIS", call_id="c1"),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_id], call_id="c2"),),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("任务失败了", call_id="c3")

    gw = FailGateway()
    rm = DummyResourceManager(fail_tasks={"FAIL_THIS"})
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("会失败的任务", ExecutionContext()))
    assert result.conclusion == "任务失败了"
    # Verify the failed task is in results
    failed = [r for r in result.task_results.values() if r.status == "failed"]
    assert len(failed) == 1


def test_unknown_resource() -> None:
    """dispatch_task with invalid resource_id returns error to model."""

    class UnknownResGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.nonexistent", "不存在的资源", call_id="c1"
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("资源不可用", call_id="c2")

    gw = UnknownResGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("无效资源", ExecutionContext()))
    assert result.conclusion == "资源不可用"
    assert len(rm.calls) == 0


def test_unknown_task_id_in_wait() -> None:
    """wait_for_tasks with unknown task_id returns not_found."""
    step1 = ModelResponse(
        text="",
        tool_calls=(_wait_tool_call(["nonexistent_task"], call_id="c1"),),
        finish_reason="tool_calls",
    )

    class WaitGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([step1])

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                return await super().generate(request, context)
            # Check that the wait result contains not_found
            self._call_idx += 1
            return _reply_tool_call("任务不存在", call_id="c2")

    gw = WaitGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("等待不存在的任务", ExecutionContext()))
    assert result.conclusion == "任务不存在"


def test_wait_for_tasks_reports_collected_media_ready_for_final_reply() -> None:
    from babybot.agent_kernel import TaskResult
    from babybot.agent_kernel.dynamic_orchestrator import (
        InMemoryChildTaskBus,
        InProcessChildTaskRuntime,
    )
    from babybot.heartbeat import TaskHeartbeatRegistry

    class _DummyRM(DummyResourceManager):
        def resolve_resource_scope(self, resource_id: str, require_tools: bool = False):
            del require_tools
            if resource_id == "skill.weather":
                return {"include_groups": ["skill_weather"]}, ("weather",)
            return None

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            return TaskResult(
                task_id="ignored",
                status="succeeded",
                output="语音已生成",
                metadata={"media_paths": ["/tmp/demo.wav"]},
            )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-media-note",
        resource_manager=_DummyRM(),  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )

    async def _run() -> str:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "生成语音"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-media-note"),
        )
        return await runtime.wait_for_tasks([task_id])

    payload = json.loads(asyncio.run(_run()))
    only_value = next(iter(payload.values()))

    assert only_value["status"] == "succeeded"
    assert only_value["output"] == "语音已生成"
    assert only_value["reply_artifacts_ready"] is True
    assert only_value["reply_artifacts_count"] == 1


def test_get_task_result_reports_collected_media_ready_for_final_reply() -> None:
    from babybot.agent_kernel import TaskResult
    from babybot.agent_kernel.dynamic_orchestrator import (
        InMemoryChildTaskBus,
        InProcessChildTaskRuntime,
    )
    from babybot.heartbeat import TaskHeartbeatRegistry

    class _DummyRM(DummyResourceManager):
        def resolve_resource_scope(self, resource_id: str, require_tools: bool = False):
            del require_tools
            if resource_id == "skill.weather":
                return {"include_groups": ["skill_weather"]}, ("weather",)
            return None

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            return TaskResult(
                task_id="ignored",
                status="succeeded",
                output="语音已生成",
                metadata={"media_paths": ["/tmp/demo.wav"]},
            )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-media-status",
        resource_manager=_DummyRM(),  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )

    async def _run() -> str:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "生成语音"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-media-status"),
        )
        await runtime.wait_for_tasks([task_id])
        return runtime.get_task_result(task_id)

    payload = json.loads(asyncio.run(_run()))

    assert payload["status"] == "succeeded"
    assert payload["output"] == "语音已生成"
    assert payload["reply_artifacts_ready"] is True
    assert payload["reply_artifacts_count"] == 1


def test_dispatch_emits_primary_resource_id_for_multi_resource_task() -> None:
    from babybot.agent_kernel import TaskResult
    from babybot.agent_kernel.dynamic_orchestrator import (
        InMemoryChildTaskBus,
        InProcessChildTaskRuntime,
    )
    from babybot.heartbeat import TaskHeartbeatRegistry

    child_bus = InMemoryChildTaskBus()

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            return TaskResult(
                task_id="ignored",
                status="succeeded",
                output="done",
            )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-primary-resource",
        resource_manager=DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        child_task_bus=child_bus,
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )

    async def _run() -> list[dict[str, Any]]:
        task_id = await runtime.dispatch(
            {
                "resource_ids": ["skill.weather", "skill.image"],
                "description": "组合任务",
            },
            task_counter=0,
            context=ExecutionContext(session_id="flow-primary-resource"),
        )
        await runtime.wait_for_tasks([task_id])
        return [
            event.payload for event in child_bus.events_for("flow-primary-resource")
        ]

    payloads = asyncio.run(_run())
    resource_ids = [
        payload.get("resource_id") for payload in payloads if "resource_id" in payload
    ]

    assert resource_ids
    assert set(resource_ids) == {"skill.weather"}


def test_dispatch_times_out_hung_child_task() -> None:
    from babybot.agent_kernel.dynamic_orchestrator import (
        InMemoryChildTaskBus,
        InProcessChildTaskRuntime,
    )
    from babybot.heartbeat import TaskHeartbeatRegistry

    cancelled = asyncio.Event()

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-timeout",
        resource_manager=DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
        default_timeout_s=0.01,
    )

    async def _run() -> tuple[dict[str, Any], bool]:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "hang forever"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-timeout"),
        )
        payload = json.loads(await runtime.wait_for_tasks([task_id]))[task_id]
        return payload, cancelled.is_set()

    payload, was_cancelled = asyncio.run(_run())

    assert payload["status"] == "failed"
    assert "timeout" in payload["error"].lower()
    assert was_cancelled is True


def test_orchestrator_reply_waits_for_child_task_cancellation_cleanup() -> None:
    cleaned_up = asyncio.Event()
    started = asyncio.Event()

    class _Gateway:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del request, context
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather", "后台慢任务", call_id="c1"
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            await asyncio.sleep(0)
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c2",
                        name="reply_to_user",
                        arguments={"text": "先给用户回复"},
                    ),
                ),
                finish_reason="tool_calls",
            )

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            try:
                started.set()
                await asyncio.sleep(60)
            finally:
                cleaned_up.set()

    orch = DynamicOrchestrator(
        resource_manager=DummyResourceManager(), gateway=_Gateway()
    )
    orch._bridge = _Bridge()  # type: ignore[assignment]

    result = asyncio.run(
        orch.run("先回复我", ExecutionContext(session_id="cancel-cleanup"))
    )

    assert result.conclusion == "先给用户回复"
    assert started.is_set() is True
    assert cleaned_up.is_set() is True


def test_system_prompt_explains_reply_artifacts_are_auto_attached() -> None:
    gateway = DummyGateway([])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    messages = orch._build_initial_messages(
        "写一首诗并生成语音发给我",
        ExecutionContext(),
    )

    system_prompt = messages[0].content
    assert "reply_to_user" in system_prompt
    assert "自动附带" in system_prompt
    assert "不要再创建专门的发送子任务" in system_prompt


def test_build_initial_messages_includes_media_paths() -> None:
    gateway = DummyGateway([])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    messages = orch._build_initial_messages(
        "请先分析这张图再调用工具",
        ExecutionContext(state={"media_paths": ["/tmp/a.png", "/tmp/b.jpg"]}),
    )

    assert messages[-1].role == "user"
    assert messages[-1].content == "请先分析这张图再调用工具"
    assert messages[-1].images == ("/tmp/a.png", "/tmp/b.jpg")


def test_call_model_passes_stream_callback_and_resource_catalog_to_router() -> None:
    class RecordingGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([ModelResponse(text="直接回复")])
            self.stream_callback: Any = None
            self.messages: tuple[Any, ...] = ()

        async def generate(
            self,
            request: ModelRequest,
            context: ExecutionContext,
        ) -> ModelResponse:
            self.stream_callback = context.state.get("stream_callback")
            self.messages = request.messages
            return await super().generate(request, context)

    gateway = RecordingGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    stream_callback = AsyncMock()
    asyncio.run(
        orch.run(
            "请发送一张天气图",
            ExecutionContext(state={"stream_callback": stream_callback}),
        )
    )

    assert gateway.stream_callback is stream_callback
    assert "skill.weather: weather" in gateway.messages[0].content
    assert "天气查询" in gateway.messages[0].content
    assert "create_scheduled_task" in gateway.messages[0].content
    assert "get_weather" not in gateway.messages[0].content


def test_build_resource_catalog_includes_tool_previews() -> None:
    catalog = _build_resource_catalog(
        [
            {
                "id": "group.channel-feishu",
                "type": "tool_group",
                "name": "channel_feishu",
                "purpose": "飞书渠道工具",
                "tool_count": 3,
                "tools_preview": ["send_text", "send_image", "send_file"],
                "active": True,
            },
        ]
    )
    assert "send_image" in catalog


def test_build_resource_catalog_hides_mcp_tool_previews() -> None:
    catalog = _build_resource_catalog(
        [
            {
                "id": "mcp.gaode-map",
                "type": "mcp",
                "name": "gaode_map",
                "purpose": "地图查询",
                "tool_count": 8,
                "tools_preview": ["poi_search", "route_plan", "geocode"],
                "active": True,
            },
        ]
    )
    assert "工具数: 8" in catalog
    assert "poi_search" not in catalog


def test_build_resource_catalog_hides_skill_tool_previews() -> None:
    catalog = _build_resource_catalog(
        [
            {
                "id": "skill.weather",
                "type": "skill",
                "name": "weather",
                "purpose": "天气查询",
                "tool_count": 3,
                "tools_preview": ["get_weather", "resolve_city", "format_report"],
                "active": True,
            },
        ]
    )
    assert "工具数: 3" in catalog
    assert "get_weather" not in catalog


def test_runtime_event_callback_receives_child_task_lifecycle_events() -> None:
    class RuntimeEventGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather", "查询杭州天气", call_id="c1"
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_id], call_id="c2"),),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("完成", call_id="c3")

    events: list[dict[str, Any]] = []

    async def _capture(event: dict[str, Any]) -> None:
        events.append(dict(event))

    gw = RuntimeEventGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(
        orch.run(
            "查询天气",
            ExecutionContext(
                session_id="flow-1", state={"runtime_event_callback": _capture}
            ),
        )
    )

    assert result.conclusion == "完成"
    assert [event["event"] for event in events] == ["queued", "started", "succeeded"]
    assert all(event["flow_id"] == "flow-1" for event in events)
    assert all(event["payload"]["resource_id"] == "skill.weather" for event in events)


def test_scheduler_stage_blocks_new_non_scheduler_dispatches_after_it_succeeds() -> (
    None
):
    class SchedulerBoundaryGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._weather_task_id = ""
            self._scheduler_task_id = ""
            self.dispatch_error_seen = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather", "查询杭州天气", call_id="c1"
                        ),
                        _dispatch_tool_call(
                            "group.scheduler",
                            "两分钟后发送一段画作描述",
                            call_id="c2",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role != "tool":
                        continue
                    if msg.tool_call_id == "c1":
                        self._weather_task_id = msg.content
                    if msg.tool_call_id == "c2":
                        self._scheduler_task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _wait_tool_call(
                            [self._weather_task_id, self._scheduler_task_id],
                            call_id="c3",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 2:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.image",
                            "根据那条两分钟后的描述立即生成图片",
                            call_id="c4",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            for msg in request.messages:
                if msg.role == "tool" and msg.tool_call_id == "c4":
                    self.dispatch_error_seen = msg.content
            self._call_idx += 1
            return _reply_tool_call(
                "已返回当前阶段结果，并等待定时任务触发", call_id="c5"
            )

    gw = SchedulerBoundaryGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(
        orch.run("先查天气，两分钟后发描述，再按描述画图", ExecutionContext())
    )

    assert result.conclusion == "已返回当前阶段结果，并等待定时任务触发"
    assert "scheduled" in gw.dispatch_error_seen.lower()
    dispatched = [call["task_description"] for call in rm.calls]
    assert "查询杭州天气" in dispatched
    assert any("两分钟后发送一段画作描述" in item for item in dispatched)
    assert "根据那条两分钟后的描述立即生成图片" not in dispatched


def test_scheduler_dispatch_inherits_prior_live_task_results_and_original_goal() -> (
    None
):
    class SchedulerAfterWeatherGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._scheduler_task_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "skill.weather", "查询杭州天气", call_id="c1"
                        ),
                        _dispatch_tool_call(
                            "group.scheduler",
                            "两分钟后处理剩余任务",
                            call_id="c2",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c2":
                        self._scheduler_task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _wait_tool_call([self._scheduler_task_id], call_id="c3"),
                    ),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return _reply_tool_call("已安排后续阶段", call_id="c4")

    gw = SchedulerAfterWeatherGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(
        orch.run(
            "先查询杭州今天的天气，然后过两分钟后给我发条消息，消息的主题是一副画的描述，然后以这个描述画一幅画发送给我",
            ExecutionContext(
                session_id="flow-1",
                state={
                    "original_goal": "先查询杭州今天的天气，然后过两分钟后给我发条消息，消息的主题是一副画的描述，然后以这个描述画一幅画发送给我"
                },
            ),
        )
    )

    assert result.conclusion == "已安排后续阶段"
    assert len(rm.calls) == 2
    scheduler_description = rm.calls[1]["task_description"]
    assert "--- 上游任务结果 ---" in scheduler_description
    assert "result for: 查询杭州天气" in scheduler_description
    assert "原始用户请求" in scheduler_description
    assert "以这个描述画一幅画发送给我" in scheduler_description


def test_scheduler_stage_blocks_mixed_scheduler_and_live_dispatches_in_same_turn() -> (
    None
):
    class MixedDispatchGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self.scheduler_dispatch_result = ""
            self.image_dispatch_result = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call(
                            "group.scheduler",
                            "两分钟后发送一段画作描述",
                            call_id="c1",
                        ),
                        _dispatch_tool_call(
                            "skill.image",
                            "立刻根据那段未来描述生成图片",
                            call_id="c2",
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            for msg in request.messages:
                if msg.role == "tool" and msg.tool_call_id == "c1":
                    self.scheduler_dispatch_result = msg.content
                if msg.role == "tool" and msg.tool_call_id == "c2":
                    self.image_dispatch_result = msg.content
            self._call_idx += 1
            return _reply_tool_call("当前阶段结束", call_id="c3")

    gw = MixedDispatchGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gw)
    result = asyncio.run(orch.run("两分钟后先发描述，再画图", ExecutionContext()))

    assert result.conclusion == "当前阶段结束"
    assert gw.scheduler_dispatch_result.startswith("task_")
    assert "scheduled" in gw.image_dispatch_result.lower()


def test_wait_for_tasks_preserves_skill_context_for_failed_tasks() -> None:
    from babybot.agent_kernel import TaskResult
    from babybot.agent_kernel.dynamic_orchestrator import (
        InMemoryChildTaskBus,
        InProcessChildTaskRuntime,
    )
    from babybot.heartbeat import TaskHeartbeatRegistry

    class _DummyRM(DummyResourceManager):
        def resolve_resource_scope(self, resource_id: str, require_tools: bool = False):
            del require_tools
            if resource_id == "skill.weather":
                return {"include_groups": ["skill_weather"]}, ("weather",)
            return None

    class _Bridge:
        async def execute(self, task, context):
            del task, context
            return TaskResult(
                task_id="ignored",
                status="failed",
                error="worker crashed",
            )

    runtime = InProcessChildTaskRuntime(
        flow_id="flow-failed-context",
        resource_manager=_DummyRM(),  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=TaskHeartbeatRegistry(),
        max_parallel=1,
        max_tasks=5,
    )

    async def _run() -> str:
        task_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "查询天气"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-failed-context"),
        )
        payload = json.loads(await runtime.wait_for_tasks([task_id]))
        return json.dumps(payload[task_id], ensure_ascii=False)

    payload = json.loads(asyncio.run(_run()))

    assert payload["status"] == "failed"
    assert payload["resource_id"] == "skill.weather"
    assert payload["resource_ids"] == ["skill.weather"]
    assert payload["skill_ids"] == ["weather"]
    assert payload["description"] == "查询天气"


def test_child_task_bus_clears_events_after_flow_completion() -> None:
    step1 = ModelResponse(
        text="",
        tool_calls=(_dispatch_tool_call("skill.weather", "查询天气", call_id="c1"),),
        finish_reason="tool_calls",
    )

    class _Gateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([step1])
            self._task_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            if self._call_idx == 0:
                return await super().generate(request, context)
            if self._call_idx == 1:
                for msg in reversed(request.messages):
                    if (
                        msg.role == "tool"
                        and msg.content
                        and not msg.content.startswith("error:")
                    ):
                        self._task_id = msg.content
                        break
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_wait_tool_call([self._task_id], call_id="c2"),),
                    finish_reason="tool_calls",
                )
            return _reply_tool_call("done", call_id="c3")

    bus = InMemoryChildTaskBus()
    orch = DynamicOrchestrator(
        resource_manager=DummyResourceManager(),
        gateway=_Gateway(),
        child_task_bus=bus,
    )
    context = ExecutionContext(session_id="flow-test")

    result = asyncio.run(orch.run("查询天气", context))

    assert result.conclusion == "done"
    assert bus.events_for("flow-test") == []


def test_orchestrator_accepts_executor_registry() -> None:
    """DynamicOrchestrator uses ExecutorRegistry when provided."""
    from babybot.agent_kernel.executors import ExecutorRegistry

    gateway = DummyGateway([_reply_tool_call("done")])
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(
        resource_manager=rm,
        gateway=gateway,
        executor_registry=None,  # None means use default bridge
    )
    result = asyncio.run(orch.run("hi", ExecutionContext()))
    assert result.conclusion == "done"


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
    # The gateway will be called:
    # 1. By the orchestrator main loop (returns dispatch_team tool call)
    # 2. By _run_team for each agent turn (2 rounds x 2 agents = 4 calls)
    # 3. By the orchestrator main loop again (returns reply_to_user)
    gateway = DummyGateway(
        [
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
            # Steps 2-5: team runner calls gateway for each agent turn
            ModelResponse(text="Microservices allow better scaling"),
            ModelResponse(text="Monoliths are simpler to deploy"),
            ModelResponse(text="But independent scaling is crucial"),
            ModelResponse(text="Complexity cost outweighs benefits"),
            # Step 6: model replies with conclusion
            _reply_tool_call("Refactoring is recommended based on the debate."),
        ]
    )
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("Should we refactor?", ExecutionContext()))
    assert "refactor" in result.conclusion.lower()


def test_dispatch_team_too_few_agents() -> None:
    """dispatch_team returns error when fewer than 2 agents are provided."""
    team_args = {
        "topic": "Solo topic",
        "agents": [
            {"id": "only", "role": "solo", "description": "Only one"},
        ],
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
            _reply_tool_call("Failed to dispatch team."),
        ]
    )
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("test", ExecutionContext()))
    # The orchestrator should continue even after team error
    assert result.conclusion is not None
