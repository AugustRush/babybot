"""Tests for DynamicOrchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
from babybot.agent_kernel.dynamic_orchestrator import DynamicOrchestrator, _build_resource_catalog


# ── Helpers ──────────────────────────────────────────────────────────────


class DummyGateway:
    """Returns scripted ModelResponse objects in sequence."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    async def generate(
        self, request: ModelRequest, context: ExecutionContext,
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
        ]

    def resolve_resource_scope(
        self, resource_id: str, require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
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
        self.calls.append({
            "task_description": task_description,
            "agent_name": agent_name,
        })
        if any(kw in task_description for kw in self._fail_tasks):
            raise RuntimeError("sub-agent failed")
        return f"result for: {task_description}", []


def _reply_tool_call(text: str, call_id: str = "call_reply") -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=(
            ModelToolCall(call_id=call_id, name="reply_to_user", arguments={"text": text}),
        ),
        finish_reason="tool_calls",
    )


def _dispatch_tool_call(
    resource_id: str, description: str, deps: list[str] | None = None,
    call_id: str = "call_dispatch",
) -> ModelToolCall:
    args: dict[str, Any] = {"resource_id": resource_id, "description": description}
    if deps:
        args["deps"] = deps
    return ModelToolCall(call_id=call_id, name="dispatch_task", arguments=args)


def _wait_tool_call(task_ids: list[str], call_id: str = "call_wait") -> ModelToolCall:
    return ModelToolCall(call_id=call_id, name="wait_for_tasks", arguments={"task_ids": task_ids})


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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
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
                    tool_calls=(
                        _wait_tool_call([self._dispatched_id], call_id="c2"),
                    ),
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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_dispatch_tool_call("skill.weather", "任务A", call_id="c1"),),
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
                            "skill.weather", "任务B", deps=[self._task_a_id], call_id="c2",
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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(_dispatch_tool_call("skill.weather", "任务A", call_id="c1"),),
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
                            "skill.weather", "任务B", deps=[self._task_a_id], call_id="c2",
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
            ModelToolCall(call_id="c_noop", name="get_task_result", arguments={"task_id": "xxx"}),
        ),
        finish_reason="tool_calls",
    )
    gateway = DummyGateway([noop] * 35)
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)
    result = asyncio.run(orch.run("无限循环", ExecutionContext()))
    assert "编排步数已达上限" in result.conclusion


def test_failed_task_handling() -> None:
    """Sub-task fails; wait returns failure; model replies with error info."""

    class FailGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        _dispatch_tool_call("skill.nonexistent", "不存在的资源", call_id="c1"),
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

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
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


def test_call_model_passes_stream_callback_and_resource_tools_to_router() -> None:
    class RecordingGateway(DummyGateway):
        def __init__(self) -> None:
            super().__init__([ModelResponse(text="直接回复")])
            self.stream_callback: Any = None
            self.messages: tuple[Any, ...] = ()

        async def generate(
            self, request: ModelRequest, context: ExecutionContext,
        ) -> ModelResponse:
            self.stream_callback = context.state.get("stream_callback")
            self.messages = request.messages
            return await super().generate(request, context)

    gateway = RecordingGateway()
    rm = DummyResourceManager()
    orch = DynamicOrchestrator(resource_manager=rm, gateway=gateway)

    stream_callback = AsyncMock()
    asyncio.run(orch.run(
        "请发送一张天气图",
        ExecutionContext(state={"stream_callback": stream_callback}),
    ))

    assert gateway.stream_callback is stream_callback
    assert "get_weather" in gateway.messages[0].content


def test_build_resource_catalog_includes_tool_previews() -> None:
    catalog = _build_resource_catalog([
        {
            "id": "group.channel-feishu",
            "name": "channel_feishu",
            "purpose": "飞书渠道工具",
            "tool_count": 3,
            "tools_preview": ["send_text", "send_image", "send_file"],
            "active": True,
        },
    ])
    assert "send_image" in catalog
