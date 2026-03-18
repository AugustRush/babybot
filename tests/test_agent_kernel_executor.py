from __future__ import annotations

import asyncio
from pathlib import Path

from babybot.agent_kernel import (
    ExecutionContext,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    SingleAgentExecutor,
    TaskContract,
    Tool,
    ToolContext,
    ToolLease,
    ToolRegistry,
    ToolResult,
)


class AddTool(Tool):
    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Add two integers."

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

    async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content=str(int(args["a"]) + int(args["b"])))


class TwoStepModel(ModelProvider):
    async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
        last = request.messages[-1]
        if last.role == "tool":
            return ModelResponse(text=f"final={last.content}")
        return ModelResponse(
            tool_calls=(ModelToolCall(call_id="c1", name="add", arguments={"a": 1, "b": 2}),)
        )


class CountingAddTool(AddTool):
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
        self.calls += 1
        return await super().invoke(args, context)


def test_single_agent_executor_runs_tool_then_returns_final_text() -> None:
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    executor = SingleAgentExecutor(model=TwoStepModel(), tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="compute"),
            ExecutionContext(session_id="s1"),
        )
    )
    assert result.status == "succeeded"
    assert result.output == "final=3"


def test_single_agent_executor_respects_tool_lease() -> None:
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    executor = SingleAgentExecutor(model=TwoStepModel(), tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(
                task_id="t1",
                description="compute",
                lease=ToolLease(include_groups=("search",)),
            ),
            ExecutionContext(session_id="s1"),
        )
    )
    assert result.status == "succeeded"
    assert "Tool unavailable: add" in result.output


def test_single_agent_executor_sets_tool_call_id_and_keeps_assistant_tool_calls() -> None:
    class InspectingModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="call-1", name="add", arguments={"a": 2, "b": 5}
                        ),
                    ),
                )
            return ModelResponse(text="done")

    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = InspectingModel()
    executor = SingleAgentExecutor(model=model, tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="compute"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    second_messages = model.requests[1].messages
    assistant_messages = [msg for msg in second_messages if msg.role == "assistant"]
    assert len(assistant_messages) == 1
    assert len(assistant_messages[0].tool_calls) == 1
    assert assistant_messages[0].tool_calls[0].call_id == "call-1"
    tool_messages = [msg for msg in second_messages if msg.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-1"


def test_single_agent_executor_continues_when_model_hits_length_finish_reason() -> None:
    class TruncatedModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(text="Part A", finish_reason="length")
            return ModelResponse(text=" + Part B", finish_reason="stop")

    executor = SingleAgentExecutor(model=TruncatedModel(), tools=ToolRegistry())
    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="continue"),
            ExecutionContext(session_id="s1"),
        )
    )
    assert result.status == "succeeded"
    assert result.output == "Part A + Part B"


def test_single_agent_executor_blocks_repeated_identical_tool_calls() -> None:
    class LoopingModel(ModelProvider):
        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"call-{len(request.messages)}",
                        name="add",
                        arguments={"a": 1, "b": 2},
                    ),
                ),
            )

    registry = ToolRegistry()
    tool = CountingAddTool()
    registry.register(tool, group="math")
    executor = SingleAgentExecutor(model=LoopingModel(), tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="loop"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "failed"
    assert tool.calls == 3


def test_single_agent_executor_collects_usage_metadata() -> None:
    class UsageModel(ModelProvider):
        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            return ModelResponse(
                text="done",
                metadata={
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                },
            )

    executor = SingleAgentExecutor(model=UsageModel(), tools=ToolRegistry())
    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="usage"),
            ExecutionContext(session_id="s1"),
        )
    )
    assert result.status == "succeeded"
    assert result.metadata["prompt_tokens"] == 10
    assert result.metadata["completion_tokens"] == 5
    assert result.metadata["total_tokens"] == 15


def test_single_agent_executor_collects_tool_artifacts_into_context(tmp_path: Path) -> None:
    image_path = tmp_path / "pig.png"
    image_path.write_bytes(b"png")

    class ArtifactTool(Tool):
        @property
        def name(self) -> str:
            return "make_image"

        @property
        def description(self) -> str:
            return "Create image."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args, context
            return ToolResult(
                ok=True,
                content="created",
                artifacts=[str(image_path)],
            )

    class ArtifactModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=(
                        ModelToolCall(call_id="c1", name="make_image", arguments={}),
                    )
                )
            assert request.messages[-1].role == "tool"
            return ModelResponse(text="done")

    registry = ToolRegistry()
    registry.register(ArtifactTool(), group="image")
    executor = SingleAgentExecutor(model=ArtifactModel(), tools=registry)
    context = ExecutionContext(session_id="s1")

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="draw"),
            context,
        )
    )

    assert result.status == "succeeded"
    assert context.state["media_paths_collected"] == [str(image_path.resolve())]


def test_single_agent_executor_enforces_token_budget() -> None:
    class BudgetModel(ModelProvider):
        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            return ModelResponse(
                text="done",
                metadata={
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    }
                },
            )

    executor = SingleAgentExecutor(model=BudgetModel(), tools=ToolRegistry())
    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="budget"),
            ExecutionContext(session_id="s1", state={"max_model_tokens": 10}),
        )
    )

    assert result.status == "failed"
    assert "token budget exceeded" in result.error.lower()
