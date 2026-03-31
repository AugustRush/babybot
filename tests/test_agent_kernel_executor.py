from __future__ import annotations

import asyncio
from pathlib import Path

from babybot.agent_kernel import (
    ExecutionContext,
    ExecutorPolicy,
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
from babybot.context import Tape


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


class CountingViewTool(Tool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "_workspace_view_text_file"

    @property
    def description(self) -> str:
        return "View workspace file."

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        }

    async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(ok=True, content=f"view:{args['file_path']}")


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


def test_single_agent_executor_fails_fast_after_consecutive_no_progress_turns() -> None:
    class StuckModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            self.calls += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"call-{self.calls}",
                        name="add",
                        arguments={"a": 1, "b": 2},
                    ),
                ),
            )

    registry = ToolRegistry()
    tool = CountingAddTool()
    registry.register(tool, group="math")
    model = StuckModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="loop"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "failed"
    assert "No progress" in result.error
    assert tool.calls == 3
    assert model.calls < 10


def test_single_agent_executor_fails_fast_on_exploration_only_turns() -> None:
    class ExploringModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del request, context
            self.calls += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"call-{self.calls}",
                        name="_workspace_view_text_file",
                        arguments={"file_path": f"skills/demo_{self.calls}.md"},
                    ),
                ),
            )

    registry = ToolRegistry()
    tool = CountingViewTool()
    registry.register(tool, group="code")
    model = ExploringModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="find and edit a skill"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "failed"
    assert "No progress" in result.error
    assert tool.calls <= 3
    assert model.calls <= 4


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


def test_single_agent_executor_isolates_parallel_tool_exceptions() -> None:
    class ExplodingTool(Tool):
        @property
        def name(self) -> str:
            return "explode"

        @property
        def description(self) -> str:
            return "Raise an exception."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args, context
            raise RuntimeError("boom")

    class ParallelToolModel(ModelProvider):
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
                        ModelToolCall(call_id="c1", name="add", arguments={"a": 2, "b": 3}),
                        ModelToolCall(call_id="c2", name="explode", arguments={}),
                    )
                )
            tool_messages = [msg.content for msg in request.messages if msg.role == "tool"]
            return ModelResponse(text=" | ".join(tool_messages))

    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    registry.register(ExplodingTool(), group="math")
    executor = SingleAgentExecutor(model=ParallelToolModel(), tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="parallel"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "5" in result.output
    assert "Tool error:" in result.output
    assert "boom" in result.output


def test_single_agent_executor_persists_tool_call_and_result_to_tape() -> None:
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    executor = SingleAgentExecutor(model=TwoStepModel(), tools=registry)
    tape = Tape("chat1")
    context = ExecutionContext(session_id="s1", state={"tape": tape})

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="compute"),
            context,
        )
    )

    assert result.status == "succeeded"
    kinds = [entry.kind for entry in tape.entries]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    tool_call = next(entry for entry in tape.entries if entry.kind == "tool_call")
    tool_result = next(entry for entry in tape.entries if entry.kind == "tool_result")
    assert tool_call.payload["name"] == "add"
    assert tool_result.payload["name"] == "add"
    assert tool_result.payload["ok"] is True


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


def test_single_agent_executor_truncates_huge_tool_outputs_before_next_model_call() -> None:
    huge = "A" * 50000

    class HugeTool(Tool):
        @property
        def name(self) -> str:
            return "huge"

        @property
        def description(self) -> str:
            return "Return huge output."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args, context
            return ToolResult(ok=True, content=huge)

    class InspectingModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0
            self.last_tool_content = ""

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            del context
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=(ModelToolCall(call_id="c1", name="huge", arguments={}),)
                )
            tool_messages = [msg for msg in request.messages if msg.role == "tool"]
            self.last_tool_content = tool_messages[-1].content
            return ModelResponse(text="done")

    registry = ToolRegistry()
    registry.register(HugeTool(), group="misc")
    model = InspectingModel()
    executor = SingleAgentExecutor(model=model, tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="huge output"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert len(model.last_tool_content) < len(huge)
    assert "truncated" in model.last_tool_content.lower()
