from __future__ import annotations

import asyncio

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
