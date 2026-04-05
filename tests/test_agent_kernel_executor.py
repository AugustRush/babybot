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
    async def generate(
        self, request: ModelRequest, context: ExecutionContext
    ) -> ModelResponse:
        last = request.messages[-1]
        if last.role == "tool":
            return ModelResponse(text=f"final={last.content}")
        return ModelResponse(
            tool_calls=(
                ModelToolCall(call_id="c1", name="add", arguments={"a": 1, "b": 2}),
            )
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


class CountingShellTool(Tool):
    def __init__(self) -> None:
        self.calls = 0
        self.commands: list[str] = []

    @property
    def name(self) -> str:
        return "_workspace_execute_shell_command"

    @property
    def description(self) -> str:
        return "Execute a shell command in the workspace."

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        }

    async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
        del context
        self.calls += 1
        self.commands.append(str(args["command"]))
        return ToolResult(ok=True, content=f"shell:{args['command']}")


class RepeatedActionModel(ModelProvider):
    def __init__(self, rounds: int = 4) -> None:
        self.calls = 0
        self.rounds = rounds

    async def generate(
        self, request: ModelRequest, context: ExecutionContext
    ) -> ModelResponse:
        del request, context
        self.calls += 1
        if self.calls <= self.rounds:
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"call-{self.calls}",
                        name="add",
                        arguments={"a": self.calls, "b": 1},
                    ),
                ),
            )
        return ModelResponse(text="action rounds completed")


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


def test_single_agent_executor_sets_tool_call_id_and_keeps_assistant_tool_calls() -> (
    None
):
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
            self.calls += 1
            del context
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id=f"call-{self.calls}",
                            name="_workspace_view_text_file",
                            arguments={"file_path": "skills/still_ignoring_hint.md"},
                        ),
                    ),
                )
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

    assert result.status == "succeeded"
    assert "探索预算已耗尽" in result.output
    assert tool.calls == 3
    assert model.calls == 4


def test_single_agent_executor_blocks_exploration_tools_once_finalize_is_required() -> (
    None
):
    class ExploringAndIgnoringFinalizeModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            self.calls += 1
            del context
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id=f"call-{self.calls}",
                            name="_workspace_view_text_file",
                            arguments={"file_path": "skills/still_ignoring_hint.md"},
                        ),
                    ),
                )
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
    model = ExploringAndIgnoringFinalizeModel()
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

    assert result.status == "succeeded"
    assert "探索预算已耗尽" in result.output
    assert tool.calls == 3
    assert model.calls == 4


def test_single_agent_executor_allows_short_exploration_burst_before_finishing() -> (
    None
):
    class ExploringThenFinishingModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del request, context
            self.calls += 1
            # 2 exploration rounds (within max_no_progress_turns=3), then finish
            if self.calls <= 2:
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
            return ModelResponse(text="done")

    registry = ToolRegistry()
    tool = CountingViewTool()
    registry.register(tool, group="code")
    model = ExploringThenFinishingModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="explore several files then finish"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert result.output == "done"
    assert tool.calls == 2


def test_single_agent_executor_warns_before_exploration_budget_is_exhausted() -> None:
    class ExploringWithHintAwareModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if any(
                msg.role == "system"
                and "连续 2 轮只使用读取/搜索/检查类工具" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(text="已确认当前差异，停止继续探索并返回缺口摘要。")
            call_id = f"call-{len(self.requests)}"
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=call_id,
                        name="_workspace_view_text_file",
                        arguments={"file_path": f"skills/demo_{call_id}.md"},
                    ),
                ),
            )

    registry = ToolRegistry()
    tool = CountingViewTool()
    registry.register(tool, group="code")
    model = ExploringWithHintAwareModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="对照参考仓库查漏补缺"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "缺口摘要" in result.output
    assert tool.calls == 2
    warning_messages = [
        msg.content
        for msg in model.requests[-1].messages
        if msg.role == "system"
        and "连续 2 轮只使用读取/搜索/检查类工具" in msg.content
    ]
    assert len(warning_messages) == 1


def test_single_agent_executor_grants_one_finalize_turn_after_exploration_budget() -> (
    None
):
    class ExploringUntilForcedToFinishModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(text="基于现有证据给出最终差异摘要。")
            call_id = f"call-{len(self.requests)}"
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=call_id,
                        name="_workspace_view_text_file",
                        arguments={"file_path": f"skills/demo_{call_id}.md"},
                    ),
                ),
            )

    registry = ToolRegistry()
    tool = CountingViewTool()
    registry.register(tool, group="code")
    model = ExploringUntilForcedToFinishModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="继续对照参考仓库查漏补缺"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "最终差异摘要" in result.output
    assert tool.calls == 3
    hard_stop_messages = [
        msg.content
        for msg in model.requests[-1].messages
        if msg.role == "system" and "探索预算已耗尽" in msg.content
    ]
    assert len(hard_stop_messages) == 1


def test_single_agent_executor_blocks_extra_exploration_turn_before_finishing() -> (
    None
):
    class ExploringOnceMoreThenFinishingModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if len(self.requests) <= 4:
                call_id = f"call-{len(self.requests)}"
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id=call_id,
                            name="_workspace_view_text_file",
                            arguments={"file_path": f"skills/demo_{call_id}.md"},
                        ),
                    ),
                )
            return ModelResponse(text="完成额外核对并给出最终差异摘要。")

    registry = ToolRegistry()
    tool = CountingViewTool()
    registry.register(tool, group="code")
    model = ExploringOnceMoreThenFinishingModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="继续核对远端仓库与本地技能差异"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "探索预算已耗尽" in result.output
    assert tool.calls == 3
    assert len(model.requests) == 4
    hard_stop_messages = [
        msg.content
        for msg in model.requests[-1].messages
        if msg.role == "system" and "探索预算已耗尽" in msg.content
    ]
    assert len(hard_stop_messages) == 1


def test_single_agent_executor_auto_summarizes_when_finalize_turn_is_ignored() -> (
    None
):
    class ExploringAndIgnoringFinalizeModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.calls += 1
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id=f"call-{self.calls}",
                            name="_workspace_view_text_file",
                            arguments={"file_path": "skills/still_ignoring_hint.md"},
                        ),
                    ),
                )
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
    model = ExploringAndIgnoringFinalizeModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="find and repair the local skill"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "探索预算已耗尽" in result.output
    assert "已收集证据" in result.output
    assert result.metadata["auto_converged"] is True
    assert result.metadata["completion_mode"] == "auto_summary_after_exploration_stall"
    assert tool.calls == 3
    assert model.calls == 4


def test_single_agent_executor_keeps_shell_action_available_on_finalize_turn() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    class ExploringThenWritingShellModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if (
                request.messages
                and request.messages[-1].role == "tool"
                and request.messages[-1].name == "_workspace_execute_shell_command"
            ):
                return ModelResponse(text="本地文件已经写入，结束任务。")
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                tool_names = {tool["function"]["name"] for tool in request.tools}
                assert "_workspace_execute_shell_command" in tool_names
                assert "_workspace_view_text_file" not in tool_names
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="call-shell-write",
                            name="_workspace_execute_shell_command",
                            arguments={"command": "printf 'done' > output.txt"},
                        ),
                    ),
                )
            call_id = f"call-{len(self.requests)}"
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=call_id,
                        name="_workspace_view_text_file",
                        arguments={"file_path": f"skills/demo_{call_id}.md"},
                    ),
                ),
            )

    registry = ToolRegistry()
    view_tool = CountingViewTool()
    shell_tool = CountingShellTool()
    registry.register(view_tool, group="code")
    registry.register(shell_tool, group="code")
    model = ExploringThenWritingShellModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )
    notebook = create_root_notebook(goal="write local skill file", flow_id="flow-shell")
    node = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Skill writer",
        objective="explore briefly, then write a local file",
        owner="worker",
    )
    context = ExecutionContext(
        session_id="s1",
        state={
            "plan_notebook": notebook,
            "plan_notebook_id": notebook.notebook_id,
            "current_notebook_node_id": node.node_id,
        },
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="先探索，再写入本地技能文件"),
            context,
        )
    )

    assert result.status == "succeeded"
    assert result.output == "本地文件已经写入，结束任务。"
    assert view_tool.calls == 3
    assert shell_tool.calls == 1
    assert notebook.progress_marker_count(node.node_id) >= 1


def test_single_agent_executor_still_rejects_read_only_shell_on_finalize_turn() -> (
    None
):
    class ExploringThenReadOnlyShellModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if any(
                msg.role == "system" and "探索预算已耗尽" in msg.content
                for msg in request.messages
            ):
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="call-shell-read",
                            name="_workspace_execute_shell_command",
                            arguments={"command": "cat skills/still_reading.md"},
                        ),
                    ),
                )
            call_id = f"call-{len(self.requests)}"
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id=call_id,
                        name="_workspace_view_text_file",
                        arguments={"file_path": f"skills/demo_{call_id}.md"},
                    ),
                ),
            )

    registry = ToolRegistry()
    view_tool = CountingViewTool()
    shell_tool = CountingShellTool()
    registry.register(view_tool, group="code")
    registry.register(shell_tool, group="code")
    model = ExploringThenReadOnlyShellModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=40, max_no_progress_turns=3),
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="继续探索并在最后尝试只读 shell"),
            ExecutionContext(session_id="s1"),
        )
    )

    assert result.status == "succeeded"
    assert "探索预算已耗尽" in result.output
    assert view_tool.calls == 3
    assert shell_tool.calls == 0


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


def test_single_agent_executor_collects_tool_artifacts_into_context(
    tmp_path: Path,
) -> None:
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
                        ModelToolCall(
                            call_id="c1", name="add", arguments={"a": 2, "b": 3}
                        ),
                        ModelToolCall(call_id="c2", name="explode", arguments={}),
                    )
                )
            tool_messages = [
                msg.content for msg in request.messages if msg.role == "tool"
            ]
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


def test_single_agent_executor_injects_pending_runtime_hints_once() -> None:
    class HintTool(Tool):
        @property
        def name(self) -> str:
            return "reload_skill"

        @property
        def description(self) -> str:
            return "Reload a skill."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args
            context.state.setdefault("pending_runtime_hints", []).append(
                "技能 helper-skill 已热重载。SKILL.md=/tmp/workspace/skills/helper-skill/SKILL.md"
            )
            return ToolResult(ok=True, content="reloaded")

    class HintModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    tool_calls=(
                        ModelToolCall(call_id="c1", name="reload_skill", arguments={}),
                    )
                )
            return ModelResponse(text="done")

    registry = ToolRegistry()
    registry.register(HintTool(), group="basic")
    model = HintModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s1")

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="reload and continue"),
            context,
        )
    )

    assert result.status == "succeeded"
    second_request = model.requests[1]
    runtime_hints = [
        msg.content
        for msg in second_request.messages
        if msg.role == "system" and "技能 helper-skill 已热重载" in msg.content
    ]
    assert len(runtime_hints) == 1
    assert context.state.get("pending_runtime_hints") == []


def test_single_agent_executor_injects_artifact_completion_hint() -> None:
    image_path = Path("/tmp/generated-report.pdf")

    class ArtifactTool(Tool):
        @property
        def name(self) -> str:
            return "generate_report"

        @property
        def description(self) -> str:
            return "Generate a report file."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args, context
            return ToolResult(
                ok=True,
                content=f"done: {image_path}",
                artifacts=[str(image_path)],
            )

    class ArtifactModel(ModelProvider):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del context
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    tool_calls=(
                        ModelToolCall(
                            call_id="c1",
                            name="generate_report",
                            arguments={},
                        ),
                    )
                )
            return ModelResponse(text="done")

    registry = ToolRegistry()
    registry.register(ArtifactTool(), group="basic")
    model = ArtifactModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s1")

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="generate a pdf report"),
            context,
        )
    )

    assert result.status == "succeeded"
    second_request = model.requests[1]
    runtime_hints = [
        msg.content
        for msg in second_request.messages
        if msg.role == "system" and str(image_path) in msg.content
    ]
    assert len(runtime_hints) == 1
    assert "stop and return" in runtime_hints[0].lower()


def test_single_agent_executor_batches_tape_store_writes_per_step() -> None:
    class RecordingTapeStore:
        def __init__(self) -> None:
            self.saved_batches: list[list[str]] = []

        def save_entries(self, chat_id: str, entries: list) -> None:  # type: ignore[no-untyped-def]
            del chat_id
            self.saved_batches.append([entry.kind for entry in entries])

        def save_entry(self, chat_id: str, entry) -> None:  # type: ignore[no-untyped-def]
            del chat_id, entry
            raise AssertionError("expected batched save_entries")

    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    executor = SingleAgentExecutor(model=TwoStepModel(), tools=registry)
    tape = Tape("chat1")
    tape_store = RecordingTapeStore()
    context = ExecutionContext(
        session_id="s1",
        state={"tape": tape, "tape_store": tape_store},
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="compute"),
            context,
        )
    )

    assert result.status == "succeeded"
    assert tape_store.saved_batches == [["tool_call"], ["tool_result"]]


def test_single_agent_executor_logs_turns_and_tool_events_to_notebook() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    executor = SingleAgentExecutor(model=TwoStepModel(), tools=registry)
    notebook = create_root_notebook(goal="compute", flow_id="flow-exec-notebook")
    node = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Compute numbers",
        objective="run add tool and summarize",
        owner="worker",
    )
    context = ExecutionContext(
        session_id="s1",
        state={
            "plan_notebook": notebook,
            "plan_notebook_id": notebook.notebook_id,
            "current_notebook_node_id": node.node_id,
        },
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="compute"),
            context,
        )
    )

    assert result.status == "succeeded"
    logged_node = notebook.get_node(node.node_id)
    assert logged_node.status == "completed"
    assert logged_node.result_text == "final=3"
    assert any(
        event.kind == "observation" and event.metadata.get("stage") == "turn_start"
        for event in logged_node.events
    )
    assert any(event.kind == "tool_call" and event.summary == "add" for event in logged_node.events)
    assert any(
        event.kind == "tool_result" and event.metadata.get("tool_name") == "add"
        for event in logged_node.events
    )
    assert result.metadata["notebook_node_id"] == node.node_id
    assert "final=3" in result.metadata["notebook_summary"]


def test_single_agent_executor_uses_notebook_progress_for_action_loops() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = RepeatedActionModel(rounds=4)
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        policy=ExecutorPolicy(max_steps=10, max_no_progress_turns=3),
    )
    notebook = create_root_notebook(goal="perform several actions", flow_id="flow-progress")
    node = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Action worker",
        objective="run several action steps",
        owner="worker",
    )
    context = ExecutionContext(
        session_id="s1",
        state={
            "plan_notebook": notebook,
            "plan_notebook_id": notebook.notebook_id,
            "current_notebook_node_id": node.node_id,
        },
    )

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="loop through actions"),
            context,
        )
    )

    assert result.status == "succeeded"
    assert result.output == "action rounds completed"
    assert model.calls == 5
    assert notebook.progress_marker_count(node.node_id) >= 4


def test_single_agent_executor_records_artifacts_in_notebook(tmp_path: Path) -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    artifact_path = tmp_path / "report.pdf"
    artifact_path.write_bytes(b"%PDF-1.4")

    class ArtifactTool(Tool):
        @property
        def name(self) -> str:
            return "generate_report"

        @property
        def description(self) -> str:
            return "Generate a report file."

        @property
        def schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def invoke(self, args: dict, context: ToolContext) -> ToolResult:
            del args, context
            return ToolResult(ok=True, content="report created", artifacts=[str(artifact_path)])

    class ArtifactModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            del request, context
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=(
                        ModelToolCall(call_id="c1", name="generate_report", arguments={}),
                    )
                )
            return ModelResponse(text="report ready")

    registry = ToolRegistry()
    registry.register(ArtifactTool(), group="basic")
    notebook = create_root_notebook(goal="generate report", flow_id="flow-artifacts")
    node = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Generate report",
        objective="produce a pdf",
        owner="worker",
    )
    context = ExecutionContext(
        session_id="s1",
        state={
            "plan_notebook": notebook,
            "plan_notebook_id": notebook.notebook_id,
            "current_notebook_node_id": node.node_id,
        },
    )
    executor = SingleAgentExecutor(model=ArtifactModel(), tools=registry)

    result = asyncio.run(
        executor.execute(
            TaskContract(task_id="t1", description="generate report"),
            context,
        )
    )

    assert result.status == "succeeded"
    logged_node = notebook.get_node(node.node_id)
    assert [artifact.path for artifact in logged_node.artifacts] == [str(artifact_path)]
    assert any(event.kind == "artifact" for event in logged_node.events)


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


def test_single_agent_executor_truncates_huge_tool_outputs_before_next_model_call() -> (
    None
):
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

        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
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


# ── Structured event emission tests ──────────────────────────────────────


def test_executor_emits_agent_start_and_agent_end_events() -> None:
    """execute() emits agent_start at the beginning and agent_end at the end."""
    registry = ToolRegistry()
    model = TwoStepModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-events")

    seen_kinds: list[str] = []
    context.on({"agent_start", "agent_end"}, lambda e: seen_kinds.append(e.kind))

    asyncio.run(executor.execute(TaskContract(task_id="t1", description="go"), context))

    assert seen_kinds[0] == "agent_start"
    assert seen_kinds[-1] == "agent_end"


def test_executor_emits_turn_start_and_turn_end_per_step() -> None:
    """Each LLM loop iteration emits a turn_start / turn_end pair."""
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = TwoStepModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-turns")

    turns: list[tuple[str, int]] = []
    context.on(
        {"turn_start", "turn_end"},
        lambda e: turns.append((e.kind, e.data.get("step", 0))),
    )

    asyncio.run(
        executor.execute(TaskContract(task_id="t1", description="add"), context)
    )

    starts = [k for k, _ in turns if k == "turn_start"]
    ends = [k for k, _ in turns if k == "turn_end"]
    assert starts  # at least one turn started
    assert ends  # and ended
    assert len(starts) == len(ends)


def test_executor_emits_llm_request_start_and_end_with_elapsed() -> None:
    """Each model.generate() call is wrapped by llm_request_start / llm_request_end."""
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = TwoStepModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-llm")

    llm_events: list[dict] = []
    context.on(
        {"llm_request_start", "llm_request_end"},
        lambda e: llm_events.append({"kind": e.kind, **e.data}),
    )

    asyncio.run(
        executor.execute(TaskContract(task_id="t1", description="add"), context)
    )

    starts = [e for e in llm_events if e["kind"] == "llm_request_start"]
    ends = [e for e in llm_events if e["kind"] == "llm_request_end"]
    assert starts
    assert ends
    assert len(starts) == len(ends)
    # end events carry elapsed_s
    for end_event in ends:
        assert "elapsed_s" in end_event
        assert end_event["elapsed_s"] >= 0


def test_executor_emits_tool_execution_start_and_end() -> None:
    """Each tool invocation is wrapped by tool_execution_start / tool_execution_end."""
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = TwoStepModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-tool-exec")

    tool_events: list[dict] = []
    context.on(
        {"tool_execution_start", "tool_execution_end"},
        lambda e: tool_events.append({"kind": e.kind, **e.data}),
    )

    asyncio.run(
        executor.execute(TaskContract(task_id="t1", description="add"), context)
    )

    starts = [e for e in tool_events if e["kind"] == "tool_execution_start"]
    ends = [e for e in tool_events if e["kind"] == "tool_execution_end"]
    assert starts
    assert ends
    # end events include ok flag and elapsed_s
    for end_event in ends:
        assert "ok" in end_event
        assert "elapsed_s" in end_event
        assert end_event["elapsed_s"] >= 0


# ── Hook tests ────────────────────────────────────────────────────────────


def test_executor_calls_transform_context_hook_before_llm() -> None:
    """transform_context hook is called before each LLM request; can modify messages."""

    class SimpleTextModel(ModelProvider):
        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            return ModelResponse(text="done")

    registry = ToolRegistry()
    model = SimpleTextModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-transform")

    transform_calls: list[int] = []

    def _transform(messages, ctx):
        transform_calls.append(len(messages))
        return messages  # pass through unchanged

    context.transform_context = _transform

    asyncio.run(
        executor.execute(TaskContract(task_id="t1", description="hello"), context)
    )

    assert len(transform_calls) >= 1, "transform_context should have been called"


def test_executor_calls_before_and_after_tool_call_hooks() -> None:
    """before_tool_call and after_tool_call hooks fire around each tool invocation."""
    registry = ToolRegistry()
    registry.register(AddTool(), group="math")
    model = TwoStepModel()
    executor = SingleAgentExecutor(model=model, tools=registry)
    context = ExecutionContext(session_id="s-hooks")

    before_calls: list[str] = []
    after_calls: list[tuple[str, bool]] = []

    context.before_tool_call = lambda name, args, ctx: before_calls.append(name)
    context.after_tool_call = lambda name, args, result, ctx: after_calls.append(
        (name, result.ok)
    )

    asyncio.run(
        executor.execute(TaskContract(task_id="t1", description="add"), context)
    )

    assert before_calls == ["add"]
    assert after_calls == [("add", True)]


def test_transform_context_hook_message_reduction_emits_context_transform_event() -> (
    None
):
    """When transform_context reduces the message list, a context_transform event is emitted."""
    from babybot.agent_kernel import SkillPack, ToolLease

    class SimpleTextModel(ModelProvider):
        async def generate(
            self, request: ModelRequest, context: ExecutionContext
        ) -> ModelResponse:
            return ModelResponse(text="done")

    registry = ToolRegistry()
    # Use a skill_resolver that returns a SkillPack with a non-empty system_prompt
    # so the initial messages list has [system, user] (2 items).
    # The transform then reduces to 1 item, triggering the context_transform event.
    skill = SkillPack(
        name="s", system_prompt="You are a helper.", tool_lease=ToolLease()
    )
    model = SimpleTextModel()
    executor = SingleAgentExecutor(
        model=model,
        tools=registry,
        skill_resolver=lambda task, ctx: skill,
    )
    context = ExecutionContext(session_id="s-ctx-event")

    transform_events: list[dict] = []
    context.on(
        {"context_transform"},
        lambda e: transform_events.append(e.data),
    )

    def _shrink(messages, ctx):
        # Remove all but the last message to trigger the size-change detection
        return messages[-1:]

    context.transform_context = _shrink

    asyncio.run(executor.execute(TaskContract(task_id="t1", description="hi"), context))

    # context_transform only fires if the message list size actually changed
    assert transform_events, (
        "expected context_transform event when messages were reduced"
    )
    for ev in transform_events:
        assert ev["messages_before"] > ev["messages_after"]
