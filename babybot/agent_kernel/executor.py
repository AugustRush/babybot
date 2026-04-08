"""Reference single-agent executor built on model/tools/skills ports."""

from __future__ import annotations

import json as _json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .loop_guard import LoopGuard, LoopGuardConfig
from .model import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from .plan_notebook import PlanNotebook
from .runtime_state import RuntimeState
from .skills import SkillPack, merge_leases, merge_prompts
from .tools import ToolContext, ToolRegistry
from .executor_history import (
    _build_context_view_messages,
    _build_history_messages,
    _estimate_token_count,
    _extract_keywords,
    _history_entry_text,
)
from .types import AgentEvent, ExecutionContext, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..context import Entry

logger = logging.getLogger(__name__)


@dataclass
class ExecutorPolicy:
    """Policy for one single-agent execution loop."""

    max_steps: int = 8
    max_continuations: int = 2
    max_no_progress_turns: int = 3
    max_tool_result_chars: int = 12000
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)


@dataclass
class ExecutionSession:
    """All mutable state shared across loop iterations in execute().

    Created once in _prepare(), passed through the entire loop.
    This replaces 20+ scattered local variables.
    """

    # Conversation
    messages: list[Any] = field(default_factory=list)
    # Token tracking
    usage_totals: dict[str, int] = field(default_factory=dict)
    max_model_tokens: int = 0
    # Loop guards
    loop_guard: Any = None
    blocked_tool_names: set[str] = field(default_factory=set)
    no_progress_turns: int = 0
    no_progress_limit: int = 5
    # Counters
    tool_call_count: int = 0
    tool_failure_count: int = 0
    loop_guard_block_count: int = 0
    # Flags
    exploration_finish_required: bool = False
    # Tools (read-only after preparation)
    available_tools: list[Any] = field(default_factory=list)
    base_lease: Any = None
    # Execution context
    tool_context: Any = None
    heartbeat: Any = None
    tape: Any = None
    tape_store: Any = None
    notebook: Any = None
    notebook_node_id: str = ""
    max_steps: int = 30
    task_id: str = ""


@dataclass
class ValidationResult:
    """Result of validating a batch of tool calls."""

    valid_calls: list[Any] = field(default_factory=list)
    error_results: list[tuple[Any, str]] = field(default_factory=list)
    forced_finalize_violations: list[Any] = field(default_factory=list)


@dataclass
class SingleAgentExecutor:
    """ExecutorPort implementation.

    It runs a compact agent loop:
    model -> tool_calls -> tool_results -> model ... -> final text

    Lifecycle events (emitted to context.event_bus):
    - agent_start / agent_end: bracket the entire execute() call
    - turn_start / turn_end: bracket each step in the main loop
    - llm_request_start / llm_request_end: bracket each model.generate() call
    - tool_execution_start / tool_execution_end: bracket each tool invocation
    - context_transform: emitted when transformContext hook modifies messages

    Hook points (set on ExecutionContext):
    - transform_context(messages, context) -> messages: modify message list before LLM
    - before_tool_call(tool_name, args, context) -> None: called before each tool invoke
    - after_tool_call(tool_name, args, result, context) -> None: called after each tool
    """

    model: ModelProvider
    tools: ToolRegistry
    skill_resolver: (
        Callable[
            [TaskContract, ExecutionContext], SkillPack | Iterable[SkillPack] | None
        ]
        | None
    ) = None
    policy: ExecutorPolicy = field(default_factory=ExecutorPolicy)

    # ── Event emission helpers ───────────────────────────────────────

    @staticmethod
    def _emit(
        context: ExecutionContext,
        kind: str,
        task_id: str = "",
        step: int = 0,
        **data: Any,
    ) -> None:
        """Emit a structured AgentEvent to the context's EventBus."""
        context.event_bus.emit(
            AgentEvent(
                kind=kind,  # type: ignore[arg-type]
                session_id=context.session_id,
                task_id=task_id,
                step=step,
                data=data,
            )
        )

    # ── Static helpers ───────────────────────────────────────────────

    @staticmethod
    def _truncate_tool_output_for_model(output: str, max_chars: int) -> str:
        if max_chars <= 0 or len(output) <= max_chars:
            return output
        note = (
            f"\n...[truncated {len(output) - max_chars} chars; "
            "narrow the command, file range, or query if more detail is needed]..."
        )
        if len(note) >= max_chars:
            return note[:max_chars]
        keep_budget = max_chars - len(note)
        head = max(1, int(keep_budget * 0.75))
        tail = max(0, keep_budget - head)
        if tail == 0:
            return output[:keep_budget] + note
        return output[:head] + note + output[-tail:]

    @staticmethod
    def _filter_non_exploration_tools(
        tool_schemas: Iterable[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            tool
            for tool in tool_schemas
            if not LoopGuard.is_exploration_tool(tool["function"]["name"])
        )

    @staticmethod
    def _consume_runtime_hint_messages(
        state: dict[str, Any],
    ) -> tuple[ModelMessage, ...]:
        if not isinstance(state, dict):
            return ()
        raw_hints = state.pop("pending_runtime_hints", None)
        if not raw_hints:
            state["pending_runtime_hints"] = []
            return ()
        messages: list[ModelMessage] = []
        for item in raw_hints:
            text = str(item).strip()
            if not text:
                continue
            messages.append(
                ModelMessage(
                    role="system",
                    content=(
                        "Runtime update:\n"
                        f"{text}\n"
                        "Do not assume your earlier skill snapshot is still current. "
                        "If you need to keep editing that skill, use the exact path above."
                    ),
                )
            )
        state["pending_runtime_hints"] = []
        return tuple(messages)

    @staticmethod
    def _resolve_notebook_binding(
        context: ExecutionContext,
    ) -> tuple[PlanNotebook | None, str]:
        binding = RuntimeState(context).notebook_binding()
        if not binding.active:
            return None, ""
        return binding.notebook, binding.node_id

    @classmethod
    def _record_notebook_event(
        cls,
        context: ExecutionContext,
        *,
        kind: str,
        summary: str,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
        progress: bool = False,
    ) -> None:
        notebook, node_id = cls._resolve_notebook_binding(context)
        if notebook is None or not summary.strip():
            return
        event_metadata = dict(metadata or {})
        if progress:
            event_metadata["progress"] = True
        notebook.record_event(
            node_id=node_id,
            kind=kind,  # type: ignore[arg-type]
            summary=summary.strip(),
            detail=detail,
            metadata=event_metadata,
        )

    @classmethod
    def _transition_notebook_node(
        cls,
        context: ExecutionContext,
        *,
        status: str,
        summary: str,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        notebook, node_id = cls._resolve_notebook_binding(context)
        if notebook is None:
            return
        notebook.transition_node(
            node_id,
            status,  # type: ignore[arg-type]
            summary=summary,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def _progress_marker_count(cls, context: ExecutionContext) -> int:
        notebook, node_id = cls._resolve_notebook_binding(context)
        if notebook is None:
            return 0
        return notebook.progress_marker_count(node_id)

    @staticmethod
    def _summarize_notebook_text(text: str, *, limit: int = 160) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return ""
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(1, limit - 3)].rstrip() + "..."

    @classmethod
    def _build_auto_converged_summary(
        cls,
        *,
        task: TaskContract,
        messages: list[ModelMessage],
    ) -> str:
        evidence: list[str] = []
        seen: set[str] = set()
        for message in reversed(messages):
            if message.role != "tool":
                continue
            content = cls._summarize_notebook_text(message.content, limit=240)
            if (
                not content
                or content.startswith("Tool error:")
                or "Exploration budget exhausted." in content
            ):
                continue
            line = f"- {message.name or 'tool'}: {content}"
            if line in seen:
                continue
            seen.add(line)
            evidence.append(line)
            if len(evidence) >= 3:
                break
        evidence.reverse()

        lines = [
            "探索预算已耗尽，已停止继续读取/搜索/检查。",
            f"任务：{str(task.description or '').strip() or task.task_id}",
            "已收集证据：",
        ]
        if evidence:
            lines.extend(evidence)
        else:
            lines.append("- 暂无可复用的有效工具结果。")
        lines.append(
            "结论：当前子任务未收敛到可直接执行的动作。"
            "如需继续推进，上层必须改写目标，或切换到编辑、写入、执行类动作。"
        )
        return "\n".join(lines)

    # ── Main execution loop ──────────────────────────────────────────

    def _validate_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        session: ExecutionSession,
    ) -> ValidationResult:
        """Run the 5 sequential validation checks on each tool call."""
        result = ValidationResult()
        for tool_call in tool_calls:
            # Check 1: Exploration budget
            if (
                session.exploration_finish_required
                and session.loop_guard.is_exploration_call(
                    tool_call.name,
                    tool_call.arguments,
                )
            ):
                result.forced_finalize_violations.append(tool_call)
                result.error_results.append(
                    (
                        tool_call,
                        (
                            "Exploration budget exhausted."
                            "\n[Hint: Read/search/check tools are no longer allowed for this task. "
                            "Return a concise conclusion or blocker summary, or switch to edit/write/action tools.]"
                        ),
                    )
                )
                continue

            # Check 2: Loop guard
            verdict = session.loop_guard.check_call(tool_call.name, tool_call.arguments)
            if verdict.blocked:
                session.loop_guard_block_count += 1
                logger.warning(
                    "Executor loop guard blocked task=%s tool=%s reason=%s",
                    session.task_id,
                    tool_call.name,
                    verdict.reason,
                )
                if verdict.disable_tool:
                    session.blocked_tool_names.add(tool_call.name)
                remaining = [
                    t["function"]["name"]
                    for t in session.available_tools
                    if t["function"]["name"] not in session.blocked_tool_names
                ]
                hint = f"Loop guard: {verdict.reason}"
                if verdict.disable_tool:
                    hint += (
                        f"\nTool '{tool_call.name}' is now disabled for this task."
                        f"\nYou MUST use a different tool. Available tools: {remaining}"
                    )
                else:
                    hint += (
                        "\nDo not repeat the same call immediately."
                        f"\nAvailable tools: {remaining}"
                    )
                result.error_results.append((tool_call, hint))
                continue

            # Check 3: Tool registered and allowed
            registered = self.tools.get(tool_call.name)
            if not registered or not self._tool_allowed(
                registered.group,
                tool_call.name,
                session.base_lease,
            ):
                logger.warning(
                    "Executor tool unavailable task=%s tool=%s",
                    session.task_id,
                    tool_call.name,
                )
                hint_parts = [
                    f"Tool unavailable: {tool_call.name}",
                ]
                if registered:
                    hint_parts.append(
                        f"This tool belongs to the '{registered.group}' group"
                        f" which is not included in the current execution scope."
                    )
                hint_parts.append(
                    "[Hint: Only use tools listed in your current tool set."
                    " Do NOT call tools you learned about through inspect_skills"
                    " or other means — they may not be available in this context."
                    " Try a different approach with available tools.]"
                )
                result.error_results.append((tool_call, "\n".join(hint_parts)))
                session.tool_failure_count += 1
                continue

            # Check 4: Argument parse error
            if isinstance(tool_call.arguments, dict) and tool_call.arguments.get(
                "__tool_argument_parse_error__"
            ):
                logger.warning(
                    "Executor invalid arguments JSON task=%s tool=%s",
                    session.task_id,
                    tool_call.name,
                )
                result.error_results.append(
                    (
                        tool_call,
                        (
                            f"Tool argument JSON parse error for {tool_call.name}: "
                            f"{tool_call.arguments.get('__raw_arguments__', '')}"
                            "\n[Hint: Fix the JSON syntax in your tool arguments.]"
                        ),
                    )
                )
                session.tool_failure_count += 1
                continue

            # Check 5: Argument validation
            tool_call.arguments = self._cast_tool_arguments(
                schema=registered.tool.schema,
                args=tool_call.arguments,
            )
            validation_error = self._validate_tool_arguments(
                schema=registered.tool.schema,
                args=tool_call.arguments,
            )
            if validation_error:
                logger.warning(
                    "Executor argument validation failed task=%s tool=%s error=%s",
                    session.task_id,
                    tool_call.name,
                    validation_error,
                )
                result.error_results.append(
                    (
                        tool_call,
                        (
                            f"Tool argument validation failed for {tool_call.name}: {validation_error}"
                            "\n[Hint: Check parameter types and values against the tool schema.]"
                        ),
                    )
                )
                session.tool_failure_count += 1
                continue

            result.valid_calls.append((tool_call, registered))
        return result

    def _prepare(
        self,
        task: TaskContract,
        context: ExecutionContext,
    ) -> ExecutionSession:
        """Initialise all per-execution state and return an ExecutionSession."""
        self._emit(
            context, "agent_start", task_id=task.task_id, description=task.description
        )

        skills = self._resolve_skills(task, context)
        base_lease = task.lease
        for skill in skills:
            base_lease = merge_leases(base_lease, skill.tool_lease)

        system_prompt = merge_prompts(skills)
        messages: list[ModelMessage] = []
        if system_prompt:
            messages.append(ModelMessage(role="system", content=system_prompt))

        tape = context.state.get("tape")
        if tape is not None:
            history_budget = context.state.get("context_history_tokens", 2000)
            tape_store = context.state.get("tape_store")
            memory_store = context.state.get("memory_store")
            messages.extend(
                _build_history_messages(
                    tape,
                    history_budget,
                    query=task.description,
                    tape_store=tape_store,
                    memory_store=memory_store,
                )
            )

        state_view = RuntimeState(context)
        media_paths = state_view.media_paths()
        messages.append(
            ModelMessage(
                role="user",
                content=task.description,
                images=tuple(media_paths),
            )
        )

        available_tools = self.tools.tool_schemas(base_lease)
        tool_names = [t["function"]["name"] for t in available_tools]
        if not tool_names:
            all_tools = {n: rt.group for n, rt in self.tools._tools.items()}
            logger.warning(
                "Executor NO TOOLS task=%s lease=%s registry_tools=%s",
                task.task_id,
                base_lease,
                all_tools,
            )
        logger.info(
            "Executor start task=%s max_steps=%d tools=%s lease_groups=%s include_tools=%s exclude_tools=%s",
            task.task_id,
            self.policy.max_steps,
            tool_names,
            list(base_lease.include_groups),
            list(base_lease.include_tools),
            list(base_lease.exclude_tools),
        )

        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        max_model_tokens = int(context.state.get("max_model_tokens", 0) or 0)
        loop_guard = LoopGuard(self.policy.loop_guard)
        blocked_tool_names: set[str] = set()
        no_progress_limit = max(1, int(self.policy.max_no_progress_turns))

        tool_context = ToolContext(session_id=context.session_id, state=context.state)
        heartbeat = state_view.get("heartbeat")
        tape = state_view.get("tape")
        tape_store = state_view.get("tape_store")
        notebook, notebook_node_id = self._resolve_notebook_binding(context)
        if (
            notebook is not None
            and notebook.get_node(notebook_node_id).status == "pending"
        ):
            self._transition_notebook_node(
                context,
                status="running",
                summary="Worker execution started",
                metadata={"source": "executor", "task_id": task.task_id},
            )

        return ExecutionSession(
            messages=messages,
            usage_totals=usage_totals,
            max_model_tokens=max_model_tokens,
            loop_guard=loop_guard,
            blocked_tool_names=blocked_tool_names,
            no_progress_turns=0,
            no_progress_limit=no_progress_limit,
            tool_call_count=0,
            tool_failure_count=0,
            loop_guard_block_count=0,
            exploration_finish_required=False,
            available_tools=available_tools,
            base_lease=base_lease,
            tool_context=tool_context,
            heartbeat=heartbeat,
            tape=tape,
            tape_store=tape_store,
            notebook=notebook,
            notebook_node_id=notebook_node_id,
            max_steps=max(1, self.policy.max_steps),
            task_id=task.task_id,
        )

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
        session = self._prepare(task, context)
        state_view = RuntimeState(context)

        for step in range(1, session.max_steps + 1):
            result = await self._execute_turn(session, task, context, step, state_view)
            if result is not None:
                return result

        logger.warning(
            "Executor exhausted steps task=%s max_steps=%d",
            task.task_id,
            self.policy.max_steps,
        )
        error = f"No terminal answer within {self.policy.max_steps} steps."
        self._transition_notebook_node(
            context,
            status="failed",
            summary="Executor exhausted step budget",
            detail=error,
            metadata={"task_id": task.task_id},
        )
        self._emit(
            context, "agent_end", task_id=task.task_id, status="failed", error=error
        )
        return self._build_task_result(
            session,
            status="failed",
            error=error,
            extra_metadata={
                "history": [
                    self._dump_message(message) for message in session.messages
                ],
                "tool_call_count": session.tool_call_count,
                "tool_failure_count": session.tool_failure_count,
                "loop_guard_block_count": session.loop_guard_block_count,
                "max_step_exhausted_count": 1,
                "notebook_node_id": session.notebook_node_id,
            },
        )

    async def _execute_turn(
        self,
        session: ExecutionSession,
        task: TaskContract,
        context: ExecutionContext,
        step: int,
        state_view: RuntimeState,
    ) -> TaskResult | None:
        """Execute one step of the agent loop.

        Returns a TaskResult to exit the loop early, or None to continue.
        """
        if session.heartbeat is not None:
            session.heartbeat.beat()

        # ── Turn start event ─────────────────────────────────
        self._emit(context, "turn_start", task_id=task.task_id, step=step)
        # Legacy event for backward compat
        context.emit("executor.step", task_id=task.task_id, step=step)
        turn_progress_before = self._progress_marker_count(context)
        self._record_notebook_event(
            context,
            kind="observation",
            summary=f"Turn {step} started",
            metadata={
                "stage": "turn_start",
                "step": step,
                "task_id": task.task_id,
            },
        )
        logger.info(
            "Executor step=%d/%d task=%s",
            step,
            self.policy.max_steps,
            task.task_id,
        )

        session.messages = session.loop_guard.compress_messages(
            session.messages,
            max_model_tokens=session.max_model_tokens,
        )

        # ── transformContext hook ─────────────────────────────
        transform_fn = context.transform_context
        if callable(transform_fn):
            pre_count = len(session.messages)
            session.messages = transform_fn(session.messages, context)
            if len(session.messages) != pre_count:
                self._emit(
                    context,
                    "context_transform",
                    task_id=task.task_id,
                    step=step,
                    messages_before=pre_count,
                    messages_after=len(session.messages),
                )

        runtime_hint_messages = self._consume_runtime_hint_messages(context.state)
        if session.blocked_tool_names:
            step_tools = [
                t
                for t in session.available_tools
                if t["function"]["name"] not in session.blocked_tool_names
            ]
        else:
            step_tools = session.available_tools
        if session.exploration_finish_required:
            step_tools = list(self._filter_non_exploration_tools(step_tools))

        # ── LLM request ──────────────────────────────────────
        self._emit(
            context,
            "llm_request_start",
            task_id=task.task_id,
            step=step,
            message_count=len(session.messages) + len(runtime_hint_messages),
            tool_count=len(step_tools),
        )
        llm_start = time.perf_counter()
        request_messages = tuple(session.messages)
        if runtime_hint_messages:
            if request_messages and request_messages[0].role == "system":
                request_messages = (
                    request_messages[:1] + runtime_hint_messages + request_messages[1:]
                )
            else:
                request_messages = runtime_hint_messages + request_messages
        response = await self.model.generate(
            ModelRequest(
                messages=request_messages,
                tools=step_tools,
                metadata={"task_id": task.task_id, "step": step},
            ),
            context,
        )
        llm_elapsed = time.perf_counter() - llm_start
        self._accumulate_usage(session.usage_totals, response)
        self._emit(
            context,
            "llm_request_end",
            task_id=task.task_id,
            step=step,
            elapsed_s=round(llm_elapsed, 3),
            has_tool_calls=bool(response.tool_calls),
            finish_reason=response.finish_reason or "stop",
        )

        budget_error = self._check_token_budget(
            session.usage_totals, session.max_model_tokens
        )
        if budget_error:
            self._transition_notebook_node(
                context,
                status="failed",
                summary="Model token budget exceeded",
                detail=budget_error,
                metadata={"task_id": task.task_id},
            )
            result = self._build_task_result(
                session,
                status="failed",
                error=budget_error,
                extra_metadata={
                    "max_model_tokens": session.max_model_tokens,
                    "notebook_node_id": session.notebook_node_id,
                },
            )
            self._emit(
                context,
                "agent_end",
                task_id=task.task_id,
                status="failed",
                error=budget_error,
            )
            return result

        if response.tool_calls:
            return await self._handle_tool_calls_branch(
                session, task, context, step, response, state_view, turn_progress_before
            )
        return await self._handle_text_response_branch(
            session, task, context, step, response
        )

    async def _handle_tool_calls_branch(
        self,
        session: ExecutionSession,
        task: TaskContract,
        context: ExecutionContext,
        step: int,
        response: ModelResponse,
        state_view: RuntimeState,
        turn_progress_before: int,
    ) -> TaskResult | None:
        """Handle the branch where the model returned tool calls.

        Returns a TaskResult to exit the loop, or None to continue.
        """
        session.tool_call_count += len(response.tool_calls)
        self._record_notebook_event(
            context,
            kind="summary",
            summary=f"Model selected {len(response.tool_calls)} tool(s)",
            detail=self._summarize_notebook_text(response.text),
            metadata={
                "stage": "model_tool_decision",
                "step": step,
                "task_id": task.task_id,
                "tool_names": [tc.name for tc in response.tool_calls],
            },
        )
        logger.info(
            "Executor tool_calls task=%s step=%d calls=%s",
            task.task_id,
            step,
            [tc.name for tc in response.tool_calls],
        )
        if session.tape is not None:
            tape_entries: list[Entry] = []
            for tool_call in response.tool_calls:
                entry = session.tape.append(
                    "tool_call",
                    {
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                        "call_id": tool_call.call_id,
                    },
                    {
                        "task_id": task.task_id,
                        "step": step,
                    },
                )
                tape_entries.append(entry)
            self._persist_tape_entries(session, tape_entries)
        session.messages.append(
            ModelMessage(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
            )
        )
        for tool_call in response.tool_calls:
            self._record_notebook_event(
                context,
                kind="tool_call",
                summary=tool_call.name,
                detail=_json.dumps(
                    tool_call.arguments, ensure_ascii=False, sort_keys=True
                )[:2000],
                metadata={
                    "step": step,
                    "call_id": tool_call.call_id,
                    "tool_name": tool_call.name,
                    "exploration": session.loop_guard.is_exploration_call(
                        tool_call.name,
                        tool_call.arguments,
                    ),
                },
            )
        # Phase 1: serial validation — lightweight checks
        validation = self._validate_tool_calls(response.tool_calls, session=session)
        valid_calls = validation.valid_calls
        error_results = validation.error_results
        forced_finalize_violations = validation.forced_finalize_violations

        # Build a map of tool_call index → result message for ordering
        tool_result_map: dict[str, ModelMessage] = {}

        # Append error results immediately
        for tc, err_output in error_results:
            self._record_notebook_event(
                context,
                kind="tool_result",
                summary=f"{tc.name} failed before execution",
                detail=err_output[:4000],
                metadata={
                    "step": step,
                    "call_id": tc.call_id,
                    "tool_name": tc.name,
                    "ok": False,
                },
            )
            tool_result_map[tc.call_id] = ModelMessage(
                role="tool",
                name=tc.name,
                content=err_output,
                tool_call_id=tc.call_id,
            )

        exploration_only_turn = bool(valid_calls) and all(
            session.loop_guard.is_exploration_call(tc.name, tc.arguments)
            for tc, _ in valid_calls
        )
        if (
            session.exploration_finish_required
            and forced_finalize_violations
            and len(forced_finalize_violations) == len(response.tool_calls)
        ):
            auto_summary = self._build_auto_converged_summary(
                task=task,
                messages=session.messages,
            )
            logger.warning(
                "Executor auto-converged task=%s turns=%d mode=forced_finalize_ignored",
                task.task_id,
                session.no_progress_turns,
            )
            self._record_notebook_event(
                context,
                kind="summary",
                summary="探索预算耗尽，自动收敛",
                detail=auto_summary,
                metadata={
                    "stage": "auto_converged_summary",
                    "step": step,
                    "task_id": task.task_id,
                    "auto_converged": True,
                },
                progress=True,
            )
            self._transition_notebook_node(
                context,
                status="completed",
                summary="探索预算耗尽，自动收敛",
                detail=auto_summary,
                metadata={
                    "task_id": task.task_id,
                    "auto_converged": True,
                },
            )
            result = self._build_task_result(
                session,
                status="succeeded",
                output=auto_summary,
                extra_metadata={
                    "no_progress_turns": session.no_progress_turns,
                    "blocked_tools": sorted(session.blocked_tool_names),
                    "tool_call_count": session.tool_call_count,
                    "tool_failure_count": session.tool_failure_count,
                    "loop_guard_block_count": session.loop_guard_block_count,
                    "max_step_exhausted_count": 0,
                    "notebook_node_id": session.notebook_node_id,
                    "auto_converged": True,
                    "completion_mode": "auto_summary_after_exploration_stall",
                },
            )
            self._emit(
                context,
                "agent_end",
                task_id=task.task_id,
                status="succeeded",
                output_len=len(auto_summary),
            )
            return result

        # Phase 2: parallel execution of validated tool calls
        if valid_calls:
            import asyncio as _aio

            def _collect_media(paths: list[str]) -> None:
                if not paths:
                    return
                bucket = state_view.collected_media_bucket()
                existing = set(bucket)
                added: list[str] = []
                for path in paths:
                    if path and path not in existing:
                        bucket.append(path)
                        existing.add(path)
                        added.append(path)
                        notebook, node_id = self._resolve_notebook_binding(context)
                        if notebook is not None:
                            notebook.add_artifact(
                                node_id=node_id,
                                path=path,
                                label="worker artifact",
                                metadata={
                                    "progress": True,
                                    "source": "tool_result",
                                },
                            )
                if added:
                    state_view.append_runtime_hint(
                        "Output artifacts detected:\n"
                        + "\n".join(f"- {path}" for path in added)
                        + "\nIf these satisfy the task, stop and return a concise final answer with the exact paths."
                        " Do not spend more turns on read-only verification."
                    )

            async def _invoke_one(
                tc: ModelToolCall,
                reg: Any,
            ) -> tuple[ModelToolCall, str, list[str], bool]:
                # ── before_tool_call hook ────────────────
                hook_before = context.before_tool_call
                if callable(hook_before):
                    hook_before(tc.name, tc.arguments, context)

                # ── tool_execution_start event ───────────
                self._emit(
                    context,
                    "tool_execution_start",
                    task_id=task.task_id,
                    step=step,
                    tool_name=tc.name,
                    call_id=tc.call_id,
                )

                logger.info(
                    "Executor invoke task=%s tool=%s args_keys=%s",
                    task.task_id,
                    tc.name,
                    list(tc.arguments.keys()),
                )
                started = time.perf_counter()
                try:
                    result = await reg.tool.invoke(tc.arguments, session.tool_context)
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    logger.exception(
                        "Executor tool crashed task=%s tool=%s elapsed=%.2fs",
                        task.task_id,
                        tc.name,
                        elapsed,
                    )
                    output = (
                        f"Tool error: {exc}"
                        "\n[Hint: Analyze the error and try a different approach instead of retrying the same way.]"
                    )
                    # ── tool_execution_end event (error) ─
                    self._emit(
                        context,
                        "tool_execution_end",
                        task_id=task.task_id,
                        step=step,
                        tool_name=tc.name,
                        call_id=tc.call_id,
                        ok=False,
                        elapsed_s=round(elapsed, 3),
                    )
                    # ── after_tool_call hook ─────────────
                    hook_after = context.after_tool_call
                    if callable(hook_after):
                        hook_after(tc.name, tc.arguments, None, context)
                    return (tc, output, [], True)

                elapsed = time.perf_counter() - started
                output = (
                    result.content
                    if result.ok
                    else (
                        f"Tool error: {result.error}"
                        "\n[Hint: Analyze the error and try a different approach instead of retrying the same way.]"
                    )
                )
                if result.ok:
                    logger.info(
                        "Executor tool done task=%s tool=%s ok=%s elapsed=%.2fs output_len=%d",
                        task.task_id,
                        tc.name,
                        result.ok,
                        elapsed,
                        len(output),
                    )
                else:
                    logger.warning(
                        "Executor tool failed task=%s tool=%s elapsed=%.2fs error=%s",
                        task.task_id,
                        tc.name,
                        elapsed,
                        (result.error or "")[:500],
                    )

                # ── tool_execution_end event (success) ───
                self._emit(
                    context,
                    "tool_execution_end",
                    task_id=task.task_id,
                    step=step,
                    tool_name=tc.name,
                    call_id=tc.call_id,
                    ok=result.ok,
                    elapsed_s=round(elapsed, 3),
                    output_len=len(output),
                )
                # ── after_tool_call hook ─────────────────
                hook_after = context.after_tool_call
                if callable(hook_after):
                    hook_after(tc.name, tc.arguments, result, context)

                return tc, output, list(result.artifacts or []), not result.ok

            done = await _aio.gather(*[_invoke_one(tc, reg) for tc, reg in valid_calls])
            result_entries: list[Entry] = []
            for tc, output, artifacts, failed in done:
                if failed:
                    session.tool_failure_count += 1
                _collect_media(artifacts)
                exploration_call = session.loop_guard.is_exploration_call(
                    tc.name,
                    tc.arguments,
                )
                progress = (not failed and not exploration_call) or bool(artifacts)
                self._record_notebook_event(
                    context,
                    kind="tool_result",
                    summary=(
                        f"{tc.name} succeeded" if not failed else f"{tc.name} failed"
                    ),
                    detail=output[:4000],
                    metadata={
                        "step": step,
                        "call_id": tc.call_id,
                        "tool_name": tc.name,
                        "ok": not failed,
                        "artifact_count": len(artifacts),
                        "exploration": exploration_call,
                    },
                    progress=progress,
                )
                if session.tape is not None:
                    entry = session.tape.append(
                        "tool_result",
                        {
                            "name": tc.name,
                            "ok": not output.startswith("Tool error:"),
                            "content_preview": output[:500],
                            "artifacts": artifacts,
                            "call_id": tc.call_id,
                        },
                        {
                            "task_id": task.task_id,
                            "step": step,
                        },
                    )
                    result_entries.append(entry)
                tool_result_map[tc.call_id] = ModelMessage(
                    role="tool",
                    name=tc.name,
                    content=self._truncate_tool_output_for_model(
                        output,
                        self.policy.max_tool_result_chars,
                    ),
                    tool_call_id=tc.call_id,
                )
            self._persist_tape_entries(session, result_entries)

        turn_made_progress = self._progress_marker_count(context) > turn_progress_before
        exploration_only_no_progress = exploration_only_turn and not turn_made_progress
        if turn_made_progress:
            session.no_progress_turns = 0
            session.exploration_finish_required = False
        else:
            session.no_progress_turns += 1
            if exploration_only_turn:
                warning_turn = max(1, session.no_progress_limit - 1)
                if (
                    session.no_progress_limit > 1
                    and session.no_progress_turns == warning_turn
                ):
                    state_view.append_runtime_hint(
                        "你已经连续 "
                        f"{session.no_progress_turns} 轮只使用读取/搜索/检查类工具。"
                        " 下一轮必须收敛：要么直接给出当前结论/缺口摘要，"
                        "要么切换到编辑、写入或执行类工具，不要继续无边界探索。"
                    )
            elif session.no_progress_turns >= session.no_progress_limit:
                last_issues = " | ".join(
                    err_output.splitlines()[0][:200]
                    for _, err_output in error_results[:3]
                    if err_output
                )
                error = f"No progress after {session.no_progress_turns} consecutive tool-only turns."
                if last_issues:
                    error += f" Last issues: {last_issues}"
                logger.warning(
                    "Executor no-progress fail task=%s turns=%d issues=%s",
                    task.task_id,
                    session.no_progress_turns,
                    last_issues,
                )
                self._transition_notebook_node(
                    context,
                    status="failed",
                    summary="Executor stalled without notebook progress",
                    detail=error,
                    metadata={"task_id": task.task_id},
                )
                result = self._build_task_result(
                    session,
                    status="failed",
                    error=error,
                    extra_metadata={
                        "no_progress_turns": session.no_progress_turns,
                        "blocked_tools": sorted(session.blocked_tool_names),
                        "tool_call_count": session.tool_call_count,
                        "tool_failure_count": session.tool_failure_count,
                        "loop_guard_block_count": session.loop_guard_block_count,
                        "max_step_exhausted_count": 0,
                        "notebook_node_id": session.notebook_node_id,
                    },
                )
                self._emit(
                    context,
                    "agent_end",
                    task_id=task.task_id,
                    status="failed",
                    error=error,
                )
                return result

        # Append tool results in original tool_calls order
        for tool_call in response.tool_calls:
            if tool_call.call_id in tool_result_map:
                session.messages.append(tool_result_map[tool_call.call_id])

        # Allow one last convergence turn after the exploration budget
        # is exhausted. That turn may finish directly or use action/edit
        # tools, but exploration tools are removed from the offered set
        # and rejected if the model still calls them.
        if (
            exploration_only_no_progress
            and session.no_progress_turns >= session.no_progress_limit
        ):
            if not session.exploration_finish_required:
                session.exploration_finish_required = True
                state_view.append_runtime_hint(
                    "探索预算已耗尽。下一轮必须直接给出当前结论/缺口摘要，"
                    "或切换到编辑、写入、执行类工具。不要再调用读取、搜索、检查类工具。"
                )

        # ── Turn end event ───────────────────────────────
        self._emit(
            context,
            "turn_end",
            task_id=task.task_id,
            step=step,
            tool_calls=[tc.name for tc in response.tool_calls],
        )

        if session.heartbeat is not None:
            session.heartbeat.beat()
        return None  # continue the loop

    async def _handle_text_response_branch(
        self,
        session: ExecutionSession,
        task: TaskContract,
        context: ExecutionContext,
        step: int,
        response: ModelResponse,
    ) -> TaskResult | None:
        """Handle the branch where the model returned text (no tool calls).

        Returns a TaskResult to exit the loop, or None to continue.
        """
        text = response.text
        if response.finish_reason == "length":
            text = await self._continue_from_truncation(
                initial_text=text,
                messages=session.messages,
                context=context,
                task_id=task.task_id,
                step=step,
                usage_totals=session.usage_totals,
                max_model_tokens=session.max_model_tokens,
            )
            if text is None:
                error = (
                    f"Model token budget exceeded "
                    f"({session.usage_totals['total_tokens']}/{session.max_model_tokens})."
                )
                self._transition_notebook_node(
                    context,
                    status="failed",
                    summary="Continuation exceeded token budget",
                    detail=error,
                    metadata={"task_id": task.task_id},
                )
                result = self._build_task_result(
                    session,
                    status="failed",
                    error=error,
                    extra_metadata={
                        "max_model_tokens": session.max_model_tokens,
                        "tool_call_count": session.tool_call_count,
                        "tool_failure_count": session.tool_failure_count,
                        "loop_guard_block_count": session.loop_guard_block_count,
                        "max_step_exhausted_count": 0,
                        "notebook_node_id": session.notebook_node_id,
                    },
                )
                self._emit(
                    context,
                    "agent_end",
                    task_id=task.task_id,
                    status="failed",
                    error=error,
                )
                return result

        text = text.strip()
        if text:
            session.exploration_finish_required = False
            final_summary = self._summarize_notebook_text(text)
            self._record_notebook_event(
                context,
                kind="summary",
                summary=final_summary or "Worker produced final output",
                detail=text,
                metadata={
                    "stage": "final_answer",
                    "step": step,
                    "task_id": task.task_id,
                },
                progress=True,
            )
            self._transition_notebook_node(
                context,
                status="completed",
                summary=final_summary or "Worker completed",
                detail=text,
                metadata={
                    "progress": True,
                    "source": "executor",
                    "task_id": task.task_id,
                },
            )
            logger.info(
                "Executor final answer task=%s step=%d output_len=%d",
                task.task_id,
                step,
                len(text),
            )
            # ── Turn end (final) + Agent end events ──────────
            self._emit(context, "turn_end", task_id=task.task_id, step=step, final=True)
            self._emit(
                context,
                "agent_end",
                task_id=task.task_id,
                status="succeeded",
                output_len=len(text),
                total_tool_calls=session.tool_call_count,
            )
            return self._build_task_result(
                session,
                status="succeeded",
                output=text,
                extra_metadata={
                    "notebook_node_id": session.notebook_node_id,
                    "notebook_summary": text,
                    "tool_call_count": session.tool_call_count,
                    "tool_failure_count": session.tool_failure_count,
                    "loop_guard_block_count": session.loop_guard_block_count,
                    "max_step_exhausted_count": 0,
                },
            )
        return None  # empty text: continue the loop

    async def _continue_from_truncation(
        self,
        *,
        initial_text: str,
        messages: list[ModelMessage],
        context: ExecutionContext,
        task_id: str,
        step: int,
        usage_totals: dict[str, int],
        max_model_tokens: int,
    ) -> str | None:
        text = initial_text or ""
        finish_reason = "length"
        for continuation_idx in range(1, max(1, self.policy.max_continuations) + 1):
            if finish_reason != "length":
                break
            messages.append(ModelMessage(role="assistant", content=text))
            messages.append(
                ModelMessage(role="user", content="Continue from where you left off.")
            )
            response = await self.model.generate(
                ModelRequest(
                    messages=tuple(messages),
                    metadata={
                        "task_id": task_id,
                        "step": step,
                        "continuation": continuation_idx,
                    },
                ),
                context,
            )
            self._accumulate_usage(usage_totals, response)
            if self._check_token_budget(usage_totals, max_model_tokens):
                return None
            text += response.text or ""
            finish_reason = response.finish_reason or "stop"
        return text

    def _resolve_skills(
        self, task: TaskContract, context: ExecutionContext
    ) -> list[SkillPack]:
        resolver = self.skill_resolver
        if resolver is None:
            return []
        result = resolver(task, context)
        if not result:
            return []
        if isinstance(result, SkillPack):
            return [result]
        return [item for item in result if isinstance(item, SkillPack)]

    @staticmethod
    def _tool_allowed(group: str, name: str, lease: ToolLease) -> bool:
        include_groups = lease.include_groups_set
        include_tools = lease.include_tools_set
        exclude_tools = lease.exclude_tools_set
        if name in exclude_tools:
            return False
        if include_tools or include_groups:
            in_tools = name in include_tools if include_tools else False
            in_groups = group in include_groups if include_groups else False
            if not (in_tools or in_groups):
                return False
        return True

    @staticmethod
    def _dump_message(message: ModelMessage) -> dict[str, str]:
        payload = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.call_id, "name": tc.name} for tc in message.tool_calls
            ]
        return payload

    @staticmethod
    def _usage_from_response(response: ModelResponse) -> tuple[int, int, int]:
        usage = response.metadata.get("usage", {}) if response.metadata else {}
        if not isinstance(usage, dict):
            return 0, 0, 0
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        return prompt, completion, total

    @classmethod
    def _accumulate_usage(
        cls,
        totals: dict[str, int],
        response: ModelResponse,
    ) -> None:
        prompt, completion, total = cls._usage_from_response(response)
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += total

    @staticmethod
    def _check_token_budget(totals: dict[str, int], max_model_tokens: int) -> str:
        if max_model_tokens <= 0:
            return ""
        if totals["total_tokens"] <= max_model_tokens:
            return ""
        return (
            f"Model token budget exceeded "
            f"({totals['total_tokens']}/{max_model_tokens})."
        )

    @staticmethod
    def _build_usage_metadata(
        totals: dict[str, int],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _build_task_result(
        self,
        session: ExecutionSession,
        *,
        status: str,
        output: str = "",
        error: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Thin wrapper: assemble TaskResult with accumulated usage metadata."""
        return TaskResult(
            task_id=session.task_id,
            status=status,
            output=output,
            error=error,
            metadata=self._build_usage_metadata(
                session.usage_totals,
                extra=extra_metadata or {},
            ),
        )

    @staticmethod
    def _persist_tape_entries(session: ExecutionSession, entries: list[Any]) -> None:
        """Persist a batch of tape entries to the tape store in one call."""
        if not entries:
            return
        if session.tape is None:
            return
        if session.tape_store is None:
            return
        save_entries = getattr(session.tape_store, "save_entries", None)
        save_entry = getattr(session.tape_store, "save_entry", None)
        chat_id = getattr(session.tape, "chat_id", "")
        if not chat_id:
            return
        if callable(save_entries):
            save_entries(chat_id, entries)
            return
        if callable(save_entry):
            for entry in entries:
                save_entry(chat_id, entry)

    @staticmethod
    def _cast_tool_arguments(
        schema: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Best-effort type coercion based on the tool schema."""
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            return args
        casted = dict(args)
        for name, value in casted.items():
            prop = properties.get(name)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            if not expected or value is None:
                continue
            if expected == "integer" and isinstance(value, str):
                try:
                    casted[name] = int(value)
                except ValueError:
                    pass
            elif expected == "number" and isinstance(value, str):
                try:
                    casted[name] = float(value)
                except ValueError:
                    pass
            elif expected == "boolean" and isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    casted[name] = True
                elif value.lower() in ("false", "0", "no"):
                    casted[name] = False
            elif expected == "array" and isinstance(value, str):
                try:
                    casted[name] = _json.loads(value)
                except (_json.JSONDecodeError, ValueError):
                    pass
        return casted

    @staticmethod
    def _validate_tool_arguments(
        schema: dict[str, Any],
        args: Any,
    ) -> str | None:
        if not isinstance(args, dict):
            return "arguments must be a JSON object"

        required = schema.get("required") or []
        if isinstance(required, list):
            missing = [name for name in required if name not in args]
            if missing:
                return f"missing required fields: {', '.join(missing)}"

        properties = schema.get("properties") or {}
        additional_allowed = schema.get("additionalProperties", True)
        if isinstance(properties, dict) and additional_allowed is False:
            extra = [key for key in args if key not in properties]
            if extra:
                return f"unexpected fields: {', '.join(extra)}"

        if isinstance(properties, dict):
            for name, value in args.items():
                prop = properties.get(name)
                if not isinstance(prop, dict):
                    continue
                expected = prop.get("type")
                if not expected:
                    continue
                if expected == "string" and not isinstance(value, str):
                    return f"field '{name}' must be string"
                if expected == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    return f"field '{name}' must be integer"
                if expected == "number" and (
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                ):
                    return f"field '{name}' must be number"
                if expected == "boolean" and not isinstance(value, bool):
                    return f"field '{name}' must be boolean"
                if expected == "array" and not isinstance(value, list):
                    return f"field '{name}' must be array"
                if expected == "object" and not isinstance(value, dict):
                    return f"field '{name}' must be object"
        return None


class EchoModelProvider:
    """Tiny debug model provider for local wiring tests."""

    async def generate(
        self, request: ModelRequest, context: ExecutionContext
    ) -> ModelResponse:
        last = (
            request.messages[-1]
            if request.messages
            else ModelMessage(role="assistant", content="")
        )
        if last.role == "tool":
            return ModelResponse(text=f"Observed: {last.content}")
        if request.tools:
            fn = request.tools[0]["function"]["name"]
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c1",
                        name=fn,
                        arguments={"input": last.content},
                    ),
                ),
            )
        return ModelResponse(text=last.content)
