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
from .skills import SkillPack, merge_leases, merge_prompts
from .tools import ToolContext, ToolRegistry
from .types import AgentEvent, ExecutionContext, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..context import Entry

logger = logging.getLogger(__name__)


# ── Kernel-internal helpers (language-agnostic fallbacks) ────────────────────


def _estimate_token_count(text: str) -> int:
    """Cheap token-count estimate: ~3 characters per token."""
    return max(1, len(str(text or "")) // 3)


def _extract_keywords(text: str) -> list[str]:
    """Very lightweight keyword extractor: unique words ≥ 3 chars."""
    words = str(text or "").lower().split()
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        w = w.strip(".,!?;:\"'()[]{}，。！？；：")
        if len(w) >= 3 and w not in seen:
            seen.add(w)
            result.append(w)
    return result[:20]


def _build_context_view_messages(
    memory_store: Any,
    chat_id: str,
    query: str,
) -> list[ModelMessage]:
    """Delegate to application-layer context_views if available, else return []."""
    try:
        from ..context_views import build_context_view_messages  # type: ignore[import]

        return build_context_view_messages(  # type: ignore[return-value]
            memory_store=memory_store,
            chat_id=chat_id,
            query=query,
        )
    except Exception:
        return []


@dataclass
class ExecutorPolicy:
    """Policy for one single-agent execution loop."""

    max_steps: int = 8
    max_continuations: int = 2
    max_no_progress_turns: int = 3
    max_tool_result_chars: int = 12000
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)


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
        notebook = context.state.get("plan_notebook")
        if not isinstance(notebook, PlanNotebook):
            return None, ""
        node_id = str(context.state.get("current_notebook_node_id", "") or "").strip()
        if not node_id or node_id not in notebook.nodes:
            node_id = str(notebook.root_node_id or "").strip()
        if not node_id or node_id not in notebook.nodes:
            return None, ""
        return notebook, node_id

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

    # ── Main execution loop ──────────────────────────────────────────

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
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

        media_paths = context.state.get("media_paths") or ()
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
        no_progress_turns = 0
        no_progress_limit = max(1, int(self.policy.max_no_progress_turns))
        tool_call_count = 0
        tool_failure_count = 0
        loop_guard_block_count = 0
        exploration_finish_required = False
        exploration_grace_turn_used = False

        tool_context = ToolContext(session_id=context.session_id, state=context.state)
        heartbeat = context.state.get("heartbeat")
        tape = context.state.get("tape")
        tape_store = context.state.get("tape_store")
        notebook, notebook_node_id = self._resolve_notebook_binding(context)
        if notebook is not None and notebook.get_node(notebook_node_id).status == "pending":
            self._transition_notebook_node(
                context,
                status="running",
                summary="Worker execution started",
                metadata={"source": "executor", "task_id": task.task_id},
            )

        def _persist_tape_entries(entries: list[Entry]) -> None:
            if not entries:
                return
            if tape is None:
                return
            if tape_store is None:
                return
            save_entries = getattr(tape_store, "save_entries", None)
            save_entry = getattr(tape_store, "save_entry", None)
            chat_id = getattr(tape, "chat_id", "")
            if not chat_id:
                return
            if callable(save_entries):
                save_entries(chat_id, entries)
                return
            if callable(save_entry):
                for entry in entries:
                    save_entry(chat_id, entry)

        for step in range(1, max(1, self.policy.max_steps) + 1):
            if heartbeat is not None:
                heartbeat.beat()

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

            messages = loop_guard.compress_messages(messages)

            # ── transformContext hook ─────────────────────────────
            transform_fn = context.transform_context
            if callable(transform_fn):
                pre_count = len(messages)
                messages = transform_fn(messages, context)
                if len(messages) != pre_count:
                    self._emit(
                        context,
                        "context_transform",
                        task_id=task.task_id,
                        step=step,
                        messages_before=pre_count,
                        messages_after=len(messages),
                    )

            runtime_hint_messages = self._consume_runtime_hint_messages(context.state)
            if blocked_tool_names:
                step_tools = [
                    t
                    for t in available_tools
                    if t["function"]["name"] not in blocked_tool_names
                ]
            else:
                step_tools = available_tools

            # ── LLM request ──────────────────────────────────────
            self._emit(
                context,
                "llm_request_start",
                task_id=task.task_id,
                step=step,
                message_count=len(messages) + len(runtime_hint_messages),
                tool_count=len(step_tools),
            )
            llm_start = time.perf_counter()
            request_messages = tuple(messages)
            if runtime_hint_messages:
                if request_messages and request_messages[0].role == "system":
                    request_messages = (
                        request_messages[:1]
                        + runtime_hint_messages
                        + request_messages[1:]
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
            self._accumulate_usage(usage_totals, response)
            self._emit(
                context,
                "llm_request_end",
                task_id=task.task_id,
                step=step,
                elapsed_s=round(llm_elapsed, 3),
                has_tool_calls=bool(response.tool_calls),
                finish_reason=response.finish_reason or "stop",
            )

            budget_error = self._check_token_budget(usage_totals, max_model_tokens)
            if budget_error:
                self._transition_notebook_node(
                    context,
                    status="failed",
                    summary="Model token budget exceeded",
                    detail=budget_error,
                    metadata={"task_id": task.task_id},
                )
                result = TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=budget_error,
                    metadata=self._build_usage_metadata(
                        usage_totals,
                        extra={
                            "max_model_tokens": max_model_tokens,
                            "notebook_node_id": notebook_node_id,
                        },
                    ),
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
                tool_call_count += len(response.tool_calls)
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
                if (
                    exploration_finish_required
                    and exploration_grace_turn_used
                    and all(
                    loop_guard.is_exploration_call(tc.name, tc.arguments)
                    for tc in response.tool_calls
                    )
                ):
                    error = (
                        f"No progress after {no_progress_turns} consecutive tool-only turns."
                        " Exploration budget was exhausted and the model still chose "
                        "read/search/check tools instead of finishing or taking action."
                    )
                    logger.warning(
                        "Executor no-progress fail task=%s turns=%d mode=forced_finalize_ignored",
                        task.task_id,
                        no_progress_turns,
                    )
                    self._transition_notebook_node(
                        context,
                        status="failed",
                        summary="Exploration budget exhausted",
                        detail=error,
                        metadata={"task_id": task.task_id},
                    )
                    result = TaskResult(
                        task_id=task.task_id,
                        status="failed",
                        error=error,
                        metadata=self._build_usage_metadata(
                            usage_totals,
                            extra={
                                "no_progress_turns": no_progress_turns,
                                "blocked_tools": sorted(blocked_tool_names),
                                "tool_call_count": tool_call_count,
                                "tool_failure_count": tool_failure_count,
                                "loop_guard_block_count": loop_guard_block_count,
                                "max_step_exhausted_count": 0,
                                "notebook_node_id": notebook_node_id,
                            },
                        ),
                    )
                    self._emit(
                        context,
                        "agent_end",
                        task_id=task.task_id,
                        status="failed",
                        error=error,
                    )
                    return result
                if tape is not None:
                    tape_entries: list[Entry] = []
                    for tool_call in response.tool_calls:
                        entry = tape.append(
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
                    _persist_tape_entries(tape_entries)
                messages.append(
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
                        detail=_json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)[:2000],
                        metadata={
                            "step": step,
                            "call_id": tool_call.call_id,
                            "tool_name": tool_call.name,
                            "exploration": loop_guard.is_exploration_call(
                                tool_call.name,
                                tool_call.arguments,
                            ),
                        },
                    )
                # Phase 1: serial validation — lightweight checks
                error_results: list[tuple[ModelToolCall, str]] = []
                valid_calls: list[tuple[ModelToolCall, Any]] = []

                for tool_call in response.tool_calls:
                    verdict = loop_guard.check_call(tool_call.name, tool_call.arguments)
                    if verdict.blocked:
                        loop_guard_block_count += 1
                        logger.warning(
                            "Executor loop guard blocked task=%s tool=%s reason=%s",
                            task.task_id,
                            tool_call.name,
                            verdict.reason,
                        )
                        if verdict.disable_tool:
                            blocked_tool_names.add(tool_call.name)
                        remaining = [
                            t["function"]["name"]
                            for t in available_tools
                            if t["function"]["name"] not in blocked_tool_names
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
                        error_results.append((tool_call, hint))
                        continue

                    registered = self.tools.get(tool_call.name)
                    if not registered or not self._tool_allowed(
                        registered.group,
                        tool_call.name,
                        base_lease,
                    ):
                        logger.warning(
                            "Executor tool unavailable task=%s tool=%s",
                            task.task_id,
                            tool_call.name,
                        )
                        error_results.append(
                            (
                                tool_call,
                                (
                                    f"Tool unavailable: {tool_call.name}"
                                    "\n[Hint: This tool is not available. Use a different tool or approach.]"
                                ),
                            )
                        )
                        tool_failure_count += 1
                        continue

                    if isinstance(
                        tool_call.arguments, dict
                    ) and tool_call.arguments.get("__tool_argument_parse_error__"):
                        logger.warning(
                            "Executor invalid arguments JSON task=%s tool=%s",
                            task.task_id,
                            tool_call.name,
                        )
                        error_results.append(
                            (
                                tool_call,
                                (
                                    f"Tool argument JSON parse error for {tool_call.name}: "
                                    f"{tool_call.arguments.get('__raw_arguments__', '')}"
                                    "\n[Hint: Fix the JSON syntax in your tool arguments.]"
                                ),
                            )
                        )
                        tool_failure_count += 1
                        continue

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
                            task.task_id,
                            tool_call.name,
                            validation_error,
                        )
                        error_results.append(
                            (
                                tool_call,
                                (
                                    f"Tool argument validation failed for {tool_call.name}: {validation_error}"
                                    "\n[Hint: Check parameter types and values against the tool schema.]"
                                ),
                            )
                        )
                        tool_failure_count += 1
                        continue

                    valid_calls.append((tool_call, registered))

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
                    loop_guard.is_exploration_call(tc.name, tc.arguments)
                    for tc, _ in valid_calls
                )

                # Phase 2: parallel execution of validated tool calls
                if valid_calls:
                    import asyncio as _aio

                    def _collect_media(paths: list[str]) -> None:
                        if not paths:
                            return
                        bucket = context.state.setdefault("media_paths_collected", [])
                        existing = set(bucket)
                        added: list[str] = []
                        for path in paths:
                            if path and path not in existing:
                                bucket.append(path)
                                existing.add(path)
                                added.append(path)
                                notebook, node_id = self._resolve_notebook_binding(
                                    context
                                )
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
                            context.state.setdefault("pending_runtime_hints", []).append(
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
                            result = await reg.tool.invoke(tc.arguments, tool_context)
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

                    done = await _aio.gather(
                        *[_invoke_one(tc, reg) for tc, reg in valid_calls]
                    )
                    result_entries: list[Entry] = []
                    for tc, output, artifacts, failed in done:
                        if failed:
                            tool_failure_count += 1
                        _collect_media(artifacts)
                        exploration_call = loop_guard.is_exploration_call(
                            tc.name,
                            tc.arguments,
                        )
                        progress = (not failed and not exploration_call) or bool(
                            artifacts
                        )
                        self._record_notebook_event(
                            context,
                            kind="tool_result",
                            summary=(
                                f"{tc.name} succeeded"
                                if not failed
                                else f"{tc.name} failed"
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
                        if tape is not None:
                            entry = tape.append(
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
                    _persist_tape_entries(result_entries)

                turn_made_progress = (
                    self._progress_marker_count(context) > turn_progress_before
                )
                exploration_only_no_progress = (
                    exploration_only_turn and not turn_made_progress
                )
                if turn_made_progress:
                    no_progress_turns = 0
                    exploration_finish_required = False
                    exploration_grace_turn_used = False
                else:
                    no_progress_turns += 1
                    if exploration_only_turn:
                        warning_turn = max(1, no_progress_limit - 1)
                        if no_progress_limit > 1 and no_progress_turns == warning_turn:
                            context.state.setdefault("pending_runtime_hints", []).append(
                                "你已经连续 "
                                f"{no_progress_turns} 轮只使用读取/搜索/检查类工具。"
                                " 下一轮必须收敛：要么直接给出当前结论/缺口摘要，"
                                "要么切换到编辑、写入或执行类工具，不要继续无边界探索。"
                            )
                    elif no_progress_turns >= no_progress_limit:
                        last_issues = " | ".join(
                            err_output.splitlines()[0][:200]
                            for _, err_output in error_results[:3]
                            if err_output
                        )
                        error = f"No progress after {no_progress_turns} consecutive tool-only turns."
                        if last_issues:
                            error += f" Last issues: {last_issues}"
                        logger.warning(
                            "Executor no-progress fail task=%s turns=%d issues=%s",
                            task.task_id,
                            no_progress_turns,
                            last_issues,
                        )
                        self._transition_notebook_node(
                            context,
                            status="failed",
                            summary="Executor stalled without notebook progress",
                            detail=error,
                            metadata={"task_id": task.task_id},
                        )
                        result = TaskResult(
                            task_id=task.task_id,
                            status="failed",
                            error=error,
                            metadata=self._build_usage_metadata(
                                usage_totals,
                                extra={
                                    "no_progress_turns": no_progress_turns,
                                    "blocked_tools": sorted(blocked_tool_names),
                                    "tool_call_count": tool_call_count,
                                    "tool_failure_count": tool_failure_count,
                                    "loop_guard_block_count": loop_guard_block_count,
                                    "max_step_exhausted_count": 0,
                                    "notebook_node_id": notebook_node_id,
                                },
                            ),
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
                        messages.append(tool_result_map[tool_call.call_id])

                # Allow one last convergence turn after the exploration budget
                # is exhausted. If the next model response still only explores,
                # the executor fails before running those tools.
                if exploration_only_no_progress and no_progress_turns >= no_progress_limit:
                    if not exploration_finish_required:
                        exploration_finish_required = True
                        exploration_grace_turn_used = False
                        context.state.setdefault("pending_runtime_hints", []).append(
                            "探索预算已耗尽。下一轮必须直接给出当前结论/缺口摘要，"
                            "或切换到编辑、写入、执行类工具。不要再调用读取、搜索、检查类工具。"
                        )
                    elif not exploration_grace_turn_used:
                        exploration_grace_turn_used = True
                        context.state.setdefault("pending_runtime_hints", []).append(
                            "你已经在探索预算耗尽后继续探索了一轮。"
                            "下一轮必须直接给出结论/缺口摘要，或改用编辑、写入、执行类工具；"
                            "如果仍继续读取、搜索、检查，将直接失败。"
                        )
                    else:
                        error = (
                            f"No progress after {no_progress_turns} consecutive tool-only turns."
                            " Only exploratory tools were used; switch to edit/write/action tools or finish."
                        )
                        logger.warning(
                            "Executor no-progress fail task=%s turns=%d mode=exploration_only",
                            task.task_id,
                            no_progress_turns,
                        )
                        self._transition_notebook_node(
                            context,
                            status="failed",
                            summary="Executor stalled in exploration-only mode",
                            detail=error,
                            metadata={"task_id": task.task_id},
                        )
                        result = TaskResult(
                            task_id=task.task_id,
                            status="failed",
                            error=error,
                            metadata=self._build_usage_metadata(
                                usage_totals,
                                extra={
                                    "no_progress_turns": no_progress_turns,
                                    "blocked_tools": sorted(blocked_tool_names),
                                    "tool_call_count": tool_call_count,
                                    "tool_failure_count": tool_failure_count,
                                    "loop_guard_block_count": loop_guard_block_count,
                                    "max_step_exhausted_count": 0,
                                    "notebook_node_id": notebook_node_id,
                                },
                            ),
                        )
                        self._emit(
                            context,
                            "agent_end",
                            task_id=task.task_id,
                            status="failed",
                            error=error,
                        )
                        return result

                # ── Turn end event ───────────────────────────────
                self._emit(
                    context,
                    "turn_end",
                    task_id=task.task_id,
                    step=step,
                    tool_calls=[tc.name for tc in response.tool_calls],
                )

                if heartbeat is not None:
                    heartbeat.beat()
                continue

            text = response.text
            if response.finish_reason == "length":
                text = await self._continue_from_truncation(
                    initial_text=text,
                    messages=messages,
                    context=context,
                    task_id=task.task_id,
                    step=step,
                    usage_totals=usage_totals,
                    max_model_tokens=max_model_tokens,
                )
                if text is None:
                    error = (
                        f"Model token budget exceeded "
                        f"({usage_totals['total_tokens']}/{max_model_tokens})."
                    )
                    self._transition_notebook_node(
                        context,
                        status="failed",
                        summary="Continuation exceeded token budget",
                        detail=error,
                        metadata={"task_id": task.task_id},
                    )
                    result = TaskResult(
                        task_id=task.task_id,
                        status="failed",
                        error=error,
                        metadata=self._build_usage_metadata(
                            usage_totals,
                            extra={
                                "max_model_tokens": max_model_tokens,
                                "tool_call_count": tool_call_count,
                                "tool_failure_count": tool_failure_count,
                                "loop_guard_block_count": loop_guard_block_count,
                                "max_step_exhausted_count": 0,
                                "notebook_node_id": notebook_node_id,
                            },
                        ),
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
                exploration_finish_required = False
                exploration_grace_turn_used = False
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
                self._emit(
                    context, "turn_end", task_id=task.task_id, step=step, final=True
                )
                self._emit(
                    context,
                    "agent_end",
                    task_id=task.task_id,
                    status="succeeded",
                    output_len=len(text),
                    total_tool_calls=tool_call_count,
                )
                return TaskResult(
                    task_id=task.task_id,
                    status="succeeded",
                    output=text,
                    metadata=self._build_usage_metadata(
                        usage_totals,
                        extra={
                            "notebook_node_id": notebook_node_id,
                            "notebook_summary": text,
                            "tool_call_count": tool_call_count,
                            "tool_failure_count": tool_failure_count,
                            "loop_guard_block_count": loop_guard_block_count,
                            "max_step_exhausted_count": 0,
                        },
                    ),
                )

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
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            error=error,
            metadata=self._build_usage_metadata(
                usage_totals,
                extra={
                    "history": [self._dump_message(message) for message in messages],
                    "tool_call_count": tool_call_count,
                    "tool_failure_count": tool_failure_count,
                    "loop_guard_block_count": loop_guard_block_count,
                    "max_step_exhausted_count": 1,
                    "notebook_node_id": notebook_node_id,
                },
            ),
        )

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


def _history_entry_text(entry: object) -> str:
    kind = getattr(entry, "kind", "")
    payload = getattr(entry, "payload", {}) or {}
    if kind == "message":
        role = payload.get("role", "?")
        content = payload.get("content", "")
        return f"{role}: {content}"
    if kind == "tool_result":
        name = str(payload.get("name", "") or "?")
        status = "ok" if payload.get("ok") else "failed"
        preview = str(payload.get("content_preview", "") or "").strip()
        artifacts = payload.get("artifacts") or []
        suffix = (
            f"\nartifacts: {', '.join(str(item) for item in artifacts)}"
            if artifacts
            else ""
        )
        return f"[tool_result][{status}] {name}: {preview}{suffix}".rstrip()
    if kind == "tool_call":
        name = str(payload.get("name", "") or "?")
        arguments = payload.get("arguments", {})
        return f"[tool_call] {name}: {_json.dumps(arguments, ensure_ascii=False)}"
    if kind == "event":
        event_name = str(payload.get("event", "") or "?")
        event_payload = payload.get("payload") or {}
        description = str(event_payload.get("description", "") or "").strip()
        error = str(event_payload.get("error", "") or "").strip()
        output = str(event_payload.get("output", "") or "").strip()
        details_parts = [part for part in (description, output, error) if part]
        details = (
            " | ".join(details_parts)
            if details_parts
            else _json.dumps(event_payload, ensure_ascii=False)
        )
        return f"[event] {event_name}: {details}"
    if kind == "anchor":
        state = payload.get("state") or {}
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        return f"[anchor_summary] {summary}".strip()
    return ""


def _build_history_messages(
    tape: object,
    token_budget: int,
    query: str = "",
    tape_store: object | None = None,
    memory_store: object | None = None,
) -> list[ModelMessage]:
    """Build history context messages from a Tape.

    Three sections (all sharing token_budget):
    1. Anchor summary → system message
    2. BM25 cross-anchor recall → [relevant_history] system message
    3. Recent entries since anchor → user/assistant messages
    """
    messages: list[ModelMessage] = []

    last_anchor = getattr(tape, "last_anchor", None)
    if last_anchor is None:
        return messages
    anchor = last_anchor()
    budget_remaining = max(0, int(token_budget))

    chat_id = getattr(tape, "chat_id", "")
    if memory_store is not None and chat_id:
        load_assistant_profile = getattr(memory_store, "load_assistant_profile", None)
        if callable(load_assistant_profile):
            assistant_profile = str(load_assistant_profile() or "").strip()
            if assistant_profile:
                profile_text = "[Assistant Profile]\n" + assistant_profile
                profile_cost = max(1, _estimate_token_count(profile_text))
                if budget_remaining >= profile_cost:
                    messages.append(ModelMessage(role="system", content=profile_text))
                    budget_remaining -= profile_cost
        memory_messages = _build_context_view_messages(
            memory_store=memory_store,
            chat_id=chat_id,
            query=query,
        )
        for message in memory_messages:
            cost = max(1, _estimate_token_count(message.content))
            if budget_remaining < cost:
                continue
            messages.append(message)
            budget_remaining -= cost

    # 1. Anchor summary → system message (with structured fields if available)
    if anchor is not None:
        state = anchor.payload.get("state", {})
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        if summary:
            parts = [f"[conversation_context]\n{summary}"]
            entities = state.get("entities")
            if entities and isinstance(entities, list):
                parts.append(f"key_entities: {', '.join(entities)}")
            intent = state.get("user_intent")
            if intent:
                parts.append(f"user_intent: {intent}")
            pending = state.get("pending")
            if pending:
                parts.append(f"pending: {pending}")
            next_steps = state.get("next_steps")
            if next_steps and isinstance(next_steps, list):
                parts.append(
                    f"next_steps: {', '.join(str(item) for item in next_steps)}"
                )
            open_questions = state.get("open_questions")
            if open_questions and isinstance(open_questions, list):
                parts.append(
                    f"open_questions: {', '.join(str(item) for item in open_questions)}"
                )
            decisions = state.get("decisions")
            if decisions and isinstance(decisions, list):
                parts.append(f"decisions: {', '.join(str(item) for item in decisions)}")
            artifacts = state.get("artifacts")
            if artifacts and isinstance(artifacts, list):
                parts.append(f"artifacts: {', '.join(str(item) for item in artifacts)}")
            anchor_text = "\n".join(parts)
            anchor_cost = len(anchor_text) // 3
            if budget_remaining >= anchor_cost:
                messages.append(ModelMessage(role="system", content=anchor_text))
                budget_remaining -= anchor_cost

    # Collect recent entries (for both section 2 exclusion and section 3)
    entries_since = getattr(tape, "entries_since_anchor", None)
    if entries_since is None:
        return messages

    recent = entries_since()
    msg_entries = [e for e in recent if e.kind == "message"]
    recent_state_entries = [e for e in recent if e.kind in {"tool_result", "event"}]
    # Exclude the last user message (it's the current turn, added by executor)
    if msg_entries and msg_entries[-1].payload.get("role") == "user":
        msg_entries = msg_entries[:-1]

    # 2. BM25 cross-anchor recall — search for relevant entries before the anchor
    search_fn = getattr(tape_store, "search_relevant", None) if tape_store else None
    chat_id = getattr(tape, "chat_id", None)
    if search_fn and chat_id and query:
        recent_ids = {e.entry_id for e in recent}
        recall_budget = budget_remaining // 4  # Reserve up to 25% for recall
        try:
            recalled = search_fn(chat_id, query, limit=5, exclude_ids=recent_ids)
        except Exception:
            recalled = []

        if recalled:
            recall_lines: list[str] = []
            recall_tokens = 0
            for entry in recalled:
                est = max(1, int(entry.token_estimate))
                if recall_tokens + est > recall_budget:
                    break
                line = _history_entry_text(entry)
                if not line:
                    continue
                recall_lines.append(line)
                recall_tokens += est
            if recall_lines:
                messages.append(
                    ModelMessage(
                        role="system",
                        content="[relevant_history]\n" + "\n".join(recall_lines),
                    )
                )
                budget_remaining -= recall_tokens

    # 2.5. Recent non-message execution state (tool results / failed events)
    if recent_state_entries and budget_remaining > 0:
        state_lines: list[str] = []
        state_tokens = 0
        for entry in recent_state_entries[-5:]:
            if entry.kind == "event" and entry.payload.get("event") not in {
                "failed",
                "dead_lettered",
                "stalled",
            }:
                continue
            line = _history_entry_text(entry)
            if not line:
                continue
            est = max(1, int(entry.token_estimate))
            if state_tokens + est > max(1, budget_remaining // 3):
                continue
            state_lines.append(line)
            state_tokens += est
        if state_lines:
            messages.append(
                ModelMessage(
                    role="system",
                    content="[近期执行状态]\n" + "\n".join(state_lines),
                )
            )
            budget_remaining -= state_tokens

    # 3. Recent entries → hybrid recency+relevance scoring
    if msg_entries and query:
        kws = _extract_keywords(query)
    else:
        kws = []

    n = len(msg_entries)
    scored_entries: list[tuple[float, int, object]] = []
    for idx, entry in enumerate(msg_entries):
        # Recency: linear 0→1, most recent = 1.0
        recency = (idx + 1) / n if n else 0.0
        # Relevance: fraction of keywords found in content
        if kws:
            content = entry.payload.get("content", "")
            hits = sum(1 for kw in kws if kw in content)
            relevance = hits / len(kws)
        else:
            relevance = 0.0
        # Weighted blend: recency dominates (0.6) to preserve conversation flow
        score = 0.6 * recency + 0.4 * relevance
        scored_entries.append((score, idx, entry))

    # Sort by score descending, greedily pick within budget
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    picked_indices: set[int] = set()
    for score, idx, entry in scored_entries:
        if budget_remaining <= 0:
            break
        est = max(1, int(entry.token_estimate))
        if est > budget_remaining:
            continue
        budget_remaining -= est
        picked_indices.add(idx)

    # Emit in original chronological order
    for idx, entry in enumerate(msg_entries):
        if idx in picked_indices:
            messages.append(
                ModelMessage(
                    role=entry.payload["role"],
                    content=entry.payload["content"],
                )
            )

    return messages
