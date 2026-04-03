"""Port implementations: build_history_summary and ResourceBridgeExecutor."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from .executor import _build_history_messages
from .model import ModelMessage
from .types import (
    ExecutionContext,
    TaskContract,
    TaskResult,
    ToolLease,
)

if TYPE_CHECKING:
    from ..context import Tape
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager

logger = logging.getLogger(__name__)


def _summarize_context_view(
    memory_store: Any,
    chat_id: str,
    query: str,
) -> str:
    """Delegate to application-layer summarize_context_view if available."""
    try:
        from ..context_views import summarize_context_view  # type: ignore[import]

        return str(
            summarize_context_view(  # type: ignore[return-value]
                memory_store=memory_store,
                chat_id=chat_id,
                query=query,
            )
            or ""
        )
    except Exception:
        return ""


# ── Shared utilities ─────────────────────────────────────────────────────


def build_history_summary(
    tape: "Tape | None",
    memory_store: Any | None = None,
    query: str = "",
) -> str:
    """Build context summary from tape for orchestration prompts.

    Includes both the anchor summary (compacted history) AND recent
    conversation entries since the last anchor so the orchestrator can make
    context-aware decisions for follow-up messages.
    """
    if tape is None:
        return ""

    parts: list[str] = []
    chat_id = getattr(tape, "chat_id", "")
    if memory_store is not None and chat_id:
        load_assistant_profile = getattr(memory_store, "load_assistant_profile", None)
        if callable(load_assistant_profile):
            assistant_profile = str(load_assistant_profile() or "").strip()
            if assistant_profile:
                parts.append("[Assistant Profile]\n" + assistant_profile)
        memory_summary = _summarize_context_view(
            memory_store=memory_store,
            chat_id=chat_id,
            query=query,
        )
        if memory_summary:
            parts.append(memory_summary)

    # 1. Anchor summary (compacted older history)
    anchor = tape.last_anchor()
    if anchor is not None:
        state = anchor.payload.get("state") or {}
        summary = state.get("summary", "")
        if summary:
            parts.append(f"conversation_summary: {summary}")
        intent = state.get("user_intent", "")
        if intent:
            parts.append(f"user_intent: {intent}")
        pending = state.get("pending", "")
        if pending:
            parts.append(f"pending: {pending}")

    # 2. Recent conversation entries (not yet compacted)
    recent = tape.entries_since_anchor()
    if recent:
        recent_lines: list[str] = []
        for e in recent:
            if e.kind != "message":
                continue
            role = e.payload.get("role", "?")
            content = e.payload.get("content", "")
            if not content:
                continue
            # Truncate long messages to keep prompt manageable
            if len(content) > 300:
                content = content[:300] + "..."
            recent_lines.append(f"{role}: {content}")
        if recent_lines:
            # Keep at most the last 10 messages to bound prompt size
            trimmed = recent_lines[-10:]
            parts.append("recent_conversation:\n" + "\n".join(trimmed))

    return "\n".join(parts)


def _supports_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        params = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name:
            return True
    return False


# ── ResourceBridgeExecutor ───────────────────────────────────────────────


class ResourceBridgeExecutor:
    """Executes tasks by delegating to ResourceManager.run_subagent_task."""

    def __init__(
        self,
        resource_manager: "ResourceManager",
        gateway: "OpenAICompatibleGateway",
    ) -> None:
        self._rm = resource_manager
        self._gateway = gateway

    def _build_capability_summary(self) -> str:
        """Build a concise capability list from active resources."""
        briefs = self._rm.get_resource_briefs()
        lines: list[str] = []
        for b in briefs:
            if b.get("active") and b.get("tool_count", 0) > 0:
                lines.append(f"- {b.get('name', '?')}: {b.get('purpose', '')}")
        if not lines:
            return ""
        return (
            "\n\nYou have access to the following capabilities:\n"
            + "\n".join(lines)
            + "\nYou do not need to use these tools for this request; answer directly."
        )

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
        # Fast path: direct answer (no tools)
        if task.metadata.get("direct_answer"):
            goal = task.description
            heartbeat = context.state.get("heartbeat")
            media_paths = context.state.get("media_paths") or ()
            stream_callback = context.state.get("stream_callback")
            tape = context.state.get("tape")
            tape_store = context.state.get("tape_store")
            memory_store = context.state.get("memory_store")
            history_budget = context.state.get("context_history_tokens", 2000)

            base_prompt = (
                "You are a helpful assistant. Answer concisely and accurately. "
                "Do not fabricate tool results. "
                "If external information is needed, state so explicitly."
            )
            capability_summary = self._build_capability_summary()

            messages: list[ModelMessage] = []
            messages.append(
                ModelMessage(
                    role="system",
                    content=base_prompt + capability_summary,
                )
            )

            # Reuse executor.py's 3-section context builder
            if tape is not None:
                messages.extend(
                    _build_history_messages(
                        tape,
                        history_budget,
                        query=goal,
                        tape_store=tape_store,
                        memory_store=memory_store,
                    )
                )

            messages.append(
                ModelMessage(
                    role="user",
                    content=goal,
                    images=tuple(media_paths),
                )
            )

            answer = await self._gateway.complete_messages(
                messages,
                heartbeat=heartbeat,
                on_stream_text=stream_callback,
            )
            return TaskResult(
                task_id=task.task_id,
                status="succeeded",
                output=answer,
            )

        resource_id = task.metadata.get("resource_id", "")
        lease_to_dict = getattr(self._rm, "_lease_to_dict", None)
        if callable(lease_to_dict):
            lease_dict = lease_to_dict(task.lease)
        else:
            lease_dict = {
                "include_groups": list(task.lease.include_groups),
                "include_tools": list(task.lease.include_tools),
                "exclude_tools": list(task.lease.exclude_tools),
            }
        skill_ids = tuple(task.metadata.get("skill_ids", ()) or ())
        if not any(lease_dict.values()) and resource_id:
            scope = self._rm.resolve_resource_scope(resource_id, require_tools=True)
            if scope is None:
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=f"Unknown or unavailable resource: {resource_id}",
                )
            lease_dict, skill_ids = scope

        # Enrich task description with upstream results
        enriched = self._enrich_with_upstream(task, context)

        tape = context.state.get("tape")
        tape_store = context.state.get("tape_store")
        memory_store = context.state.get("memory_store")
        heartbeat = context.state.get("heartbeat")
        media_paths = context.state.get("media_paths")

        run_subagent_task = self._rm.run_subagent_task
        run_subagent_task_result = getattr(self._rm, "run_subagent_task_result", None)
        kwargs: dict[str, Any] = {
            "task_description": enriched,
            "lease": lease_dict,
            "agent_name": f"DAG-{task.task_id}",
            "tape": tape,
            "tape_store": tape_store,
            "heartbeat": heartbeat,
            "media_paths": media_paths,
            "skill_ids": list(skill_ids),
        }
        if memory_store is not None and _supports_kwarg(
            run_subagent_task, "memory_store"
        ):
            kwargs["memory_store"] = memory_store

        if callable(run_subagent_task_result):
            detailed_kwargs = dict(kwargs)
            if memory_store is not None and not _supports_kwarg(
                run_subagent_task_result, "memory_store"
            ):
                detailed_kwargs.pop("memory_store", None)
            try:
                detailed_result = await run_subagent_task_result(**detailed_kwargs)
            except Exception as exc:
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=str(exc),
                )

            artifacts = tuple(detailed_result.artifacts or ())
            if detailed_result.status == "succeeded":
                context.state.setdefault("upstream_results", {})[task.task_id] = (
                    detailed_result.output
                )
            context.state.setdefault("media_paths_collected", []).extend(artifacts)
            return TaskResult(
                task_id=task.task_id,
                status=detailed_result.status,
                output=detailed_result.output,
                error=detailed_result.error,
                artifacts=artifacts,
                attempts=detailed_result.attempts,
                metadata=dict(detailed_result.metadata or {}),
            )

        try:
            output, media = await run_subagent_task(**kwargs)
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=str(exc),
            )

        # Store result for downstream tasks
        context.state.setdefault("upstream_results", {})[task.task_id] = output
        context.state.setdefault("media_paths_collected", []).extend(media or [])

        return TaskResult(
            task_id=task.task_id,
            status="succeeded",
            output=output,
            artifacts=tuple(media or ()),
        )

    @staticmethod
    def _enrich_with_upstream(task: TaskContract, context: ExecutionContext) -> str:
        """Append runtime context needed by downstream tasks."""
        upstream = task.metadata.get("upstream_results") or context.state.get(
            "upstream_results", {}
        )
        original_goal = str(context.state.get("original_goal", "") or "").strip()
        original_request_header = str(
            context.state.get("original_request_header", "--- original_request ---")
            or "--- original_request ---"
        ).strip()
        upstream_results_header = str(
            context.state.get("upstream_results_header", "--- upstream_results ---")
            or "--- upstream_results ---"
        ).strip()
        parts = [task.description]
        enriched = False

        if original_goal and original_goal != task.description.strip():
            parts.append(f"\n\n{original_request_header}")
            parts.append(original_goal)
            enriched = True

        if not task.deps or not upstream:
            return "\n".join(parts) if enriched else task.description

        parts.append(f"\n\n{upstream_results_header}")
        for dep_id in task.deps:
            result = upstream.get(dep_id)
            if result:
                parts.append(f"\n[{dep_id}]:\n{result}")
                enriched = True
        return "\n".join(parts) if enriched else task.description
