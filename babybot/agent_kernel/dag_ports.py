"""Port implementations: build_history_summary and ResourceBridgeExecutor."""

from __future__ import annotations

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


# ── Shared utilities ─────────────────────────────────────────────────────


def build_history_summary(tape: "Tape | None") -> str:
    """Build context summary from tape for orchestration prompts.

    Includes both the anchor summary (compacted history) AND recent
    conversation entries since the last anchor so the orchestrator can make
    context-aware decisions for follow-up messages.
    """
    if tape is None:
        return ""

    parts: list[str] = []

    # 1. Anchor summary (compacted older history)
    anchor = tape.last_anchor()
    if anchor is not None:
        state = anchor.payload.get("state") or {}
        summary = state.get("summary", "")
        if summary:
            parts.append(f"对话摘要: {summary}")
        intent = state.get("user_intent", "")
        if intent:
            parts.append(f"用户意图: {intent}")
        pending = state.get("pending", "")
        if pending:
            parts.append(f"待办: {pending}")

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
            parts.append("近期对话:\n" + "\n".join(trimmed))

    return "\n".join(parts)


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
            "\n\n你具备以下能力（用户询问时可介绍）：\n"
            + "\n".join(lines)
            + "\n当前对话不需要使用这些工具，直接回答即可。"
        )

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        # Fast path: direct answer (no tools)
        if task.metadata.get("direct_answer"):
            goal = task.description
            heartbeat = context.state.get("heartbeat")
            media_paths = context.state.get("media_paths") or ()
            stream_callback = context.state.get("stream_callback")
            tape = context.state.get("tape")
            tape_store = context.state.get("tape_store")
            history_budget = context.state.get("context_history_tokens", 2000)

            base_prompt = "你是高效助手。简洁准确地回答用户问题。不虚构工具执行结果。如需外部信息必须明确指出。"
            capability_summary = self._build_capability_summary()

            messages: list[ModelMessage] = []
            messages.append(ModelMessage(
                role="system",
                content=base_prompt + capability_summary,
            ))

            # Reuse executor.py's 3-section context builder
            if tape is not None:
                messages.extend(_build_history_messages(
                    tape, history_budget, query=goal, tape_store=tape_store,
                ))

            messages.append(ModelMessage(
                role="user",
                content=goal,
                images=tuple(media_paths),
            ))

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
        heartbeat = context.state.get("heartbeat")
        media_paths = context.state.get("media_paths")

        try:
            output, media = await self._rm.run_subagent_task(
                task_description=enriched,
                lease=lease_dict,
                agent_name=f"DAG-{task.task_id}",
                tape=tape,
                tape_store=tape_store,
                heartbeat=heartbeat,
                media_paths=media_paths,
                skill_ids=list(skill_ids),
            )
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
        )

    @staticmethod
    def _enrich_with_upstream(task: TaskContract, context: ExecutionContext) -> str:
        """Append runtime context needed by downstream scheduler-style tasks."""
        upstream = task.metadata.get("upstream_results") or context.state.get("upstream_results", {})
        resource_id = str(task.metadata.get("resource_id", "") or "")
        original_goal = str(context.state.get("original_goal", "") or "").strip()
        parts = [task.description]
        enriched = False

        if (
            resource_id == "group.scheduler"
            and original_goal
            and original_goal != task.description.strip()
        ):
            parts.append("\n\n--- 原始用户请求 ---")
            parts.append(original_goal)
            enriched = True

        if not task.deps or not upstream:
            return "\n".join(parts) if enriched else task.description

        parts.append("\n\n--- 上游任务结果 ---")
        for dep_id in task.deps:
            result = upstream.get(dep_id)
            if result:
                parts.append(f"\n[{dep_id}]:\n{result}")
                enriched = True
        return "\n".join(parts) if enriched else task.description
