"""DAG-mode port implementations: LLMPlanner, ResourceBridgeExecutor, LLMSynthesizer."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .executor import _build_history_messages
from .model import ModelMessage
from .types import (
    ExecutionContext,
    ExecutionPlan,
    FinalResult,
    TaskContract,
    TaskResult,
    ToolLease,
)

if TYPE_CHECKING:
    from ..context import Tape
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────────


class PlannedTask(BaseModel):
    """One planned subtask from the LLM planner."""

    task_id: str
    resource_id: str
    description: str
    deps: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    """Structured output from the LLM planner."""

    need_tools: bool = False
    direct_answer: str = ""
    tasks: list[PlannedTask] = Field(default_factory=list)
    rationale: str = ""


# ── LLMPlanner ───────────────────────────────────────────────────────────


class LLMPlanner:
    """Plans task DAGs via LLM structured output."""

    _IMAGE_PATTERNS = [
        r"画一?张", r"画一个", r"画一只", r"画一下",
        r"生成.*图", r"生成.*图片", r"做一张图", r"绘制",
        r"\bdraw\b", r"\bimage\b",
    ]

    _TIME_PATTERNS = [
        r"\d+\s*分钟", r"\d+\s*小时", r"\d+\s*秒",
        r"\d+\s*天后", r"每天", r"每周", r"每月",
        r"每隔", r"每\s*\d+",
        r"\d{1,2}[:\uff1a]\d{2}",
        r"\bin\s+\d+\s+minutes?\b", r"\bevery\s+\d+\b",
    ]

    _ACTION_PATTERNS = [
        r"提醒", r"发送", r"发消息", r"通知", r"推送",
        r"定时", r"延时", r"定期",
        r"\bremind\b", r"\bschedule\b", r"\bsend\b",
    ]

    def __init__(
        self,
        gateway: "OpenAICompatibleGateway",
        resource_manager: "ResourceManager",
    ) -> None:
        self._gateway = gateway
        self._rm = resource_manager

    def _needs_tools_heuristic(self, goal: str, briefs: list[dict[str, Any]]) -> bool:
        """Local fast check: does the goal likely need tools?

        Returns True if tools *might* be needed (fall through to LLM planner).
        Returns False only when we are confident no tools are required.
        """
        if self._looks_like_image_generation(goal):
            return True
        if self._looks_like_scheduling(goal):
            return True

        # Check if any active resource with tools has keyword overlap
        goal_lower = goal.lower()
        has_any_active_tool = False
        for brief in briefs:
            if not brief.get("active", False) or brief.get("tool_count", 0) <= 0:
                continue
            has_any_active_tool = True
            for field in ("name", "purpose"):
                value = brief.get(field, "")
                if not value:
                    continue
                val_lower = value.lower()
                # Direct substring match
                if val_lower in goal_lower:
                    return True
                # Space-separated word match (for English/mixed text)
                for word in val_lower.split():
                    if len(word) >= 2 and word in goal_lower:
                        return True
                # Character bigram match (for CJK text without spaces)
                for i in range(len(val_lower) - 1):
                    bigram = val_lower[i:i + 2]
                    if bigram in goal_lower:
                        return True

        # No active tools at all → definitely no tools needed
        if not has_any_active_tool:
            return False

        # Have active tools but no keyword match → confidently no tools
        return False

    async def plan(self, goal: str, context: ExecutionContext) -> ExecutionPlan:
        briefs = self._rm.get_resource_briefs()

        # Local fast path: skip LLM planner if clearly no tools needed
        if not self._needs_tools_heuristic(goal, briefs):
            logger.info("Planner fast-path: no tools needed for goal=%r", goal)
            return ExecutionPlan(tasks=(
                TaskContract(
                    task_id="direct_answer",
                    description=goal,
                    metadata={"direct_answer": True},
                ),
            ))

        # Build history summary from tape if available
        tape: Tape | None = context.state.get("tape")
        history_summary = self._build_history_summary(tape)

        heartbeat = context.state.get("heartbeat")

        plan_output = await self._gateway.complete_structured(
            system_prompt=self._build_planner_prompt(briefs, history_summary),
            user_prompt=goal,
            model_cls=PlanOutput,
            heartbeat=heartbeat,
        )

        if plan_output is None:
            # LLM parse failure → fallback to direct answer
            return ExecutionPlan(tasks=(
                TaskContract(
                    task_id="direct_answer",
                    description=goal,
                    metadata={"direct_answer": True},
                ),
            ))

        # Apply heuristics before fast-path check (e.g. image generation override)
        plan_output = self._apply_heuristics(goal, plan_output, briefs)

        # Fast path: no tools needed → single direct-answer task
        if not plan_output.need_tools:
            return ExecutionPlan(tasks=(
                TaskContract(
                    task_id="direct_answer",
                    description=plan_output.direct_answer or goal,
                    metadata={"direct_answer": True},
                ),
            ))

        # Convert to ExecutionPlan
        contracts: list[TaskContract] = []
        for t in plan_output.tasks:
            scope = self._rm.resolve_resource_scope(t.resource_id)
            lease = self._build_lease(scope) if scope else ToolLease()
            skill_ids = scope[1] if scope else ()
            contracts.append(TaskContract(
                task_id=t.task_id,
                description=t.description,
                deps=tuple(t.deps),
                lease=lease,
                metadata={
                    "resource_id": t.resource_id,
                    "skill_ids": list(skill_ids),
                },
            ))

        if not contracts:
            # No valid tasks → fallback to direct answer
            return ExecutionPlan(tasks=(
                TaskContract(
                    task_id="direct_answer",
                    description=goal,
                    metadata={"direct_answer": True},
                ),
            ))

        return ExecutionPlan(
            tasks=tuple(contracts),
            rationale=plan_output.rationale,
        )

    @staticmethod
    def _build_lease(scope: tuple[dict[str, Any], tuple[str, ...]] | None) -> ToolLease:
        if scope is None:
            return ToolLease()
        lease_dict = scope[0]
        return ToolLease(
            include_groups=tuple(lease_dict.get("include_groups", ())),
            include_tools=tuple(lease_dict.get("include_tools", ())),
            exclude_tools=tuple(lease_dict.get("exclude_tools", ())),
        )

    @staticmethod
    def _build_history_summary(tape: "Tape | None") -> str:
        """Build context for the planner from tape.

        Includes both the anchor summary (compacted history) AND recent
        conversation entries since the last anchor so the planner can make
        context-aware routing decisions for follow-up messages.
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
                # Truncate long messages to keep planner prompt manageable
                if len(content) > 300:
                    content = content[:300] + "..."
                recent_lines.append(f"{role}: {content}")
            if recent_lines:
                # Keep at most the last 10 messages to bound prompt size
                trimmed = recent_lines[-10:]
                parts.append("近期对话:\n" + "\n".join(trimmed))

        return "\n".join(parts)

    @staticmethod
    def _build_planner_prompt(
        briefs: list[dict[str, Any]],
        history_summary: str,
    ) -> str:
        catalog = json.dumps(briefs, ensure_ascii=False)
        history_block = ""
        if history_summary:
            history_block = f"\n\n上下文信息:\n{history_summary}\n"
        return (
            "你是任务规划Agent。根据用户请求和资源目录，决定是否需要工具，并生成执行计划。\n"
            "请严格输出 JSON，字段：\n"
            '{"need_tools":true/false,"direct_answer":"字符串",'
            '"tasks":[{"task_id":"简短标识","resource_id":"资源ID","description":"子任务说明","deps":["依赖的task_id"]}],'
            '"rationale":"规划理由"}\n'
            "规则：\n"
            "1. 不需要工具时，need_tools=false，给出 direct_answer，tasks 为空。\n"
            "2. 需要工具时，need_tools=true，direct_answer 置空，tasks 至少一项。\n"
            "3. tasks 中的 resource_id 必须来自资源目录，禁止虚构ID。\n"
            "4. 每个 task 只绑定一个 resource_id。\n"
            "5. need_tools=true 时，只能选择 active=true 且 tool_count>0 的资源。\n"
            "6. 独立任务不设 deps → 并行执行。\n"
            "7. 有依赖的任务设置 deps → 串行等待上游结果。\n"
            "8. task_id 简短有意义（如 search_web, gen_image）。\n"
            "9. 定时/延时发送 → 只需一个 task 绑定含定时工具的资源，定时系统会自动投递消息，不要额外创建发送任务。\n"
            f"{history_block}"
            f"资源目录：{catalog}"
        )

    @classmethod
    def _looks_like_image_generation(cls, text: str) -> bool:
        text = (text or "").strip().lower()
        if not text:
            return False
        return any(re.search(p, text) for p in cls._IMAGE_PATTERNS)

    @classmethod
    def _looks_like_scheduling(cls, text: str) -> bool:
        t = (text or "").strip().lower()
        has_time = any(re.search(p, t) for p in cls._TIME_PATTERNS)
        has_action = any(re.search(p, t) for p in cls._ACTION_PATTERNS)
        return has_time and has_action

    def _apply_heuristics(
        self,
        goal: str,
        plan: PlanOutput,
        briefs: list[dict[str, Any]],
    ) -> PlanOutput:
        # Scheduling heuristic: force single task to avoid split schedule+send
        if self._looks_like_scheduling(goal):
            for item in briefs:
                if item.get("id") == "group.basic" and item.get("active"):
                    return PlanOutput(
                        need_tools=True,
                        tasks=[PlannedTask(
                            task_id="schedule_task",
                            resource_id="group.basic",
                            description=goal.strip(),
                        )],
                        rationale="Scheduling heuristic",
                    )

        if plan.need_tools and plan.tasks:
            return plan
        if not self._looks_like_image_generation(goal):
            return plan
        for item in briefs:
            if item.get("id") != "skill.text-to-image":
                continue
            if not item.get("active") or int(item.get("tool_count") or 0) <= 0:
                return plan
            return PlanOutput(
                need_tools=True,
                tasks=[PlannedTask(
                    task_id="gen_image",
                    resource_id="skill.text-to-image",
                    description=goal.strip() or "根据用户要求生成图片",
                )],
                rationale="Image generation heuristic",
            )
        return plan


# ── ResourceBridgeExecutor ───────────────────────────────────────────────


class ResourceBridgeExecutor:
    """Executes DAG tasks by delegating to ResourceManager.run_subagent_task."""

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

            messages.append(ModelMessage(role="user", content=goal))

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
        """Append upstream task results to the task description."""
        upstream = context.state.get("upstream_results", {})
        if not task.deps or not upstream:
            return task.description

        parts = [task.description, "\n\n--- 上游任务结果 ---"]
        for dep_id in task.deps:
            result = upstream.get(dep_id)
            if result:
                parts.append(f"\n[{dep_id}]:\n{result}")
        return "\n".join(parts)


# ── LLMSynthesizer ───────────────────────────────────────────────────────


class LLMSynthesizer:
    """Synthesizes final output from DAG execution results."""

    def __init__(self, gateway: "OpenAICompatibleGateway") -> None:
        self._gateway = gateway

    async def synthesize(
        self,
        goal: str,
        plan: ExecutionPlan,
        results: dict[str, TaskResult],
        context: ExecutionContext,
    ) -> FinalResult:
        # Single task: return directly
        if len(results) == 1:
            r = next(iter(results.values()))
            if r.status == "succeeded":
                return FinalResult(
                    conclusion=r.output,
                    task_results=dict(results),
                )
            return FinalResult(
                conclusion=r.error or "任务失败。",
                failed_tasks=[r.task_id],
                task_results=dict(results),
            )

        # Multiple tasks: LLM merge
        heartbeat = context.state.get("heartbeat")
        stream_callback = context.state.get("stream_callback")
        merge_prompt = (
            "你是主Agent，负责整合多个子Agent结果并回复用户。\n"
            "要求：\n"
            "1. 只基于给定子任务结果回答，禁止编造未执行结果。\n"
            "2. 对失败子任务明确标注失败原因。\n"
            "3. 输出简洁、结论优先。"
        )

        subtask_results = []
        failed: list[str] = []
        for task_id, result in results.items():
            entry: dict[str, Any] = {
                "task_id": task_id,
                "ok": result.status == "succeeded",
                "output": result.output if result.status == "succeeded" else "",
                "error": result.error if result.status != "succeeded" else "",
            }
            subtask_results.append(entry)
            if result.status != "succeeded":
                failed.append(task_id)

        user_payload = json.dumps(
            {"user_request": goal, "subtask_results": subtask_results},
            ensure_ascii=False,
            indent=2,
        )

        merged = await self._gateway.complete(
            system_prompt=merge_prompt,
            user_prompt=user_payload,
            heartbeat=heartbeat,
            on_stream_text=stream_callback,
        )

        return FinalResult(
            conclusion=merged.strip() or "任务完成，但没有可返回的结果。",
            failed_tasks=failed,
            task_results=dict(results),
        )
