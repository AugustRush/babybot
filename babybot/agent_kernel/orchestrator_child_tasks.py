"""Child-task dispatch/prompt policy helper for DynamicOrchestrator."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .model import ModelToolCall
from .types import ExecutionContext, TaskResult

logger = logging.getLogger(__name__)


def _dispatch_resource_ids(args: dict[str, Any]) -> tuple[str, ...]:
    resource_ids = args.get("resource_ids")
    if isinstance(resource_ids, (list, tuple)):
        normalized = tuple(
            str(item).strip() for item in resource_ids if str(item).strip()
        )
        if normalized:
            return normalized
    resource_id = str(args.get("resource_id", "") or "").strip()
    return (resource_id,) if resource_id else ()


def _default_is_maintenance_goal(goal: str, config: Any) -> bool:
    del config
    lowered = str(goal or "").lower()
    return any(
        token in lowered
        for token in ("repair", "fix", "maintain", "维护", "修复", "skill")
    )


def _default_has_parallel_intent(goal: str, config: Any) -> bool:
    del config
    lowered = str(goal or "").lower()
    return any(token in lowered for token in ("parallel", "同时", "并行"))


class ChildTaskRuntimeHelper:
    """Encapsulates child-task merge, dependency, and prompt policies."""

    def __init__(
        self,
        config: Any,
        *,
        is_maintenance_goal: Callable[[str, Any], bool] | None = None,
        has_parallel_intent: Callable[[str, Any], bool] | None = None,
    ) -> None:
        self._config = config
        self._is_maintenance_goal = is_maintenance_goal or _default_is_maintenance_goal
        self._has_parallel_intent = (
            has_parallel_intent or _default_has_parallel_intent
        )

    def merge_dispatch_calls_for_maintenance(
        self,
        tool_calls: tuple[ModelToolCall, ...],
        *,
        goal: str,
    ) -> tuple[ModelToolCall, ...]:
        if (
            len(tool_calls) < 2
            or not self._is_maintenance_goal(goal, self._config)
            or self._has_parallel_intent(goal, self._config)
        ):
            return tool_calls

        mergeable_ids: list[str] = []
        merged_resource_ids: list[str] = []
        merged_descriptions: list[str] = []
        timeout_candidates: list[float] = []

        for tool_call in tool_calls:
            if tool_call.name != "dispatch_task":
                continue
            resource_ids = _dispatch_resource_ids(tool_call.arguments)
            if not resource_ids or "group.scheduler" in resource_ids:
                continue
            if tool_call.arguments.get("deps"):
                continue
            mergeable_ids.append(tool_call.call_id)
            merged_resource_ids.extend(resource_ids)
            description = str(tool_call.arguments.get("description", "") or "").strip()
            if description and description not in merged_descriptions:
                merged_descriptions.append(description)
            raw_timeout = tool_call.arguments.get("timeout_s")
            try:
                if raw_timeout is not None:
                    timeout_candidates.append(float(raw_timeout))
            except (TypeError, ValueError):
                pass

        if len(mergeable_ids) < 2:
            return tool_calls

        merged_description = (
            "\n".join(
                [
                    "同一维护目标的合并子任务：",
                    *[f"- {item}" for item in merged_descriptions],
                ]
            )
            if len(merged_descriptions) > 1
            else (
                merged_descriptions[0]
                if merged_descriptions
                else str(goal or "").strip()
            )
        )
        merged_resources = list(dict.fromkeys(rid for rid in merged_resource_ids if rid))
        first_call_id = mergeable_ids[0]
        normalized_calls: list[ModelToolCall] = []
        merged_inserted = False

        for tool_call in tool_calls:
            if tool_call.call_id not in mergeable_ids:
                normalized_calls.append(tool_call)
                continue
            if merged_inserted:
                continue
            merged_args = dict(tool_call.arguments)
            merged_args.pop("resource_id", None)
            merged_args["resource_ids"] = merged_resources
            merged_args["description"] = merged_description
            merged_args.pop("deps", None)
            if timeout_candidates:
                merged_args["timeout_s"] = max(timeout_candidates)
            normalized_calls.append(
                ModelToolCall(
                    call_id=first_call_id,
                    name="dispatch_task",
                    arguments=merged_args,
                )
            )
            merged_inserted = True

        logger.info(
            "DynamicOrchestrator: coalesced %d maintenance dispatches into one task resources=%s",
            len(mergeable_ids),
            merged_resources,
        )
        return tuple(normalized_calls)

    def maintenance_serial_dependency_ids(
        self,
        runtime: Any,
        *,
        prior_live_task_ids_this_turn: tuple[str, ...],
    ) -> list[str]:
        dependency_ids: list[str] = []

        for task_id in prior_live_task_ids_this_turn:
            normalized = str(task_id or "").strip()
            if normalized and normalized not in dependency_ids:
                dependency_ids.append(normalized)
        if dependency_ids:
            return dependency_ids

        for task_id in runtime.pending_reply_blocking_task_ids():
            normalized = str(task_id or "").strip()
            if normalized and normalized not in dependency_ids:
                dependency_ids.append(normalized)
        if dependency_ids:
            return dependency_ids

        recent_success = self.recent_successful_upstream_results(
            runtime.results,
            limit=1,
        )
        if recent_success:
            return list(recent_success.keys())

        for task_id, result in reversed(list(runtime.results.items())):
            resource_id = str(result.metadata.get("resource_id", "") or "").strip()
            if resource_id == "group.scheduler":
                continue
            normalized = str(task_id or "").strip()
            if normalized:
                return [normalized]
        return []

    def normalize_child_task_description(
        self,
        *,
        description: str,
        resource_ids: tuple[str, ...],
        context: ExecutionContext,
        upstream_results: dict[str, TaskResult] | None = None,
    ) -> str:
        raw_description = str(description or "").strip()
        if not raw_description:
            return raw_description

        sentinel = getattr(self._config, "child_task_sentinel", "")
        if sentinel and sentinel in raw_description:
            return raw_description

        builder = getattr(self._config, "build_child_task_prompt", None)
        if builder is None:
            return raw_description

        original_goal = str(context.state.get("original_goal", "") or "").strip()
        upstream_outputs: dict[str, Any] = {}
        if upstream_results:
            for tid, result in upstream_results.items():
                output_snippet = str(result.output or "").strip()
                if output_snippet:
                    upstream_outputs[tid] = output_snippet

        try:
            built = builder(
                raw_description=raw_description,
                original_goal=original_goal,
                resource_ids=resource_ids,
                upstream_results=upstream_outputs,
            )
        except Exception:
            logger.exception("build_child_task_prompt raised; using raw description")
            return raw_description
        return built if built else raw_description

    def build_child_task_feedback_label(
        self,
        *,
        description: str,
        resource_ids: tuple[str, ...],
        context: ExecutionContext,
    ) -> str:
        raw_description = str(description or "").strip()
        if not raw_description:
            return "执行子任务"

        builder = getattr(self._config, "build_child_task_feedback_label", None)
        if builder is None:
            return raw_description.splitlines()[0].strip()[:80]

        original_goal = str(context.state.get("original_goal", "") or "").strip()
        try:
            built = builder(
                raw_description=raw_description,
                original_goal=original_goal,
                resource_ids=resource_ids,
            )
        except Exception:
            logger.exception(
                "build_child_task_feedback_label raised; using raw description"
            )
            return raw_description.splitlines()[0].strip()[:80]
        return str(built or raw_description).strip()[:80] or "执行子任务"

    @staticmethod
    def recent_successful_upstream_results(
        results: dict[str, TaskResult],
        *,
        limit: int = 1,
    ) -> dict[str, TaskResult] | None:
        if not results or limit <= 0:
            return None
        selected: list[tuple[str, TaskResult]] = []
        for task_id, result in reversed(list(results.items())):
            if result.status != "succeeded":
                continue
            if str(result.output or "").strip() == "":
                continue
            if result.metadata.get("auto_converged") is True:
                continue
            resource_id = str(result.metadata.get("resource_id", "") or "").strip()
            if resource_id == "group.scheduler":
                continue
            selected.append((task_id, result))
            if len(selected) >= limit:
                break
        if not selected:
            return None
        selected.reverse()
        return dict(selected)
