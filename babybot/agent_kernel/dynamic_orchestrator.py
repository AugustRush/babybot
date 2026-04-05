"""Dynamic orchestration loop — main agent dispatches sub-agents via tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import uuid
import inspect
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from ..task_contract import assert_runtime_matches_contract
from .context import ContextManager
from .dag_ports import ResourceBridgeExecutor, build_history_summary
from .errors import classify_error, retry_delay_seconds as default_retry_delay_seconds
from .execution_constraints import (
    build_team_execution_policy,
    format_execution_constraints_for_prompt,
    normalize_execution_constraints,
)
from .model import ModelMessage, ModelRequest, ModelResponse, ModelToolCall
from .orchestrator_config import OrchestratorConfig
from .plan_notebook import PlanNotebook, create_root_notebook
from .plan_notebook_context import (
    build_completion_context_view,
    build_orchestrator_context_view,
)
from .types import ExecutionContext, FinalResult, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..heartbeat import TaskHeartbeatRegistry
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager
    from .protocols import ExecutorPort

logger = logging.getLogger(__name__)

_EXECUTION_NODE_KINDS = {
    "tool_workflow",
    "team_debate",
    "team_cooperative",
    "plan_step",
}
_TEAM_NODE_KINDS = {"team_debate", "team_cooperative"}
_FORCE_CONVERGE_TOOL_NAMES = frozenset(
    {"wait_for_tasks", "get_task_result", "reply_to_user"}
)


def _resolve_orchestrator_config(
    config: OrchestratorConfig | None,
) -> OrchestratorConfig:
    if config is not None:
        return config
    try:
        from ..orchestrator_prompts import build_orchestrator_config

        resolved = build_orchestrator_config()
        if isinstance(resolved, OrchestratorConfig):
            return resolved
    except Exception:
        logger.exception(
            "Failed to load application orchestrator config; falling back to generic defaults"
        )
    return OrchestratorConfig()


# ── Orchestration tool schemas (OpenAI function-calling format) ──────────

_ORCHESTRATION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "dispatch_task",
            "description": (
                "Create a sub-agent task and immediately return a task_id (non-blocking). "
                "The sub-agent will execute the task using the specified resource(s)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "Single resource ID from the available-resources list.",
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple resource IDs when a sub-task needs combined capabilities.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of the sub-task.",
                    },
                    "deps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task_ids that must complete before this task starts.",
                        "default": [],
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Sub-task timeout in seconds. Omit to use the runtime default.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_tasks",
            "description": (
                "Block until all specified tasks complete and return a JSON result map. "
                "Each result contains status/output/error and reply_artifacts_ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task_ids to wait for.",
                    },
                },
                "required": ["task_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_result",
            "description": (
                "Query the current status and result of a task (non-blocking, returns JSON). "
                "Result contains status/output/error and reply_artifacts_ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task_id to query.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_user",
            "description": (
                "Send the final reply to the user. The orchestration loop ends after this call. "
                "This tool must be called alone as the last action. "
                "The runtime will automatically attach any collected artifacts to the reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to send to the user.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_team",
            "description": (
                "Launch a group of agents for collaborative work. Supports two modes:\n"
                "- debate (default): multi-round debate/review/brainstorm with agents taking turns.\n"
                "- cooperative: parallel task execution where agents pick tasks from a shared list "
                "and broadcast results to downstream dependents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Collaboration topic / high-level goal.",
                    },
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string"},
                                "description": {"type": "string"},
                                "resource_id": {
                                    "type": "string",
                                    "description": "Optional: resource ID for this agent.",
                                },
                                "skill_id": {
                                    "type": "string",
                                    "description": "Optional: skill name whose role/description/prompt will be inherited.",
                                },
                            },
                            "required": ["id", "role", "description"],
                        },
                        "description": "Agents participating in the collaboration (at least 2).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["debate", "cooperative"],
                        "description": (
                            "Collaboration mode. debate=multi-round discussion (default), "
                            "cooperative=parallel task execution (requires tasks parameter)."
                        ),
                        "default": "debate",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Maximum discussion rounds in debate mode (default 5).",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "Unique task identifier.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Task description.",
                                },
                                "deps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of task_ids this task depends on.",
                                    "default": [],
                                },
                            },
                            "required": ["task_id", "description"],
                        },
                        "description": (
                            "Task list for cooperative mode. Each task may declare deps; "
                            "agents will automatically pick up available tasks and broadcast results."
                        ),
                    },
                },
                "required": ["topic", "agents"],
            },
        },
    },
)

_ORCHESTRATION_TOOL_BY_NAME: dict[str, dict[str, Any]] = {
    tool["function"]["name"]: tool for tool in _ORCHESTRATION_TOOLS
}
# NOTE: _ORCHESTRATION_TOOL_BY_NAME is kept for backward compatibility with any
# external callers. Internally DynamicOrchestrator uses self._orchestration_tools.


# ── System prompt builder ────────────────────────────────────────────────

# Minimal language-agnostic fallback used when no OrchestratorConfig is
# injected.  Application-specific content belongs in OrchestratorConfig
# (supplied by the application layer, e.g. orchestrator_prompts.py).
_SYSTEM_PROMPT_ROLE = (
    "You are an orchestration agent. "
    "Dispatch sub-tasks to available resources and reply to the user when done."
)

# Empty by default — patterns are supplied via OrchestratorConfig.
_DEFERRED_TASK_PATTERNS: tuple[str, ...] = ()

_DEFERRED_TASK_GUIDANCE = ""

# Default NLU token lists — empty by design (language-agnostic fallback).
# Override via OrchestratorConfig.multi_step_tokens / parallel_tokens.
_MULTI_STEP_TOKENS: tuple[str, ...] = ()
_PARALLEL_TOKENS: tuple[str, ...] = ()


def _build_resource_catalog(
    briefs: list[dict[str, Any]],
    config: OrchestratorConfig | None = None,
) -> str:
    cfg = config or OrchestratorConfig()

    specialist_types = {"skill", "mcp"}
    specialist_lines: list[str] = []
    general_lines: list[str] = []

    for b in briefs:
        if not b.get("active"):
            continue
        rid = b.get("id", "?")
        resource_type = b.get("type", "")
        name = b.get("name", "?")
        purpose = b.get("purpose", "")
        tc = b.get("tool_count", 0)
        preview = (
            ""
            if resource_type in {"mcp", "skill"}
            else ", ".join(b.get("tools_preview") or [])
        )
        preview_text = (
            (cfg.resource_catalog_preview_prefix + preview) if preview else ""
        )
        line = cfg.resource_catalog_line.format(
            rid=rid,
            name=name,
            purpose=purpose,
            tc=tc,
            preview_text=preview_text,
        )
        if resource_type in specialist_types:
            specialist_lines.append(line)
        else:
            general_lines.append(line)

    if not specialist_lines and not general_lines:
        return cfg.resource_catalog_empty

    # When tier headers are configured and both tiers have content, display
    # a tiered catalog; otherwise fall back to flat list.
    use_tiers = bool(
        cfg.resource_catalog_specialist_header
        and cfg.resource_catalog_general_header
        and specialist_lines
        and general_lines
    )

    if use_tiers:
        parts = [cfg.resource_catalog_header]
        parts.append(cfg.resource_catalog_specialist_header)
        parts.extend(f"  {line}" for line in specialist_lines)
        parts.append(cfg.resource_catalog_general_header)
        parts.extend(f"  {line}" for line in general_lines)
        return "\n".join(parts)

    all_lines = specialist_lines + general_lines
    return cfg.resource_catalog_header + "\n".join(all_lines)


def _needs_deferred_task_guidance(
    goal: str, config: OrchestratorConfig | None = None
) -> bool:
    lowered = (goal or "").strip()
    patterns = (
        config.deferred_task_patterns if config else None
    ) or _DEFERRED_TASK_PATTERNS
    return any(pattern in lowered for pattern in patterns)


def _dispatch_resource_ids(args: dict[str, Any]) -> tuple[str, ...]:
    raw_multi = args.get("resource_ids")
    resource_ids: list[str] = []
    if isinstance(raw_multi, (list, tuple)):
        resource_ids.extend(
            str(item).strip() for item in raw_multi if str(item).strip()
        )
    single = str(args.get("resource_id", "") or "").strip()
    if single:
        resource_ids.append(single)
    return tuple(dict.fromkeys(resource_ids))


def _provider_policy_hints(
    resource_manager: "ResourceManager",
    goal: str,
    config: OrchestratorConfig | None = None,
) -> list[str]:
    provider = getattr(resource_manager, "_observability_provider", None)
    if provider is None:
        return []
    cfg = config or OrchestratorConfig()
    text = str(goal or "").strip()
    build_features = getattr(provider, "_build_policy_state_features", None)
    multi_step_tokens = cfg.multi_step_tokens or _MULTI_STEP_TOKENS
    parallel_tokens = cfg.parallel_tokens or _PARALLEL_TOKENS
    features: dict[str, Any] = {
        "task_shape": "multi_step"
        if any(token in text for token in multi_step_tokens)
        else "single_step",
        "input_length": len(text),
    }
    if callable(build_features):
        _raw = build_features(goal)
        result: dict[str, Any]
        result = _raw if isinstance(_raw, dict) else {}
        if result:
            features.update(result)
    independent_subtasks = 1
    for token in parallel_tokens:
        independent_subtasks += text.count(token)
    features["independent_subtasks"] = max(1, independent_subtasks)
    hints: list[str] = []
    for method_name in ("choose_scheduling_policy", "choose_worker_policy"):
        chooser = getattr(provider, method_name, None)
        if not callable(chooser):
            continue
        payload = chooser(features=features)
        if isinstance(payload, dict):
            action_name = str(
                payload.get("action_name") or payload.get("name") or ""
            ).strip()
            hint = str(payload.get("hint") or "").strip()
        else:
            action_name = str(
                getattr(payload, "action_name", "")
                or getattr(payload, "name", "")
                or ""
            ).strip()
            hint = str(getattr(payload, "hint", "") or "").strip()
        if not hint:
            continue
        if method_name == "choose_worker_policy" and action_name == "allow_worker":
            continue
        hints.append(hint)
    return hints


def _goal_has_explicit_parallel_intent(
    goal: str, config: OrchestratorConfig | None = None
) -> bool:
    text = str(goal or "").strip()
    if not text:
        return False
    tokens = (config.parallel_tokens if config else None) or _PARALLEL_TOKENS
    return any(token in text for token in tokens)


def _is_maintenance_goal(goal: str, config: OrchestratorConfig | None = None) -> bool:
    text = str(goal or "").strip()
    if not text:
        return False
    detector = config.is_maintenance_task if config else None
    if callable(detector):
        try:
            return bool(detector(text))
        except Exception:
            logger.exception("maintenance task detector raised; falling back to False")
    return False


def _emit_policy_decision(
    context: ExecutionContext,
    *,
    decision_kind: str,
    action_name: str,
    state_features: dict[str, Any] | None = None,
) -> None:
    context.emit(
        "policy_decision",
        decision_kind=decision_kind,
        action_name=action_name,
        state_features=dict(state_features or {}),
    )


@dataclass(frozen=True)
class ChildTaskEvent:
    """Lifecycle event emitted for one child task."""

    flow_id: str
    task_id: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)


class InMemoryChildTaskBus:
    """Simple in-memory event sink for child-task lifecycle events."""

    def __init__(self) -> None:
        self._events: dict[str, list[ChildTaskEvent]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[ChildTaskEvent]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: ChildTaskEvent) -> None:
        async with self._lock:
            self._events.setdefault(event.flow_id, []).append(event)
            for queue in list(self._subscribers.get(event.flow_id, ())):
                queue.put_nowait(event)

    def events_for(self, flow_id: str) -> list[ChildTaskEvent]:
        return list(self._events.get(flow_id, ()))

    async def subscribe(self, flow_id: str) -> AsyncIterator[ChildTaskEvent]:
        queue: asyncio.Queue[ChildTaskEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(flow_id, []).append(queue)
            for event in self._events.get(flow_id, ()):
                queue.put_nowait(event)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(flow_id, [])
                if queue in subscribers:
                    subscribers.remove(queue)
                if not subscribers:
                    self._subscribers.pop(flow_id, None)

    def clear_flow(self, flow_id: str) -> None:
        """Remove all stored events and subscribers for a completed flow."""
        self._events.pop(flow_id, None)
        self._subscribers.pop(flow_id, None)


class InProcessChildTaskRuntime:
    """Current child-task runtime backed by local asyncio tasks."""

    MAX_RETRY_CAP = 8

    def __init__(
        self,
        *,
        flow_id: str,
        resource_manager: "ResourceManager",
        bridge: ResourceBridgeExecutor,
        child_task_bus: InMemoryChildTaskBus,
        task_heartbeat_registry: "TaskHeartbeatRegistry",
        max_parallel: int,
        max_tasks: int,
        max_retries: int = 0,
        retry_delay_seconds: Callable[[int], float] = default_retry_delay_seconds,
        default_timeout_s: float | None = 300.0,
        stale_after_s: float | None = None,
        progress_poll_interval_s: float = 0.05,
        plan_step_id: str = "",
    ) -> None:
        self._flow_id = flow_id
        self._rm = resource_manager
        self._bridge = bridge
        self._child_task_bus = child_task_bus
        self._task_heartbeat_registry = task_heartbeat_registry
        self._max_tasks = max_tasks
        self._max_retries = min(self.MAX_RETRY_CAP, max(0, int(max_retries)))
        self._retry_delay_seconds = retry_delay_seconds
        self._default_timeout_s = self._coerce_timeout(default_timeout_s)
        self._stale_after_s = stale_after_s
        self._progress_poll_interval_s = max(0.02, float(progress_poll_interval_s))
        self._plan_step_id = str(plan_step_id or "").strip()
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._in_flight: dict[str, asyncio.Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._task_state: dict[str, dict[str, Any]] = {}
        self._dead_letters: dict[str, dict[str, Any]] = {}
        self._cancelling = False

    @staticmethod
    def _coerce_timeout(raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        try:
            timeout_s = float(raw_value)
        except (TypeError, ValueError):
            return None
        if timeout_s <= 0:
            return None
        return timeout_s

    @property
    def in_flight(self) -> dict[str, asyncio.Task]:
        return self._in_flight

    @property
    def results(self) -> dict[str, TaskResult]:
        return self._results

    def task_state_snapshot(self, task_id: str) -> dict[str, Any]:
        return dict(self._task_state.get(task_id, {}))

    def pending_task_ids(self) -> list[str]:
        return sorted(
            task_id
            for task_id, task in self._in_flight.items()
            if not task.done() and task_id not in self._results
        )

    def pending_reply_blocking_task_ids(self) -> list[str]:
        blocking: list[str] = []
        for task_id in self.pending_task_ids():
            resource_id = str(
                self._task_state.get(task_id, {}).get("resource_id", "") or ""
            )
            if resource_id == "group.scheduler":
                continue
            blocking.append(task_id)
        return blocking

    def all_tasks_dead_lettered_with_no_success(self) -> tuple[bool, list[str]]:
        """Return (True, dead_letter_ids) if every dispatched task dead-lettered and
        none succeeded, so the model cannot fabricate a reply from real results."""
        if not self._results:
            return False, []
        succeeded = [tid for tid, r in self._results.items() if r.status == "succeeded"]
        if succeeded:
            return False, []
        dead = [
            tid
            for tid, r in self._results.items()
            if r.status == "failed" and r.metadata.get("dead_lettered") is True
        ]
        if not dead:
            return False, []
        # Only trigger if *all* non-scheduler tasks failed (dead_lettered)
        non_scheduler_results = [
            tid
            for tid, r in self._results.items()
            if str(r.metadata.get("resource_id", "") or "") != "group.scheduler"
        ]
        if non_scheduler_results and all(tid in dead for tid in non_scheduler_results):
            return True, dead
        return False, []

    def _update_task_state(self, task_id: str, **payload: Any) -> None:
        current = self._task_state.setdefault(task_id, {})
        current.update(payload)
        status = payload.get("status")
        if isinstance(status, str) and status:
            self._task_heartbeat_registry.beat(
                self._flow_id,
                task_id,
                status=status,
            )

    def _stale_task_ids(self, task_ids: list[str]) -> set[str]:
        if self._stale_after_s is None:
            return set()
        stale = self._task_heartbeat_registry.stale_tasks(
            self._flow_id,
            stale_after_s=self._stale_after_s,
        )
        return {
            task_id
            for task_id in task_ids
            if task_id in self._in_flight
            and task_id in stale
            and task_id not in self._results
        }

    def _event_payload(self, **payload: Any) -> dict[str, Any]:
        normalized = dict(payload)
        if self._plan_step_id:
            normalized.setdefault("plan_step_id", self._plan_step_id)
        return normalized

    @staticmethod
    def _normalize_dispatch_description(description: str) -> str:
        return " ".join(str(description or "").split()).strip().lower()

    @classmethod
    def _dispatch_signature(
        cls,
        resource_ids: tuple[str, ...],
        description: str,
    ) -> tuple[tuple[str, ...], str]:
        normalized_ids = tuple(
            sorted({rid.strip() for rid in resource_ids if rid.strip()})
        )
        return normalized_ids, cls._normalize_dispatch_description(description)

    def _duplicate_dispatch_reason(
        self,
        *,
        resource_ids: tuple[str, ...],
        description: str,
    ) -> str | None:
        target_signature = self._dispatch_signature(resource_ids, description)

        for task_id, state in self._task_state.items():
            state_resource_ids = state.get("resource_ids")
            if isinstance(state_resource_ids, (list, tuple)):
                state_ids = tuple(str(item).strip() for item in state_resource_ids)
            else:
                state_resource_id = str(state.get("resource_id", "") or "").strip()
                state_ids = (state_resource_id,) if state_resource_id else ()
            state_description = str(state.get("description", "") or "")
            if (
                self._dispatch_signature(state_ids, state_description)
                != target_signature
            ):
                continue
            state_status = str(state.get("status", "") or "").strip().lower()
            if (
                state_status in {"queued", "started", "retrying"}
                and task_id in self._in_flight
                and task_id not in self._results
            ):
                return f"error: similar task already in progress: {task_id}"

        for task_id, result in self._results.items():
            metadata = dict(result.metadata or {})
            metadata_resource_ids = metadata.get("resource_ids")
            if isinstance(metadata_resource_ids, (list, tuple)):
                metadata_ids = tuple(
                    str(item).strip() for item in metadata_resource_ids
                )
            else:
                metadata_resource_id = str(
                    metadata.get("resource_id", "") or ""
                ).strip()
                metadata_ids = (metadata_resource_id,) if metadata_resource_id else ()
            metadata_description = str(metadata.get("description", "") or "")
            if (
                self._dispatch_signature(metadata_ids, metadata_description)
                != target_signature
            ):
                continue
            if result.status == "failed" and metadata.get("dead_lettered") is True:
                last_error = str(result.error or "").strip()
                reason = f" ({last_error})" if last_error else ""
                return (
                    f"error: task {task_id} permanently failed (dead-lettered){reason}; "
                    "do NOT retry with a similar description — "
                    "summarize partial results or inform the user of the failure"
                )
        return None

    async def _promote_stalled_tasks(self, task_ids: list[str]) -> None:
        for task_id in self._stale_task_ids(task_ids):
            running = self._in_flight.pop(task_id, None)
            if running is not None:
                running.cancel()
            self._update_task_state(
                task_id,
                status="recoverable",
                error="child task heartbeat stalled",
            )
            await self._child_task_bus.publish(
                ChildTaskEvent(
                    flow_id=self._flow_id,
                    task_id=task_id,
                    event="stalled",
                    payload=self._event_payload(error="child task heartbeat stalled"),
                )
            )

    @staticmethod
    def _normalize_result(
        task_id: str,
        result: TaskResult,
        *,
        attempts: int,
        base_metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        metadata = dict(base_metadata or {})
        metadata.update(dict(result.metadata))
        return TaskResult(
            task_id=task_id,
            status=result.status,
            output=result.output,
            error=result.error,
            artifacts=tuple(result.artifacts or ()),
            attempts=attempts,
            metadata=metadata,
        )

    @staticmethod
    def _result_artifacts(result: TaskResult) -> tuple[str, ...]:
        if result.artifacts:
            return tuple(result.artifacts)
        legacy = result.metadata.get("media_paths")
        if isinstance(legacy, (list, tuple)):
            return tuple(str(item) for item in legacy if str(item).strip())
        return ()

    @classmethod
    def _result_payload(cls, result: TaskResult) -> dict[str, Any]:
        artifacts = cls._result_artifacts(result)
        payload: dict[str, Any] = {
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "reply_artifacts_ready": bool(artifacts),
            "reply_artifacts_count": len(artifacts),
        }
        metadata = dict(result.metadata or {})
        for key in ("resource_id", "description", "error_type"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value
        for key in ("resource_ids", "skill_ids"):
            value = metadata.get(key)
            if isinstance(value, (list, tuple)):
                normalized = [str(item) for item in value if str(item).strip()]
                if normalized:
                    payload[key] = normalized
        if metadata.get("dead_lettered") is True:
            payload["dead_lettered"] = True
        if isinstance(metadata.get("retryable"), bool):
            payload["retryable"] = metadata["retryable"]
        if artifacts:
            payload["reply_artifacts_preview"] = list(artifacts[:3])
        return payload

    def _record_dead_letter(
        self,
        task_id: str,
        result: TaskResult,
        *,
        error_type: str,
        retryable: bool,
        max_attempts: int,
    ) -> TaskResult:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "dead_lettered": True,
                "error_type": error_type,
                "retryable": retryable,
                "max_attempts": max_attempts,
            }
        )
        final_result = TaskResult(
            task_id=task_id,
            status="failed",
            output=result.output,
            error=result.error,
            artifacts=tuple(result.artifacts or ()),
            attempts=result.attempts,
            metadata=metadata,
        )
        self._results[task_id] = final_result
        self._dead_letters[task_id] = {
            "task_id": task_id,
            "attempts": final_result.attempts,
            "max_attempts": max_attempts,
            "last_error": final_result.error,
            "error_type": error_type,
            "retryable": retryable,
        }
        self._update_task_state(
            task_id,
            status="failed",
            output=final_result.output,
            error=final_result.error,
            attempts=final_result.attempts,
            metadata=final_result.metadata,
            max_attempts=max_attempts,
            last_error=final_result.error,
            error_type=error_type,
        )
        return final_result

    async def dispatch(
        self,
        args: dict[str, Any],
        *,
        task_counter: int,
        context: ExecutionContext,
    ) -> str:
        if task_counter >= self._max_tasks:
            return f"error: task limit reached (max {self._max_tasks})"

        resource_ids = _dispatch_resource_ids(args)
        description: str = args.get("description", "")
        deps: list[str] = args.get("deps", [])
        if not resource_ids:
            return "error: resource_id or resource_ids is required"
        resolved_scopes: list[tuple[dict[str, Any], tuple[str, ...]]] = []
        for resource_id in resource_ids:
            scope = self._rm.resolve_resource_scope(resource_id, require_tools=True)
            if scope is None:
                return f"error: resource not available: {resource_id}"
            resolved_scopes.append(scope)

        for dep in deps:
            if dep not in self._in_flight and dep not in self._results:
                return f"error: unknown dep task_id: {dep}"

        duplicate_reason = self._duplicate_dispatch_reason(
            resource_ids=resource_ids,
            description=description,
        )
        if duplicate_reason is not None:
            return duplicate_reason

        task_id = f"task_{task_counter}_{uuid.uuid4().hex[:6]}"
        flow_id = self._flow_id
        task_kind = str(args.get("__task_kind", "child_task") or "child_task")
        include_groups: set[str] = set()
        include_tools: set[str] = set()
        exclude_tools: set[str] = set()
        merged_skill_ids: list[str] = []
        seen_skill_ids: set[str] = set()
        for lease_dict, skill_ids in resolved_scopes:
            include_groups.update(lease_dict.get("include_groups", ()))
            include_tools.update(lease_dict.get("include_tools", ()))
            exclude_tools.update(lease_dict.get("exclude_tools", ()))
            for skill_id in skill_ids:
                normalized = str(skill_id).strip()
                if normalized and normalized not in seen_skill_ids:
                    seen_skill_ids.add(normalized)
                    merged_skill_ids.append(normalized)
        lease = ToolLease(
            include_groups=tuple(sorted(include_groups)),
            include_tools=tuple(sorted(include_tools)),
            exclude_tools=tuple(sorted(exclude_tools)),
        )
        primary_resource_id = resource_ids[0]
        notebook_node_id = ""
        notebook = context.state.get("plan_notebook")
        if isinstance(notebook, PlanNotebook):
            parent_id = str(
                context.state.get("current_notebook_node_id", "") or notebook.root_node_id
            )
            if parent_id not in notebook.nodes:
                parent_id = notebook.root_node_id
            task_map = context.state.setdefault("notebook_task_map", {})
            dep_node_ids = tuple(
                task_map[dep_id]
                for dep_id in deps
                if isinstance(task_map, dict)
                and dep_id in task_map
                and str(task_map[dep_id]).strip()
            )
            node = notebook.add_child_node(
                parent_id=parent_id,
                kind=task_kind,
                title=DynamicOrchestrator._task_title_from_description(description),
                objective=str(description or "").strip(),
                owner="subagent",
                resource_ids=tuple(resource_ids),
                deps=dep_node_ids,
                metadata={
                    "task_id": task_id,
                    "dispatch_deps": list(deps),
                },
            )
            notebook.transition_node(
                node.node_id,
                "running",
                summary="已派发子任务",
                detail=task_id,
                metadata={
                    "task_id": task_id,
                    "resource_ids": list(resource_ids),
                },
            )
            if isinstance(task_map, dict):
                task_map[task_id] = node.node_id
            notebook_node_id = node.node_id
        contract = TaskContract(
            task_id=task_id,
            description=description,
            deps=tuple(deps),
            lease=lease,
            timeout_s=self._coerce_timeout(args.get("timeout_s"))
            or self._default_timeout_s,
            metadata={
                "resource_id": primary_resource_id,
                "resource_ids": list(resource_ids),
                "skill_ids": merged_skill_ids,
                "description": description,
                "notebook_node_id": notebook_node_id,
            },
        )
        child_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{task_id}",
        )
        if notebook_node_id:
            child_context.state["current_notebook_node_id"] = notebook_node_id
        child_context.state["upstream_results"] = context.state.setdefault(
            "upstream_results", {}
        )
        child_context.state["heartbeat"] = self._task_heartbeat_registry.handle(
            flow_id,
            task_id,
            parent=context.state.get("heartbeat"),
        )

        def _notebook_feedback_payload() -> dict[str, Any]:
            if not isinstance(notebook, PlanNotebook):
                return {}
            if not notebook_node_id or notebook_node_id not in notebook.nodes:
                return {}
            node = notebook.get_node(notebook_node_id)
            completed_steps = [
                candidate.title
                for candidate in notebook.nodes.values()
                if candidate.parent_id == node.parent_id and candidate.status == "completed"
            ][-3:]
            blockers = [
                checkpoint.message
                for checkpoint in node.checkpoints
                if checkpoint.status == "open"
            ]
            blockers.extend(
                issue.title for issue in node.issues if issue.status == "open"
            )
            return {
                "notebook_phase": node.status,
                "notebook_owner": node.owner,
                "notebook_completed_steps": completed_steps,
                "notebook_blockers": blockers[:2],
                "notebook_next_action": (
                    node.objective
                    if node.status not in {"completed", "failed", "cancelled"}
                    else ""
                ),
            }

        await self._child_task_bus.publish(
            ChildTaskEvent(
                flow_id=flow_id,
                task_id=task_id,
                event="queued",
                payload=self._event_payload(
                    resource_id=primary_resource_id,
                    resource_ids=list(resource_ids),
                    description=description,
                    deps=list(deps),
                    **_notebook_feedback_payload(),
                ),
            )
        )
        self._update_task_state(
            task_id,
            status="queued",
            resource_id=primary_resource_id,
            resource_ids=list(resource_ids),
            description=description,
            deps=list(deps),
        )

        async def _run_with_deps() -> None:
            progress_monitor = asyncio.create_task(
                self._monitor_task_progress(
                    task_id=task_id,
                    resource_id=primary_resource_id,
                    description=description,
                )
            )
            if deps:
                dep_tasks = [self._in_flight[d] for d in deps if d in self._in_flight]
                if dep_tasks:
                    await asyncio.gather(*dep_tasks, return_exceptions=True)
            try:
                max_attempts = 1 + self._max_retries
                attempt = 0
                while True:
                    attempt += 1
                    await self._child_task_bus.publish(
                        ChildTaskEvent(
                            flow_id=flow_id,
                            task_id=task_id,
                            event="started",
                            payload=self._event_payload(
                                resource_id=primary_resource_id,
                                description=description,
                                **_notebook_feedback_payload(),
                            ),
                        )
                    )
                    self._update_task_state(
                        task_id,
                        status="started",
                        attempts=attempt,
                        max_attempts=max_attempts,
                    )
                    upstream_results = {
                        dep_id: dep_result.output
                        for dep_id in deps
                        if (dep_result := self._results.get(dep_id)) is not None
                        and dep_result.status == "succeeded"
                    }
                    execution_contract = replace(
                        contract,
                        metadata={
                            **contract.metadata,
                            "upstream_results": upstream_results,
                        },
                    )
                    try:
                        async with self._semaphore:
                            if execution_contract.timeout_s is not None:
                                raw_result = await asyncio.wait_for(
                                    self._bridge.execute(
                                        execution_contract, child_context
                                    ),
                                    timeout=execution_contract.timeout_s,
                                )
                            else:
                                raw_result = await self._bridge.execute(
                                    execution_contract,
                                    child_context,
                                )
                        result = self._normalize_result(
                            task_id,
                            raw_result,
                            attempts=attempt,
                            base_metadata=dict(execution_contract.metadata),
                        )
                    except asyncio.TimeoutError:
                        timeout_s = (
                            execution_contract.timeout_s or self._default_timeout_s
                        )
                        result = TaskResult(
                            task_id=task_id,
                            status="failed",
                            error=f"child task timeout after {timeout_s:.2f}s",
                            attempts=attempt,
                            metadata=dict(execution_contract.metadata),
                        )
                    except Exception as exc:
                        result = TaskResult(
                            task_id=task_id,
                            status="failed",
                            error=str(exc),
                            attempts=attempt,
                            metadata=dict(execution_contract.metadata),
                        )

                    if result.status == "succeeded":
                        self._results[task_id] = result
                        self._update_task_state(
                            task_id,
                            status=result.status,
                            output=result.output,
                            error=result.error,
                            attempts=result.attempts,
                            metadata=result.metadata,
                        )
                        await self._child_task_bus.publish(
                            ChildTaskEvent(
                                flow_id=flow_id,
                                task_id=task_id,
                                event="succeeded",
                                payload=self._event_payload(
                                    resource_id=primary_resource_id,
                                    message="子任务已完成",
                                    status=result.status,
                                    error=result.error,
                                    output=result.output,
                                    task_description=description,
                                    **_notebook_feedback_payload(),
                                ),
                            )
                        )
                        child_media = child_context.state.get(
                            "media_paths_collected", []
                        )
                        all_media = list(
                            dict.fromkeys(
                                list(self._result_artifacts(result))
                                + list(child_media or ())
                            )
                        )
                        if all_media:
                            context.state.setdefault(
                                "media_paths_collected", []
                            ).extend(all_media)
                        break

                    decision = classify_error(result.error)
                    if decision.retryable and attempt < max_attempts:
                        self._update_task_state(
                            task_id,
                            status="retrying",
                            output=result.output,
                            error=result.error,
                            attempts=attempt,
                            metadata=result.metadata,
                            max_attempts=max_attempts,
                            last_error=result.error,
                            error_type=decision.error_type,
                        )
                        await self._child_task_bus.publish(
                            ChildTaskEvent(
                                flow_id=flow_id,
                                task_id=task_id,
                                event="retrying",
                                payload=self._event_payload(
                                    resource_id=primary_resource_id,
                                    description=description,
                                    attempt=attempt,
                                    max_attempts=max_attempts,
                                    error=result.error,
                                    error_type=decision.error_type,
                                    **_notebook_feedback_payload(),
                                ),
                            )
                        )
                        delay_s = float(self._retry_delay_seconds(attempt - 1))
                        if delay_s > 0:
                            await asyncio.sleep(delay_s)
                        continue

                    final_result = self._record_dead_letter(
                        task_id,
                        result,
                        error_type=decision.error_type,
                        retryable=decision.retryable,
                        max_attempts=max_attempts,
                    )
                    await self._child_task_bus.publish(
                        ChildTaskEvent(
                            flow_id=flow_id,
                            task_id=task_id,
                            event="dead_lettered",
                            payload=self._event_payload(
                                resource_id=primary_resource_id,
                                message="子任务执行失败",
                                status=final_result.status,
                                error=final_result.error,
                                attempts=final_result.attempts,
                                max_attempts=max_attempts,
                                error_type=decision.error_type,
                                task_description=description,
                                **_notebook_feedback_payload(),
                            ),
                        )
                    )
                    break
            finally:
                progress_monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_monitor
                self._in_flight.pop(task_id, None)

        self._in_flight[task_id] = asyncio.create_task(_run_with_deps())
        return task_id

    async def _monitor_task_progress(
        self,
        *,
        task_id: str,
        resource_id: str,
        description: str,
    ) -> None:
        last_status = ""
        last_progress: float | None = None
        while task_id not in self._results:
            await asyncio.sleep(self._progress_poll_interval_s)
            snapshot = self._task_heartbeat_registry.snapshot(self._flow_id).get(
                task_id
            )
            if not snapshot:
                continue
            status = str(snapshot.get("status", "") or "").strip()
            progress_raw = snapshot.get("progress")
            progress = None
            if isinstance(progress_raw, (int, float)):
                progress = max(0.0, min(1.0, float(progress_raw)))
            rounded_progress = None if progress is None else round(progress, 2)
            if (
                status in {"", "idle", "queued", "started", "retrying"}
                and rounded_progress is None
            ):
                continue
            if status == last_status and rounded_progress == last_progress:
                continue
            last_status = status
            last_progress = rounded_progress
            payload: dict[str, Any] = {
                "resource_id": resource_id,
                "description": description,
            }
            if status:
                payload["status"] = status
            if rounded_progress is not None:
                payload["progress"] = rounded_progress
            await self._child_task_bus.publish(
                ChildTaskEvent(
                    flow_id=self._flow_id,
                    task_id=task_id,
                    event="progress",
                    payload=self._event_payload(**payload),
                )
            )

    async def wait_for_tasks(self, task_ids: list[str]) -> str:
        if self._stale_after_s is None:
            awaitables = []
            for task_id in task_ids:
                if task_id in self._results:
                    continue
                if task_id in self._in_flight:
                    awaitables.append(self._in_flight[task_id])

            if awaitables:
                await asyncio.gather(*awaitables, return_exceptions=True)
        else:
            while True:
                await self._promote_stalled_tasks(task_ids)
                awaitables = [
                    self._in_flight[task_id]
                    for task_id in task_ids
                    if task_id not in self._results and task_id in self._in_flight
                ]
                if not awaitables:
                    break
                done, pending = await asyncio.wait(
                    awaitables,
                    timeout=max(0.01, min(self._stale_after_s, 0.1)),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                del done, pending

        out: dict[str, dict[str, Any]] = {}
        for task_id in task_ids:
            if task_id in self._results:
                result = self._results[task_id]
                out[task_id] = self._result_payload(result)
            elif self._task_state.get(task_id, {}).get("status") == "recoverable":
                out[task_id] = {
                    "status": "recoverable",
                    "output": "",
                    "error": "child task heartbeat stalled",
                    "reply_artifacts_ready": False,
                    "reply_artifacts_count": 0,
                }
            else:
                out[task_id] = {
                    "status": "not_found",
                    "output": "",
                    "error": f"task not found: {task_id}",
                    "reply_artifacts_ready": False,
                    "reply_artifacts_count": 0,
                }
        return json.dumps(out, ensure_ascii=False)

    def get_task_result(self, task_id: str) -> str:
        if task_id in self._in_flight and task_id in self._stale_task_ids([task_id]):
            return json.dumps(
                {
                    "status": "recoverable",
                    "output": "",
                    "error": "child task heartbeat stalled",
                    "reply_artifacts_ready": False,
                    "reply_artifacts_count": 0,
                },
                ensure_ascii=False,
            )
        if task_id in self._results:
            result = self._results[task_id]
            return json.dumps(self._result_payload(result), ensure_ascii=False)
        if task_id in self._in_flight:
            return json.dumps(
                {
                    "status": "pending",
                    "output": "",
                    "error": "",
                    "reply_artifacts_ready": False,
                    "reply_artifacts_count": 0,
                },
                ensure_ascii=False,
            )
        if task_id in self._task_state:
            status = str(self._task_state[task_id].get("status", "") or "")
            if status in {"queued", "started", "retrying"}:
                return json.dumps(
                    {
                        "status": "pending",
                        "output": "",
                        "error": "",
                        "reply_artifacts_ready": False,
                        "reply_artifacts_count": 0,
                    },
                    ensure_ascii=False,
                )
            if status == "recoverable":
                error = str(
                    self._task_state[task_id].get("error", "")
                    or "previous run interrupted before completion"
                )
                return json.dumps(
                    {
                        "status": "recoverable",
                        "output": "",
                        "error": error,
                        "reply_artifacts_ready": False,
                        "reply_artifacts_count": 0,
                    },
                    ensure_ascii=False,
                )
        return json.dumps(
            {
                "status": "not_found",
                "output": "",
                "error": f"task not found: {task_id}",
                "reply_artifacts_ready": False,
                "reply_artifacts_count": 0,
            },
            ensure_ascii=False,
        )

    async def cancel_all(self, *, grace_period_s: float = 0.0) -> None:
        # Prevent new dispatches by clearing the semaphore budget.
        self._cancelling = True
        try:
            tasks = [task for task in self._in_flight.values() if not task.done()]
            if not tasks:
                return
            if grace_period_s > 0:
                done, pending = await asyncio.wait(tasks, timeout=grace_period_s)
                del done
                for task in pending:
                    task.cancel()
            else:
                for task in tasks:
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._cancelling = False


# ── DynamicOrchestrator ──────────────────────────────────────────────────


class DynamicOrchestrator:
    """Dynamic orchestration loop driven by model tool calls."""

    MAX_STEPS = 30
    MAX_TASKS = 20

    def __init__(
        self,
        resource_manager: "ResourceManager",
        gateway: "OpenAICompatibleGateway",
        child_task_bus: InMemoryChildTaskBus | None = None,
        task_heartbeat_registry: "TaskHeartbeatRegistry | None" = None,
        task_stale_after_s: float | None = None,
        max_steps: int | None = None,
        default_task_timeout_s: float | None = 300.0,
        executor_registry: "ExecutorPort | None" = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        from ..heartbeat import TaskHeartbeatRegistry

        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
        self._executor_registry = executor_registry
        self._child_task_bus = child_task_bus or InMemoryChildTaskBus()
        self._task_heartbeat_registry = (
            task_heartbeat_registry or TaskHeartbeatRegistry()
        )
        self._task_stale_after_s = task_stale_after_s
        self._max_steps = max(1, int(max_steps or self.MAX_STEPS))
        self._default_task_timeout_s = default_task_timeout_s
        self._resource_catalog_cache_key: tuple[str, ...] | None = None
        self._resource_catalog_cache_value = ""
        self._config = _resolve_orchestrator_config(config)
        self._orchestration_tools = self._build_orchestration_tools()

    @property
    def _executor(self) -> ExecutorPort:
        """Return the executor registry if provided, otherwise fall back to the bridge."""
        if self._executor_registry is not None:
            return self._executor_registry
        return self._bridge

    def _build_orchestration_tools(self) -> tuple[dict[str, Any], ...]:
        """Merge config-supplied descriptions into the static tool schemas."""
        cfg = self._config
        td = cfg.tool_descriptions
        pd = cfg.tool_param_descriptions

        def _desc(tool_name: str, fallback: str) -> str:
            return td.get(tool_name) or fallback

        def _pdesc(tool_name: str, param: str, fallback: str) -> str:
            return (pd.get(tool_name) or {}).get(param) or fallback

        tools = list(_ORCHESTRATION_TOOLS)
        result: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool["function"]
            name = fn["name"]
            patched_fn: dict[str, Any] = dict(fn)
            patched_fn["description"] = _desc(name, fn.get("description", ""))
            # Patch parameter descriptions if provided
            orig_params: dict[str, Any] = fn.get("parameters", {})
            if pd.get(name):
                patched_props = {}
                for pname, pschema in orig_params.get("properties", {}).items():
                    patched_prop = dict(pschema)
                    override = _pdesc(name, pname, "")
                    if override:
                        patched_prop["description"] = override
                    patched_props[pname] = patched_prop
                patched_params = dict(orig_params)
                patched_params["properties"] = patched_props
                patched_fn["parameters"] = patched_params
            result.append({"type": "function", "function": patched_fn})
        return tuple(result)

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
        context.state.setdefault("original_goal", goal)
        self._ensure_plan_notebook(goal, context)
        context.state.setdefault(
            "original_request_header",
            self._config.original_request_header,
        )
        context.state.setdefault(
            "upstream_results_header",
            self._config.upstream_results_header,
        )
        heartbeat = context.state.get("heartbeat")
        reply_text: str | None = None
        scheduler_handoff_created = False
        flow_id = context.session_id or "orchestrator"
        runtime = InProcessChildTaskRuntime(
            flow_id=flow_id,
            resource_manager=self._rm,
            bridge=self._executor,
            child_task_bus=self._child_task_bus,
            task_heartbeat_registry=self._task_heartbeat_registry,
            max_parallel=4,
            max_tasks=self.MAX_TASKS,
            default_timeout_s=self._default_task_timeout_s,
            stale_after_s=self._task_stale_after_s,
            plan_step_id=(
                context.state.get("execution_plan").steps[0].step_id
                if getattr(context.state.get("execution_plan"), "steps", None)
                else ""
            ),
        )
        runtime_event_callback = context.state.get("runtime_event_callback")
        forwarder_task: asyncio.Task[None] | None = None
        if runtime_event_callback is not None:
            forwarder_task = asyncio.create_task(
                self._forward_runtime_events(flow_id, runtime_event_callback)
            )

        messages = self._build_initial_messages(goal, context)
        logger.info("DynamicOrchestrator: initial messages built, entering main loop")
        try:
            for step in range(self._max_steps):
                if heartbeat is not None:
                    heartbeat.beat()

                messages = self._prune_stale_wait_history(messages)
                messages = self._prune_messages_by_count(messages)
                force_converge_reason = self._refresh_force_converge_state(
                    runtime=runtime,
                    context=context,
                )
                if force_converge_reason:
                    messages.append(
                        ModelMessage(
                            role="system",
                            content=(
                                "Runtime update:\n"
                                f"{force_converge_reason}\n"
                                "Only use get_task_result, wait_for_tasks, or "
                                "reply_to_user until this stage converges."
                            ),
                        )
                    )
                logger.debug("DynamicOrchestrator: step=%d calling model", step)
                response = await self._call_model(messages, context, step=step)
                response = ModelResponse(
                    text=response.text,
                    tool_calls=self._merge_dispatch_calls_for_maintenance(
                        response.tool_calls,
                        goal=str(context.state.get("original_goal", goal) or goal),
                    ),
                    finish_reason=response.finish_reason,
                    metadata=dict(response.metadata),
                )

                # Model responded with plain text (no tool calls)
                if not response.tool_calls:
                    return FinalResult(conclusion=response.text)

                reply_call_count = sum(
                    1
                    for tool_call in response.tool_calls
                    if tool_call.name == "reply_to_user"
                )
                if reply_call_count and len(response.tool_calls) > 1:
                    response = ModelResponse(
                        text=response.text,
                        tool_calls=tuple(
                            tool_call
                            for tool_call in response.tool_calls
                            if tool_call.name != "reply_to_user"
                        )
                        + (
                            ModelToolCall(
                                call_id="reply_guard",
                                name="reply_to_user",
                                arguments={
                                    "text": (
                                        "error: reply_to_user must be the only tool call in its turn"
                                    )
                                },
                            ),
                        ),
                        finish_reason="tool_calls",
                    )

                # Append assistant message with all tool_calls
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )

                # Process each tool call
                scheduler_dispatch_succeeded_this_turn = False
                prior_live_task_ids_this_turn: list[str] = []
                scheduler_dispatch_present_this_turn = any(
                    tc.name == "dispatch_task"
                    and tc.arguments.get("resource_id") == "group.scheduler"
                    for tc in response.tool_calls
                )
                for tc in response.tool_calls:
                    result_text = await self._dispatch_tool(
                        tc,
                        runtime,
                        context,
                        task_counter,
                        scheduler_handoff_created=scheduler_handoff_created,
                        scheduler_dispatch_present_this_turn=scheduler_dispatch_present_this_turn,
                        scheduler_dispatch_seen_before_call=scheduler_dispatch_succeeded_this_turn,
                        prior_live_task_ids_this_turn=tuple(
                            prior_live_task_ids_this_turn
                        ),
                    )
                    if tc.name == "dispatch_task" and not result_text.startswith(
                        "error:"
                    ):
                        task_counter += 1
                        resource_id = tc.arguments.get("resource_id")
                        if resource_id == "group.scheduler":
                            scheduler_dispatch_succeeded_this_turn = True
                        else:
                            prior_live_task_ids_this_turn.append(result_text)
                    messages.append(
                        ModelMessage(
                            role="tool",
                            content=result_text,
                            tool_call_id=tc.call_id,
                        )
                    )
                    if tc.name == "reply_to_user":
                        if result_text.startswith("error:"):
                            continue
                        reply_text = tc.arguments.get("text", result_text)

                if scheduler_dispatch_succeeded_this_turn:
                    scheduler_handoff_created = True

                if reply_text is not None:
                    await runtime.cancel_all()
                    return FinalResult(
                        conclusion=reply_text, task_results=runtime.results
                    )

            await runtime.cancel_all(grace_period_s=5.0)
            return self._build_fallback_result(
                goal,
                runtime.results,
                notebook=context.state.get("plan_notebook"),
            )
        finally:
            if forwarder_task is not None:
                forwarder_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder_task
            self._child_task_bus.clear_flow(flow_id)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_initial_messages(
        self,
        goal: str,
        context: ExecutionContext,
    ) -> list[ModelMessage]:
        briefs = self._rm.get_resource_briefs()
        tape = context.state.get("tape")
        memory_store = context.state.get("memory_store")
        history = build_history_summary(tape, memory_store=memory_store, query=goal)
        media_paths = context.state.get("media_paths") or ()
        execution_constraints = context.state.get("execution_constraints")
        policy_hints = [
            str(item).strip()
            for item in (context.state.get("policy_hints") or ())
            if str(item).strip()
        ]
        policy_hints.extend(_provider_policy_hints(self._rm, goal, self._config))
        deduped_policy_hints: list[str] = []
        for hint in policy_hints:
            if hint and hint not in deduped_policy_hints:
                deduped_policy_hints.append(hint)
        # Cap at 20 hints to prevent prompt bloat from unbounded accumulation.
        _MAX_POLICY_HINTS = 20
        if len(deduped_policy_hints) > _MAX_POLICY_HINTS:
            deduped_policy_hints = deduped_policy_hints[:_MAX_POLICY_HINTS]
        context.state["policy_hints"] = tuple(deduped_policy_hints)

        system_parts = [
            self._config.system_prompt or _SYSTEM_PROMPT_ROLE,
            self._resource_catalog_text(briefs),
        ]
        if _needs_deferred_task_guidance(goal, self._config):
            system_parts.insert(
                1, self._config.deferred_task_guidance or _DEFERRED_TASK_GUIDANCE
            )
        # Dynamic resource selection addendum — injected when the callable
        # is configured and the current resource mix warrants guidance.
        addendum_builder = getattr(
            self._config, "build_resource_selection_addendum", None
        )
        if callable(addendum_builder):
            addendum = addendum_builder(briefs)
            if addendum:
                system_parts.append(addendum)
        notebook = context.state.get("plan_notebook")
        if isinstance(notebook, PlanNotebook):
            notebook_context_budget = int(
                context.state.get("notebook_context_budget", 2400) or 2400
            )
            notebook_view = build_orchestrator_context_view(
                notebook,
                token_budget=notebook_context_budget,
                current_node_id=str(
                    context.state.get("current_notebook_node_id", "") or ""
                ),
            )
            if notebook_view.text:
                system_parts.append("\n[Notebook Context]\n" + notebook_view.text)
        if history:
            system_parts.append(f"\n{history}")
        if deduped_policy_hints:
            header = self._config.policy_hints_header or "\nPolicy hints:\n- "
            system_parts.append(header + "\n- ".join(deduped_policy_hints))

        # execution_constraints is dynamic and request-scoped — inject it as a
        # dedicated context block immediately before the user message so it is
        # clearly separated from the static role/catalog and from conversation
        # history, making it easier for the model to locate and respect.
        user_context_prefix = ""
        if execution_constraints:
            constraints_text = format_execution_constraints_for_prompt(
                execution_constraints
            )
            wrapper = self._config.execution_constraints_wrapper
            if wrapper and "{constraints}" in wrapper and "{goal}" in wrapper:
                user_context_prefix = wrapper.split("{goal}")[0].replace(
                    "{constraints}", constraints_text
                )
            else:
                user_context_prefix = (
                    f"[Constraints]\n{constraints_text}\n\n[Request]\n"
                )

        return [
            ModelMessage(role="system", content="\n".join(system_parts)),
            ModelMessage(
                role="user",
                content=user_context_prefix + goal,
                images=tuple(media_paths),
            ),
        ]

    def _resource_catalog_text(self, briefs: list[dict[str, Any]]) -> str:
        cache_key = tuple(
            json.dumps(brief, ensure_ascii=False, sort_keys=True) for brief in briefs
        )
        if cache_key != self._resource_catalog_cache_key:
            self._resource_catalog_cache_key = cache_key
            self._resource_catalog_cache_value = _build_resource_catalog(
                briefs, self._config
            )
        return self._resource_catalog_cache_value

    def _ensure_plan_notebook(
        self,
        goal: str,
        context: ExecutionContext,
    ) -> PlanNotebook:
        notebook = context.state.get("plan_notebook")
        if isinstance(notebook, PlanNotebook):
            context.state.setdefault("plan_notebook_id", notebook.notebook_id)
            context.state.setdefault(
                "current_notebook_node_id",
                notebook.primary_frontier_node_id(),
            )
            context.state.setdefault("notebook_context_budget", 2400)
            return notebook
        notebook = create_root_notebook(
            goal=goal,
            flow_id=context.session_id,
            plan_id=str(
                getattr(context.state.get("execution_plan"), "plan_id", "") or ""
            ),
            metadata={"original_goal": str(context.state.get("original_goal", goal) or goal)},
        )
        context.state["plan_notebook"] = notebook
        context.state["plan_notebook_id"] = notebook.notebook_id
        context.state["current_notebook_node_id"] = notebook.primary_frontier_node_id()
        context.state.setdefault("notebook_context_budget", 2400)
        return notebook

    @staticmethod
    def _notebook_task_map(context: ExecutionContext) -> dict[str, str]:
        mapping = context.state.get("notebook_task_map")
        if isinstance(mapping, dict):
            return mapping
        context.state["notebook_task_map"] = {}
        return context.state["notebook_task_map"]

    @staticmethod
    def _task_title_from_description(description: str) -> str:
        for line in str(description or "").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
        return "Child task"

    @staticmethod
    def _current_notebook_node(
        notebook: PlanNotebook,
        context: ExecutionContext,
    ) -> Any:
        node_id = str(
            context.state.get("current_notebook_node_id", "")
            or notebook.primary_frontier_node_id()
        )
        if node_id in notebook.nodes:
            return notebook.get_node(node_id)
        return notebook.get_node(notebook.primary_frontier_node_id())

    @staticmethod
    def _child_notebook_nodes(notebook: PlanNotebook, parent_id: str) -> list[Any]:
        return [
            node
            for node in notebook.nodes.values()
            if str(node.parent_id or "") == str(parent_id)
        ]

    def _ensure_team_notebook_node(
        self,
        *,
        notebook: PlanNotebook,
        context: ExecutionContext,
        topic: str,
        mode: str,
        agents: list[dict[str, Any]],
        max_rounds: int,
    ) -> Any:
        current_node = self._current_notebook_node(notebook, context)
        node_kind = "team_cooperative" if mode == "cooperative" else "team_debate"
        node = current_node if current_node.kind == node_kind else None
        if node is None:
            node = notebook.add_child_node(
                parent_id=current_node.node_id,
                kind=node_kind,
                title=self._task_title_from_description(topic) or "Team execution",
                objective=str(topic or "").strip(),
                owner="team",
                metadata={
                    "team_mode": mode,
                    "agents": [
                        {
                            "id": str(agent.get("id", "") or "").strip(),
                            "role": str(agent.get("role", "") or "").strip(),
                        }
                        for agent in agents
                    ],
                    "max_rounds": max_rounds,
                },
            )
        if node.status == "pending":
            notebook.transition_node(
                node.node_id,
                "running",
                summary="已启动团队节点",
                detail=str(topic or ""),
                metadata={"team_mode": mode, "max_rounds": max_rounds},
            )
        return node

    def _complete_converged_execution_nodes(self, notebook: PlanNotebook) -> None:
        for node in notebook.nodes.values():
            if node.kind not in _EXECUTION_NODE_KINDS:
                continue
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            children = [
                child
                for child in self._child_notebook_nodes(notebook, node.node_id)
                if child.kind != "scheduled_task"
            ]
            if children and any(
                child.status not in {"completed", "failed", "cancelled"}
                for child in children
            ):
                continue
            summary = node.latest_summary or "执行节点完成"
            detail = node.result_text or node.objective
            notebook.transition_node(
                node.node_id,
                "completed",
                summary=summary,
                detail=detail,
            )

    @staticmethod
    def _build_team_debate_payload(
        *,
        result: Any,
        agents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        last_arguments = [
            {
                "agent": entry["agent"],
                "role": entry["role"],
                "content": entry["content"][:500],
            }
            for entry in result.transcript[-len(agents) :]
        ]
        recommendation = str(result.summary or "").strip().splitlines()[0].strip()
        disagreements = [
            f"{entry['role']}: {entry['content'][:200]}" for entry in last_arguments
        ]
        return {
            "topic": result.topic,
            "rounds": result.rounds,
            "summary": result.summary,
            "completed": result.completed,
            "termination_reason": result.termination_reason,
            "transcript_length": len(result.transcript),
            "recommendation": recommendation,
            "disagreements": disagreements,
            "last_arguments": last_arguments,
        }

    def _finalize_team_debate_node(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        payload: dict[str, Any],
    ) -> None:
        recommendation = str(payload.get("recommendation", "") or "").strip()
        if recommendation:
            rationale = "\n".join(str(item) for item in payload.get("disagreements", []))
            notebook.add_decision(
                node_id=node.node_id,
                summary=recommendation,
                rationale=rationale,
                metadata={
                    "team_mode": "debate",
                    "rounds": payload.get("rounds"),
                    "termination_reason": payload.get("termination_reason"),
                    "progress": True,
                },
            )
        notebook.record_event(
            node_id=node.node_id,
            kind="summary",
            summary=str(payload.get("summary", "") or "团队讨论完成")[:200],
            detail=json.dumps(payload, ensure_ascii=False),
            metadata={"team_mode": "debate", "progress": True},
        )
        notebook.transition_node(
            node.node_id,
            "completed" if payload.get("completed") else "failed",
            summary=str(payload.get("recommendation") or payload.get("summary") or "团队讨论完成")[:200],
            detail=str(payload.get("summary", "") or ""),
            metadata={
                "team_mode": "debate",
                "rounds": payload.get("rounds"),
                "termination_reason": payload.get("termination_reason"),
            },
        )
        if not payload.get("completed"):
            notebook.mark_needs_repair(
                node.node_id,
                message=str(payload.get("termination_reason", "") or "team debate incomplete"),
            )

    def _initialize_team_cooperative_children(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        tasks: list[dict[str, Any]],
    ) -> dict[str, str]:
        task_nodes: dict[str, str] = {}
        for task in tasks:
            task_id = str(task.get("task_id", "") or "").strip()
            if not task_id:
                continue
            dep_node_ids = tuple(
                task_nodes[dep_id]
                for dep_id in (task.get("deps") or [])
                if dep_id in task_nodes
            )
            task_node = notebook.add_child_node(
                parent_id=node.node_id,
                kind="team_task",
                title=task_id,
                objective=str(task.get("description", "") or task_id),
                owner="team",
                deps=dep_node_ids,
                metadata={"team_task_id": task_id},
            )
            task_nodes[task_id] = task_node.node_id
        return task_nodes

    def _finalize_team_cooperative_node(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        result: Any,
        task_nodes: dict[str, str],
    ) -> dict[str, Any]:
        for task_id, task_payload in dict(result.task_statuses or {}).items():
            node_id = task_nodes.get(task_id)
            if not node_id or node_id not in notebook.nodes:
                continue
            status = str(task_payload.get("status", "") or "pending")
            output = str(task_payload.get("output", "") or "")
            if status == "completed":
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary=f"团队子任务完成: {task_id}",
                    detail=output,
                )
            elif status == "failed":
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary=f"团队子任务失败: {task_id}",
                    detail=output,
                )
                notebook.add_issue(
                    node_id=node_id,
                    title=f"团队子任务失败: {task_id}",
                    detail=output,
                    severity="high",
                )
            else:
                notebook.transition_node(
                    node_id,
                    "cancelled",
                    summary=f"团队子任务未完成: {task_id}",
                    detail=output or status,
                )
        payload = {
            "topic": result.topic,
            "tasks_completed": result.tasks_completed,
            "tasks_failed": result.tasks_failed,
            "tasks_total": result.tasks_total,
            "summary": result.summary,
            "completed": result.completed,
            "termination_reason": result.termination_reason,
            "task_statuses": result.task_statuses,
        }
        notebook.record_event(
            node_id=node.node_id,
            kind="summary",
            summary=str(result.summary or "团队协作完成")[:200],
            detail=json.dumps(payload, ensure_ascii=False),
            metadata={"team_mode": "cooperative", "progress": True},
        )
        notebook.transition_node(
            node.node_id,
            "completed" if result.completed else "failed",
            summary=str(result.summary or "团队协作完成")[:200],
            detail=str(result.summary or ""),
            metadata={
                "team_mode": "cooperative",
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
            },
        )
        if not result.completed:
            notebook.mark_needs_repair(
                node.node_id,
                message=str(result.termination_reason or "team cooperative incomplete"),
            )
        return payload

    def _update_notebook_from_task_payloads(
        self,
        *,
        context: ExecutionContext,
        runtime: InProcessChildTaskRuntime,
        task_ids: list[str],
    ) -> None:
        notebook = self._ensure_plan_notebook(
            str(context.state.get("original_goal", "") or ""),
            context,
        )
        task_map = self._notebook_task_map(context)
        candidate_task_ids = list(dict.fromkeys(list(task_ids) + list(runtime.results.keys())))
        for task_id in candidate_task_ids:
            node_id = task_map.get(task_id)
            if not node_id or node_id not in notebook.nodes:
                continue
            result = runtime.results.get(task_id)
            if result is None:
                continue
            if result.status == "succeeded":
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary="子任务完成",
                    detail=str(result.output or ""),
                )
                for artifact in getattr(result, "artifacts", ()) or ():
                    if str(artifact).strip():
                        notebook.add_artifact(
                            node_id=node_id,
                            path=str(artifact),
                            label="runtime artifact",
                        )
            elif result.status == "failed":
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary="子任务失败",
                    detail=str(result.error or ""),
                )
                node = notebook.get_node(node_id)
                task_error = str(result.error or "")
                issue_exists = any(
                    issue.status == "open"
                    and issue.title == "子任务失败"
                    and issue.detail == task_error
                    for issue in node.issues
                )
                if not issue_exists:
                    notebook.add_issue(
                        node_id=node_id,
                        title="子任务失败",
                        detail=task_error,
                        severity="high",
                    )
                repair_message = task_error or "子任务失败，需要修复"
                repair_checkpoint_exists = any(
                    checkpoint.status == "open"
                    and checkpoint.kind == "needs_repair"
                    and checkpoint.message == repair_message
                    and str(checkpoint.metadata.get("task_id", "") or "") == task_id
                    for checkpoint in node.checkpoints
                )
                if not repair_checkpoint_exists:
                    notebook.mark_needs_repair(
                        node_id,
                        message=repair_message,
                        metadata={"task_id": task_id},
                    )

    def _notebook_blocking_node_ids(
        self,
        notebook: PlanNotebook,
    ) -> list[str]:
        blocking: list[str] = []
        for node_id, node in notebook.nodes.items():
            if node.kind in {"root", "scheduled_task"}:
                continue
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            blocking.append(node_id)
        return blocking

    @staticmethod
    def _notebook_reply_blocking_checkpoints(
        notebook: PlanNotebook,
    ) -> list[str]:
        blocking_kinds = {"needs_human_input", "verification_failed"}
        return [
            f"{checkpoint.node_id}:{checkpoint.kind}"
            for checkpoint in notebook.open_checkpoints()
            if checkpoint.kind in blocking_kinds
        ]

    def _force_converge_state(
        self,
        *,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
    ) -> tuple[str, dict[str, Any]] | None:
        threshold = max(
            1,
            int(
                getattr(
                    self._config,
                    "force_converge_dead_letter_threshold",
                    3,
                )
                or 3
            ),
        )
        notebook = self._ensure_plan_notebook(
            str(context.state.get("original_goal", "") or ""),
            context,
        )
        current_node = self._current_notebook_node(notebook, context)
        task_map = self._notebook_task_map(context)
        dead_ids: list[str] = []
        success_ids: list[str] = []
        pending_ids: list[str] = []

        for task_id, node_id in task_map.items():
            normalized_node_id = str(node_id or "").strip()
            if not normalized_node_id or normalized_node_id not in notebook.nodes:
                continue
            node = notebook.get_node(normalized_node_id)
            if str(node.parent_id or "") != current_node.node_id:
                continue
            if node.kind == "scheduled_task":
                continue

            result = runtime.results.get(task_id)
            if result is not None:
                resource_id = str(result.metadata.get("resource_id", "") or "").strip()
                if resource_id == "group.scheduler":
                    continue
                if result.status == "succeeded":
                    success_ids.append(task_id)
                elif (
                    result.status == "failed"
                    and result.metadata.get("dead_lettered") is True
                ):
                    dead_ids.append(task_id)
                continue

            if task_id in runtime.in_flight:
                state = runtime.task_state_snapshot(task_id)
                resource_id = str(state.get("resource_id", "") or "").strip()
                if resource_id != "group.scheduler":
                    pending_ids.append(task_id)

        if len(dead_ids) < threshold or success_ids or pending_ids:
            return None

        dead_preview = ", ".join(dead_ids[-threshold:])
        reason = (
            "convergence required: repeated child-task failures exhausted the retry "
            f"budget for the current notebook node ({dead_preview}). "
            "Do not dispatch more child tasks. Inspect existing task results and "
            "reply with a concise blocker/failure summary."
        )
        return reason, {
            "node_id": current_node.node_id,
            "dead_task_ids": dead_ids,
            "successful_task_ids": success_ids,
            "pending_task_ids": pending_ids,
        }

    def _refresh_force_converge_state(
        self,
        *,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
    ) -> str | None:
        payload = self._force_converge_state(runtime=runtime, context=context)
        current_reason = str(
            context.state.get("orchestrator_force_converge", "") or ""
        ).strip()

        if payload is None:
            if current_reason:
                logger.info("DynamicOrchestrator: clearing force-converge mode")
            context.state.pop("orchestrator_force_converge", None)
            context.state.pop("orchestrator_force_converge_node_id", None)
            context.state.pop("orchestrator_force_converge_dead_task_ids", None)
            return None

        reason, metadata = payload
        context.state["orchestrator_force_converge"] = reason
        context.state["orchestrator_force_converge_node_id"] = metadata["node_id"]
        context.state["orchestrator_force_converge_dead_task_ids"] = list(
            metadata["dead_task_ids"]
        )
        if current_reason == reason:
            return None

        logger.warning(
            "DynamicOrchestrator: force-converge activated node=%s dead_tasks=%s",
            metadata["node_id"],
            metadata["dead_task_ids"],
        )
        notebook = context.state.get("plan_notebook")
        if isinstance(notebook, PlanNotebook) and metadata["node_id"] in notebook.nodes:
            notebook.record_event(
                node_id=metadata["node_id"],
                kind="decision",
                summary="进入强制收敛模式",
                detail=reason,
                metadata={
                    "force_converge": True,
                    "dead_task_ids": list(metadata["dead_task_ids"]),
                },
            )
        return reason

    def _finalize_notebook_for_reply(
        self,
        *,
        context: ExecutionContext,
        reply_text: str,
    ) -> None:
        notebook = self._ensure_plan_notebook(
            str(context.state.get("original_goal", "") or reply_text),
            context,
        )
        self._complete_converged_execution_nodes(notebook)
        if not notebook.completion_summary:
            notebook.set_completion_summary(
                {
                    "final_summary": str(reply_text or "").strip(),
                    "decision_register": [
                        decision.summary
                        for node in notebook.nodes.values()
                        for decision in node.decisions[-3:]
                    ],
                    "artifact_manifest": [
                        artifact.path
                        for node in notebook.nodes.values()
                        for artifact in node.artifacts
                    ],
                }
            )
        for node_id, node in notebook.nodes.items():
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            if node.kind == "root":
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary="编排收尾完成",
                    detail=str(reply_text or ""),
                )

    def _merge_dispatch_calls_for_maintenance(
        self,
        tool_calls: tuple[ModelToolCall, ...],
        *,
        goal: str,
    ) -> tuple[ModelToolCall, ...]:
        if (
            len(tool_calls) < 2
            or not _is_maintenance_goal(goal, self._config)
            or _goal_has_explicit_parallel_intent(goal, self._config)
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

    def _maintenance_serial_dependency_ids(
        self,
        runtime: InProcessChildTaskRuntime,
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

        recent_success = self._recent_successful_upstream_results(
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

    def _normalize_child_task_description(
        self,
        *,
        description: str,
        resource_ids: tuple[str, ...],
        context: ExecutionContext,
        upstream_results: dict[str, "TaskResult"] | None = None,
    ) -> str:
        raw_description = str(description or "").strip()
        if not raw_description:
            return raw_description

        # If a sentinel is configured and already present, skip normalisation.
        sentinel = self._config.child_task_sentinel
        if sentinel and sentinel in raw_description:
            return raw_description

        # Delegate to injected builder if provided.
        builder = self._config.build_child_task_prompt
        if builder is None:
            return raw_description

        original_goal = str(context.state.get("original_goal", "") or "").strip()

        # Build a plain upstream dict: tid -> output str (successful only)
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

    @staticmethod
    def _recent_successful_upstream_results(
        results: dict[str, "TaskResult"],
        *,
        limit: int = 1,
    ) -> dict[str, "TaskResult"] | None:
        if not results or limit <= 0:
            return None
        selected: list[tuple[str, "TaskResult"]] = []
        for task_id, result in reversed(list(results.items())):
            if result.status != "succeeded":
                continue
            if str(result.output or "").strip() == "":
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

    @staticmethod
    def _prune_stale_wait_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        wait_call_ids: list[str] = []
        for message in messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if tool_call.name == "wait_for_tasks" and tool_call.call_id:
                    wait_call_ids.append(tool_call.call_id)

        if len(wait_call_ids) <= 1:
            return messages

        keep_wait_call_id = wait_call_ids[-1]
        stale_wait_call_ids = set(wait_call_ids[:-1])
        pruned: list[ModelMessage] = []

        for message in messages:
            if message.role == "tool" and message.tool_call_id in stale_wait_call_ids:
                continue
            if message.role == "assistant" and message.tool_calls:
                filtered_tool_calls = tuple(
                    tool_call
                    for tool_call in message.tool_calls
                    if tool_call.call_id not in stale_wait_call_ids
                )
                if filtered_tool_calls != message.tool_calls:
                    if not filtered_tool_calls and not message.content.strip():
                        continue
                    pruned.append(
                        ModelMessage(
                            role=message.role,
                            content=message.content,
                            name=message.name,
                            tool_call_id=message.tool_call_id,
                            tool_calls=filtered_tool_calls,
                            images=message.images,
                        )
                    )
                    continue
            pruned.append(message)

        tool_ids_present = {
            message.tool_call_id
            for message in pruned
            if message.role == "tool" and message.tool_call_id
        }
        if keep_wait_call_id not in tool_ids_present:
            return messages
        return pruned

    @staticmethod
    def _prune_messages_by_count(
        messages: list[ModelMessage],
        *,
        max_non_system: int = 80,
        keep_recent: int = 60,
    ) -> list[ModelMessage]:
        """Fallback count-based pruning to prevent unbounded message growth.

        When the total number of non-system messages exceeds ``max_non_system``,
        only the system message (index 0) plus the most recent ``keep_recent``
        non-system messages are retained.  This is a safety net — it does NOT
        do token-based compaction, but it bounds worst-case prompt size in long
        task chains.
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= max_non_system:
            return messages

        logger.warning(
            "DynamicOrchestrator: message count exceeded %d (actual=%d); "
            "pruning to %d most recent non-system messages",
            max_non_system,
            len(non_system),
            keep_recent,
        )
        return system_msgs + non_system[-keep_recent:]

    async def _call_model(
        self,
        messages: list[ModelMessage],
        context: ExecutionContext,
        *,
        step: int = 0,
    ) -> ModelResponse:
        heartbeat = context.state.get("heartbeat")
        stream_callback = context.state.get("stream_callback")
        request = ModelRequest(
            messages=tuple(messages),
            tools=self._tool_schemas_for_context(context),
        )
        state = {
            k: v
            for k, v in [
                ("heartbeat", heartbeat),
                ("stream_callback", stream_callback),
            ]
            if v is not None
        }
        ctx = ExecutionContext(state=state)
        logger.info("DynamicOrchestrator: step=%d LLM call start", step)
        if heartbeat is not None:
            async with heartbeat.keep_alive():
                response = await self._gateway.generate(request, ctx)
        else:
            response = await self._gateway.generate(request, ctx)
        logger.info(
            "DynamicOrchestrator: step=%d LLM call done finish_reason=%s tool_calls=%d",
            step,
            response.finish_reason,
            len(response.tool_calls),
        )
        return response

    def _tool_schemas_for_context(
        self,
        context: ExecutionContext,
    ) -> tuple[dict[str, Any], ...]:
        execution_plan = context.state.get("execution_plan")
        allowed = ()
        if getattr(execution_plan, "steps", None):
            payload = getattr(execution_plan.steps[0], "payload", {}) or {}
            allowed = tuple(payload.get("allowed_tools") or ())
        if not allowed:
            task_contract = context.state.get("task_contract")
            allowed = tuple(getattr(task_contract, "allowed_tools", ()) or ())
        if not allowed:
            selected = self._orchestration_tools
        else:
            tool_by_name = {t["function"]["name"]: t for t in self._orchestration_tools}
            filtered = tuple(tool_by_name[name] for name in allowed if name in tool_by_name)
            selected = filtered or self._orchestration_tools
        if str(context.state.get("orchestrator_force_converge", "") or "").strip():
            convergence_tools = tuple(
                tool
                for tool in selected
                if tool["function"]["name"] in _FORCE_CONVERGE_TOOL_NAMES
            )
            if convergence_tools:
                return convergence_tools
            return tuple(
                tool
                for tool in self._orchestration_tools
                if tool["function"]["name"] in _FORCE_CONVERGE_TOOL_NAMES
            )
        return selected

    async def _dispatch_tool(
        self,
        tool_call: ModelToolCall,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
        task_counter: int,
        *,
        scheduler_handoff_created: bool,
        scheduler_dispatch_present_this_turn: bool,
        scheduler_dispatch_seen_before_call: bool,
        prior_live_task_ids_this_turn: tuple[str, ...],
    ) -> str:
        name = tool_call.name
        args = tool_call.arguments

        if args.get("__tool_argument_parse_error__"):
            return f"error: invalid arguments: {args.get('__raw_arguments__', '')}"

        force_converge_reason = str(
            context.state.get("orchestrator_force_converge", "") or ""
        ).strip()
        if force_converge_reason and name in {"dispatch_task", "dispatch_team"}:
            return (
                "error: convergence required: "
                f"{force_converge_reason} "
                "Do not dispatch new child tasks; use get_task_result, "
                "wait_for_tasks, or reply_to_user."
            )

        if name == "dispatch_task":
            dispatch_args = dict(args)
            resource_ids = _dispatch_resource_ids(dispatch_args)
            deps = list(dispatch_args.get("deps", []) or [])
            active_in_flight = len(runtime.in_flight)
            scheduler_dispatch = "group.scheduler" in resource_ids
            maintenance_single_owner = (
                not scheduler_dispatch
                and _is_maintenance_goal(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
                and not _goal_has_explicit_parallel_intent(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
            )
            if maintenance_single_owner and not deps:
                serialized_deps = self._maintenance_serial_dependency_ids(
                    runtime,
                    prior_live_task_ids_this_turn=prior_live_task_ids_this_turn,
                )
                if serialized_deps:
                    dispatch_args["deps"] = serialized_deps
                    deps = list(serialized_deps)
            if (
                not deps
                and not scheduler_dispatch
                and prior_live_task_ids_this_turn
                and not _goal_has_explicit_parallel_intent(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
            ):
                current_resources = {rid for rid in resource_ids if rid}
                for prior_task_id in prior_live_task_ids_this_turn:
                    prior_state = runtime.task_state_snapshot(prior_task_id)
                    prior_resources = {
                        str(item).strip()
                        for item in (prior_state.get("resource_ids") or [])
                        if str(item).strip()
                    }
                    if not current_resources or not prior_resources:
                        continue
                    if (
                        current_resources & prior_resources
                        and (
                            len(current_resources) > 1
                            or len(prior_resources) > 1
                        )
                    ):
                        return (
                            "error: overlapping sibling dispatches in the same turn "
                            "must be merged into one dispatch_task with resource_ids, "
                            "or the later task must declare deps for serial execution"
                        )
            if "group.scheduler" not in resource_ids:
                # Collect completed upstream results for dep tasks so the child task
                # knows what its dependencies produced (break the black-box problem).
                upstream_results = {
                    tid: runtime.results[tid] for tid in deps if tid in runtime.results
                } or None
                if (
                    upstream_results is None
                    and not deps
                    and not prior_live_task_ids_this_turn
                    and active_in_flight == 0
                ):
                    upstream_results = self._recent_successful_upstream_results(
                        runtime.results
                    )
                dispatch_args["description"] = self._normalize_child_task_description(
                    description=str(dispatch_args.get("description", "") or ""),
                    resource_ids=resource_ids,
                    context=context,
                    upstream_results=upstream_results,
                )
            scheduling_action = (
                "parallel_dispatch"
                if not deps and (prior_live_task_ids_this_turn or active_in_flight > 0)
                else "serial_dispatch"
            )
            _emit_policy_decision(
                context,
                decision_kind="scheduling",
                action_name=scheduling_action,
                state_features={
                    "deps_count": len(deps),
                    "active_in_flight": active_in_flight,
                    "resource_count": len(resource_ids),
                },
            )
            if (
                scheduler_dispatch
                and not dispatch_args.get("deps")
                and prior_live_task_ids_this_turn
            ):
                dispatch_args["deps"] = list(prior_live_task_ids_this_turn)
            if scheduler_handoff_created and not scheduler_dispatch:
                return (
                    "error: scheduled handoff already created in this flow; "
                    "do not dispatch additional live tasks after scheduling future work"
                )
            if scheduler_dispatch_seen_before_call and not scheduler_dispatch:
                return (
                    "error: scheduled handoff was already created earlier in this turn; "
                    "do not dispatch additional live tasks after it"
                )
            if (
                runtime.results
                and scheduler_dispatch_present_this_turn
                and not scheduler_dispatch
            ):
                return (
                    "error: scheduled handoff is being created for a later stage; "
                    "do not mix new live tasks into the same turn"
                )
            runtime_dispatch_args = dict(dispatch_args)
            runtime_dispatch_args["__task_kind"] = (
                "scheduled_task" if scheduler_dispatch else "child_task"
            )
            task_id = await runtime.dispatch(
                runtime_dispatch_args, task_counter=task_counter, context=context
            )
            return task_id
        if name == "wait_for_tasks":
            _emit_policy_decision(
                context,
                decision_kind="scheduling",
                action_name="wait_barrier",
                state_features={
                    "task_count": len(args.get("task_ids", []) or []),
                },
            )
            task_ids = list(args.get("task_ids", []) or [])
            payload = await runtime.wait_for_tasks(task_ids)
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=task_ids,
            )
            return payload
        if name == "get_task_result":
            task_id = args.get("task_id", "")
            payload = runtime.get_task_result(task_id)
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=[task_id],
            )
            return payload
        if name == "reply_to_user":
            pending = runtime.pending_reply_blocking_task_ids()
            if pending:
                return (
                    "error: reply_to_user called before child tasks finished; "
                    f"pending tasks: {', '.join(pending)}"
                )
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=list(runtime.results.keys()),
            )
            notebook = self._ensure_plan_notebook(
                str(context.state.get("original_goal", "") or ""),
                context,
            )
            self._complete_converged_execution_nodes(notebook)
            all_dead, dead_ids = runtime.all_tasks_dead_lettered_with_no_success()
            blocking_nodes = self._notebook_blocking_node_ids(notebook)
            if blocking_nodes:
                return (
                    "error: reply_to_user called before notebook frontier converged; "
                    f"blocking notebook nodes: {', '.join(blocking_nodes)}"
                )
            open_checkpoints = self._notebook_reply_blocking_checkpoints(notebook)
            if open_checkpoints:
                return (
                    "error: reply_to_user called before notebook checkpoints were resolved; "
                    f"open notebook checkpoints: {', '.join(open_checkpoints)}"
                )
            if all_dead:
                errors = "; ".join(
                    str(runtime.results[tid].error or "unknown error")[:120]
                    for tid in dead_ids
                )
                logger.warning(
                    "reply_to_user after all tasks dead-lettered (%s); "
                    "allowing direct failure reply. errors: %s",
                    ", ".join(dead_ids),
                    errors,
                )
            self._finalize_notebook_for_reply(
                context=context,
                reply_text=str(args.get("text", "") or ""),
            )
            return args.get("text", "")
        if name == "dispatch_team":
            return await self._run_team(args, context)
        return f"error: unknown tool: {name}"

    async def _run_team(self, args: dict[str, Any], context: ExecutionContext) -> str:
        import functools
        from .team import TeamRunner

        topic = args.get("topic", "")
        agents = args.get("agents", [])
        mode = str(args.get("mode", "debate") or "debate").strip().lower()
        task_contract = context.state.get("task_contract")
        execution_constraints = normalize_execution_constraints(
            context.state.get("execution_constraints")
        )
        effective_args = dict(args)
        if getattr(task_contract, "round_budget", None) is not None:
            effective_args["max_rounds"] = int(task_contract.round_budget)
        team_policy = build_team_execution_policy(effective_args, execution_constraints)
        if task_contract is not None:
            assert_runtime_matches_contract(
                task_contract,
                max_rounds=team_policy.max_rounds,
            )
        max_rounds = team_policy.max_rounds

        if len(agents) < 2:
            return "error: dispatch_team requires at least 2 agents"

        if mode == "cooperative":
            tasks = args.get("tasks") or []
            if not tasks:
                return "error: cooperative mode requires a non-empty 'tasks' list"
        else:
            tasks = []

        notebook = self._ensure_plan_notebook(
            str(context.state.get("original_goal", "") or topic),
            context,
        )

        logger.info(
            "Team dispatch: topic=%r agents=%d max_rounds=%d mode=%s",
            topic[:80],
            len(agents),
            max_rounds,
            mode,
        )

        heartbeat = context.state.get("heartbeat")
        send_intermediate = context.state.get("send_intermediate_message")
        stream_callback = context.state.get("stream_callback")
        reset_stream = (
            getattr(stream_callback, "reset", None) if stream_callback else None
        )
        team_streaming = stream_callback is not None and reset_stream is not None
        runtime_event_callback = context.state.get("runtime_event_callback")
        stage_name = "cooperative" if mode == "cooperative" else "debate"
        plan_step_id = ""
        execution_plan = context.state.get("execution_plan")
        if getattr(execution_plan, "steps", None):
            plan_step_id = str(execution_plan.steps[0].step_id or "").strip()

        async def _emit_team_event(
            event_name: str,
            *,
            state: str,
            message: str,
            progress: float | None = None,
            error: str = "",
        ) -> None:
            text = str(message or "").strip()
            if (
                event_name == "progress"
                and team_streaming
                and send_intermediate is not None
                and text
            ):
                await send_intermediate(text)
            if runtime_event_callback is None:
                return
            payload: dict[str, Any] = {
                "state": state,
                "stage": stage_name,
                "message": text,
            }
            if plan_step_id:
                payload["plan_step_id"] = plan_step_id
            if progress is not None:
                payload["progress"] = progress
            if error:
                payload["error"] = error
            maybe = runtime_event_callback(
                {
                    "event": event_name,
                    "flow_id": context.session_id,
                    "task_id": stage_name,
                    "payload": payload,
                }
            )
            if inspect.isawaitable(maybe):
                await maybe

        async def gateway_executor(
            agent_id: str, prompt: str, ctx: dict[str, Any]
        ) -> str:
            """Fallback executor: calls gateway directly (no tools)."""
            sys_prompt = ctx.get(
                "system_prompt",
                self._config.team_default_agent_system_prompt,
            )
            messages = [
                ModelMessage(role="system", content=sys_prompt),
                ModelMessage(role="user", content=prompt),
            ]
            request = ModelRequest(messages=tuple(messages))
            gen_ctx = ExecutionContext(
                state={"stream_callback": stream_callback} if team_streaming else {},
            )
            if heartbeat is not None:
                async with heartbeat.keep_alive():
                    response = await self._gateway.generate(request, gen_ctx)
            else:
                response = await self._gateway.generate(request, gen_ctx)
            return response.text

        # Resolve skill_id references and prepare per-agent executors
        enriched_agents: list[dict[str, Any]] = []
        skills = getattr(self._rm, "skills", {})
        for agent in agents:
            agent_copy = dict(agent)
            skill_id = agent_copy.pop("skill_id", None) or agent_copy.pop(
                "profile_id", None
            )
            if skill_id:
                skill = skills.get(skill_id) or skills.get(skill_id.strip().lower())
                if skill is not None:
                    if not agent_copy.get("role") and skill.role:
                        agent_copy["role"] = skill.role
                    if not agent_copy.get("description"):
                        agent_copy["description"] = skill.description
                    if not agent_copy.get("system_prompt") and skill.prompt:
                        agent_copy["system_prompt"] = skill.prompt
            enriched_agents.append(agent_copy)

        if (
            team_policy.max_agents is not None
            and len(enriched_agents) > team_policy.max_agents
        ):
            enriched_agents = enriched_agents[: team_policy.max_agents]
        if len(enriched_agents) < 2:
            return "error: dispatch_team requires at least 2 agents after applying constraints"

        for ea in enriched_agents:
            logger.info(
                "Team agent resolved: id=%s role=%s skill=%s resource=%s",
                ea.get("id"),
                ea.get("role", ""),
                ea.get("skill_id", ea.get("profile_id", "")),
                ea.get("resource_id", ""),
            )

        team_node = self._ensure_team_notebook_node(
            notebook=notebook,
            context=context,
            topic=topic,
            mode=mode,
            agents=enriched_agents,
            max_rounds=max_rounds,
        )
        team_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{team_node.node_id}",
        )
        team_context.state["current_notebook_node_id"] = team_node.node_id

        async def resource_executor(
            resource_id: str, agent_id: str, prompt: str, ctx: dict[str, Any]
        ) -> str:
            """Resource-backed executor: runs through bridge with full tool access."""
            task = TaskContract(
                task_id=f"team_{agent_id}",
                description=prompt,
                metadata={"resource_id": resource_id, "team_node_id": team_node.node_id},
            )
            result = await self._executor.execute(task, team_context)
            if result.status != "succeeded":
                return f"[error: {result.error}]"
            return result.output

        for agent_copy in enriched_agents:
            rid = agent_copy.get("resource_id")
            if not rid:
                continue
            scope = self._rm.resolve_resource_scope(rid, require_tools=True)
            if scope is not None:
                agent_copy["executor"] = functools.partial(resource_executor, rid)

        # ── Cooperative mode: task-based parallel execution ──────────────
        if mode == "cooperative":
            return await self._run_team_cooperative(
                args=args,
                topic=topic,
                enriched_agents=enriched_agents,
                runner_executor=gateway_executor,
                team_policy=team_policy,
                max_rounds=max_rounds,
                heartbeat=heartbeat,
                send_intermediate=send_intermediate,
                team_streaming=team_streaming,
                reset_stream=reset_stream,
                _emit_team_event=_emit_team_event,
                context=context,
                notebook=notebook,
                team_node=team_node,
            )

        # ── Debate mode (default) ───────────────────────────────────────

        async def on_turn(agent_id: str, role: str, round_num: int, text: str) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            notebook.record_event(
                node_id=team_node.node_id,
                kind="observation",
                summary=f"Round {round_num}: {role}",
                detail=str(text or ""),
                metadata={
                    "agent_id": agent_id,
                    "role": role,
                    "round": round_num,
                },
            )
            # Always send an intermediate message for each turn so the user
            # sees individual agent outputs regardless of streaming mode.
            if send_intermediate is not None:
                header = f"**[{role} — Round {round_num}]**"
                await send_intermediate(header + "\n" + text)
            if team_streaming:
                await reset_stream()

        logger.info(
            "Team streaming=%s intermediate=%s heartbeat=%s",
            team_streaming,
            send_intermediate is not None,
            heartbeat is not None,
        )

        await _emit_team_event(
            "progress",
            state="planning",
            message=self._config.team_debate_started_message.format(
                n_agents=len(enriched_agents), max_rounds=max_rounds
            ),
            progress=0.0,
        )

        async def on_round_start(round_num: int, total_rounds: int) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            await _emit_team_event(
                "progress",
                state="running",
                message=self._config.team_debate_round_message.format(
                    round_num=round_num, total_rounds=total_rounds
                ),
                progress=round_num / max(1, total_rounds + 1),
            )

        if team_streaming:
            await reset_stream()

        runner = TeamRunner(
            executor=gateway_executor,
            max_rounds=max_rounds,
            policy=team_policy,
        )
        result = await runner.run_debate(
            topic=topic,
            agents=enriched_agents,
            on_turn=on_turn,
            on_round_start=on_round_start,
        )

        logger.info(
            "Team debate finished: topic=%r rounds=%d transcript_len=%d",
            result.topic[:80],
            result.rounds,
            len(result.transcript),
        )

        if team_streaming:
            await reset_stream()

        await _emit_team_event(
            "completed" if result.completed else "failed",
            state="completed" if result.completed else "repairing",
            message=result.summary[:200] or self._config.team_debate_ended_message,
            progress=1.0 if result.completed else None,
        )
        payload = self._build_team_debate_payload(
            result=result,
            agents=enriched_agents,
        )
        self._finalize_team_debate_node(
            notebook=notebook,
            node=team_node,
            payload=payload,
        )
        self._complete_converged_execution_nodes(notebook)
        return json.dumps(payload, ensure_ascii=False)

    async def _run_team_cooperative(
        self,
        *,
        args: dict[str, Any],
        topic: str,
        enriched_agents: list[dict[str, Any]],
        runner_executor: Any,
        team_policy: Any,
        max_rounds: int,
        heartbeat: Any,
        send_intermediate: Any,
        team_streaming: bool,
        reset_stream: Any,
        _emit_team_event: Any,
        context: ExecutionContext,
        notebook: PlanNotebook,
        team_node: Any,
    ) -> str:
        """Run team in cooperative (task-based) mode."""
        from .team import TeamRunner

        tasks = args.get("tasks") or []

        logger.info(
            "Team cooperative: topic=%r agents=%d tasks=%d",
            topic[:80],
            len(enriched_agents),
            len(tasks),
        )

        await _emit_team_event(
            "progress",
            state="planning",
            message=self._config.team_coop_started_message.format(
                n_agents=len(enriched_agents), n_tasks=len(tasks)
            ),
            progress=0.0,
        )
        task_nodes = self._initialize_team_cooperative_children(
            notebook=notebook,
            node=team_node,
            tasks=tasks,
        )

        completed_count = {"n": 0}

        async def on_task_complete(agent_id: str, task_id: str, output: str) -> None:
            completed_count["n"] += 1
            if heartbeat is not None:
                heartbeat.beat()
            node_id = task_nodes.get(task_id, "")
            if node_id and node_id in notebook.nodes:
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary=f"团队子任务完成: {task_id}",
                    detail=str(output or ""),
                )
            progress = completed_count["n"] / max(1, len(tasks))
            await _emit_team_event(
                "progress",
                state="running",
                message=self._config.team_coop_task_done_message.format(
                    task_id=task_id,
                    done=completed_count["n"],
                    total=len(tasks),
                ),
                progress=progress,
            )

        async def on_task_failed(agent_id: str, task_id: str, error: str) -> None:
            del agent_id
            node_id = task_nodes.get(task_id, "")
            if node_id and node_id in notebook.nodes:
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary=f"团队子任务失败: {task_id}",
                    detail=str(error or ""),
                )
                notebook.add_issue(
                    node_id=node_id,
                    title=f"团队子任务失败: {task_id}",
                    detail=str(error or ""),
                    severity="high",
                )

        if team_streaming and reset_stream is not None:
            await reset_stream()

        runner = TeamRunner(
            executor=runner_executor,
            max_rounds=max_rounds,
            policy=team_policy,
        )
        result = await runner.run_cooperative(
            topic=topic,
            agents=enriched_agents,
            tasks=tasks,
            on_task_complete=on_task_complete,
            on_task_failed=on_task_failed,
        )

        logger.info(
            "Team cooperative finished: topic=%r completed=%d/%d failed=%d",
            result.topic[:80],
            result.tasks_completed,
            result.tasks_total,
            result.tasks_failed,
        )

        if team_streaming and reset_stream is not None:
            await reset_stream()

        await _emit_team_event(
            "completed" if result.completed else "failed",
            state="completed" if result.completed else "repairing",
            message=result.summary[:200] or self._config.team_coop_ended_message,
            progress=1.0 if result.completed else None,
        )
        payload = self._finalize_team_cooperative_node(
            notebook=notebook,
            node=team_node,
            result=result,
            task_nodes=task_nodes,
        )
        payload["mode"] = "cooperative"
        payload["task_outputs"] = {
            tid: output[:500] for tid, output in result.task_outputs.items()
        }
        payload["mailbox_messages"] = len(result.mailbox_log)
        self._complete_converged_execution_nodes(notebook)
        return json.dumps(payload, ensure_ascii=False)

    # DAG task events that carry meaningful user-facing progress.
    # "started" is included so the progress card updates when a subtask
    # begins executing, giving the user visible activity during long tasks.
    _FORWARD_EVENT_ALLOWLIST: frozenset[str] = frozenset(
        {
            "queued",
            "started",
            "succeeded",
            "failed",
            "dead_lettered",
            "stalled",
            "cancelled",
        }
    )

    async def _forward_runtime_events(
        self,
        flow_id: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        async for event in self._child_task_bus.subscribe(flow_id):
            if event.event not in self._FORWARD_EVENT_ALLOWLIST:
                continue
            result = callback(dataclasses.asdict(event))
            if inspect.isawaitable(result):
                await result

    def _build_fallback_result(
        self,
        goal: str,
        results: dict[str, TaskResult],
        *,
        notebook: PlanNotebook | None = None,
    ) -> FinalResult:
        del goal
        parts = [self._config.step_budget_exhausted_header]
        if notebook is not None:
            completion_view = build_completion_context_view(notebook, token_budget=1200)
            if completion_view.text:
                parts.append(completion_view.text)
        for task_id, r in results.items():
            if r.status == "succeeded":
                parts.append(
                    self._config.step_budget_succeeded_line.format(
                        task_id=task_id, output=r.output or "done"
                    )
                )
            else:
                parts.append(
                    self._config.step_budget_failed_line.format(
                        task_id=task_id, error=r.error or ""
                    )
                )
        return FinalResult(conclusion="\n".join(parts), task_results=results)
