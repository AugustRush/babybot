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
from .types import ExecutionContext, FinalResult, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..heartbeat import TaskHeartbeatRegistry
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager
    from .protocols import ExecutorPort

logger = logging.getLogger(__name__)


# ── Orchestration tool schemas (OpenAI function-calling format) ──────────

_ORCHESTRATION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "dispatch_task",
            "description": (
                "创建一个子Agent任务并立即返回 task_id（非阻塞）。"
                "子Agent将使用指定资源执行任务。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "单个资源ID，必须来自可用资源列表",
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "当一个子任务需要多种能力时，传入多个资源ID并合并使用",
                    },
                    "description": {
                        "type": "string",
                        "description": "子任务的完整描述",
                    },
                    "deps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "依赖的 task_id 列表，这些任务必须先完成",
                        "default": [],
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "子任务超时时间（秒）。未提供时使用运行时默认超时。",
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
                "等待指定任务完成并返回 JSON 结果映射（阻塞直到全部完成）。"
                "每个任务结果都包含 status/output/error，以及 reply_artifacts_ready 等字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要等待的 task_id 列表",
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
                "查询任务当前状态和结果（非阻塞，返回 JSON 对象）。"
                "结果包含 status/output/error，以及 reply_artifacts_ready 等字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "要查询的 task_id",
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
                "向用户发送最终回复。调用后编排循环结束。"
                "此工具应作为最后一个工具调用单独使用，不与其他工具混用。"
                "宿主会自动附带当前已收集的产物/附件到最终回复，无需再创建专门的发送子任务。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "回复给用户的文本内容",
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
                "启动一组Agent进行协作。支持两种模式：\n"
                "- debate（默认）：多轮辩论/评审/头脑风暴，Agent交替发言。\n"
                "- cooperative：任务分工协作，Agent从共享任务列表中领取任务并行执行，"
                "通过Mailbox广播结果给下游依赖。适用于可拆分为多个子任务的复杂工作。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "协作主题/高层目标描述",
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
                                    "description": "可选：指定该Agent使用的资源ID",
                                },
                                "skill_id": {
                                    "type": "string",
                                    "description": "可选：引用预定义的 skill name，自动继承其 role/description/prompt",
                                },
                            },
                            "required": ["id", "role", "description"],
                        },
                        "description": "参与协作的Agent列表，至少2个",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["debate", "cooperative"],
                        "description": (
                            "协作模式。debate=多轮辩论（默认），"
                            "cooperative=任务分工（需配合 tasks 参数）"
                        ),
                        "default": "debate",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "debate模式下的最大讨论轮数，默认5",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "任务唯一标识",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "任务描述",
                                },
                                "deps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "依赖的 task_id 列表",
                                    "default": [],
                                },
                            },
                            "required": ["task_id", "description"],
                        },
                        "description": (
                            "cooperative模式下的任务列表。每个任务可声明依赖(deps)，"
                            "Agent会自动领取可执行的任务并广播结果。"
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


# ── System prompt builder ────────────────────────────────────────────────

_SYSTEM_PROMPT_ROLE = (
    "你是任务编排Agent，负责在最少步骤内调度子Agent完成任务并回复用户。\n\n"
    "核心规则：\n"
    "1. 简单问题直接 reply_to_user；需要外部能力时才 dispatch_task。\n"
    "2. 并行任务不要设 deps；有依赖的任务必须显式声明 deps。\n"
    "3. reply_to_user 必须单独调用且作为收尾；不能与其他工具混用。\n"
    "4. 禁止虚构结果；用户明确表达的执行限制与偏好必须优先遵守。\n"
    "5. 多资源任务优先在一次 dispatch_task 中用 resource_ids 组合能力；查看网页/仓库后创建或更新技能时，不要靠 create_worker 套娃补能力。\n"
    "6. 需要多Agent协作讨论、辩论、评审或头脑风暴时，使用 dispatch_team（debate模式）。\n"
    "7. 需要多Agent并行分工执行可拆分的复杂任务时，使用 dispatch_team（cooperative模式，需提供tasks列表）。\n"
    "\n\n执行阶段协议：\n"
    "按以下四阶段有序推进，不允许跨阶段跳跃或在错误阶段执行动作：\n"
    "  [Research]     — 信息收集：仅读取、搜索、获取外部数据，不做修改或写入。\n"
    "  [Synthesis]    — 分析综合：基于 Research 阶段结果制定具体方案，不执行任何写入。\n"
    "  [Implementation] — 执行：按 Synthesis 方案执行写入、调用、修改等动作。\n"
    "  [Verification] — 验证：检查执行结果，与目标对比，发现问题则局部补救，完成后 reply_to_user。\n"
    "- 简单任务（直接问答、无外部调用）可省略 Research/Synthesis/Verification，直接 reply_to_user。\n"
    "- 每阶段的子任务描述中必须标注所在阶段，例如：[Research] 搜索相关文档。\n"
    "\n\n任务结果协议：\n"
    "- wait_for_tasks / get_task_result 返回 JSON，不是自由文本。\n"
    "- 当结果中的 reply_artifacts_ready=true 时，表示子任务已经产出可随最终回复自动附带给用户的附件/媒体。\n"
    "- 出现 reply_artifacts_ready=true 后，不要再创建专门的发送子任务；直接调用 reply_to_user 收尾。"
)

_DEFERRED_TASK_PATTERNS = (
    "两分钟后",
    "一分钟后",
    "稍后",
    "待会",
    "过会",
    "定时",
    "预约",
    "提醒我",
    "之后再",
)

_DEFERRED_TASK_GUIDANCE = (
    "\n\n延时/未来任务规则：\n"
    "7. 如果用户要求稍后、几分钟后、定时或未来某个时间执行动作，当前只应创建/更新定时任务，不要立刻执行未来动作。\n"
    "8. 未来一次性任务的描述必须自包含，写入定时任务时要包含届时需要完成的完整步骤，不能依赖当前这次对话还保存在上下文中。"
)


def _build_resource_catalog(briefs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in briefs:
        if b.get("active"):
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
            preview_text = f"; 示例工具: {preview}" if preview else ""
            lines.append(f"- {rid}: {name} — {purpose} (工具数: {tc}{preview_text})")
    if not lines:
        return "\n可用资源：无"
    return "\n可用资源：\n" + "\n".join(lines)


def _needs_deferred_task_guidance(goal: str) -> bool:
    lowered = (goal or "").strip()
    return any(pattern in lowered for pattern in _DEFERRED_TASK_PATTERNS)


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


def _looks_like_repo_or_skill_maintenance(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    markers = (
        "查漏补缺",
        "补齐",
        "对比",
        "比较",
        "同步",
        "skill.md",
        "skill 文档",
        "github",
        "仓库",
        "repo",
        "文档",
        "技能",
    )
    return any(marker in normalized for marker in markers)


def _provider_policy_hints(resource_manager: "ResourceManager", goal: str) -> list[str]:
    provider = getattr(resource_manager, "_observability_provider", None)
    if provider is None:
        return []
    text = str(goal or "").strip()
    build_features = getattr(provider, "_build_policy_state_features", None)
    features: dict[str, Any] = {
        "task_shape": "multi_step"
        if any(token in text for token in ("然后", "再", "并且", "同时", "先"))
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
    for token in ("同时", "分别", "并行", "并且"):
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
            },
        )
        child_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{task_id}",
        )
        child_context.state["upstream_results"] = context.state.setdefault(
            "upstream_results", {}
        )
        child_context.state["heartbeat"] = self._task_heartbeat_registry.handle(
            flow_id,
            task_id,
            parent=context.state.get("heartbeat"),
        )

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
                                    description=description,
                                    status=result.status,
                                    output=result.output,
                                    error=result.error,
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
                                description=description,
                                status=final_result.status,
                                error=final_result.error,
                                attempts=final_result.attempts,
                                max_attempts=max_attempts,
                                error_type=decision.error_type,
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

    @property
    def _executor(self) -> ExecutorPort:
        """Return the executor registry if provided, otherwise fall back to the bridge."""
        if self._executor_registry is not None:
            return self._executor_registry
        return self._bridge

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
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
                logger.debug("DynamicOrchestrator: step=%d calling model", step)
                response = await self._call_model(messages, context, step=step)

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
            return self._build_fallback_result(goal, runtime.results)
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
        policy_hints.extend(_provider_policy_hints(self._rm, goal))
        deduped_policy_hints: list[str] = []
        for hint in policy_hints:
            if hint and hint not in deduped_policy_hints:
                deduped_policy_hints.append(hint)
        # Cap at 20 hints to prevent prompt bloat from unbounded accumulation.
        _MAX_POLICY_HINTS = 20
        if len(deduped_policy_hints) > _MAX_POLICY_HINTS:
            deduped_policy_hints = deduped_policy_hints[:_MAX_POLICY_HINTS]
        context.state["policy_hints"] = tuple(deduped_policy_hints)

        system_parts = [_SYSTEM_PROMPT_ROLE, self._resource_catalog_text(briefs)]
        if _needs_deferred_task_guidance(goal):
            system_parts.insert(1, _DEFERRED_TASK_GUIDANCE)
        if history:
            system_parts.append(f"\n{history}")
        if deduped_policy_hints:
            system_parts.append("\n策略建议：\n- " + "\n- ".join(deduped_policy_hints))

        # execution_constraints is dynamic and request-scoped — inject it as a
        # dedicated context block immediately before the user message so it is
        # clearly separated from the static role/catalog and from conversation
        # history, making it easier for the model to locate and respect.
        user_context_prefix = ""
        if execution_constraints:
            constraints_text = format_execution_constraints_for_prompt(
                execution_constraints
            )
            user_context_prefix = f"[执行约束]\n{constraints_text}\n\n[用户请求]\n"

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
            self._resource_catalog_cache_value = _build_resource_catalog(briefs)
        return self._resource_catalog_cache_value

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
        if "[执行型子任务]" in raw_description:
            return raw_description

        original_goal = str(context.state.get("original_goal", "") or "").strip()
        resource_summary = ", ".join(resource_ids) if resource_ids else "-"
        maintenance_like = _looks_like_repo_or_skill_maintenance(
            f"{raw_description}\n{original_goal}"
        )

        # Build upstream results section — only include successful outputs from dep tasks.
        upstream_lines: list[str] = []
        if upstream_results:
            for tid, result in upstream_results.items():
                output_snippet = str(result.output or "").strip()
                if output_snippet:
                    # Truncate to 300 chars to avoid prompt bloat.
                    truncated = output_snippet[:300] + (
                        "…" if len(output_snippet) > 300 else ""
                    )
                    upstream_lines.append(f"  [{tid}]: {truncated}")
        upstream_section: list[str] = []
        if upstream_lines:
            upstream_section = [
                "- upstream_results (上游依赖任务的输出摘要):"
            ] + upstream_lines

        expected_output_lines = [
            "- 返回当前子任务的执行结果摘要。",
            "- 如无法继续，明确说明缺少的输入、目标路径或阻塞原因。",
        ]
        done_when_lines = [
            "- 已完成当前子任务说明中的单一目标，或已明确指出无法继续的原因。",
            "- 不再需要额外派生任务或向用户追问即可交回主 agent。",
        ]
        if maintenance_like:
            expected_output_lines.insert(
                1, "- 若存在差异，列出差异项、证据和建议动作。"
            )
            done_when_lines.insert(
                1, "- 已确认明确的目标文件；若无法确认，立即停止并返回缺口。"
            )

        lines = [
            "[执行型子任务]",
            "你是执行型子任务，不是任务编排器，也不是最终回复器。",
            f"原始子任务：{raw_description}",
            "[输入]",
            f"- parent_goal: {original_goal or '-'}",
            f"- resource_ids: {resource_summary}",
            *upstream_section,
            "[预期输出]",
            *expected_output_lines,
            "[完成条件]",
            *done_when_lines,
            "[禁止事项]",
            "- 不要创建或派生新的 worker。",
            "- 不要进入 team、讨论、评审或新的编排流程。",
            "- 不要直接向用户发送消息。",
            "- 不要在无明确目标时持续进行大范围扫描或无边界探索。",
        ]
        return "\n".join(lines)

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

    @staticmethod
    def _tool_schemas_for_context(
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
            return _ORCHESTRATION_TOOLS
        filtered = tuple(
            _ORCHESTRATION_TOOL_BY_NAME[name]
            for name in allowed
            if name in _ORCHESTRATION_TOOL_BY_NAME
        )
        return filtered or _ORCHESTRATION_TOOLS

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

        if name == "dispatch_task":
            dispatch_args = dict(args)
            resource_ids = _dispatch_resource_ids(dispatch_args)
            deps = list(dispatch_args.get("deps", []) or [])
            if "group.scheduler" not in resource_ids:
                # Collect completed upstream results for dep tasks so the child task
                # knows what its dependencies produced (break the black-box problem).
                upstream_results = {
                    tid: runtime.results[tid] for tid in deps if tid in runtime.results
                } or None
                dispatch_args["description"] = self._normalize_child_task_description(
                    description=str(dispatch_args.get("description", "") or ""),
                    resource_ids=resource_ids,
                    context=context,
                    upstream_results=upstream_results,
                )
            active_in_flight = len(runtime.in_flight)
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
            scheduler_dispatch = "group.scheduler" in resource_ids
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
            return await runtime.dispatch(
                dispatch_args, task_counter=task_counter, context=context
            )
        if name == "wait_for_tasks":
            _emit_policy_decision(
                context,
                decision_kind="scheduling",
                action_name="wait_barrier",
                state_features={
                    "task_count": len(args.get("task_ids", []) or []),
                },
            )
            return await runtime.wait_for_tasks(args.get("task_ids", []))
        if name == "get_task_result":
            return runtime.get_task_result(args.get("task_id", ""))
        if name == "reply_to_user":
            pending = runtime.pending_reply_blocking_task_ids()
            if pending:
                return (
                    "error: reply_to_user called before child tasks finished; "
                    f"pending tasks: {', '.join(pending)}"
                )
            all_dead, dead_ids = runtime.all_tasks_dead_lettered_with_no_success()
            if all_dead:
                # All child tasks failed — inject an explicit honesty reminder into the
                # tool result so the model cannot silently fabricate success.
                errors = "; ".join(
                    str(runtime.results[tid].error or "unknown error")[:120]
                    for tid in dead_ids
                )
                logger.warning(
                    "reply_to_user after all tasks dead-lettered (%s); "
                    "injecting honesty reminder. errors: %s",
                    ", ".join(dead_ids),
                    errors,
                )
                dead_reminder = (
                    "[SYSTEM NOTICE] 所有子任务均已失败，没有任何成功的执行结果。\n"
                    f"失败任务: {', '.join(dead_ids)}\n"
                    f"错误摘要: {errors}\n"
                    "你必须如实向用户报告任务失败及原因，禁止虚构或推断未执行的结果。"
                )
                user_text = args.get("text", "")
                return f"{dead_reminder}\n\n你当前提供的回复文本:\n{user_text}"
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
            if runtime_event_callback is None:
                return
            payload: dict[str, Any] = {
                "state": state,
                "stage": "debate",
                "message": message,
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
                    "task_id": "team_debate",
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
                "你是讨论参与者。根据你的角色，针对主题发表观点。",
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

        async def resource_executor(
            resource_id: str, agent_id: str, prompt: str, ctx: dict[str, Any]
        ) -> str:
            """Resource-backed executor: runs through bridge with full tool access."""
            task = TaskContract(
                task_id=f"team_{agent_id}",
                description=prompt,
                metadata={"resource_id": resource_id},
            )
            result = await self._executor.execute(task, context)
            if result.status != "succeeded":
                return f"[error: {result.error}]"
            return result.output

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
            rid = agent_copy.get("resource_id")
            if rid:
                scope = self._rm.resolve_resource_scope(rid, require_tools=True)
                if scope is not None:
                    agent_copy["executor"] = functools.partial(resource_executor, rid)
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
            )

        # ── Debate mode (default) ───────────────────────────────────────

        async def on_turn(agent_id: str, role: str, round_num: int, text: str) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            if team_streaming:
                await reset_stream()
            elif send_intermediate is not None:
                header = f"**[{role} — Round {round_num}]**"
                await send_intermediate(header + "\n" + text)

        logger.info(
            "Team streaming=%s intermediate=%s heartbeat=%s",
            team_streaming,
            send_intermediate is not None,
            heartbeat is not None,
        )

        await _emit_team_event(
            "progress",
            state="planning",
            message=f"已启动 {len(enriched_agents)} 位专家讨论，最多 {max_rounds} 轮。",
            progress=0.0,
        )

        async def on_round_start(round_num: int, total_rounds: int) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            await _emit_team_event(
                "progress",
                state="running",
                message=f"第 {round_num}/{total_rounds} 轮讨论进行中。",
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
            message=result.summary[:200] or "讨论结束",
            progress=1.0 if result.completed else None,
        )

        return json.dumps(
            {
                "topic": result.topic,
                "rounds": result.rounds,
                "summary": result.summary,
                "completed": result.completed,
                "termination_reason": result.termination_reason,
                "transcript_length": len(result.transcript),
                "last_arguments": [
                    {
                        "agent": e["agent"],
                        "role": e["role"],
                        "content": e["content"][:500],
                    }
                    for e in result.transcript[-len(agents) :]
                ],
            },
            ensure_ascii=False,
        )

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
            message=(
                f"已启动 {len(enriched_agents)} 位专家协作执行 {len(tasks)} 个任务。"
            ),
            progress=0.0,
        )

        completed_count = {"n": 0}

        async def on_task_complete(agent_id: str, task_id: str, output: str) -> None:
            completed_count["n"] += 1
            if heartbeat is not None:
                heartbeat.beat()
            progress = completed_count["n"] / max(1, len(tasks))
            await _emit_team_event(
                "progress",
                state="running",
                message=f"任务 {task_id} 已完成 ({completed_count['n']}/{len(tasks)})",
                progress=progress,
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
            message=result.summary[:200] or "协作执行结束",
            progress=1.0 if result.completed else None,
        )

        return json.dumps(
            {
                "mode": "cooperative",
                "topic": result.topic,
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
                "tasks_total": result.tasks_total,
                "summary": result.summary,
                "completed": result.completed,
                "termination_reason": result.termination_reason,
                "task_outputs": {
                    tid: output[:500] for tid, output in result.task_outputs.items()
                },
                "mailbox_messages": len(result.mailbox_log),
            },
            ensure_ascii=False,
        )

    # DAG task events that carry meaningful user-facing progress.
    # queued/started are internal lifecycle noise; only terminal states matter.
    _FORWARD_EVENT_ALLOWLIST: frozenset[str] = frozenset(
        {"succeeded", "failed", "dead_lettered", "stalled", "cancelled"}
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

    @staticmethod
    def _build_fallback_result(
        goal: str,
        results: dict[str, TaskResult],
    ) -> FinalResult:
        del goal
        parts = ["（编排步数已达上限，以下为已完成的任务结果）"]
        for task_id, r in results.items():
            if r.status == "succeeded":
                parts.append(f"- {task_id}: {r.output or '完成'}")
            else:
                parts.append(f"- {task_id}: 失败 — {r.error}")
        return FinalResult(conclusion="\n".join(parts), task_results=results)
