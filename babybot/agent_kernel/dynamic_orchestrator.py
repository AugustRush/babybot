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

from .context import ContextManager
from .dag_ports import ResourceBridgeExecutor, build_history_summary
from .errors import classify_error, retry_delay_seconds as default_retry_delay_seconds
from .model import ModelMessage, ModelRequest, ModelResponse, ModelToolCall
from .types import ExecutionContext, FinalResult, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..heartbeat import TaskHeartbeatRegistry
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager

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
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_tasks",
            "description": "等待指定任务完成并返回结果（阻塞直到全部完成）。",
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
            "description": "查询任务当前状态和结果（非阻塞）。",
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
)


# ── System prompt builder ────────────────────────────────────────────────

_SYSTEM_PROMPT_ROLE = (
    "你是任务编排Agent。理解用户请求，动态调度子Agent完成任务，最终向用户回复结果。\n\n"
    "编排规则：\n"
    "1. 简单问题（聊天、知识问答）→ 直接调用 reply_to_user，无需创建子任务\n"
    "2. 需要工具的任务 → dispatch_task 创建子Agent，wait_for_tasks 等待结果，reply_to_user 回复\n"
    "3. 可并行的任务 → 同时 dispatch 多个（不设deps），再 wait_for_tasks 全部等待\n"
    "4. 有依赖的任务 → 在 deps 中声明依赖，任务内部自动等待前置任务完成\n"
    "5. 拿到结果后 → 调用 reply_to_user 汇总并回复用户，reply_to_user 必须单独调用且为最后一步\n"
    "6. 禁止虚构执行结果；需要外部信息必须通过 dispatch_task 获取\n"
    "7. 如果单个子任务同时需要多种能力，可在一次 dispatch_task 中使用 resource_ids 组合多个资源。\n"
    "8. 对于需要查看网页/仓库说明再创建或更新技能的任务，优先在同一个子任务里组合相关技能、browser、必要的 code 资源；不要靠 create_worker 套娃补能力。"
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
            str(item).strip()
            for item in raw_multi
            if str(item).strip()
        )
    single = str(args.get("resource_id", "") or "").strip()
    if single:
        resource_ids.append(single)
    return tuple(dict.fromkeys(resource_ids))


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

    async def publish(self, event: ChildTaskEvent) -> None:
        self._events.setdefault(event.flow_id, []).append(event)
        for queue in list(self._subscribers.get(event.flow_id, ())):
            queue.put_nowait(event)

    def events_for(self, flow_id: str) -> list[ChildTaskEvent]:
        return list(self._events.get(flow_id, ()))

    async def subscribe(self, flow_id: str) -> AsyncIterator[ChildTaskEvent]:
        queue: asyncio.Queue[ChildTaskEvent] = asyncio.Queue()
        self._subscribers.setdefault(flow_id, []).append(queue)
        for event in self._events.get(flow_id, ()):
            queue.put_nowait(event)
        try:
            while True:
                yield await queue.get()
        finally:
            subscribers = self._subscribers.get(flow_id, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                self._subscribers.pop(flow_id, None)


class InProcessChildTaskRuntime:
    """Current child-task runtime backed by local asyncio tasks."""

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
        stale_after_s: float | None = None,
    ) -> None:
        self._flow_id = flow_id
        self._rm = resource_manager
        self._bridge = bridge
        self._child_task_bus = child_task_bus
        self._task_heartbeat_registry = task_heartbeat_registry
        self._max_tasks = max_tasks
        self._max_retries = max(0, int(max_retries))
        self._retry_delay_seconds = retry_delay_seconds
        self._stale_after_s = stale_after_s
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._in_flight: dict[str, asyncio.Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._task_state: dict[str, dict[str, Any]] = {}
        self._dead_letters: dict[str, dict[str, Any]] = {}

    @property
    def in_flight(self) -> dict[str, asyncio.Task]:
        return self._in_flight

    @property
    def results(self) -> dict[str, TaskResult]:
        return self._results

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
            if task_id in self._in_flight and task_id in stale and task_id not in self._results
        }

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
                    payload={"error": "child task heartbeat stalled"},
                )
            )

    @staticmethod
    def _normalize_result(task_id: str, result: TaskResult, *, attempts: int) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status=result.status,
            output=result.output,
            error=result.error,
            attempts=attempts,
            metadata=dict(result.metadata),
        )

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
        metadata.update({
            "dead_lettered": True,
            "error_type": error_type,
            "retryable": retryable,
            "max_attempts": max_attempts,
        })
        final_result = TaskResult(
            task_id=task_id,
            status="failed",
            output=result.output,
            error=result.error,
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
            metadata={
                "resource_id": primary_resource_id,
                "resource_ids": list(resource_ids),
                "skill_ids": merged_skill_ids,
            },
        )
        child_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{task_id}",
        )
        child_context.state["upstream_results"] = context.state.setdefault("upstream_results", {})
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
                payload={
                    "resource_id": primary_resource_id,
                    "resource_ids": list(resource_ids),
                    "description": description,
                    "deps": list(deps),
                },
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
                            payload={
                                "resource_id": resource_id,
                                "description": description,
                            },
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
                            raw_result = await self._bridge.execute(execution_contract, child_context)
                        result = self._normalize_result(task_id, raw_result, attempts=attempt)
                    except Exception as exc:
                        result = TaskResult(
                            task_id=task_id,
                            status="failed",
                            error=str(exc),
                            attempts=attempt,
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
                                payload={
                                    "resource_id": resource_id,
                                    "description": description,
                                    "status": result.status,
                                    "output": result.output,
                                    "error": result.error,
                                },
                            )
                        )
                        child_media = child_context.state.get("media_paths_collected", [])
                        if child_media:
                            context.state.setdefault("media_paths_collected", []).extend(child_media)
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
                                payload={
                                    "resource_id": resource_id,
                                    "description": description,
                                    "attempt": attempt,
                                    "max_attempts": max_attempts,
                                    "error": result.error,
                                    "error_type": decision.error_type,
                                },
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
                            payload={
                                "resource_id": resource_id,
                                "description": description,
                                "status": final_result.status,
                                "error": final_result.error,
                                "attempts": final_result.attempts,
                                "max_attempts": max_attempts,
                                "error_type": decision.error_type,
                            },
                        )
                    )
                    break
            finally:
                self._in_flight.pop(task_id, None)

        self._in_flight[task_id] = asyncio.create_task(_run_with_deps())
        return task_id

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

        out: dict[str, str] = {}
        for task_id in task_ids:
            if task_id in self._results:
                result = self._results[task_id]
                if result.status == "succeeded":
                    out[task_id] = f"succeeded: {result.output}"
                else:
                    out[task_id] = f"failed: {result.error}"
            elif self._task_state.get(task_id, {}).get("status") == "recoverable":
                out[task_id] = "recoverable: child task heartbeat stalled"
            else:
                out[task_id] = f"not_found: {task_id}"
        return json.dumps(out, ensure_ascii=False)

    def get_task_result(self, task_id: str) -> str:
        if task_id in self._in_flight and task_id in self._stale_task_ids([task_id]):
            return "recoverable: child task heartbeat stalled"
        if task_id in self._results:
            result = self._results[task_id]
            if result.status == "succeeded":
                return f"succeeded: {result.output}"
            return f"failed: {result.error}"
        if task_id in self._in_flight:
            return "pending"
        if task_id in self._task_state:
            status = str(self._task_state[task_id].get("status", "") or "")
            if status in {"queued", "started", "retrying"}:
                return "pending"
            if status == "recoverable":
                error = str(
                    self._task_state[task_id].get("error", "") or "previous run interrupted before completion"
                )
                return f"recoverable: {error}"
        return f"not_found: {task_id}"

    def cancel_all(self) -> None:
        for task in self._in_flight.values():
            task.cancel()


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
    ) -> None:
        from ..heartbeat import TaskHeartbeatRegistry

        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
        self._child_task_bus = child_task_bus or InMemoryChildTaskBus()
        self._task_heartbeat_registry = task_heartbeat_registry or TaskHeartbeatRegistry()
        self._task_stale_after_s = task_stale_after_s
        self._max_steps = max(1, int(max_steps or self.MAX_STEPS))

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
        heartbeat = context.state.get("heartbeat")
        reply_text: str | None = None
        scheduler_handoff_created = False
        flow_id = context.session_id or "orchestrator"
        runtime = InProcessChildTaskRuntime(
            flow_id=flow_id,
            resource_manager=self._rm,
            bridge=self._bridge,
            child_task_bus=self._child_task_bus,
            task_heartbeat_registry=self._task_heartbeat_registry,
            max_parallel=4,
            max_tasks=self.MAX_TASKS,
            stale_after_s=self._task_stale_after_s,
        )
        runtime_event_callback = context.state.get("runtime_event_callback")
        forwarder_task: asyncio.Task[None] | None = None
        if runtime_event_callback is not None:
            forwarder_task = asyncio.create_task(
                self._forward_runtime_events(flow_id, runtime_event_callback)
            )

        messages = self._build_initial_messages(goal, context)
        try:
            for step in range(self._max_steps):
                if heartbeat is not None:
                    heartbeat.beat()

                response = await self._call_model(messages, context)

                # Model responded with plain text (no tool calls)
                if not response.tool_calls:
                    return FinalResult(conclusion=response.text)

                # Append assistant message with all tool_calls
                messages.append(ModelMessage(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                ))

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
                        prior_live_task_ids_this_turn=tuple(prior_live_task_ids_this_turn),
                    )
                    if tc.name == "dispatch_task" and not result_text.startswith("error:"):
                        task_counter += 1
                        resource_id = tc.arguments.get("resource_id")
                        if resource_id == "group.scheduler":
                            scheduler_dispatch_succeeded_this_turn = True
                        else:
                            prior_live_task_ids_this_turn.append(result_text)
                    messages.append(ModelMessage(
                        role="tool",
                        content=result_text,
                        tool_call_id=tc.call_id,
                    ))
                    if tc.name == "reply_to_user":
                        reply_text = tc.arguments.get("text", result_text)

                if scheduler_dispatch_succeeded_this_turn:
                    scheduler_handoff_created = True

                if reply_text is not None:
                    runtime.cancel_all()
                    return FinalResult(conclusion=reply_text, task_results=runtime.results)

            return self._build_fallback_result(goal, runtime.results, runtime.in_flight)
        finally:
            if forwarder_task is not None:
                forwarder_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder_task

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_initial_messages(
        self, goal: str, context: ExecutionContext,
    ) -> list[ModelMessage]:
        briefs = self._rm.get_resource_briefs()
        tape = context.state.get("tape")
        memory_store = context.state.get("memory_store")
        history = build_history_summary(tape, memory_store=memory_store, query=goal)
        media_paths = context.state.get("media_paths") or ()

        system_parts = [_SYSTEM_PROMPT_ROLE, _build_resource_catalog(briefs)]
        if _needs_deferred_task_guidance(goal):
            system_parts.insert(1, _DEFERRED_TASK_GUIDANCE)
        if history:
            system_parts.append(f"\n{history}")

        return [
            ModelMessage(role="system", content="\n".join(system_parts)),
            ModelMessage(
                role="user",
                content=goal,
                images=tuple(media_paths),
            ),
        ]

    async def _call_model(
        self,
        messages: list[ModelMessage],
        context: ExecutionContext,
    ) -> ModelResponse:
        heartbeat = context.state.get("heartbeat")
        stream_callback = context.state.get("stream_callback")
        request = ModelRequest(
            messages=tuple(messages),
            tools=_ORCHESTRATION_TOOLS,
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
        if heartbeat is not None:
            async with heartbeat.keep_alive():
                return await self._gateway.generate(request, ctx)
        return await self._gateway.generate(request, ctx)

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
            return await runtime.dispatch(dispatch_args, task_counter=task_counter, context=context)
        if name == "wait_for_tasks":
            return await runtime.wait_for_tasks(args.get("task_ids", []))
        if name == "get_task_result":
            return runtime.get_task_result(args.get("task_id", ""))
        if name == "reply_to_user":
            return args.get("text", "")
        return f"error: unknown tool: {name}"

    async def _forward_runtime_events(
        self,
        flow_id: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        async for event in self._child_task_bus.subscribe(flow_id):
            result = callback(dataclasses.asdict(event))
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _build_fallback_result(
        goal: str,
        results: dict[str, TaskResult],
        in_flight: dict[str, asyncio.Task],
    ) -> FinalResult:
        for task in in_flight.values():
            task.cancel()
        parts = ["（编排步数已达上限，以下为已完成的任务结果）"]
        for task_id, r in results.items():
            if r.status == "succeeded":
                parts.append(f"- {task_id}: {r.output or '完成'}")
            else:
                parts.append(f"- {task_id}: 失败 — {r.error}")
        return FinalResult(conclusion="\n".join(parts), task_results=results)
