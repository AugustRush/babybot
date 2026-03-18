"""Dynamic orchestration loop — main agent dispatches sub-agents via tool calls."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
                        "description": "资源ID，必须来自可用资源列表",
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
                "required": ["resource_id", "description"],
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
    "6. 禁止虚构执行结果；需要外部信息必须通过 dispatch_task 获取"
)


def _build_resource_catalog(briefs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in briefs:
        if b.get("active"):
            rid = b.get("id", "?")
            name = b.get("name", "?")
            purpose = b.get("purpose", "")
            tc = b.get("tool_count", 0)
            preview = ", ".join(b.get("tools_preview") or [])
            preview_text = f"; 示例工具: {preview}" if preview else ""
            lines.append(f"- {rid}: {name} — {purpose} (工具数: {tc}{preview_text})")
    if not lines:
        return "\n可用资源：无"
    return "\n可用资源：\n" + "\n".join(lines)


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

    async def publish(self, event: ChildTaskEvent) -> None:
        self._events.setdefault(event.flow_id, []).append(event)

    def events_for(self, flow_id: str) -> list[ChildTaskEvent]:
        return list(self._events.get(flow_id, ()))


class FileChildTaskStateStore:
    """JSON snapshot store for child-task runtime state."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(flow_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in flow_id)
        return cleaned or "orchestrator"

    def _path_for(self, flow_id: str) -> Path:
        return self._base_dir / f"{self._slug(flow_id)}.json"

    def save_snapshot(self, flow_id: str, payload: dict[str, Any]) -> None:
        self._path_for(flow_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_snapshot(self, flow_id: str) -> dict[str, Any] | None:
        path = self._path_for(flow_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


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
        state_store: FileChildTaskStateStore | None = None,
        max_retries: int = 0,
        retry_delay_seconds: Callable[[int], float] = default_retry_delay_seconds,
    ) -> None:
        self._flow_id = flow_id
        self._rm = resource_manager
        self._bridge = bridge
        self._child_task_bus = child_task_bus
        self._task_heartbeat_registry = task_heartbeat_registry
        self._max_tasks = max_tasks
        self._state_store = state_store
        self._max_retries = max(0, int(max_retries))
        self._retry_delay_seconds = retry_delay_seconds
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._in_flight: dict[str, asyncio.Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._task_state: dict[str, dict[str, Any]] = {}
        self._dead_letters: dict[str, dict[str, Any]] = {}
        self._restore_snapshot()

    @property
    def in_flight(self) -> dict[str, asyncio.Task]:
        return self._in_flight

    @property
    def results(self) -> dict[str, TaskResult]:
        return self._results

    def _restore_snapshot(self) -> None:
        if self._state_store is None:
            return
        payload = self._state_store.load_snapshot(self._flow_id) or {}
        dead_letters = payload.get("dead_letters", {})
        if isinstance(dead_letters, dict):
            for task_id, task_payload in dead_letters.items():
                if isinstance(task_payload, dict):
                    self._dead_letters[task_id] = dict(task_payload)
        tasks = payload.get("tasks", {})
        if not isinstance(tasks, dict):
            return
        mutated = False
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            restored_payload = dict(task_payload)
            status = task_payload.get("status", "")
            if status in {"succeeded", "failed"}:
                self._task_state[task_id] = restored_payload
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=status,
                    output=str(task_payload.get("output", "") or ""),
                    error=str(task_payload.get("error", "") or ""),
                    attempts=int(task_payload.get("attempts", 1) or 1),
                    metadata=dict(task_payload.get("metadata") or {}),
                )
                continue
            if status:
                restored_payload["status"] = "recoverable"
                restored_payload["error"] = (
                    str(restored_payload.get("error", "") or "")
                    or "previous run interrupted before completion"
                )
                self._task_state[task_id] = restored_payload
                mutated = True
            else:
                self._task_state[task_id] = restored_payload
        if mutated:
            self._persist_snapshot()

    def _snapshot_payload(self) -> dict[str, Any]:
        tasks: dict[str, dict[str, Any]] = {}
        for task_id, payload in self._task_state.items():
            tasks[task_id] = dict(payload)
        for task_id, result in self._results.items():
            existing = tasks.get(task_id, {})
            existing.update({
                "status": result.status,
                "output": result.output,
                "error": result.error,
                "attempts": result.attempts,
                "metadata": result.metadata,
            })
            tasks[task_id] = existing
        return {
            "flow_id": self._flow_id,
            "tasks": tasks,
            "dead_letters": {
                task_id: dict(payload) for task_id, payload in self._dead_letters.items()
            },
        }

    def _persist_snapshot(self) -> None:
        if self._state_store is None:
            return
        self._state_store.save_snapshot(self._flow_id, self._snapshot_payload())

    def _update_task_state(self, task_id: str, **payload: Any) -> None:
        current = self._task_state.setdefault(task_id, {})
        current.update(payload)
        self._persist_snapshot()

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

        resource_id: str = args.get("resource_id", "")
        description: str = args.get("description", "")
        deps: list[str] = args.get("deps", [])

        scope = self._rm.resolve_resource_scope(resource_id, require_tools=True)
        if scope is None:
            return f"error: resource not available: {resource_id}"

        for dep in deps:
            if dep not in self._in_flight and dep not in self._results:
                return f"error: unknown dep task_id: {dep}"

        task_id = f"task_{task_counter}_{uuid.uuid4().hex[:6]}"
        flow_id = self._flow_id
        lease_dict, skill_ids = scope
        lease = ToolLease(
            include_groups=tuple(lease_dict.get("include_groups", ())),
            include_tools=tuple(lease_dict.get("include_tools", ())),
            exclude_tools=tuple(lease_dict.get("exclude_tools", ())),
        )
        contract = TaskContract(
            task_id=task_id,
            description=description,
            deps=tuple(deps),
            lease=lease,
            metadata={"resource_id": resource_id, "skill_ids": list(skill_ids)},
        )
        child_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{task_id}",
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
                payload={"resource_id": resource_id, "deps": list(deps)},
            )
        )
        self._update_task_state(
            task_id,
            status="queued",
            resource_id=resource_id,
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
                        ChildTaskEvent(flow_id=flow_id, task_id=task_id, event="started")
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
                                payload={"status": result.status, "error": result.error},
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
        awaitables = []
        for task_id in task_ids:
            if task_id in self._results:
                continue
            if task_id in self._in_flight:
                awaitables.append(self._in_flight[task_id])

        if awaitables:
            await asyncio.gather(*awaitables, return_exceptions=True)

        out: dict[str, str] = {}
        for task_id in task_ids:
            if task_id in self._results:
                result = self._results[task_id]
                if result.status == "succeeded":
                    out[task_id] = f"succeeded: {result.output}"
                else:
                    out[task_id] = f"failed: {result.error}"
            else:
                out[task_id] = f"not_found: {task_id}"
        return json.dumps(out, ensure_ascii=False)

    def get_task_result(self, task_id: str) -> str:
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
        state_store: FileChildTaskStateStore | None = None,
    ) -> None:
        from ..heartbeat import TaskHeartbeatRegistry

        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
        self._child_task_bus = child_task_bus or InMemoryChildTaskBus()
        self._task_heartbeat_registry = task_heartbeat_registry or TaskHeartbeatRegistry()
        if state_store is not None:
            self._state_store = state_store
        elif hasattr(resource_manager, "config"):
            home_dir = getattr(resource_manager.config, "home_dir", None)
            self._state_store = (
                FileChildTaskStateStore(Path(home_dir) / "runtime" / "child_task_state")
                if home_dir is not None
                else None
            )
        else:
            self._state_store = None

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
        heartbeat = context.state.get("heartbeat")
        reply_text: str | None = None
        runtime = InProcessChildTaskRuntime(
            flow_id=context.session_id or "orchestrator",
            resource_manager=self._rm,
            bridge=self._bridge,
            child_task_bus=self._child_task_bus,
            task_heartbeat_registry=self._task_heartbeat_registry,
            max_parallel=4,
            max_tasks=self.MAX_TASKS,
            state_store=self._state_store,
        )

        messages = self._build_initial_messages(goal, context)

        for step in range(self.MAX_STEPS):
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
            for tc in response.tool_calls:
                result_text = await self._dispatch_tool(
                    tc, runtime, context, task_counter,
                )
                if tc.name == "dispatch_task" and not result_text.startswith("error:"):
                    task_counter += 1
                messages.append(ModelMessage(
                    role="tool",
                    content=result_text,
                    tool_call_id=tc.call_id,
                ))
                if tc.name == "reply_to_user":
                    reply_text = tc.arguments.get("text", result_text)

            if reply_text is not None:
                runtime.cancel_all()
                return FinalResult(conclusion=reply_text, task_results=runtime.results)

        return self._build_fallback_result(goal, runtime.results, runtime.in_flight)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_initial_messages(
        self, goal: str, context: ExecutionContext,
    ) -> list[ModelMessage]:
        briefs = self._rm.get_resource_briefs()
        tape = context.state.get("tape")
        history = build_history_summary(tape)
        media_paths = context.state.get("media_paths") or ()

        system_parts = [_SYSTEM_PROMPT_ROLE, _build_resource_catalog(briefs)]
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
    ) -> str:
        name = tool_call.name
        args = tool_call.arguments

        if args.get("__tool_argument_parse_error__"):
            return f"error: invalid arguments: {args.get('__raw_arguments__', '')}"

        if name == "dispatch_task":
            return await runtime.dispatch(args, task_counter=task_counter, context=context)
        if name == "wait_for_tasks":
            return await runtime.wait_for_tasks(args.get("task_ids", []))
        if name == "get_task_result":
            return runtime.get_task_result(args.get("task_id", ""))
        if name == "reply_to_user":
            return args.get("text", "")
        return f"error: unknown tool: {name}"

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
