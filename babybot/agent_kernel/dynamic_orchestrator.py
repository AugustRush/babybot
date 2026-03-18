"""Dynamic orchestration loop — main agent dispatches sub-agents via tool calls."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .context import ContextManager
from .dag_ports import ResourceBridgeExecutor, build_history_summary
from .model import ModelMessage, ModelRequest, ModelResponse, ModelToolCall
from .types import ExecutionContext, FinalResult, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
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


class InProcessChildTaskRuntime:
    """Current child-task runtime backed by local asyncio tasks."""

    def __init__(
        self,
        *,
        resource_manager: "ResourceManager",
        bridge: ResourceBridgeExecutor,
        child_task_bus: InMemoryChildTaskBus,
        max_parallel: int,
        max_tasks: int,
    ) -> None:
        self._rm = resource_manager
        self._bridge = bridge
        self._child_task_bus = child_task_bus
        self._max_tasks = max_tasks
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._in_flight: dict[str, asyncio.Task] = {}
        self._results: dict[str, TaskResult] = {}

    @property
    def in_flight(self) -> dict[str, asyncio.Task]:
        return self._in_flight

    @property
    def results(self) -> dict[str, TaskResult]:
        return self._results

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
        flow_id = context.session_id or "orchestrator"
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

        await self._child_task_bus.publish(
            ChildTaskEvent(
                flow_id=flow_id,
                task_id=task_id,
                event="queued",
                payload={"resource_id": resource_id, "deps": list(deps)},
            )
        )

        async def _run_with_deps() -> None:
            if deps:
                dep_tasks = [self._in_flight[d] for d in deps if d in self._in_flight]
                if dep_tasks:
                    await asyncio.gather(*dep_tasks, return_exceptions=True)
            try:
                await self._child_task_bus.publish(
                    ChildTaskEvent(flow_id=flow_id, task_id=task_id, event="started")
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
                async with self._semaphore:
                    result = await self._bridge.execute(execution_contract, child_context)
                self._results[task_id] = result
                await self._child_task_bus.publish(
                    ChildTaskEvent(
                        flow_id=flow_id,
                        task_id=task_id,
                        event="succeeded" if result.status == "succeeded" else "failed",
                        payload={
                            "status": result.status,
                            "error": result.error,
                        },
                    )
                )
                child_media = child_context.state.get("media_paths_collected", [])
                if child_media:
                    context.state.setdefault("media_paths_collected", []).extend(child_media)
            except Exception as exc:
                self._results[task_id] = TaskResult(
                    task_id=task_id, status="failed", error=str(exc),
                )
                await self._child_task_bus.publish(
                    ChildTaskEvent(
                        flow_id=flow_id,
                        task_id=task_id,
                        event="failed",
                        payload={"error": str(exc)},
                    )
                )
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
    ) -> None:
        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
        self._child_task_bus = child_task_bus or InMemoryChildTaskBus()

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
        heartbeat = context.state.get("heartbeat")
        reply_text: str | None = None
        runtime = InProcessChildTaskRuntime(
            resource_manager=self._rm,
            bridge=self._bridge,
            child_task_bus=self._child_task_bus,
            max_parallel=4,
            max_tasks=self.MAX_TASKS,
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
