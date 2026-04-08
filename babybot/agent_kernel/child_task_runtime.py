"""Local in-process child-task runtime for the dynamic orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from .child_task_events import (
    ChildTaskEvent,
    ChildTaskView,
    InMemoryChildTaskBus,
    notebook_feedback_payload,
)
from .context import ContextManager
from .errors import classify_error, retry_delay_seconds as default_retry_delay_seconds
from .orchestrator_child_tasks import dispatch_resource_ids
from .orchestrator_notebook import NotebookRuntimeHelper
from .plan_notebook import PlanNotebook
from .types import ExecutionContext, TaskContract, TaskResult, ToolLease

if TYPE_CHECKING:
    from ..heartbeat import TaskHeartbeatRegistry
    from ..resource import ResourceManager
    from .protocols import ExecutorPort


def _default_task_title_builder(description: str) -> str:
    return NotebookRuntimeHelper.task_title_from_description(description)


class InProcessChildTaskRuntime:
    """Current child-task runtime backed by local asyncio tasks."""

    MAX_RETRY_CAP = 8

    def __init__(
        self,
        *,
        flow_id: str,
        resource_manager: "ResourceManager",
        bridge: "ExecutorPort",
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
        task_title_builder: Callable[[str], str] = _default_task_title_builder,
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
        self._task_title_builder = task_title_builder
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
        """Return (True, dead_letter_ids) when every dispatched non-scheduler task failed."""
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

    def _task_event_payload(
        self,
        view: ChildTaskView,
        **payload: Any,
    ) -> dict[str, Any]:
        return self._event_payload(**view.event_payload(**payload))

    async def _publish_task_event(
        self,
        *,
        flow_id: str,
        task_id: str,
        event: str,
        view: ChildTaskView,
        notebook: PlanNotebook | Any = None,
        notebook_node_id: str = "",
        **payload: Any,
    ) -> None:
        await self._child_task_bus.publish(
            ChildTaskEvent(
                flow_id=flow_id,
                task_id=task_id,
                event=event,
                payload=self._task_event_payload(
                    view,
                    **payload,
                    **notebook_feedback_payload(notebook, notebook_node_id),
                ),
            )
        )

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
        metadata = dict(result.metadata or {})
        payload: dict[str, Any] = {
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "reply_artifacts_ready": bool(artifacts),
            "reply_artifacts_count": len(artifacts),
        }
        for key in ("resource_id", "description", "error_type", "user_label"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value
        if metadata.get("auto_converged") is True:
            payload["auto_converged"] = True
        completion_mode = metadata.get("completion_mode")
        if isinstance(completion_mode, str) and completion_mode.strip():
            payload["completion_mode"] = completion_mode
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

        resource_ids = dispatch_resource_ids(args)
        description: str = args.get("description", "")
        public_description = str(
            args.get("__public_description", "")
            or args.get("__feedback_label", "")
            or description
        ).strip()
        feedback_label = str(
            args.get("__feedback_label", "") or public_description
        ).strip()
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
            description=public_description,
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
        view = ChildTaskView(
            resource_ids=tuple(resource_ids),
            primary_resource_id=resource_ids[0],
            execution_description=description,
            public_description=public_description,
            user_label=feedback_label,
            deps=tuple(deps),
            skill_ids=tuple(merged_skill_ids),
        )
        notebook_node_id = ""
        notebook = context.state.get("plan_notebook")
        if isinstance(notebook, PlanNotebook):
            parent_id = str(
                context.state.get("current_notebook_node_id", "")
                or notebook.root_node_id
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
                title=self._task_title_builder(view.public_description),
                objective=view.public_description,
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
                **view.metadata_payload(notebook_node_id=notebook_node_id),
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
        await self._publish_task_event(
            flow_id=flow_id,
            task_id=task_id,
            event="queued",
            view=view,
            notebook=notebook,
            notebook_node_id=notebook_node_id,
        )
        self._update_task_state(
            task_id,
            **view.state_payload(status="queued"),
        )

        async def _run_with_deps() -> None:
            progress_monitor = asyncio.create_task(
                self._monitor_task_progress(
                    task_id=task_id,
                    resource_id=view.primary_resource_id,
                    description=view.public_description,
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
                    await self._publish_task_event(
                        flow_id=flow_id,
                        task_id=task_id,
                        event="started",
                        view=view,
                        notebook=notebook,
                        notebook_node_id=notebook_node_id,
                    )
                    self._update_task_state(
                        task_id,
                        status="started",
                        attempts=attempt,
                        max_attempts=max_attempts,
                    )
                    # NOTE: upstream_results are NOT injected into
                    # contract.metadata here.  The full upstream outputs
                    # are injected by _enrich_with_upstream in
                    # dag_ports.py via the shared upstream_results bucket
                    # in context.state.  Removing this redundant copy
                    # saves tokens in the worker prompt.
                    execution_contract = contract
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
                        await self._publish_task_event(
                            flow_id=flow_id,
                            task_id=task_id,
                            event="succeeded",
                            view=view,
                            notebook=notebook,
                            notebook_node_id=notebook_node_id,
                            message="子任务已完成",
                            status=result.status,
                            error=result.error,
                            output=result.output,
                            task_description=view.public_description,
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
                        await self._publish_task_event(
                            flow_id=flow_id,
                            task_id=task_id,
                            event="retrying",
                            view=view,
                            notebook=notebook,
                            notebook_node_id=notebook_node_id,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            error=result.error,
                            error_type=decision.error_type,
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
                    await self._publish_task_event(
                        flow_id=flow_id,
                        task_id=task_id,
                        event="dead_lettered",
                        view=view,
                        notebook=notebook,
                        notebook_node_id=notebook_node_id,
                        message="子任务执行失败",
                        status=final_result.status,
                        error=final_result.error,
                        attempts=final_result.attempts,
                        max_attempts=max_attempts,
                        error_type=decision.error_type,
                        task_description=view.public_description,
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


__all__ = ["InProcessChildTaskRuntime"]
