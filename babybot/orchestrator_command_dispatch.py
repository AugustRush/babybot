"""CommandDispatch — handles @policy, @job, @session prefix commands.

Single Responsibility: Parse and dispatch special command prefixes before
the main DAG execution path.

Design: try_dispatch() returns a TaskResponse if the input is a command,
or None if it should fall through to the normal DAG pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .orchestration_policy_types import PolicyOutcomeRecord
from .orchestrator_interactive_support import OrchestratorInteractiveSessionSupport
from .runtime_feedback_commands import parse_policy_command

if TYPE_CHECKING:
    from .orchestrator_inspect import InspectService

logger = logging.getLogger(__name__)


class CommandDispatch:
    """Parses and handles @policy, @job, @session commands.

    All command handlers return TaskResponse-compatible dicts/objects.
    The orchestrator wraps those into actual TaskResponse instances.
    """

    def __init__(
        self,
        *,
        inspect_service: InspectService | None,
        policy_store: Any,
        job_store: Any,
        process_task_fn: Callable[..., Awaitable[Any]],
        runtime_support_fn: Callable[[], Any],
        interactive_support_fn: Callable[[], Any],
        response_factory: Callable[..., Any],
    ) -> None:
        self._inspect = inspect_service
        self._policy_store = policy_store
        self._job_store = job_store
        self._process_task_fn = process_task_fn
        self._runtime_support_fn = runtime_support_fn
        self._interactive_support_fn = interactive_support_fn
        self._response = response_factory

    # ------------------------------------------------------------------ #
    # Parsers (pure functions — no I/O)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_policy_command(user_input: str) -> dict[str, str] | None:
        return parse_policy_command(user_input)

    @staticmethod
    def parse_job_command(user_input: str) -> dict[str, str] | None:
        text = (user_input or "").strip()
        if not text.lower().startswith("@job"):
            return None
        parts = text.split()
        action = parts[1].lower() if len(parts) >= 2 else "status"
        target = parts[2].strip() if len(parts) >= 3 else "latest"
        return {"action": action, "target": target}

    @staticmethod
    def parse_interactive_command(user_input: str) -> dict[str, str] | None:
        return OrchestratorInteractiveSessionSupport.parse_command(user_input)

    # ------------------------------------------------------------------ #
    # Command handlers                                                      #
    # ------------------------------------------------------------------ #

    async def handle_policy_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> Any:
        if control.get("action", "") == "inspect":
            target = str(control.get("target", "") or "").strip()
            if self._inspect is None:
                return self._response(text="[Policy]\n- no_stats")
            if target and not any(
                target == decision_kind
                for decision_kind in ("decomposition", "scheduling", "worker")
            ):
                return self._response(
                    text=self._inspect.inspect_runtime_flow(flow_id=target)
                )
            return self._response(
                text=self._inspect.inspect_policy(
                    chat_key=chat_key,
                    decision_kind=target,
                )
            )
        if control.get("action", "") != "feedback":
            return self._response(
                text="支持的命令：@policy feedback <flow_id|latest> good|bad <reason> / @policy inspect [decision_kind|flow_id]"
            )
        rating = str(control.get("rating", "") or "").strip().lower()
        reason = str(control.get("reason", "") or "").strip()
        if rating not in {"good", "bad"} or not reason:
            return self._response(
                text="用法：@policy feedback <flow_id|latest> good|bad <reason>"
            )
        if not chat_key:
            return self._response(text="缺少 chat_key，无法记录策略反馈。")
        flow_id, error = self._resolve_policy_feedback_flow_id(
            chat_key=chat_key,
            target=str(control.get("target", "") or "").strip(),
        )
        if error:
            return self._response(text=error)
        if self._policy_store is None:
            return self._response(text="当前未启用策略反馈存储。")
        self._policy_store.record_feedback(
            flow_id=flow_id,
            chat_key=chat_key,
            rating=rating,
            reason=reason,
        )
        return self._response(text=f"已记录策略反馈：{rating}。")

    async def handle_job_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> Any:
        action = str(control.get("action", "") or "status").strip()
        target = str(control.get("target", "") or "latest").strip()
        if action == "cleanup":
            runtime_support = self._runtime_support_fn()
            recent_flows: dict[str, Any] = {}
            if self._inspect is not None:
                recent_flows = dict(self._inspect._recent_flows_by_chat)
            return self._response(
                text=runtime_support.runtime_maintenance_report(
                    recent_flows_by_chat=recent_flows,
                    interactive_sessions=None,
                )
            )
        if action not in {"status", "resume"}:
            return self._response(
                text="支持的命令：@job status <job_id|latest> / @job resume <job_id|latest> / @job cleanup"
            )
        runtime_support = self._runtime_support_fn()
        job, error = runtime_support.resolve_job_target(
            chat_key=chat_key, target=target
        )
        if error:
            return self._response(text=error)
        if job is None:
            return self._response(text="未找到对应作业。")
        if action == "resume":
            if job.state == "completed":
                return self._response(text="该作业已完成，无需恢复。")
            if job.state == "running":
                return self._response(
                    text="该作业仍在运行，请先使用 @job status 查询。"
                )
            if self._job_store is None:
                return self._response(text="当前未启用运行时作业跟踪。")
            self._job_store.transition(
                job.job_id,
                "repairing",
                progress_message="准备恢复执行",
                metadata={"resume_requested": True},
            )
            resume_metadata = dict(job.metadata)
            resume_metadata["resumed_from"] = job.job_id
            return await self._process_task_fn(
                job.goal,
                chat_key=job.chat_key or chat_key,
                media_paths=list(job.metadata.get("media_paths") or []),
                job_metadata_override=resume_metadata,
            )
        return self._response(
            text=(
                f"[Job]\njob_id={job.job_id}\nstate={job.state}\n"
                f"progress={job.progress_message or '-'}\n"
                f"flow_id={str(job.metadata.get('flow_id', '') or '-')}"
            )
        )

    async def handle_interactive_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> Any:
        return await self._interactive_support_fn().handle_command(
            chat_key=chat_key,
            control=control,
        )

    async def handle_interactive_message(
        self,
        chat_key: str,
        user_input: str,
        **kwargs: Any,
    ) -> Any | None:
        try:
            return await self._interactive_support_fn().handle_message(
                chat_key=chat_key,
                user_input=user_input,
                **kwargs,
            )
        except RuntimeError:
            logger.warning(
                "Interactive session send failed; falling back to DAG chat_key=%s",
                chat_key,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _resolve_policy_feedback_flow_id(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[str, str]:
        normalized_target = str(target or "").strip()
        store = self._job_store
        # Access flow caches through InspectService (authoritative owner)
        recent_latest: dict[str, str] = {}
        recent_history: dict[str, list[str]] = {}
        if self._inspect is not None:
            recent_latest = dict(self._inspect._recent_flow_ids_by_chat)
            recent_history = dict(self._inspect._recent_flows_by_chat)
        history = [
            str(item).strip()
            for item in recent_history.get(chat_key, []) or []
            if str(item).strip()
        ]
        if normalized_target and normalized_target not in {"latest"}:
            target_job = store.get(normalized_target) if store is not None else None
            if target_job is not None:
                flow_id = str(target_job.metadata.get("flow_id", "") or "").strip()
                if flow_id:
                    return flow_id, ""
                return "", "该 job 尚未关联 flow_id，请指定 flow_id 再反馈。"
        if not history:
            latest = str(recent_latest.get(chat_key, "") or "").strip()
            if latest:
                history = [latest]
        if normalized_target and normalized_target not in {"latest"}:
            return normalized_target, ""
        if not history:
            return "", "当前会话没有可反馈的最近任务。"
        if normalized_target == "latest" and len(history) > 1:
            return "", "最近有多个运行记录，请指定 flow_id 再反馈。"
        return history[0], ""
