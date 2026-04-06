"""Runtime support helpers for orchestrator task lifecycle bookkeeping."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from .context import Tape
from .runtime_jobs import JOB_STATES, project_job_state_from_runtime_event


class OrchestratorRuntimeSupport:
    """Encapsulates tape/job/runtime-event bookkeeping for OrchestratorAgent."""

    def __init__(
        self,
        *,
        config: Any,
        tape_store: Any = None,
        memory_store: Any = None,
        runtime_job_store: Any = None,
        invoke_callback: Callable[[Callable[[Any], Awaitable[None] | None] | None, Any], Awaitable[None]]
        | None = None,
        spawn_background_task: Callable[[Awaitable[Any], str], Any] | None = None,
        maybe_handoff: Callable[[Tape, str], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._tape_store = tape_store
        self._memory_store = memory_store
        self._runtime_job_store = runtime_job_store
        self._invoke_callback = invoke_callback
        self._spawn_background_task = spawn_background_task
        self._maybe_handoff = maybe_handoff

    def prepare_tape(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None = None,
    ) -> Tape | None:
        if not chat_key or self._tape_store is None:
            return None
        tape = self._tape_store.get_or_create(chat_key)
        if self._memory_store is not None:
            self._memory_store.observe_user_message(chat_key, user_input)
        pending_entries = []
        if tape.last_anchor() is None:
            anchor = tape.append("anchor", {"name": "session/start", "state": {}})
            pending_entries.append(anchor)
        content_for_tape = user_input
        if media_paths:
            content_for_tape = f"{user_input}\n[附带 {len(media_paths)} 张图片]"
        user_entry = tape.append(
            "message", {"role": "user", "content": content_for_tape}
        )
        pending_entries.append(user_entry)
        self._tape_store.save_entries(chat_key, pending_entries)
        return tape

    def create_runtime_job(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None = None,
        job_metadata_override: dict[str, Any] | None = None,
    ) -> Any | None:
        if not chat_key or self._runtime_job_store is None:
            return None
        runtime_metadata = dict(job_metadata_override or {})
        runtime_metadata.setdefault("media_paths", list(media_paths or []))
        runtime_job = self._runtime_job_store.create(
            chat_key=chat_key,
            goal=user_input,
            metadata=runtime_metadata,
        )
        self._runtime_job_store.transition(
            runtime_job.job_id,
            "planning",
            progress_message="已接收任务，准备执行",
        )
        return runtime_job

    def build_runtime_event_recorder(
        self,
        *,
        chat_key: str,
        tape: Tape | None,
        runtime_job: Any | None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None,
    ) -> Callable[[Any], Awaitable[None] | None] | None:
        if runtime_job is None and (tape is None or not chat_key):
            return runtime_event_callback

        async def _record_runtime_event(event: Any) -> None:
            payload = self.runtime_event_payload(event)
            if self._runtime_job_store is not None and runtime_job is not None:
                payload["job_id"] = str(payload.get("job_id", "") or runtime_job.job_id)
                inner_payload = dict(payload.get("payload") or {})
                inner_payload.setdefault("job_id", runtime_job.job_id)
                payload["payload"] = inner_payload
                state, progress_message = self.job_state_from_runtime_event(payload)
                metadata_update: dict[str, Any] = {}
                flow_id = str(payload.get("flow_id", "") or "").strip()
                task_id = str(payload.get("task_id", "") or "").strip()
                stage = str(inner_payload.get("stage", "") or "").strip()
                if flow_id:
                    metadata_update["flow_id"] = flow_id
                if task_id:
                    metadata_update["last_task_id"] = task_id
                if stage:
                    metadata_update["last_stage"] = stage
                self._runtime_job_store.transition(
                    runtime_job.job_id,
                    state,
                    progress_message=progress_message,
                    metadata=metadata_update,
                )
            if tape is not None and chat_key and self._tape_store is not None:
                entry = tape.append(
                    "event",
                    {
                        "event": str(payload.get("event", "") or ""),
                        "payload": dict(payload.get("payload") or {}),
                    },
                    {
                        "task_id": str(payload.get("task_id", "") or ""),
                        "flow_id": str(payload.get("flow_id", "") or ""),
                    },
                )
                self._tape_store.save_entry(chat_key, entry)
                if self._memory_store is not None:
                    self._memory_store.observe_runtime_event(chat_key, payload)
            if self._invoke_callback is None:
                if runtime_event_callback is None:
                    return
                maybe = runtime_event_callback(payload)
                if inspect.isawaitable(maybe):
                    await maybe
                return
            await self._invoke_callback(runtime_event_callback, payload)

        return _record_runtime_event

    def record_assistant_reply(
        self,
        *,
        chat_key: str,
        tape: Tape | None,
        text: str,
    ) -> None:
        if not tape or not chat_key or self._tape_store is None:
            return
        asst_entry = tape.append("message", {"role": "assistant", "content": text})
        self._tape_store.save_entry(chat_key, asst_entry)
        if self._spawn_background_task is not None and self._maybe_handoff is not None:
            self._spawn_background_task(
                self._maybe_handoff(tape, chat_key),
                label=f"handoff:{chat_key}",
            )

    @staticmethod
    def runtime_event_payload(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            payload = dict(event)
        else:
            payload = {
                "event": getattr(event, "event", ""),
                "task_id": getattr(event, "task_id", ""),
                "flow_id": getattr(event, "flow_id", ""),
                "payload": dict(getattr(event, "payload", {}) or {}),
            }
        payload["payload"] = dict(payload.get("payload") or {})
        return payload

    @staticmethod
    def job_state_from_runtime_event(event_payload: dict[str, Any]) -> tuple[str, str]:
        state, progress_message = project_job_state_from_runtime_event(event_payload)
        if state not in JOB_STATES:
            return "running", progress_message
        return state, progress_message

    def resolve_job_target(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[Any, str]:
        normalized_target = str(target or "latest").strip() or "latest"
        if self._runtime_job_store is None:
            return None, "当前未启用运行时作业跟踪。"
        job = (
            self._runtime_job_store.latest_for_chat(chat_key)
            if normalized_target == "latest"
            else self._runtime_job_store.get(normalized_target)
        )
        if job is None:
            return None, "未找到对应作业。"
        return job, ""

    def runtime_maintenance_report(
        self,
        *,
        recent_flows_by_chat: dict[str, list[str]] | None = None,
        interactive_sessions: Any = None,
    ) -> str:
        report = (
            self._runtime_job_store.run_maintenance(retention_seconds=0)
            if self._runtime_job_store is not None
            else {}
        )
        stale_sessions = (
            int(interactive_sessions.cleanup())
            if interactive_sessions is not None
            and hasattr(interactive_sessions, "cleanup")
            else 0
        )
        recent_flows = recent_flows_by_chat or {}
        known_flow_ids = (
            self._runtime_job_store.known_flow_ids()
            if self._runtime_job_store is not None
            else set()
        )
        unmatched_recent_flows = sorted(
            {
                str(flow_id).strip()
                for flows in recent_flows.values()
                for flow_id in flows or []
                if str(flow_id).strip() and str(flow_id).strip() not in known_flow_ids
            }
        )
        lines = [
            "[Runtime Maintenance]",
            f"orphaned_jobs_pruned={int(report.get('orphaned_jobs_pruned', 0) or 0)}",
            f"stale_interactive_sessions={stale_sessions}",
            f"unmatched_recent_flows={len(unmatched_recent_flows)}",
        ]
        orphaned_job_ids = report.get("orphaned_job_ids") or []
        if unmatched_recent_flows:
            lines.append(f"flows={', '.join(unmatched_recent_flows[:10])}")
        if orphaned_job_ids:
            lines.append(
                f"jobs={', '.join(str(item) for item in orphaned_job_ids[:10])}"
            )
        return "\n".join(lines)


__all__ = ["OrchestratorRuntimeSupport"]
