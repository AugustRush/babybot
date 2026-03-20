from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from babybot.channels.tools import ChannelToolContext
    from babybot.cron import ScheduledTaskManager


def require_scheduled_task_manager(owner: Any) -> "ScheduledTaskManager":
    if owner.scheduled_task_manager is None:
        raise RuntimeError("Scheduled task manager is unavailable in this runtime.")
    return owner.scheduled_task_manager


def resolve_scheduled_task_target(
    channel: str | None,
    chat_id: str | None,
) -> tuple[str, str]:
    if channel and chat_id:
        return channel, chat_id

    from ..channels.tools import ChannelToolContext

    ctx = ChannelToolContext.get_current()
    resolved_channel = (channel or "").strip()
    resolved_chat_id = (chat_id or "").strip()
    if ctx is not None:
        if not resolved_channel:
            resolved_channel = getattr(ctx, "channel_name", "") or str(
                (ctx.metadata or {}).get("channel", "")
            )
        if not resolved_chat_id:
            resolved_chat_id = ctx.chat_id

    if not resolved_channel or not resolved_chat_id:
        raise RuntimeError(
            "Scheduled task target is missing. In channel conversations this "
            "should be inherited automatically; otherwise provide channel and chat_id."
        )
    return resolved_channel, resolved_chat_id


def anchor_delay_to_request_time(
    *,
    delay_seconds: float | None,
    run_at: str | None,
) -> tuple[float | None, str | None]:
    if delay_seconds is None or run_at is not None:
        return delay_seconds, run_at

    from ..channels.tools import ChannelToolContext

    ctx = ChannelToolContext.get_current()
    if ctx is None:
        return delay_seconds, run_at
    raw_received_at = (ctx.metadata or {}).get("request_received_at")
    if not isinstance(raw_received_at, str) or not raw_received_at.strip():
        return delay_seconds, run_at
    received_at = raw_received_at.strip()
    if received_at.endswith("Z"):
        received_at = received_at[:-1] + "+00:00"
    base = datetime.datetime.fromisoformat(received_at)
    if base.tzinfo is None:
        base = base.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
    anchored = base.astimezone() + datetime.timedelta(seconds=float(delay_seconds))
    return None, anchored.isoformat(timespec="seconds")


def build_list_scheduled_tasks_tool(owner: Any) -> Any:
    def list_scheduled_tasks() -> str:
        """List all scheduled tasks from the workspace task file."""
        return require_scheduled_task_manager(owner).render_tasks()

    return list_scheduled_tasks


def build_create_scheduled_task_tool(owner: Any) -> Any:
    def create_scheduled_task(
        prompt: str,
        channel: str | None = None,
        chat_id: str | None = None,
        name: str | None = None,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool = True,
        require_active_runtime: bool = True,
    ) -> str:
        """Create a scheduled task.

        Scheduling options (provide exactly one):
        - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
        - run_at: execute once at absolute time (ISO format or HH:MM)
        - cron: recurring cron expression
        - interval_seconds: recurring every N seconds
        """
        channel_name, target_chat_id = resolve_scheduled_task_target(channel, chat_id)
        delay_seconds_resolved, run_at_resolved = anchor_delay_to_request_time(
            delay_seconds=delay_seconds,
            run_at=run_at,
        )
        task = require_scheduled_task_manager(owner).create_task(
            name=name,
            prompt=prompt,
            channel=channel_name,
            chat_id=target_chat_id,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at_resolved,
            delay_seconds=delay_seconds_resolved,
            enabled=enabled,
            require_active_runtime=require_active_runtime,
        )
        return json.dumps(task, ensure_ascii=False, indent=2)

    return create_scheduled_task


def build_save_scheduled_task_tool(owner: Any) -> Any:
    def save_scheduled_task(
        prompt: str,
        channel: str | None = None,
        chat_id: str | None = None,
        name: str | None = None,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool = True,
        require_active_runtime: bool = True,
    ) -> str:
        """Create or update a scheduled task. Prefer this for natural-language task management.

        Scheduling options (provide exactly one):
        - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
        - run_at: execute once at absolute time (ISO format or HH:MM)
        - cron: recurring cron expression
        - interval_seconds: recurring every N seconds
        """
        channel_name, target_chat_id = resolve_scheduled_task_target(channel, chat_id)
        delay_seconds_resolved, run_at_resolved = anchor_delay_to_request_time(
            delay_seconds=delay_seconds,
            run_at=run_at,
        )
        task = require_scheduled_task_manager(owner).save_task(
            name=name,
            prompt=prompt,
            channel=channel_name,
            chat_id=target_chat_id,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at_resolved,
            delay_seconds=delay_seconds_resolved,
            enabled=enabled,
            require_active_runtime=require_active_runtime,
        )
        return json.dumps(task, ensure_ascii=False, indent=2)

    return save_scheduled_task


def build_update_scheduled_task_tool(owner: Any) -> Any:
    def update_scheduled_task(
        name: str,
        prompt: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool | None = None,
        require_active_runtime: bool = True,
    ) -> str:
        """Update one scheduled task by name.

        Scheduling options (provide exactly one to switch schedule):
        - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
        - run_at: execute once at absolute time (ISO format or HH:MM)
        - cron: recurring cron expression
        - interval_seconds: recurring every N seconds
        """
        channel_name = channel
        target_chat_id = chat_id
        if channel_name is None or target_chat_id is None:
            try:
                channel_name, target_chat_id = resolve_scheduled_task_target(
                    channel_name,
                    target_chat_id,
                )
            except RuntimeError:
                pass
        delay_seconds_resolved, run_at_resolved = anchor_delay_to_request_time(
            delay_seconds=delay_seconds,
            run_at=run_at,
        )
        task = require_scheduled_task_manager(owner).update_task(
            name=name,
            prompt=prompt,
            channel=channel_name,
            chat_id=target_chat_id,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at_resolved,
            delay_seconds=delay_seconds_resolved,
            enabled=enabled,
            require_active_runtime=require_active_runtime,
        )
        return json.dumps(task, ensure_ascii=False, indent=2)

    return update_scheduled_task


def build_delete_scheduled_task_tool(owner: Any) -> Any:
    def delete_scheduled_task(name: str) -> str:
        """Delete one scheduled task by name."""
        deleted = require_scheduled_task_manager(owner).delete_task(name)
        return json.dumps({"name": name, "deleted": deleted}, ensure_ascii=False, indent=2)

    return delete_scheduled_task


def iter_scheduled_task_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (build_list_scheduled_tasks_tool(owner), "basic"),
        (build_save_scheduled_task_tool(owner), "basic"),
        (build_create_scheduled_task_tool(owner), "basic"),
        (build_update_scheduled_task_tool(owner), "basic"),
        (build_delete_scheduled_task_tool(owner), "basic"),
    )
