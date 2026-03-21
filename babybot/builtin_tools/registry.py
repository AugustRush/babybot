from __future__ import annotations

from typing import Any

from .code import iter_code_tool_registrations
from .observability import iter_observability_tool_registrations
from .scheduled_tasks import iter_scheduled_task_tool_registrations
from .workers import iter_worker_tool_registrations


def iter_builtin_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        *iter_worker_tool_registrations(owner),
        *iter_scheduled_task_tool_registrations(owner),
        *iter_observability_tool_registrations(owner),
        *iter_code_tool_registrations(owner),
    )
