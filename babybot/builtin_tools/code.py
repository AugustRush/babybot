from __future__ import annotations

from typing import Any


def iter_code_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (owner._workspace_execute_python_code, "code"),
        (owner._workspace_execute_shell_command, "code"),
        (owner._workspace_view_text_file, "code"),
        (owner._workspace_write_text_file, "code"),
        (owner._workspace_insert_text_file, "code"),
    )
