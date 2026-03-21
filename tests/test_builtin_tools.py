import datetime

from babybot.builtin_tools import iter_builtin_tool_registrations
from babybot.builtin_tools.time import build_get_current_time_tool


class _DummyOwner:
    def create_worker_tool(self):
        async def create_worker(task_description: str) -> str:
            return task_description

        return create_worker

    def dispatch_workers_tool(self):
        async def dispatch_workers(tasks: list[str]) -> str:
            return ",".join(tasks)

        return dispatch_workers

    def list_scheduled_tasks_tool(self):
        def list_scheduled_tasks() -> str:
            return "[]"

        return list_scheduled_tasks

    def save_scheduled_task_tool(self):
        def save_scheduled_task(prompt: str) -> str:
            return prompt

        return save_scheduled_task

    def create_scheduled_task_tool(self):
        def create_scheduled_task(prompt: str) -> str:
            return prompt

        return create_scheduled_task

    def update_scheduled_task_tool(self):
        def update_scheduled_task(name: str) -> str:
            return name

        return update_scheduled_task

    def delete_scheduled_task_tool(self):
        def delete_scheduled_task(name: str) -> str:
            return name

        return delete_scheduled_task

    async def _workspace_execute_python_code(self, code: str) -> str:
        return code

    async def _workspace_execute_shell_command(self, command: str) -> str:
        return command

    async def _workspace_view_text_file(self, file_path: str) -> str:
        return file_path

    async def _workspace_write_text_file(self, file_path: str, content: str) -> str:
        return f"{file_path}:{content}"

    async def _workspace_insert_text_file(
        self,
        file_path: str,
        content: str,
        line_number: int,
    ) -> str:
        return f"{file_path}:{line_number}:{content}"

    def _inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        return flow_id or chat_key or "runtime"

    def _inspect_chat_context(self, chat_key: str = "", query: str = "") -> str:
        return chat_key or query or "context"


def test_iter_builtin_tool_registrations_exposes_expected_groups_and_names() -> None:
    items = list(iter_builtin_tool_registrations(_DummyOwner()))

    assert [(group, func.__name__) for func, group in items] == [
        ("basic", "create_worker"),
        ("basic", "dispatch_workers"),
        ("basic", "list_scheduled_tasks"),
        ("basic", "save_scheduled_task"),
        ("basic", "create_scheduled_task"),
        ("basic", "update_scheduled_task"),
        ("basic", "delete_scheduled_task"),
        ("basic", "inspect_runtime_flow"),
        ("basic", "inspect_chat_context"),
        ("basic", "get_current_time"),
        ("code", "_workspace_execute_python_code"),
        ("code", "_workspace_execute_shell_command"),
        ("code", "_workspace_view_text_file"),
        ("code", "_workspace_write_text_file"),
        ("code", "_workspace_insert_text_file"),
    ]


def test_get_current_time_supports_common_formats(monkeypatch) -> None:
    fixed_now = datetime.datetime(
        2026,
        3,
        21,
        19,
        30,
        45,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8), name="CST"),
    )
    monkeypatch.setattr(
        "babybot.builtin_tools.time._now_local",
        lambda: fixed_now,
    )

    tool = build_get_current_time_tool(_DummyOwner())

    assert tool() == "2026-03-21 19:30:45 CST (UTC+08:00)"
    assert tool(format="iso") == "2026-03-21T19:30:45+08:00"
    assert tool(format="date") == "2026-03-21"
    assert tool(format="time") == "19:30:45"
    assert tool(format="datetime") == "2026-03-21 19:30:45"
    assert tool(format="timestamp") == str(int(fixed_now.timestamp()))
    assert tool(format="timestamp_ms") == str(int(fixed_now.timestamp() * 1000))


def test_get_current_time_rejects_unknown_format(monkeypatch) -> None:
    monkeypatch.setattr(
        "babybot.builtin_tools.time._now_local",
        lambda: datetime.datetime(
            2026,
            3,
            21,
            19,
            30,
            45,
            tzinfo=datetime.timezone.utc,
        ),
    )

    tool = build_get_current_time_tool(_DummyOwner())

    assert tool(format="weird") == (
        "Unsupported format 'weird'. "
        "Supported formats: default, iso, date, time, datetime, timestamp, timestamp_ms."
    )
