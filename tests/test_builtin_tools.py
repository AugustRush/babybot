import asyncio
import contextvars
import datetime
from types import SimpleNamespace

from babybot.builtin_tools import iter_builtin_tool_registrations
from babybot.builtin_tools.time import build_get_current_time_tool
from babybot.builtin_tools.workers import (
    build_create_worker_tool,
    build_dispatch_workers_tool,
)


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

    async def _workspace_edit_text_file(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        return f"{file_path}:{old_text}:{new_text}:{replace_all}"

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

    def _inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        return decision_kind or chat_key or "policy"

    def _inspect_tools(self, query: str = "", group: str = "", active_only: bool = False) -> str:
        return f"tools:{query}:{group}:{active_only}"

    def _inspect_skills(self, query: str = "", active_only: bool = False) -> str:
        return f"skills:{query}:{active_only}"

    def _inspect_skill_load_errors(self, limit: int = 20) -> str:
        return f"errors:{limit}"

    def reload_skill(self, skill_path: str) -> str:
        return f"reloaded {skill_path}"


def test_iter_builtin_tool_registrations_exposes_expected_groups_and_names() -> None:
    items = list(iter_builtin_tool_registrations(_DummyOwner()))

    assert [(group, func.__name__) for func, group in items] == [
        ("worker_control", "create_worker"),
        ("worker_control", "dispatch_workers"),
        ("basic", "list_scheduled_tasks"),
        ("basic", "save_scheduled_task"),
        ("basic", "create_scheduled_task"),
        ("basic", "update_scheduled_task"),
        ("basic", "delete_scheduled_task"),
        ("basic", "inspect_runtime_flow"),
        ("basic", "inspect_chat_context"),
        ("basic", "inspect_policy"),
        ("basic", "inspect_tools"),
        ("basic", "inspect_skills"),
        ("basic", "inspect_skill_load_errors"),
        ("basic", "get_current_time"),
        ("code", "_workspace_execute_python_code"),
        ("code", "_workspace_execute_shell_command"),
        ("code", "_workspace_view_text_file"),
        ("code", "_workspace_write_text_file"),
        ("code", "_workspace_edit_text_file"),
        ("code", "_workspace_insert_text_file"),
        ("basic", "reload_skill"),
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


def test_worker_tool_is_blocked_when_policy_marks_worker_usage_high_risk() -> None:
    class _PolicyProvider:
        def choose_worker_policy(self, *, features):
            del features
            return {"action_name": "deny_worker", "hint": "high risk"}

    class _Owner:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system=SimpleNamespace(worker_max_depth=3))
            self._lease_var = contextvars.ContextVar("lease_var_builtin_worker", default=None)
            self._skill_ids_var = contextvars.ContextVar("skill_ids_var_builtin_worker", default=None)
            self._worker_depth_var = contextvars.ContextVar(
                "worker_depth_var_builtin_worker", default=0
            )
            self._observability_provider = _PolicyProvider()
            self.called = False

        def _get_current_task_lease_var(self):
            return self._lease_var

        def _get_current_skill_ids_var(self):
            return self._skill_ids_var

        def _get_current_worker_depth_var(self):
            return self._worker_depth_var

        async def run_subagent_task(self, *args, **kwargs):
            del args, kwargs
            self.called = True
            return "done", []

    owner = _Owner()

    result = asyncio.run(build_create_worker_tool(owner)("high risk task"))

    assert "policy denied" in result.lower()
    assert owner.called is False


def test_dispatch_workers_tool_is_blocked_when_policy_marks_worker_usage_high_risk() -> None:
    class _PolicyProvider:
        def choose_worker_policy(self, *, features):
            del features
            return {"action_name": "deny_worker", "hint": "high risk"}

    class _Owner:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system=SimpleNamespace(worker_subtask_timeout=10))
            self._lease_var = contextvars.ContextVar("lease_var_builtin_dispatch", default=None)
            self._skill_ids_var = contextvars.ContextVar("skill_ids_var_builtin_dispatch", default=None)
            self._observability_provider = _PolicyProvider()
            self.called = False

        def _get_current_task_lease_var(self):
            return self._lease_var

        def _get_current_skill_ids_var(self):
            return self._skill_ids_var

        async def run_subagent_task(self, *args, **kwargs):
            del args, kwargs
            self.called = True
            return "done", []

    owner = _Owner()

    result = asyncio.run(
        build_dispatch_workers_tool(owner)(["task one", "task two"], max_concurrency=2)
    )

    assert "policy denied" in result.lower()
    assert owner.called is False
