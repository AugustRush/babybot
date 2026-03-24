# tests/test_claude_code_executor.py
"""Tests for ClaudeCodeExecutor."""

from __future__ import annotations
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from babybot.agent_kernel.types import ExecutionContext, TaskContract
from babybot.agent_kernel.executors.claude_code import ClaudeCodeExecutor


def _make_process_mock(stdout: str, returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_basic_execution() -> None:
    output_json = json.dumps(
        {"result": "Fixed the bug in auth.py", "session_id": "sess_123"}
    )
    with patch(
        "asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)
    ):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t1", description="Fix the bug in auth.py")
        result = await executor.execute(task, ExecutionContext())
    assert result.status == "succeeded"
    assert "Fixed the bug" in result.output
    assert result.metadata.get("session_id") == "sess_123"


@pytest.mark.asyncio
async def test_resume_session() -> None:
    output_json = json.dumps({"result": "Continued work", "session_id": "sess_123"})
    with patch(
        "asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)
    ) as mock_exec:
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(
            task_id="t2",
            description="Continue the review",
            metadata={"session_id": "sess_123"},
        )
        result = await executor.execute(task, ExecutionContext())
    call_args = mock_exec.call_args
    assert "--resume" in call_args[0]
    assert "sess_123" in call_args[0]
    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_nonzero_exit_code() -> None:
    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_make_process_mock("error output", returncode=1),
    ):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t3", description="Fail")
        result = await executor.execute(task, ExecutionContext())
    assert result.status == "failed"
    assert result.error


@pytest.mark.asyncio
async def test_timeout() -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(task_id="t4", description="Hang", timeout_s=1.0)
        result = await executor.execute(task, ExecutionContext())
    assert result.status == "failed"
    assert "timeout" in result.error.lower()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_allowed_tools_passed() -> None:
    output_json = json.dumps({"result": "ok", "session_id": "s1"})
    with patch(
        "asyncio.create_subprocess_exec", return_value=_make_process_mock(output_json)
    ) as mock_exec:
        executor = ClaudeCodeExecutor(workdir="/tmp/project")
        task = TaskContract(
            task_id="t5",
            description="Read only",
            metadata={"allowed_tools": ["Read", "Grep"]},
        )
        await executor.execute(task, ExecutionContext())
    call_args = mock_exec.call_args[0]
    assert "--allowedTools" in call_args
    idx = call_args.index("--allowedTools")
    assert call_args[idx + 1] == "Read,Grep"


# ---------------------------------------------------------------------------
# E2E integration tests (skipped when `claude` CLI is not installed)
# ---------------------------------------------------------------------------
import shutil

_CLAUDE_AVAILABLE = shutil.which("claude") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_basic_prompt() -> None:
    """E2E: ClaudeCodeExecutor sends a real prompt to claude CLI."""
    executor = ClaudeCodeExecutor(workdir="/tmp", default_timeout_s=60.0)
    task = TaskContract(
        task_id="e2e_1",
        description="Reply with exactly the word HELLO and nothing else.",
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "succeeded", f"Expected success, got: {result.error}"
    assert result.task_id == "e2e_1"
    assert "HELLO" in result.output.upper()


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_session_id_returned() -> None:
    """E2E: ClaudeCodeExecutor returns a session_id in metadata."""
    executor = ClaudeCodeExecutor(workdir="/tmp", default_timeout_s=60.0)
    task = TaskContract(
        task_id="e2e_2",
        description="Say OK.",
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "succeeded", f"Expected success, got: {result.error}"
    assert result.metadata.get("session_id"), "Expected session_id in metadata"


@pytest.mark.asyncio
@pytest.mark.skipif(not _CLAUDE_AVAILABLE, reason="claude CLI not found in PATH")
async def test_integration_timeout_enforcement() -> None:
    """E2E: ClaudeCodeExecutor enforces timeout on real subprocess."""
    executor = ClaudeCodeExecutor(workdir="/tmp")
    task = TaskContract(
        task_id="e2e_3",
        description=(
            "Write a Python script that computes the first 10 million prime numbers "
            "and prints each one. Do not stop until all are printed."
        ),
        timeout_s=3.0,
    )
    result = await executor.execute(task, ExecutionContext())

    assert result.status == "failed"
    assert "timeout" in result.error.lower()
