from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class _FakeReader:
    def __init__(self) -> None:
        self._lines: list[bytes] = []

    def push_json(self, payload: dict[str, object]) -> None:
        import json

        self._lines.append((json.dumps(payload, ensure_ascii=False) + "\n").encode())

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, *, pid: int = 4321) -> None:
        self.pid = pid
        self.stdin = _FakeWriter()
        self.stdout = _FakeReader()
        self.stderr = _FakeReader()
        self.returncode: int | None = None
        self.terminate = AsyncMock(side_effect=self._terminate)
        self.kill = AsyncMock(side_effect=self._kill)
        self.wait = AsyncMock(return_value=0)

    async def _terminate(self) -> None:
        self.returncode = 0

    async def _kill(self) -> None:
        self.returncode = -9


def test_claude_backend_start_uses_isolated_environment(tmp_path: Path):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )

    async def _run():
        fake_process = _FakeProcess()
        with patch("asyncio.create_subprocess_exec", return_value=fake_process) as mock_exec:
            session = await backend.start(chat_key="feishu:c1")
        return session, mock_exec

    session, mock_exec = asyncio.run(_run())

    env = mock_exec.call_args.kwargs["env"]
    assert env["HOME"].startswith(str(tmp_path))
    assert session.session_id
    assert session.process_pid == 4321
    assert session.mode == "resident"


def test_claude_backend_send_keeps_resident_process_and_returns_backend_reply(
    tmp_path: Path,
):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )
    async def _run():
        fake_process = _FakeProcess()
        fake_process.stdout.push_json({"type": "result", "subtype": "success", "result": "available models"})
        fake_process.stdout.push_json({"type": "result", "subtype": "success", "result": "current branch"})
        with patch("asyncio.create_subprocess_exec", return_value=fake_process) as mock_exec:
            session = await backend.start(chat_key="feishu:c1")
            reply = await backend.send(session, "/models")
            second = await backend.send(session, "/status")
        return reply, second, mock_exec, fake_process

    reply, second, mock_exec, fake_process = asyncio.run(_run())

    assert reply.text == "available models"
    assert second.text == "current branch"
    assert mock_exec.call_count == 1
    assert len(fake_process.stdin.writes) == 2


def test_claude_backend_stop_cleans_runtime_root_and_blocks_reuse(tmp_path: Path):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )
    async def _run():
        fake_process = _FakeProcess()
        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            session = await backend.start(chat_key="feishu:c1")
        assert session.runtime_root.exists()
        await backend.stop(session, reason="user_stop")
        return session, fake_process

    session, fake_process = asyncio.run(_run())

    assert not session.runtime_root.exists()
    fake_process.terminate.assert_awaited_once()

    async def _send_again() -> None:
        with pytest.raises(RuntimeError, match="已关闭"):
            await backend.send(session, "hello again")

    asyncio.run(_send_again())


def test_claude_backend_send_timeout_kills_process_and_raises_runtime_error(
    tmp_path: Path,
):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
        default_timeout_s=0.01,
    )
    proc = _FakeProcess()

    async def _never_readline() -> bytes:
        await asyncio.sleep(1)
        return b""

    proc.stdout.readline = _never_readline  # type: ignore[method-assign]

    async def _run() -> None:
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            session = await backend.start(chat_key="feishu:c1")
            with pytest.raises(RuntimeError, match="timed out"):
                await backend.send(session, "slow request")

    asyncio.run(_run())

    proc.kill.assert_awaited_once()
    proc.wait.assert_awaited_once()
