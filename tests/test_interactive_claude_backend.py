from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _make_process(stdout: str, returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_claude_backend_start_uses_isolated_environment(tmp_path: Path):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )

    with patch("asyncio.create_subprocess_exec", return_value=_make_process('{"session_id":"sess_1","result":"started"}')) as mock_exec:
        session = await backend.start(chat_key="feishu:c1")

    env = mock_exec.call_args.kwargs["env"]
    assert env["HOME"].startswith(str(tmp_path))
    assert session.session_id == "sess_1"


@pytest.mark.asyncio
async def test_claude_backend_send_returns_backend_reply(tmp_path: Path):
    from babybot.interactive_sessions.backends.claude import ClaudeInteractiveBackend

    backend = ClaudeInteractiveBackend(
        claude_bin="claude",
        workspace_root=tmp_path,
    )
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=[
            _make_process('{"session_id":"sess_1","result":"started"}'),
            _make_process(json.dumps({"session_id": "sess_1", "result": "available models"})),
        ],
    ):
        session = await backend.start(chat_key="feishu:c1")
        reply = await backend.send(session, "/models")

    assert reply.text == "available models"
