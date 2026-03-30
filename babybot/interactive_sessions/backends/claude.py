from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import InteractiveReply, InteractiveRequest


@dataclass
class ClaudeSessionHandle:
    chat_key: str
    session_id: str
    runtime_root: Path
    env: dict[str, str]
    is_stopped: bool = False
    stop_reason: str = ""


class ClaudeInteractiveBackend:
    def __init__(
        self,
        *,
        claude_bin: str = "claude",
        workspace_root: str | Path = ".",
        default_timeout_s: float = 60.0,
    ) -> None:
        self._claude_bin = claude_bin
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._default_timeout_s = float(default_timeout_s)

    async def start(self, chat_key: str) -> ClaudeSessionHandle:
        runtime_root = self._runtime_root(chat_key)
        env = self._build_isolated_env(runtime_root)
        output = await self._run_claude(
            prompt="Reply with exactly STARTED.",
            env=env,
        )
        data = self._parse_json_output(output)
        session_id = str(data.get("session_id", "") or "").strip()
        if not session_id:
            raise RuntimeError("Claude interactive backend did not return session_id")
        return ClaudeSessionHandle(
            chat_key=chat_key,
            session_id=session_id,
            runtime_root=runtime_root,
            env=env,
        )

    async def send(
        self, handle: ClaudeSessionHandle, message: InteractiveRequest
    ) -> InteractiveReply:
        if handle.is_stopped:
            raise RuntimeError("Claude 交互会话已关闭，请重新启动。")
        output = await self._run_claude(
            prompt=message.text,
            env=handle.env,
            session_id=handle.session_id,
        )
        data = self._parse_json_output(output)
        text = str(data.get("result", "") or "").strip()
        session_id = str(data.get("session_id", "") or "").strip()
        if session_id:
            handle.session_id = session_id
        return InteractiveReply(text=text)

    async def stop(
        self, handle: ClaudeSessionHandle, reason: str = "user_stop"
    ) -> None:
        handle.is_stopped = True
        handle.stop_reason = str(reason or "user_stop")
        handle.session_id = ""
        shutil.rmtree(handle.runtime_root, ignore_errors=True)
        return None

    def status(self, handle: ClaudeSessionHandle) -> dict[str, Any]:
        return {
            "backend": "claude",
            "session_id": handle.session_id,
            "runtime_root": str(handle.runtime_root),
            "is_stopped": handle.is_stopped,
            "stop_reason": handle.stop_reason,
        }

    def _runtime_root(self, chat_key: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in chat_key).strip("_")
        runtime_root = (
            self._workspace_root / ".interactive_sessions" / "claude" / safe_name
        )
        runtime_root.mkdir(parents=True, exist_ok=True)
        return runtime_root

    def _build_isolated_env(self, runtime_root: Path) -> dict[str, str]:
        home_dir = runtime_root / "home"
        tmp_dir = runtime_root / "tmp"
        state_dir = runtime_root / "state"
        home_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["HOME"] = str(home_dir)
        env["TMPDIR"] = str(tmp_dir)
        env["CLAUDE_CONFIG_DIR"] = str(state_dir)
        return env

    async def _run_claude(
        self,
        *,
        prompt: str,
        env: dict[str, str],
        session_id: str | None = None,
    ) -> str:
        cmd = [self._claude_bin, "-p", prompt, "--output-format", "json"]
        if session_id:
            cmd.extend(["--resume", session_id])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self._workspace_root),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._default_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            maybe_kill = proc.kill()
            if inspect.isawaitable(maybe_kill):
                await maybe_kill
            maybe_wait = proc.wait()
            if inspect.isawaitable(maybe_wait):
                await maybe_wait
            raise RuntimeError(
                f"Claude CLI timed out after {self._default_timeout_s:.2f}s"
            ) from exc
        if proc.returncode != 0:
            detail = (
                stderr.decode(errors="replace").strip()
                or stdout.decode(errors="replace").strip()
            )
            if detail:
                detail = f": {detail}"
            raise RuntimeError(
                f"Claude CLI exited with code {proc.returncode}{detail}"
            )
        return stdout.decode(errors="replace")

    @staticmethod
    def _parse_json_output(stdout: str) -> dict[str, Any]:
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Claude JSON output: {stdout}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid Claude response payload: {data!r}")
        return data
