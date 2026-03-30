from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import InteractiveReply, InteractiveRequest


@dataclass
class ClaudeSessionHandle:
    chat_key: str
    session_id: str
    runtime_root: Path
    env: dict[str, str]
    process: Any
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stderr_task: asyncio.Task[Any] | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    mode: str = "resident"
    is_stopped: bool = False
    stop_reason: str = ""
    last_error: str = ""

    @property
    def process_pid(self) -> int | None:
        return getattr(self.process, "pid", None)


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
        session_id = str(uuid.uuid4())
        proc = await asyncio.create_subprocess_exec(
            *self._build_command(session_id=session_id),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self._workspace_root),
        )
        handle = ClaudeSessionHandle(
            chat_key=chat_key,
            session_id=session_id,
            runtime_root=runtime_root,
            env=env,
            process=proc,
        )
        if proc.stderr is not None:
            handle.stderr_task = asyncio.create_task(self._drain_stderr(handle))
        return handle

    async def send(
        self, handle: ClaudeSessionHandle, message: InteractiveRequest | str
    ) -> InteractiveReply:
        if handle.is_stopped:
            raise RuntimeError("Claude 交互会话已关闭，请重新启动。")
        if handle.process.returncode is not None:
            handle.last_error = handle.last_error or "Claude resident process exited"
            raise RuntimeError("Claude 交互会话已失效，请重新启动。")
        request = (
            message
            if isinstance(message, InteractiveRequest)
            else InteractiveRequest(text=str(message or ""))
        )
        stdin = handle.process.stdin
        if stdin is None:
            raise RuntimeError("Claude 交互会话缺少 stdin，无法继续发送消息。")
        async with handle.send_lock:
            payload = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": request.text}],
                },
            }
            stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await stdin.drain()
            try:
                text = await asyncio.wait_for(
                    self._read_turn_output(handle),
                    timeout=self._default_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await self._kill_process(handle)
                handle.is_stopped = True
                handle.stop_reason = "timeout"
                handle.last_error = (
                    f"Claude interactive backend timed out after {self._default_timeout_s:.2f}s"
                )
                raise RuntimeError(handle.last_error) from exc
            return InteractiveReply(text=text)

    async def stop(
        self, handle: ClaudeSessionHandle, reason: str = "user_stop"
    ) -> None:
        handle.is_stopped = True
        handle.stop_reason = str(reason or "user_stop")
        await self._terminate_process(handle)
        if handle.stderr_task is not None:
            handle.stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.stderr_task
            handle.stderr_task = None
        shutil.rmtree(handle.runtime_root, ignore_errors=True)

    def status(self, handle: ClaudeSessionHandle) -> dict[str, Any]:
        proc = handle.process
        return {
            "backend": "claude",
            "mode": handle.mode,
            "session_id": handle.session_id,
            "runtime_root": str(handle.runtime_root),
            "pid": getattr(proc, "pid", None),
            "alive": bool(not handle.is_stopped and getattr(proc, "returncode", None) is None),
            "is_stopped": handle.is_stopped,
            "stop_reason": handle.stop_reason,
            "last_error": handle.last_error,
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

    def _build_command(self, *, session_id: str) -> list[str]:
        return [
            self._claude_bin,
            "-p",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--session-id",
            session_id,
        ]

    async def _read_turn_output(self, handle: ClaudeSessionHandle) -> str:
        stdout = handle.process.stdout
        if stdout is None:
            raise RuntimeError("Claude 交互会话缺少 stdout，无法读取输出。")
        latest_text = ""
        while True:
            raw_line = await stdout.readline()
            if not raw_line:
                detail = handle.last_error or "Claude resident process closed stdout"
                raise RuntimeError(detail)
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            maybe_session_id = str(payload.get("session_id", "") or "").strip()
            if maybe_session_id:
                handle.session_id = maybe_session_id
            event_type = str(payload.get("type", "") or "").strip().lower()
            if event_type == "result":
                subtype = str(payload.get("subtype", "") or "").strip().lower()
                if subtype in {"error", "failed"}:
                    detail = self._extract_result_text(payload) or payload.get("error") or "Claude 交互执行失败"
                    handle.last_error = str(detail)
                    raise RuntimeError(str(detail))
                return (self._extract_result_text(payload) or latest_text).strip()
            if event_type == "error":
                detail = str(payload.get("error", "") or payload.get("message", "") or "Claude 交互执行失败").strip()
                handle.last_error = detail
                raise RuntimeError(detail)
            extracted = self._extract_result_text(payload)
            if extracted:
                latest_text = extracted

    @staticmethod
    def _extract_result_text(payload: dict[str, Any]) -> str:
        result = payload.get("result")
        if isinstance(result, str):
            return result
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        text = str(item.get("text", "") or "").strip()
                        if text:
                            parts.append(text)
                if parts:
                    return "".join(parts)
        return ""

    async def _drain_stderr(self, handle: ClaudeSessionHandle) -> None:
        stderr = handle.process.stderr
        if stderr is None:
            return
        while True:
            raw_line = await stderr.readline()
            if not raw_line:
                return
            text = raw_line.decode(errors="replace").strip()
            if not text:
                continue
            handle.stderr_tail.append(text)
            handle.last_error = text

    async def _terminate_process(self, handle: ClaudeSessionHandle) -> None:
        proc = handle.process
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            stdin.close()
            maybe_wait_closed = getattr(stdin, "wait_closed", None)
            if callable(maybe_wait_closed):
                with contextlib.suppress(Exception):
                    await maybe_wait_closed()
        if getattr(proc, "returncode", None) is not None:
            return
        terminate = getattr(proc, "terminate", None)
        if callable(terminate):
            maybe = terminate()
            if asyncio.iscoroutine(maybe):
                await maybe
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            await self._kill_process(handle)

    async def _kill_process(self, handle: ClaudeSessionHandle) -> None:
        proc = handle.process
        kill = getattr(proc, "kill", None)
        if callable(kill):
            maybe = kill()
            if asyncio.iscoroutine(maybe):
                await maybe
        await proc.wait()
