"""Executor that drives Claude Code via its CLI (claude -p)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..types import ExecutionContext, TaskContract, TaskResult

logger = logging.getLogger(__name__)

__all__ = ["ClaudeCodeExecutor"]

_DEFAULT_TIMEOUT_S = 300.0


class ClaudeCodeExecutor:
    """Runs tasks by invoking ``claude -p`` as a subprocess.

    Supports:
    - One-shot execution (``claude -p "prompt" --output-format json``)
    - Session resumption via ``--resume session_id``
    - Tool allowlisting via ``--allowedTools``
    - Timeout enforcement via ``task.timeout_s``
    """

    def __init__(
        self,
        workdir: str = ".",
        claude_bin: str = "claude",
        default_timeout_s: float = _DEFAULT_TIMEOUT_S,
        extra_flags: tuple[str, ...] = (),
    ) -> None:
        self._workdir = workdir
        self._claude_bin = claude_bin
        self._default_timeout_s = default_timeout_s
        self._extra_flags = extra_flags

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
        cmd = self._build_command(task)
        timeout = task.timeout_s or self._default_timeout_s

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workdir,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=f"Timeout after {timeout}s",
                )
        except FileNotFoundError:
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=f"Claude Code binary not found: {self._claude_bin}",
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=stderr or stdout or f"Exit code {proc.returncode}",
            )

        return self._parse_output(task.task_id, stdout)

    def _build_command(self, task: TaskContract) -> list[str]:
        cmd = [self._claude_bin, "-p", task.description, "--output-format", "json"]

        session_id = task.metadata.get("session_id")
        if session_id:
            cmd.extend(["--resume", session_id])

        allowed_tools = task.metadata.get("allowed_tools")
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        system_prompt = task.metadata.get("system_prompt")
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        cmd.extend(self._extra_flags)
        return cmd

    @staticmethod
    def _parse_output(task_id: str, stdout: str) -> TaskResult:
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return TaskResult(
                task_id=task_id, status="succeeded", output=stdout.strip()
            )

        result_text = data.get("result", stdout.strip())
        session_id = data.get("session_id")
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["session_id"] = session_id

        return TaskResult(
            task_id=task_id, status="succeeded", output=result_text, metadata=metadata
        )
