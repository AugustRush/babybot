from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any


_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+(-[^\s]*\s+)*-[^\s]*[rR]", "recursive delete"),
    (r"rm\s+-rf\s+/", "recursive delete from root"),
    (r"mkfs", "filesystem format"),
    (r"dd\s+if=", "disk overwrite"),
    (r">\s*/dev/sd", "device overwrite"),
    (r":()\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"chmod\s+-R\s+777\s+/", "recursive permission change on root"),
    (r"curl[^|]*\|\s*(sudo\s+)?bash", "pipe to shell"),
    (r"wget[^|]*\|\s*(sudo\s+)?bash", "pipe to shell"),
    (r"curl[^|]*\|\s*(sudo\s+)?sh\b", "pipe to shell"),
    (r"wget[^|]*\|\s*(sudo\s+)?sh\b", "pipe to shell"),
    # Encoded command bypass attempts
    (r"base64\s+.*\|\s*(ba)?sh", "encoded pipe to shell"),
    (r"\bsudo\s+rm\b", "sudo delete"),
    (r"\bsudo\s+dd\b", "sudo disk write"),
    (r"\bsudo\s+mkfs\b", "sudo filesystem format"),
    # Python / Perl / Ruby one-liners for destructive ops
    (r"python[23]?\s+-c\s+.*shutil\.rmtree", "python recursive delete"),
    (r"python[23]?\s+-c\s+.*os\.remove", "python file delete"),
    (r"perl\s+-e\s+.*unlink", "perl file delete"),
    # Prevent overwriting critical system files
    (r">\s*/etc/", "write to /etc"),
    (r"tee\s+/etc/", "write to /etc via tee"),
]

_DANGEROUS_PYTHON_PATTERNS: list[tuple[str, str]] = [
    (r"shutil\.rmtree\s*\(", "recursive delete"),
    (r"os\.(remove|unlink|rmdir)\s*\(", "file delete"),
    (r"Path\s*\([^)]*\)\.(unlink|rmdir)\s*\(", "path delete"),
    (r"subprocess\.(run|Popen|call|check_output)\s*\(", "subprocess execution"),
    (r"os\.system\s*\(", "shell execution"),
    (r"eval\s*\(", "dynamic eval"),
    (r"exec\s*\(", "dynamic exec"),
]


def check_shell_safety(command: str) -> str | None:
    for pattern, label in _DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked: command matches dangerous pattern ({label})"
    return None


def check_python_safety(code: str) -> str | None:
    for pattern, label in _DANGEROUS_PYTHON_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
            return f"Blocked: python code matches dangerous pattern ({label})"
    return None


class WorkspaceToolSuite:
    def __init__(self, owner: Any) -> None:
        self._owner = owner

    async def execute_python_code(
        self,
        code: str,
        timeout: float | int | str | None = 300,
        **kwargs: Any,
    ) -> str:
        del kwargs
        safety_error = check_python_safety(code)
        if safety_error:
            return safety_error
        ws = str(self._owner._get_active_write_root())
        proc = await asyncio.create_subprocess_exec(
            self._owner._get_user_python(),
            "-c",
            f"import os\nos.chdir({ws!r})\n{code}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._owner._clean_env(),
        )
        timeout_s = self._owner._coerce_timeout(timeout, default=300.0)
        communicate_coro = proc.communicate()
        try:
            if timeout_s and timeout_s > 0:
                stdout, stderr = await asyncio.wait_for(
                    communicate_coro, timeout=timeout_s
                )
            else:
                stdout, stderr = await communicate_coro
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            return f"Timeout: python execution exceeded {timeout_s}s."
        except Exception:
            try:
                communicate_coro.close()
            except Exception:
                pass
            raise
        out = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        text = out.strip()
        if err.strip():
            text = f"{text}\n{err.strip()}".strip()
        return text

    async def execute_shell_command(
        self,
        command: str,
        timeout: float | int | str | None = 300,
        **kwargs: Any,
    ) -> str:
        del kwargs
        safety_error = check_shell_safety(command)
        if safety_error:
            return safety_error
        ws = shlex.quote(str(self._owner._get_active_write_root()))
        guarded = f"cd {ws} && {command}"
        proc = await asyncio.create_subprocess_shell(
            guarded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._owner._clean_env(),
        )
        timeout_s = self._owner._coerce_timeout(timeout, default=300.0)
        communicate_coro = proc.communicate()
        try:
            if timeout_s and timeout_s > 0:
                stdout, stderr = await asyncio.wait_for(
                    communicate_coro, timeout=timeout_s
                )
            else:
                stdout, stderr = await communicate_coro
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            return f"Timeout: shell command exceeded {timeout_s}s."
        except Exception:
            try:
                communicate_coro.close()
            except Exception:
                pass
            raise
        out = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        text = out.strip()
        if err.strip():
            text = f"{text}\n{err.strip()}".strip()
        return text

    async def view_text_file(
        self,
        file_path: str,
        ranges: list[int] | None = None,
    ) -> str:
        resolved, err = self._owner._resolve_workspace_file(file_path)
        if err:
            return err
        lines = await asyncio.to_thread(
            Path(resolved).read_text,
            encoding="utf-8",
            errors="ignore",
        )
        line_list = lines.splitlines(keepends=True)
        if not ranges:
            return "".join(line_list)
        chunks: list[str] = []
        for i in range(0, len(ranges), 2):
            start = max(1, int(ranges[i]))
            end = int(ranges[i + 1]) if i + 1 < len(ranges) else start
            chunks.extend(line_list[start - 1 : end])
        return "".join(chunks)

    async def write_text_file(
        self,
        file_path: str,
        content: str,
        ranges: list[int] | None = None,
    ) -> str:
        resolved, err = self._owner._resolve_workspace_file(file_path)
        if err:
            return err
        target = Path(resolved)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        if not ranges:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            return f"Wrote file: {target}"
        if target.exists():
            existing = await asyncio.to_thread(target.read_text, encoding="utf-8")
            lines = existing.splitlines(keepends=True)
        else:
            lines = []
        start = max(1, int(ranges[0])) if ranges else 1
        end = int(ranges[1]) if ranges and len(ranges) > 1 else start
        replacement = content.splitlines(keepends=True)
        lines[start - 1 : end] = replacement
        await asyncio.to_thread(target.write_text, "".join(lines), encoding="utf-8")
        return f"Updated file range in: {target}"

    async def insert_text_file(
        self,
        file_path: str,
        content: str,
        line_number: int,
    ) -> str:
        resolved, err = self._owner._resolve_workspace_file(file_path)
        if err:
            return err
        target = Path(resolved)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        if target.exists():
            existing = await asyncio.to_thread(target.read_text, encoding="utf-8")
            lines = existing.splitlines(keepends=True)
        else:
            lines = []
        idx = max(0, min(len(lines), int(line_number) - 1))
        lines[idx:idx] = content.splitlines(keepends=True)
        await asyncio.to_thread(target.write_text, "".join(lines), encoding="utf-8")
        return f"Inserted text into: {target}"
