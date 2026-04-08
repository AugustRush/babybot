"""Unified subprocess sandbox — Layer 2 of defense-in-depth.

Provides:
  - Environment variable allowlist filtering (no secrets leaking)
  - Process group isolation (start_new_session=True for all paths)
  - Resource limits via rlimit (CPU, memory, file descriptors, processes)
  - Clean process tree killing on timeout
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SandboxConfig

logger = logging.getLogger(__name__)

_IS_DARWIN = platform.system() == "Darwin"
_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class SubprocessResult:
    """Result of a sandboxed subprocess execution."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


class SubprocessSandbox:
    """Unified sandboxed subprocess executor.

    Replaces the ad-hoc subprocess creation scattered across
    resource_workspace_tools.py and resource_python_runner.py.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    async def run(
        self,
        executable: str,
        args: list[str] | tuple[str, ...] = (),
        *,
        stdin_data: bytes | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> SubprocessResult:
        """Run a subprocess in a sandboxed environment.

        Args:
            executable: Path to the executable.
            args: Command-line arguments.
            stdin_data: Optional data to feed to stdin.
            cwd: Working directory.
            env: Base environment (will be filtered through allowlist).
            timeout: Maximum execution time in seconds.

        Returns:
            SubprocessResult with stdout, stderr, returncode, and timeout flag.
        """
        safe_env = self.build_safe_env(env)
        preexec = self._make_preexec_fn() if not _IS_WINDOWS else None

        try:
            proc = await asyncio.create_subprocess_exec(
                executable,
                *args,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
                env=safe_env,
                start_new_session=True,  # ALWAYS: unified process group isolation
                preexec_fn=preexec,
            )
        except OSError as exc:
            return SubprocessResult(
                stdout="",
                stderr=f"Failed to start process: {exc}",
                returncode=-1,
            )

        effective_timeout = timeout if timeout and timeout > 0 else 300.0
        communicate_kwargs: dict[str, Any] = {}
        if stdin_data is not None:
            communicate_kwargs["input"] = stdin_data

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(**communicate_kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            await self._kill_process_tree(proc)
            try:
                await proc.communicate()
            except Exception:
                pass
            return SubprocessResult(
                stdout="",
                stderr=f"Timeout: execution exceeded {effective_timeout}s.",
                returncode=-1,
                timed_out=True,
            )
        except Exception:
            # Clean up on unexpected errors
            try:
                await self._kill_process_tree(proc)
                await proc.communicate()
            except Exception:
                pass
            raise

        stdout = (stdout_bytes or b"").decode("utf-8", errors="ignore")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="ignore")
        return SubprocessResult(
            stdout=stdout,
            stderr=stderr,
            returncode=int(proc.returncode or 0),
        )

    def build_safe_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build a safe environment by allowlist-filtering the base env.

        Only variables in the allowlist are passed through.
        Variables whose names match secret patterns are explicitly excluded.
        """
        source = base_env if base_env is not None else dict(os.environ)
        config = self._config
        safe: dict[str, str] = {}

        for key, value in source.items():
            # Check allowlist
            if key not in config.env_allowlist:
                continue
            # Double-check: reject if name matches a secret pattern
            upper_key = key.upper()
            if any(pattern in upper_key for pattern in config.env_secret_patterns):
                continue
            safe[key] = value

        # Always ensure PATH is present (subprocess needs it)
        if "PATH" not in safe:
            safe["PATH"] = "/usr/local/bin:/usr/bin:/bin"

        return safe

    def build_shell_args(self, command: str) -> tuple[str, list[str]]:
        """Build shell executable and args.

        Uses plain -c (not -lc) by default to avoid loading login profiles
        that may contain secrets or aliases.
        """
        shell = "/bin/sh"
        if self._config.login_shell:
            return shell, ["-lc", command]
        return shell, ["-c", command]

    def _make_preexec_fn(self) -> Any | None:
        """Create a preexec_fn that sets resource limits.

        Called after fork() but before exec() in the child process.
        Returns None if resource limits are disabled.
        """
        if not self._config.resource_limits_enabled:
            return None

        config = self._config

        def _apply_limits() -> None:
            try:
                import resource as rlimit_mod
            except ImportError:
                return  # Windows — no rlimit support

            try:
                # CPU time limit
                rlimit_mod.setrlimit(
                    rlimit_mod.RLIMIT_CPU,
                    (config.cpu_limit_seconds, config.cpu_limit_seconds),
                )
            except (ValueError, OSError):
                pass  # Some platforms may not support this

            try:
                # Memory limit
                mem_bytes = config.memory_limit_mb * 1024 * 1024
                if _IS_DARWIN:
                    # macOS: RLIMIT_AS is unreliable; use RLIMIT_RSS
                    rlimit_mod.setrlimit(rlimit_mod.RLIMIT_RSS, (mem_bytes, mem_bytes))
                else:
                    rlimit_mod.setrlimit(rlimit_mod.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError, AttributeError):
                pass

            try:
                # File descriptor limit
                rlimit_mod.setrlimit(
                    rlimit_mod.RLIMIT_NOFILE,
                    (config.max_file_descriptors, config.max_file_descriptors),
                )
            except (ValueError, OSError):
                pass

            try:
                # Process limit (prevent fork bombs)
                rlimit_mod.setrlimit(
                    rlimit_mod.RLIMIT_NPROC,
                    (config.max_child_processes, config.max_child_processes),
                )
            except (ValueError, OSError, AttributeError):
                pass  # Not available on all platforms

        return _apply_limits

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
        """Kill a process and its entire process group."""
        if proc.returncode is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
