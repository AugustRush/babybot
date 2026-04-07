"""CallableTool — wraps Python callables into the kernel Tool protocol.

Also provides the write-root context-variable helpers used by
ResourceManager and CallableTool to scope output file writes.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_kernel import ToolResult
from .resource_path_utils import (
    _collect_artifact_paths,
    _looks_like_path_candidate,
    _normalize_artifact_path_for_manager,
)

if TYPE_CHECKING:
    from .resource import ResourceManager

# ── ContextVar declarations (moved from resource.py) ──
_CURRENT_CALLABLE_TOOL_WRITE_ROOT: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("current_callable_tool_write_root", default=None)
)
_CURRENT_DEFAULT_WRITE_ROOT: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("current_default_write_root", default=None)
)


class CallableTool:
    """Wrap a python callable into kernel Tool protocol."""

    def __init__(
        self,
        func: Any,
        name: str,
        description: str,
        schema: dict[str, Any],
        preset_kwargs: dict[str, Any] | None = None,
        resource_manager: "ResourceManager | None" = None,
        collect_artifacts: bool = True,
    ):
        self._func = func
        self._name = name
        self._description = description
        self._schema = schema
        self._preset_kwargs = dict(preset_kwargs or {})
        self._resource_manager = resource_manager
        self._collect_artifacts_enabled = collect_artifacts

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    async def invoke(self, args: dict[str, Any], context: Any) -> ToolResult:
        tool_context_token: contextvars.Token[Any | None] | None = None
        write_root_token: contextvars.Token[str | None] | None = None
        try:
            kwargs = dict(self._preset_kwargs)
            kwargs.update(args or {})
            artifact_base = self._current_write_root()
            write_root = artifact_base
            state = getattr(context, "state", None)
            if isinstance(state, dict):
                state.setdefault("write_root", str(write_root))
            if self._resource_manager is not None:
                tool_context_token = (
                    self._resource_manager._get_current_tool_context_var().set(context)
                )
            with override_current_write_root(write_root):
                write_root_token = _CURRENT_CALLABLE_TOOL_WRITE_ROOT.set(
                    str(write_root)
                )
                if inspect.iscoroutinefunction(self._func):
                    value = await self._func(**kwargs)
                else:
                    # Run sync callables in a thread to avoid blocking the loop.
                    # Use contextvars.copy_context() so channel context etc.
                    # are visible inside the thread (Python <3.12 doesn't copy
                    # context automatically in run_in_executor).
                    loop = asyncio.get_running_loop()
                    ctx = contextvars.copy_context()
                    value = await loop.run_in_executor(
                        None,
                        ctx.run,
                        lambda: self._func(**kwargs),
                    )
            return ToolResult(
                ok=True,
                content=self._normalize_result(value),
                artifacts=(
                    self._collect_artifacts(value, base_dir=artifact_base)
                    if self._collect_artifacts_enabled
                    else []
                ),
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
        finally:
            if write_root_token is not None:
                _CURRENT_CALLABLE_TOOL_WRITE_ROOT.reset(write_root_token)
            if tool_context_token is not None and self._resource_manager is not None:
                self._resource_manager._get_current_tool_context_var().reset(
                    tool_context_token
                )

    @staticmethod
    def _normalize_result(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _current_write_root(self) -> Path:
        if self._resource_manager is None:
            return get_current_write_root()
        return self._resource_manager._get_active_write_root()

    @staticmethod
    def _looks_like_path_candidate(candidate: str) -> bool:
        return _looks_like_path_candidate(candidate)

    def _collect_artifacts(
        self,
        value: Any,
        *,
        base_dir: Path | None = None,
    ) -> list[str]:
        return _collect_artifact_paths(
            value,
            base_dir=(base_dir or self._current_write_root()).resolve(),
            normalize_path=self._normalize_artifact_path,
        )

    def _normalize_artifact_path(self, resolved: Path) -> Path:
        return _normalize_artifact_path_for_manager(self._resource_manager, resolved)


@contextmanager
def override_current_write_root(path: str | os.PathLike[str]) -> Any:
    resolved = Path(path).expanduser().resolve()
    token = _CURRENT_DEFAULT_WRITE_ROOT.set(str(resolved))
    try:
        yield resolved
    finally:
        _CURRENT_DEFAULT_WRITE_ROOT.reset(token)


def get_current_write_root() -> Path:
    raw = _CURRENT_CALLABLE_TOOL_WRITE_ROOT.get()
    if raw:
        return Path(raw).expanduser().resolve()
    fallback = _CURRENT_DEFAULT_WRITE_ROOT.get()
    if fallback:
        return Path(fallback).expanduser().resolve()
    return Path.cwd().resolve()
