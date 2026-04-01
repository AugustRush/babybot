from types import SimpleNamespace
import asyncio
import contextvars

from babybot.agent_kernel import ToolRegistry
from babybot.worker import WorkerRuntimeConfig, create_worker_executor
from babybot.builtin_tools.workers import (
    build_create_worker_tool,
    build_dispatch_workers_tool,
)


def test_create_worker_executor_uses_configured_worker_max_steps() -> None:
    config = SimpleNamespace(system=SimpleNamespace(worker_max_steps=23))

    executor = create_worker_executor(
        config=config,  # type: ignore[arg-type]
        tools=ToolRegistry(),
        sys_prompt="sys",
        gateway=object(),  # type: ignore[arg-type]
    )

    assert executor.policy.max_steps == 23


def test_create_worker_executor_defaults_worker_max_steps_to_14() -> None:
    config = SimpleNamespace(system=SimpleNamespace())

    executor = create_worker_executor(
        config=config,  # type: ignore[arg-type]
        tools=ToolRegistry(),
        sys_prompt="sys",
        gateway=object(),  # type: ignore[arg-type]
    )

    assert executor.policy.max_steps == 14


def test_create_worker_executor_clamps_runtime_max_steps_to_at_least_one() -> None:
    config = SimpleNamespace(system=SimpleNamespace(worker_max_steps=0))

    executor = create_worker_executor(
        config=config,  # type: ignore[arg-type]
        tools=ToolRegistry(),
        sys_prompt="sys",
        runtime=WorkerRuntimeConfig(max_steps=0),
        gateway=object(),  # type: ignore[arg-type]
    )

    assert executor.policy.max_steps == 1


def test_create_worker_tool_denies_nested_worker_creation() -> None:
    depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
        "worker_depth", default=1
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(system=SimpleNamespace(worker_max_depth=3)),
        _observability_provider=None,
        _get_current_worker_depth_var=lambda: depth_var,
    )

    result = asyncio.run(build_create_worker_tool(owner)("collect facts"))

    assert "cannot create nested workers" in result.lower()


def test_dispatch_workers_tool_denies_nested_worker_dispatch() -> None:
    depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
        "worker_depth", default=1
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(system=SimpleNamespace(worker_max_depth=3)),
        _observability_provider=None,
        _get_current_worker_depth_var=lambda: depth_var,
    )

    result = asyncio.run(
        build_dispatch_workers_tool(owner)(["task one", "task two"])
    )

    assert "cannot dispatch nested workers" in result.lower()
