from types import SimpleNamespace

from babybot.agent_kernel import ToolRegistry
from babybot.worker import WorkerRuntimeConfig, create_worker_executor


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
