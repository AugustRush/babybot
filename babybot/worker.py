"""Worker executor factory built on the in-repo agent kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agent_kernel import ExecutorPolicy, SingleAgentExecutor, SkillPack, ToolRegistry
from .config import Config
from .model_gateway import OpenAICompatibleGateway


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    """Runtime knobs for worker execution."""

    max_steps: int = 14


def create_worker_executor(
    config: Config,
    tools: ToolRegistry,
    sys_prompt: str,
    skill_packs: Iterable[SkillPack] | None = None,
    runtime: WorkerRuntimeConfig | None = None,
    gateway: OpenAICompatibleGateway | None = None,
) -> SingleAgentExecutor:
    """Create one worker executor with provided system prompt."""
    runtime = runtime or WorkerRuntimeConfig()
    if gateway is None:
        gateway = OpenAICompatibleGateway(config)
    base_skill = SkillPack(name="worker_base", system_prompt=sys_prompt)
    extra_skills = list(skill_packs or [])
    return SingleAgentExecutor(
        model=gateway,
        tools=tools,
        skill_resolver=lambda task, ctx: [base_skill, *extra_skills],
        policy=ExecutorPolicy(max_steps=max(1, runtime.max_steps)),
    )
