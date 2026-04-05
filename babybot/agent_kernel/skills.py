"""Skill contracts for prompt/tool policy composition."""

from __future__ import annotations

from dataclasses import dataclass

from .lease_utils import merge_tool_leases
from .types import SystemPromptBuilder, ToolLease


@dataclass(frozen=True)
class SkillPack:
    """Composable skill unit.

    A skill can contribute system prompt fragments and tool policy hints.
    """

    name: str
    system_prompt: str = ""
    tool_lease: ToolLease = ToolLease()


def merge_leases(primary: ToolLease, secondary: ToolLease) -> ToolLease:
    """Merge two leases with additive include semantics."""
    return merge_tool_leases(primary, secondary)


def merge_prompts(skills: list[SkillPack]) -> str:
    """Combine non-empty system prompt fragments in stable order."""
    return "\n\n".join(
        skill.system_prompt.strip() for skill in skills if skill.system_prompt.strip()
    )


def merge_prompts_as_sections(
    skills: list[SkillPack],
    builder: SystemPromptBuilder | None = None,
    *,
    base_priority: int = 70,
) -> SystemPromptBuilder:
    """Merge skill prompts into named sections inside a SystemPromptBuilder.

    Each skill's prompt becomes a section named ``skill:<skill_name>`` so
    that individual skill contributions are observable and cacheable.

    If *builder* is None a fresh one is created; otherwise sections are
    appended to the existing builder.
    """
    if builder is None:
        builder = SystemPromptBuilder()
    for idx, skill in enumerate(skills):
        text = skill.system_prompt.strip()
        if text:
            builder.add(
                f"skill:{skill.name}",
                text,
                priority=base_priority + idx,
            )
    return builder
