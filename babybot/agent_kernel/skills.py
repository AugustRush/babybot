"""Skill contracts for prompt/tool policy composition."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ToolLease


@dataclass(frozen=True)
class SkillPack:
    """Composable skill unit.

    A skill can contribute system prompt fragments and tool policy hints.
    """

    name: str
    system_prompt: str = ""
    tool_lease: ToolLease = ToolLease()


def merge_leases(primary: ToolLease, secondary: ToolLease) -> ToolLease:
    """Merge two leases with additive include semantics.

    - include_groups/include_tools are UNIONED (additive access)
    - exclude_tools are unioned (deny wins)

    This means skills ADD their tool groups to the available set rather
    than restricting them.
    """
    include_groups = set(primary.include_groups) | set(secondary.include_groups)
    include_tools = set(primary.include_tools) | set(secondary.include_tools)

    return ToolLease(
        include_groups=tuple(sorted(include_groups)),
        include_tools=tuple(sorted(include_tools)),
        exclude_tools=tuple(sorted(set(primary.exclude_tools) | set(secondary.exclude_tools))),
    )


def merge_prompts(skills: list[SkillPack]) -> str:
    """Combine non-empty system prompt fragments in stable order."""
    return "\n\n".join(skill.system_prompt.strip() for skill in skills if skill.system_prompt.strip())
