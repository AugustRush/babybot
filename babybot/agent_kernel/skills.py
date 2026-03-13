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
    """Merge two leases with conservative defaults.

    - include_* are intersected when both sides define constraints
      (empty means "no explicit include constraint")
    - exclude_tools are unioned (deny wins)
    """
    primary_groups = set(primary.include_groups)
    secondary_groups = set(secondary.include_groups)
    primary_tools = set(primary.include_tools)
    secondary_tools = set(secondary.include_tools)

    if primary_groups and secondary_groups:
        include_groups = primary_groups & secondary_groups
    else:
        include_groups = primary_groups | secondary_groups

    if primary_tools and secondary_tools:
        include_tools = primary_tools & secondary_tools
    else:
        include_tools = primary_tools | secondary_tools

    return ToolLease(
        include_groups=tuple(sorted(include_groups)),
        include_tools=tuple(sorted(include_tools)),
        exclude_tools=tuple(sorted(set(primary.exclude_tools) | set(secondary.exclude_tools))),
    )


def merge_prompts(skills: list[SkillPack]) -> str:
    """Combine non-empty system prompt fragments in stable order."""
    return "\n\n".join(skill.system_prompt.strip() for skill in skills if skill.system_prompt.strip())
