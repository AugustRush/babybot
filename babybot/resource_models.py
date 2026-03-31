from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_kernel import ToolLease


@dataclass
class ToolGroup:
    name: str
    description: str
    notes: str = ""
    active: bool = False


@dataclass
class ResourceBrief:
    """One concise resource record exposed to the main routing agent."""

    resource_id: str
    resource_type: str
    name: str
    purpose: str
    group: str = ""
    tool_count: int = 0
    tools_preview: tuple[str, ...] = ()
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.resource_id,
            "type": self.resource_type,
            "name": self.name,
            "purpose": self.purpose,
            "group": self.group,
            "tool_count": self.tool_count,
            "tools_preview": list(self.tools_preview),
            "active": self.active,
        }


@dataclass(frozen=True)
class SkillRuntimeConfig:
    python_executable: str = ""
    python_fallback_executables: tuple[str, ...] = ()
    python_required_modules: tuple[str, ...] = ()


@dataclass
class LoadedSkill:
    """Resolved skill metadata for runtime routing."""

    name: str
    description: str
    directory: str
    prompt: str = ""
    prompt_body: str = ""
    prompt_body_path: str = ""
    role: str = ""
    keywords: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    lease: ToolLease = ToolLease()
    source: str = "auto"
    active: bool = True
    tool_group: str = ""
    tools: tuple[str, ...] = ()
    runtime: SkillRuntimeConfig = field(default_factory=lambda: SkillRuntimeConfig())

    def resolve_prompt(self, *, load_full: bool = False) -> str:
        if not load_full:
            return self.prompt
        if self.prompt_body:
            return self.prompt_body
        if self.prompt_body_path:
            skill_md = Path(self.prompt_body_path)
            try:
                text = skill_md.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return self.prompt
            if text.startswith("---\n"):
                end = text.find("\n---", 4)
                if end != -1:
                    text = text[end + 4 :]
            self.prompt_body = text.strip()
            if self.prompt_body:
                return self.prompt_body
        return self.prompt


@dataclass(frozen=True)
class ScriptFunctionSpec:
    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class CliArgumentSpec:
    name: str
    flag: str
    schema: dict[str, Any]
    required: bool = False
    action: str | None = None


@dataclass(frozen=True)
class CliToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    arguments: tuple[CliArgumentSpec, ...]
