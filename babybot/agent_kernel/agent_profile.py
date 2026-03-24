"""Declarative agent profiles loaded from AGENT.md files."""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["AgentProfile", "AgentProfileLoader"]

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


@dataclass
class AgentProfile:
    """A reusable agent identity for team interactions."""

    name: str
    role: str
    description: str = ""
    resource_id: str = ""
    system_prompt: str = ""

    def to_agent_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with dispatch_team agent schema."""
        d: dict[str, Any] = {
            "id": self.name,
            "role": self.role,
            "description": self.description,
        }
        if self.resource_id:
            d["resource_id"] = self.resource_id
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        return d


class AgentProfileLoader:
    """Loads AgentProfile instances from AGENT.md files."""

    @staticmethod
    def load_file(path: str) -> AgentProfile:
        """Parse a single AGENT.md file into an AgentProfile."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        match = _FRONTMATTER_RE.match(raw)
        if not match:
            raise ValueError(f"Invalid AGENT.md format (no frontmatter): {path}")

        frontmatter_text = match.group(1)
        body = match.group(2).strip()

        meta = _parse_yaml_frontmatter(frontmatter_text)

        name = meta.get("name", "").strip()
        if not name:
            raise ValueError(f"AGENT.md missing required field 'name': {path}")

        role = meta.get("role", "").strip()
        if not role:
            raise ValueError(f"AGENT.md missing required field 'role': {path}")

        return AgentProfile(
            name=name,
            role=role,
            description=meta.get("description", "").strip(),
            resource_id=meta.get("resource_id", "").strip(),
            system_prompt=body,
        )

    @classmethod
    def load_dir(cls, directory: str) -> list[AgentProfile]:
        """Scan a directory for subdirectories containing AGENT.md."""
        profiles: list[AgentProfile] = []
        if not os.path.isdir(directory):
            return profiles

        for entry in sorted(os.listdir(directory)):
            agent_md = os.path.join(directory, entry, "AGENT.md")
            if os.path.isfile(agent_md):
                try:
                    profiles.append(cls.load_file(agent_md))
                except (ValueError, OSError) as exc:
                    logger.warning(
                        "Skipping invalid agent profile %s: %s", agent_md, exc
                    )

        return profiles


def _parse_yaml_frontmatter(text: str) -> dict[str, str]:
    """Minimal YAML-like key: value parser (no dependency on PyYAML)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result
