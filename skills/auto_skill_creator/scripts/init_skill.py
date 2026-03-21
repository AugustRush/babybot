#!/usr/bin/env python3
"""Initialize babybot skill directories with a deterministic template."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_SKILL_NAME_LENGTH = 64
ALLOWED_RESOURCES = {"scripts", "references", "assets"}
EXAMPLE_RESOURCE_ALIASES = {"example", "examples"}
RESOURCE_ALIASES = {
    "script": "scripts",
    "scripts": "scripts",
    "code": "scripts",
    "codes": "scripts",
    "reference": "references",
    "references": "references",
    "ref": "references",
    "refs": "references",
    "doc": "references",
    "docs": "references",
    "documentation": "references",
    "asset": "assets",
    "assets": "assets",
    "template": "assets",
    "templates": "assets",
}

SKILL_TEMPLATE = """---
name: {skill_name}
description: {description}
---

# {skill_title}

## When to Use

- Use when the request clearly matches {skill_title} work and benefits from specialized workflow guidance.
- Do not use for unrelated general-purpose tasks.

## Workflow

1. Confirm the request matches this skill's trigger conditions.
2. Read only the bundled resources needed for the current task.
3. Execute the minimum necessary steps and return the result clearly.

## Resources

- `scripts/`: deterministic helpers for repeatable operations.
- `references/`: detailed documentation to load only when needed.
- `assets/`: templates or supporting files used in outputs.

## Constraints

- Keep instructions concise and task-focused.
- Do not add extra files unless they are required by repeated real usage.
"""

EXAMPLE_SCRIPT = """#!/usr/bin/env python3
def example_tool(input_text: str = "{skill_name}") -> str:
    return f"example script for {{input_text}}"
"""

EXAMPLE_REFERENCE = """# Reference

Add detailed reference material for {skill_title}.
"""

EXAMPLE_ASSET = "Replace this placeholder with a real asset if needed.\n"


def normalize_skill_name(skill_name: str) -> str:
    normalized = skill_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    if not normalized:
        raise ValueError("Skill name cannot be empty after normalization")
    if len(normalized) > MAX_SKILL_NAME_LENGTH:
        raise ValueError(
            f"Skill name too long ({len(normalized)} > {MAX_SKILL_NAME_LENGTH})"
        )
    return normalized


def title_case_skill_name(skill_name: str) -> str:
    return " ".join(part.capitalize() for part in skill_name.split("-"))


def _resource_items(raw_resources: str | list[str] | tuple[str, ...] | None) -> list[str]:
    def _flatten(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    decoded = json.loads(text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    decoded = None
                if isinstance(decoded, (list, tuple)):
                    flattened: list[str] = []
                    for item in decoded:
                        flattened.extend(_flatten(item))
                    return flattened
            return [item.strip().lower() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(_flatten(item))
            return flattened
        text = str(value).strip()
        return [text.lower()] if text else []

    return _flatten(raw_resources)


def parse_resources(raw_resources: str | list[str] | tuple[str, ...] | None) -> list[str]:
    resources = [
        RESOURCE_ALIASES.get(item, item)
        for item in _resource_items(raw_resources)
        if item not in EXAMPLE_RESOURCE_ALIASES
    ]
    invalid = sorted({item for item in resources if item not in ALLOWED_RESOURCES})
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_RESOURCES))
        print(f"[ERROR] Unknown resource type(s): {', '.join(invalid)}")
        print(f"Allowed: {allowed}")
        raise SystemExit(1)
    deduped: list[str] = []
    seen: set[str] = set()
    for resource in resources:
        if resource not in seen:
            deduped.append(resource)
            seen.add(resource)
    return deduped


def _default_description(skill_name: str) -> str:
    return (
        f"Use when handling {skill_name} requests that need specialized workflow "
        "guidance, references, or helper scripts."
    )


def _create_resource_dirs(
    skill_dir: Path,
    skill_name: str,
    skill_title: str,
    resources: list[str],
    include_examples: bool,
) -> None:
    for resource in resources:
        resource_dir = skill_dir / resource
        resource_dir.mkdir(parents=True, exist_ok=True)
        if not include_examples:
            continue
        if resource == "scripts":
            example_script = resource_dir / "_example.py"
            if not example_script.exists():
                example_script.write_text(
                    EXAMPLE_SCRIPT.format(skill_name=skill_name),
                    encoding="utf-8",
                )
                example_script.chmod(0o755)
        elif resource == "references":
            example_reference = resource_dir / "reference.md"
            if not example_reference.exists():
                example_reference.write_text(
                    EXAMPLE_REFERENCE.format(skill_title=skill_title),
                    encoding="utf-8",
                )
        elif resource == "assets":
            example_asset = resource_dir / "example.txt"
            if not example_asset.exists():
                example_asset.write_text(EXAMPLE_ASSET, encoding="utf-8")


def init_skill(
    skill_name: str,
    path: str | Path | None = None,
    resources: str | list[str] | tuple[str, ...] | None = None,
    include_examples: bool = False,
    *,
    target: str = "workspace",
    workspace_skills_dir: str | Path | None = None,
    builtin_skills_dir: str | Path | None = None,
) -> Path:
    normalized_name = normalize_skill_name(skill_name)
    skill_title = title_case_skill_name(normalized_name)
    include_examples = include_examples or any(
        item in EXAMPLE_RESOURCE_ALIASES for item in _resource_items(resources)
    )
    parsed_resources = parse_resources(resources)

    if path is not None:
        root = Path(path).expanduser().resolve()
    elif target == "builtin":
        if builtin_skills_dir is None:
            root = Path(__file__).resolve().parents[3] / "skills"
        else:
            root = Path(builtin_skills_dir).expanduser().resolve()
    elif target == "workspace":
        if workspace_skills_dir is None:
            root = Path("~/.babybot/workspace/skills").expanduser().resolve()
        else:
            root = Path(workspace_skills_dir).expanduser().resolve()
    else:
        raise ValueError("target must be 'workspace' or 'builtin'")

    root.mkdir(parents=True, exist_ok=True)
    skill_dir = root / normalized_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        skill_md.write_text(
            SKILL_TEMPLATE.format(
                skill_name=normalized_name,
                skill_title=skill_title,
                description=_default_description(normalized_name),
            ),
            encoding="utf-8",
        )

    _create_resource_dirs(
        skill_dir=skill_dir,
        skill_name=normalized_name,
        skill_title=skill_title,
        resources=parsed_resources,
        include_examples=include_examples,
    )
    return skill_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a babybot skill")
    parser.add_argument("skill_name")
    parser.add_argument("--target", choices=("workspace", "builtin"), default="workspace")
    parser.add_argument("--path")
    parser.add_argument("--workspace-skills-dir")
    parser.add_argument("--builtin-skills-dir")
    parser.add_argument("--resources", default="")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()

    skill_dir = init_skill(
        args.skill_name,
        path=args.path,
        target=args.target,
        workspace_skills_dir=args.workspace_skills_dir,
        builtin_skills_dir=args.builtin_skills_dir,
        resources=args.resources,
        include_examples=args.examples,
    )
    print(skill_dir)


if __name__ == "__main__":
    main()
