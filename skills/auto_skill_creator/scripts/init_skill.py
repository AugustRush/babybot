#!/usr/bin/env python3
"""Initialize babybot skill directories with a deterministic template."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_SKILL_NAME_LENGTH = 64
ALLOWED_RESOURCES = {"scripts", "references", "assets"}

SKILL_TEMPLATE = """---
name: {skill_name}
description: {description}
---

# {skill_title}

## Overview

Describe what this skill enables and when it should be used.

## Workflow

1. Explain how the skill should be triggered.
2. List the core steps the agent should follow.
3. Reference bundled resources when needed.
"""

EXAMPLE_SCRIPT = """#!/usr/bin/env python3
def main() -> None:
    print("example script for {skill_name}")


if __name__ == "__main__":
    main()
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


def parse_resources(raw_resources: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw_resources is None:
        return []
    if isinstance(raw_resources, str):
        resources = [item.strip() for item in raw_resources.split(",") if item.strip()]
    else:
        resources = [str(item).strip() for item in raw_resources if str(item).strip()]
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
    return f"Use this skill when working with {skill_name} tasks."


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
            example_script = resource_dir / "example.py"
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
