#!/usr/bin/env python3
"""Lightweight validator for babybot skill folders."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_ROOT_DIRS = {"scripts", "references", "assets"}
PLACEHOLDER_MARKERS = ("[todo", "todo:")
MAX_SKILL_NAME_LENGTH = 64


def _extract_frontmatter(content: str) -> tuple[dict[str, str], str] | tuple[None, None]:
    if not content.startswith("---\n"):
        return None, None
    end = content.find("\n---", 4)
    if end == -1:
        return None, None
    header = content[4:end].strip()
    body = content[end + 4 :].strip()
    parsed: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed, body


def _validate_skill_name(name: str, folder_name: str) -> str | None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        return "Skill name must be hyphen-case using lowercase letters, digits, and hyphens only"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return f"Skill name is too long ({len(name)} > {MAX_SKILL_NAME_LENGTH})"
    normalized_folder = folder_name.replace("_", "-")
    if name != normalized_folder:
        return (
            f"Skill name '{name}' must match directory name '{folder_name}' "
            "after normalizing underscores to hyphens"
        )
    return None


def _validate_description(description: str) -> str | None:
    trimmed = description.strip()
    if not trimmed:
        return "Description cannot be empty"
    lowered = trimmed.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return "Description still contains TODO placeholder text"
    return None


def validate_skill(skill_path: str | Path) -> tuple[bool, str]:
    skill_dir = Path(skill_path).expanduser().resolve()
    if not skill_dir.exists():
        return False, f"Skill folder not found: {skill_dir}"
    if not skill_dir.is_dir():
        return False, f"Path is not a directory: {skill_dir}"

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found"

    content = skill_md.read_text(encoding="utf-8")
    meta, _body = _extract_frontmatter(content)
    if meta is None:
        return False, "Invalid frontmatter format"
    if "name" not in meta:
        return False, "Missing 'name' in frontmatter"
    if "description" not in meta:
        return False, "Missing 'description' in frontmatter"

    name_error = _validate_skill_name(meta["name"], skill_dir.name)
    if name_error:
        return False, name_error

    description_error = _validate_description(meta["description"])
    if description_error:
        return False, description_error

    for child in skill_dir.iterdir():
        if child.name == "SKILL.md":
            continue
        if child.is_dir() and child.name in ALLOWED_ROOT_DIRS:
            continue
        if child.is_symlink():
            continue
        return False, f"Unexpected file or directory in skill root: {child.name}"

    return True, "Skill is valid!"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: quick_validate.py <skill-directory>")
        raise SystemExit(1)
    valid, message = validate_skill(sys.argv[1])
    print(message)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
