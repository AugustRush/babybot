#!/usr/bin/env python3
"""Lightweight validator for babybot skill folders."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ALLOWED_ROOT_DIRS = {"scripts", "references", "assets"}
PLACEHOLDER_MARKERS = ("[todo", "todo:")
MAX_SKILL_NAME_LENGTH = 64
SKIP_SCRIPT_FUNCTIONS = {"main", "parse_arguments", "create_client"}
BODY_PLACEHOLDER_MARKERS = (
    "describe what this skill enables",
    "explain how the skill should be triggered",
    "list the core steps the agent should follow",
)


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
    if not lowered.startswith("use when "):
        return "Description must start with 'Use when' and describe trigger conditions"
    if "use this skill when" in lowered:
        return "Description must describe trigger conditions directly, not say 'Use this skill when'"
    return None


def _validate_body(body: str) -> str | None:
    lowered = body.strip().lower()
    if not lowered:
        return "Skill body cannot be empty"
    for marker in PLACEHOLDER_MARKERS:
        if marker in lowered:
            return "Skill body still contains TODO placeholder text"
    for marker in BODY_PLACEHOLDER_MARKERS:
        if marker in lowered:
            return "Skill body still contains placeholder guidance and must be rewritten"
    return None


def _validate_scripts_dir(scripts_dir: Path) -> str | None:
    for script_path in sorted(scripts_dir.rglob("*.py")):
        if script_path.name.startswith("_"):
            continue
        try:
            tree = ast.parse(script_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"Failed to parse script {script_path.name}: {exc}"

        public_functions = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") or node.name in SKIP_SCRIPT_FUNCTIONS:
                continue
            public_functions.append(node.name)

        if public_functions:
            continue

        return (
            f"Script '{script_path.name}' does not expose a public callable function. "
            "Public scripts in scripts/ must define at least one agent-callable function; "
            "pure CLI/demo/helper modules should be renamed with a leading underscore "
            "(for example '_helper.py') or wrapped by a public function."
        )
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
    meta, body = _extract_frontmatter(content)
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
    body_error = _validate_body(body)
    if body_error:
        return False, body_error

    for child in skill_dir.iterdir():
        if child.name == "SKILL.md":
            continue
        if child.is_dir() and child.name in ALLOWED_ROOT_DIRS:
            continue
        if child.is_symlink():
            continue
        return False, f"Unexpected file or directory in skill root: {child.name}"

    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        scripts_error = _validate_scripts_dir(scripts_dir)
        if scripts_error:
            return False, scripts_error

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
