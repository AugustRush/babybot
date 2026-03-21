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
ALLOWED_TOOL_KINDS = {"prompt", "scripts", "hybrid"}
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

- {when_to_use_line}
- Do not use for unrelated general-purpose tasks or requests better served by a broader general skill.

## Example Requests

{example_requests_block}

## Workflow

{workflow_block}

## Resources

{resources_block}

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


def _normalize_skill_name(skill_name: str) -> str:
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


def _title_case_skill_name(skill_name: str) -> str:
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


def _parse_resources(raw_resources: str | list[str] | tuple[str, ...] | None) -> list[str]:
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


def _normalize_summary(summary: str | None, skill_title: str) -> str:
    text = (summary or "").strip()
    if text:
        return text.rstrip(".")
    return f"{skill_title} tasks that need specialized workflow guidance"


def _default_example_requests(skill_title: str, tool_kind: str) -> list[str]:
    base = skill_title.lower()
    if tool_kind == "scripts":
        return [
            f"Run the {base} workflow on this input and return the result",
            f"Use a helper tool to process this {base} task for me",
            f"Automate this {base} job and tell me what files were produced",
        ]
    if tool_kind == "hybrid":
        return [
            f"Help me reason through this {base} task and use tools if needed",
            f"Use the {base} skill to analyze the request and produce the final output",
            f"Figure out the right workflow for this {base} job, then execute it",
        ]
    return [
        f"Help me handle this {base} task",
        f"What is the right workflow for this {base} request?",
        f"Guide me through this {base} problem step by step",
    ]


def _normalize_example_requests(
    example_requests: list[str] | tuple[str, ...] | None,
    *,
    skill_title: str,
    tool_kind: str,
) -> list[str]:
    items = [str(item).strip() for item in (example_requests or ()) if str(item).strip()]
    if items:
        return items
    return _default_example_requests(skill_title, tool_kind)


def _default_resources_for_tool_kind(tool_kind: str) -> list[str]:
    if tool_kind == "scripts":
        return ["scripts"]
    if tool_kind == "hybrid":
        return ["scripts", "references"]
    return []


def _merge_resources(parsed_resources: list[str], tool_kind: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for resource in [*_default_resources_for_tool_kind(tool_kind), *parsed_resources]:
        if resource not in seen:
            merged.append(resource)
            seen.add(resource)
    return merged


def _when_to_use_line(summary_text: str) -> str:
    return f"Use when the request involves {summary_text}"


def _example_requests_block(example_requests: list[str]) -> str:
    return "\n".join(f"- {item}" for item in example_requests)


def _workflow_block(tool_kind: str) -> str:
    if tool_kind == "scripts":
        return "\n".join(
            (
                "1. Confirm the request matches this skill's trigger conditions.",
                "2. Use the public helper scripts for the repeatable or fragile steps.",
                "3. Return the result clearly, including any output paths or artifacts.",
            )
        )
    if tool_kind == "hybrid":
        return "\n".join(
            (
                "1. Confirm the request matches this skill's trigger conditions.",
                "2. Read only the needed references, then decide whether scripts should run.",
                "3. Use tools for execution-heavy steps and summarize the outcome clearly.",
            )
        )
    return "\n".join(
        (
            "1. Confirm the request matches this skill's trigger conditions.",
            "2. Read only the bundled references needed for the current task.",
            "3. Explain or execute the minimum necessary workflow and return the result clearly.",
        )
    )


def _resources_block(tool_kind: str) -> str:
    if tool_kind == "scripts":
        return "\n".join(
            (
                "- `scripts/`: required for deterministic helper functions that the runtime can register as tools.",
                "- `references/`: optional background material if the scripts need supporting docs.",
                "- `assets/`: optional templates or files used in outputs.",
            )
        )
    if tool_kind == "hybrid":
        return "\n".join(
            (
                "- `scripts/`: helper functions for execution-heavy or repetitive steps.",
                "- `references/`: detailed documentation loaded only when needed.",
                "- `assets/`: optional templates or supporting files used in outputs.",
            )
        )
    return "\n".join(
        (
            "- `scripts/`: add only if this skill later needs deterministic helper functions.",
            "- `references/`: preferred for deeper documentation and decision support.",
            "- `assets/`: optional templates or supporting files used in outputs.",
        )
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
    summary: str | None = None,
    example_requests: list[str] | tuple[str, ...] | None = None,
    tool_kind: str = "prompt",
    *,
    target: str = "workspace",
    workspace_skills_dir: str | Path | None = None,
    builtin_skills_dir: str | Path | None = None,
) -> Path:
    normalized_name = _normalize_skill_name(skill_name)
    skill_title = _title_case_skill_name(normalized_name)
    normalized_tool_kind = str(tool_kind or "prompt").strip().lower()
    if normalized_tool_kind not in ALLOWED_TOOL_KINDS:
        raise ValueError(
            f"tool_kind must be one of: {', '.join(sorted(ALLOWED_TOOL_KINDS))}"
        )
    include_examples = include_examples or any(
        item in EXAMPLE_RESOURCE_ALIASES for item in _resource_items(resources)
    )
    parsed_resources = _merge_resources(_parse_resources(resources), normalized_tool_kind)
    summary_text = _normalize_summary(summary, skill_title)
    example_request_items = _normalize_example_requests(
        example_requests,
        skill_title=skill_title,
        tool_kind=normalized_tool_kind,
    )

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
                description=(
                    f"Use when the request involves {summary_text}."
                ),
                when_to_use_line=_when_to_use_line(summary_text),
                example_requests_block=_example_requests_block(example_request_items),
                workflow_block=_workflow_block(normalized_tool_kind),
                resources_block=_resources_block(normalized_tool_kind),
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
    parser.add_argument("--summary", default="")
    parser.add_argument(
        "--example-request",
        action="append",
        default=[],
        dest="example_requests",
    )
    parser.add_argument(
        "--tool-kind",
        choices=tuple(sorted(ALLOWED_TOOL_KINDS)),
        default="prompt",
    )
    args = parser.parse_args()

    skill_dir = init_skill(
        args.skill_name,
        path=args.path,
        target=args.target,
        workspace_skills_dir=args.workspace_skills_dir,
        builtin_skills_dir=args.builtin_skills_dir,
        resources=args.resources,
        include_examples=args.examples,
        summary=args.summary,
        example_requests=args.example_requests,
        tool_kind=args.tool_kind,
    )
    print(skill_dir)


if __name__ == "__main__":
    main()
