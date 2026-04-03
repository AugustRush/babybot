import importlib
import sys
from pathlib import Path


SCRIPT_DIR = Path("skills/skill-manager/scripts").resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

init_skill = importlib.import_module("init_skill")
quick_validate = importlib.import_module("quick_validate")


def test_init_skill_creates_workspace_skill_at_default_root(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "Demo Skill",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["scripts", "references"],
        include_examples=False,
    )

    assert skill_dir == tmp_path / "workspace" / "skills" / "demo-skill"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "scripts").is_dir()
    assert (skill_dir / "references").is_dir()
    assert not (skill_dir / "assets").exists()


def test_init_skill_creates_builtin_skill_at_default_root(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "builtin helper",
        target="builtin",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["assets"],
        include_examples=False,
    )

    assert skill_dir == tmp_path / "builtin" / "skills" / "builtin-helper"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "assets").is_dir()


def test_parse_resources_rejects_unknown_resource_types() -> None:
    try:
        init_skill._parse_resources("scripts,unknown")
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError(
            "Expected parse_resources to exit on invalid resource type"
        )


def test_parse_resources_ignores_examples_alias() -> None:
    resources = init_skill._parse_resources("scripts,examples,references")

    assert resources == ["scripts", "references"]


def test_parse_resources_accepts_json_stringified_array_items() -> None:
    resources = init_skill._parse_resources(['["scripts"]', "references"])

    assert resources == ["scripts", "references"]


def test_parse_resources_accepts_common_resource_aliases() -> None:
    resources = init_skill._parse_resources(["code", "docs", "template"])

    assert resources == ["scripts", "references", "assets"]


def test_init_skill_accepts_json_stringified_resource_array_items(
    tmp_path: Path,
) -> None:
    skill_dir = init_skill.init_skill(
        "json-array-skill",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=['["scripts"]'],
        include_examples=False,
    )

    assert (skill_dir / "scripts").is_dir()


def test_init_skill_treats_examples_resource_alias_as_include_examples(
    tmp_path: Path,
) -> None:
    skill_dir = init_skill.init_skill(
        "glm ocr",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["scripts", "examples"],
        include_examples=False,
    )

    assert (skill_dir / "scripts" / "_example.py").exists()


def test_init_skill_removes_placeholder_examples_on_rerun_without_examples(
    tmp_path: Path,
) -> None:
    skill_dir = init_skill.init_skill(
        "cleanup skill",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["scripts", "references", "assets"],
        include_examples=True,
    )

    assert (skill_dir / "scripts" / "_example.py").exists()
    assert (skill_dir / "references" / "reference.md").exists()
    assert (skill_dir / "assets" / "example.txt").exists()

    skill_dir = init_skill.init_skill(
        "cleanup skill",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["scripts", "references", "assets"],
        include_examples=False,
    )

    assert not (skill_dir / "scripts" / "_example.py").exists()
    assert not (skill_dir / "references" / "reference.md").exists()
    assert not (skill_dir / "assets" / "example.txt").exists()


def test_validate_skill_accepts_generated_skill(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "validator-skill",
        path=tmp_path,
        resources=[],
        include_examples=False,
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert valid, message


def test_init_skill_generates_guideline_aligned_skill_document(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "validator-skill",
        path=tmp_path,
        resources=[],
        include_examples=False,
    )

    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    assert "description: Use when " in content
    assert "Use this skill when" not in content
    assert "## When to Use" in content
    assert "## Example Requests" in content
    assert "## Workflow" in content
    assert "## Resources" in content
    assert "## Constraints" in content
    assert "- " in content


def test_init_skill_supports_summary_examples_and_tool_kind_defaults(
    tmp_path: Path,
) -> None:
    skill_dir = init_skill.init_skill(
        "receipt parser",
        path=tmp_path,
        resources=[],
        include_examples=False,
        summary="extracting structured fields from receipt photos and scanned invoices",
        example_requests=[
            "帮我提取这张小票里的商家、金额和日期",
            "Parse this receipt image into JSON",
        ],
        tool_kind="scripts",
    )

    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    assert (
        "extracting structured fields from receipt photos and scanned invoices"
        in content
    )
    assert "帮我提取这张小票里的商家、金额和日期" in content
    assert "Parse this receipt image into JSON" in content
    assert (skill_dir / "scripts").is_dir()


def test_init_skill_hybrid_kind_creates_scripts_and_references_by_default(
    tmp_path: Path,
) -> None:
    skill_dir = init_skill.init_skill(
        "finance helper",
        path=tmp_path,
        resources=[],
        include_examples=False,
        tool_kind="hybrid",
    )

    assert (skill_dir / "scripts").is_dir()
    assert (skill_dir / "references").is_dir()


def test_validate_skill_rejects_todo_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        '---\nname: bad-skill\ndescription: "[TODO: fill me in]"\n---\n\n# Bad Skill\n',
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "TODO" in message


def test_validate_skill_rejects_placeholder_skill_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "placeholder-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: placeholder-skill\n"
        "description: Use when handling placeholder skill tasks.\n"
        "---\n\n"
        "# Placeholder Skill\n\n"
        "## Overview\n\n"
        "Describe what this skill enables and when it should be used.\n\n"
        "## Workflow\n\n"
        "1. Explain how the skill should be triggered.\n"
        "2. List the core steps the agent should follow.\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "placeholder" in message.lower()


def test_validate_skill_rejects_generated_placeholder_resources(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "placeholder-resources",
        path=tmp_path,
        resources=["scripts", "references", "assets"],
        include_examples=True,
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "placeholder resource" in message.lower()


def test_validate_skill_rejects_missing_example_requests_section(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "missing-examples-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: missing-examples-skill\n"
        "description: Use when handling receipt extraction and invoice parsing requests.\n"
        "---\n\n"
        "# Missing Examples Skill\n\n"
        "## When to Use\n\n"
        "- Use when a user asks to extract receipt data.\n\n"
        "## Workflow\n\n"
        "1. Inspect the input files.\n"
        "2. Extract the requested fields.\n\n"
        "## Resources\n\n"
        "- `scripts/`: OCR helpers.\n\n"
        "## Constraints\n\n"
        "- Return structured data.\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "Example Requests" in message


def test_validate_skill_rejects_unexpected_root_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-root-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bad-root-skill\n"
        "description: Use when handling bad root skill requests.\n"
        "---\n\n"
        "# Skill\n\n"
        "## When to Use\n\n"
        "- Use when a bad root skill request needs handling.\n\n"
        "## Example Requests\n\n"
        "- Help me validate this bad root skill.\n\n"
        "## Workflow\n\n"
        "1. Check the files.\n"
        "2. Return the result.\n\n"
        "## Resources\n\n"
        "- `references/`: Optional supporting docs.\n\n"
        "## Constraints\n\n"
        "- Keep the root clean.\n",
        encoding="utf-8",
    )
    (skill_dir / "README.md").write_text("extra\n", encoding="utf-8")

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "Unexpected file or directory" in message


def test_validate_skill_accepts_current_skill_manager_folder_naming() -> None:
    valid, message = quick_validate.validate_skill(
        Path("skills/skill-manager").resolve()
    )

    assert valid, message


def test_validate_skill_accepts_non_english_description_prefix(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bilingual-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bilingual-skill\n"
        "description: 适用于处理发票识别和小票抽取请求。\n"
        "---\n\n"
        "# Bilingual Skill\n\n"
        "## Example Requests\n\n"
        "- 帮我提取这张发票里的金额和日期。\n\n"
        "## Workflow\n\n"
        "1. 读取输入。\n"
        "2. 返回结构化结果。\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert valid, message


def test_validate_skill_allows_common_noise_files_in_root(tmp_path: Path) -> None:
    skill_dir = tmp_path / "noise-ok-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: noise-ok-skill\n"
        "description: Use when handling noise-tolerant skill requests.\n"
        "---\n\n"
        "# Noise OK Skill\n\n"
        "## Example Requests\n\n"
        "- Validate this skill folder.\n\n"
        "## Workflow\n\n"
        "1. Inspect the files.\n"
        "2. Return the result.\n",
        encoding="utf-8",
    )
    (skill_dir / ".DS_Store").write_text("x", encoding="utf-8")
    (skill_dir / "__pycache__").mkdir()

    valid, message = quick_validate.validate_skill(skill_dir)

    assert valid, message


def test_skill_manager_skill_forbids_workspace_output_artifacts() -> None:
    content = Path("skills/skill-manager/SKILL.md").read_text(encoding="utf-8")

    assert "/workspace/output" in content
    # The skill must explicitly forbid writing to workspace/output
    assert "workspace/output" in content.lower()


def test_validate_skill_rejects_public_cli_only_script(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "cli-only-skill",
        path=tmp_path,
        resources=["scripts"],
        include_examples=False,
    )
    (skill_dir / "scripts" / "generate_image.py").write_text(
        "import argparse\n"
        "def parse_arguments():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-p', '--prompt', required=True)\n"
        "    return parser.parse_args()\n\n"
        "def main():\n"
        "    args = parse_arguments()\n"
        "    print(args.prompt)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "public callable function" in message


def test_validate_skill_accepts_public_callable_script(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "callable-skill",
        path=tmp_path,
        resources=["scripts"],
        include_examples=False,
    )
    (skill_dir / "scripts" / "generate_image.py").write_text(
        "def generate_image(prompt: str) -> str:\n    return f'img:{prompt}'\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert valid, message
