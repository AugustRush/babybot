import asyncio
from pathlib import Path

from babybot.agent_kernel import ToolLease
from babybot.resource import LoadedSkill, ResourceManager


def test_parse_frontmatter_from_skill_markdown() -> None:
    text = """---
name: sample-skill
description: test skill
---

# Title

Do something.
"""
    meta, body = ResourceManager._parse_frontmatter(text)
    assert meta["name"] == "sample-skill"
    assert meta["description"] == "test skill"
    assert "Do something." in body


def test_select_skill_packs_returns_all_active_skills() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "code-review": LoadedSkill(
            name="code-review",
            description="Review code",
            directory="/tmp/skills/code-review",
            prompt="review prompt",
            active=True,
        ),
        "text-to-image": LoadedSkill(
            name="text-to-image",
            description="Generate images",
            directory="/tmp/skills/text-to-image",
            prompt="image prompt",
            active=True,
        ),
        "disabled-skill": LoadedSkill(
            name="disabled-skill",
            description="disabled",
            directory="/tmp/skills/disabled",
            prompt="disabled prompt",
            active=False,
        ),
    }
    packs = asyncio.run(manager._select_skill_packs("请使用 $text-to-image 画一只兔子"))
    names = sorted(p.name for p in packs)
    assert names == ["code-review", "text-to-image"]


def test_tokenize_supports_cjk_and_latin_terms() -> None:
    terms = ResourceManager._tokenize("画一只机械小鸟 and generate image")
    assert "generate" in terms
    assert "image" in terms
    assert "机械" in terms


def test_build_worker_prompt_contains_skill_catalog() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "code-review": LoadedSkill(
            name="code-review",
            description="Review code quality",
            directory="/tmp/skills/code-review",
            prompt="You are code reviewer.",
            active=True,
        ),
        "data-analysis": LoadedSkill(
            name="data-analysis",
            description="Analyze datasets",
            directory="/tmp/skills/data-analysis",
            prompt="You are data analyst.",
            active=True,
        ),
    }
    prompt = manager._build_worker_sys_prompt(
        agent_name="Worker",
        task_description="你可以做什么？",
        tools_text="execute_shell_command, view_text_file",
        selected_skill_packs=[],
    )
    assert "可用技能目录" in prompt
    assert "code-review: Review code quality" in prompt
    assert "data-analysis: Analyze datasets" in prompt


def test_coerce_timeout_handles_invalid_values() -> None:
    assert ResourceManager._coerce_timeout(None) == 300.0
    assert ResourceManager._coerce_timeout("12") == 12.0
    assert ResourceManager._coerce_timeout("bad") == 300.0
    assert ResourceManager._coerce_timeout(-1) == 300.0


def test_register_skill_tools_from_scripts(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {"resolve_workspace_path": staticmethod(lambda value: str(value))},
    )()
    skill_dir = tmp_path / "text_to_image"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "tool_impl.py").write_text(
        "def generate_image(prompt: str) -> str:\n    return f'img:{prompt}'\n",
        encoding="utf-8",
    )
    group, tools = manager._register_skill_tools("text-to-image", skill_dir)
    assert group == "skill_text_to_image"
    assert any(name.endswith("__generate_image") for name in tools)


def test_select_skill_packs_loads_all_active_skills() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "a": LoadedSkill(
            name="a",
            description="A",
            directory="/tmp/a",
            prompt="A",
            active=True,
        ),
        "b": LoadedSkill(
            name="b",
            description="B",
            directory="/tmp/b",
            prompt="B",
            active=True,
        ),
        "c": LoadedSkill(
            name="c",
            description="C",
            directory="/tmp/c",
            prompt="C",
            active=True,
        ),
    }

    packs = asyncio.run(manager._select_skill_packs("draw an image"))
    assert sorted(p.name for p in packs) == ["a", "b", "c"]
