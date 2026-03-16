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


def test_scheduled_task_tools_delegate_to_manager() -> None:
    class _TaskManager:
        def render_tasks(self) -> str:
            return '{"tasks":[]}'

        def create_task(self, **kwargs):
            return {"name": kwargs["name"], "schedule": kwargs["cron"]}

        def save_task(self, **kwargs):
            return {"name": kwargs.get("name") or "auto-name", "_action": "created"}

        def update_task(self, name: str, **kwargs):
            return {"name": name, "enabled": kwargs["enabled"]}

        def delete_task(self, name: str) -> bool:
            return name == "t1"

    manager = object.__new__(ResourceManager)
    manager.scheduled_task_manager = _TaskManager()

    assert manager.list_scheduled_tasks_tool()() == '{"tasks":[]}'
    created = manager.create_scheduled_task_tool()(
        name="t1",
        prompt="p",
        channel="feishu",
        chat_id="c1",
        cron="0 9 * * *",
    )
    assert '"name": "t1"' in created
    updated = manager.update_scheduled_task_tool()(name="t1", enabled=False)
    assert '"enabled": false' in updated
    saved = manager.save_scheduled_task_tool()(
        prompt="p",
        channel="feishu",
        chat_id="c1",
        interval_seconds=60,
    )
    assert '"_action": "created"' in saved
    deleted = manager.delete_scheduled_task_tool()(name="t1")
    assert '"deleted": true' in deleted


def test_save_scheduled_task_uses_channel_context_defaults() -> None:
    from babybot.channels.tools import ChannelToolContext

    class _TaskManager:
        def save_task(self, **kwargs):
            return {
                "name": "auto-name",
                "channel": kwargs["channel"],
                "chat_id": kwargs["chat_id"],
                "_action": "created",
            }

    manager = object.__new__(ResourceManager)
    manager.scheduled_task_manager = _TaskManager()
    ChannelToolContext.set_current(
        ChannelToolContext(channel_name="feishu", chat_id="oc_test", sender_id="u1")
    )
    try:
        saved = manager.save_scheduled_task_tool()(
            prompt="测试测试",
            cron="10 17 * * *",
        )
    finally:
        ChannelToolContext.set_current(None)

    assert '"channel": "feishu"' in saved
    assert '"chat_id": "oc_test"' in saved
