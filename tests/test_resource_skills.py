import asyncio
import contextvars
import json
from pathlib import Path
from types import SimpleNamespace

from babybot.agent_kernel import TaskResult, ToolLease
from babybot.agent_kernel.tools import ToolContext
from babybot.resource import CallableTool, LoadedSkill, ResourceManager, ToolGroup


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


def test_select_skill_packs_with_explicit_ids_filters_result() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "code-review": LoadedSkill(
            name="code-review",
            description="Review code",
            directory="/tmp/code-review",
            prompt="review prompt",
            active=True,
        ),
        "weather-query": LoadedSkill(
            name="weather-query",
            description="Weather skill",
            directory="/tmp/weather-query",
            prompt="weather prompt",
            active=True,
        ),
    }
    packs = asyncio.run(
        manager._select_skill_packs(
            "查天气",
            skill_ids=["skill.weather-query"],
        )
    )
    assert [pack.name for pack in packs] == ["weather-query"]


def test_resource_briefs_and_scope_resolution() -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {
        "basic": ToolGroup(name="basic", description="Core tools", active=True),
        "skill_weather_query": ToolGroup(
            name="skill_weather_query",
            description="Weather tools",
            active=True,
        ),
        "map_services": ToolGroup(
            name="map_services",
            description="Map MCP tools",
            active=True,
        ),
    }
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.mcp_server_groups = {"gaode_map": "map_services"}
    manager.skills = {
        "weather-query": LoadedSkill(
            name="weather-query",
            description="用于查询天气",
            directory="/tmp/weather-query",
            prompt="weather prompt",
            active=True,
            lease=ToolLease(include_groups=("skill_weather_query",)),
            tool_group="skill_weather_query",
            tools=("weather_query__fetch_weather",),
        )
    }

    def weather_query__fetch_weather(city_name: str) -> str:
        return city_name

    def gaode_map__search(keyword: str) -> str:
        return keyword

    manager.register_tool(
        weather_query__fetch_weather,
        group_name="skill_weather_query",
        func_name="weather_query__fetch_weather",
    )
    manager.register_tool(
        gaode_map__search,
        group_name="map_services",
        func_name="gaode_map__search",
    )

    briefs = manager.get_resource_briefs()
    ids = {item["id"] for item in briefs}
    assert "skill.weather-query" in ids
    assert "mcp.gaode-map" in ids

    scope = manager.resolve_resource_scope("skill.weather-query")
    assert scope is not None
    lease_dict, skill_ids = scope
    assert lease_dict["include_groups"] == ["skill_weather_query"]
    assert skill_ids == ("weather-query",)

    mcp_scope = manager.resolve_resource_scope("mcp.gaode-map")
    assert mcp_scope is not None
    mcp_lease, mcp_skills = mcp_scope
    assert mcp_lease["include_groups"] == ["map_services"]
    assert mcp_skills == ()

    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    unavailable_scope = manager.resolve_resource_scope(
        "skill.weather-query",
        require_tools=True,
    )
    assert unavailable_scope is None


def test_json_schema_for_callable_handles_collections_and_kwargs() -> None:
    def dispatch_workers(
        tasks: list[str],
        max_concurrency: int = 3,
        lease: dict[str, str] | None = None,
        **kwargs,
    ) -> str:
        return ""

    schema = ResourceManager._json_schema_for_callable(dispatch_workers)
    assert schema["properties"]["tasks"]["type"] == "array"
    assert schema["properties"]["max_concurrency"]["type"] == "integer"
    assert schema["properties"]["lease"]["type"] == "object"
    assert "kwargs" not in schema["properties"]


def test_run_subagent_task_returns_collected_media_from_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "pig.png"
    image_path.write_bytes(b"png")

    manager = object.__new__(ResourceManager)
    manager.config = SimpleNamespace(
        system=SimpleNamespace(context_history_tokens=2000),
        workspace_dir=tmp_path,
    )
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.groups = {}
    manager.skills = {}
    manager._shared_gateway = object()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_test",
        default=str(tmp_path),
    )
    manager._get_output_dir = lambda: tmp_path
    manager._build_task_lease = lambda lease: ToolLease()
    manager._build_worker_sys_prompt = lambda **kwargs: "sys"
    manager.get_shared_gateway = lambda: object()

    async def _select_skill_packs(task_description: str, skill_ids=None):
        del task_description, skill_ids
        return []

    manager._select_skill_packs = _select_skill_packs

    class _FakeExecutor:
        async def execute(self, task, context):
            del task
            context.state["media_paths_collected"] = [str(image_path.resolve())]
            return TaskResult(task_id="worker", status="succeeded", output="done")

    monkeypatch.setattr(
        "babybot.resource.create_worker_executor",
        lambda **kwargs: _FakeExecutor(),
    )

    text, media = asyncio.run(manager.run_subagent_task("draw a pig"))

    assert text == "done"
    assert media == [str(image_path.resolve())]


def test_run_subagent_task_propagates_channel_context_to_worker_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from babybot.channels.tools import ChannelToolContext

    manager = object.__new__(ResourceManager)
    manager.config = SimpleNamespace(
        system=SimpleNamespace(context_history_tokens=2000),
        workspace_dir=tmp_path,
    )
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.groups = {}
    manager.skills = {}
    manager._shared_gateway = object()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_test",
        default=str(tmp_path),
    )
    manager._get_output_dir = lambda: tmp_path
    manager._build_task_lease = lambda lease: ToolLease()
    manager._build_worker_sys_prompt = lambda **kwargs: "sys"
    manager.get_shared_gateway = lambda: object()

    async def _select_skill_packs(task_description: str, skill_ids=None):
        del task_description, skill_ids
        return []

    manager._select_skill_packs = _select_skill_packs
    seen: dict[str, object] = {}

    class _FakeExecutor:
        async def execute(self, task, context):
            del task
            seen["channel_context"] = context.state.get("channel_context")
            return TaskResult(task_id="worker", status="succeeded", output="done")

    monkeypatch.setattr(
        "babybot.resource.create_worker_executor",
        lambda **kwargs: _FakeExecutor(),
    )

    parent_ctx = ChannelToolContext(
        channel_name="feishu",
        chat_id="oc_test",
        sender_id="u1",
    )
    ChannelToolContext.set_current(parent_ctx)
    try:
        asyncio.run(manager.run_subagent_task("schedule something"))
    finally:
        ChannelToolContext.set_current(None)

    assert seen["channel_context"] is parent_ctx


def test_register_skill_tools_falls_back_to_proxy_when_import_fails(tmp_path: Path) -> None:
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
        "import definitely_missing_package\n"
        "def generate_image(prompt: str, output_path: str = 'a.jpg') -> str:\n"
        "    return output_path\n",
        encoding="utf-8",
    )

    group, tools = manager._register_skill_tools("text-to-image", skill_dir)
    assert group == "skill_text_to_image"
    assert any(name.endswith("__generate_image") for name in tools)
    reg = manager.registry.get("text_to_image__generate_image")
    assert reg is not None
    assert reg.tool.schema["properties"]["prompt"]["type"] == "string"


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


def test_save_scheduled_task_anchors_delay_to_request_received_time() -> None:
    from babybot.channels.tools import ChannelToolContext

    captured: dict[str, object] = {}

    class _TaskManager:
        def save_task(self, **kwargs):
            captured.update(kwargs)
            return {
                "name": "rain_art",
                "channel": kwargs["channel"],
                "chat_id": kwargs["chat_id"],
                "run_at": kwargs.get("run_at"),
                "_action": "created",
            }

    manager = object.__new__(ResourceManager)
    manager.scheduled_task_manager = _TaskManager()
    ChannelToolContext.set_current(
        ChannelToolContext(
            channel_name="feishu",
            chat_id="oc_test",
            sender_id="u1",
            metadata={"request_received_at": "2026-03-18T17:54:27+08:00"},
        )
    )
    try:
        manager.save_scheduled_task_tool()(
            prompt="生成杭州雨天图片",
            name="rain_art",
            delay_seconds=120,
        )
    finally:
        ChannelToolContext.set_current(None)

    assert captured["channel"] == "feishu"
    assert captured["chat_id"] == "oc_test"
    assert captured["delay_seconds"] is None
    assert captured["run_at"] == "2026-03-18T17:56:27+08:00"


def test_callable_tool_does_not_treat_long_json_text_as_artifact_path() -> None:
    payload = {
        "name": "杭州西湖画作描述消息",
        "prompt": "发送飞书消息：一幅描绘杭州西湖美景的画作",
        "schedule": {"run_at": "2026-03-19T11:41:23+08:00"},
        "target": {
            "channel": "feishu",
            "chat_id": "oc_cfcc4eb536ba962ffc1f58a3d4c51279",
        },
        "enabled": True,
        "_action": "created",
    }

    def return_long_json_text() -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    tool = CallableTool(
        func=return_long_json_text,
        name="return_long_json_text",
        description="return json text",
        schema={"type": "object", "properties": {}},
    )

    result = asyncio.run(tool.invoke({}, ToolContext(session_id="s1", state={})))

    assert result.ok is True
    assert '"_action": "created"' in result.content
    assert result.artifacts == []
