import asyncio
import ast
import contextvars
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import importlib

from babybot.agent_kernel import SkillPack, TaskResult, ToolLease
from babybot.agent_kernel.tools import ToolContext
from babybot.resource import CallableTool, LoadedSkill, ResourceManager, ToolGroup
from babybot.config import Config


_AUTO_SKILL_CREATOR_SCRIPTS = Path("skills/auto_skill_creator/scripts").resolve()
if str(_AUTO_SKILL_CREATOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_AUTO_SKILL_CREATOR_SCRIPTS))
auto_init_skill = importlib.import_module("init_skill")
from babybot.builtin_tools.workers import build_create_worker_tool, build_dispatch_workers_tool


def test_resource_manager_exposes_expected_registration_and_runtime_entrypoints() -> None:
    assert hasattr(ResourceManager, "_register_skill_tools")
    assert hasattr(ResourceManager, "_invoke_external_skill_function")
    assert hasattr(ResourceManager, "_build_external_cli_script_callable")
    assert hasattr(ResourceManager, "_run_subagent_task")


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


def test_parse_frontmatter_supports_colons_and_yaml_lists() -> None:
    text = """---
name: sample-skill
description: "Use when: parsing data"
include_groups:
  - code
  - browser
---

# Title

Do something.
"""
    meta, body = ResourceManager._parse_frontmatter(text)
    assert meta["description"] == "Use when: parsing data"
    assert meta["include_groups"] == ["code", "browser"]
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
    assert names == ["text-to-image"]


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


def test_format_skill_catalog_for_lease_filters_inaccessible_skills() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "prompt-only": LoadedSkill(
            name="prompt-only",
            description="No tool lease",
            directory="/tmp/prompt-only",
            prompt="prompt-only",
            active=True,
        ),
        "weather-query": LoadedSkill(
            name="weather-query",
            description="Weather tools",
            directory="/tmp/weather-query",
            prompt="weather",
            active=True,
            lease=ToolLease(include_groups=("skill_weather_query",)),
        ),
        "image-gen": LoadedSkill(
            name="image-gen",
            description="Image tools",
            directory="/tmp/image-gen",
            prompt="image",
            active=True,
            lease=ToolLease(include_groups=("skill_image_gen",)),
        ),
    }

    catalog = manager._format_skill_catalog_for_lease(
        ToolLease(include_groups=("skill_weather_query",))
    )

    assert "prompt-only: No tool lease" in catalog
    assert "weather-query: Weather tools" in catalog
    assert "image-gen: Image tools" not in catalog


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


def test_register_skill_tools_for_auto_skill_creator_exposes_only_agent_facing_tools() -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {"resolve_workspace_path": staticmethod(lambda value: str(value))},
    )()

    skill_dir = Path("skills/auto_skill_creator").resolve()

    group, tools = manager._register_skill_tools("auto-skill-creator", skill_dir)

    assert group == "skill_auto_skill_creator"
    assert tools == (
        "auto_skill_creator__init_skill",
        "auto_skill_creator__validate_skill",
    )


def test_discovered_generated_skill_uses_example_requests_for_keywords(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_skills_dir = workspace_dir / "skills"
    builtin_skills_dir = tmp_path / "builtin" / "skills"
    workspace_skills_dir.mkdir(parents=True, exist_ok=True)
    builtin_skills_dir.mkdir(parents=True, exist_ok=True)

    auto_init_skill.init_skill(
        "receipt parser",
        target="workspace",
        workspace_skills_dir=workspace_skills_dir,
        builtin_skills_dir=builtin_skills_dir,
        resources=[],
        include_examples=False,
        summary="extracting fields from receipts and invoices",
        example_requests=[
            "帮我提取这张小票里的金额和日期",
            "parse this invoice photo into json",
        ],
        tool_kind="prompt",
    )

    cfg = Config()
    cfg.workspace_dir = workspace_dir
    cfg.workspace_skills_dir = workspace_skills_dir
    cfg.builtin_skills_dir = builtin_skills_dir
    cfg.workspace_tools_dir = workspace_dir / "tools"
    cfg.scheduled_tasks_file = workspace_dir / "scheduled_tasks.json"

    manager = ResourceManager(cfg)
    skill = manager.skills["receipt-parser"]

    assert "receipt" in skill.keywords
    assert "invoice" in skill.keywords
    assert "小票" in skill.keywords or "金额" in skill.keywords


def test_discovered_skill_frontmatter_can_extend_include_groups(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_skills_dir = workspace_dir / "skills"
    builtin_skills_dir = tmp_path / "builtin" / "skills"
    skill_dir = workspace_skills_dir / "creator-helper"
    skill_dir.mkdir(parents=True, exist_ok=True)
    builtin_skills_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: creator-helper\n"
        "description: Use when creating helper skills that also need code tools.\n"
        "include_groups: code\n"
        "---\n\n"
        "# Creator Helper\n\n"
        "## When to Use\n\n"
        "- Use when creating helper skills that also need code tools.\n\n"
        "## Example Requests\n\n"
        "- Create a helper skill from this GitHub code sample.\n\n"
        "## Workflow\n\n"
        "1. Read the references.\n"
        "2. Write the needed files.\n\n"
        "## Resources\n\n"
        "- `scripts/`: optional helper tools.\n\n"
        "## Constraints\n\n"
        "- Keep the skill focused.\n",
        encoding="utf-8",
    )

    cfg = Config()
    cfg.workspace_dir = workspace_dir
    cfg.workspace_skills_dir = workspace_skills_dir
    cfg.builtin_skills_dir = builtin_skills_dir
    cfg.workspace_tools_dir = workspace_dir / "tools"
    cfg.scheduled_tasks_file = workspace_dir / "scheduled_tasks.json"

    manager = ResourceManager(cfg)
    skill = manager.skills["creator-helper"]

    assert skill.lease.include_groups == ("code",)


def test_workspace_skill_overrides_builtin_skill_with_same_name(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_skills_dir = workspace_dir / "skills"
    builtin_skills_dir = tmp_path / "builtin" / "skills"
    workspace_skill_dir = workspace_skills_dir / "shared-skill"
    builtin_skill_dir = builtin_skills_dir / "shared-skill"
    workspace_skill_dir.mkdir(parents=True, exist_ok=True)
    builtin_skill_dir.mkdir(parents=True, exist_ok=True)

    (builtin_skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: shared-skill\n"
        "description: Use when loading the builtin variant.\n"
        "---\n\n"
        "# Builtin Shared Skill\n\n"
        "## Example Requests\n\n"
        "- Use the builtin version.\n",
        encoding="utf-8",
    )
    (workspace_skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: shared-skill\n"
        "description: Use when loading the workspace override.\n"
        "---\n\n"
        "# Workspace Shared Skill\n\n"
        "## Example Requests\n\n"
        "- Use the workspace version.\n",
        encoding="utf-8",
    )

    cfg = Config()
    cfg.workspace_dir = workspace_dir
    cfg.workspace_skills_dir = workspace_skills_dir
    cfg.builtin_skills_dir = builtin_skills_dir
    cfg.workspace_tools_dir = workspace_dir / "tools"
    cfg.scheduled_tasks_file = workspace_dir / "scheduled_tasks.json"

    manager = ResourceManager(cfg)
    skill = manager.skills["shared-skill"]

    assert skill.description == "Use when loading the workspace override."
    assert Path(skill.directory) == workspace_skill_dir.resolve()


def test_register_skill_tools_avoids_import_side_effects_for_function_scripts(
    tmp_path: Path,
) -> None:
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
    side_effect_path = tmp_path / "imported.txt"
    (scripts / "tool_impl.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(side_effect_path)!r}).write_text('imported', encoding='utf-8')\n"
        "def generate_image(prompt: str) -> str:\n"
        "    return f'img:{prompt}'\n",
        encoding="utf-8",
    )

    group, tools = manager._register_skill_tools("text-to-image", skill_dir)

    assert group == "skill_text_to_image"
    assert "text_to_image__generate_image" in tools
    assert not side_effect_path.exists()


def test_register_skill_tools_skips_cli_main_function(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {"resolve_workspace_path": staticmethod(lambda value: str(value))},
    )()
    skill_dir = tmp_path / "ocr_helper"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "tool_impl.py").write_text(
        "def recognize_text(file_path: str) -> str:\n"
        "    return file_path\n\n"
        "def main() -> None:\n"
        "    raise SystemExit(0)\n",
        encoding="utf-8",
    )

    group, tools = manager._register_skill_tools("ocr-helper", skill_dir)

    assert group == "skill_ocr_helper"
    assert "ocr_helper__recognize_text" in tools
    assert "ocr_helper__main" not in tools
    assert manager.registry.get("ocr_helper__main") is None


def test_register_skill_tools_registers_cli_script_proxy(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {"resolve_workspace_path": staticmethod(lambda value: str(value))},
    )()
    skill_dir = tmp_path / "image_cli"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "generate_image.py").write_text(
        "import argparse\n"
        "def parse_arguments():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-p', '--prompt', required=True)\n"
        "    parser.add_argument('--count', type=int, default=1)\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    return parser.parse_args()\n\n"
        "def main():\n"
        "    args = parse_arguments()\n"
        "    print(f'{args.prompt}|{args.count}|{args.verbose}')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    group, tools = manager._register_skill_tools("image-cli", skill_dir)

    assert group == "skill_image_cli"
    assert "image_cli__generate_image" in tools
    assert "image_cli__parse_arguments" not in tools
    reg = manager.registry.get("image_cli__generate_image")
    assert reg is not None
    assert reg.tool.schema["properties"]["prompt"]["type"] == "string"
    assert reg.tool.schema["properties"]["count"]["type"] == "integer"
    assert reg.tool.schema["properties"]["verbose"]["type"] == "boolean"


def test_cli_script_proxy_invokes_script_with_named_arguments(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {
            "resolve_workspace_path": staticmethod(lambda value: str(value)),
            "workspace_dir": tmp_path,
            "system": SimpleNamespace(shell_command_timeout=300, python_executable="python3"),
        },
    )()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_cli_proxy",
        default=str(tmp_path),
    )
    skill_dir = tmp_path / "image_cli"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "generate_image.py").write_text(
        "import argparse\n"
        "def parse_arguments():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-p', '--prompt', required=True)\n"
        "    parser.add_argument('--count', type=int, default=1)\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    return parser.parse_args()\n\n"
        "def main():\n"
        "    args = parse_arguments()\n"
        "    print(f'{args.prompt}|{args.count}|{args.verbose}')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    manager._register_skill_tools("image-cli", skill_dir)
    reg = manager.registry.get("image_cli__generate_image")

    result = asyncio.run(
        reg.tool.invoke(  # type: ignore[union-attr]
            {"prompt": "bird", "count": 2, "verbose": True},
            ToolContext(session_id="test", state={}),
        )
    )

    assert result.ok is True
    assert "bird|2|True" in result.content


def test_cli_script_proxy_retries_with_fallback_python_on_env_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.config = type(
        "DummyConfig",
        (),
        {
            "resolve_workspace_path": staticmethod(lambda value: str(value)),
            "workspace_dir": tmp_path,
            "system": SimpleNamespace(
                shell_command_timeout=300,
                python_executable="",
                python_fallback_executables=[],
            ),
        },
    )()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_cli_proxy_fallback",
        default=str(tmp_path),
    )
    skill_dir = tmp_path / "image_cli"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "generate_image.py").write_text(
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

    monkeypatch.setattr(manager, "_get_python_candidates", lambda skill_runtime=None: [
        {
            "executable": "/broken/python",
            "required_modules": (),
            "source": "skill.python_executable",
        },
        {
            "executable": "/healthy/python",
            "required_modules": (),
            "source": "skill.python_fallback_executables",
        },
    ])
    monkeypatch.setattr(manager, "_probe_python_candidate", lambda candidate: None)

    calls: list[str] = []

    class _Proc:
        def __init__(self, returncode: int, stdout: bytes, stderr: bytes):
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        async def communicate(self):
            return self._stdout, self._stderr

        def kill(self) -> None:
            return None

    async def _fake_exec(program, *args, **kwargs):
        del args, kwargs
        calls.append(program)
        if program == "/broken/python":
            return _Proc(1, b"", b"ModuleNotFoundError: No module named 'PIL'\n")
        return _Proc(0, b"bird\n", b"")

    monkeypatch.setattr("babybot.resource.asyncio.create_subprocess_exec", _fake_exec)

    manager._register_skill_tools(
        "image-cli",
        skill_dir,
        runtime=ResourceManager._build_skill_runtime(
            {
                "python_executable": "/broken/python",
                "python_fallback_executables": ["/healthy/python"],
            }
        ),
    )
    reg = manager.registry.get("image_cli__generate_image")

    result = asyncio.run(
        reg.tool.invoke(  # type: ignore[union-attr]
            {"prompt": "bird"},
            ToolContext(session_id="test", state={}),
        )
    )

    assert result.ok is True
    assert result.content == "bird"
    assert calls == ["/broken/python", "/healthy/python"]


def test_get_user_python_prefers_path_python3_over_hardcoded_system_python(tmp_path: Path, monkeypatch) -> None:
    manager = object.__new__(ResourceManager)
    manager.config = type(
        "DummyConfig",
        (),
        {
            "system": SimpleNamespace(shell_command_timeout=300, python_executable=""),
            "workspace_dir": tmp_path,
        },
    )()
    monkeypatch.setattr("babybot.resource.shutil.which", lambda name: {
        "python3": "/Users/test/miniconda3/bin/python3",
        "python": "/Users/test/miniconda3/bin/python",
    }.get(name))
    monkeypatch.setattr("babybot.resource.os.path.isfile", lambda path: path == "/usr/bin/python3")
    monkeypatch.setattr("babybot.resource.os.access", lambda path, mode: path == "/usr/bin/python3")

    assert manager._get_user_python() == "/Users/test/miniconda3/bin/python3"


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


def test_select_skill_packs_prefers_keyword_and_phrase_matches() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {
        "weather-query": LoadedSkill(
            name="weather-query",
            description="Check weather forecasts",
            directory="/tmp/weather-query",
            prompt="weather prompt",
            active=True,
            keywords=("weather", "forecast", "天气"),
            phrases=("weather forecast", "查天气"),
        ),
        "image-gen": LoadedSkill(
            name="image-gen",
            description="Generate images",
            directory="/tmp/image-gen",
            prompt="image prompt",
            active=True,
            keywords=("image", "draw", "画图"),
            phrases=("generate image",),
        ),
    }

    packs = asyncio.run(manager._select_skill_packs("请帮我查天气并给出 weather forecast"))

    assert [pack.name for pack in packs] == ["weather-query"]


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


def test_search_resources_filters_groups_tools_skills_and_mcp_servers() -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {
        "basic": ToolGroup(name="basic", description="Core tools", active=True),
        "map_services": ToolGroup(
            name="map_services",
            description="Map MCP tools",
            active=True,
        ),
    }
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.mcp_server_groups = {"gaode_map": "map_services"}
    manager.mcp_clients = {"gaode_map": object()}
    manager.skills = {
        "weather-query": LoadedSkill(
            name="weather-query",
            description="用于查询天气",
            directory="/tmp/weather-query",
            prompt="weather prompt",
            active=True,
        )
    }

    def gaode_map__search(keyword: str) -> str:
        return keyword

    manager.register_tool(
        gaode_map__search,
        group_name="map_services",
        func_name="gaode_map__search",
    )

    result = manager.search_resources("map")

    assert result["query"] == "map"
    assert result["mcp_servers"] == ["gaode_map"]


def test_get_resource_briefs_keeps_prompt_only_skill_active_without_unrelated_tools() -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {
        "basic": ToolGroup(name="basic", description="Core tools", active=True),
    }
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    manager.mcp_server_groups = {}
    manager.skills = {
        "prompt-helper": LoadedSkill(
            name="prompt-helper",
            description="Use when a prompt-only helper skill is needed.",
            directory="/tmp/prompt-helper",
            prompt="prompt helper prompt",
            active=True,
            lease=ToolLease(),
        )
    }

    def list_files(path: str) -> str:
        return path

    manager.register_tool(
        list_files,
        group_name="basic",
        func_name="list_files",
    )

    briefs = manager.get_resource_briefs()
    prompt_brief = next(item for item in briefs if item["id"] == "skill.prompt-helper")

    assert prompt_brief["active"] is True
    assert prompt_brief["tool_count"] == 0
    assert prompt_brief["tools_preview"] == []


def test_get_resource_briefs_reuses_single_registry_snapshot() -> None:
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
        )
    }

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Registered:
        def __init__(self, name: str, group: str) -> None:
            self.tool = _Tool(name)
            self.group = group

    class _Registry:
        def __init__(self) -> None:
            self.calls = 0
            self.items = [
                _Registered("weather_query__fetch_weather", "skill_weather_query"),
                _Registered("gaode_map__search", "map_services"),
                _Registered("list_files", "basic"),
            ]

        def list(self, lease=None):
            self.calls += 1
            if lease is None:
                return list(self.items)
            include_groups = set(getattr(lease, "include_groups", ()) or ())
            if include_groups:
                return [item for item in self.items if item.group in include_groups]
            return list(self.items)

    manager.registry = _Registry()

    briefs = manager.get_resource_briefs()

    assert briefs
    assert manager.registry.calls == 1


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


def test_schema_for_ast_annotation_does_not_misclassify_literal_strings() -> None:
    node = ast.parse("def demo(value: Literal['internal']):\n    pass\n").body[0]

    schema = ResourceManager._schema_for_ast_annotation(node.args.args[0].annotation)

    assert schema["type"] == "string"


def test_load_tool_module_raises_when_script_calls_sys_exit(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    script_path = tmp_path / "exit_tool.py"
    script_path.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    manager.config = SimpleNamespace(resolve_workspace_path=lambda value: str(value))

    try:
        manager._load_tool_module(str(script_path))
    except ModuleNotFoundError as exc:
        assert "called sys.exit() during import" in str(exc)
    else:
        raise AssertionError("expected ModuleNotFoundError")


def test_discover_workspace_tools_registers_public_functions(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.groups = {}
    manager.registry = __import__("babybot.agent_kernel", fromlist=["ToolRegistry"]).ToolRegistry()
    tools_root = tmp_path / "tools"
    analysis_dir = tools_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "demo.py").write_text(
        "def make_summary(topic: str) -> str:\n"
        "    return topic\n\n"
        "def _hidden() -> str:\n"
        "    return 'hidden'\n",
        encoding="utf-8",
    )
    manager.config = SimpleNamespace(
        workspace_dir=tmp_path,
        workspace_tools_dir=tools_root,
        resolve_workspace_path=lambda value: str(tmp_path / value),
    )

    manager._discover_workspace_tools()

    registered = manager.registry.get("make_summary")
    assert registered is not None
    assert registered.group == "analysis"
    assert manager.registry.get("_hidden") is None


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


def test_run_subagent_task_merges_skill_leases_before_executor(monkeypatch, tmp_path: Path) -> None:
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
        "active_write_root_test_merge_lease",
        default=str(tmp_path),
    )
    manager._get_output_dir = lambda: tmp_path
    manager._build_task_lease = lambda lease: ToolLease(
        include_groups=("basic",),
        include_tools=("regular_tool",),
        exclude_tools=("create_worker",),
    )
    manager._build_worker_sys_prompt = lambda **kwargs: "sys"
    manager.get_shared_gateway = lambda: object()

    async def _select_skill_packs(task_description: str, skill_ids=None):
        del task_description, skill_ids
        return [
            SkillPack(
                name="weather-query",
                system_prompt="weather",
                tool_lease=ToolLease(
                    include_groups=("skill_weather_query",),
                    include_tools=("weather_query__fetch_weather",),
                    exclude_tools=("dispatch_workers",),
                ),
            )
        ]

    manager._select_skill_packs = _select_skill_packs
    captured: dict[str, object] = {}

    class _FakeExecutor:
        async def execute(self, task, context):
            del context
            captured["lease"] = task.lease
            return TaskResult(task_id="worker", status="succeeded", output="done")

    def _fake_create_worker_executor(**kwargs):
        captured["skill_packs"] = kwargs["skill_packs"]
        return _FakeExecutor()

    monkeypatch.setattr("babybot.resource.create_worker_executor", _fake_create_worker_executor)

    text, media = asyncio.run(manager.run_subagent_task("check weather"))

    assert text == "done"
    assert media == []
    merged_lease = captured["lease"]
    assert isinstance(merged_lease, ToolLease)
    assert merged_lease.include_groups == ("basic", "skill_weather_query")
    assert merged_lease.include_tools == ("regular_tool", "weather_query__fetch_weather")
    assert merged_lease.exclude_tools == ("create_worker", "dispatch_workers")

    executor_skill_packs = captured["skill_packs"]
    assert isinstance(executor_skill_packs, list)
    assert [pack.name for pack in executor_skill_packs] == ["weather-query"]
    assert all(pack.tool_lease == ToolLease() for pack in executor_skill_packs)


def test_create_worker_tool_enforces_max_depth() -> None:
    class _Owner:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system=SimpleNamespace(worker_max_depth=2))
            self._lease_var = contextvars.ContextVar("lease_var", default=None)
            self._skill_ids_var = contextvars.ContextVar("skill_ids_var", default=None)
            self._worker_depth_var = contextvars.ContextVar("worker_depth_var", default=2)
            self.called = False

        def _get_current_task_lease_var(self):
            return self._lease_var

        def _get_current_skill_ids_var(self):
            return self._skill_ids_var

        def _get_current_worker_depth_var(self):
            return self._worker_depth_var

        async def run_subagent_task(self, *args, **kwargs):
            del args, kwargs
            self.called = True
            return "done", []

    owner = _Owner()
    tool = build_create_worker_tool(owner)

    result = asyncio.run(tool("nested task"))

    assert "max worker depth" in result.lower()
    assert owner.called is False


def test_dispatch_workers_tool_applies_timeout_to_hung_subtasks() -> None:
    class _Owner:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system=SimpleNamespace(worker_subtask_timeout=0.01))
            self._lease_var = contextvars.ContextVar("lease_var", default=None)
            self._skill_ids_var = contextvars.ContextVar("skill_ids_var", default=None)

        def _get_current_task_lease_var(self):
            return self._lease_var

        def _get_current_skill_ids_var(self):
            return self._skill_ids_var

        async def run_subagent_task(self, *args, **kwargs):
            del args, kwargs
            await asyncio.sleep(60)
            return "done", []

    owner = _Owner()
    tool = build_dispatch_workers_tool(owner)

    payload = json.loads(asyncio.run(tool(["slow task"])))

    assert payload["results"][0]["error"].lower().startswith("timeout")


def test_run_subagent_task_includes_current_channel_scope_for_live_delivery(monkeypatch, tmp_path: Path) -> None:
    from babybot.agent_kernel import ToolRegistry
    from babybot.channels.tools import ChannelToolContext

    manager = object.__new__(ResourceManager)
    manager.config = SimpleNamespace(
        system=SimpleNamespace(context_history_tokens=2000),
        workspace_dir=tmp_path,
    )
    manager.registry = ToolRegistry()
    manager.groups = {
        "basic": ToolGroup("basic", "core", active=True),
        "code": ToolGroup("code", "code", active=True),
        "channel_feishu": ToolGroup("channel_feishu", "feishu", active=True),
    }
    manager.skills = {}
    manager._shared_gateway = object()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_test_strip_channel",
        default=str(tmp_path),
    )
    manager._get_output_dir = lambda: tmp_path
    manager._build_task_lease = lambda lease: ToolLease(
        include_groups=("basic", "code"),
        include_tools=("inspect_chat_context",),
    )
    manager._build_worker_sys_prompt = lambda **kwargs: "sys"
    manager.get_shared_gateway = lambda: object()

    def _tool_ok() -> str:
        return "ok"

    manager.register_tool(_tool_ok, group_name="basic", func_name="inspect_chat_context")
    manager.register_tool(_tool_ok, group_name="code", func_name="_workspace_execute_shell_command")
    manager.register_tool(_tool_ok, group_name="channel_feishu", func_name="send_audio")
    manager.register_tool(_tool_ok, group_name="channel_feishu", func_name="send_file")

    async def _select_skill_packs(task_description: str, skill_ids=None):
        del task_description, skill_ids
        return []

    manager._select_skill_packs = _select_skill_packs
    captured: dict[str, object] = {}

    class _FakeExecutor:
        async def execute(self, task, context):
            captured["lease"] = task.lease
            captured["channel_context"] = context.state.get("channel_context")
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
        text, media = asyncio.run(manager.run_subagent_task("send audio carefully"))
    finally:
        ChannelToolContext.set_current(None)

    assert text == "done"
    assert media == []
    stripped = captured["lease"]
    assert isinstance(stripped, ToolLease)
    assert stripped.include_groups == ("basic", "code", "channel_feishu")
    assert stripped.include_tools == ("inspect_chat_context",)
    assert stripped.exclude_tools == ()
    assert captured["channel_context"] is parent_ctx


def test_create_worker_tool_inherits_parent_scope_by_default() -> None:
    manager = object.__new__(ResourceManager)
    captured: dict[str, object] = {}

    lease_var = contextvars.ContextVar("current_task_lease_test", default=None)
    skill_ids_var = contextvars.ContextVar("current_skill_ids_test", default=None)
    lease_var.set(ToolLease(include_groups=("basic", "skill_auto_skill_creator")))
    skill_ids_var.set(("auto-skill-creator",))
    manager._current_task_lease = lease_var
    manager._current_skill_ids = skill_ids_var
    manager._lease_to_dict = ResourceManager._lease_to_dict

    async def _run_subagent_task(
        task_description: str,
        lease: dict[str, object] | None = None,
        agent_name: str = "Worker",
        tape=None,
        tape_store=None,
        heartbeat=None,
        media_paths=None,
        skill_ids=None,
    ) -> tuple[str, list[str]]:
        del tape, tape_store, heartbeat, media_paths
        captured["task_description"] = task_description
        captured["lease"] = lease
        captured["agent_name"] = agent_name
        captured["skill_ids"] = skill_ids
        return "done", []

    manager.run_subagent_task = _run_subagent_task

    text = asyncio.run(
        manager.create_worker_tool()("inspect the project homepage")
    )

    assert text == "done"
    assert captured["task_description"] == "inspect the project homepage"
    assert captured["agent_name"] == "Worker"
    assert captured["lease"] == {"include_groups": ["basic", "skill_auto_skill_creator"]}
    assert captured["skill_ids"] == ["auto-skill-creator"]


def test_build_worker_prompt_allows_live_subagents_to_send_final_delivery() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {}
    prompt = manager._build_worker_sys_prompt(
        agent_name="Worker",
        task_description="生成一段语音并返回结果",
        tools_text="mlx_audio__generate_speech",
        selected_skill_packs=[],
    )

    assert "可直接发送最终内容" in prompt
    assert "避免发送中间状态" in prompt


def test_build_worker_prompt_guides_skill_edits_to_skill_md_and_verification() -> None:
    manager = object.__new__(ResourceManager)
    manager.skills = {}
    prompt = manager._build_worker_sys_prompt(
        agent_name="Worker",
        task_description="删除一个技能并给另一个技能增加模型支持",
        tools_text="_workspace_view_text_file, _workspace_write_text_file",
        selected_skill_packs=[],
    )

    assert "SKILL.md" in prompt
    assert "skill.yaml" in prompt
    assert "config.yaml" in prompt
    assert "先检查目标技能是否存在" in prompt
    assert "output" in prompt


def test_build_task_lease_excludes_nested_orchestration_tools_by_default() -> None:
    from babybot.agent_kernel import ToolRegistry

    manager = object.__new__(ResourceManager)
    manager.groups = {"basic": ToolGroup("basic", "core", active=True)}
    manager.registry = ToolRegistry()

    def _tool_ok() -> str:
        return "ok"

    manager.register_tool(_tool_ok, group_name="basic", func_name="regular_tool")
    manager.register_tool(_tool_ok, group_name="basic", func_name="create_worker")
    manager.register_tool(_tool_ok, group_name="basic", func_name="dispatch_workers")

    lease = manager._build_task_lease({})
    names = sorted(t.tool.name for t in manager.registry.list(lease))

    assert "regular_tool" in names
    assert "create_worker" not in names
    assert "dispatch_workers" not in names


def test_build_task_lease_adds_basic_group_and_filters_unknown_include_tools() -> None:
    from babybot.agent_kernel import ToolRegistry

    manager = object.__new__(ResourceManager)
    manager.groups = {
        "basic": ToolGroup("basic", "core", active=True),
        "analysis": ToolGroup("analysis", "analysis", active=False),
    }
    manager.registry = ToolRegistry()

    def _tool_ok() -> str:
        return "ok"

    manager.register_tool(_tool_ok, group_name="basic", func_name="regular_tool")

    lease = manager._build_task_lease(
        {
            "include_groups": ["analysis"],
            "include_tools": ["regular_tool", "missing_tool"],
        }
    )

    assert lease.include_groups == ("analysis", "basic")
    assert lease.include_tools == ("regular_tool",)


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


def test_callable_tool_can_disable_implicit_artifact_collection(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")

    def return_shell_like_output() -> str:
        return f"found artifact: {image_path}"

    tool = CallableTool(
        func=return_shell_like_output,
        name="_workspace_execute_shell_command",
        description="shell",
        schema={"type": "object", "properties": {}},
        collect_artifacts=False,
    )

    result = asyncio.run(tool.invoke({}, ToolContext(session_id="s1", state={})))

    assert result.ok is True
    assert str(image_path) in result.content
    assert result.artifacts == []


def test_callable_tool_relocates_external_artifact_into_workspace_output(tmp_path: Path) -> None:
    workspace_output = tmp_path / "output"
    workspace_output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"png-bytes")
        external_path = Path(f.name).resolve()

    manager = object.__new__(ResourceManager)
    manager.config = SimpleNamespace(workspace_dir=tmp_path)
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_artifact_relocation",
        default=str(workspace_output),
    )
    manager._get_output_dir = lambda: workspace_output

    def return_external_path() -> str:
        return str(external_path)

    tool = CallableTool(
        func=return_external_path,
        name="return_external_path",
        description="return external image path",
        schema={"type": "object", "properties": {}},
        resource_manager=manager,
    )

    try:
        result = asyncio.run(tool.invoke({}, ToolContext(session_id="s1", state={})))
    finally:
        if external_path.exists():
            external_path.unlink()

    assert result.ok is True
    assert len(result.artifacts) == 1
    relocated = Path(result.artifacts[0])
    assert relocated.parent == workspace_output.resolve()
    assert relocated.name == external_path.name
    assert relocated.read_bytes() == b"png-bytes"


def test_get_python_candidates_prefers_skill_runtime_then_fallbacks(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager.config = type(
        "DummyConfig",
        (),
        {
            "system": SimpleNamespace(
                shell_command_timeout=300,
                python_executable="/global/python",
                python_fallback_executables=["/global/fallback"],
            ),
            "workspace_dir": tmp_path,
        },
    )()

    runtime = ResourceManager._build_skill_runtime(
        {
            "python_executable": "/skill/python",
            "python_fallback_executables": "/skill/fallback-a, /skill/fallback-b",
            "python_required_modules": "mlx_audio, soundfile",
        }
    )

    candidates = manager._get_python_candidates(runtime)

    assert [item["executable"] for item in candidates[:5]] == [
        "/skill/python",
        "/skill/fallback-a",
        "/skill/fallback-b",
        "/global/python",
        "/global/fallback",
    ]
    assert candidates[0]["required_modules"] == ("mlx_audio", "soundfile")


def test_invoke_external_skill_function_retries_with_fallback_python_on_env_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = object.__new__(ResourceManager)
    manager.config = type(
        "DummyConfig",
        (),
        {
            "system": SimpleNamespace(
                shell_command_timeout=300,
                python_executable="",
                python_fallback_executables=[],
            ),
            "workspace_dir": tmp_path,
        },
    )()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_external_skill_fallback",
        default=str(tmp_path),
    )

    runtime = ResourceManager._build_skill_runtime(
        {
            "python_executable": "/broken/python",
            "python_fallback_executables": ["/healthy/python"],
        }
    )
    monkeypatch.setattr(manager, "_get_python_candidates", lambda skill_runtime=None: [
        {
            "executable": "/broken/python",
            "required_modules": (),
            "source": "skill.python_executable",
        },
        {
            "executable": "/healthy/python",
            "required_modules": (),
            "source": "skill.python_fallback_executables",
        },
    ])
    monkeypatch.setattr(manager, "_probe_python_candidate", lambda candidate: None)

    calls: list[str] = []

    class _Proc:
        def __init__(self, returncode: int, stdout: bytes, stderr: bytes):
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        async def communicate(self):
            return self._stdout, self._stderr

        def kill(self) -> None:
            return None

    async def _fake_exec(program, *args, **kwargs):
        del args, kwargs
        calls.append(program)
        if program == "/broken/python":
            return _Proc(
                1,
                b"",
                b"Traceback (most recent call last):\nModuleNotFoundError: No module named 'mlx_audio'\n",
            )
        return _Proc(
            0,
            b'__BABYBOT_RESULT__{"ok": true, "result": "audio.wav"}\n',
            b"",
        )

    monkeypatch.setattr("babybot.resource.asyncio.create_subprocess_exec", _fake_exec)

    result = asyncio.run(
        manager._invoke_external_skill_function(
            script_path=str(tmp_path / "tool.py"),
            function_name="generate_speech",
            arguments={"text": "hello"},
            runtime=runtime,
        )
    )

    assert result == "audio.wav"
    assert calls == ["/broken/python", "/healthy/python"]


def test_invoke_external_skill_function_does_not_retry_business_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = object.__new__(ResourceManager)
    manager.config = type(
        "DummyConfig",
        (),
        {
            "system": SimpleNamespace(
                shell_command_timeout=300,
                python_executable="",
                python_fallback_executables=[],
            ),
            "workspace_dir": tmp_path,
        },
    )()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_external_skill_business_failure",
        default=str(tmp_path),
    )

    monkeypatch.setattr(manager, "_get_python_candidates", lambda skill_runtime=None: [
        {"executable": "/primary/python", "required_modules": (), "source": "auto"},
        {"executable": "/fallback/python", "required_modules": (), "source": "auto"},
    ])
    monkeypatch.setattr(manager, "_probe_python_candidate", lambda candidate: None)

    calls: list[str] = []

    class _Proc:
        def __init__(self, returncode: int, stdout: bytes, stderr: bytes):
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        async def communicate(self):
            return self._stdout, self._stderr

        def kill(self) -> None:
            return None

    async def _fake_exec(program, *args, **kwargs):
        del args, kwargs
        calls.append(program)
        return _Proc(
            0,
            b'__BABYBOT_RESULT__{"ok": false, "error": "text is empty"}\n',
            b"",
        )

    monkeypatch.setattr("babybot.resource.asyncio.create_subprocess_exec", _fake_exec)

    result = asyncio.run(
        manager._invoke_external_skill_function(
            script_path=str(tmp_path / "tool.py"),
            function_name="generate_speech",
            arguments={"text": ""},
        )
    )

    assert result == "Tool error: text is empty"
    assert calls == ["/primary/python"]


def test_invoke_external_skill_function_beats_heartbeat_from_process_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = object.__new__(ResourceManager)
    manager.config = type(
        "DummyConfig",
        (),
        {
            "system": SimpleNamespace(
                shell_command_timeout=300,
                python_executable="",
                python_fallback_executables=[],
            ),
            "workspace_dir": tmp_path,
        },
    )()
    manager._active_write_root = contextvars.ContextVar(
        "active_write_root_external_skill_progress",
        default=str(tmp_path),
    )

    monkeypatch.setattr(manager, "_get_python_candidates", lambda skill_runtime=None: [
        {"executable": "/primary/python", "required_modules": (), "source": "auto"},
    ])
    monkeypatch.setattr(manager, "_probe_python_candidate", lambda candidate: None)

    beats: list[tuple[str | None, float | None]] = []

    class _Heartbeat:
        def beat(self, *, progress: float | None = None, status: str | None = None) -> None:
            beats.append((status, progress))

    monkeypatch.setattr(manager, "_get_current_task_heartbeat", lambda: _Heartbeat())

    class _FakeStream:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        async def readline(self) -> bytes:
            if self._lines:
                return self._lines.pop(0)
            return b""

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = _FakeStream(
                [
                    b"Downloading model 25%\n",
                    b"Downloading model 75%\n",
                    b'__BABYBOT_RESULT__{"ok": true, "result": "audio.wav"}\n',
                ]
            )
            self.stderr = _FakeStream([])

        async def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            return None

    async def _fake_exec(program, *args, **kwargs):
        del program, args, kwargs
        return _Proc()

    monkeypatch.setattr("babybot.resource.asyncio.create_subprocess_exec", _fake_exec)

    result = asyncio.run(
        manager._invoke_external_skill_function(
            script_path=str(tmp_path / "tool.py"),
            function_name="generate_speech",
            arguments={"text": "hello"},
        )
    )

    assert result == "audio.wav"
    assert any(progress == 0.25 for _, progress in beats)
    assert any(progress == 0.75 for _, progress in beats)


def test_inspect_chat_context_uses_channel_context_default_chat_key(tmp_path: Path) -> None:
    from babybot.channels.tools import ChannelToolContext

    manager = object.__new__(ResourceManager)
    manager._observability_provider = type(
        "Provider",
        (),
        {
            "inspect_chat_context": staticmethod(
                lambda chat_key, query="": f"chat={chat_key};query={query}"
            )
        },
    )()

    ChannelToolContext.set_current(
        ChannelToolContext(channel_name="feishu", chat_id="oc_test", sender_id="u1")
    )
    try:
        result = manager._inspect_chat_context(query="继续语音任务")
    finally:
        ChannelToolContext.set_current(None)

    assert "chat=feishu:oc_test" in result
    assert "query=继续语音任务" in result


def test_inspect_runtime_flow_uses_provider_snapshot(tmp_path: Path) -> None:
    manager = object.__new__(ResourceManager)
    manager._observability_provider = type(
        "Provider",
        (),
        {
            "inspect_runtime_flow": staticmethod(
                lambda flow_id="", chat_key="": f"flow={flow_id};chat={chat_key}"
            )
        },
    )()

    result = manager._inspect_runtime_flow(flow_id="orchestrator:abc123")

    assert result == "flow=orchestrator:abc123;chat="
