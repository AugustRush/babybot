from __future__ import annotations

from pathlib import Path

from babybot.config import Config
from babybot.resource import ResourceManager


def test_resource_manager_instances_keep_their_own_config(tmp_path: Path) -> None:
    config_one = Config(config_file=str(tmp_path / "config-one.json"))
    config_two = Config(config_file=str(tmp_path / "config-two.json"))

    manager_one = ResourceManager(config_one)
    manager_two = ResourceManager(config_two)

    assert manager_one is not manager_two
    assert manager_one.config is config_one
    assert manager_two.config is config_two


def test_playwright_stdio_mcp_defaults_output_dir_to_workspace_output(
    tmp_path: Path,
) -> None:
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    command, args, cwd, env = manager._prepare_mcp_stdio_launch(
        "playwright",
        "npx",
        ["@playwright/mcp@latest"],
        {},
    )

    assert command == "npx"
    assert args == ["@playwright/mcp@latest"]
    assert cwd is None
    assert env is not None
    assert env["PLAYWRIGHT_MCP_OUTPUT_DIR"] == str(
        config.workspace_dir.resolve() / "output"
    )


def test_stdio_mcp_launch_expands_configured_cwd_and_env_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    _command, _args, cwd, env = manager._prepare_mcp_stdio_launch(
        "playwright",
        "npx",
        ["@playwright/mcp@latest"],
        {
            "cwd": "~/browser-mcp",
            "env": {
                "PLAYWRIGHT_MCP_OUTPUT_DIR": "${HOME}/custom-output",
                "DEBUG": "1",
            },
        },
    )

    assert cwd == str((tmp_path / "home" / "browser-mcp").resolve())
    assert env == {
        "PLAYWRIGHT_MCP_OUTPUT_DIR": str(
            (tmp_path / "home" / "custom-output").resolve()
        ),
        "DEBUG": "1",
    }
