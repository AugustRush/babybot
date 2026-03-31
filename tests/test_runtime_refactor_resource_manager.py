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


def test_stdio_mcp_defaults_to_isolated_artifact_root(
    tmp_path: Path,
) -> None:
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    command, args, cwd, env = manager._prepare_mcp_stdio_launch(
        "demo_server",
        "python",
        ["-m", "demo"],
        {},
    )

    artifact_root = config.workspace_dir.resolve() / "output" / "mcp" / "demo_server"

    assert command == "python"
    assert args == ["-m", "demo"]
    assert cwd == str(artifact_root)
    assert env is not None
    assert env["BABYBOT_MCP_SERVER_NAME"] == "demo_server"
    assert env["BABYBOT_MCP_WORKSPACE_ROOT"] == str(config.workspace_dir.resolve())
    assert env["BABYBOT_MCP_ARTIFACT_ROOT"] == str(artifact_root)


def test_stdio_mcp_launch_expands_configured_cwd_and_env_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    _command, _args, cwd, env = manager._prepare_mcp_stdio_launch(
        "demo_server",
        "npx",
        ["@demo/mcp"],
        {
            "cwd": "~/browser-mcp",
            "env": {
                "BABYBOT_MCP_ARTIFACT_ROOT": "${HOME}/custom-output",
                "DEBUG": "1",
            },
        },
    )

    assert cwd == str((tmp_path / "home" / "browser-mcp").resolve())
    assert env == {
        "BABYBOT_MCP_SERVER_NAME": "demo_server",
        "BABYBOT_MCP_WORKSPACE_ROOT": str(config.workspace_dir.resolve()),
        "BABYBOT_MCP_ARTIFACT_ROOT": str(
            (tmp_path / "home" / "custom-output").resolve()
        ),
        "DEBUG": "1",
    }


def test_http_mcp_defaults_to_metadata_headers(tmp_path: Path) -> None:
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    url, headers = manager._prepare_mcp_http_launch(
        "remote_demo",
        "https://example.com/mcp",
        {},
    )

    artifact_root = config.workspace_dir.resolve() / "output" / "mcp" / "remote_demo"

    assert url == "https://example.com/mcp"
    assert headers == {
        "X-Babybot-Mcp-Server": "remote_demo",
        "X-Babybot-Workspace-Root": str(config.workspace_dir.resolve()),
        "X-Babybot-Artifact-Root": str(artifact_root),
    }


def test_http_mcp_merges_user_headers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = Config(config_file=str(tmp_path / "config.json"))
    manager = ResourceManager(config)

    _url, headers = manager._prepare_mcp_http_launch(
        "remote_demo",
        "https://example.com/mcp",
        {
            "headers": {
                "Authorization": "Bearer token",
                "X-Babybot-Artifact-Root": "${HOME}/custom-http-root",
            }
        },
    )

    assert headers == {
        "X-Babybot-Mcp-Server": "remote_demo",
        "X-Babybot-Workspace-Root": str(config.workspace_dir.resolve()),
        "X-Babybot-Artifact-Root": str(
            (tmp_path / "home" / "custom-http-root").resolve()
        ),
        "Authorization": "Bearer token",
    }
