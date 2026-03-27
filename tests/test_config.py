import json

import pytest

from babybot.config import Config


def test_policy_learning_config_fields_load_from_system(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "model": {"api_key": "test"},
                "system": {
                    "policy_learning_enabled": True,
                    "policy_learning_min_samples": 5,
                    "policy_learning_explore_ratio": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_path))

    assert cfg.system.policy_learning_enabled is True
    assert cfg.system.policy_learning_min_samples == 5
    assert cfg.system.policy_learning_explore_ratio == 0.05


def test_system_context_fields_loaded_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "system": {
                    "context_history_tokens": 123,
                    "context_compact_threshold": 456,
                    "context_max_chats": 7,
                    "worker_max_steps": 21,
                    "orchestrator_max_steps": 31,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))
    assert cfg.system.context_history_tokens == 123
    assert cfg.system.context_compact_threshold == 456
    assert cfg.system.context_max_chats == 7
    assert cfg.system.worker_max_steps == 21
    assert cfg.system.orchestrator_max_steps == 31


def test_to_dict_includes_system_context_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "system": {
                    "context_history_tokens": 321,
                    "context_compact_threshold": 654,
                    "context_max_chats": 9,
                    "worker_max_steps": 18,
                    "orchestrator_max_steps": 28,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))
    payload = cfg.to_dict()["system"]
    assert payload["context_history_tokens"] == 321
    assert payload["context_compact_threshold"] == 654
    assert payload["context_max_chats"] == 9
    assert payload["worker_max_steps"] == 18
    assert payload["orchestrator_max_steps"] == 28
    assert payload["interactive_session_max_age_seconds"] == 7200


def test_interactive_session_max_age_loaded_from_system(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "model": {"api_key": "test"},
                "system": {"interactive_session_max_age_seconds": 1800},
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_path))

    assert cfg.system.interactive_session_max_age_seconds == 1800


def test_scheduled_tasks_migrated_to_workspace_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    legacy_tasks = [
        {
            "name": "news",
            "prompt": "summarize news",
            "schedule": "0 9 * * *",
            "target": {"channel": "feishu", "chat_id": "c1"},
        }
    ]
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "scheduled_tasks": legacy_tasks,
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))

    assert cfg.get_scheduled_tasks() == legacy_tasks
    assert cfg.scheduled_tasks_file.exists()
    assert json.loads(cfg.scheduled_tasks_file.read_text(encoding="utf-8")) == legacy_tasks


def test_invalid_scheduled_tasks_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "home" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "scheduled_tasks.json").write_text('{"bad": true}', encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Scheduled tasks file must contain a JSON list"):
        Config(config_file=str(config_path))


def test_config_repr_no_longer_reports_custom_tool_count(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "custom_tools": {
                    "demo": {
                        "module": "tools.search",
                        "function": "web_search",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))

    assert "tools=" not in repr(cfg)


def test_weixin_channel_config_loaded_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "channels": {
                    "weixin": {
                        "enabled": True,
                        "base_url": "https://example.invalid",
                        "token": "tok",
                        "poll_timeout": 12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))
    weixin = cfg.get_channel_config("weixin")

    assert weixin.enabled is True
    assert weixin.base_url == "https://example.invalid"
    assert weixin.token == "tok"
    assert weixin.poll_timeout == 12


def test_to_dict_includes_weixin_channel_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_name": "test-model",
                    "api_key": "k",
                    "api_base": "",
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                "channels": {
                    "weixin": {
                        "enabled": True,
                        "token": "tok",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))
    payload = cfg.to_dict()["channels"]["weixin"]

    assert payload["enabled"] is True
    assert payload["token"] == "***"
