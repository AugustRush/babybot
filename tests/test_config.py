import json

from babybot.config import Config


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
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(config_file=str(config_path))
    assert cfg.system.context_history_tokens == 123
    assert cfg.system.context_compact_threshold == 456
    assert cfg.system.context_max_chats == 7


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
