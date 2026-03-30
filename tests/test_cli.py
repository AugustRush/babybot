from __future__ import annotations

from types import SimpleNamespace

import babybot.cli as cli_module


class _Response:
    def __init__(self, text: str = "ok", media_paths: list[str] | None = None) -> None:
        self.text = text
        self.media_paths = list(media_paths or [])


class _Orchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.status_payload = {
            "available_tools": 3,
            "interactive_sessions": {
                "active_count": 1,
                "chat_keys": ["cli:local"],
                "sessions": [
                    {
                        "chat_key": "cli:local",
                        "backend_name": "claude",
                        "backend_status": {"mode": "resident", "alive": True, "pid": 4321},
                    }
                ],
            },
        }

    async def process_task(self, user_input: str, chat_key: str = "", **_: object) -> _Response:
        self.calls.append((user_input, chat_key))
        return _Response(text=f"handled:{user_input}")

    def get_status(self) -> dict[str, object]:
        return dict(self.status_payload)

    def reset(self) -> None:
        return None


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(system=SimpleNamespace(timeout=1.0))


def test_cli_run_routes_messages_with_stable_chat_key(
    monkeypatch,
    capsys,
) -> None:
    orchestrator = _Orchestrator()
    inputs = iter(["@session start claude", "quit"])

    monkeypatch.setattr(
        cli_module,
        "_init_orchestrator",
        lambda gateway_mode=False: (_fake_config(), orchestrator),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    cli_module.run()

    _ = capsys.readouterr()
    assert orchestrator.calls == [
        ("@session start claude", cli_module._CLI_CHAT_KEY),
    ]


def test_cli_status_prints_interactive_session_summary(
    monkeypatch,
    capsys,
) -> None:
    orchestrator = _Orchestrator()
    inputs = iter(["status", "quit"])

    monkeypatch.setattr(
        cli_module,
        "_init_orchestrator",
        lambda gateway_mode=False: (_fake_config(), orchestrator),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    cli_module.run()

    output = capsys.readouterr().out
    assert "Available Tools: 3" in output
    assert "Interactive Sessions: 1" in output
    assert "cli:local" in output
    assert "resident" in output
    assert "4321" in output
