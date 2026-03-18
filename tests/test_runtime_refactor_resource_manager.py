from __future__ import annotations

from pathlib import Path

from babybot.config import Config
from babybot.resource import ResourceManager


def _reset_resource_manager_singleton() -> None:
    ResourceManager._instance = None
    ResourceManager._initialized = False


def test_resource_manager_singleton_reuses_first_config(tmp_path: Path) -> None:
    _reset_resource_manager_singleton()
    try:
        config_one = Config(config_file=str(tmp_path / "config-one.json"))
        config_two = Config(config_file=str(tmp_path / "config-two.json"))

        manager_one = ResourceManager(config_one)
        manager_two = ResourceManager(config_two)

        assert manager_one is manager_two
        assert manager_two.config is config_one
    finally:
        _reset_resource_manager_singleton()
