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
