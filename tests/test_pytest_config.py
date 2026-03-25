from __future__ import annotations

from pathlib import Path
import tomllib


def _load_pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_declares_pytest_asyncio_in_dev_dependencies() -> None:
    pyproject = _load_pyproject()

    dev_dependencies = pyproject.get("dependency-groups", {}).get("dev", [])

    assert any(str(item).startswith("pytest-asyncio") for item in dev_dependencies)


def test_pyproject_configures_pytest_pythonpath_and_asyncio_mode() -> None:
    pyproject = _load_pyproject()

    pytest_config = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})

    assert pytest_config.get("pythonpath") == ["."]
    assert pytest_config.get("asyncio_mode") == "auto"
