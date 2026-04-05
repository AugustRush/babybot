from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import uuid


def test_marked_async_test_executes_without_external_pytest_asyncio_plugin(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = (
        repo_root
        / "tests"
        / f"_pytest_asyncio_fallback_probe_{uuid.uuid4().hex}.py"
    )
    try:
        test_file.write_text(
            "import pytest\n"
            "\n"
            "@pytest.mark.asyncio\n"
            "async def test_probe():\n"
            "    assert True\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert "1 passed" in result.stdout
        assert "skipped" not in result.stdout.lower()
    finally:
        test_file.unlink(missing_ok=True)
