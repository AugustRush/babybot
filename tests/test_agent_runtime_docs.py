from __future__ import annotations

from pathlib import Path

from babybot.runtime_jobs import JOB_STATES


def _extract_bullet_states(path: Path) -> set[str]:
    states: set[str] = set()
    in_state_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped in {"权威状态：", "长任务统一映射为持久化 `JobRuntime`。权威状态只有："}:
            in_state_block = True
            continue
        if in_state_block and stripped and not stripped.startswith("- `"):
            break
        if stripped.startswith("- `") and stripped.endswith("`"):
            states.add(stripped.removeprefix("- `").removesuffix("`"))
    return states


def test_runtime_docs_only_reference_canonical_job_states() -> None:
    docs_dir = Path("docs/agent-runtime")
    for name in (
        "long-running-jobs.md",
        "feedback-state-machine.md",
    ):
        states = _extract_bullet_states(docs_dir / name)
        assert states
        assert states <= set(JOB_STATES)
