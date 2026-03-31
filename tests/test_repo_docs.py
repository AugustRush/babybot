from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_docs_track_current_kernel_and_router_defaults() -> None:
    engine_stub = REPO_ROOT / "babybot" / "agent_kernel" / "engine.py"
    kernel_readme = (REPO_ROOT / "babybot" / "agent_kernel" / "README.md").read_text(
        encoding="utf-8"
    )
    main_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    config_example = json.loads(
        (REPO_ROOT / "config.json.example").read_text(encoding="utf-8")
    )

    assert not engine_stub.exists()
    assert "engine.py" not in kernel_readme
    assert "scheduler.py" not in kernel_readme
    assert "`routing_timeout` 默认 `3.0` 秒" in main_readme
    assert "skipped:*" in main_readme
    assert "skip_breakdown" in main_readme
    assert "ContextRouter" not in main_readme
    assert "orchestration_router.py" in main_readme
    assert "shadow routing" not in main_readme
    assert "shadow_routing_eval_rate" not in main_readme
    assert "内置工具、通道工具与 workspace 自定义工具的注册/加载" in main_readme
    assert config_example.get("system", {}).get("routing_timeout") == 3.0


def test_readme_project_tree_matches_current_repo_shape() -> None:
    main_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "├── config.json                  # 配置文件" not in main_readme
    assert "├── config.json.example          # 配置模板" in main_readme
    assert "├── scheduled_tasks.json.example # 定时任务模板" in main_readme
    assert "│   ├── orchestration_router.py  # 轻量路由判定与门控" in main_readme
    assert "│   ├── orchestration_policy.py  # 保守策略选择器" in main_readme
    assert "│   ├── orchestration_policy_store.py # 策略/路由 telemetry 持久化" in main_readme
    assert "│   ├── runtime_feedback_commands.py # @policy / @job / @session 命令解析" in main_readme
    assert "│   ├── interactive_sessions/    # 交互式 session 管理与 backend" in main_readme
