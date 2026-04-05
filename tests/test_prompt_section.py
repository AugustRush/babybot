"""Tests for SystemPromptBuilder, WorkerState/OrchestratorState typed views,
and task-class classification in resource_skill_runtime."""

from __future__ import annotations

from babybot.agent_kernel import (
    ExecutionContext,
    OrchestratorState,
    RuntimeState,
    SystemPromptBuilder,
    SystemPromptSection,
    WorkerState,
)
from babybot.agent_kernel.prompt_assembly import (
    add_list_section,
    add_text_section,
    dedupe_prompt_items,
)
from babybot.resource_skill_runtime import (
    _EXECUTION_RULES,
    _SKILL_MAINTENANCE_RULES,
    _TASK_CLASS_RULES,
    _classify_task_from_tools,
)


# ── SystemPromptBuilder ───────────────────────────────────────────────────


def test_system_prompt_builder_sections_sorted_by_priority() -> None:
    builder = SystemPromptBuilder()
    builder.add("third", "C", priority=30)
    builder.add("first", "A", priority=10)
    builder.add("second", "B", priority=20)

    sections = builder.sections
    assert [s.name for s in sections] == ["first", "second", "third"]
    assert [s.content for s in sections] == ["A", "B", "C"]


def test_system_prompt_builder_build_joins_by_newline() -> None:
    builder = SystemPromptBuilder()
    builder.add("a", "Hello", priority=0)
    builder.add("b", "World", priority=1)
    result = builder.build()
    assert result == "Hello\nWorld"


def test_system_prompt_builder_skips_empty_sections() -> None:
    builder = SystemPromptBuilder()
    builder.add("empty", "", priority=5)
    builder.add("whitespace", "   ", priority=6)
    builder.add("real", "content", priority=7)
    assert len(builder.sections) == 1
    assert builder.sections[0].name == "real"


def test_system_prompt_builder_section_names_returns_ordered_list() -> None:
    builder = SystemPromptBuilder()
    builder.add("z", "Z", priority=99)
    builder.add("a", "A", priority=1)
    assert builder.section_names() == ["a", "z"]


def test_system_prompt_builder_add_returns_self_for_chaining() -> None:
    builder = SystemPromptBuilder()
    result = builder.add("x", "X").add("y", "Y")
    assert result is builder
    assert len(builder.sections) == 2


def test_system_prompt_section_is_frozen() -> None:
    section = SystemPromptSection(name="s", content="c", priority=0)
    try:
        section.content = "changed"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass  # expected


def test_system_prompt_builder_cacheable_flag_is_preserved() -> None:
    builder = SystemPromptBuilder()
    builder.add("static", "fixed content", priority=0, cacheable=True)
    builder.add("dynamic", "varies", priority=1, cacheable=False)
    sections = builder.sections
    assert sections[0].cacheable is True
    assert sections[1].cacheable is False


# ── Typed state views ─────────────────────────────────────────────────────


def test_worker_state_view_reads_and_writes_underlying_dict() -> None:
    ctx = ExecutionContext(session_id="ws-test")
    ctx.state["tape"] = "my_tape"
    ctx.state["heartbeat"] = None
    ctx.state["max_model_tokens"] = 4096

    ws: WorkerState = ctx.worker_state
    assert ws.get("tape") == "my_tape"
    assert ws.get("max_model_tokens") == 4096
    # Writing through the typed view also updates the underlying dict
    ws["summary"] = "result summary"  # type: ignore[index]
    assert ctx.state["summary"] == "result summary"


def test_orchestrator_state_view_reads_and_writes_underlying_dict() -> None:
    ctx = ExecutionContext(session_id="os-test")
    ctx.state["original_goal"] = "build something"
    ctx.state["status"] = "running"

    os_: OrchestratorState = ctx.orchestrator_state
    assert os_.get("original_goal") == "build something"
    assert os_.get("status") == "running"


def test_worker_state_and_orchestrator_state_share_same_dict() -> None:
    ctx = ExecutionContext(session_id="shared-test")
    ctx.worker_state["tape"] = "shared"  # type: ignore[index]
    # Both views point to ctx.state
    assert ctx.state["tape"] == "shared"
    assert ctx.orchestrator_state.get("tape") == "shared"


def test_worker_state_missing_key_returns_none_via_get() -> None:
    ctx = ExecutionContext(session_id="missing-test")
    ws: WorkerState = ctx.worker_state
    assert ws.get("tape") is None
    assert ws.get("max_model_tokens", 0) == 0


def test_runtime_state_centralizes_media_and_runtime_hints() -> None:
    ctx = ExecutionContext(
        session_id="runtime-state",
        state={"media_paths": ["a.png", "b.png"]},
    )

    runtime = RuntimeState(ctx)
    assert runtime.media_paths() == ("a.png", "b.png")

    runtime.append_runtime_hint("first")
    runtime.append_runtime_hint("second")

    assert ctx.state["pending_runtime_hints"] == ["first", "second"]


def test_runtime_state_notebook_binding_falls_back_to_root_node() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="repair runtime state", flow_id="flow-runtime")
    ctx = ExecutionContext(
        session_id="runtime-notebook",
        state={
            "plan_notebook": notebook,
            "current_notebook_node_id": "missing-node",
        },
    )

    binding = RuntimeState(ctx).notebook_binding()

    assert binding.active is True
    assert binding.notebook is notebook
    assert binding.node_id == notebook.root_node_id


def test_dedupe_prompt_items_filters_blanks_and_caps_result() -> None:
    assert dedupe_prompt_items(
        [" alpha ", "", "alpha", "beta", "  ", "gamma"],
        limit=2,
    ) == ["alpha", "beta"]


def test_prompt_assembly_helpers_build_consistent_sections() -> None:
    builder = SystemPromptBuilder()

    add_text_section(
        builder,
        "catalog",
        "worker-a, worker-b",
        priority=10,
        header="Catalog:\n",
    )
    add_list_section(
        builder,
        "hints",
        [" first ", "first", "second"],
        priority=20,
        header="Hints:\n",
    )

    assert builder.build() == "Catalog:\nworker-a, worker-b\nHints:\n- first\n- second"


# ── Task-class classification ─────────────────────────────────────────────


def test_classify_task_no_tools_returns_text_gen() -> None:
    assert _classify_task_from_tools("无", None) == "text_gen"
    assert _classify_task_from_tools("", None) == "text_gen"
    assert _classify_task_from_tools("   ", None) == "text_gen"


def test_classify_task_code_tools_returns_code() -> None:
    assert _classify_task_from_tools("execute_shell_command, view_file", None) == "code"
    assert _classify_task_from_tools("run_python, list_files", None) == "code"
    assert _classify_task_from_tools("bash_execute", None) == "code"


def test_classify_task_skill_maintenance_tools_returns_skill_maintenance() -> None:
    assert (
        _classify_task_from_tools("reload_skill, view_file", None)
        == "skill_maintenance"
    )
    assert _classify_task_from_tools("delete_skill, bash", None) == "skill_maintenance"


def test_classify_task_skill_maintenance_takes_priority_over_code() -> None:
    # reload_skill + execute_shell: skill_maintenance wins
    assert (
        _classify_task_from_tools("reload_skill, execute_shell_command", None)
        == "skill_maintenance"
    )


def test_classify_task_generic_tools_returns_tool_action() -> None:
    assert _classify_task_from_tools("view_file, send_message", None) == "tool_action"
    assert _classify_task_from_tools("search_web, write_file", None) == "tool_action"


def test_task_class_rules_dict_contains_expected_classes() -> None:
    assert "code" in _TASK_CLASS_RULES
    assert "tool_action" in _TASK_CLASS_RULES
    assert "skill_maintenance" in _TASK_CLASS_RULES
    # text_gen has no extra rules (prompt is shorter by design)
    assert "text_gen" not in _TASK_CLASS_RULES


# ── Static rule content sanity checks ────────────────────────────────────


def test_execution_rules_contains_key_directives() -> None:
    assert "文本生成" in _EXECUTION_RULES
    assert "不要直接向用户发送消息" in _EXECUTION_RULES
    assert "缺少输入" in _EXECUTION_RULES


def test_skill_maintenance_rules_contains_key_directives() -> None:
    assert "SKILL.md" in _SKILL_MAINTENANCE_RULES
    assert "skill.yaml" in _SKILL_MAINTENANCE_RULES
    assert "config.yaml" in _SKILL_MAINTENANCE_RULES
    assert "reload_skill" in _SKILL_MAINTENANCE_RULES
    assert "output" in _SKILL_MAINTENANCE_RULES
