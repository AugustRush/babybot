"""Configuration contract for DynamicOrchestrator.

All application-specific content (prompts, tool descriptions, NLU patterns,
sub-task templates, UI strings) is declared here and injected at construction
time.  The kernel itself stays language- and domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorConfig:
    """Injectable configuration for DynamicOrchestrator.

    Pass an instance to ``DynamicOrchestrator.__init__`` to supply all
    application-layer content.  Any field left as ``None`` / empty causes the
    orchestrator to fall back to a minimal, language-agnostic default.
    """

    # ── System prompt ────────────────────────────────────────────────────
    # Static role/rules section prepended to every orchestration request.
    system_prompt: str = ""

    # ── Tool descriptions ────────────────────────────────────────────────
    # Map from tool name to its natural-language description shown to the LLM.
    # Keys: "dispatch_task", "wait_for_tasks", "get_task_result",
    #        "reply_to_user", "dispatch_team"
    # Any missing key keeps the bare English skeleton.
    tool_descriptions: dict[str, str] = field(default_factory=dict)

    # ── Tool parameter descriptions ──────────────────────────────────────
    # Nested map: tool_name -> param_name -> description string.
    tool_param_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)

    # ── Deferred-task patterns ───────────────────────────────────────────
    # Substrings that, if found in the goal, trigger injection of
    # ``deferred_task_guidance`` into the system prompt.
    deferred_task_patterns: tuple[str, ...] = ()

    # Additional system-prompt section appended when a deferred-task pattern
    # is detected in the goal.
    deferred_task_guidance: str = ""

    # ── Resource catalog formatting ──────────────────────────────────────
    # Label shown before the resource list  (e.g. "\nAvailable resources:\n")
    resource_catalog_header: str = "\nAvailable resources:\n"
    # Label used when there are no active resources.
    resource_catalog_empty: str = "\nNo active resources."
    # Template for a single resource line.
    # Receives keyword args: rid, name, purpose, tc, preview_text.
    resource_catalog_line: str = (
        "- {rid}: {name} — {purpose} (tools: {tc}{preview_text})"
    )
    # Label prefix for the tools-preview part of a resource line.
    resource_catalog_preview_prefix: str = "; sample tools: "

    # Section headers for tiered resource catalog display.
    # When non-empty, the catalog groups resources by tier instead of flat list.
    resource_catalog_specialist_header: str = ""
    resource_catalog_general_header: str = ""

    # ── Policy hints section header ──────────────────────────────────────
    # Text prepended to the bullet list of policy hints in the system prompt.
    policy_hints_header: str = "\nPolicy hints:\n- "

    # ── Execution constraints section labels ─────────────────────────────
    # Wrapper injected around the user goal when execution constraints exist.
    # Use ``{constraints}`` and ``{goal}`` placeholders.
    execution_constraints_wrapper: str = (
        "[Constraints]\n{constraints}\n\n[Request]\n{goal}"
    )

    # ── Child-task enrichment section headers ───────────────────────────
    # Used when downstream tasks inherit the original request or upstream
    # task outputs outside the structured child-task prompt path.
    original_request_header: str = "--- original_request ---"
    upstream_results_header: str = "--- upstream_results ---"

    # ── Sub-task description template ────────────────────────────────────
    # Sentinel string: if already present in a child-task description the
    # normalisation step is skipped entirely.
    child_task_sentinel: str = ""

    # Callable ``(raw_description, original_goal, resource_ids,
    # upstream_results) -> str`` that builds the structured child-task prompt.
    # Signature::
    #
    #   def build_child_task_prompt(
    #       raw_description: str,
    #       original_goal: str,
    #       resource_ids: tuple[str, ...],
    #       upstream_results: dict[str, Any],   # tid -> TaskResult-like
    #   ) -> str: ...
    #
    # Return ``""`` to keep ``raw_description`` unchanged.
    # If ``None`` the normalisation step is skipped entirely.
    build_child_task_prompt: Any = None  # Callable | None

    # ── Repo/skill maintenance detector ─────────────────────────────────
    # Callable ``(text: str) -> bool`` used to detect maintenance-like tasks
    # and add extra guidance lines to the child-task prompt.
    # If ``None`` the heuristic is disabled (always returns False).
    is_maintenance_task: Any = None  # Callable[[str], bool] | None

    # ── Honesty-reminder (all tasks dead-lettered) ───────────────────────
    # Injected into the reply_to_user result when every child task failed.
    # Use ``{dead_ids}`` and ``{errors}`` placeholders.
    all_tasks_failed_reminder: str = (
        "[SYSTEM NOTICE] All child tasks failed with no successful results.\n"
        "Failed tasks: {dead_ids}\n"
        "Error summary: {errors}\n"
        "You MUST report the failure and its reasons honestly. "
        "Do not fabricate or infer results that were never executed."
    )

    # ── Step-budget fallback conclusion ──────────────────────────────────
    # Header line when the orchestrator exhausts its step budget.
    step_budget_exhausted_header: str = (
        "(Step budget exhausted. Partial results below.)"
    )
    # Template for a succeeded task line.  Receives ``task_id`` and ``output``.
    step_budget_succeeded_line: str = "- {task_id}: {output}"
    # Template for a failed task line.  Receives ``task_id`` and ``error``.
    step_budget_failed_line: str = "- {task_id}: failed — {error}"

    # ── Team-mode progress messages ───────────────────────────────────────
    # Receives keyword args as noted.
    # debate mode
    team_debate_started_message: str = (
        "Started {n_agents} agents, up to {max_rounds} rounds."
    )
    team_debate_round_message: str = "Round {round_num}/{total_rounds} in progress."
    team_debate_ended_message: str = "Discussion ended."
    # cooperative mode
    team_coop_started_message: str = "Started {n_agents} agents on {n_tasks} tasks."
    team_coop_task_done_message: str = "Task {task_id} done ({done}/{total})."
    team_coop_ended_message: str = "Cooperative run ended."

    # ── Team default system prompt (fallback when agent has none) ────────
    team_default_agent_system_prompt: str = (
        "You are a discussion participant. "
        "Share your perspective on the topic according to your role."
    )

    # ── NLU helpers (task-shape inference) ───────────────────────────────
    # Token lists for detecting multi-step / parallel intent in the goal.
    # Used only by the optional _provider_policy_hints path.
    multi_step_tokens: tuple[str, ...] = ()
    parallel_tokens: tuple[str, ...] = ()

    # ── Dynamic resource selection addendum ──────────────────────────────
    # Callable ``(briefs: list[dict[str, Any]]) -> str`` that builds a
    # dynamic addendum for the system prompt based on the currently active
    # resource composition.  Return ``""`` to skip.
    # If ``None``, no addendum is generated.
    build_resource_selection_addendum: Any = None  # Callable | None

    # ── Force-converge policy ────────────────────────────────────────────
    # After this many dead-lettered child tasks under the same notebook
    # execution node, the orchestrator stops offering dispatch tools and
    # requires the model to converge via existing results / failure summary.
    force_converge_dead_letter_threshold: int = 3
