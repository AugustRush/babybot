"""Prompt-assembly and policy-hint helpers for DynamicOrchestrator.

Extracted from dynamic_orchestrator.py. All symbols re-imported there
for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .orchestrator_config import OrchestratorConfig
from .types import ExecutionContext

if TYPE_CHECKING:
    from ..resource import ResourceManager

logger = logging.getLogger(__name__)

# ── System prompt builder ────────────────────────────────────────────────

# Minimal language-agnostic fallback used when no OrchestratorConfig is
# injected.  Application-specific content belongs in OrchestratorConfig
# (supplied by the application layer, e.g. orchestrator_prompts.py).
_SYSTEM_PROMPT_ROLE = (
    "You are an orchestration agent. "
    "Dispatch sub-tasks to available resources and reply to the user when done."
)

# Empty by default — patterns are supplied via OrchestratorConfig.
_DEFERRED_TASK_PATTERNS: tuple[str, ...] = ()

_DEFERRED_TASK_GUIDANCE = ""

# Default NLU token lists — empty by design (language-agnostic fallback).
# Override via OrchestratorConfig.multi_step_tokens / parallel_tokens.
_MULTI_STEP_TOKENS: tuple[str, ...] = ()
_PARALLEL_TOKENS: tuple[str, ...] = ()


def _build_resource_catalog(
    briefs: list[dict[str, Any]],
    config: OrchestratorConfig | None = None,
) -> str:
    cfg = config or OrchestratorConfig()

    specialist_types = {"skill", "mcp"}
    specialist_lines: list[str] = []
    general_lines: list[str] = []

    for b in briefs:
        if not b.get("active"):
            continue
        rid = b.get("id", "?")
        resource_type = b.get("type", "")
        name = b.get("name", "?")
        purpose = b.get("purpose", "")
        tc = b.get("tool_count", 0)
        preview = (
            ""
            if resource_type in {"mcp", "skill"}
            else ", ".join(b.get("tools_preview") or [])
        )
        preview_text = (
            (cfg.resource_catalog_preview_prefix + preview) if preview else ""
        )
        line = cfg.resource_catalog_line.format(
            rid=rid,
            name=name,
            purpose=purpose,
            tc=tc,
            preview_text=preview_text,
        )
        if resource_type in specialist_types:
            specialist_lines.append(line)
        else:
            general_lines.append(line)

    if not specialist_lines and not general_lines:
        return cfg.resource_catalog_empty

    # When tier headers are configured and both tiers have content, display
    # a tiered catalog; otherwise fall back to flat list.
    use_tiers = bool(
        cfg.resource_catalog_specialist_header
        and cfg.resource_catalog_general_header
        and specialist_lines
        and general_lines
    )

    if use_tiers:
        parts = [cfg.resource_catalog_header]
        parts.append(cfg.resource_catalog_specialist_header)
        parts.extend(f"  {line}" for line in specialist_lines)
        parts.append(cfg.resource_catalog_general_header)
        parts.extend(f"  {line}" for line in general_lines)
        return "\n".join(parts)

    all_lines = specialist_lines + general_lines
    return cfg.resource_catalog_header + "\n".join(all_lines)


def _needs_deferred_task_guidance(
    goal: str, config: OrchestratorConfig | None = None
) -> bool:
    lowered = (goal or "").strip()
    patterns = (
        config.deferred_task_patterns if config else None
    ) or _DEFERRED_TASK_PATTERNS
    return any(pattern in lowered for pattern in patterns)


def _normalize_recommended_resource_ids(
    payload: dict[str, Any] | None,
    key: str,
) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def _provider_policy_hints(
    resource_manager: "ResourceManager",
    goal: str,
    config: OrchestratorConfig | None = None,
) -> list[str]:
    provider = getattr(resource_manager, "_observability_provider", None)
    if provider is None:
        return []
    cfg = config or OrchestratorConfig()
    text = str(goal or "").strip()
    build_features = getattr(provider, "_build_policy_state_features", None)
    multi_step_tokens = cfg.multi_step_tokens or _MULTI_STEP_TOKENS
    parallel_tokens = cfg.parallel_tokens or _PARALLEL_TOKENS
    features: dict[str, Any] = {
        "task_shape": "multi_step"
        if any(token in text for token in multi_step_tokens)
        else "single_step",
        "input_length": len(text),
    }
    if callable(build_features):
        _raw = build_features(goal)
        result: dict[str, Any]
        result = _raw if isinstance(_raw, dict) else {}
        if result:
            features.update(result)
    independent_subtasks = 1
    for token in parallel_tokens:
        independent_subtasks += text.count(token)
    features["independent_subtasks"] = max(1, independent_subtasks)
    hints: list[str] = []
    for method_name in ("choose_scheduling_policy", "choose_worker_policy"):
        chooser = getattr(provider, method_name, None)
        if not callable(chooser):
            continue
        payload = chooser(features=features)
        if isinstance(payload, dict):
            action_name = str(
                payload.get("action_name") or payload.get("name") or ""
            ).strip()
            hint = str(payload.get("hint") or "").strip()
        else:
            action_name = str(
                getattr(payload, "action_name", "")
                or getattr(payload, "name", "")
                or ""
            ).strip()
            hint = str(getattr(payload, "hint", "") or "").strip()
        if not hint:
            continue
        if method_name == "choose_worker_policy" and action_name == "allow_worker":
            continue
        hints.append(hint)
    return hints


def _goal_has_explicit_parallel_intent(
    goal: str, config: OrchestratorConfig | None = None
) -> bool:
    text = str(goal or "").strip()
    if not text:
        return False
    tokens = (config.parallel_tokens if config else None) or _PARALLEL_TOKENS
    return any(token in text for token in tokens)


def _is_maintenance_goal(goal: str, config: OrchestratorConfig | None = None) -> bool:
    text = str(goal or "").strip()
    if not text:
        return False
    detector = config.is_maintenance_task if config else None
    if callable(detector):
        try:
            return bool(detector(text))
        except Exception:
            logger.exception("maintenance task detector raised; falling back to False")
    return False


def _emit_policy_decision(
    context: ExecutionContext,
    *,
    decision_kind: str,
    action_name: str,
    state_features: dict[str, Any] | None = None,
) -> None:
    context.emit(
        "policy_decision",
        decision_kind=decision_kind,
        action_name=action_name,
        state_features=dict(state_features or {}),
    )
