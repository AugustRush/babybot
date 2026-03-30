"""Execution-constraint parsing and normalization for orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ExecutionHardLimits:
    max_rounds: int | None = None
    max_agents: int | None = None
    max_total_seconds: float | None = None
    max_turn_seconds: float | None = None


@dataclass(frozen=True)
class ExecutionSoftPreferences:
    resolution_style: Literal["balanced", "single_pass", "fast_consensus", "thorough"] = (
        "balanced"
    )


@dataclass(frozen=True)
class DegradationPolicy:
    on_budget_exhausted: Literal["summarize_partial", "raise_timeout"] = (
        "summarize_partial"
    )


@dataclass(frozen=True)
class ExecutionConstraints:
    mode: Literal["interactive", "deferred"] = "interactive"
    hard_limits: ExecutionHardLimits = field(default_factory=ExecutionHardLimits)
    soft_preferences: ExecutionSoftPreferences = field(
        default_factory=ExecutionSoftPreferences
    )
    degradation: DegradationPolicy = field(default_factory=DegradationPolicy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamExecutionPolicy:
    max_rounds: int = 5
    max_agents: int | None = None
    max_total_seconds: float | None = None
    max_turn_seconds: float | None = None
    resolution_style: Literal["balanced", "single_pass", "fast_consensus", "thorough"] = (
        "balanced"
    )
    on_budget_exhausted: Literal["summarize_partial", "raise_timeout"] = (
        "summarize_partial"
    )


class ExecutionHardLimitsModel(BaseModel):
    max_rounds: int | None = None
    max_agents: int | None = None
    max_total_seconds: float | None = None
    max_turn_seconds: float | None = None


class ExecutionSoftPreferencesModel(BaseModel):
    resolution_style: Literal["balanced", "single_pass", "fast_consensus", "thorough"] = (
        "balanced"
    )


class DegradationPolicyModel(BaseModel):
    on_budget_exhausted: Literal["summarize_partial", "raise_timeout"] = (
        "summarize_partial"
    )


class ExecutionConstraintsModel(BaseModel):
    mode: Literal["interactive", "deferred"] = "interactive"
    hard_limits: ExecutionHardLimitsModel = Field(default_factory=ExecutionHardLimitsModel)
    soft_preferences: ExecutionSoftPreferencesModel = Field(
        default_factory=ExecutionSoftPreferencesModel
    )
    degradation: DegradationPolicyModel = Field(default_factory=DegradationPolicyModel)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def default_execution_constraints(
    *,
    default_max_total_seconds: float | None = None,
) -> dict[str, Any]:
    return ExecutionConstraints(
        mode="interactive",
        hard_limits=ExecutionHardLimits(
            max_total_seconds=default_max_total_seconds,
        ),
        degradation=DegradationPolicy(on_budget_exhausted="summarize_partial"),
    ).to_dict()


def _structured_to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    return {}


def normalize_execution_constraints(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    hard_limits = payload.get("hard_limits") if isinstance(payload.get("hard_limits"), dict) else {}
    soft_preferences = (
        payload.get("soft_preferences")
        if isinstance(payload.get("soft_preferences"), dict)
        else {}
    )
    degradation = (
        payload.get("degradation") if isinstance(payload.get("degradation"), dict) else {}
    )
    constraints = ExecutionConstraints(
        mode=(
            "deferred"
            if str(payload.get("mode", "") or "").strip().lower() == "deferred"
            else "interactive"
        ),
        hard_limits=ExecutionHardLimits(
            max_rounds=_coerce_int(hard_limits.get("max_rounds")),
            max_agents=_coerce_int(hard_limits.get("max_agents")),
            max_total_seconds=_coerce_float(hard_limits.get("max_total_seconds")),
            max_turn_seconds=_coerce_float(hard_limits.get("max_turn_seconds")),
        ),
        soft_preferences=ExecutionSoftPreferences(
            resolution_style=(
                str(soft_preferences.get("resolution_style", "") or "").strip().lower()
                if str(soft_preferences.get("resolution_style", "") or "").strip().lower()
                in {"balanced", "single_pass", "fast_consensus", "thorough"}
                else "balanced"
            )
        ),
        degradation=DegradationPolicy(
            on_budget_exhausted=(
                "raise_timeout"
                if str(degradation.get("on_budget_exhausted", "") or "").strip().lower()
                == "raise_timeout"
                else "summarize_partial"
            )
        ),
    )
    return constraints.to_dict()


async def infer_execution_constraints(
    gateway: Any,
    text: str,
    *,
    heartbeat: Any = None,
    default_max_total_seconds: float | None = None,
) -> dict[str, Any]:
    defaults = default_execution_constraints(
        default_max_total_seconds=default_max_total_seconds
    )
    complete_structured = getattr(gateway, "complete_structured", None)
    if not callable(complete_structured):
        return defaults
    goal = str(text or "").strip()
    if not goal:
        return defaults
    system_prompt = (
        "你负责从用户请求中抽取执行约束，输出结构化 JSON。"
        "不要规划任务，不要解释，只抽取用户明确表达或强烈暗示的执行限制与偏好。"
        "数字字段必须输出阿拉伯数字或 null；无法确定时输出 null。"
    )
    user_prompt = (
        "请根据下面的用户请求，提取执行约束。\n"
        "字段含义：\n"
        "- hard_limits.max_rounds: 多Agent讨论允许的最大轮数\n"
        "- hard_limits.max_agents: 允许的最大Agent数量\n"
        "- hard_limits.max_total_seconds: 整体执行预算秒数\n"
        "- hard_limits.max_turn_seconds: 单个agent发言/turn预算秒数\n"
        "- soft_preferences.resolution_style: balanced/single_pass/fast_consensus/thorough\n"
        "- degradation.on_budget_exhausted: summarize_partial/raise_timeout\n\n"
        f"用户请求：{goal}"
    )
    structured = await complete_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_cls=ExecutionConstraintsModel,
        heartbeat=heartbeat,
    )
    payload = normalize_execution_constraints(_structured_to_dict(structured))
    if payload["hard_limits"].get("max_total_seconds") is None:
        payload["hard_limits"]["max_total_seconds"] = default_max_total_seconds
    if not payload.get("degradation"):
        payload["degradation"] = defaults["degradation"]
    if not payload.get("soft_preferences"):
        payload["soft_preferences"] = defaults["soft_preferences"]
    return payload


def build_execution_constraint_hints(constraints: Any) -> list[str]:
    normalized = normalize_execution_constraints(constraints)
    hard_limits = normalized["hard_limits"]
    soft_preferences = normalized["soft_preferences"]
    hints: list[str] = []
    if hard_limits.get("max_rounds") is not None:
        hints.append(
            f"若选择多Agent讨论，最大讨论轮数不得超过 {int(hard_limits['max_rounds'])} 轮。"
        )
    if hard_limits.get("max_total_seconds") is not None:
        hints.append(
            f"整体执行预算约为 {int(float(hard_limits['max_total_seconds']))} 秒，预算不足时应优先总结已有结果。"
        )
    if soft_preferences.get("resolution_style") == "single_pass":
        hints.append("用户偏好单轮收敛，避免默认扩展为多轮辩论。")
    elif soft_preferences.get("resolution_style") == "fast_consensus":
        hints.append("用户偏好快速收敛，优先给出结论而不是展开冗长讨论。")
    elif soft_preferences.get("resolution_style") == "thorough":
        hints.append("用户偏好充分讨论，可以在预算内适度展开论证。")
    return hints


def format_execution_constraints_for_prompt(constraints: Any) -> str:
    normalized = normalize_execution_constraints(constraints)
    hard_limits = normalized["hard_limits"]
    soft_preferences = normalized["soft_preferences"]
    degradation = normalized["degradation"]
    lines = [f"- mode: {normalized['mode']}"]
    if hard_limits.get("max_rounds") is not None:
        lines.append(f"- max_rounds: {int(hard_limits['max_rounds'])}")
    if hard_limits.get("max_agents") is not None:
        lines.append(f"- max_agents: {int(hard_limits['max_agents'])}")
    if hard_limits.get("max_total_seconds") is not None:
        lines.append(f"- max_total_seconds: {float(hard_limits['max_total_seconds']):.1f}")
    if hard_limits.get("max_turn_seconds") is not None:
        lines.append(f"- max_turn_seconds: {float(hard_limits['max_turn_seconds']):.1f}")
    lines.append(f"- resolution_style: {soft_preferences['resolution_style']}")
    lines.append(f"- on_budget_exhausted: {degradation['on_budget_exhausted']}")
    return "\n".join(lines)


def build_team_execution_policy(
    args: dict[str, Any],
    constraints: Any,
) -> TeamExecutionPolicy:
    normalized = normalize_execution_constraints(constraints)
    hard_limits = normalized["hard_limits"]
    soft_preferences = normalized["soft_preferences"]
    degradation = normalized["degradation"]
    raw_rounds = args.get("max_rounds")
    requested_rounds = _coerce_int(raw_rounds)
    user_round_cap = hard_limits.get("max_rounds")
    if requested_rounds is None and user_round_cap is None:
        max_rounds = 5
    elif requested_rounds is None:
        max_rounds = int(user_round_cap)
    elif user_round_cap is None:
        max_rounds = requested_rounds
    else:
        max_rounds = min(requested_rounds, int(user_round_cap))
    max_agents = hard_limits.get("max_agents")
    return TeamExecutionPolicy(
        max_rounds=max(1, int(max_rounds)),
        max_agents=(int(max_agents) if max_agents is not None else None),
        max_total_seconds=(
            float(hard_limits["max_total_seconds"])
            if hard_limits.get("max_total_seconds") is not None
            else None
        ),
        max_turn_seconds=(
            float(hard_limits["max_turn_seconds"])
            if hard_limits.get("max_turn_seconds") is not None
            else None
        ),
        resolution_style=soft_preferences["resolution_style"],
        on_budget_exhausted=degradation["on_budget_exhausted"],
    )
