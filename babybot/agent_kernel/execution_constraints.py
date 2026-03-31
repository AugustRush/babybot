"""Execution-constraint parsing and normalization for orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..orchestration_router import is_trivial_social_message


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


_CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_cn_number(text: str) -> int | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _CN_NUM_MAP:
        return _CN_NUM_MAP[raw]
    if raw.startswith("十"):
        tail = _CN_NUM_MAP.get(raw[1:], 0) if len(raw) > 1 else 0
        return 10 + int(tail)
    if raw.endswith("十"):
        head = _CN_NUM_MAP.get(raw[:-1], 1)
        return int(head) * 10
    if "十" in raw:
        head, tail = raw.split("十", 1)
        high = _CN_NUM_MAP.get(head, 1)
        low = _CN_NUM_MAP.get(tail, 0)
        return int(high) * 10 + int(low)
    return None


def _parse_loose_number(raw: str) -> int | None:
    value = _coerce_int(raw)
    if value is not None:
        return value
    return _parse_cn_number(raw)


def _parse_duration_to_seconds(value: str, unit: str) -> float | None:
    amount = _coerce_float(value)
    if amount is None:
        parsed = _parse_cn_number(value)
        amount = float(parsed) if parsed is not None else None
    if amount is None:
        return None
    normalized_unit = str(unit or "").strip().lower()
    if normalized_unit in {"秒", "秒钟", "second", "seconds", "s"}:
        return float(amount)
    if normalized_unit in {"分钟", "分", "minute", "minutes", "min", "mins", "m"}:
        return float(amount) * 60.0
    if normalized_unit in {"小时", "时", "hour", "hours", "h"}:
        return float(amount) * 3600.0
    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _infer_hard_limits(goal: str) -> dict[str, Any]:
    hard_limits: dict[str, Any] = {}

    round_matches = [
        re.search(pattern, goal)
        for pattern in (
            r"(?:最多|至多|不超过|最多进行)?\s*([0-9零〇一二两三四五六七八九十]+)\s*轮",
            r"([0-9零〇一二两三四五六七八九十]+)\s*轮(?:内|结束|搞定|定胜负)?",
        )
    ]
    for match in round_matches:
        if not match:
            continue
        parsed = _parse_loose_number(match.group(1))
        if parsed is not None:
            hard_limits["max_rounds"] = max(1, int(parsed))
            break
    if "单轮" in goal or "一轮定胜负" in goal:
        hard_limits["max_rounds"] = 1

    agent_matches = [
        re.search(pattern, goal)
        for pattern in (
            r"([0-9零〇一二两三四五六七八九十]+)\s*(?:个)?(?:专家|agent|智能体|助手|人)(?:讨论|协作|评审|辩论)?",
            r"(?:最多|至多|不超过)\s*([0-9零〇一二两三四五六七八九十]+)\s*(?:个)?(?:专家|agent|智能体|助手|人)",
        )
    ]
    for match in agent_matches:
        if not match:
            continue
        parsed = _parse_loose_number(match.group(1))
        if parsed is not None:
            hard_limits["max_agents"] = max(1, int(parsed))
            break

    total_budget_match = re.search(
        r"(?:总时长|整体执行预算|执行预算|总预算|限时|在)\s*(?:不超过|最多|控制在|限制在)?\s*([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十]+)\s*(秒钟?|秒|分钟?|分|小时|时)",
        goal,
    )
    if total_budget_match:
        seconds = _parse_duration_to_seconds(
            total_budget_match.group(1), total_budget_match.group(2)
        )
        if seconds is not None:
            hard_limits["max_total_seconds"] = seconds

    turn_budget_match = re.search(
        r"(?:每轮|单轮|每个(?:agent|智能体|专家)?(?:发言|turn)?|单个agent发言/turn预算)\s*(?:不超过|最多|控制在|限制在)?\s*([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十]+)\s*(秒钟?|秒|分钟?|分|小时|时)",
        goal,
    )
    if turn_budget_match:
        seconds = _parse_duration_to_seconds(
            turn_budget_match.group(1), turn_budget_match.group(2)
        )
        if seconds is not None:
            hard_limits["max_turn_seconds"] = seconds

    return hard_limits


def _infer_resolution_style(goal: str) -> str:
    if _contains_any(
        goal,
        (
            "单轮",
            "一轮定胜负",
            "一步到位",
            "直接给最终答案",
            "只要一个方案",
        ),
    ):
        return "single_pass"
    if _contains_any(
        goal,
        (
            "尽快",
            "快速",
            "快点",
            "简洁",
            "简明",
            "先给结论",
            "不要展开",
            "不要长篇讨论",
            "快速收敛",
        ),
    ):
        return "fast_consensus"
    if _contains_any(
        goal,
        (
            "详细",
            "充分",
            "深入",
            "全面",
            "仔细",
            "展开",
            "完整分析",
        ),
    ):
        return "thorough"
    return "balanced"


def _infer_degradation_policy(goal: str) -> str:
    if _contains_any(
        goal,
        (
            "不要部分结果",
            "超时就报错",
            "失败就报错",
            "不要总结已有结果",
            "超时直接失败",
        ),
    ):
        return "raise_timeout"
    return "summarize_partial"


def infer_execution_constraints_from_text(
    text: str,
    *,
    default_max_total_seconds: float | None = None,
) -> dict[str, Any]:
    defaults = default_execution_constraints(
        default_max_total_seconds=default_max_total_seconds
    )
    goal = str(text or "").strip()
    if not goal or is_trivial_social_message(goal):
        return defaults

    hard_limits = _infer_hard_limits(goal)
    if (
        hard_limits.get("max_total_seconds") is None
        and default_max_total_seconds is not None
    ):
        hard_limits["max_total_seconds"] = float(default_max_total_seconds)

    return normalize_execution_constraints(
        {
            "mode": "interactive",
            "hard_limits": hard_limits,
            "soft_preferences": {
                "resolution_style": _infer_resolution_style(goal),
            },
            "degradation": {
                "on_budget_exhausted": _infer_degradation_policy(goal),
            },
        }
    )


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
    del gateway, heartbeat
    return infer_execution_constraints_from_text(
        text,
        default_max_total_seconds=default_max_total_seconds,
    )


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
