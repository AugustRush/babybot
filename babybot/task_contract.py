"""User-facing task contract for harness-led orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent_kernel.execution_constraints import normalize_execution_constraints

_DEBATE_KEYWORDS = ("讨论", "辩论", "专家", "评审", "对比", "比较", "胜负")
_NO_CLARIFICATION_KEYWORDS = (
    "直接给我最终答案",
    "直接给最终答案",
    "不要追问",
    "不要问我",
    "不要澄清",
    "无需确认",
)
_SINGLE_ROUND_KEYWORDS = ("一轮定胜负", "一轮", "单轮", "一回合")

_ANSWER_ALLOWED_TOOLS = (
    "dispatch_task",
    "wait_for_tasks",
    "get_task_result",
    "reply_to_user",
)
_DEBATE_ALLOWED_TOOLS = (
    "dispatch_team",
    "reply_to_user",
)


@dataclass(frozen=True)
class TaskContract:
    chat_key: str
    goal: str
    mode: str
    deliverable: str
    round_budget: int | None
    termination_rule: str
    allow_clarification: bool
    allowed_tools: tuple[str, ...] = ()
    allowed_agents: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _infer_round_budget(goal: str, constraints: dict[str, Any]) -> int | None:
    configured = constraints.get("hard_limits", {}).get("max_rounds")
    if configured is not None:
        return int(configured)
    if any(token in goal for token in _SINGLE_ROUND_KEYWORDS):
        return 1
    return None


def _infer_mode(goal: str, round_budget: int | None) -> str:
    if round_budget is not None:
        return "debate"
    if any(token in goal for token in _DEBATE_KEYWORDS):
        return "debate"
    return "answer"


def _infer_termination_rule(goal: str, round_budget: int | None) -> str:
    if round_budget == 1 or any(token in goal for token in _SINGLE_ROUND_KEYWORDS):
        return "single_round"
    if round_budget is not None:
        return "round_budget"
    return "final_answer"


def _infer_allow_clarification(goal: str) -> bool:
    return not any(token in goal for token in _NO_CLARIFICATION_KEYWORDS)


def _infer_allowed_tools(mode: str) -> tuple[str, ...]:
    if mode == "debate":
        return _DEBATE_ALLOWED_TOOLS
    return _ANSWER_ALLOWED_TOOLS


def build_task_contract(
    *,
    user_input: str,
    chat_key: str,
    execution_constraints: Any = None,
) -> TaskContract:
    goal = str(user_input or "").strip()
    normalized_constraints = normalize_execution_constraints(execution_constraints)
    round_budget = _infer_round_budget(goal, normalized_constraints)
    mode = _infer_mode(goal, round_budget)
    metadata: dict[str, Any] = {}
    if execution_constraints is not None:
        metadata["execution_constraints"] = normalized_constraints
    return TaskContract(
        chat_key=chat_key,
        goal=goal,
        mode=mode,
        deliverable="final_answer",
        round_budget=round_budget,
        termination_rule=_infer_termination_rule(goal, round_budget),
        allow_clarification=_infer_allow_clarification(goal),
        allowed_tools=_infer_allowed_tools(mode),
        metadata=metadata,
    )


def assert_runtime_matches_contract(
    contract: TaskContract,
    *,
    max_rounds: int | None,
) -> None:
    if contract.round_budget is None:
        return
    if max_rounds is None or int(max_rounds) != int(contract.round_budget):
        raise ValueError(
            "runtime round budget does not match task contract round budget"
        )
