from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .context_views import build_context_view

logger = logging.getLogger(__name__)

_TRIVIAL_SOCIAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
}

_EXPLICIT_DEBATE_MARKERS = (
    "辩论",
    "专家讨论",
    "正方",
    "反方",
    "支持方",
    "反对方",
    "不同角色",
    "不同观点",
)

_RETRIEVE_FIRST_MARKERS = (
    "查一下",
    "查询",
    "搜索",
    "检索",
    "找一下",
    "看看",
    "浏览",
    "打开",
    "天气",
    "汇率",
    "股价",
    "文档",
    "资料",
    "状态",
)

_ANALYZE_FIRST_MARKERS = (
    "修复",
    "修改",
    "实现",
    "编写",
    "生成",
    "画",
    "创建",
    "整理",
    "总结",
    "分析",
    "提取",
    "翻译",
    "测试",
    "运行",
)


def is_trivial_social_message(text: str) -> bool:
    normalized = (
        str(text or "")
        .strip()
        .lower()
        .replace("！", "")
        .replace("!", "")
        .replace("。", "")
        .replace(".", "")
        .replace("？", "")
        .replace("?", "")
        .replace("呀", "")
        .replace("啊", "")
        .replace("哈", "")
    )
    return normalized in _TRIVIAL_SOCIAL_MESSAGES


def _estimate_subtask_count(text: str) -> int:
    count = 1
    for token in ("同时", "分别", "并行", "并且"):
        count += str(text or "").count(token)
    return max(1, count)


class RoutingDecision(BaseModel):
    route_mode: Literal["tool_workflow", "answer", "debate"] = "tool_workflow"
    need_clarification: bool = False
    execution_style: Literal[
        "direct_execute",
        "analyze_first",
        "retrieve_first",
        "verify_first",
    ] = "analyze_first"
    parallelism_hint: Literal["serial", "bounded_parallel"] = "serial"
    worker_hint: Literal["allow", "deny"] = "deny"
    explain: str = Field(default="", max_length=200)
    decision_source: Literal["model", "rule", "reflection", "intent_cache"] = "model"


def build_routing_intent_bucket(goal: str, *, has_media: bool = False) -> str:
    text = str(goal or "").strip()
    if is_trivial_social_message(text):
        kind = "social"
    elif any(marker in text for marker in _EXPLICIT_DEBATE_MARKERS):
        kind = "debate"
    elif any(marker in text for marker in _RETRIEVE_FIRST_MARKERS):
        kind = "retrieve"
    elif any(marker in text for marker in _ANALYZE_FIRST_MARKERS):
        kind = "analyze"
    else:
        kind = "other"
    question_flag = (
        "q1"
        if any(token in text for token in ("?", "？", "吗", "么", "是否", "可不可以"))
        else "q0"
    )
    subtask_flag = "s2p" if _estimate_subtask_count(text) >= 2 else "s1"
    text_length = len(text)
    if text_length <= 12:
        length_flag = "l_short"
    elif text_length <= 32:
        length_flag = "l_medium"
    else:
        length_flag = "l_long"
    media_flag = "m1" if has_media else "m0"
    return "|".join((kind, question_flag, subtask_flag, length_flag, media_flag))


@dataclass(frozen=True)
class RoutingContextSnapshot:
    chat_key: str
    goal: str
    anchor_summary: str = ""
    hot_context: tuple[str, ...] = ()
    warm_context: tuple[str, ...] = ()
    cold_context: tuple[str, ...] = ()
    runtime_state: str = ""
    runtime_progress: str = ""
    recent_flow_ids: tuple[str, ...] = ()
    execution_constraints: dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "chat_key": self.chat_key,
            "goal": self.goal,
            "anchor_summary": self.anchor_summary,
            "hot_context": list(self.hot_context),
            "warm_context": list(self.warm_context),
            "cold_context": list(self.cold_context),
            "runtime_state": self.runtime_state,
            "runtime_progress": self.runtime_progress,
            "recent_flow_ids": list(self.recent_flow_ids),
            "execution_constraints": self.execution_constraints,
        }


def build_routing_snapshot(
    *,
    chat_key: str,
    goal: str,
    tape: Any = None,
    memory_store: Any = None,
    runtime_job: Any = None,
    recent_flow_ids: list[str] | tuple[str, ...] | None = None,
    execution_constraints: dict[str, Any] | None = None,
) -> RoutingContextSnapshot:
    anchor_summary = ""
    if tape is not None and hasattr(tape, "last_anchor"):
        anchor = tape.last_anchor()
        if anchor is not None:
            state = anchor.payload.get("state") if isinstance(anchor.payload, dict) else {}
            anchor_summary = str((state or {}).get("summary", "") or "").strip()
    hot_context: tuple[str, ...] = ()
    warm_context: tuple[str, ...] = ()
    cold_context: tuple[str, ...] = ()
    if chat_key and memory_store is not None:
        try:
            view = build_context_view(memory_store=memory_store, chat_id=chat_key, query=goal)
        except Exception:
            logger.exception("Failed to build routing context view")
        else:
            hot_context = tuple(view.hot[:3])
            warm_context = tuple(view.warm[:3])
            cold_context = tuple(view.cold[:2])
    runtime_state = ""
    runtime_progress = ""
    if runtime_job is not None:
        runtime_state = str(getattr(runtime_job, "state", "") or "").strip()
        runtime_progress = str(getattr(runtime_job, "progress_message", "") or "").strip()
    return RoutingContextSnapshot(
        chat_key=str(chat_key or "").strip(),
        goal=str(goal or "").strip(),
        anchor_summary=anchor_summary,
        hot_context=hot_context,
        warm_context=warm_context,
        cold_context=cold_context,
        runtime_state=runtime_state,
        runtime_progress=runtime_progress,
        recent_flow_ids=tuple(
            item for item in (recent_flow_ids or ()) if str(item).strip()
        )[:3],
        execution_constraints=dict(execution_constraints or {}),
    )


def route_mode_to_contract_mode(route_mode: str) -> str:
    return "debate" if str(route_mode or "").strip().lower() == "debate" else "answer"


def route_mode_to_step_kind(route_mode: str) -> str:
    return "debate" if str(route_mode or "").strip().lower() == "debate" else "tool_workflow"


def match_rule_based_routing(snapshot: RoutingContextSnapshot) -> RoutingDecision | None:
    goal = str(snapshot.goal or "").strip()
    if not goal:
        return None
    if is_trivial_social_message(goal):
        return RoutingDecision(
            route_mode="answer",
            need_clarification=False,
            execution_style="direct_execute",
            parallelism_hint="serial",
            worker_hint="deny",
            explain="简单问候直接回复",
            decision_source="rule",
        )
    if any(marker in goal for marker in _EXPLICIT_DEBATE_MARKERS):
        return RoutingDecision(
            route_mode="debate",
            need_clarification=False,
            execution_style="analyze_first",
            parallelism_hint="serial",
            worker_hint="deny",
            explain="显式多观点讨论请求",
            decision_source="rule",
        )
    subtask_count = _estimate_subtask_count(goal)
    parallelism_hint: Literal["serial", "bounded_parallel"] = (
        "bounded_parallel" if subtask_count >= 2 else "serial"
    )
    if any(marker in goal for marker in _RETRIEVE_FIRST_MARKERS):
        return RoutingDecision(
            route_mode="tool_workflow",
            need_clarification=False,
            execution_style="retrieve_first",
            parallelism_hint=parallelism_hint,
            worker_hint="deny",
            explain="显式查询或检索请求",
            decision_source="rule",
        )
    if any(marker in goal for marker in _ANALYZE_FIRST_MARKERS):
        return RoutingDecision(
            route_mode="tool_workflow",
            need_clarification=False,
            execution_style="analyze_first",
            parallelism_hint=parallelism_hint,
            worker_hint="deny",
            explain="显式执行型任务请求",
            decision_source="rule",
        )
    return None


async def route_task(
    gateway: Any,
    snapshot: RoutingContextSnapshot,
    *,
    heartbeat: Any = None,
    model_name: str = "",
    timeout: float = 2.0,
    allow_rule_based: bool = True,
) -> RoutingDecision | None:
    goal = str(snapshot.goal or "").strip()
    if not goal:
        return None
    if allow_rule_based:
        rule_match = match_rule_based_routing(snapshot)
        if rule_match is not None:
            return rule_match
    complete_structured = getattr(gateway, "complete_structured", None)
    if not callable(complete_structured):
        return None
    system_prompt = (
        "你是一个轻量任务路由器，只做一次保守判定。"
        "优先保持 tool_workflow，只有明显需要多方比较时才升级为 debate。"
    )
    user_prompt = (
        "请基于下面的紧凑上下文快照输出结构化 JSON。\n"
        "- route_mode: tool_workflow / answer / debate\n"
        "- need_clarification: 只有信息缺失会阻塞执行时才为 true\n"
        "- execution_style: direct_execute / analyze_first / retrieve_first / verify_first\n"
        "- parallelism_hint: serial / bounded_parallel\n"
        "- worker_hint: allow / deny\n"
        "- explain: 中文一句话，不超过40字\n\n"
        f"上下文快照：{json.dumps(snapshot.to_prompt_payload(), ensure_ascii=False)}"
    )
    try:
        try:
            structured = await asyncio.wait_for(
                complete_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_cls=RoutingDecision,
                    heartbeat=heartbeat,
                    model_name=str(model_name or "").strip() or None,
                    timeout=max(0.5, float(timeout or 0.0)),
                ),
                timeout=max(0.5, float(timeout or 0.0) + 0.2),
            )
        except TypeError:
            structured = await asyncio.wait_for(
                complete_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_cls=RoutingDecision,
                    heartbeat=heartbeat,
                ),
                timeout=max(0.5, float(timeout or 0.0) + 0.2),
            )
    except asyncio.TimeoutError:
        logger.warning(
            "Routing decision timed out after %.2fs; falling back to default contract",
            max(0.5, float(timeout or 0.0)),
        )
        return None
    except Exception:
        logger.exception("Routing decision failed; falling back to default contract")
        return None
    if structured is None:
        return None
    if isinstance(structured, RoutingDecision):
        return structured
    payload = (
        structured.model_dump()
        if hasattr(structured, "model_dump")
        else dict(structured)
        if isinstance(structured, dict)
        else None
    )
    if not isinstance(payload, dict):
        return None
    required_keys = {
        "route_mode",
        "need_clarification",
        "execution_style",
        "parallelism_hint",
        "worker_hint",
    }
    if not required_keys.issubset(payload):
        return None
    try:
        return RoutingDecision.model_validate(payload)
    except Exception:
        logger.exception("Invalid routing decision payload")
        return None
