from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import time

from .context import _extract_keywords
from .memory_models import MemoryRecord

if TYPE_CHECKING:
    from .agent_kernel.model import ModelMessage
    from .memory_store import HybridMemoryStore


@dataclass
class ContextView:
    hot: list[str] = field(default_factory=list)
    warm: list[str] = field(default_factory=list)
    cold: list[str] = field(default_factory=list)


def build_context_view(
    *,
    memory_store: "HybridMemoryStore",
    chat_id: str,
    query: str = "",
) -> ContextView:
    view = ContextView()
    records = memory_store.list_memories(chat_id=chat_id)
    query_keywords = _extract_keywords(query) if query else []
    hot_records: list[tuple[float, str]] = []
    warm_records: list[tuple[float, str]] = []
    cold_records: list[tuple[float, str]] = []

    for record in records:
        score = _memory_score(record, query_keywords)
        if record.tier == "hard":
            hot_records.append((score, record.summary))
            continue
        if record.tier == "ephemeral" and record.memory_type == "task_state":
            if query_keywords and _memory_keyword_hits(record, query_keywords) == 0:
                warm_records.append((score, record.summary))
            else:
                hot_records.append((score, record.summary))
            continue
        if record.tier == "soft" and record.memory_type in {
            "relationship_policy",
            "user_profile",
            "task_decision",
        }:
            if record.status == "decaying":
                cold_records.append((score, record.summary))
                continue
            warm_records.append((score, record.summary))
            continue
        cold_records.append((score, record.summary))

    view.hot = _dedupe(_sorted_summaries(hot_records))[:6]
    view.warm = _dedupe(_sorted_summaries(warm_records))[:6]
    view.cold = _dedupe(_sorted_summaries(cold_records))[:6]
    return view


def build_context_view_messages(
    *,
    memory_store: "HybridMemoryStore",
    chat_id: str,
    query: str = "",
) -> list["ModelMessage"]:
    from .agent_kernel.model import ModelMessage

    view = build_context_view(memory_store=memory_store, chat_id=chat_id, query=query)
    messages: list[ModelMessage] = []
    if view.hot:
        messages.append(
            ModelMessage(
                role="system",
                content="[Hot Context]\n" + "\n".join(f"- {line}" for line in view.hot),
            )
        )
    if view.warm:
        messages.append(
            ModelMessage(
                role="system",
                content="[Warm Context]\n"
                + "\n".join(f"- {line}" for line in view.warm),
            )
        )
    if view.cold:
        messages.append(
            ModelMessage(
                role="system",
                content="[Cold Context]\n"
                + "\n".join(f"- {line}" for line in view.cold),
            )
        )
    return messages


def summarize_context_view(
    *,
    memory_store: "HybridMemoryStore",
    chat_id: str,
    query: str = "",
) -> str:
    view = build_context_view(memory_store=memory_store, chat_id=chat_id, query=query)
    parts: list[str] = []
    if view.hot:
        parts.append("Hot:\n" + "\n".join(f"- {line}" for line in view.hot))
    if view.warm:
        parts.append("Warm:\n" + "\n".join(f"- {line}" for line in view.warm))
    if view.cold:
        parts.append("Cold:\n" + "\n".join(f"- {line}" for line in view.cold))
    return "\n".join(parts)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _sorted_summaries(items: list[tuple[float, str]]) -> list[str]:
    return [
        summary for _, summary in sorted(items, key=lambda item: item[0], reverse=True)
    ]


def _memory_score(record: MemoryRecord, query_keywords: list[str]) -> float:
    base = {
        "hard": 100.0,
        "ephemeral": 70.0,
        "soft": 40.0,
    }.get(record.tier, 10.0)
    if record.memory_type == "task_state":
        base += 10.0
    if record.memory_type == "task_decision":
        base += 6.0
    if record.memory_type in {"relationship_policy", "user_profile"}:
        base += 4.0
    base += {
        "active": 8.0,
        "candidate": 3.0,
        "decaying": -8.0,
    }.get(record.status, 0.0)
    hits = _memory_keyword_hits(record, query_keywords)
    # Decay recency over hours: a record updated just now scores 10, one
    # updated >10 hours ago scores 0.
    age_hours = max(0.0, (time.time() - record.updated_at)) / 3600.0
    recency = max(0.0, 10.0 - age_hours)
    confidence = max(0.0, min(1.0, float(record.confidence))) * 5.0
    return base + hits * 20.0 + confidence + recency


def _memory_keyword_hits(record: MemoryRecord, query_keywords: list[str]) -> int:
    if not query_keywords:
        return 0
    haystack = _record_search_text(record)
    return sum(1 for keyword in query_keywords if keyword and keyword in haystack)


def _record_search_text(record: MemoryRecord) -> str:
    value = record.value
    if isinstance(value, list):
        value_text = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value_text = " ".join(f"{k} {v}" for k, v in value.items())
    else:
        value_text = str(value)
    return " ".join(
        part
        for part in (
            record.memory_type,
            record.key,
            record.summary,
            value_text,
            " ".join(record.tags),
        )
        if part
    )
