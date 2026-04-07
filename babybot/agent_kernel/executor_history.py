"""History-building helper functions extracted from executor.py."""

from __future__ import annotations

import json as _json
from typing import Any

from ..feedback_events import (
    normalize_runtime_feedback_event,
    runtime_event_primary_label,
)
from .model import ModelMessage


# ── Kernel-internal helpers (language-agnostic fallbacks) ────────────────────


def _estimate_token_count(text: str) -> int:
    """Cheap token-count estimate: ~3 characters per token."""
    return max(1, len(str(text or "")) // 3)


def _extract_keywords(text: str) -> list[str]:
    """Very lightweight keyword extractor: unique words ≥ 3 chars."""
    words = str(text or "").lower().split()
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        w = w.strip(".,!?;:\"'()[]{}，。！？；：")
        if len(w) >= 3 and w not in seen:
            seen.add(w)
            result.append(w)
    return result[:20]


def _build_context_view_messages(
    memory_store: Any,
    chat_id: str,
    query: str,
) -> list[ModelMessage]:
    """Delegate to application-layer context_views if available, else return []."""
    try:
        from ..context_views import build_context_view_messages  # type: ignore[import]

        return build_context_view_messages(  # type: ignore[return-value]
            memory_store=memory_store,
            chat_id=chat_id,
            query=query,
        )
    except Exception:
        return []


def _history_entry_text(entry: object) -> str:
    kind = getattr(entry, "kind", "")
    payload = getattr(entry, "payload", {}) or {}
    if kind == "message":
        role = payload.get("role", "?")
        content = payload.get("content", "")
        return f"{role}: {content}"
    if kind == "tool_result":
        name = str(payload.get("name", "") or "?")
        status = "ok" if payload.get("ok") else "failed"
        preview = str(payload.get("content_preview", "") or "").strip()
        artifacts = payload.get("artifacts") or []
        suffix = (
            f"\nartifacts: {', '.join(str(item) for item in artifacts)}"
            if artifacts
            else ""
        )
        return f"[tool_result][{status}] {name}: {preview}{suffix}".rstrip()
    if kind == "tool_call":
        name = str(payload.get("name", "") or "?")
        arguments = payload.get("arguments", {})
        return f"[tool_call] {name}: {_json.dumps(arguments, ensure_ascii=False)}"
    if kind == "event":
        event_name = str(payload.get("event", "") or "?")
        event_payload = payload.get("payload") or {}
        normalized = normalize_runtime_feedback_event(
            {
                "event": event_name,
                "task_id": str(payload.get("task_id", "") or ""),
                "flow_id": str(payload.get("flow_id", "") or ""),
                "payload": dict(event_payload or {}),
            }
        )
        description = runtime_event_primary_label(normalized)
        error = str(normalized.error or "").strip()
        output = str(
            normalized.output_summary or event_payload.get("output", "") or ""
        ).strip()
        details_parts = [part for part in (description, output, error) if part]
        details = (
            " | ".join(details_parts)
            if details_parts
            else _json.dumps(event_payload, ensure_ascii=False)
        )
        return f"[event] {event_name}: {details}"
    if kind == "anchor":
        state = payload.get("state") or {}
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        return f"[anchor_summary] {summary}".strip()
    return ""


def _build_history_messages(
    tape: object,
    token_budget: int,
    query: str = "",
    tape_store: object | None = None,
    memory_store: object | None = None,
) -> list[ModelMessage]:
    """Build history context messages from a Tape.

    Three sections (all sharing token_budget):
    1. Anchor summary → system message
    2. BM25 cross-anchor recall → [relevant_history] system message
    3. Recent entries since anchor → user/assistant messages
    """
    messages: list[ModelMessage] = []

    last_anchor = getattr(tape, "last_anchor", None)
    if last_anchor is None:
        return messages
    anchor = last_anchor()
    budget_remaining = max(0, int(token_budget))

    chat_id = getattr(tape, "chat_id", "")
    if memory_store is not None and chat_id:
        load_assistant_profile = getattr(memory_store, "load_assistant_profile", None)
        if callable(load_assistant_profile):
            assistant_profile = str(load_assistant_profile() or "").strip()
            if assistant_profile:
                profile_text = "[Assistant Profile]\n" + assistant_profile
                profile_cost = max(1, _estimate_token_count(profile_text))
                if budget_remaining >= profile_cost:
                    messages.append(ModelMessage(role="system", content=profile_text))
                    budget_remaining -= profile_cost
        memory_messages = _build_context_view_messages(
            memory_store=memory_store,
            chat_id=chat_id,
            query=query,
        )
        for message in memory_messages:
            cost = max(1, _estimate_token_count(message.content))
            if budget_remaining < cost:
                continue
            messages.append(message)
            budget_remaining -= cost

    # 1. Anchor summary → system message (with structured fields if available)
    if anchor is not None:
        state = anchor.payload.get("state", {})
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        if summary:
            parts = [f"[conversation_context]\n{summary}"]
            entities = state.get("entities")
            if entities and isinstance(entities, list):
                parts.append(f"key_entities: {', '.join(entities)}")
            intent = state.get("user_intent")
            if intent:
                parts.append(f"user_intent: {intent}")
            pending = state.get("pending")
            if pending:
                parts.append(f"pending: {pending}")
            next_steps = state.get("next_steps")
            if next_steps and isinstance(next_steps, list):
                parts.append(
                    f"next_steps: {', '.join(str(item) for item in next_steps)}"
                )
            open_questions = state.get("open_questions")
            if open_questions and isinstance(open_questions, list):
                parts.append(
                    f"open_questions: {', '.join(str(item) for item in open_questions)}"
                )
            decisions = state.get("decisions")
            if decisions and isinstance(decisions, list):
                parts.append(f"decisions: {', '.join(str(item) for item in decisions)}")
            artifacts = state.get("artifacts")
            if artifacts and isinstance(artifacts, list):
                parts.append(f"artifacts: {', '.join(str(item) for item in artifacts)}")
            anchor_text = "\n".join(parts)
            anchor_cost = len(anchor_text) // 3
            if budget_remaining >= anchor_cost:
                messages.append(ModelMessage(role="system", content=anchor_text))
                budget_remaining -= anchor_cost

    # Collect recent entries (for both section 2 exclusion and section 3)
    entries_since = getattr(tape, "entries_since_anchor", None)
    if entries_since is None:
        return messages

    recent = entries_since()
    msg_entries = [e for e in recent if e.kind == "message"]
    recent_state_entries = [e for e in recent if e.kind in {"tool_result", "event"}]
    # Exclude the last user message (it's the current turn, added by executor)
    if msg_entries and msg_entries[-1].payload.get("role") == "user":
        msg_entries = msg_entries[:-1]

    # 2. BM25 cross-anchor recall — search for relevant entries before the anchor
    search_fn = getattr(tape_store, "search_relevant", None) if tape_store else None
    chat_id = getattr(tape, "chat_id", None)
    if search_fn and chat_id and query:
        recent_ids = {e.entry_id for e in recent}
        recall_budget = budget_remaining // 4  # Reserve up to 25% for recall
        try:
            recalled = search_fn(chat_id, query, limit=5, exclude_ids=recent_ids)
        except Exception:
            recalled = []

        if recalled:
            recall_lines: list[str] = []
            recall_tokens = 0
            for entry in recalled:
                est = max(1, int(entry.token_estimate))
                if recall_tokens + est > recall_budget:
                    break
                line = _history_entry_text(entry)
                if not line:
                    continue
                recall_lines.append(line)
                recall_tokens += est
            if recall_lines:
                messages.append(
                    ModelMessage(
                        role="system",
                        content="[relevant_history]\n" + "\n".join(recall_lines),
                    )
                )
                budget_remaining -= recall_tokens

    # 2.5. Recent non-message execution state (tool results / failed events)
    if recent_state_entries and budget_remaining > 0:
        state_lines: list[str] = []
        state_tokens = 0
        for entry in recent_state_entries[-5:]:
            if entry.kind == "event" and entry.payload.get("event") not in {
                "failed",
                "dead_lettered",
                "stalled",
            }:
                continue
            line = _history_entry_text(entry)
            if not line:
                continue
            est = max(1, int(entry.token_estimate))
            if state_tokens + est > max(1, budget_remaining // 3):
                continue
            state_lines.append(line)
            state_tokens += est
        if state_lines:
            messages.append(
                ModelMessage(
                    role="system",
                    content="[近期执行状态]\n" + "\n".join(state_lines),
                )
            )
            budget_remaining -= state_tokens

    # 3. Recent entries → hybrid recency+relevance scoring
    if msg_entries and query:
        kws = _extract_keywords(query)
    else:
        kws = []

    n = len(msg_entries)
    scored_entries: list[tuple[float, int, object]] = []
    for idx, entry in enumerate(msg_entries):
        # Recency: linear 0→1, most recent = 1.0
        recency = (idx + 1) / n if n else 0.0
        # Relevance: fraction of keywords found in content
        if kws:
            content = entry.payload.get("content", "")
            hits = sum(1 for kw in kws if kw in content)
            relevance = hits / len(kws)
        else:
            relevance = 0.0
        # Weighted blend: recency dominates (0.6) to preserve conversation flow
        score = 0.6 * recency + 0.4 * relevance
        scored_entries.append((score, idx, entry))

    # Sort by score descending, greedily pick within budget
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    picked_indices: set[int] = set()
    for score, idx, entry in scored_entries:
        if budget_remaining <= 0:
            break
        est = max(1, int(entry.token_estimate))
        if est > budget_remaining:
            continue
        budget_remaining -= est
        picked_indices.add(idx)

    # Emit in original chronological order
    for idx, entry in enumerate(msg_entries):
        if idx in picked_indices:
            messages.append(
                ModelMessage(
                    role=entry.payload["role"],
                    content=entry.payload["content"],
                )
            )

    return messages
