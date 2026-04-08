"""HandoffManager — conversation context compaction.

Single Responsibility: When accumulated conversation tokens exceed a threshold,
summarize the history via LLM and create a compact anchor entry.

Key design fix vs the original:
  The original held an asyncio.Lock while awaiting the LLM call (up to 30s),
  blocking all concurrent requests for the same chat_key.

  Fixed protocol:
    1. Acquire lock → snapshot entries → check threshold → release lock.
    2. If threshold exceeded: await LLM OUTSIDE the lock.
    3. Re-acquire lock → write anchor → compact → release lock.

  This reduces lock hold time from ~30s to microseconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from typing import Any

from .context import Tape, _extract_keywords

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = (
    "CRITICAL: 仅输出纯文本 JSON，禁止调用任何工具，禁止任何 markdown 代码块包裹。\n\n"
    "如果提供了 '## 上一次摘要'，请在其基础上整合新对话内容，保留仍然相关的信息，更新已完成的事项，移除已过时的内容。\n\n"
    "请将以下对话历史浓缩为 JSON 格式（用中文填写），严格按以下结构输出，不要输出其他内容：\n"
    '{"summary":"不超过200字的摘要，保留关键事实和已完成操作",'
    '"entities":["提到的关键实体，如人名、物品、话题等，最多5个"],'
    '"user_intent":"用户当前最可能的意图，一句话",'
    '"pending":"未完成的事项，如无则为空字符串",'
    '"next_steps":["建议的下一步，最多3条"],'
    '"artifacts":["重要产物文件名或标识，最多5条"],'
    '"open_questions":["仍需用户确认的问题，最多3条"],'
    '"decisions":["已经确认的重要决定，最多3条"]}\n\n'
)

_HANDOFF_LOCK_LIMIT = 256


class HandoffManager:
    """Manages per-chat handoff locks and context compaction."""

    def __init__(
        self,
        *,
        gateway: Any,
        tape_store: Any,
        memory_store: Any,
        compact_threshold: int,
    ) -> None:
        self._gateway = gateway
        self._tape_store = tape_store
        self._memory_store = memory_store
        self._compact_threshold = compact_threshold
        # LRU of per-chat asyncio.Locks — bounded to prevent unbounded growth.
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def _get_lock(self, chat_key: str) -> asyncio.Lock:
        lock = self._locks.pop(chat_key, None)
        if lock is None:
            lock = asyncio.Lock()
        self._locks[chat_key] = lock
        while len(self._locks) > _HANDOFF_LOCK_LIMIT:
            self._locks.popitem(last=False)
        return lock

    async def maybe_handoff(self, tape: Tape, chat_key: str) -> None:
        """
        Check whether accumulated tokens exceed the threshold.
        If so, summarize via LLM and create a compact anchor.

        Lock protocol (fixes the original hold-lock-during-LLM bug):
          Phase A — under lock: snapshot entries, check threshold
          Phase B — NO lock: await LLM (expensive I/O, up to 30s)
          Phase C — under lock: write anchor, compact entries
        """
        lock = self._get_lock(chat_key)
        try:
            # ── Phase A: check under lock ──────────────────────────────
            async with lock:
                old_entries = tape.entries_since_anchor()
                if not old_entries:
                    return
                total_tokens = sum(e.token_estimate for e in old_entries)
                if total_tokens <= self._compact_threshold:
                    return
                # Build the text to summarize (snapshot under lock)
                lines: list[str] = []
                for e in old_entries:
                    if e.kind == "message":
                        role = e.payload.get("role", "?")
                        content = e.payload.get("content", "")
                        lines.append(f"{role}: {content}")
                if not lines:
                    return
                history_text = "\n".join(lines)
                source_ids = [e.entry_id for e in old_entries]
                prev_anchor = tape.last_anchor()
                prev_summary = ""
                if prev_anchor:
                    prev_summary = str(
                        (prev_anchor.payload.get("state") or {}).get("summary", "")
                        or ""
                    )
            # ── Phase B: LLM call outside lock ────────────────────────
            # Build user content with optional previous summary for
            # incremental updates instead of rebuilding from scratch.
            if prev_summary:
                prev_state = (
                    (prev_anchor.payload.get("state") or {}) if prev_anchor else {}
                )
                prev_context_parts = [f"## 上一次摘要\n{prev_summary}"]
                # Add structured context from previous anchor
                for key in ("entities", "artifacts", "decisions"):
                    items = prev_state.get(key) or []
                    if items:
                        prev_context_parts.append(
                            f"上次{key}: {', '.join(str(x) for x in items)}"
                        )
                user_content = (
                    "\n\n".join(prev_context_parts) + f"\n\n## 最近对话\n{history_text}"
                )
            else:
                user_content = history_text
            raw_summary = await self._gateway.complete(_SUMMARIZE_PROMPT, user_content)

            # Parse structured JSON from LLM, fallback to plain summary
            structured: dict[str, Any] = {}
            try:
                text = raw_summary.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                structured = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                structured = {"summary": raw_summary.strip()}

            summary_text = structured.get("summary", raw_summary.strip())
            entities = structured.get("entities", [])
            next_steps = structured.get("next_steps", [])
            artifacts = structured.get("artifacts", [])
            open_questions = structured.get("open_questions", [])
            decisions = structured.get("decisions", [])
            for lst in (entities, next_steps, artifacts, open_questions, decisions):
                if not isinstance(lst, list):
                    lst = []  # noqa: PLW2901 — local rebind intentional

            # Detect topic shift by comparing keyword overlap
            phase = "continuation"
            if prev_summary:
                prev_kws = set(_extract_keywords(prev_summary, max_keywords=12))
                recent_user_text = " ".join(
                    line.split(":", 1)[-1].strip()
                    for line in lines
                    if line.startswith("user:")
                )
                curr_kws = set(_extract_keywords(recent_user_text, max_keywords=12))
                if prev_kws and curr_kws:
                    overlap = len(prev_kws & curr_kws) / max(
                        len(prev_kws), len(curr_kws)
                    )
                    if overlap < 0.15:
                        phase = "topic_shift"
                        logger.info(
                            "Topic shift detected chat_key=%s overlap=%.2f",
                            chat_key,
                            overlap,
                        )

            # ── Phase C: write results under lock ──────────────────────
            async with lock:
                anchor = tape.append(
                    "anchor",
                    {
                        "name": f"compact/{tape.turn_count()}",
                        "state": {
                            "summary": summary_text,
                            "entities": entities if isinstance(entities, list) else [],
                            "user_intent": structured.get("user_intent", ""),
                            "pending": structured.get("pending", ""),
                            "next_steps": [
                                str(i)
                                for i in (
                                    next_steps if isinstance(next_steps, list) else []
                                )[:3]
                            ],
                            "artifacts": [
                                str(i)
                                for i in (
                                    artifacts if isinstance(artifacts, list) else []
                                )[:5]
                            ],
                            "open_questions": [
                                str(i)
                                for i in (
                                    open_questions
                                    if isinstance(open_questions, list)
                                    else []
                                )[:3]
                            ],
                            "decisions": [
                                str(i)
                                for i in (
                                    decisions if isinstance(decisions, list) else []
                                )[:3]
                            ],
                            "phase": phase,
                            "source_ids": source_ids,
                            "turn_count": tape.turn_count(),
                        },
                    },
                )
                self._tape_store.save_entry(chat_key, anchor)
                if self._memory_store is not None:
                    self._memory_store.observe_anchor_state(
                        chat_key,
                        anchor.payload.get("state") or {},
                        source_ids=source_ids,
                    )
                tape.compact_entries()
                logger.info(
                    "Handoff created anchor chat_key=%s entry_id=%d summarized=%d entries",
                    chat_key,
                    anchor.entry_id,
                    len(source_ids),
                )
        except Exception:
            logger.exception("Error in maybe_handoff for chat_key=%s", chat_key)
