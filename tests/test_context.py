"""Tests for babybot.context — Entry, Tape, TapeStore."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from babybot.context import Entry, Tape, TapeStore


# ── Entry ──────────────────────────────────────────────────────────

class TestEntry:
    def test_message_factory(self):
        e = Entry.message(1, "user", "hello")
        assert e.entry_id == 1
        assert e.kind == "message"
        assert e.payload == {"role": "user", "content": "hello"}
        assert e.timestamp > 0

    def test_anchor_factory(self):
        e = Entry.anchor(2, "session/start", {"summary": "x"})
        assert e.kind == "anchor"
        assert e.payload["name"] == "session/start"
        assert e.payload["state"]["summary"] == "x"

    def test_anchor_factory_default_state(self):
        e = Entry.anchor(1, "start")
        assert e.payload["state"] == {}

    def test_token_estimate(self):
        e = Entry.message(1, "user", "hello world")
        assert e.token_estimate > 0
        # Longer content → higher estimate
        e2 = Entry.message(2, "user", "hello world " * 100)
        assert e2.token_estimate > e.token_estimate

    def test_frozen(self):
        e = Entry.message(1, "user", "hi")
        with pytest.raises(AttributeError):
            e.kind = "anchor"


# ── Tape ───────────────────────────────────────────────────────────

class TestTape:
    def test_append(self):
        tape = Tape("chat1")
        e = tape.append("message", {"role": "user", "content": "hi"})
        assert e.entry_id == 1
        assert len(tape.entries) == 1

    def test_monotonic_ids(self):
        tape = Tape("chat1")
        e1 = tape.append("message", {"role": "user", "content": "a"})
        e2 = tape.append("message", {"role": "assistant", "content": "b"})
        assert e2.entry_id == e1.entry_id + 1

    def test_last_anchor_none(self):
        tape = Tape("chat1")
        tape.append("message", {"role": "user", "content": "a"})
        assert tape.last_anchor() is None

    def test_last_anchor(self):
        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "user", "content": "a"})
        tape.append("anchor", {"name": "compact/1", "state": {"summary": "x"}})
        tape.append("message", {"role": "user", "content": "b"})
        anchor = tape.last_anchor()
        assert anchor is not None
        assert anchor.payload["name"] == "compact/1"

    def test_entries_since_anchor(self):
        tape = Tape("chat1")
        tape.append("message", {"role": "user", "content": "before"})
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "user", "content": "after1"})
        tape.append("message", {"role": "assistant", "content": "after2"})

        since = tape.entries_since_anchor()
        assert len(since) == 2
        assert since[0].payload["content"] == "after1"
        assert since[1].payload["content"] == "after2"

    def test_entries_since_anchor_no_anchor(self):
        tape = Tape("chat1")
        tape.append("message", {"role": "user", "content": "a"})
        tape.append("message", {"role": "user", "content": "b"})
        assert len(tape.entries_since_anchor()) == 2

    def test_total_tokens_since_anchor(self):
        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "user", "content": "hello " * 50})
        tokens = tape.total_tokens_since_anchor()
        assert tokens > 0

    def test_turn_count(self):
        tape = Tape("chat1")
        tape.append("message", {"role": "user", "content": "a"})
        tape.append("message", {"role": "assistant", "content": "b"})
        tape.append("message", {"role": "user", "content": "c"})
        assert tape.turn_count() == 2

    def test_compact_entries(self):
        tape = Tape("chat1")
        tape.append("message", {"role": "user", "content": "old"})
        tape.append("anchor", {"name": "a1", "state": {}})
        tape.append("message", {"role": "user", "content": "new"})

        tape.compact_entries()
        assert len(tape.entries) == 2  # anchor + new message
        assert tape.entries[0].kind == "anchor"
        assert tape.entries[1].payload["content"] == "new"

    def test_init_with_existing_entries(self):
        entries = [
            Entry.message(5, "user", "hi"),
            Entry.message(6, "assistant", "hello"),
        ]
        tape = Tape("chat1", entries)
        e = tape.append("message", {"role": "user", "content": "next"})
        assert e.entry_id == 7


# ── TapeStore ──────────────────────────────────────────────────────

class TestTapeStore:
    def _make_store(self, tmp_path: Path) -> TapeStore:
        return TapeStore(db_path=tmp_path / "test.db", max_chats=3)

    def test_get_or_create_new(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        assert tape.chat_id == "chat1"
        assert len(tape.entries) == 0

    def test_save_and_reload(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append("anchor", {"name": "start", "state": {}})
        store.save_entry("chat1", e1)
        e2 = tape.append("message", {"role": "user", "content": "hello"})
        store.save_entry("chat1", e2)

        # Clear cache and reload from DB
        store.clear()
        tape2 = store.get_or_create("chat1")
        assert len(tape2.entries) == 2
        assert tape2.entries[0].kind == "anchor"
        assert tape2.entries[1].payload["content"] == "hello"

    def test_anchor_based_loading(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")

        # Add some entries, then an anchor, then more entries
        e1 = tape.append("message", {"role": "user", "content": "old msg"})
        store.save_entry("chat1", e1)
        e2 = tape.append("anchor", {"name": "compact/1", "state": {"summary": "previous stuff"}})
        store.save_entry("chat1", e2)
        e3 = tape.append("message", {"role": "user", "content": "new msg"})
        store.save_entry("chat1", e3)

        # Reload — should only load from anchor onwards
        store.clear()
        tape2 = store.get_or_create("chat1")
        assert len(tape2.entries) == 2  # anchor + new msg
        assert tape2.entries[0].kind == "anchor"
        assert tape2.entries[0].payload["state"]["summary"] == "previous stuff"
        assert tape2.entries[1].payload["content"] == "new msg"

    def test_lru_eviction(self, tmp_path):
        store = self._make_store(tmp_path)  # max_chats=3
        for i in range(5):
            tape = store.get_or_create(f"chat{i}")
            e = tape.append("message", {"role": "user", "content": f"msg{i}"})
            store.save_entry(f"chat{i}", e)

        # Only 3 should be in cache
        assert len(store._cache) == 3
        # chat0 and chat1 should have been evicted
        assert "chat0" not in store._cache
        assert "chat1" not in store._cache
        # But they're still in DB
        store.clear()
        tape0 = store.get_or_create("chat0")
        assert len(tape0.entries) == 1

    def test_multiple_anchors_loads_from_latest(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")

        e1 = tape.append("message", {"role": "user", "content": "ancient"})
        store.save_entry("chat1", e1)
        e2 = tape.append("anchor", {"name": "a1", "state": {"summary": "first"}})
        store.save_entry("chat1", e2)
        e3 = tape.append("message", {"role": "user", "content": "middle"})
        store.save_entry("chat1", e3)
        e4 = tape.append("anchor", {"name": "a2", "state": {"summary": "second"}})
        store.save_entry("chat1", e4)
        e5 = tape.append("message", {"role": "user", "content": "recent"})
        store.save_entry("chat1", e5)

        store.clear()
        tape2 = store.get_or_create("chat1")
        # Should load from a2 onwards (a2 + recent)
        assert len(tape2.entries) == 2
        assert tape2.entries[0].payload["name"] == "a2"
        assert tape2.entries[1].payload["content"] == "recent"

    def test_save_entries_batch(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append("anchor", {"name": "start", "state": {}})
        e2 = tape.append("message", {"role": "user", "content": "hello"})
        e3 = tape.append("message", {"role": "assistant", "content": "world"})
        store.save_entries("chat1", [e1, e2, e3])

        store.clear()
        tape2 = store.get_or_create("chat1")
        assert len(tape2.entries) == 3
        assert tape2.entries[1].payload["content"] == "hello"
        assert tape2.entries[2].payload["content"] == "world"

    def test_search_relevant_basic(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append("message", {"role": "user", "content": "画一只黑色小猪"})
        e2 = tape.append("message", {"role": "assistant", "content": "好的，已生成黑色小猪图片"})
        e3 = tape.append("anchor", {"name": "compact/1", "state": {"summary": "画了小猪"}})
        e4 = tape.append("message", {"role": "user", "content": "今天天气怎么样"})
        store.save_entries("chat1", [e1, e2, e3, e4])

        # Search for "小猪" — should find pre-anchor entries
        results = store.search_relevant("chat1", "小猪", limit=5)
        assert len(results) >= 1
        contents = [r.payload.get("content", "") for r in results]
        assert any("小猪" in c for c in contents)

    def test_search_relevant_excludes_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append("message", {"role": "user", "content": "画一只小猪"})
        e2 = tape.append("message", {"role": "user", "content": "小猪很可爱"})
        store.save_entries("chat1", [e1, e2])

        results = store.search_relevant("chat1", "小猪", limit=5, exclude_ids={e1.entry_id})
        ids = {r.entry_id for r in results}
        assert e1.entry_id not in ids

    def test_search_relevant_empty_query(self, tmp_path):
        store = self._make_store(tmp_path)
        results = store.search_relevant("chat1", "", limit=5)
        assert results == []

    def test_search_relevant_no_match(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append("message", {"role": "user", "content": "hello world"})
        store.save_entries("chat1", [e1])

        results = store.search_relevant("chat1", "zzzznotexist", limit=5)
        assert results == []

    def test_search_relevant_bm25_idf_ranking(self, tmp_path):
        """BM25 should rank entries with rare keywords higher than common ones."""
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        # "你好" appears in many messages (common), "小猪" only in one (rare)
        entries = [
            tape.append("message", {"role": "user", "content": "你好，今天天气不错"}),
            tape.append("message", {"role": "user", "content": "你好，我想聊天"}),
            tape.append("message", {"role": "user", "content": "你好，帮我查东西"}),
            tape.append("message", {"role": "user", "content": "画一只小猪，你好"}),
        ]
        store.save_entries("chat1", entries)

        # Search for "小猪 你好" — the entry with rare "小猪" should rank first
        results = store.search_relevant("chat1", "小猪你好", limit=4)
        assert len(results) >= 1
        assert "小猪" in results[0].payload["content"]


# ── _build_history_messages ────────────────────────────────────────

class TestBuildHistoryMessages:
    def test_no_tape(self):
        from babybot.agent_kernel.executor import _build_history_messages
        msgs = _build_history_messages(None, 2000)
        assert msgs == []

    def test_empty_tape(self):
        from babybot.agent_kernel.executor import _build_history_messages
        tape = Tape("chat1")
        msgs = _build_history_messages(tape, 2000)
        assert msgs == []

    def test_with_anchor_summary(self):
        from babybot.agent_kernel.executor import _build_history_messages
        tape = Tape("chat1")
        tape.append("anchor", {"name": "compact/1", "state": {"summary": "用户画了小猪"}})
        tape.append("message", {"role": "user", "content": "把背景改蓝"})

        msgs = _build_history_messages(tape, 2000)
        # First message should be the summary
        assert msgs[0].role == "system"
        assert "用户画了小猪" in msgs[0].content
        # Last user message should be excluded (it's the current turn)
        assert len(msgs) == 1

    def test_structured_anchor_state(self):
        from babybot.agent_kernel.executor import _build_history_messages
        tape = Tape("chat1")
        tape.append("anchor", {"name": "compact/1", "state": {
            "summary": "用户要求画小猪",
            "entities": ["小猪", "黑色"],
            "user_intent": "生成图片",
            "pending": "等待用户确认颜色",
        }})
        tape.append("message", {"role": "user", "content": "继续"})

        msgs = _build_history_messages(tape, 2000)
        content = msgs[0].content
        assert "用户要求画小猪" in content
        assert "小猪" in content
        assert "生成图片" in content
        assert "等待用户确认颜色" in content

    def test_with_history_messages(self):
        from babybot.agent_kernel.executor import _build_history_messages
        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "user", "content": "hi"})
        tape.append("message", {"role": "assistant", "content": "hello"})
        tape.append("message", {"role": "user", "content": "current"})

        msgs = _build_history_messages(tape, 2000)
        # No summary (empty state), but should have hi + hello (current user excluded)
        roles = [m.role for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_token_budget_limits(self):
        from babybot.agent_kernel.executor import _build_history_messages
        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        # Add many messages
        for i in range(20):
            tape.append("message", {"role": "user", "content": f"message {i} " * 50})
            tape.append("message", {"role": "assistant", "content": f"reply {i} " * 50})
        tape.append("message", {"role": "user", "content": "current"})

        msgs = _build_history_messages(tape, 500)
        # Should be limited by budget, not all 40 messages
        assert len(msgs) < 40

    def test_token_budget_prefers_recent_entries(self):
        from babybot.agent_kernel.executor import _build_history_messages

        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "user", "content": "old user " * 60})
        tape.append("message", {"role": "assistant", "content": "old assistant " * 60})
        tape.append("message", {"role": "user", "content": "recent user " * 60})
        recent_asst = tape.append("message", {"role": "assistant", "content": "recent assistant " * 60})
        tape.append("message", {"role": "user", "content": "current"})

        msgs = _build_history_messages(tape, recent_asst.token_estimate + 2)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert "recent assistant" in msgs[0].content

    def test_token_budget_skips_oversized_recent_entry(self):
        from babybot.agent_kernel.executor import _build_history_messages

        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        tape.append("message", {"role": "assistant", "content": "small keep me"})
        tape.append("message", {"role": "assistant", "content": "x" * 5000})
        tape.append("message", {"role": "user", "content": "current"})

        msgs = _build_history_messages(tape, 50)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert "small keep me" in msgs[0].content

    def test_hybrid_scoring_prefers_relevant_recent(self):
        """P1: When budget is tight, hybrid scoring picks relevant entries over purely old ones."""
        from babybot.agent_kernel.executor import _build_history_messages

        tape = Tape("chat1")
        tape.append("anchor", {"name": "start", "state": {}})
        # Old irrelevant message
        tape.append("message", {"role": "user", "content": "无关内容填充" * 30})
        tape.append("message", {"role": "assistant", "content": "好的无关回复" * 30})
        # Relevant message about 小猪
        tape.append("message", {"role": "user", "content": "画一只小猪"})
        tape.append("message", {"role": "assistant", "content": "好的已生成小猪图片"})
        # More irrelevant filler
        tape.append("message", {"role": "user", "content": "其他话题讨论" * 30})
        tape.append("message", {"role": "assistant", "content": "其他回复内容" * 30})
        # Current turn
        tape.append("message", {"role": "user", "content": "把小猪改成白色"})

        # Tight budget: can only fit ~2 messages
        msgs = _build_history_messages(tape, 120, query="把小猪改成白色")
        contents = " ".join(m.content for m in msgs)
        # Relevant "小猪" entries should be picked despite not being the most recent
        assert "小猪" in contents

    def test_cross_anchor_recall(self, tmp_path):
        from babybot.agent_kernel.executor import _build_history_messages

        store = TapeStore(db_path=tmp_path / "test.db")
        tape = store.get_or_create("chat1")

        # Pre-anchor history about "小猪"
        e1 = tape.append("message", {"role": "user", "content": "画一只黑色小猪"})
        e2 = tape.append("message", {"role": "assistant", "content": "好的，已生成黑色小猪图片"})
        e3 = tape.append("anchor", {"name": "compact/1", "state": {"summary": "画了小猪"}})
        # Post-anchor: unrelated topic
        e4 = tape.append("message", {"role": "user", "content": "今天天气怎么样"})
        e5 = tape.append("message", {"role": "assistant", "content": "今天晴天"})
        # Current turn: user asks about 小猪 again
        e6 = tape.append("message", {"role": "user", "content": "之前那个小猪改成白色"})
        store.save_entries("chat1", [e1, e2, e3, e4, e5, e6])

        msgs = _build_history_messages(
            tape, 2000,
            query="之前那个小猪改成白色",
            tape_store=store,
        )

        # Should have: [对话背景], [相关历史] with 小猪 content, recent msgs
        all_content = " ".join(m.content for m in msgs)
        assert "画了小猪" in all_content  # anchor summary
        assert "黑色小猪" in all_content  # BM25 recalled from pre-anchor
