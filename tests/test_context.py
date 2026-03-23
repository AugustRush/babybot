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

    def test_tool_call_factory(self):
        e = Entry.tool_call(3, "add", {"a": 1, "b": 2}, task_id="t1", step=1)
        assert e.kind == "tool_call"
        assert e.payload["name"] == "add"
        assert e.payload["arguments"] == {"a": 1, "b": 2}
        assert e.meta["task_id"] == "t1"

    def test_tool_result_factory(self):
        e = Entry.tool_result(
            4,
            "add",
            ok=True,
            content_preview="3",
            artifacts=["/tmp/out.txt"],
            task_id="t1",
            step=1,
        )
        assert e.kind == "tool_result"
        assert e.payload["name"] == "add"
        assert e.payload["ok"] is True
        assert e.payload["content_preview"] == "3"
        assert e.payload["artifacts"] == ["/tmp/out.txt"]

    def test_event_factory(self):
        e = Entry.event(5, "started", task_id="task-1", flow_id="flow-1")
        assert e.kind == "event"
        assert e.payload["event"] == "started"
        assert e.meta["task_id"] == "task-1"
        assert e.meta["flow_id"] == "flow-1"

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

    def test_search_relevant_matches_anchor_and_tool_result_content(self, tmp_path):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        e1 = tape.append(
            "anchor",
            {
                "name": "compact/1",
                "state": {
                    "summary": "用户正在编辑小猪图片",
                    "pending": "等待确认是否改成白色",
                },
            },
        )
        e2 = tape.append(
            "tool_result",
            {
                "name": "generate_image",
                "ok": True,
                "content_preview": "已生成黑色小猪图片",
                "artifacts": [],
            },
        )
        store.save_entries("chat1", [e1, e2])

        results = store.search_relevant("chat1", "小猪 白色", limit=5)
        payload_text = json.dumps([r.payload for r in results], ensure_ascii=False)
        assert "小猪" in payload_text
        assert "白色" in payload_text or "黑色小猪图片" in payload_text

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

    def test_search_relevant_uses_bounded_sql_queries(self, tmp_path, monkeypatch):
        store = self._make_store(tmp_path)
        tape = store.get_or_create("chat1")
        entries = [
            tape.append("message", {"role": "user", "content": "画一只黑色小猪"}),
            tape.append("message", {"role": "assistant", "content": "好的，已生成黑色小猪图片"}),
            tape.append("tool_result", {"name": "generate_image", "ok": True, "content_preview": "黑色小猪", "artifacts": []}),
        ]
        store.save_entries("chat1", entries)

        db = store._ensure_db()
        execute_count = {"value": 0}

        class _DBProxy:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def execute(self, *args, **kwargs):
                execute_count["value"] += 1
                return self._wrapped.execute(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        monkeypatch.setattr(store, "_ensure_db", lambda: _DBProxy(db))

        results = store.search_relevant("chat1", "黑色 小猪", limit=5)

        assert results
        assert execute_count["value"] <= 3


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
            "next_steps": ["把小猪改成白色"],
            "artifacts": ["pig.png"],
            "open_questions": ["是否保留背景"],
            "decisions": ["继续使用上一张图"],
        }})
        tape.append("message", {"role": "user", "content": "继续"})

        msgs = _build_history_messages(tape, 2000)
        content = msgs[0].content
        assert "用户要求画小猪" in content
        assert "小猪" in content
        assert "生成图片" in content
        assert "等待用户确认颜色" in content
        assert "把小猪改成白色" in content
        assert "pig.png" in content
        assert "是否保留背景" in content
        assert "继续使用上一张图" in content

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

    def test_cross_anchor_recall_formats_tool_results_and_failed_events(self, tmp_path):
        from babybot.agent_kernel.executor import _build_history_messages

        store = TapeStore(db_path=tmp_path / "test.db")
        tape = store.get_or_create("chat1")

        e1 = tape.append(
            "tool_result",
            {
                "name": "generate_image",
                "ok": True,
                "content_preview": "已生成黑色小猪图片",
                "artifacts": ["pig.png"],
            },
        )
        e2 = tape.append(
            "event",
            {
                "event": "failed",
                "payload": {
                    "description": "发送语音",
                    "error": "tts timeout",
                },
            },
        )
        e3 = tape.append("anchor", {"name": "compact/1", "state": {"summary": "前面做过图片和语音尝试"}})
        e4 = tape.append("message", {"role": "user", "content": "把之前那个小猪发我，并说明语音为什么失败"})
        store.save_entries("chat1", [e1, e2, e3, e4])

        msgs = _build_history_messages(
            tape,
            2000,
            query="之前那个小猪和语音失败",
            tape_store=store,
        )

        system_texts = [m.content for m in msgs if m.role == "system"]
        merged = "\n".join(system_texts)
        assert "generate_image" in merged
        assert "已生成黑色小猪图片" in merged
        assert "发送语音" in merged
        assert "tts timeout" in merged

    def test_history_view_includes_recent_tool_results_and_failed_events(self):
        from babybot.agent_kernel.executor import _build_history_messages

        tape = Tape("chat1")
        tape.append("anchor", {"name": "compact/1", "state": {"summary": "前面做过图片尝试"}})
        tape.append(
            "tool_result",
            {
                "name": "generate_image",
                "ok": True,
                "content_preview": "已生成白色小猪图片",
                "artifacts": ["pig-white.png"],
            },
        )
        tape.append(
            "event",
            {
                "event": "failed",
                "payload": {
                    "description": "发送语音",
                    "error": "tts timeout",
                },
            },
        )
        tape.append("message", {"role": "user", "content": "把结果再发我"})

        msgs = _build_history_messages(tape, 2000, query="发我之前的小猪结果")
        system_texts = "\n".join(m.content for m in msgs if m.role == "system")

        assert "近期执行状态" in system_texts
        assert "generate_image" in system_texts
        assert "已生成白色小猪图片" in system_texts
        assert "发送语音" in system_texts

    def test_history_view_includes_hot_warm_cold_memory_layers(self, tmp_path):
        from babybot.agent_kernel.executor import _build_history_messages
        from babybot.memory_store import HybridMemoryStore

        tape = Tape("chat1")
        tape.append("anchor", {"name": "compact/1", "state": {"summary": "用户正在改图"}})
        tape.append("message", {"role": "user", "content": "以后默认中文，而且回答简洁一点"})
        tape.append("message", {"role": "assistant", "content": "好的"})
        tape.append("message", {"role": "user", "content": "继续处理那张小猪图"})

        store = TapeStore(db_path=tmp_path / "context.db")
        memory_store = HybridMemoryStore(
            db_path=tmp_path / "context.db",
            memory_dir=tmp_path / "memory",
        )
        memory_store.ensure_bootstrap()
        memory_store.observe_user_message("chat1", "以后默认中文，而且回答简洁一点")
        memory_store.observe_anchor_state(
            "chat1",
            {
                "pending": "等待确认颜色",
                "next_steps": ["把小猪改成白色"],
                "artifacts": ["pig.png"],
            },
            source_ids=[1, 2],
        )

        msgs = _build_history_messages(
            tape,
            2000,
            query="继续处理小猪图",
            tape_store=store,
            memory_store=memory_store,
        )
        system_texts = "\n".join(m.content for m in msgs if m.role == "system")

        assert "[Hot Context]" in system_texts
        assert "[Warm Context]" in system_texts
        assert "等待确认颜色" in system_texts
        assert "pig.png" in system_texts
        assert "默认中文" in system_texts
        assert "简洁" in system_texts

    def test_history_view_uses_query_to_promote_relevant_memory(self, tmp_path):
        from babybot.agent_kernel.executor import _build_history_messages
        from babybot.memory_store import HybridMemoryStore

        tape = Tape("chat1")
        tape.append("anchor", {"name": "compact/1", "state": {"summary": "用户在并行处理图片和语音"}})

        memory_store = HybridMemoryStore(
            db_path=tmp_path / "context.db",
            memory_dir=tmp_path / "memory",
        )
        memory_store.ensure_bootstrap()
        memory_store.observe_user_message("chat1", "我是独立开发者，你现在是我的代码架构助手")
        memory_store.observe_anchor_state(
            "chat1",
            {
                "pending": "继续处理语音生成失败",
                "artifacts": ["speech.wav", "pig.png"],
                "decisions": ["语音失败优先回退宿主 Python"],
            },
            source_ids=[1, 2],
        )
        memory_store.observe_runtime_event(
            "chat1",
            {
                "event": "failed",
                "payload": {
                    "description": "生成语音",
                    "error": "tts timeout",
                },
            },
        )

        msgs = _build_history_messages(
            tape,
            2000,
            query="继续修复语音 tts 问题",
            memory_store=memory_store,
        )
        system_messages = [m.content for m in msgs if m.role == "system"]
        hot_text = "\n".join(text for text in system_messages if text.startswith("[Hot Context]"))
        warm_text = "\n".join(text for text in system_messages if text.startswith("[Warm Context]"))

        assert "tts timeout" in hot_text
        assert "继续处理语音生成失败" in hot_text
        assert "独立开发者" in warm_text
        assert "代码架构助手" in warm_text

    def test_context_view_demotes_decaying_soft_memory(self, tmp_path):
        from babybot.context_views import build_context_view
        from babybot.memory_store import HybridMemoryStore

        memory_store = HybridMemoryStore(
            db_path=tmp_path / "context.db",
            memory_dir=tmp_path / "memory",
        )
        memory_store.ensure_bootstrap()

        memory_store.observe_user_message("chat1", "以后默认用中文")
        memory_store.observe_user_message("chat1", "以后默认用中文")
        memory_store.observe_user_message("chat1", "我是独立开发者")
        memory_store.run_maintenance(now=time.time() + 8 * 24 * 3600)

        view = build_context_view(
            memory_store=memory_store,
            chat_id="chat1",
            query="继续聊天",
        )

        warm_text = "\n".join(view.warm)
        cold_text = "\n".join(view.cold)
        assert "默认中文" in warm_text
        assert "独立开发者" in cold_text
