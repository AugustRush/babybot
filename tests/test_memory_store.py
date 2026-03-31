from __future__ import annotations

import logging
from pathlib import Path
import time

from babybot.memory_store import HybridMemoryStore


def test_hybrid_memory_store_bootstraps_hard_memory_files(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )

    store.ensure_bootstrap()

    assert (tmp_path / "assistant_profile.md").exists()
    assert not (tmp_path / "memory" / "identity.json").exists()
    assert not (tmp_path / "memory" / "policies.json").exists()


def test_hybrid_memory_store_loads_assistant_profile_markdown(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )

    store.ensure_bootstrap()
    profile_text = store.load_assistant_profile()

    assert "# Assistant Profile" in profile_text
    assert "技术助手" in profile_text


def test_hybrid_memory_store_enables_wal_and_busy_timeout(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )

    store.ensure_bootstrap()
    db = store._ensure_db()

    assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert int(db.execute("PRAGMA busy_timeout").fetchone()[0]) >= 3000


def test_hybrid_memory_store_extracts_soft_preferences_from_user_message(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_user_message("feishu:chat-1", "以后回答尽量简洁一点，默认用中文")

    records = store.list_memories(chat_id="feishu:chat-1")
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}
    assert ("relationship_policy", "response_style", "concise") in keys
    assert ("relationship_policy", "default_language", "zh-CN") in keys


def test_hybrid_memory_store_extracts_user_profile_and_assistant_role(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_user_message(
        "feishu:chat-1",
        "我是独立开发者，你现在是我的代码架构助手",
    )

    records = store.list_memories(chat_id="feishu:chat-1")
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}
    assert ("user_profile", "self_description", "独立开发者") in keys
    assert ("relationship_policy", "assistant_role", "代码架构助手") in keys


def test_hybrid_memory_store_updates_task_state_from_anchor_and_failures(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_anchor_state(
        "feishu:chat-1",
        {
            "pending": "等待确认颜色",
            "next_steps": ["把小猪改成白色"],
            "artifacts": ["pig.png"],
        },
        source_ids=[1, 2, 3],
    )
    store.observe_runtime_event(
        "feishu:chat-1",
        {
            "event": "failed",
            "task_id": "task-1",
            "payload": {
                "description": "发送语音",
                "error": "tts timeout",
            },
        },
    )

    records = store.list_memories(chat_id="feishu:chat-1")
    summaries = "\n".join(record.summary for record in records)
    assert "等待确认颜色" in summaries
    assert "把小猪改成白色" in summaries
    assert "pig.png" in summaries
    assert "tts timeout" in summaries


def test_hybrid_memory_store_ignores_assistant_reply_for_long_term_preferences_but_tracks_success_event(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_assistant_message(
        "feishu:chat-1",
        "好的，后续我会默认用中文，回答保持简洁，并作为你的代码架构助手继续协助你。",
    )
    store.observe_runtime_event(
        "feishu:chat-1",
        {
            "event": "succeeded",
            "task_id": "task-1",
            "payload": {
                "description": "生成语音",
                "output": "已生成 speech.wav",
            },
        },
    )

    records = store.list_memories(chat_id="feishu:chat-1")
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}
    summaries = "\n".join(record.summary for record in records)

    assert ("relationship_policy", "default_language", "zh-CN") not in keys
    assert ("relationship_policy", "response_style", "concise") not in keys
    assert ("relationship_policy", "assistant_role", "代码架构助手") not in keys
    assert "生成语音" in summaries
    assert "speech.wav" in summaries


def test_hybrid_memory_store_applies_user_corrections_and_supersedes_old_preferences(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_user_message("feishu:chat-1", "以后默认用中文，回答尽量简洁")
    store.observe_user_message("feishu:chat-1", "不要中文了，改成英文回复，并且详细一点")

    records = store.list_memories(chat_id="feishu:chat-1")
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}

    assert ("relationship_policy", "default_language", "en-US") in keys
    assert ("relationship_policy", "response_style", "detailed") in keys
    assert ("relationship_policy", "default_language", "zh-CN") not in keys
    assert ("relationship_policy", "response_style", "concise") not in keys


def test_hybrid_memory_store_maintenance_decays_and_expires_stale_candidates(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_user_message("feishu:chat-1", "以后默认用中文")

    first_now = time.time() + 8 * 24 * 3600
    store.run_maintenance(now=first_now)
    records = store.list_memories(chat_id="feishu:chat-1")
    language = next(
        record
        for record in records
        if record.memory_type == "relationship_policy" and record.key == "default_language"
    )
    assert language.status == "decaying"

    second_now = first_now + 40 * 24 * 3600
    store.run_maintenance(now=second_now)
    records = store.list_memories(chat_id="feishu:chat-1")
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}
    assert ("relationship_policy", "default_language", "zh-CN") not in keys


def test_hybrid_memory_store_throttles_maintenance_between_list_calls(tmp_path, monkeypatch) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    current_time = {"value": 1000.0}
    runs: list[float] = []

    monkeypatch.setattr("babybot.memory_store.time.time", lambda: current_time["value"])

    original_run_maintenance = store.run_maintenance

    def _wrapped_run_maintenance(now=None):
        runs.append(float(now if now is not None else current_time["value"]))
        return original_run_maintenance(now=now)

    monkeypatch.setattr(store, "run_maintenance", _wrapped_run_maintenance)

    store.list_memories(chat_id="feishu:chat-1")
    current_time["value"] += 1
    store.list_memories(chat_id="feishu:chat-1")
    current_time["value"] += 301
    store.list_memories(chat_id="feishu:chat-1")

    assert len(runs) == 2


def test_hybrid_memory_store_run_maintenance_batches_record_saves(tmp_path, monkeypatch) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    store.observe_user_message("feishu:chat-1", "以后默认用中文")
    store.observe_user_message("feishu:chat-2", "Please reply in English from now on")

    commit_flags: list[bool] = []
    original_save_record = store._save_record

    def _wrapped_save_record(record, *args, **kwargs):
        commit_flags.append(bool(kwargs.get("commit", True)))
        return original_save_record(record, *args, **kwargs)

    monkeypatch.setattr(store, "_save_record", _wrapped_save_record)

    store.run_maintenance(now=time.time() + 8 * 24 * 3600)

    assert commit_flags
    assert all(flag is False for flag in commit_flags)


def test_hybrid_memory_store_caches_assistant_profile_between_reads(tmp_path, monkeypatch) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    read_counts = {"assistant_profile.md": 0}
    original_read_text = Path.read_text

    def _wrapped_read_text(path_obj, *args, **kwargs):
        if path_obj.name in read_counts:
            read_counts[path_obj.name] += 1
        return original_read_text(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _wrapped_read_text)

    store.load_assistant_profile()
    store.load_assistant_profile()
    store.load_assistant_profile()

    assert read_counts == {"assistant_profile.md": 1}


def test_hybrid_memory_store_close_logs_db_close_failures(tmp_path, monkeypatch, caplog) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    store.ensure_bootstrap()

    class _BrokenConnection:
        def close(self) -> None:
            raise RuntimeError("boom")

    store._db = _BrokenConnection()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        store.close()

    assert "Failed to close memory store DB" in caplog.text
