from __future__ import annotations

import time

from babybot.memory_store import HybridMemoryStore


def test_hybrid_memory_store_bootstraps_hard_memory_files(tmp_path) -> None:
    store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )

    store.ensure_bootstrap()

    assert (tmp_path / "memory" / "identity.json").exists()
    assert (tmp_path / "memory" / "policies.json").exists()


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


def test_hybrid_memory_store_updates_from_assistant_reply_and_success_event(tmp_path) -> None:
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

    assert ("relationship_policy", "default_language", "zh-CN") in keys
    assert ("relationship_policy", "response_style", "concise") in keys
    assert ("relationship_policy", "assistant_role", "代码架构助手") in keys
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
