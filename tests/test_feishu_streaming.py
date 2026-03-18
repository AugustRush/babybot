from __future__ import annotations

import asyncio

from babybot.channels.feishu import FeishuChannel
from babybot.config import FeishuConfig


def test_create_stream_message_uses_interactive_card_payload() -> None:
    channel = FeishuChannel(FeishuConfig(enabled=False, stream_reply=True), manager=None)

    created: list[tuple[str, str]] = []

    def _fake_create_with_id(
        receive_id_type: str,
        receive_id: str,
        msg_type: str,
        content: str,
    ) -> str | None:
        del receive_id_type, receive_id
        created.append((msg_type, content))
        return "om_mock_message"

    channel._send_message_with_id_sync = _fake_create_with_id  # type: ignore[method-assign]

    message_id = asyncio.run(
        channel.create_stream_message(
            chat_id="oc_mock_chat",
            text="streaming output demo",
            sender_id="ou_user_1",
        )
    )

    assert message_id == "om_mock_message"
    assert len(created) == 1
    assert created[0][0] == "interactive"
    assert "streaming output demo" in created[0][1]


def test_patch_stream_message_uses_interactive_card_payload() -> None:
    channel = FeishuChannel(FeishuConfig(enabled=False, stream_reply=True), manager=None)

    patched: list[tuple[str, str]] = []

    def _fake_patch(message_id: str, content: str) -> bool:
        patched.append((message_id, content))
        return True

    channel._patch_message_sync = _fake_patch  # type: ignore[method-assign]

    ok = asyncio.run(channel.patch_stream_message("om_mock_message", "hello world"))

    assert ok is True
    assert patched[0][0] == "om_mock_message"
    assert "hello world" in patched[0][1]
