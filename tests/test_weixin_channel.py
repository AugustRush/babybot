from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from babybot.channels.base import InboundMessage
from babybot.channels.registry import discover_channels
from babybot.channels.weixin import WeixinChannel
from babybot.config import WeixinConfig
from babybot.orchestrator import TaskResponse


class _Manager:
    def __init__(self) -> None:
        self.messages: list[InboundMessage] = []

    async def handle_message(self, msg: InboundMessage) -> None:
        self.messages.append(msg)


def test_channel_registry_discovers_weixin_channel() -> None:
    channels = discover_channels()
    assert "weixin" in channels


def test_weixin_send_response_sends_text_chunks() -> None:
    channel = WeixinChannel(WeixinConfig(enabled=False, token="tok"), manager=None)
    sent: list[tuple[str, str, str]] = []

    async def _fake_send_text(chat_id: str, text: str, context_token: str = "") -> None:
        sent.append((chat_id, text, context_token))

    channel._send_text = _fake_send_text  # type: ignore[method-assign]
    channel._context_tokens["wx_user"] = "ctx-1"

    asyncio.run(
        channel.send_response(
            "wx_user",
            TaskResponse(text="你好微信"),
        )
    )

    assert sent == [("wx_user", "你好微信", "ctx-1")]


def test_weixin_process_text_message_enqueues_inbound_message() -> None:
    manager = _Manager()
    channel = WeixinChannel(WeixinConfig(enabled=False, token="tok"), manager=manager)

    async def _run() -> None:
        await channel._process_message(
            {
                "message_type": 1,
                "message_id": "m1",
                "from_user_id": "wx_user",
                "context_token": "ctx-1",
                "item_list": [
                    {
                        "type": 1,
                        "text_item": {"text": "你好"},
                    }
                ],
            }
        )

    asyncio.run(_run())

    assert len(manager.messages) == 1
    msg = manager.messages[0]
    assert msg.channel == "weixin"
    assert msg.chat_id == "wx_user"
    assert msg.sender_id == "wx_user"
    assert msg.content == "你好"
    assert msg.metadata["message_id"] == "m1"
    assert channel._context_tokens["wx_user"] == "ctx-1"



def test_weixin_login_persists_state_from_qr_flow(tmp_path: Path) -> None:
    channel = WeixinChannel(
        WeixinConfig(enabled=True, token="", state_dir=str(tmp_path)),
        manager=None,
    )
    shown: list[str] = []
    saved: list[str] = []

    async def _fake_api_get(
        endpoint: str,
        params: dict | None = None,
        *,
        auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        del params, auth, extra_headers
        if endpoint == "ilink/bot/get_bot_qrcode":
            return {
                "qrcode": "qr-id",
                "qrcode_img_content": "https://login.example/qr",
            }
        if endpoint == "ilink/bot/get_qrcode_status":
            return {
                "status": "confirmed",
                "bot_token": "tok-from-qr",
                "baseurl": "https://wx.example",
            }
        raise AssertionError(endpoint)

    channel._api_get = _fake_api_get  # type: ignore[method-assign]
    channel._print_qr_code = lambda url: shown.append(url)  # type: ignore[method-assign]
    channel._save_qr_png = lambda url: saved.append(url) or str(tmp_path / "weixin-login-qr.png")  # type: ignore[method-assign]

    assert asyncio.run(channel.login()) is True
    state = json.loads((tmp_path / "account.json").read_text(encoding="utf-8"))
    assert shown == ["https://login.example/qr"]
    assert saved == ["https://login.example/qr"]
    assert state["token"] == "tok-from-qr"
    assert state["base_url"] == "https://wx.example"



def test_weixin_save_qr_png_writes_file(tmp_path: Path) -> None:
    channel = WeixinChannel(
        WeixinConfig(enabled=True, token="", state_dir=str(tmp_path)),
        manager=None,
    )

    class _FakeImage:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"png")

    fake_module = types.SimpleNamespace(make=lambda url: _FakeImage())
    old_module = sys.modules.get("qrcode")
    sys.modules["qrcode"] = fake_module
    try:
        saved_path = channel._save_qr_png("https://login.example/qr")
    finally:
        if old_module is None:
            sys.modules.pop("qrcode", None)
        else:
            sys.modules["qrcode"] = old_module

    assert saved_path == str(tmp_path / "weixin-login-qr.png")
    assert (tmp_path / "weixin-login-qr.png").read_bytes() == b"png"



def test_weixin_start_triggers_login_when_token_missing(tmp_path: Path) -> None:
    channel = WeixinChannel(
        WeixinConfig(enabled=True, token="", state_dir=str(tmp_path)),
        manager=None,
    )
    called: dict[str, int] = {"login": 0, "poll": 0}

    async def _fake_login(force: bool = False) -> bool:
        del force
        called["login"] += 1
        return True

    async def _fake_poll_loop() -> None:
        called["poll"] += 1

    channel.login = _fake_login  # type: ignore[method-assign]
    channel._poll_loop = _fake_poll_loop  # type: ignore[method-assign]

    async def _run() -> None:
        await channel.start()
        await asyncio.sleep(0)
        await channel.stop()

    asyncio.run(_run())

    assert called == {"login": 1, "poll": 1}



def test_weixin_process_image_message_downloads_media(tmp_path: Path) -> None:
    manager = _Manager()
    channel = WeixinChannel(
        WeixinConfig(enabled=False, token="tok", state_dir=str(tmp_path)),
        manager=manager,
    )
    media_path = str(tmp_path / "image.png")

    async def _fake_download(
        typed_item: dict,
        media_type: str,
        filename: str | None = None,
    ) -> str | None:
        del typed_item, filename
        assert media_type == "image"
        return media_path

    channel._download_media_item = _fake_download  # type: ignore[method-assign]

    async def _run() -> None:
        await channel._process_message(
            {
                "message_type": 1,
                "message_id": "m-img",
                "from_user_id": "wx_user",
                "context_token": "ctx-1",
                "item_list": [
                    {
                        "type": 2,
                        "image_item": {"media": {"encrypt_query_param": "abc"}},
                    }
                ],
            }
        )

    asyncio.run(_run())

    assert len(manager.messages) == 1
    msg = manager.messages[0]
    assert msg.content == "[image]"
    assert msg.media_paths == [media_path]



def test_weixin_send_response_sends_media_before_text(tmp_path: Path) -> None:
    img = tmp_path / "demo.png"
    doc = tmp_path / "demo.txt"
    img.write_bytes(b"img")
    doc.write_text("doc", encoding="utf-8")

    channel = WeixinChannel(WeixinConfig(enabled=False, token="tok"), manager=None)
    channel._context_tokens["wx_user"] = "ctx-1"
    calls: list[tuple[str, str, str]] = []

    async def _fake_send_media_file(
        chat_id: str,
        file_path: str,
        context_token: str,
    ) -> None:
        calls.append(("media", file_path, context_token))

    async def _fake_send_text(chat_id: str, text: str, context_token: str = "") -> None:
        del chat_id
        calls.append(("text", text, context_token))

    channel._send_media_file = _fake_send_media_file  # type: ignore[method-assign]
    channel._send_text = _fake_send_text  # type: ignore[method-assign]

    asyncio.run(
        channel.send_response(
            "wx_user",
            TaskResponse(text="处理完成", media_paths=[str(img), str(doc)]),
        )
    )

    assert calls == [
        ("media", str(img), "ctx-1"),
        ("media", str(doc), "ctx-1"),
        ("text", "处理完成", "ctx-1"),
    ]



def test_weixin_channel_tools_send_image_and_file(tmp_path: Path) -> None:
    from babybot.channels.tools import ChannelToolContext

    img = tmp_path / "demo.png"
    doc = tmp_path / "demo.txt"
    img.write_bytes(b"img")
    doc.write_text("doc", encoding="utf-8")

    channel = WeixinChannel(WeixinConfig(enabled=False, token="tok"), manager=None)
    channel._context_tokens["wx_user"] = "ctx-1"
    tools = channel.get_channel_tools()
    assert tools is not None
    sent: list[str] = []

    async def _fake_send_media_file(
        chat_id: str,
        file_path: str,
        context_token: str,
    ) -> None:
        assert chat_id == "wx_user"
        assert context_token == "ctx-1"
        sent.append(file_path)

    channel._send_media_file = _fake_send_media_file  # type: ignore[method-assign]
    ChannelToolContext.set_current(ChannelToolContext(chat_id="wx_user", channel_name="weixin"))
    try:
        image_result = asyncio.run(tools.send_image(str(img)))
        file_result = asyncio.run(tools.send_file(str(doc)))
    finally:
        ChannelToolContext.set_current(None)

    assert image_result == "已发送图片: demo.png"
    assert file_result == "已发送文件: demo.txt"
    assert sent == [str(img), str(doc)]
