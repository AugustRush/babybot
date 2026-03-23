from __future__ import annotations

import asyncio

from babybot.channels.feishu import FeishuChannel, _extract_interactive_content
from babybot.config import FeishuConfig


def test_normalize_markdown_images_to_links() -> None:
    text = "结果如下：![生成图](https://example.com/a.png)"
    normalized = FeishuChannel._normalize_markdown_images(text)
    assert "![生成图]" not in normalized
    assert "[生成图](https://example.com/a.png)" in normalized



def test_extract_interactive_content_supports_flat_elements_list() -> None:
    parts = _extract_interactive_content(
        {
            "elements": [
                {"tag": "markdown", "content": "hello"},
                {"tag": "a", "href": "https://example.com", "text": "link"},
            ]
        }
    )

    assert "hello" in parts
    assert "link: https://example.com" in parts
    assert "link" in parts


def test_feishu_channel_uses_async_processed_lock() -> None:
    channel = FeishuChannel(FeishuConfig(enabled=False), manager=None)
    assert isinstance(channel._processed_lock, asyncio.Lock)



def test_feishu_stop_joins_thread_and_closes_client() -> None:
    channel = FeishuChannel(FeishuConfig(enabled=False), manager=None)
    called = {"join": 0, "close": 0}

    class _Thread:
        def join(self, timeout: float | None = None) -> None:
            del timeout
            called["join"] += 1

    class _Client:
        def stop(self) -> None:
            called["close"] += 1

    channel._running = True
    channel._ws_thread = _Thread()
    channel._ws_client = _Client()

    asyncio.run(channel.stop())

    assert called == {"join": 1, "close": 1}
