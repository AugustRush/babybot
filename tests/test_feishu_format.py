from __future__ import annotations

from babybot.channels.feishu import FeishuChannel


def test_normalize_markdown_images_to_links() -> None:
    text = "结果如下：![生成图](https://example.com/a.png)"
    normalized = FeishuChannel._normalize_markdown_images(text)
    assert "![生成图]" not in normalized
    assert "[生成图](https://example.com/a.png)" in normalized
