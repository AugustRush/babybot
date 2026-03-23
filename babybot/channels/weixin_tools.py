"""Weixin channel tools."""

from __future__ import annotations

import os
from typing import Any

from .tools import ChannelCapabilities, ChannelTools


class WeixinChannelTools(ChannelTools):
    channel_name = "weixin"
    channel_display_name = "微信"

    def __init__(self, channel: Any):
        self._channel = channel

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_text=True,
            supports_image=True,
            supports_file=True,
            max_text_length=4000,
            max_image_size=20 * 1024 * 1024,
            max_file_size=30 * 1024 * 1024,
            supported_image_formats=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
        )

    def get_tools(self) -> list:
        return [self.send_text, self.send_image, self.send_file]

    def get_prompt(self) -> str:
        return """## 微信渠道交互

你正在通过微信与用户交流，可以使用以下工具发送内容：

- `send_text(text)`: 发送文字消息
- `send_image(file_path)`: 发送图片文件
- `send_file(file_path)`: 发送普通文件

使用建议：
- 普通回复 → send_text
- 展示图片/图表 → send_image
- 发送文档/附件 → send_file"""

    async def send_text(self, text: str) -> str:
        """发送文本到微信。

        Args:
            text (str): 要发送的文本内容。
        """
        ctx = self._get_context()
        if not ctx:
            return "无法获取当前微信会话"
        context_token = self._channel._context_tokens.get(ctx.chat_id, "")
        if not context_token:
            return "无法获取当前微信 context_token"
        await self._channel._send_text(ctx.chat_id, text, context_token)
        return "已发送文本"

    async def send_image(self, file_path: str) -> str:
        """发送图片到微信。

        Args:
            file_path (str): 图片文件绝对路径。
        """
        return await self._send_media(file_path, kind="图片")

    async def send_file(self, file_path: str) -> str:
        """发送文件到微信。

        Args:
            file_path (str): 文件绝对路径。
        """
        return await self._send_media(file_path, kind="文件")

    async def _send_media(self, file_path: str, *, kind: str) -> str:
        ctx = self._get_context()
        if not ctx:
            return f"无法获取当前微信会话，不能发送{kind}"
        if not os.path.isfile(file_path):
            return f"文件不存在: {file_path}"
        context_token = self._channel._context_tokens.get(ctx.chat_id, "")
        if not context_token:
            return "无法获取当前微信 context_token"
        await self._channel._send_media_file(ctx.chat_id, file_path, context_token)
        return f"已发送{kind}: {os.path.basename(file_path)}"
