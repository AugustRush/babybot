"""Feishu channel tools - simplified with auto docstring extraction."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .tools import ChannelCapabilities, ChannelTools, ChannelToolContext

logger = logging.getLogger(__name__)


class FeishuChannelTools(ChannelTools):
    """Feishu channel tools for rich media interactions.

    Tools are auto-registered with JSON schema extracted from docstrings.
    """

    channel_name = "feishu"
    channel_display_name = "飞书"

    def __init__(self, channel: Any):
        self._channel = channel

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_text=True,
            supports_image=True,
            supports_audio=True,
            supports_video=True,
            supports_file=True,
            supports_card=True,
            supports_reaction=True,
            max_text_length=30000,
            max_image_size=20 * 1024 * 1024,
            max_file_size=30 * 1024 * 1024,
            supported_image_formats=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
            supported_audio_formats=["opus", "mp3", "wav", "aac"],
            supported_video_formats=["mp4", "mov", "avi"],
        )

    def get_tools(self) -> list:
        """Return Feishu tool functions."""
        return [
            self.send_text,
            self.send_image,
            self.send_file,
            self.send_audio,
            self.send_card,
            self.add_reaction,
        ]

    def get_prompt(self) -> str:
        caps = self.capabilities
        return f"""## 飞书渠道交互

你正在通过飞书与用户交流，可以使用以下工具发送富媒体内容：

- **send_text(text)**: 发送文字消息
- **send_image(file_path)**: 发送图片 (格式: {", ".join(caps.supported_image_formats)})
- **send_file(file_path)**: 发送文件 (最大 {caps.max_file_size // 1024 // 1024}MB)
- **send_audio(file_path)**: 发送音频
- **send_card(title, content)**: 发送交互卡片，适合结构化信息展示
- **add_reaction(emoji)**: 添加表情反应 (THUMBSUP, HEART, LAUGH, OK)

使用建议：
- 简单回复 → send_text
- 展示图片/图表 → send_image
- 发送文档/报表 → send_file
- 结构化信息 → send_card
- 快速确认 → add_reaction("THUMBSUP")"""

    def _get_receive_info(self) -> tuple[str, str]:
        """Get receive_id_type and receive_id from context."""
        ctx = self._get_context()
        if not ctx:
            raise RuntimeError("No channel context available")

        config = self._channel.config
        receive_id = ctx.sender_id if config.reply_mode == "p2p" else ctx.chat_id

        if not receive_id:
            receive_id = ctx.chat_id

        if receive_id.startswith("oc_"):
            return "chat_id", receive_id
        elif receive_id.startswith("ou_") or receive_id.startswith("on_"):
            return "open_id", receive_id
        else:
            return "chat_id", receive_id

    def _send_message(self, msg_type: str, content: str) -> str:
        """Helper to send a message via Feishu API."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        receive_id_type, receive_id = self._get_receive_info()

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )

        response = self._channel._client.im.v1.message.create(request)
        if response.success():
            return "发送成功"
        return f"发送失败: {response.msg}"

    def send_text(self, text: str) -> str:
        """发送文字消息到飞书。

        Args:
            text (str): 要发送的文字内容。
        """
        content = json.dumps({"text": text}, ensure_ascii=False)
        result = self._send_message("text", content)
        return result

    def send_image(self, file_path: str) -> str:
        """发送图片到飞书。

        Args:
            file_path (str): 图片文件的绝对路径。
        """
        if not os.path.isfile(file_path):
            return f"文件不存在: {file_path}"

        key = self._channel._upload_image_sync(file_path)
        if not key:
            return "图片上传失败"

        content = json.dumps({"image_key": key}, ensure_ascii=False)
        result = self._send_message("image", content)
        return f"{result}: {os.path.basename(file_path)}"

    def send_file(self, file_path: str) -> str:
        """发送文件到飞书。

        Args:
            file_path (str): 文件的绝对路径。
        """
        if not os.path.isfile(file_path):
            return f"文件不存在: {file_path}"

        key = self._channel._upload_file_sync(file_path)
        if not key:
            return "文件上传失败"

        content = json.dumps({"file_key": key}, ensure_ascii=False)
        result = self._send_message("file", content)
        return f"{result}: {os.path.basename(file_path)}"

    def send_audio(self, file_path: str) -> str:
        """发送音频文件到飞书。

        Args:
            file_path (str): 音频文件的绝对路径。
        """
        if not os.path.isfile(file_path):
            return f"文件不存在: {file_path}"

        key = self._channel._upload_file_sync(file_path)
        if not key:
            return "音频上传失败"

        content = json.dumps({"file_key": key}, ensure_ascii=False)
        result = self._send_message("media", content)
        return f"{result}: {os.path.basename(file_path)}"

    def send_card(self, content: str, title: str = "") -> str:
        """发送交互卡片到飞书，适合展示结构化信息。

        Args:
            content (str): 卡片内容，支持 Markdown 格式。
            title (str): 卡片标题（可选）。
        """
        elements = []
        if title:
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**{title}**"},
                }
            )
        elements.append({"tag": "markdown", "content": content})

        card = {"config": {"wide_screen_mode": True}, "elements": elements}
        card_content = json.dumps(card, ensure_ascii=False)
        result = self._send_message("interactive", card_content)
        return result

    def add_reaction(self, emoji: str = "THUMBSUP") -> str:
        """对消息添加表情反应。

        Args:
            emoji (str): 表情类型，可选值: THUMBSUP, HEART, LAUGH, OK, CONFUSED, SAD。
        """
        ctx = self._get_context()
        if not ctx or not ctx.message_id:
            return "无法获取消息ID"

        from lark_oapi.api.im.v1 import (
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
        )

        request = (
            CreateMessageReactionRequest.builder()
            .message_id(ctx.message_id)
            .request_body(
                CreateMessageReactionRequestBody.builder()
                .reaction_type(Emoji.builder().emoji_type(emoji).build())
                .build()
            )
            .build()
        )

        response = self._channel._client.im.v1.message_reaction.create(request)
        if response.success():
            return f"已添加表情: {emoji}"
        return f"添加失败: {response.msg}"
