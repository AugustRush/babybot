"""Feishu channel integration using lark-oapi WebSocket long connection.

Supports:
- Inbound: text, post (rich text), image, audio, file, media, interactive cards,
  share_chat, share_user, share_calendar_event, system, merge_forward
- Outbound: smart format detection (text / post / interactive card), media upload
- Reactions: auto-react to received messages
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from ..config import FeishuConfig
from ..orchestrator import TaskResponse
from .base import BaseChannel, InboundMessage
from .tools import ChannelCapabilities
from .feishu_tools import FeishuChannelTools

logger = logging.getLogger(__name__)


FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
    "media": "[media]",
}


# ── Content extraction helpers ───────────────────────────────────────

from .feishu_content_extractor import (
    _extract_element_content,
    _extract_interactive_content,
    _extract_post_content,
    _extract_share_card_content,
)


# ── FeishuChannel ────────────────────────────────────────────────────


class FeishuChannel(BaseChannel):
    """Feishu bot channel with full message type support."""

    name = "feishu"
    display_name = "Feishu"

    # File extension sets for media type detection
    _IMAGE_EXTS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".ico",
        ".tiff",
        ".tif",
    }
    _AUDIO_EXTS = {".aac", ".mp3", ".opus", ".wav"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi"}
    _FILE_TYPE_MAP = {
        ".opus": "opus",
        ".mp4": "mp4",
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "doc",
        ".xls": "xls",
        ".xlsx": "xls",
        ".ppt": "ppt",
        ".pptx": "ppt",
    }

    @classmethod
    def _message_type_for_file(cls, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        file_type = cls._FILE_TYPE_MAP.get(ext, "stream")
        if file_type in {"opus", "mp4"}:
            return "media"
        return "file"

    # Regex patterns for smart format detection and card building
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    _COMPLEX_MD_RE = re.compile(
        r"```"
        r"|^\|.+\|.*\n\s*\|[-:\s|]+\|"
        r"|^#{1,6}\s+",
        re.MULTILINE,
    )
    _SIMPLE_MD_RE = re.compile(
        r"\*\*.+?\*\*"
        r"|__.+?__"
        r"|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"
        r"|~~.+?~~",
        re.DOTALL,
    )
    _MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    _MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    _LIST_RE = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
    _OLIST_RE = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
    _TEXT_MAX_LEN = 200
    _POST_MAX_LEN = 2000

    def __init__(self, config: FeishuConfig, manager: Any):
        super().__init__(config, manager)
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._processed_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Resolve media directory
        media_dir = config.media_dir
        if media_dir:
            self._media_dir = Path(media_dir).expanduser()
        else:
            self._media_dir = (
                Path(os.getenv("BABYBOT_HOME", "~/.babybot")).expanduser()
                / "media"
                / "feishu"
            )

    # ── Streaming capability ──────────────────────────���──────────────

    @property
    def supports_streaming(self) -> bool:
        """Return True when streaming replies are enabled in Feishu config."""
        return bool(
            getattr(self.config, "streaming", False)
            or getattr(self.config, "stream_reply", False)
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start Feishu WebSocket client and keep running."""
        feishu_cfg = self.config
        if not FEISHU_AVAILABLE:
            raise RuntimeError("Feishu SDK not installed. Run: uv sync")
        if not feishu_cfg.app_id or not feishu_cfg.app_secret:
            raise ValueError(
                "Feishu app_id/app_secret not configured in channels.feishu"
            )

        import lark_oapi as lark

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._media_dir.mkdir(parents=True, exist_ok=True)

        self._client = (
            lark.Client.builder()
            .app_id(feishu_cfg.app_id)
            .app_secret(feishu_cfg.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        builder = lark.EventDispatcherHandler.builder(
            feishu_cfg.encrypt_key or "",
            feishu_cfg.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync,
        )
        builder = self._register_optional_event(
            builder,
            "register_p2_im_message_reaction_created_v1",
            self._on_reaction_created,
        )
        builder = self._register_optional_event(
            builder, "register_p2_im_message_message_read_v1", self._on_message_read
        )
        builder = self._register_optional_event(
            builder,
            "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
            self._on_bot_p2p_chat_entered,
        )
        event_handler = builder.build()

        self._ws_client = lark.ws.Client(
            feishu_cfg.app_id,
            feishu_cfg.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        def run_ws() -> None:
            import time
            import lark_oapi.ws.client as _lark_ws_client

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            _lark_ws_client.loop = ws_loop
            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception:
                        if self._running:
                            time.sleep(3)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        logger.info("Feishu channel started (WebSocket long connection).")

    async def stop(self) -> None:
        """Stop Feishu channel."""
        self._running = False
        ws_client = self._ws_client
        ws_thread = self._ws_thread
        self._ws_client = None
        self._ws_thread = None

        for obj, method_name in ((ws_client, "stop"), (self._client, "close")):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    result = method()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning(
                        "Error stopping Feishu resource %s: %s", method_name, exc
                    )

        self._client = None

        if ws_thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, ws_thread.join, 5.0)

    def get_channel_tools(self) -> FeishuChannelTools | None:
        """Return Feishu channel tools for registration."""
        if not FEISHU_AVAILABLE:
            return None
        return FeishuChannelTools(self)

    @property
    def capabilities(self) -> ChannelCapabilities:
        """Return Feishu capabilities."""
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

    @staticmethod
    def _register_optional_event(builder: Any, method_name: str, handler: Any) -> Any:
        """Register an event handler only when the SDK supports it."""
        method = getattr(builder, method_name, None)
        return method(handler) if callable(method) else builder

    # ── Reactions ────────────────────────────────────────────────────

    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        from lark_oapi.api.im.v1 import (
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
        )

        try:
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message_reaction.create(request)
            if not response.success():
                logger.warning(
                    f"Feishu: failed to add reaction: code={response.code}, msg={response.msg}"
                )
        except Exception as e:
            logger.warning("Feishu: error adding reaction: %s", e)

    async def _add_reaction(
        self, message_id: str, emoji_type: str = "THUMBSUP"
    ) -> None:
        """Add a reaction emoji to a message (non-blocking)."""
        if not self._client:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._add_reaction_sync, message_id, emoji_type
        )

    # ── Group message filtering ──────────────────────────────────────

    def _is_bot_mentioned(self, message: Any) -> bool:
        """Check if the bot is @mentioned in the message."""
        raw_content = message.content or ""
        if "@_all" in raw_content:
            return True
        for mention in getattr(message, "mentions", None) or []:
            mid = getattr(mention, "id", None)
            if not mid:
                continue
            if getattr(mid, "user_id", None) == "" and (
                getattr(mid, "open_id", None) or ""
            ).startswith("ou_"):
                return True
        return False

    def _allow_group_message(self, message: Any) -> bool:
        policy: Literal["open", "mention"] = self.config.group_policy
        if policy == "open":
            return True
        return self._is_bot_mentioned(message)

    # ── Media download ───────────────────────────────────────────────

    def _download_image_sync(
        self, message_id: str, image_key: str
    ) -> tuple[bytes | None, str | None]:
        """Download an image from Feishu message."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                if hasattr(file_data, "read"):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.warning(
                    "Feishu: failed to download image: code=%s, msg=%s",
                    response.code,
                    response.msg,
                )
                return None, None
        except Exception as e:
            logger.warning("Feishu: error downloading image %s: %s", image_key, e)
            return None, None

    def _download_file_sync(
        self, message_id: str, file_key: str, resource_type: str = "file"
    ) -> tuple[bytes | None, str | None]:
        """Download a file/audio/media from a Feishu message."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        # Feishu API only accepts 'image' or 'file' as type parameter
        if resource_type == "audio":
            resource_type = "file"
        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                if hasattr(file_data, "read"):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.warning(
                    f"Feishu: failed to download {resource_type}: code={response.code}, msg={response.msg}"
                )
                return None, None
        except Exception as e:
            logger.warning(
                "Feishu: error downloading %s %s: %s",
                resource_type,
                file_key,
                e,
            )
            return None, None

    async def _download_and_save_media(
        self,
        msg_type: str,
        content_json: dict,
        message_id: str | None = None,
    ) -> tuple[str | None, str]:
        """Download media from Feishu and save to local disk.

        Returns (file_path, content_text) — file_path is None if download failed.
        """
        loop = asyncio.get_running_loop()
        data, filename = None, None

        if msg_type == "image":
            image_key = content_json.get("image_key")
            if image_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_image_sync, message_id, image_key
                )
                if not filename:
                    filename = f"{image_key[:16]}.jpg"

        elif msg_type in ("audio", "file", "media"):
            file_key = content_json.get("file_key")
            if file_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_file_sync, message_id, file_key, msg_type
                )
                if not filename:
                    filename = file_key[:16]
                if msg_type == "audio" and not filename.endswith(".opus"):
                    filename = f"{filename}.opus"

        if data and filename:
            # Sanitize filename to prevent path traversal attacks
            filename = Path(filename).name
            file_path = self._media_dir / filename
            file_path.write_bytes(data)
            return str(file_path), f"[{msg_type}: {filename}]"

        return None, f"[{msg_type}: download failed]"

    # ── Media upload ─────────────────────────────────────────────────

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        try:
            with open(file_path, "rb") as f:
                request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.image.create(request)
                if response.success():
                    return response.data.image_key
                else:
                    logger.warning(
                        f"Feishu: failed to upload image: code={response.code}, msg={response.msg}"
                    )
                    return None
        except Exception as e:
            logger.warning("Feishu: error uploading image %s: %s", file_path, e)
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                request = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.file.create(request)
                if response.success():
                    return response.data.file_key
                else:
                    logger.warning(
                        f"Feishu: failed to upload file: code={response.code}, msg={response.msg}"
                    )
                    return None
        except Exception as e:
            logger.warning("Feishu: error uploading file %s: %s", file_path, e)
            return None

    # ── Send message (sync helper) ───────────────────────────────────

    def _send_message_sync(
        self, receive_id_type: str, receive_id: str, msg_type: str, content: str
    ) -> bool:
        """Send a single message synchronously."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build(),
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.warning(
                    f"Feishu: send failed: code={response.code}, msg={response.msg}, "
                    f"log_id={response.get_log_id()}"
                )
                return False
            return True
        except Exception as e:
            logger.warning("Feishu: error sending %s message: %s", msg_type, e)
            return False

    def _send_message_with_id_sync(
        self, receive_id_type: str, receive_id: str, msg_type: str, content: str
    ) -> str | None:
        """Send a message and return Feishu ``message_id`` when successful."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build(),
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.warning(
                    f"Feishu: send failed: code={response.code}, msg={response.msg}, "
                    f"log_id={response.get_log_id()}"
                )
                return None
            return getattr(getattr(response, "data", None), "message_id", None)
        except Exception as e:
            logger.warning("Feishu: error sending %s message: %s", msg_type, e)
            return None

    def _patch_message_sync(self, message_id: str, content: str) -> bool:
        """Patch an existing Feishu message content."""
        from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

        try:
            request = (
                PatchMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    PatchMessageRequestBody.builder().content(content).build()
                )
                .build()
            )
            response = self._client.im.v1.message.patch(request)
            if not response.success():
                logger.warning(
                    f"Feishu: patch failed: code={response.code}, msg={response.msg}, "
                    f"log_id={response.get_log_id()}"
                )
                return False
            return True
        except Exception as e:
            logger.warning("Feishu: error patching message %s: %s", message_id, e)
            return False

    # ── Smart format detection ───────────────────────────────────────

    @classmethod
    def _detect_msg_format(cls, content: str) -> str:
        """Determine the optimal Feishu message format for *content*.

        Returns "text", "post", or "interactive".
        """
        stripped = content.strip()
        if cls._COMPLEX_MD_RE.search(stripped):
            return "interactive"
        if len(stripped) > cls._POST_MAX_LEN:
            return "interactive"
        if cls._SIMPLE_MD_RE.search(stripped):
            return "interactive"
        if cls._LIST_RE.search(stripped) or cls._OLIST_RE.search(stripped):
            return "interactive"
        if cls._MD_LINK_RE.search(stripped):
            return "post"
        if len(stripped) <= cls._TEXT_MAX_LEN:
            return "text"
        return "post"

    @classmethod
    def _markdown_to_post(cls, content: str) -> str:
        """Convert markdown content to Feishu post message JSON."""
        lines = content.strip().split("\n")
        paragraphs: list[list[dict]] = []
        for line in lines:
            elements: list[dict] = []
            last_end = 0
            for m in cls._MD_LINK_RE.finditer(line):
                before = line[last_end : m.start()]
                if before:
                    elements.append({"tag": "text", "text": before})
                elements.append({"tag": "a", "text": m.group(1), "href": m.group(2)})
                last_end = m.end()
            remaining = line[last_end:]
            if remaining:
                elements.append({"tag": "text", "text": remaining})
            if not elements:
                elements.append({"tag": "text", "text": ""})
            paragraphs.append(elements)
        post_body = {"zh_cn": {"content": paragraphs}}
        return json.dumps(post_body, ensure_ascii=False)

    @classmethod
    def _normalize_markdown_images(cls, content: str) -> str:
        """Replace markdown image syntax to plain link text for Feishu compatibility."""

        def _replace(match: re.Match[str]) -> str:
            alt = (match.group(1) or "图片").strip() or "图片"
            url = (match.group(2) or "").strip()
            if url:
                return f"[{alt}]({url})"
            return alt

        return cls._MD_IMAGE_RE.sub(_replace, content or "")

    # ── Card building ────────────────────────────────────────────────

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [
            _line.strip() for _line in table_text.strip().split("\n") if _line.strip()
        ]
        if len(lines) < 3:
            return None

        def split(_line: str) -> list[str]:
            return [c.strip() for c in _line.strip("|").split("|")]

        headers = split(lines[0])
        rows = [split(_line) for _line in lines[2:]]
        columns = [
            {"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
            for i, h in enumerate(headers)
        ]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [
                {f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))}
                for r in rows
            ],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements: list[dict] = []
        last_end = 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end : m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(
                self._parse_md_table(m.group(1))
                or {"tag": "markdown", "content": m.group(1)}
            )
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    @staticmethod
    def _split_elements_by_table_limit(
        elements: list[dict], max_tables: int = 1
    ) -> list[list[dict]]:
        """Split card elements into groups with at most *max_tables* table each.

        Feishu cards have a hard limit of one table per card (API error 11310).
        """
        if not elements:
            return [[]]
        groups: list[list[dict]] = []
        current: list[dict] = []
        table_count = 0
        for el in elements:
            if el.get("tag") == "table":
                if table_count >= max_tables:
                    if current:
                        groups.append(current)
                    current = []
                    table_count = 0
                current.append(el)
                table_count += 1
            else:
                current.append(el)
        if current:
            groups.append(current)
        return groups or [[]]

    def _build_single_stream_card(self, content: str) -> str | None:
        """Build one interactive-card payload suitable for message patch."""
        elements = self._build_card_elements(content)
        chunks = self._split_elements_by_table_limit(elements)
        if len(chunks) != 1:
            return None
        card = {"config": {"wide_screen_mode": True}, "elements": chunks[0]}
        return json.dumps(card, ensure_ascii=False)

    def _resolve_receive_target(
        self,
        chat_id: str,
        sender_id: str | None,
    ) -> tuple[str, str]:
        feishu_cfg = self.config
        receive_id = (
            sender_id if sender_id and feishu_cfg.reply_mode == "p2p" else chat_id
        )
        if receive_id.startswith("oc_"):
            return "chat_id", receive_id
        if receive_id.startswith("ou_") or receive_id.startswith("on_"):
            return "open_id", receive_id
        return "chat_id", receive_id

    async def create_stream_message(
        self,
        chat_id: str,
        text: str,
        *,
        sender_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        del metadata
        content = self._normalize_markdown_images(text or "").strip()
        if not content:
            return None
        card = self._build_single_stream_card(content)
        if not card:
            return None
        receive_id_type, receive_id = self._resolve_receive_target(chat_id, sender_id)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._send_message_with_id_sync,
            receive_id_type,
            receive_id,
            "interactive",
            card,
        )

    async def patch_stream_message(self, message_id: str, text: str) -> bool:
        content = self._normalize_markdown_images(text or "").strip()
        if not content:
            return False
        card = self._build_single_stream_card(content)
        if not card:
            return False
        loop = asyncio.get_running_loop()
        return bool(
            await loop.run_in_executor(
                None,
                self._patch_message_sync,
                message_id,
                card,
            )
        )

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks: list[str] = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(
                m.group(1), f"\x00CODE{len(code_blocks) - 1}\x00", 1
            )

        elements: list[dict] = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end : m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            text = m.group(2).strip()
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**{text}**"},
                }
            )
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    # ── Reply (outbound) ─────────────────────────────────────────────

    async def send_response(
        self,
        chat_id: str,
        response: TaskResponse,
        **kwargs: Any,
    ) -> None:
        """Send reply with smart formatting and optional media attachments."""
        text = self._normalize_markdown_images(response.text)
        media_paths = response.media_paths
        sender_id = kwargs.get("sender_id")
        receive_id_type, receive_id = self._resolve_receive_target(chat_id, sender_id)

        loop = asyncio.get_running_loop()

        # Send media attachments first
        if media_paths:
            for file_path in media_paths:
                if not os.path.isfile(file_path):
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    key = await loop.run_in_executor(
                        None, self._upload_image_sync, file_path
                    )
                    if key:
                        await loop.run_in_executor(
                            None,
                            self._send_message_sync,
                            receive_id_type,
                            receive_id,
                            "image",
                            json.dumps({"image_key": key}, ensure_ascii=False),
                        )
                else:
                    key = await loop.run_in_executor(
                        None, self._upload_file_sync, file_path
                    )
                    if key:
                        media_type = self._message_type_for_file(file_path)
                        await loop.run_in_executor(
                            None,
                            self._send_message_sync,
                            receive_id_type,
                            receive_id,
                            media_type,
                            json.dumps({"file_key": key}, ensure_ascii=False),
                        )

        # Send text content with smart formatting
        if not text or not text.strip():
            if not media_paths:
                text = "任务已处理，但没有生成可发送内容。"
            else:
                return

        forced_format = str(kwargs.get("message_format", "") or "").strip().lower()
        fmt = (
            forced_format
            if forced_format in {"text", "post", "interactive"}
            else self._detect_msg_format(text)
        )

        if fmt == "text":
            text_body = json.dumps({"text": text.strip()}, ensure_ascii=False)
            await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                receive_id,
                "text",
                text_body,
            )

        elif fmt == "post":
            post_body = self._markdown_to_post(text)
            await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                receive_id,
                "post",
                post_body,
            )

        else:
            # Interactive card
            elements = self._build_card_elements(text)
            for chunk in self._split_elements_by_table_limit(elements):
                card = {"config": {"wide_screen_mode": True}, "elements": chunk}
                await loop.run_in_executor(
                    None,
                    self._send_message_sync,
                    receive_id_type,
                    receive_id,
                    "interactive",
                    json.dumps(card, ensure_ascii=False),
                )

    # ── Inbound message handling ─────────────────────────────────────

    def _on_message_sync(self, data: Any) -> None:
        """Sync callback from Feishu SDK thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        """Process one incoming Feishu message."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            sender_id = (
                sender.sender_id.open_id if sender and sender.sender_id else "unknown"
            )

            if sender and sender.sender_type == "bot":
                return

            message_id = getattr(message, "message_id", "")
            async with self._processed_lock:
                if message_id in self._processed_message_ids:
                    return
                self._processed_message_ids[message_id] = None
                while len(self._processed_message_ids) > 1000:
                    self._processed_message_ids.popitem(last=False)

            chat_id = getattr(message, "chat_id", "")
            chat_type = getattr(message, "chat_type", "p2p")
            if chat_type == "group" and not self._allow_group_message(message):
                return

            msg_type = getattr(message, "message_type", "text")

            # Add reaction
            await self._add_reaction(message_id, self.config.react_emoji)

            # Parse content
            content_parts: list[str] = []
            media_paths: list[str] = []

            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}

            if msg_type == "text":
                text = content_json.get("text", "")
                if text:
                    content_parts.append(text)

            elif msg_type == "post":
                text, image_keys = _extract_post_content(content_json)
                if text:
                    content_parts.append(text)
                for img_key in image_keys:
                    file_path, content_text = await self._download_and_save_media(
                        "image", {"image_key": img_key}, message_id
                    )
                    if file_path:
                        media_paths.append(file_path)
                    content_parts.append(content_text)

            elif msg_type in ("image", "audio", "file", "media"):
                file_path, content_text = await self._download_and_save_media(
                    msg_type, content_json, message_id
                )
                if file_path:
                    media_paths.append(file_path)
                content_parts.append(content_text)

            elif msg_type in (
                "share_chat",
                "share_user",
                "interactive",
                "share_calendar_event",
                "system",
                "merge_forward",
            ):
                text = _extract_share_card_content(content_json, msg_type)
                if text:
                    content_parts.append(text)

            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            user_input = "\n".join(content_parts).strip()
            if not user_input and not media_paths:
                return

            # If media but no text, provide a placeholder
            if not user_input and media_paths:
                user_input = f"[收到 {len(media_paths)} 个媒体文件]"

            msg = InboundMessage(
                channel="feishu",
                sender_id=sender_id,
                chat_id=chat_id,
                content=user_input,
                media_paths=media_paths,
                metadata={"chat_type": chat_type, "message_id": message_id},
            )
            await self.manager.handle_message(msg)
        except Exception as e:
            logger.warning("Feishu message handling error: %s", e)

    # ── No-op event handlers (suppress SDK noise) ────────────────────

    def _on_reaction_created(self, data: Any) -> None:
        pass

    def _on_message_read(self, data: Any) -> None:
        pass

    def _on_bot_p2p_chat_entered(self, data: Any) -> None:
        pass

    async def health_check(self) -> dict[str, Any]:
        """Check Feishu channel health status."""
        if not self._client:
            return {"status": "not_initialized", "channel": "feishu"}
        if not self._running:
            return {"status": "not_running", "channel": "feishu"}
        return {
            "status": "healthy",
            "channel": "feishu",
            "websocket_connected": self._ws_thread is not None
            and self._ws_thread.is_alive(),
        }
