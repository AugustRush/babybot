"""Feishu channel integration using lark-oapi WebSocket long connection."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from collections import OrderedDict
from typing import Any, Literal

from ..config import Config
from ..orchestrator import OrchestratorAgent


FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None


class FeishuChannel:
    """Minimal Feishu bot channel."""

    def __init__(self, config: Config, orchestrator: OrchestratorAgent):
        self.config = config
        self.orchestrator = orchestrator
        self._client: Any = None
        self._ws_client: Any = None
        self._running = False
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Start Feishu WebSocket client and keep running."""
        feishu_cfg = self.config.feishu
        if not FEISHU_AVAILABLE:
            raise RuntimeError("Feishu SDK not installed. Run: uv sync")
        if not feishu_cfg.app_id or not feishu_cfg.app_secret:
            raise ValueError("Feishu app_id/app_secret not configured in channels.feishu")

        import lark_oapi as lark

        self._loop = asyncio.get_running_loop()
        self._running = True

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
        print("Feishu channel started (WebSocket long connection).")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop Feishu channel."""
        self._running = False

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
            sender_id = sender.sender_id.open_id if sender and sender.sender_id else "unknown"

            if sender and sender.sender_type == "bot":
                return

            message_id = getattr(message, "message_id", "")
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            chat_type = getattr(message, "chat_type", "p2p")
            if chat_type == "group" and not self._allow_group_message(message):
                return

            msg_type = getattr(message, "message_type", "text")
            if msg_type != "text":
                await self._reply_text(message.chat_id, "目前仅支持文本消息。")
                return

            content_json = json.loads(message.content or "{}")
            user_input = (content_json.get("text", "") or "").strip()
            if not user_input:
                return

            lock = self._chat_locks.setdefault(message.chat_id, asyncio.Lock())
            async with lock:
                try:
                    answer = await self.orchestrator.process_task(user_input)
                except Exception as e:
                    answer = f"处理失败：{e}"

            await self._reply_text(message.chat_id, answer, sender_id=sender_id)
        except Exception as e:
            print(f"Feishu message handling error: {e}")

    def _allow_group_message(self, message: Any) -> bool:
        policy: Literal["open", "mention"] = self.config.feishu.group_policy
        if policy == "open":
            return True
        raw_content = getattr(message, "content", "") or ""
        return "@_all" in raw_content or "<at" in raw_content

    async def _reply_text(self, chat_id: str, text: str, sender_id: str | None = None) -> None:
        """Send text message to chat."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        receive_id = sender_id if sender_id and self.config.feishu.reply_mode == "p2p" else chat_id
        receive_id_type = "open_id" if sender_id and self.config.feishu.reply_mode == "p2p" else "chat_id"

        content = json.dumps({"text": text}, ensure_ascii=False)
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("text")
                .content(content)
                .build(),
            )
            .build()
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._client.im.v1.message.create, request)
        if not response.success():
            print(f"Feishu reply failed: code={response.code}, msg={response.msg}")
