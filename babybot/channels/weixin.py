"""Minimal Weixin channel integration for BabyBot."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import quote
import uuid

import httpx

from ..config import WeixinConfig
from ..orchestrator import TaskResponse
from .base import BaseChannel, InboundMessage
from .tools import ChannelCapabilities
from .weixin_tools import WeixinChannelTools

logger = logging.getLogger(__name__)

_ITEM_TEXT = 1
_ITEM_IMAGE = 2
_ITEM_FILE = 4
_MESSAGE_TYPE_USER = 1
_MESSAGE_TYPE_BOT = 2
_MESSAGE_STATE_FINISH = 2
_UPLOAD_MEDIA_IMAGE = 1
_UPLOAD_MEDIA_FILE = 3
_BASE_INFO: dict[str, str] = {"channel_version": "1.0.2"}
_MAX_TEXT_CHUNK_LEN = 4000
_DEDUP_LIMIT = 1000
_QR_PNG_NAME = "weixin-login-qr.png"
_STATE_FILE_NAME = "account.json"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


class WeixinChannel(BaseChannel):
    """Weixin channel backed by the ilinkai long-poll HTTP API."""

    name = "weixin"
    display_name = "Weixin"

    def __init__(self, config: WeixinConfig, manager: Any) -> None:
        super().__init__(config, manager)
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._get_updates_buf = ""
        self._context_tokens: dict[str, str] = {}
        self._processed_ids: OrderedDict[str, None] = OrderedDict()
        self._next_poll_timeout_s = max(int(self.config.poll_timeout or 35), 5)
        self._token = self.config.token or ""
        self._state_dir: Path | None = None
        media_dir = self.config.media_dir
        if media_dir:
            self._media_dir = Path(media_dir).expanduser()
        elif self.config.state_dir:
            self._media_dir = Path(self.config.state_dir).expanduser() / "media"
        else:
            self._media_dir = (
                Path(os.getenv("BABYBOT_HOME", "~/.babybot")).expanduser()
                / "media"
                / "weixin"
            )

    async def start(self) -> None:
        if self._running:
            return
        if not self.config.enabled:
            logger.info("Weixin channel disabled; skipping start")
            return
        if self.config.token:
            self._token = self.config.token
        elif not self._token and not self._load_state():
            logged_in = await self.login()
            if not logged_in:
                logger.warning("Weixin QR login failed; channel not started")
                return

        self._media_dir.mkdir(parents=True, exist_ok=True)
        timeout = httpx.Timeout(self._next_poll_timeout_s + 10, connect=30)
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Weixin channel started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("Weixin channel stopped")

    def get_channel_tools(self) -> WeixinChannelTools:
        return WeixinChannelTools(self)

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_text=True,
            supports_image=True,
            supports_file=True,
            max_text_length=_MAX_TEXT_CHUNK_LEN,
            max_image_size=20 * 1024 * 1024,
            max_file_size=30 * 1024 * 1024,
            supported_image_formats=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
        )

    async def send_response(
        self, chat_id: str, response: TaskResponse, **kwargs: Any
    ) -> None:
        del kwargs
        context_token = self._context_tokens.get(chat_id, "")
        if not context_token:
            logger.warning(
                "Weixin context_token missing for chat_id=%s; skipping reply", chat_id
            )
            return
        for media_path in response.media_paths or []:
            await self._send_media_file(chat_id, media_path, context_token)
        text = (response.text or "").strip()
        if not text:
            return
        for chunk in _chunk_text(text, _MAX_TEXT_CHUNK_LEN):
            await self._send_text(chat_id, chunk, context_token)

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "running" if self._running else "stopped",
            "channel": self.name,
            "authenticated": bool(self._token),
        }

    async def login(self, force: bool = False) -> bool:
        if force:
            self._token = ""
            self._get_updates_buf = ""
            state_file = self._get_state_file()
            if state_file.exists():
                state_file.unlink()
        elif self.config.token:
            self._token = self.config.token
            return True
        elif self._token or self._load_state():
            return True

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=30),
            follow_redirects=True,
        )
        self._running = True
        try:
            return await self._qr_login()
        finally:
            self._running = False
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Weixin poll failed: %s", exc)
                await asyncio.sleep(2)

    async def _poll_once(self) -> None:
        data = await self._api_post(
            "ilink/bot/getupdates",
            {"get_updates_buf": self._get_updates_buf, "base_info": _BASE_INFO},
        )
        server_timeout_ms = data.get("longpolling_timeout_ms")
        if isinstance(server_timeout_ms, int) and server_timeout_ms > 0:
            self._next_poll_timeout_s = max(server_timeout_ms // 1000, 5)
            if self._client is not None:
                self._client.timeout = httpx.Timeout(
                    self._next_poll_timeout_s + 10,
                    connect=30,
                )
        new_buf = data.get("get_updates_buf", "")
        if new_buf:
            self._get_updates_buf = new_buf
            self._save_state()
        for msg in data.get("msgs") or []:
            await self._process_message(msg)

    async def _qr_login(self) -> bool:
        logger.info("Starting Weixin QR login")
        data = await self._api_get(
            "ilink/bot/get_bot_qrcode",
            params={"bot_type": "3"},
            auth=False,
        )
        qrcode_id = data.get("qrcode", "")
        scan_url = data.get("qrcode_img_content", "") or qrcode_id
        if not qrcode_id or not scan_url:
            logger.warning("Weixin QR login did not return a usable QR code")
            return False

        self._print_qr_code(scan_url)
        saved_path = self._save_qr_png(scan_url)
        if saved_path:
            logger.info("Weixin QR code saved to %s", saved_path)

        while self._running:
            status_data = await self._api_get(
                "ilink/bot/get_qrcode_status",
                params={"qrcode": qrcode_id},
                auth=False,
                extra_headers={"iLink-App-ClientVersion": "1"},
            )
            status = status_data.get("status", "")
            if status == "confirmed":
                token = status_data.get("bot_token", "")
                if not token:
                    logger.warning("Weixin QR login confirmed but no token returned")
                    return False
                self._token = token
                self.config.token = token
                base_url = status_data.get("baseurl", "")
                if base_url:
                    self.config.base_url = base_url
                self._save_state()
                logger.info("Weixin QR login succeeded")
                return True
            if status == "expired":
                logger.warning("Weixin QR code expired")
                return False
            if status == "scaned":
                logger.info("Weixin QR code scanned; waiting for confirmation")
            await asyncio.sleep(1)
        return False

    async def _process_message(self, msg: dict[str, Any]) -> None:
        if msg.get("message_type") == _MESSAGE_TYPE_BOT:
            return
        from_user_id = str(msg.get("from_user_id", "") or "")
        if not from_user_id:
            return
        if self.config.allow_from and from_user_id not in set(self.config.allow_from):
            return

        message_id = str(msg.get("message_id", "") or msg.get("seq", ""))
        if not message_id:
            message_id = f"{from_user_id}_{msg.get('create_time_ms', '')}"
        if message_id in self._processed_ids:
            return
        self._processed_ids[message_id] = None
        while len(self._processed_ids) > _DEDUP_LIMIT:
            self._processed_ids.popitem(last=False)

        context_token = str(msg.get("context_token", "") or "")
        if context_token:
            self._context_tokens[from_user_id] = context_token

        content_parts: list[str] = []
        media_paths: list[str] = []
        for item in msg.get("item_list") or []:
            item_type = item.get("type")
            if item_type == _ITEM_TEXT:
                text = ((item.get("text_item") or {}).get("text", "") or "").strip()
                if text:
                    content_parts.append(text)
            elif item_type == _ITEM_IMAGE:
                image_item = item.get("image_item") or {}
                file_path = await self._download_media_item(image_item, "image")
                if file_path:
                    media_paths.append(file_path)
                content_parts.append("[image]")
            elif item_type == _ITEM_FILE:
                file_item = item.get("file_item") or {}
                file_name = str(file_item.get("file_name", "") or "")
                file_path = await self._download_media_item(file_item, "file", file_name or None)
                if file_path:
                    media_paths.append(file_path)
                content_parts.append(f"[file: {file_name}]" if file_name else "[file]")

        if not content_parts and not media_paths:
            return
        inbound = InboundMessage(
            channel=self.name,
            sender_id=from_user_id,
            chat_id=from_user_id,
            content="\n".join(content_parts) if content_parts else f"[收到 {len(media_paths)} 个媒体文件]",
            media_paths=media_paths,
            metadata={"message_id": message_id},
        )
        if self.manager is not None:
            await self.manager.handle_message(inbound)

    async def _download_media_item(
        self,
        typed_item: dict[str, Any],
        media_type: str,
        filename: str | None = None,
    ) -> str | None:
        if self._client is None:
            return None
        try:
            media = typed_item.get("media") or {}
            encrypt_query_param = media.get("encrypt_query_param", "")
            if not encrypt_query_param:
                return None
            raw_aeskey_hex = str(typed_item.get("aeskey", "") or "")
            media_aes_key_b64 = str(media.get("aes_key", "") or "")
            aes_key_b64 = ""
            if raw_aeskey_hex:
                aes_key_b64 = base64.b64encode(bytes.fromhex(raw_aeskey_hex)).decode()
            elif media_aes_key_b64:
                aes_key_b64 = media_aes_key_b64
            response = await self._client.get(
                f"{self.config.cdn_base_url.rstrip('/')}/download?encrypted_query_param={quote(encrypt_query_param)}"
            )
            response.raise_for_status()
            data = response.content
            if aes_key_b64 and data:
                data = _decrypt_aes_ecb(data, aes_key_b64)
            if not data:
                return None
            self._media_dir.mkdir(parents=True, exist_ok=True)
            if not filename:
                filename = f"{media_type}_{int(time.time())}_{abs(hash(encrypt_query_param)) % 100000}{_ext_for_type(media_type)}"
            safe_name = os.path.basename(filename)
            file_path = self._media_dir / safe_name
            file_path.write_bytes(data)
            return str(file_path)
        except Exception as exc:
            logger.warning("Failed to download Weixin %s media: %s", media_type, exc)
            return None

    async def _send_text(
        self,
        to_user_id: str,
        text: str,
        context_token: str = "",
    ) -> None:
        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": f"babybot-{uuid.uuid4().hex[:12]}",
                "message_type": _MESSAGE_TYPE_BOT,
                "message_state": _MESSAGE_STATE_FINISH,
                "item_list": [{"type": _ITEM_TEXT, "text_item": {"text": text}}],
                "context_token": context_token,
            },
            "base_info": _BASE_INFO,
        }
        data = await self._api_post("ilink/bot/sendmessage", body)
        errcode = data.get("errcode", 0)
        ret = data.get("ret", 0)
        if errcode not in (0, None) or ret not in (0, None):
            logger.warning(
                "Weixin send returned ret=%s errcode=%s errmsg=%s",
                ret,
                errcode,
                data.get("errmsg", ""),
            )

    async def _send_media_file(
        self,
        to_user_id: str,
        media_path: str,
        context_token: str,
    ) -> None:
        file_obj = Path(media_path)
        if not file_obj.is_file():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        if self._client is None:
            raise RuntimeError("Weixin client not started")

        raw_data = file_obj.read_bytes()
        raw_size = len(raw_data)
        raw_md5 = hashlib.md5(raw_data).hexdigest()
        ext = file_obj.suffix.lower()
        if ext in _IMAGE_EXTS:
            upload_type = _UPLOAD_MEDIA_IMAGE
            item_type = _ITEM_IMAGE
            item_key = "image_item"
        else:
            upload_type = _UPLOAD_MEDIA_FILE
            item_type = _ITEM_FILE
            item_key = "file_item"

        aes_key_raw = os.urandom(16)
        aes_key_hex = aes_key_raw.hex()
        padded_size = ((raw_size + 1 + 15) // 16) * 16
        file_key = os.urandom(16).hex()
        upload_resp = await self._api_post(
            "ilink/bot/getuploadurl",
            {
                "filekey": file_key,
                "media_type": upload_type,
                "to_user_id": to_user_id,
                "rawsize": raw_size,
                "rawfilemd5": raw_md5,
                "filesize": padded_size,
                "no_need_thumb": True,
                "aeskey": aes_key_hex,
            },
        )
        upload_param = upload_resp.get("upload_param", "")
        if not upload_param:
            raise RuntimeError(f"getuploadurl returned no upload_param: {upload_resp}")
        encrypted_data = _encrypt_aes_ecb(raw_data, base64.b64encode(aes_key_raw).decode())
        cdn_resp = await self._client.post(
            f"{self.config.cdn_base_url.rstrip('/')}/upload?encrypted_query_param={quote(upload_param)}&filekey={quote(file_key)}",
            content=encrypted_data,
            headers={"Content-Type": "application/octet-stream"},
        )
        cdn_resp.raise_for_status()
        download_param = cdn_resp.headers.get("x-encrypted-param", "")
        if not download_param:
            raise RuntimeError("CDN upload response missing x-encrypted-param header")
        cdn_aes_key_b64 = base64.b64encode(aes_key_hex.encode()).decode()
        media_item: dict[str, Any] = {
            "media": {
                "encrypt_query_param": download_param,
                "aes_key": cdn_aes_key_b64,
                "encrypt_type": 1,
            }
        }
        if item_type == _ITEM_IMAGE:
            media_item["mid_size"] = padded_size
        else:
            media_item["file_name"] = file_obj.name
            media_item["len"] = str(raw_size)
        await self._api_post(
            "ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": to_user_id,
                    "client_id": f"babybot-{uuid.uuid4().hex[:12]}",
                    "message_type": _MESSAGE_TYPE_BOT,
                    "message_state": _MESSAGE_STATE_FINISH,
                    "item_list": [{"type": item_type, item_key: media_item}],
                    "context_token": context_token,
                },
                "base_info": _BASE_INFO,
            },
        )

    async def _api_get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Weixin client not started")
        headers = self._make_headers(auth=auth)
        if extra_headers:
            headers.update(extra_headers)
        response = await self._client.get(
            f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Weixin API returned non-object response")
        return data

    async def _api_post(
        self,
        endpoint: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Weixin client not started")
        payload = dict(body or {})
        payload.setdefault("base_info", _BASE_INFO)
        response = await self._client.post(
            f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            json=payload,
            headers=self._make_headers(),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Weixin API returned non-object response")
        return data

    def _make_headers(self, *, auth: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
        }
        token = self._token or self.config.token
        if auth and token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_state_dir(self) -> Path:
        if self._state_dir is not None:
            return self._state_dir
        if self.config.state_dir:
            state_dir = Path(self.config.state_dir).expanduser()
        else:
            state_dir = Path(os.getenv("BABYBOT_HOME", "~/.babybot")).expanduser() / "weixin"
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = state_dir
        return state_dir

    def _get_state_file(self) -> Path:
        return self._get_state_dir() / _STATE_FILE_NAME

    def _load_state(self) -> bool:
        state_file = self._get_state_file()
        if not state_file.exists():
            return False
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load Weixin state file %s: %s", state_file, exc)
            return False
        token = str(state.get("token", "") or "")
        if not token:
            return False
        self._token = token
        self.config.token = token
        self._get_updates_buf = str(state.get("get_updates_buf", "") or "")
        base_url = str(state.get("base_url", "") or "")
        if base_url:
            self.config.base_url = base_url
        return True

    def _save_state(self) -> None:
        state_file = self._get_state_file()
        payload = {
            "token": self._token,
            "base_url": self.config.base_url,
            "get_updates_buf": self._get_updates_buf,
        }
        state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _print_qr_code(self, url: str) -> None:
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception:
            logger.info("Weixin login QR URL: %s", url)
            print(f"\nWeixin login QR URL: {url}\n")

    def _save_qr_png(self, url: str) -> str:
        target = self._get_state_dir() / _QR_PNG_NAME
        try:
            import qrcode

            image = qrcode.make(url)
            image.save(str(target))
            return str(target)
        except Exception as exc:
            logger.warning("Failed to save Weixin QR PNG to %s: %s", target, exc)
            return ""


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


def _parse_aes_key(aes_key_b64: str) -> bytes:
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32 and re.fullmatch(rb"[0-9a-fA-F]{32}", decoded):
        return bytes.fromhex(decoded.decode("ascii"))
    raise ValueError(
        f"aes_key must decode to 16 raw bytes or 32-char hex string, got {len(decoded)} bytes"
    )


def _encrypt_aes_ecb(data: bytes, aes_key_b64: str) -> bytes:
    try:
        key = _parse_aes_key(aes_key_b64)
    except Exception as exc:
        logger.warning("Failed to parse AES key for encryption, sending raw: %s", exc)
        return data
    pad_len = 16 - len(data) % 16
    padded = data + bytes([pad_len] * pad_len)
    try:
        from Crypto.Cipher import AES

        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(padded)
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher_obj = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher_obj.encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    except ImportError:
        logger.warning("Cannot encrypt media: install 'pycryptodome' or 'cryptography'")
        return data


def _decrypt_aes_ecb(data: bytes, aes_key_b64: str) -> bytes:
    try:
        key = _parse_aes_key(aes_key_b64)
    except Exception as exc:
        logger.warning("Failed to parse AES key, returning raw data: %s", exc)
        return data
    try:
        from Crypto.Cipher import AES

        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.decrypt(data)
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher_obj = Cipher(algorithms.AES(key), modes.ECB())
        decryptor = cipher_obj.decryptor()
        return decryptor.update(data) + decryptor.finalize()
    except ImportError:
        logger.warning("Cannot decrypt media: install 'pycryptodome' or 'cryptography'")
        return data


def _ext_for_type(media_type: str) -> str:
    return {
        "image": ".jpg",
        "file": "",
    }.get(media_type, "")
