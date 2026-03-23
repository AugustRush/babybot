"""OpenAI-compatible model gateway used by orchestrator and kernel executors."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel

from .agent_kernel import (
    ExecutionContext,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from .config import Config

T = TypeVar("T", bound=BaseModel)
StreamTextCallback = Callable[[str], Awaitable[None] | None]
logger = logging.getLogger(__name__)


def _image_to_content_part(image_ref: str) -> dict[str, Any]:
    """File path or data URI → OpenAI image_url content part."""
    if image_ref.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": image_ref}}
    path = Path(image_ref)
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _is_supported_image_ref(image_ref: str) -> bool:
    """Only pass actual image inputs to multimodal APIs."""
    if image_ref.startswith("data:"):
        return image_ref.startswith("data:image/")
    mime = mimetypes.guess_type(image_ref)[0] or ""
    return mime.startswith("image/")


class OpenAICompatibleGateway(ModelProvider):
    """Model gateway for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: Config):
        self._config = config
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self._config.model.api_key}
        if self._config.model.api_base:
            kwargs["base_url"] = self._config.model.api_base
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def close(self) -> None:
        """Release the underlying HTTP client resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    async def _call_stream_callback(
        self,
        callback: StreamTextCallback | None,
        text: str,
    ) -> None:
        if callback is None:
            return
        try:
            maybe_awaitable = callback(text)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception:
            logger.warning("Streaming callback failed", exc_info=True)

    @staticmethod
    def _extract_stream_delta_text(delta: Any) -> str:
        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    item_type = getattr(item, "type", None)
                    txt = getattr(item, "text", "")
                    if item_type == "text" and isinstance(txt, str):
                        parts.append(txt)
                    continue
                if item.get("type") == "text":
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "".join(parts)
        return ""

    async def _complete_streaming(
        self,
        messages: list[dict[str, Any]],
        heartbeat: Any = None,
        on_stream_text: StreamTextCallback | None = None,
    ) -> str:
        client = self._ensure_client()
        per_call_timeout = max(1.0, float(self._config.system.subtask_timeout))
        kwargs: dict[str, Any] = {
            "model": self._config.model.model_name,
            "messages": messages,
            "temperature": self._config.model.temperature,
            "max_tokens": self._config.model.max_tokens,
            "stream": True,
        }
        accumulated_text = ""

        async def _consume() -> None:
            nonlocal accumulated_text
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                for choice in getattr(chunk, "choices", []) or []:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    piece = self._extract_stream_delta_text(delta)
                    if not piece:
                        continue
                    accumulated_text += piece
                    await self._call_stream_callback(on_stream_text, accumulated_text)

        if heartbeat is not None:
            async with heartbeat.keep_alive():
                await asyncio.wait_for(_consume(), timeout=per_call_timeout)
        else:
            await asyncio.wait_for(_consume(), timeout=per_call_timeout)
        return accumulated_text.strip()

    @staticmethod
    def _delta_tool_calls(delta: Any) -> list[Any]:
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls is None and isinstance(delta, dict):
            tool_calls = delta.get("tool_calls")
        if not tool_calls:
            return []
        return list(tool_calls)

    @staticmethod
    def _read_delta_field(value: Any, field: str) -> Any:
        if isinstance(value, dict):
            return value.get(field)
        return getattr(value, field, None)

    def _extract_reply_text_from_stream_tool_calls(
        self,
        tool_calls: dict[int, dict[str, str]],
    ) -> str:
        for item in tool_calls.values():
            if item.get("name") != "reply_to_user":
                continue
            raw_arguments = item.get("arguments", "")
            return self._extract_text_field_from_partial_json(raw_arguments)
        return ""

    @staticmethod
    def _extract_text_field_from_partial_json(raw: str) -> str:
        marker = '"text"'
        start = raw.find(marker)
        if start < 0:
            return ""
        colon = raw.find(":", start + len(marker))
        if colon < 0:
            return ""
        idx = colon + 1
        while idx < len(raw) and raw[idx] in " \t\r\n":
            idx += 1
        if idx >= len(raw) or raw[idx] != '"':
            return ""
        idx += 1
        chars: list[str] = []
        escaped = False
        while idx < len(raw):
            ch = raw[idx]
            if escaped:
                chars.append("\\" + ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                break
            else:
                chars.append(ch)
            idx += 1
        return OpenAICompatibleGateway._decode_partial_json_string("".join(chars))

    @staticmethod
    def _decode_partial_json_string(raw: str) -> str:
        candidate = raw
        while candidate:
            try:
                return json.loads(f'"{candidate}"')
            except json.JSONDecodeError:
                candidate = candidate[:-1]
        return ""

    def _parse_tool_calls(
        self,
        raw_calls: list[tuple[str, str, str]],
        *,
        task_id: Any,
        step: Any,
    ) -> list[ModelToolCall]:
        tool_calls: list[ModelToolCall] = []
        for call_id, call_name, raw_arguments in raw_calls:
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid tool arguments JSON task=%s step=%s tool=%s raw=%s",
                    task_id,
                    step,
                    call_name,
                    raw_arguments,
                )
                arguments = {
                    "__tool_argument_parse_error__": True,
                    "__raw_arguments__": raw_arguments,
                }
            tool_calls.append(
                ModelToolCall(
                    call_id=call_id,
                    name=call_name,
                    arguments=arguments,
                )
            )
        return tool_calls

    async def _generate_streaming(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        heartbeat: Any = None,
        on_stream_text: StreamTextCallback | None = None,
        task_id: Any,
        step: Any,
        per_call_timeout: float,
    ) -> ModelResponse:
        streamed_text = ""
        saw_content_text = False
        finish_reason = "stop"
        tool_calls: dict[int, dict[str, str]] = {}

        async def _consume() -> None:
            nonlocal streamed_text, finish_reason, saw_content_text
            stream = await client.chat.completions.create(**kwargs, stream=True)
            async for chunk in stream:
                for choice in getattr(chunk, "choices", []) or []:
                    finish_reason = (
                        getattr(choice, "finish_reason", None) or finish_reason
                    )
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    piece = self._extract_stream_delta_text(delta)
                    if piece:
                        saw_content_text = True
                        streamed_text += piece

                    for item in self._delta_tool_calls(delta):
                        index = self._read_delta_field(item, "index")
                        if index is None:
                            index = len(tool_calls)
                        acc = tool_calls.setdefault(
                            int(index),
                            {"id": "", "name": "", "arguments": ""},
                        )
                        item_id = self._read_delta_field(item, "id")
                        if isinstance(item_id, str) and item_id:
                            acc["id"] = item_id
                        function = self._read_delta_field(item, "function")
                        if function is None:
                            continue
                        name = self._read_delta_field(function, "name")
                        if isinstance(name, str) and name:
                            acc["name"] += name
                        arguments = self._read_delta_field(function, "arguments")
                        if isinstance(arguments, str) and arguments:
                            acc["arguments"] += arguments

                    next_text = streamed_text
                    if not saw_content_text:
                        next_text = self._extract_reply_text_from_stream_tool_calls(
                            tool_calls
                        )
                    if next_text and next_text != streamed_text:
                        streamed_text = next_text
                        await self._call_stream_callback(on_stream_text, streamed_text)
                    elif piece:
                        await self._call_stream_callback(on_stream_text, streamed_text)

        if heartbeat is not None:
            async with heartbeat.keep_alive():
                await asyncio.wait_for(_consume(), timeout=per_call_timeout)
        else:
            await asyncio.wait_for(_consume(), timeout=per_call_timeout)

        parsed_tool_calls = self._parse_tool_calls(
            [
                (
                    item.get("id", f"tool_call_{index}"),
                    item.get("name", ""),
                    item.get("arguments", ""),
                )
                for index, item in sorted(tool_calls.items())
            ],
            task_id=task_id,
            step=step,
        )
        return ModelResponse(
            text=(streamed_text.strip() if saw_content_text else ""),
            tool_calls=tuple(parsed_tool_calls),
            finish_reason=finish_reason or "stop",
            metadata={
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "model": self._config.model.model_name,
            },
        )

    async def generate(
        self,
        request: ModelRequest,
        context: ExecutionContext,
    ) -> ModelResponse:
        client = self._ensure_client()
        messages = [self._to_openai_message(msg) for msg in request.messages]
        kwargs: dict[str, Any] = {
            "model": self._config.model.model_name,
            "messages": messages,
            "temperature": self._config.model.temperature,
            "max_tokens": self._config.model.max_tokens,
        }
        tool_count = 0
        if request.tools:
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = "auto"
            tool_count = len(request.tools)

        meta = request.metadata or {}
        task_id = meta.get("task_id", "?")
        step = meta.get("step", "?")
        logger.info(
            "LLM request task=%s step=%s model=%s tools=%d msgs=%d",
            task_id,
            step,
            self._config.model.model_name,
            tool_count,
            len(messages),
        )
        per_call_timeout = max(1.0, float(self._config.system.subtask_timeout))
        logger.info(
            "LLM request timeout task=%s step=%s timeout=%.1fs",
            task_id,
            step,
            per_call_timeout,
        )
        started = time.perf_counter()
        heartbeat = context.state.get("heartbeat")
        stream_callback = context.state.get("stream_callback")
        try:
            if stream_callback is not None:
                return await self._generate_streaming(
                    client,
                    kwargs,
                    heartbeat=heartbeat,
                    on_stream_text=stream_callback,
                    task_id=task_id,
                    step=step,
                    per_call_timeout=per_call_timeout,
                )
            if heartbeat is not None:
                async with heartbeat.keep_alive():
                    completion = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=per_call_timeout,
                    )
            else:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=per_call_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "LLM request timeout task=%s step=%s timeout=%.1fs",
                task_id,
                step,
                per_call_timeout,
            )
            raise
        except Exception:
            logger.exception("LLM request failed task=%s step=%s", task_id, step)
            raise
        elapsed = time.perf_counter() - started

        choice = completion.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = self._parse_tool_calls(
            [
                (
                    call.id,
                    call.function.name,
                    call.function.arguments or "{}",
                )
                for call in msg.tool_calls or []
            ],
            task_id=task_id,
            step=step,
        )

        tc_names = [tc.name for tc in tool_calls]
        logger.info(
            "LLM response task=%s step=%s elapsed=%.2fs finish=%s text_len=%d tool_calls=%s",
            task_id,
            step,
            elapsed,
            choice.finish_reason,
            len(content),
            tc_names or "none",
        )
        usage = completion.usage
        metadata: dict[str, Any] = {
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            "model": getattr(completion, "model", self._config.model.model_name),
        }
        return ModelResponse(
            text=content,
            tool_calls=tuple(tool_calls),
            finish_reason=choice.finish_reason or "stop",
            metadata=metadata,
        )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        heartbeat: Any = None,
        on_stream_text: StreamTextCallback | None = None,
    ) -> str:
        if on_stream_text is not None:
            return await self._complete_streaming(
                [
                    self._to_openai_message(
                        ModelMessage(role="system", content=system_prompt)
                    ),
                    self._to_openai_message(
                        ModelMessage(role="user", content=user_prompt)
                    ),
                ],
                heartbeat=heartbeat,
                on_stream_text=on_stream_text,
            )
        req = ModelRequest(
            messages=(
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_prompt),
            )
        )
        state: dict[str, Any] = {}
        if heartbeat is not None:
            state["heartbeat"] = heartbeat
        resp = await self.generate(req, ExecutionContext(state=state))
        return resp.text.strip()

    async def complete_messages(
        self,
        messages: list[ModelMessage],
        heartbeat: Any = None,
        on_stream_text: StreamTextCallback | None = None,
    ) -> str:
        """Send a full multi-turn message list to the model and return text."""
        if on_stream_text is not None:
            payload = [self._to_openai_message(msg) for msg in messages]
            return await self._complete_streaming(
                payload,
                heartbeat=heartbeat,
                on_stream_text=on_stream_text,
            )
        req = ModelRequest(messages=tuple(messages))
        state: dict[str, Any] = {}
        if heartbeat is not None:
            state["heartbeat"] = heartbeat
        resp = await self.generate(req, ExecutionContext(state=state))
        return resp.text.strip()

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
        heartbeat: Any = None,
    ) -> T | None:
        instruction = (
            "请严格输出 JSON 对象，不要输出 markdown 或额外解释。"
            f"JSON 字段要求：{json.dumps(model_cls.model_json_schema(), ensure_ascii=False)}"
        )
        text = await self.complete(
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\n{instruction}",
            heartbeat=heartbeat,
        )
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Try parsing the full cleaned text first
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: extract outermost braces
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning(
                        "LLM structured parse failed model=%s", model_cls.__name__
                    )
                    return None
            else:
                logger.warning(
                    "LLM structured parse failed model=%s", model_cls.__name__
                )
                return None
        try:
            return model_cls.model_validate(data)
        except Exception:
            logger.warning(
                "LLM structured validation failed model=%s", model_cls.__name__
            )
            return None

    @staticmethod
    def _to_openai_message(message: ModelMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.images:
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"type": "text", "text": message.content})
            for img in message.images:
                if not _is_supported_image_ref(img):
                    logger.warning(
                        "Skipping non-image attachment for multimodal input: %s", img
                    )
                    continue
                try:
                    parts.append(_image_to_content_part(img))
                except FileNotFoundError:
                    logger.warning("Image not found: %s", img)
            payload["content"] = parts or message.content
        else:
            payload["content"] = message.content
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.role == "assistant" and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return payload
