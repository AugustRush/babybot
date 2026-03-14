"""OpenAI-compatible model gateway used by orchestrator and kernel executors."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

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
logger = logging.getLogger(__name__)


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
            task_id, step, self._config.model.model_name,
            tool_count, len(messages),
        )
        started = time.perf_counter()
        try:
            completion = await client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("LLM request failed task=%s step=%s", task_id, step)
            raise
        elapsed = time.perf_counter() - started

        choice = completion.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls: list[ModelToolCall] = []
        for call in msg.tool_calls or []:
            raw_arguments = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ModelToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        tc_names = [tc.name for tc in tool_calls]
        logger.info(
            "LLM response task=%s step=%s elapsed=%.2fs finish=%s text_len=%d tool_calls=%s",
            task_id, step, elapsed, choice.finish_reason,
            len(content), tc_names or "none",
        )
        return ModelResponse(
            text=content,
            tool_calls=tuple(tool_calls),
            finish_reason=choice.finish_reason or "stop",
        )

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        req = ModelRequest(
            messages=(
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_prompt),
            )
        )
        resp = await self.generate(req, ExecutionContext())
        return resp.text.strip()

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
    ) -> T | None:
        instruction = (
            "请严格输出 JSON 对象，不要输出 markdown 或额外解释。"
            f"JSON 字段要求：{json.dumps(model_cls.model_json_schema(), ensure_ascii=False)}"
        )
        text = await self.complete(
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\n{instruction}",
        )
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM structured parse failed model=%s", model_cls.__name__)
            return None
        try:
            return model_cls.model_validate(data)
        except Exception:
            logger.warning("LLM structured validation failed model=%s", model_cls.__name__)
            return None

    @staticmethod
    def _to_openai_message(message: ModelMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
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
