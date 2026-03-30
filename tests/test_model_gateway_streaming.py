from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from babybot.agent_kernel import ExecutionContext, ModelMessage, ModelRequest
from babybot.model_gateway import OpenAICompatibleGateway


class _Config:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            model_name="mock-model",
            api_key="mock-key",
            api_base="",
            temperature=0.1,
            max_tokens=128,
        )
        self.system = SimpleNamespace(subtask_timeout=10)


class _FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._iterator = iter(chunks)

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeCompletions:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.calls: list[dict] = []
        self._chunks = chunks

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return _FakeStream(self._chunks)


class _TimeoutCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        raise asyncio.TimeoutError


def test_complete_streams_accumulated_text_to_callback() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="你"), finish_reason=None)
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="好"), finish_reason=None)
            ]
        ),
    ]
    completions = _FakeCompletions(chunks)
    gateway = OpenAICompatibleGateway(_Config())  # type: ignore[arg-type]
    gateway._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    updates: list[str] = []

    async def _run() -> str:
        return await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            on_stream_text=updates.append,
        )

    text = asyncio.run(_run())

    assert text == "你好"
    assert updates == ["你", "你好"]
    assert completions.calls[0]["stream"] is True


def test_complete_messages_streaming_supports_list_delta_content() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=[{"type": "text", "text": "A"}]),
                    finish_reason=None,
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=[{"type": "text", "text": "B"}]),
                    finish_reason="stop",
                )
            ]
        ),
    ]
    completions = _FakeCompletions(chunks)
    gateway = OpenAICompatibleGateway(_Config())  # type: ignore[arg-type]
    gateway._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    updates: list[str] = []

    async def _run() -> str:
        return await gateway.complete_messages(
            messages=[ModelMessage(role="user", content="hello")],
            on_stream_text=updates.append,
        )

    text = asyncio.run(_run())

    assert text == "AB"
    assert updates == ["A", "AB"]


def test_generate_streams_reply_to_user_tool_arguments() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(
                                    name="reply_to_user", arguments='{"text":"你'
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                ),
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="", arguments='好"}'),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
            ],
        ),
    ]
    completions = _FakeCompletions(chunks)
    gateway = OpenAICompatibleGateway(_Config())  # type: ignore[arg-type]
    gateway._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    updates: list[str] = []

    async def _run():
        return await gateway.generate(
            ModelRequest(
                messages=(ModelMessage(role="user", content="hello"),),
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "reply_to_user",
                            "description": "reply",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                    },
                ),
            ),
            ExecutionContext(state={"stream_callback": updates.append}),
        )

    response = asyncio.run(_run())

    assert updates == ["你", "你好"]
    assert response.tool_calls[0].name == "reply_to_user"
    assert response.tool_calls[0].arguments == {"text": "你好"}
    assert completions.calls[0]["stream"] is True


def test_decode_partial_json_string_keeps_valid_prefix_before_trailing_escape() -> None:
    gateway = OpenAICompatibleGateway(_Config())  # type: ignore[arg-type]

    assert gateway._decode_partial_json_string('\\u4f60\\') == '你'
    assert gateway._decode_partial_json_string('abc\\u4f60\\') == 'abc你'


def test_complete_expected_timeout_logs_warning_not_error(caplog: pytest.LogCaptureFixture) -> None:
    completions = _TimeoutCompletions()
    gateway = OpenAICompatibleGateway(_Config())  # type: ignore[arg-type]
    gateway._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def _run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await gateway.complete(
                system_prompt="sys",
                user_prompt="user",
                timeout=1.0,
                expected_timeout=True,
            )

    with caplog.at_level(logging.WARNING, logger="babybot.model_gateway"):
        asyncio.run(_run())

    assert any(
        record.levelno == logging.WARNING
        and "LLM request timeout" in record.message
        for record in caplog.records
    )
    assert not any(
        record.levelno >= logging.ERROR and "LLM request timeout" in record.message
        for record in caplog.records
    )
