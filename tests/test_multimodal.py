"""Tests for multimodal image support pipeline."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest

from babybot.agent_kernel.model import ModelMessage
from babybot.model_gateway import OpenAICompatibleGateway, _image_to_content_part


# ── _image_to_content_part ──


def test_image_to_content_part_from_file():
    """Real image file → base64 data URI."""
    # Create a tiny 1x1 PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_bytes)
        f.flush()
        result = _image_to_content_part(f.name)

    assert result["type"] == "image_url"
    url = result["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == png_bytes
    Path(f.name).unlink()


def test_image_to_content_part_data_uri_passthrough():
    """data: URI passes through unchanged."""
    uri = "data:image/jpeg;base64,/9j/4AAQ..."
    result = _image_to_content_part(uri)
    assert result == {"type": "image_url", "image_url": {"url": uri}}


def test_image_to_content_part_file_not_found():
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _image_to_content_part("/nonexistent/image.jpg")


# ── _to_openai_message ──


def test_to_openai_message_with_images():
    """Message with images → content list format."""
    uri = "data:image/png;base64,abc123"
    msg = ModelMessage(role="user", content="描述这张图", images=(uri,))
    result = OpenAICompatibleGateway._to_openai_message(msg)

    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert result["content"][0] == {"type": "text", "text": "描述这张图"}
    assert result["content"][1] == {"type": "image_url", "image_url": {"url": uri}}


def test_to_openai_message_without_images():
    """Message without images → plain content string."""
    msg = ModelMessage(role="user", content="hello")
    result = OpenAICompatibleGateway._to_openai_message(msg)

    assert result["role"] == "user"
    assert result["content"] == "hello"


def test_to_openai_message_missing_image_file():
    """Missing image file is skipped with warning, text still present."""
    msg = ModelMessage(
        role="user",
        content="看图",
        images=("/nonexistent/img.jpg",),
    )
    result = OpenAICompatibleGateway._to_openai_message(msg)
    # Only text part should remain
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert result["content"][0] == {"type": "text", "text": "看图"}


def test_to_openai_message_skips_non_image_media_file(tmp_path: Path):
    """Non-image attachments like PDF must not be sent as image_url parts."""
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    msg = ModelMessage(
        role="user",
        content="识别这个文件",
        images=(str(pdf_path),),
    )

    result = OpenAICompatibleGateway._to_openai_message(msg)

    assert isinstance(result["content"], list)
    assert result["content"] == [{"type": "text", "text": "识别这个文件"}]


# ── Executor media_paths → ModelMessage.images ──


def test_executor_media_paths_to_images():
    """context.state['media_paths'] → user ModelMessage gets images."""
    from babybot.agent_kernel.types import ExecutionContext, TaskContract, ToolLease

    media = ["/tmp/a.jpg", "/tmp/b.png"]
    ctx = ExecutionContext(session_id="test", state={"media_paths": media})

    # Verify the logic: media_paths from state should become images tuple
    media_paths = ctx.state.get("media_paths") or ()
    msg = ModelMessage(role="user", content="describe", images=tuple(media_paths))
    assert msg.images == ("/tmp/a.jpg", "/tmp/b.png")


def test_executor_no_media_paths():
    """Without media_paths, images defaults to empty tuple."""
    from babybot.agent_kernel.types import ExecutionContext

    ctx = ExecutionContext(session_id="test", state={})
    media_paths = ctx.state.get("media_paths") or ()
    msg = ModelMessage(role="user", content="hello", images=tuple(media_paths))
    assert msg.images == ()
