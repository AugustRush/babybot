"""Tests for babybot.builtin_tools.web (web_fetch & web_search)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest

from babybot.builtin_tools.web import (
    _extract_main_content,
    _format_search_results,
    _html_to_markdown,
    _html_to_text_fallback,
    _truncate,
    build_web_fetch_tool,
    build_web_search_tool,
    iter_web_tool_registrations,
)


class _DummyOwner:
    def __init__(self, tavily_api_key: str = "") -> None:
        self.config = SimpleNamespace(
            web=SimpleNamespace(tavily_api_key=tavily_api_key),
        )


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self) -> None:
        text = "a" * 200
        result = _truncate(text, 100)
        assert len(result) < 250
        assert "[Truncated" in result
        assert "200 chars total" in result


class TestHtmlToTextFallback:
    def test_strips_tags(self) -> None:
        html = "<p>Hello <b>world</b></p>"
        assert _html_to_text_fallback(html) == "Hello world"

    def test_collapses_whitespace(self) -> None:
        html = "line1\n\n\n\n\nline2"
        assert _html_to_text_fallback(html) == "line1\n\nline2"


class TestExtractMainContent:
    def test_prefers_article_tag(self) -> None:
        html = "<html><body><nav>nav</nav><article>main content</article></body></html>"
        result = _extract_main_content(html)
        assert "main content" in result
        assert "nav" not in result

    def test_prefers_main_tag(self) -> None:
        html = "<html><body><header>h</header><main>real stuff</main></body></html>"
        result = _extract_main_content(html)
        assert "real stuff" in result

    def test_falls_back_to_body(self) -> None:
        html = "<html><body><div>everything</div></body></html>"
        result = _extract_main_content(html)
        assert "everything" in result

    def test_returns_full_html_when_no_body(self) -> None:
        html = "<div>no body tag</div>"
        result = _extract_main_content(html)
        assert "no body tag" in result


class TestFormatSearchResults:
    def test_with_answer_and_results(self) -> None:
        data = {
            "answer": "Python is a programming language.",
            "results": [
                {
                    "title": "Python.org",
                    "url": "https://python.org",
                    "content": "Welcome to Python.",
                    "score": 0.95,
                },
            ],
        }
        output = _format_search_results(data, include_answer=True)
        assert "**Answer:**" in output
        assert "Python is a programming language" in output
        assert "### 1. Python.org" in output
        assert "https://python.org" in output
        assert "0.95" in output

    def test_without_answer(self) -> None:
        data = {
            "results": [
                {"title": "Example", "url": "https://example.com", "content": "text"},
            ],
        }
        output = _format_search_results(data, include_answer=False)
        assert "**Answer:**" not in output
        assert "### 1. Example" in output

    def test_no_results(self) -> None:
        data = {"results": []}
        output = _format_search_results(data, include_answer=True)
        assert "No results found" in output


# ---------------------------------------------------------------------------
# web_fetch tool tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(
        self,
        text: str = "",
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        return self._response

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return self._response

    async def aclose(self) -> None:
        pass


class TestWebFetch:
    def test_rejects_invalid_scheme(self) -> None:
        tool = build_web_fetch_tool(_DummyOwner())
        result = asyncio.run(tool(url="ftp://example.com"))
        assert "Invalid URL scheme" in result

    def test_rejects_missing_host(self) -> None:
        tool = build_web_fetch_tool(_DummyOwner())
        result = asyncio.run(tool(url="http://"))
        assert "missing host" in result

    def test_returns_html_format(self) -> None:
        html = "<html><body><article><p>Hello world</p></article></body></html>"
        fake_client = _FakeClient(_FakeResponse(text=html))

        tool = build_web_fetch_tool(_DummyOwner())
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            result = asyncio.run(tool(url="https://example.com", format="html"))
        assert "Hello world" in result

    def test_returns_text_format(self) -> None:
        html = "<html><body><article><p>Hello world</p></article></body></html>"
        fake_client = _FakeClient(_FakeResponse(text=html))

        tool = build_web_fetch_tool(_DummyOwner())
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            result = asyncio.run(tool(url="https://example.com", format="text"))
        assert "Hello world" in result
        # Should not contain HTML tags.
        assert "<p>" not in result

    def test_returns_markdown_format(self) -> None:
        html = "<html><body><article><h1>Title</h1><p>Paragraph</p></article></body></html>"
        fake_client = _FakeClient(_FakeResponse(text=html))

        tool = build_web_fetch_tool(_DummyOwner())
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            result = asyncio.run(tool(url="https://example.com", format="markdown"))
        # markdownify should convert <h1> to "# Title"
        assert "Title" in result
        assert "Paragraph" in result

    def test_non_html_content_returned_as_is(self) -> None:
        plain = "just plain text"
        fake_client = _FakeClient(_FakeResponse(text=plain, content_type="text/plain"))

        tool = build_web_fetch_tool(_DummyOwner())
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            result = asyncio.run(tool(url="https://example.com/data.txt"))
        assert result == plain

    def test_handles_http_error(self) -> None:
        fake_client = _FakeClient(_FakeResponse(status_code=404))

        tool = build_web_fetch_tool(_DummyOwner())
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                asyncio.run(tool(url="https://example.com/missing"))


# ---------------------------------------------------------------------------
# web_search tool tests
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_rejects_empty_query(self) -> None:
        tool = build_web_search_tool(_DummyOwner(tavily_api_key="test-key"))
        result = asyncio.run(tool(query=""))
        assert "empty" in result.lower()

    def test_rejects_missing_api_key(self) -> None:
        tool = build_web_search_tool(_DummyOwner(tavily_api_key=""))
        result = asyncio.run(tool(query="python"))
        assert "tavily_api_key" in result

    def test_successful_search(self) -> None:
        response_data = {
            "answer": "Python is great.",
            "results": [
                {
                    "title": "Python.org",
                    "url": "https://python.org",
                    "content": "Official site.",
                    "score": 0.9,
                },
            ],
        }
        fake_response = _FakeResponse(
            text=json.dumps(response_data),
            content_type="application/json",
        )
        # Make .json() work on our fake.
        fake_response.json = lambda: response_data  # type: ignore[attr-defined]
        fake_client = _FakeClient(fake_response)

        tool = build_web_search_tool(_DummyOwner(tavily_api_key="test-key"))
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            result = asyncio.run(tool(query="python programming"))
        assert "Python is great" in result
        assert "Python.org" in result

    def test_handles_search_error(self) -> None:
        fake_client = _FakeClient(_FakeResponse(status_code=500))

        tool = build_web_search_tool(_DummyOwner(tavily_api_key="test-key"))
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            with pytest.raises(RuntimeError, match="Search request failed"):
                asyncio.run(tool(query="test query"))

    def test_clamps_max_results(self) -> None:
        response_data = {"results": []}
        fake_response = _FakeResponse(
            text=json.dumps(response_data),
            content_type="application/json",
        )
        fake_response.json = lambda: response_data  # type: ignore[attr-defined]
        fake_client = _FakeClient(fake_response)

        tool = build_web_search_tool(_DummyOwner(tavily_api_key="test-key"))
        with patch(
            "babybot.builtin_tools.web._get_http_client",
            return_value=fake_client,
        ):
            # max_results should be clamped to 10
            result = asyncio.run(tool(query="test", max_results=100))
        assert "No results found" in result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_iter_web_tool_registrations(self) -> None:
        items = iter_web_tool_registrations(_DummyOwner())
        assert len(items) == 2
        names = [func.__name__ for func, group in items]
        groups = [group for func, group in items]
        assert names == ["web_fetch", "web_search"]
        assert all(g == "web" for g in groups)
