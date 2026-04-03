"""Built-in web tools: URL fetching and web search."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0
_MAX_CONTENT_LENGTH = 50_000  # characters, not bytes
_USER_AGENT = "Mozilla/5.0 (compatible; BabyBot/1.0; +https://github.com/babybot)"

# Simple patterns for stripping HTML to readable text when markdownify
# is not installed.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _html_to_text_fallback(html: str) -> str:
    """Best-effort HTML-to-text without external dependencies."""
    text = _TAG_RE.sub("", html)
    text = _WS_RE.sub("\n\n", text)
    return text.strip()


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown, falling back to plain text stripping."""
    try:
        from markdownify import markdownify  # type: ignore[import-untyped]

        return markdownify(html, heading_style="ATX", strip=["img", "script", "style"])
    except ImportError:
        pass
    try:
        import html2text  # type: ignore[import-untyped]

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html)
    except ImportError:
        pass
    return _html_to_text_fallback(html)


def _extract_main_content(html: str) -> str:
    """Try to extract the main content block from a full HTML page.

    Falls back to the full ``<body>`` or the entire document.
    """
    try:
        from readability import Document  # type: ignore[import-untyped]

        doc = Document(html)
        return doc.summary()
    except ImportError:
        pass

    # Lightweight heuristic: prefer <article> or <main> tags.
    for tag in ("article", "main"):
        pattern = re.compile(rf"<{tag}[^>]*>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        m = pattern.search(html)
        if m:
            return m.group(1)

    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1)
    return html


def _truncate(text: str, max_chars: int = _MAX_CONTENT_LENGTH) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[Truncated — {len(text)} chars total]"


async def _get_http_client() -> Any:
    """Lazy-import and return an httpx.AsyncClient."""
    import httpx

    return httpx.AsyncClient(
        timeout=httpx.Timeout(_DEFAULT_TIMEOUT, connect=10),
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------


def build_web_fetch_tool(owner: Any) -> Any:
    """Build the ``web_fetch`` tool function."""

    async def web_fetch(
        url: str,
        format: Literal["markdown", "text", "html"] = "markdown",
        extract_main: bool = True,
    ) -> str:
        """Fetch a web page by URL and return its content.

        Args:
            url: The URL to fetch. Must be a valid HTTP(S) URL.
            format: Output format — "markdown" (default), "text", or "html".
            extract_main: If True, try to extract the main content block
                (removing nav, footer, ads) before converting.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Invalid URL scheme '{parsed.scheme}'. Only http and https are supported."
        if not parsed.netloc:
            return "Invalid URL: missing host."

        client = await _get_http_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as exc:
            return f"Failed to fetch {url}: {exc}"
        finally:
            await client.aclose()

        content_type = response.headers.get("content-type", "")
        raw = response.text

        # For non-HTML content, return as-is.
        if "html" not in content_type and format != "html":
            return _truncate(raw)

        html = raw
        if extract_main:
            html = _extract_main_content(html)

        if format == "html":
            return _truncate(html)
        if format == "text":
            return _truncate(_html_to_text_fallback(html))
        # Default: markdown
        return _truncate(_html_to_markdown(html))

    return web_fetch


# ---------------------------------------------------------------------------
# web_search  (Tavily API)
# ---------------------------------------------------------------------------

_TAVILY_API_URL = "https://api.tavily.com/search"


def _get_tavily_api_key(owner: Any) -> str:
    """Resolve Tavily API key from config (preferred) or environment."""
    config = getattr(owner, "config", None)
    web_conf = getattr(config, "web", None)
    key = getattr(web_conf, "tavily_api_key", "") or ""
    return key


def build_web_search_tool(owner: Any) -> Any:
    """Build the ``web_search`` tool function (Tavily backend)."""

    async def web_search(
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_answer: bool = True,
        topic: Literal["general", "news"] = "general",
    ) -> str:
        """Search the web using the Tavily search API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (1-10).
            search_depth: "basic" for fast results, "advanced" for deeper research.
            include_answer: If True, include an AI-generated answer summary.
            topic: Search topic — "general" or "news".
        """
        api_key = _get_tavily_api_key(owner)
        if not api_key:
            return (
                "Tavily API key is not configured. "
                'Set tavily_api_key in the "web" section of config.json.'
            )

        query = query.strip()
        if not query:
            return "Search query cannot be empty."

        max_results = max(1, min(int(max_results), 10))

        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "topic": topic,
        }

        client = await _get_http_client()
        try:
            response = await client.post(
                _TAVILY_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            response.raise_for_status()
        except Exception as exc:
            return f"Search request failed: {exc}"
        finally:
            await client.aclose()

        try:
            data = response.json()
        except Exception:
            return "Failed to parse search response."

        return _format_search_results(data, include_answer)

    return web_search


def _format_search_results(data: dict[str, Any], include_answer: bool) -> str:
    """Format Tavily API response into readable output."""
    parts: list[str] = []

    # AI-generated answer
    answer = data.get("answer")
    if include_answer and answer:
        parts.append(f"**Answer:** {answer}")
        parts.append("")

    # Individual results
    results = data.get("results", [])
    if not results:
        parts.append("No results found.")
        return "\n".join(parts)

    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("score")

        parts.append(f"### {i}. {title}")
        if url:
            parts.append(f"URL: {url}")
        if content:
            parts.append(content)
        if score is not None:
            parts.append(f"(relevance: {score:.2f})")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def iter_web_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    """Return ``(callable, group_name)`` pairs for web tools."""
    return (
        (build_web_fetch_tool(owner), "web"),
        (build_web_search_tool(owner), "web"),
    )
