"""Feishu outbound card and markdown formatting helpers.

Extracted from feishu.py. FeishuChannel keeps thin wrapper methods/staticmethods
that delegate to these module-level functions for backward compatibility.
"""

from __future__ import annotations

import json
import re

# ── Constants (moved from FeishuChannel) ──────────────────────────────────────

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


# ── Module-level functions ────────────────────────────────────────────────────


def _detect_msg_format(text: str) -> str:
    """Determine the optimal Feishu message format for *text*.

    Returns "text", "post", or "interactive".
    """
    stripped = text.strip()
    if _COMPLEX_MD_RE.search(stripped):
        return "interactive"
    if len(stripped) > _POST_MAX_LEN:
        return "interactive"
    if _SIMPLE_MD_RE.search(stripped):
        return "interactive"
    if _LIST_RE.search(stripped) or _OLIST_RE.search(stripped):
        return "interactive"
    if _MD_LINK_RE.search(stripped):
        return "post"
    if len(stripped) <= _TEXT_MAX_LEN:
        return "text"
    return "post"


def _normalize_markdown_images(text: str) -> str:
    """Replace markdown image syntax to plain link text for Feishu compatibility."""

    def _replace(match: re.Match[str]) -> str:
        alt = (match.group(1) or "图片").strip() or "图片"
        url = (match.group(2) or "").strip()
        if url:
            return f"[{alt}]({url})"
        return alt

    return _MD_IMAGE_RE.sub(_replace, text or "")


def _markdown_to_post(text: str) -> str:
    """Convert markdown content to Feishu post message JSON."""
    lines = text.strip().split("\n")
    paragraphs: list[list[dict]] = []
    for line in lines:
        elements: list[dict] = []
        last_end = 0
        for m in _MD_LINK_RE.finditer(line):
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


def _parse_md_table(table_text: str) -> dict | None:
    """Parse a markdown table into a Feishu table element."""
    lines = [_line.strip() for _line in table_text.strip().split("\n") if _line.strip()]
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


def _split_headings(content: str) -> list[dict]:
    """Split content by headings, converting headings to div elements."""
    protected = content
    code_blocks: list[str] = []
    for m in _CODE_BLOCK_RE.finditer(content):
        code_blocks.append(m.group(1))
        protected = protected.replace(
            m.group(1), f"\x00CODE{len(code_blocks) - 1}\x00", 1
        )

    elements: list[dict] = []
    last_end = 0
    for m in _HEADING_RE.finditer(protected):
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


def _build_card_elements(content: str) -> list[dict]:
    """Split content into div/markdown + table elements for Feishu card."""
    elements: list[dict] = []
    last_end = 0
    for m in _TABLE_RE.finditer(content):
        before = content[last_end : m.start()]
        if before.strip():
            elements.extend(_split_headings(before))
        elements.append(
            _parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)}
        )
        last_end = m.end()
    remaining = content[last_end:]
    if remaining.strip():
        elements.extend(_split_headings(remaining))
    return elements or [{"tag": "markdown", "content": content}]


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


def _build_single_stream_card(content: str) -> str | None:
    """Build one interactive-card payload suitable for message patch."""
    elements = _build_card_elements(content)
    chunks = _split_elements_by_table_limit(elements)
    if len(chunks) != 1:
        return None
    card = {"config": {"wide_screen_mode": True}, "elements": chunks[0]}
    return json.dumps(card, ensure_ascii=False)
