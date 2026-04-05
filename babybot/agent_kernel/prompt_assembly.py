"""Shared helpers for composing prompt sections consistently."""

from __future__ import annotations

from collections.abc import Iterable

from .types import SystemPromptBuilder


def dedupe_prompt_items(
    items: Iterable[object],
    *,
    limit: int | None = None,
) -> list[str]:
    """Normalize prompt items into a trimmed, unique, ordered list."""
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if limit is not None and limit > 0 and len(normalized) >= limit:
            break
    return normalized


def add_text_section(
    builder: SystemPromptBuilder,
    name: str,
    content: object,
    *,
    priority: int = 50,
    header: str = "",
    cacheable: bool = False,
) -> SystemPromptBuilder:
    """Add a text section with optional header prefix."""
    text = str(content or "").strip()
    if not text:
        return builder
    prefix = str(header or "")
    body = f"{prefix}{text}" if prefix else text
    return builder.add(name, body, priority=priority, cacheable=cacheable)


def add_list_section(
    builder: SystemPromptBuilder,
    name: str,
    items: Iterable[object],
    *,
    priority: int = 50,
    header: str = "",
    bullet: str = "- ",
    cacheable: bool = False,
    limit: int | None = None,
) -> SystemPromptBuilder:
    """Add a bullet-list section after trimming, deduping, and limiting items."""
    normalized = dedupe_prompt_items(items, limit=limit)
    if not normalized:
        return builder
    lines = "\n".join(f"{bullet}{item}" for item in normalized)
    return add_text_section(
        builder,
        name,
        lines,
        priority=priority,
        header=header,
        cacheable=cacheable,
    )
