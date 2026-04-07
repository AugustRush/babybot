"""Path-detection and artifact-collection helpers.

Used by ResourceManager and CallableTool to identify and normalise file
paths embedded in tool return values.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .resource import ResourceManager


def _looks_like_path_candidate(candidate: str) -> bool:
    text = candidate.strip()
    if not text:
        return False
    if "\n" in text or "\r" in text:
        return False
    if len(text) > 240:
        return False
    if text.startswith("{") or text.startswith("["):
        return False
    if "://" in text:
        return False
    suffix = Path(text).suffix.lower()
    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".pdf",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".xlsx",
        ".pptx",
        ".docx",
        ".mp4",
        ".mp3",
        ".wav",
    }:
        return True
    return "/" in text or "\\" in text or text.startswith(".") or text.startswith("~")


def _normalize_artifact_path_for_manager(
    resource_manager: "ResourceManager | None",
    resolved: Path,
) -> Path:
    if resource_manager is None:
        return resolved
    try:
        workspace = resource_manager.config.workspace_dir.resolve()
    except Exception:
        return resolved
    try:
        resolved.relative_to(workspace)
        return resolved
    except ValueError:
        pass

    output_dir = resource_manager._get_output_dir()
    target = output_dir / resolved.name
    if target == resolved:
        return resolved
    stem = resolved.stem
    suffix = resolved.suffix
    counter = 1
    while target.exists():
        try:
            if target.samefile(resolved):
                return target.resolve()
        except OSError:
            pass
        target = output_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(resolved, target)
    return target.resolve()


def _collect_artifact_paths(
    value: Any,
    *,
    base_dir: Path,
    normalize_path: Callable[[Path], Path] | None = None,
    media_path_re: re.Pattern[str] | None = None,
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    source_seen: set[str] = set()
    if media_path_re is None:
        from .resource import (
            ResourceManager,
        )  # lazy import to avoid circular dependency

        media_path_re = ResourceManager._MEDIA_PATH_RE

    def _add_path(raw: str) -> None:
        candidate = raw.strip().strip("\"'`")
        if not candidate:
            return
        path = Path(os.path.expanduser(candidate))
        if not path.is_absolute():
            path = base_dir / path
        try:
            resolved = path.resolve()
        except OSError:
            return
        try:
            if not resolved.is_file():
                return
        except OSError:
            return
        source_key = str(resolved)
        if source_key in source_seen:
            return
        source_seen.add(source_key)
        if normalize_path is not None:
            resolved = normalize_path(resolved)
        resolved_str = str(resolved)
        if resolved_str not in seen:
            seen.add(resolved_str)
            found.append(resolved_str)

    def _walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, os.PathLike):
            _add_path(os.fspath(item))
            return
        if isinstance(item, str):
            if _looks_like_path_candidate(item):
                _add_path(item)
            for match in media_path_re.finditer(item):
                _add_path(match.group(1))
            return
        if isinstance(item, dict):
            for nested in item.values():
                _walk(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                _walk(nested)

    _walk(value)
    return found
