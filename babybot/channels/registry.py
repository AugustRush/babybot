"""Auto-discover BaseChannel subclasses in the channels package."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseChannel

_SKIP_MODULES = {"base", "manager", "registry", "__init__"}
logger = logging.getLogger(__name__)


def discover_channels() -> dict[str, type[BaseChannel]]:
    """Scan ``babybot.channels`` and return ``{name: ChannelClass}``."""
    from .base import BaseChannel as _Base

    package = importlib.import_module("babybot.channels")
    result: dict[str, type[BaseChannel]] = {}

    for info in pkgutil.iter_modules(package.__path__):
        if info.name in _SKIP_MODULES or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"babybot.channels.{info.name}")
        except Exception as exc:
            logger.warning(
                "Failed to import channel module %s: %s",
                info.name,
                exc,
            )
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, _Base)
                and obj is not _Base
                and getattr(obj, "name", "")
            ):
                result[obj.name] = obj

    return result
