from __future__ import annotations

import datetime
from typing import Any


_SUPPORTED_FORMATS = (
    "default",
    "iso",
    "date",
    "time",
    "datetime",
    "timestamp",
    "timestamp_ms",
)


def _now_local() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


def _format_utc_offset(value: datetime.datetime) -> str:
    offset = value.utcoffset() or datetime.timedelta()
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def build_get_current_time_tool(owner: Any) -> Any:
    def get_current_time(format: str = "default") -> str:
        """Return the current local time in a common format."""
        normalized = str(format or "default").strip().lower()
        if normalized not in _SUPPORTED_FORMATS:
            supported = ", ".join(_SUPPORTED_FORMATS)
            return (
                f"Unsupported format '{normalized}'. "
                f"Supported formats: {supported}."
            )

        current = _now_local()
        if normalized == "iso":
            return current.isoformat(timespec="seconds")
        if normalized == "date":
            return current.strftime("%Y-%m-%d")
        if normalized == "time":
            return current.strftime("%H:%M:%S")
        if normalized == "datetime":
            return current.strftime("%Y-%m-%d %H:%M:%S")
        if normalized == "timestamp":
            return str(int(current.timestamp()))
        if normalized == "timestamp_ms":
            return str(int(current.timestamp() * 1000))

        tz_name = current.tzname() or "local"
        utc_offset = _format_utc_offset(current)
        return f"{current.strftime('%Y-%m-%d %H:%M:%S')} {tz_name} ({utc_offset})"

    return get_current_time


def iter_time_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return ((build_get_current_time_tool(owner), "basic"),)
