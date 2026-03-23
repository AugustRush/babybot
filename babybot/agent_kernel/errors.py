"""Error classification helpers for workflow retries."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDecision:
    """Classification result for one error string."""

    error_type: str
    retryable: bool
    suggested_action: str


# HTTP status code patterns are checked first for unambiguous classification.
_HTTP_STATUS_RETRYABLE = re.compile(r"\b(429|502|503)\b")
_HTTP_STATUS_FATAL = re.compile(r"\b(401|403|404)\b")

_RETRYABLE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary",
    "transient",
    "unavailable",
)

_NON_RETRYABLE_PATTERNS = (
    "permission",
    "access denied",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "not found",
)


def classify_error(error: str | Exception) -> ErrorDecision:
    """Classify an error into retryable or non-retryable.

    HTTP status codes take priority over substring pattern matching to
    avoid ambiguity (e.g. "503 … access denied" should be retryable).
    """
    message = str(error).lower()

    # 1. HTTP status codes — highest priority, unambiguous.
    if _HTTP_STATUS_RETRYABLE.search(message):
        return ErrorDecision(
            error_type="retryable",
            retryable=True,
            suggested_action="Transient HTTP error; retry is safe.",
        )
    if _HTTP_STATUS_FATAL.search(message):
        return ErrorDecision(
            error_type="fatal",
            retryable=False,
            suggested_action="Check credentials/permissions or endpoint configuration.",
        )

    # 2. Keyword patterns — retryable checked first since false-negative
    #    (not retrying a retryable error) is worse than false-positive.
    for pattern in _RETRYABLE_PATTERNS:
        if pattern in message:
            return ErrorDecision(
                error_type="retryable",
                retryable=True,
                suggested_action="Transient issue detected; retry is safe.",
            )
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern in message:
            return ErrorDecision(
                error_type="fatal",
                retryable=False,
                suggested_action="Check credentials/permissions or endpoint configuration.",
            )

    return ErrorDecision(
        error_type="unknown",
        retryable=False,
        suggested_action="Unknown failure; inspect logs before retrying.",
    )


def retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped for interactive workloads."""
    base = min(2 ** max(0, attempt), 30)
    return base * (0.5 + random.random() * 0.5)
