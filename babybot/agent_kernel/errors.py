"""Error classification helpers for workflow retries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDecision:
    """Classification result for one error string."""

    error_type: str
    retryable: bool
    suggested_action: str


_RETRYABLE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary",
    "unavailable",
    "503",
    "429",
    "502",
)

_NON_RETRYABLE_PATTERNS = (
    "permission",
    "access denied",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "not found",
    "401",
    "403",
)


def classify_error(error: str | Exception) -> ErrorDecision:
    """Classify an error into retryable or non-retryable."""
    message = str(error).lower()
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern in message:
            return ErrorDecision(
                error_type="fatal",
                retryable=False,
                suggested_action="Check credentials/permissions or endpoint configuration.",
            )
    for pattern in _RETRYABLE_PATTERNS:
        if pattern in message:
            return ErrorDecision(
                error_type="retryable",
                retryable=True,
                suggested_action="Transient issue detected; retry is safe.",
            )
    return ErrorDecision(
        error_type="retryable",
        retryable=True,
        suggested_action="Unknown failure; retry once or inspect logs.",
    )


def retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff in seconds with a low cap for interactive workloads."""
    return min(2 ** max(0, attempt), 30)
