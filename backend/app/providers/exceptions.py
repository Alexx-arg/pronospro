"""Provider-side exception hierarchy.

Errors raised BY the data provider layer are defined HERE so that sync
services can distinguish them from generic Python exceptions. The hierarchy
is intentionally narrow:

* :class:`ProviderError`  — root
    * :class:`ProviderUnavailable`     — 5xx / network / DNS / no response
    * :class:`ProviderAuthError`       — 401/403, bad/missing key
    * :class:`ProviderRateLimitError`  — 429, retry-after-aware
    * :class:`ProviderTimeoutError`    — read/connect timeout
    * :class:`InvalidProviderResponse` — schema/format violation

CRITICAL: error messages MUST NOT contain the API key, header values, or
the raw URL with credentials inlined. The HTTP client strips them before
constructing these exceptions.
"""

from __future__ import annotations

from typing import Any


class ProviderError(Exception):
    """Base class for all data-provider errors."""

    #: Stable machine-readable code (forwarded to API clients by the
    #: ``app.api`` layer in later phases). Never changes within a major
    #: version.
    code: str = "provider_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ProviderUnavailable(ProviderError):
    """The provider returned 5xx or was unreachable on the network."""

    code = "provider_unavailable"


class ProviderAuthError(ProviderError):
    """Authentication / authorisation failed (HTTP 401 / 403).

    This is NON-retryable at the sync-service level: there is no point in
    retrying a sync with the same key.
    """

    code = "provider_auth_error"


class ProviderRateLimitError(ProviderError):
    """The provider returned 429 (or equivalent rate-limit signal).

    The HTTP client may already have retried internally (with the
    ``Retry-After`` header if present). If this bubbles up it means all
    retries were exhausted.
    """

    code = "provider_rate_limit"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeoutError(ProviderError):
    """A network read/connect timeout occurred (after internal retries)."""

    code = "provider_timeout"


class InvalidProviderResponse(ProviderError):
    """The provider returned an HTTP success but the body could not be
    parsed / validated into the expected DTO.

    Includes the raw payload fragment under ``details["response_fragment"]``
    (truncated and stripped of credentials) to aid debugging without
    leaking secrets.
    """

    code = "invalid_provider_response"
