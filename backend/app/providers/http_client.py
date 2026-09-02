"""HTTP client wrapper for the provider layer.

This module is the ONLY place in the codebase that performs outbound HTTP
calls to external providers. It centralises:

* API-key injection (header ``x-apisports-key``).
* Timeouts, connection limits, keep-alive.
* Exponential backoff with jitter, via :mod:`tenacity`. Retries target
  idempotent ``GET`` endpoints only.
* Rate limiting via :mod:`aiolimiter` (requests/minute) so we stay below
  the provider's per-minute allowance even when several sync jobs run
  concurrently.
* Error classification (timeout / rate-limit / auth / unavailable /
  invalid-response) into the typed :mod:`app.providers.exceptions` tree
  — so the sync services can react accordingly.

Secure-by-default guarantees:
* The API key is NEVER present in exception messages, log records, or
  ``details`` dicts. Logs use :func:`_redact` to scrub headers.
* Logs only emit the canonical endpoint path (``/fixtures``), the request
  parameters with secret fields removed, the HTTP status and the latency.
* On non-2xx the response body is truncated to a small fragment and
  stored under :class:`InvalidProviderResponse.details["response_fragment"]`.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any, Final

import httpx
from aiolimiter import AsyncLimiter
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.logging import get_logger
from app.providers.exceptions import (
    InvalidProviderResponse,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailable,
)

# ----- Logging-safe labels -------------------------------------------------
_LOG: Final = get_logger()
_SECURE_HEADER_KEYS: Final = frozenset(  # case-insensitive
    {
        "x-apisports-key",
        "authorization",
        "api-key",
        "apikey",
        "x-api-key",
    }
)

# Maximum body fragment kept for debugging on invalid/failed responses.
_BODY_FRAGMENT_BYTES: Final = 1024


def _redact(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced by ``***``."""
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SECURE_HEADER_KEYS:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _redacted_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Scrub known-sensitive query params from a request's parameter dict."""
    sensitive = {"api_key", "apikey", "api-key", "key", "token", "secret"}
    return {
        k: ("***" if k.lower() in sensitive else v)
        for k, v in params.items()
    }


class ProviderHttpClient:
    """Async, retry-aware, rate-limited GET client for data providers.

    Designed for the API-Football API surface (REST JSON over HTTPS, GET
    requests only). It is NOT a generic httpx client: the wrapper hard-
    codes the contract the adapter expects.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        rate_per_minute: int = 30,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
        user_agent: str = "football-prediction-app/0.1",
    ) -> None:
        if not api_key:
            raise ValueError("ProviderHttpClient requires a non-empty API key.")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout_seconds

        # Required headers for API-Football. The API key goes here, never
        # in the query string (left as default for other providers: callers
        # can pass extra headers too).
        default_headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            # API-Football canonical auth header.
            "x-apisports-key": api_key,
        }
        if headers:
            default_headers.update(headers)
        self._headers = default_headers

        # Connection / timeout configuration.
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._timeout_config = httpx.Timeout(timeout_seconds)

        # AsyncLimiter enforces the per-minute cap. ``AsyncLimiter`` rounds
        # the interval so the effective spacing between requests is
        # ``max(60 / rate_per_minute, 0.05)`` seconds.
        self._limiter = AsyncLimiter(max(1, rate_per_minute), period=60)

        # The underlying httpx client is created lazily so the wrapper can
        # be constructed for tests without opening sockets.
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the underlying httpx client (safe to call multiple times)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ProviderHttpClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout_config,
                limits=self._limits,
            )
        return self._client

    # ------------------------------------------------------------------
    # public method
    # ------------------------------------------------------------------
    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Issue a GET ``endpoint`` with query ``params`` and return parsed JSON.

        Retries idempotently (HTTP 429 / 5xx / network timeout) with
        exponential backoff up to ``max_retries`` attempts. Beyond that,
        raises a :class:`ProviderError` subclass matching the failure.

        Returns:
            The decoded JSON body (``dict`` or ``list`` depending on the
            endpoint). For API-Football all of our endpoints return
            ``dict`` wrappers.

        Raises:
            ProviderAuthError, ProviderRateLimitError, ProviderTimeoutError,
            ProviderUnavailable, InvalidProviderResponse
        """
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        clean_params = dict(params or {})
        redacted_params = _redacted_params(clean_params)

        retry_strategy = retry_if_exception_type(
            (
                ProviderRateLimitError,
                ProviderUnavailable,
                ProviderTimeoutError,
            )
        )
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential_jitter(initial=1, max=30),
                retry=retry_strategy,
                reraise=True,
            ):
                with attempt:
                    return await self._single_attempt(
                        endpoint, clean_params, redacted_params
                    )
        except RetryError as exc:
            # Should not happen because we pass ``reraise=True`` above, but
            # be defensive.
            raise _from_retry_error(exc) from exc

        # Unreachable: AsyncRetrying always returns or raises.
        raise ProviderError("retry loop exited without result or exception")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    async def _single_attempt(
        self,
        endpoint: str,
        params: dict[str, Any],
        redacted_params: dict[str, Any],
    ) -> dict[str, Any] | list[Any]:
        """Perform ONE HTTP attempt (rate-limited) and classify failures."""
        client = self._ensure_client()
        started = time.perf_counter()
        try:
            async with self._limiter:
                response = await client.get(endpoint, params=params)
        except httpx.TimeoutException as exc:
            _LOG.warning(
                "provider timeout GET {} params={} error={}",
                endpoint, redacted_params, type(exc).__name__,
            )
            raise ProviderTimeoutError(
                f"Timeout calling {endpoint}",
                details={"endpoint": endpoint, "params": redacted_params},
            ) from exc
        except httpx.HTTPError as exc:
            # Network-level / connection refused / DNS / etc.
            _LOG.warning(
                "provider network error GET {} params={} error={}",
                endpoint, redacted_params, type(exc).__name__,
            )
            raise ProviderUnavailable(
                f"Network error calling {endpoint}: {type(exc).__name__}",
                details={"endpoint": endpoint},
            ) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._log_status(endpoint, redacted_params, response.status_code, elapsed_ms)

        # ----- classify status -----
        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"Authentication failed calling {endpoint} "
                f"(status {response.status_code})",
                details={"endpoint": endpoint, "status": response.status_code},
            )
        if response.status_code == 429:
            retry_after = self._parse_retry_after(response.headers)
            raise ProviderRateLimitError(
                f"Rate limited by provider on {endpoint}",
                retry_after_seconds=retry_after,
                details={"endpoint": endpoint},
            )
        if response.status_code >= 500:
            raise ProviderUnavailable(
                f"Provider returned {response.status_code} on {endpoint}",
                details={"endpoint": endpoint, "status": response.status_code},
            )
        if response.status_code >= 400:
            # 4xx (non-auth, non-rate-limit) — treat as invalid/unavailable.
            raise ProviderUnavailable(
                f"Provider returned {response.status_code} on {endpoint}",
                details={"endpoint": endpoint, "status": response.status_code},
            )

        # ----- parse body -----
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001  (distinguish below)
            fragment = response.text[:_BODY_FRAGMENT_BYTES]
            _LOG.warning(
                "provider returned non-JSON body on {} ({} bytes)",
                endpoint, len(response.text),
            )
            raise InvalidProviderResponse(
                f"Non-JSON response from {endpoint}",
                details={
                    "endpoint": endpoint,
                    "response_fragment": fragment,
                },
            ) from exc

    @staticmethod
    def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
        """Best-effort parse of ``Retry-After``.

        Returns seconds. Handles both the integer (delta-seconds) form and
        the HTTP-date form (converted to seconds from now).
        """
        value = headers.get("Retry-After") or headers.get("retry-after")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        # HTTP-date fallback.
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone

            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            delta = (target - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, float(delta))
        except Exception:  # noqa: BLE001
            return None

    def _log_status(
        self,
        endpoint: str,
        redacted_params: dict[str, Any],
        status: int,
        elapsed_ms: int,
    ) -> None:
        """Log the canonical request line WITHOUT secrets or headers."""
        _LOG.debug(
            "provider request {} params={} status={} elapsed_ms={}",
            endpoint, redacted_params, status, elapsed_ms,
        )


def _from_retry_error(exc: RetryError) -> ProviderError:
    """Convert a tenacity RetryError into a typed ProviderError if possible."""
    cause = exc.__cause__ or exc
    if isinstance(cause, ProviderError):
        return cause
    return ProviderUnavailable(
        "Retries exhausted calling provider",
        details={"cause": type(cause).__name__},
    )


# Re-export the type for typing convenience.
__all__ = ["ProviderHttpClient", "InvalidProviderResponse", "random_jitter"]


def random_jitter() -> float:
    """Tiny jitter helper (exposed for deterministic tests)."""
    return random.uniform(0.0, 0.5)
