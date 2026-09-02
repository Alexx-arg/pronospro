"""Provider registry / factory.

Selects a concrete :class:`DataProvider` implementation based on the
``DATA_PROVIDER`` env var. The default and only implemented option in
Phase 3 is ``api_football``.

A new provider only needs:

1. ``backend/app/providers/<new_name>.py`` implementing the
   :class:`DataProvider` Protocol.
2. A branch added to :func:`get_provider` (and tests).

Nothing else in the codebase needs to change: repositories, sync services
and tasks only depend on the Protocol + DTOs.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.api_football import APIFootballProvider
from app.providers.base import DataProvider
from app.providers.exceptions import ProviderError


class UnknownProviderError(ProviderError):
    """Raised when ``DATA_PROVIDER`` references a non-existent adapter."""

    code = "unknown_provider"


_cached_provider: DataProvider | None = None


def get_provider(settings: Settings | None = None) -> DataProvider:
    """Return a cached :class:`DataProvider` instance for the configured env.

    The instance is constructed from settings (env vars). The first call
    opens the underlying HTTP client lazily; subsequent calls reuse the
    same provider instance for the lifetime of the worker.

    Warning:
        The cached instance is global; tests must call
        :func:`reset_provider_cache` to swap mocks.
    """
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider
    s = settings or get_settings()
    name = s.data_provider.strip().lower()
    if name == "api_football":
        if not s.api_football_key:
            raise UnknownProviderError(
                "DATA_PROVIDER=api_football but API_FOOTBALL_KEY is not set "
                "in the environment. The provider cannot authenticate without "
                "the key."
            )
        _cached_provider = APIFootballProvider(
            base_url=s.api_football_base_url,
            api_key=s.api_football_key,
            timeout_seconds=s.api_football_timeout_seconds,
            max_retries=s.api_football_max_retries,
            rate_per_minute=s.api_football_rate_per_minute,
            max_connections=s.api_football_max_connections,
            max_keepalive_connections=s.api_football_max_keepalive_connections,
            user_agent=s.api_football_user_agent,
        )
        return _cached_provider
    if name == "bzzoiro":
        if not s.bzzoiro_api_key:
            raise UnknownProviderError(
                "DATA_PROVIDER=bzzoiro but BZZOIRO_API_KEY is not set "
                "in the environment. The provider cannot authenticate without "
                "the key."
            )
        from app.providers.bzzoiro import BzzoiroProvider

        _cached_provider = BzzoiroProvider(
            base_url=s.bzzoiro_base_url,
            api_key=s.bzzoiro_api_key,
            timeout_seconds=s.api_football_timeout_seconds,
        )
        return _cached_provider
    raise UnknownProviderError(
        f"Unknown DATA_PROVIDER={name!r}. "
        "Implement the adapter under backend/app/providers/<name>.py and "
        "register it in get_provider()."
    )


def reset_provider_cache() -> None:
    """Clear the cached provider instance (useful for tests)."""
    global _cached_provider
    _cached_provider = None


__all__ = ["UnknownProviderError", "get_provider", "reset_provider_cache"]
