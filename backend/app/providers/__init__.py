"""Provider package — pure isolated data access layer.

The ``providers`` package is the **only** place in the codebase that knows
about API-Football (or any other external data source). The rest of the
backend consumes :class:`DataProvider` instances and the DTOs defined in
:mod:`app.providers.dto`.

Public surface (stable across phases):

* :class:`app.providers.base.DataProvider`  — Protocol
* :class:`app.providers.registry.get_provider` — factory by env var
* :data:`app.providers.dto` — normalised Pydantic DTOs
* :class:`app.providers.exceptions.*` — provider-side errors

Hard guarantees:
* No API key ever leaves this package (logs, error messages, DTO fields).
* No HTTP response object ever leaves this package either: the adapter
  ALWAYS returns normalised DTOs or raises a typed exception.
"""
