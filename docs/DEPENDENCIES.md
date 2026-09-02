# Dependencias

## Backend — Python 3.12

### Runtime (`backend/requirements.txt`)

| Paquete | versión | Por qué |
|---|---|---|
| `fastapi` | 0.115.0 | Framework HTTP async |
| `uvicorn[standard]` | 0.30.6 | ASGI server |
| `gunicorn` | 22.0.0 | producción (solo si se sirve en Linux) |
| `sqlalchemy[asyncio]` | 2.0.35 | ORM async |
| `asyncpg` | 0.29.0 | driver async PostgreSQL |
| `alembic` | 1.13.3 | migraciones |
| `pydantic` | 2.9.2 | validación de DTOs |
| `pydantic-settings` | 2.5.2 | `.env` -> Settings |
| `httpx` | 0.27.2 | cliente HTTP async (provider + GLM) |
| `tenacity` | 9.0.0 | retries |
| `aiolimiter` | 1.1.0 | rate limiting async |
| `apscheduler` | 3.10.4 | tareas programadas (`worker`) |
| `loguru` | 0.7.2 | logging estructurado |
| `orjson` | 3.10.7 | serialización JSON rápida |
| `python-dotenv` | 1.0.1 | carga `.env` |
| `scikit-learn` | 1.5.2 | Gradient Boosting + métricas (brier, log_loss) |
| `pandas` | 2.2.3 | backtesting / agregaciones de métricas |
| `numpy` | 2.1.1 | álgebra |
| `statsmodels` | 0.14.2 | Poisson y odds ratios |
| `openai` | 1.50.2 | cliente OpenAI-compatible para GLM-5.2 |

`openai` se usa apuntando a un endpoint OpenAI-compatible (GLM-5.2 por
Zhipu/NVIDIA puede servirse así). Alternativamente `zhipuai` — decisión
postergada a la fase de implementación de GLM.

### Dev / quality (en `requirements-dev.txt`)

| Paquete | versión | Uso |
|---|---|---|
| `pytest` | 8.3.3 | runner |
| `pytest-asyncio` | 0.24.0 | async tests |
| `pytest-cov` | 5.0.0 | cobertura |
| `respx` | 0.21.1 | mock httpx (provider y GLM) |
| `factory-boy` | 3.3.1 | factories de entidades |
| `freezegun` | 1.5.1 | control de tiempo (evaluaciones, cutoffs) |
| `testcontainers` | 4.8.1 | Postgres real para tests de integración |
| `ruff` | 0.6.8 | linter + formatter |
| `mypy` | 1.11.2 | type checker (`--strict`) |
| `types-requests` / `types-python-dateutil` | latest | stubs |

### Configuración de herramientas (ver `pyproject.toml`)

- `ruff`: line-length 100, target py312, reglas E/F/I/UP/B/SIM/C4/N.
- `mypy`: `strict=true`, plugin `pydantic.mypy`.
- `pytest`: `asyncio_mode=auto`, marker `integration` para tests con DB.

---

## Frontend — Flutter / Dart

### `frontend/pubspec.yaml`

| Paquete | versión | Por qué |
|---|---|---|
| `flutter_riverpod` | ^2.5.1 | state management |
| `riverpod_annotation` | ^2.3.5 | anotaciones codegen |
| `go_router` | ^14.2.0 | navegación declarativa |
| `dio` | ^5.4.3+1 | HTTP client |
| `freezed_annotation` | ^2.4.4 | modelos inmutables |
| `json_annotation` | ^4.9.0 | serialización |
| `cached_network_image` | ^3.3.1 | logos de equipos |
| `fl_chart` | ^0.69.0 | gráficos de performance / calibration |
| `intl` | ^0.19.0 | fechas y formatos |
| `flutter_dotenv` | ^5.1.0 | carga `.env` (solo URL backend!) |
| `shimmer` | ^3.0.0 | loading placeholders |
| `footLoading copyright` | builtin | |

dev_dependencies:

| Paquete | versión |
|---|---|
| `flutter_test` | sdk |
| `flutter_lints` | ^4.0.0 |
| `build_runner` | ^2.4.11 |
| `freezed` | ^2.5.7 |
| `json_serializable` | ^6.8.0 |
| `riverpod_generator` | ^2.4.2 |
| `custom_lint` | ^0.6.4 |
| `riverpod_lint` | ^2.3.12 |

### Privacidad de claves

`flutter_dotenv` recibe UNICAMENTE `BACKEND_BASE_URL` y `ENV`. Jamás se
incluirá una API key de API-Football o de GLM en el bundle de la app.

### Dart SDK

- `environment.sdk: ">=3.4.0 <4.0.0"` (Dart 3, records, patterns).
- `flutter`: >= 3.22 (Material 3 default).

---

## Infraestructura

| Imagen | versión | uso |
|---|---|---|
| `postgres` | 16-alpine | DB |
| `python` | 3.12-slim | base backend (FastAPI + worker comparten image) |

Imágenes adicionales:
- `redis` (futuro, solo si pasamos a cola `arq`/`celery`). No en v1.
- No se construye imagen Docker del frontend (es Android, build con Gradle).
