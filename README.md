# Football Prediction App

Aplicación móvil Flutter de análisis y predicción deportiva de fútbol con
backend FastAPI, provider API-Football y capa de explicación GLM-5.2.

> Regla de oro: **GLM-5.2 nunca decide ni modifica probabilidades.** Solo
> produce explicaciones a partir de un JSON estructurado calculado por los
> modelos (Elo / Poisson / Gradient Boosting). API keys viven exclusivamente
> en el backend — nunca en el bundle de Flutter.

---

## Estado — Fase 2 completada

| Fase | Estado | Contenido |
|---|---|---|
| 1 | Diseño | Arquitectura, árbol, DDL, API contracts, dependencias, flujos |
| 2 | Persistencia | Modelos SQLAlchemy 2.x async, Alembic + migración `0001_initial`, triggers de inmutabilidad, repositorios async, tests de integración |

### Cómo levantar la base y correr las validaciones

```bash
cp .env.example .env                          # completar POSTGRES_*
docker compose up -d postgres                # 1) Postgres
docker compose exec backend alembic upgrade head   # 2) migración inicial
docker compose exec backend pytest            # 3) tests (requiere Postgres)
docker compose exec backend ruff check app    # 4) lint
docker compose exec backend mypy app           # 5) typecheck
```

### Inmutabilidad de `predictions` (capas de defensa)

1. **Python**: `PredictionRepository` expone solo `insert` + lecturas. Los
   métodos `update`/`delete`/`merge` levantan `PredictionImmutableError`
   sin tocar la DB (ver `app/repositories/prediction.py`).
2. **Trigger DB**: `trg_no_update_predictions` y `trg_no_delete_predictions`
   lanzan `RAISE EXCEPTION` ante cualquier `UPDATE`/`DELETE` (incluso por
   roles con permisos elevados).
3. **Permisos**: el rol `app_user` solo recibe `SELECT, INSERT` sobre la
   tabla; migración 0001 emite `REVOKE UPDATE, DELETE`.

Lo mismo aplica a `prediction_outcomes` (insert-only) y a
`prediction_explanations`.

## Stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy async, Alembic, Pydantic v2,
  httpx, PostgreSQL 16. APScheduler para tareas programadas.
- **Frontend:** Flutter, Dart 3, Material 3, Riverpod 2, GoRouter, Dio, fl_chart.
- **Infra:** Docker Compose (`postgres`, `backend`, `worker`), `.env`.

## Estado — Fase 1: Diseño

Esta primera entrega contiene **únicamente** los artefactos de diseño:
arquitectura, árbol de carpetas, esquema de PostgreSQL, contratos de API,
dependencias y flujos de datos. No hay código de aplicación todavía.

| Documento | Contenido |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Contexto, capas backend/frontend, aislamiento de GLM, anti-leakage |
| [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md) | Árbol completo de carpetas con anotaciones (*) diseñado / (~) siguiente fase |
| [docs/SCHEMA.md](docs/SCHEMA.md) | DDL PostgreSQL 1:1 con entidades; inmutabilidad de `predictions` (permisos + trigger) |
| [docs/API_CONTRACTS.md](docs/API_CONTRACTS.md) | 8 endpoints públicos v1 + endpoints admin + formato errores/paginación |
| [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) | Backend y frontend con versiones y justificación |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | 6 flujos: sync, evaluación, stats, predicción, request, performance |

Archivos base creados:

```
docker-compose.yml        .env.example        .gitignore
backend/requirements.txt   backend/requirements-dev.txt
backend/pyproject.toml     backend/Dockerfile
frontend/pubspec.yaml      frontend/analysis_options.yaml
```

## Modelo de datos (entidades)

`Competition`, `Season`, `Team`, `Player`, `Fixture`, `TeamStatistics`,
`PlayerStatistics`, `Injury`, `Lineup`, `Prediction`, `PredictionOutcome`,
`ModelVersion` (+ auxiliares `LineupPlayer`, `PlayerTeamSeason`,
`PredictionExplanation`).

`Prediction` almacena: `fixture_id`, `model_version_id`, `created_at`,
`kickoff_time`, `home_probability`, `draw_probability`, `away_probability`,
`expected_home_goals`, `expected_away_goals`, `confidence`,
`features_snapshot (JSONB)` y es **inmutable** post-INSERT.

## Próximas fases (pendiente de tu próxima instrucción)

1. Capa dominio: SQLAlchemy models + Alembic migration inicial + DB seed.
2. Provider API-Football (httpx + tenacity + aiolimiter) + repositorios.
3. Servicios: sync, prediction (Elo, Poisson), evaluation, explanation (GLM).
4. Endpoints FastAPI con schemas Pydantic + tests unitarios/integración.
5. Frontend Flutter completo (screens, providers, repositorios, widgets).
6. Backtesting walk-forward + Gradient Boosting.

## Ejecución (cuándo haya código)

```bash
cp .env.example .env          # completar API_FOOTBALL_API_KEY, GLM_API_KEY
docker compose up --build
# backend en http://localhost:8000  | worker en background
```
