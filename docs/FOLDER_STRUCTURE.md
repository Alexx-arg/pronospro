# Árbol de carpetas

Convención: `(*)` = archivo a crear en esta fase de diseño.
`(~)` = archivo/contenido de implementación que llegará en fases posteriores.

```
football-prediction-app/
|-- README.md                                    (*)
|-- docker-compose.yml                           (*)
|-- .env.example                                 (*)
|-- .gitignore                                   (*)
|-- docs/
|   |-- ARCHITECTURE.md                          (*)
|   |-- FOLDER_STRUCTURE.md                      (*)
|   |-- SCHEMA.md                                (*)
|   |-- API_CONTRACTS.md                         (*)
|   |-- DEPENDENCIES.md                          (*)
|   `-- DATA_FLOW.md                             (*)
|
|-- backend/
|   |-- Dockerfile                               (*)
|   |-- requirements.txt                         (*)
|   |-- pyproject.toml                           (*  ruff/mypy/pytest config)
|   |-- alembic.ini                              (~)
|   |-- alembic/
|   |   |-- env.py                               (~)
|   |   |-- script.py.mako                       (~)
|   |   `-- versions/                            (~)
|   `-- app/
|       |-- __init__.py                          (~)
|       |-- main.py                              (~  FastAPI app factory)
|       |-- config.py                            (~  Settings via pydantic-settings)
|       |-- core/
|       |   |-- logging.py                       (~  loguru setup)
|       |   |-- exceptions.py                    (~  domain + provider errors)
|       |   |-- rate_limit.py                     (~  aiolimiter helpers)
|       |   `-- retries.py                        (~  tenacity policies)
|       |-- db/
|       |   |-- base.py                          (~  DeclarativeBase)
|       |   `-- session.py                       (~  async engine/session)
|       |-- models/                              (~  SQLAlchemy ORM, 1:1 con SCHEMA)
|       |   |-- competition.py
|       |   |-- season.py
|       |   |-- team.py
|       |   |-- player.py
|       |   |-- fixture.py
|       |   |-- team_statistics.py
|       |   |-- player_statistics.py
|       |   |-- injury.py
|       |   |-- lineup.py
|       |   |-- prediction.py
|       |   |-- prediction_outcome.py
|       |   `-- model_version.py
|       |-- schemas/                             (~  Pydantic v2 DTOs)
|       |   |-- fixture.py
|       |   |-- team.py
|       |   |-- prediction.py
|       |   `-- performance.py
|       |-- api/
|       |   `-- v1/
|       |       |-- router.py                    (~  agrega routers)
|       |       |-- fixtures.py                  (~ upcoming + fixture detail)
|       |       |-- teams.py                     (~ team + stats)
|       |       |-- predictions.py               (~ list + detail)
|       |       |-- performance.py               (~ metrics)
|       |       `-- models.py                    (~ model versions)
|       |-- providers/
|       |   |-- base.py                          (~  DataProvider Protocol)
|       |   |-- api_football.py                  (~  adapter httpx)
|       |   `-- registry.py                      (~  factory segun .env)
|       |-- repositories/                         (~  SQLAlchemy async repos)
|       |   |-- competition.py
|       |   |-- season.py
|       |   |-- team.py
|       |   |-- player.py
|       |   |-- fixture.py
|       |   |-- statistics.py
|       |   |-- injury.py
|       |   |-- lineup.py
|       |   `-- prediction.py
|       |-- services/
|       |   |-- sync/
|       |   |   |-- fixtures.py                  (~  upcoming + finished)
|       |   |   |-- statistics.py                (~  team/player stats)
|       |   |   `-- injuries_lineups.py          (~  injuries + lineups)
|       |   |-- prediction/
|       |   |   |-- features.py                  (~  snapshot builder)
|       |   |   |-- elo.py
|       |   |   |-- poisson.py
|       |   |   |-- gradient_boosting.py
|       |   |   `-- orchestrator.py              (~  coordina modelo+GLM+repo)
|       |   |-- evaluation/
|       |   |   |-- outcomes.py                  (~  crea PredictionOutcome)
|       |   |   `-- metrics.py                   (~  accuracy/brier/logloss/calib)
|       |   `-- explanation/
|       |       `-- glm.py                       (~  cliente GLM-5.2 aislado)
|       |-- tasks/
|       |   |-- scheduler.py                     (~  APScheduler entrypoint)
|       |   |-- sync_upcoming.py
|       |   |-- sync_finished.py
|       |   |-- sync_stats.py
|       |   `-- evaluate_predictions.py
|       `-- tests/
|           |-- conftest.py                      (~  DB fixture + respx mock)
|           |-- unit/
|           |   |-- test_elo.py
|           |   |-- test_poisson.py
|           |   |-- test_metrics.py
|           |   `-- test_features_snapshot.py
|           `-- integration/
|               |-- test_api_fixtures.py
|               |-- test_predictions_immutability.py
|               `-- test_scheduled_evaluation.py
|
`-- frontend/
    |-- pubspec.yaml                             (*  dependencias congeladas)
    |-- analysis_options.yaml                    (*)
    `-- lib/
        |-- main.dart                           (~  entrypoint + APP bootstrap)
        |-- app.dart                            (~  MaterialApp.router + theme)
        |-- core/
        |   |-- config/app_config.dart          (~  URL del backend desde .env)
        |   |-- theme/
        |   |   |-- app_theme.dart              (~  Material 3 colorScheme)
        |   |   `-- colors.dart
        |   |-- routing/app_router.dart          (~  GoRouter routes)
        |   |-- network/
        |   |   |-- dio_client.dart              (~  interceptors, baseUrl)
        |   |   `-- api_endpoints.dart
        |   |-- errors/failures.dart
        |   `-- utils/
        |-- data/
        |   |-- models/                          (~  DTOs @freezed @JsonSerializable)
        |   |-- datasources/remote/
        |   |   |-- fixtures_remote_ds.dart
        |   |   |-- teams_remote_ds.dart
        |   |   |-- predictions_remote_ds.dart
        |   |   `-- performance_remote_ds.dart
        |   `-- repositories/
        |-- domain/
        |   |-- entities/
        |   |-- repositories/                    (~  contratos abstractos)
        |   `-- usecases/
        `-- presentation/
            |-- providers/                       (~  Riverpod Notifier/Future)
            |-- screens/
            |   |-- home/
            |   |-- match_detail/
            |   |-- predictions/
            |   |-- performance/
            |   `-- prediction_history/
            `-- widgets/
```

Notas:
- `backend/app/tests/` vive dentro del paquete para que `import app...`
  funcione tanto en runtime como en tests.
- `providers/` es la única carpeta que conoce API-Football. Cambiar de
  proveedor no afecta al resto del código.
- `services/explanation/glm.py` importa solo httpx y los schemas de
  explicación: no toca la DB ni los modelos.
