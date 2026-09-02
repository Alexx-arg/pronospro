# Flujos de datos

Esta sección detalla los caminos que sigue la información en cada operación
crítica. Referencia cruzada con `ARCHITECTURE.md`.

---

## Flujo 1 — Sincronización de próximos partidos

```
[Scheduler: cada 60 min]
        |
        v
 tasks/sync_upcoming
        |
        v
 services/sync/fixtures.sync_upcoming(league_ids=current leagues)
        |
        v
 providers.api_football.fetch_upcoming_fixtures
        |  (httpx + tenacity + aiolimiter; key solo aquí)
        v
 DTO normalizado (Provider DTO)
        |
        v
 repositories.fixture.upsert_many(dto) -> INSERT ... ON CONFLICT (external_id) DO UPDATE
        |     (actualiza kickoff_time, status, venue, round si cambiaron)
        v
 DB: fixtures (status scheduled/in_play)
        |
        v
 log: N fixtures upserted, watermark actualizado
```

Watermark: `sync_watermarks(competition_id, kind, last_run_at)` (tabla auxiliar
opcional, squadra del FIXTURES v1 evita duplicados vía `external_id`) —
mantiene el último `from` consultado para no saltar partidos.

---

## Flujo 2 — Sincronización de partidos terminados + evaluación

```
[Scheduler: cada 15 min]
        |
        v
 tasks/sync_finished
        |
        v
 services.sync.fixtures.sync_finished(window=now-6h..now)
        |
        v
 providers.fetch_finished_fixtures -> repositories.fixture.mark_finished
        v
 DB: fixtures.home_goals, away_goals, status='finished', finished_at
        |
        v
 tasks/evaluate_predictions  (puede ser parte del mismo job)
        |
        v
 services.evaluation.outcomes.evaluate_pending()
        |
        |   for fixture in fixtures_finished_sin_outcome_completo:
        |       preds = repositories.prediction.find_by_fixture_created_before_kickoff(fixture_id)
        |       for p in preds:
        |           actual_result = result(home_goals, away_goals)
        |           predicted_result = argmax(p.home/draw/away)
        |           correct = (actual==predicted)
        |           brier  = sum((p_home-actual_home)^2 + ...)
        |           log_loss = -log(p_actual_result + eps)
        |           prediction_outcome INSERT
        v
 DB: prediction_outcomes (+1 por prediction evaluada)
```

Importante: nunca se inserta `PredictionOutcome` si no existe `Prediction`
con `created_at < kickoff_time` (inmutabilidad + no retro fitting).

---

## Flujo 3 — Sincronización de estadísticas

```
[Scheduler: cada 6 h]
        |
        v
 tasks/sync_stats
        |
        v
 services.sync.statistics.sync_team_statistics(competitions=current)
        v
 providers.fetch_team_statistics -> repositories.team_statistics.upsert(as_of_date=today)
        v
 DB: team_statistics (unique team_id+competition_id+season_id+as_of_date)
        |
        v
 services.sync.statistics.sync_player_statistics(...)
        v
 DB: player_statistics
        v
 services.sync.injuries_lineups.sync_injuries(); sync_lineups(for upcoming fixtures only)
```

---

## Flujo 4 — Generación de predicción

Trigger: Scheduler (`tasks/sync_upcoming` final) + endpoint
`POST /admin/predict` para forzar. Solo se predice si `kickoff_time - now >=
MIN_HOURS_BEFORE_KICKOFF` (default 1h) y no existe predicción previa para
`(fixture_id, model_version_id)`.

```
PredictorOrchestrator.predict(fixture_id, model_version_id)
        |
        v
 features.build_snapshot(fixture_id)
   -Season/Competition del fixture
   - home/away team_statistics mas reciente con as_of_date < kickoff_time
   - forma reciente (W/D/L) reconstruida desde fixtures terminados < kickoff
   - H2H (ultimos N fixtures < kickoff entre los dos equipos)
   - injuries activas con start_date < kickoff_time
   - lineup si esta disponible y updated_external_at < kickoff_time
        |
        v
 features_snapshot: JSONB  (clave para reproducibilidad y anti leakage)
        |
        v
 model.predict(features_snapshot) -> PredictionDTO
        |
        v
 repositories.prediction.insert(dto)        # INSERT ONLY, nunca UPDATE
        |
        v
 ExplanationService.explain(context ) -> GLM
        |   -> httpx POST a GLM con JSON estructurado (no prompting de probabilidades)
        |   -> sistema prompt prohíbe alterar números; valida schema de salida
        v
 repositories.prediction_explanations.insert(prediction_id, ...)
        |
        v
 return Prediction (con explanation)
```

Si GLM no responde a tiempo o invalida schema: la prediction **ya está
persistida** (numbers-first). Se inserta `prediction_explanations` luego
(vía job best-effort) o se devuelve `explanation: null`. Los números no se
alteran.

---

## Flujo 5 — Request del frontend (p.ej. pantalla Home)

```
Flutter Riverpod: upcomingFixturesProvider
        |
        v
 usecase GetUpcomingFixturesUseCase
        |
        v
 data/repositories/FixtureRepository (impl) -> datasource/remote/FixturesRemoteDS
        |
        v
 Dio.get('/api/v1/fixtures/upcoming?include_prediction=true')
        |
        v
 FastAPI router -> FixtureService.get_upcoming()
        |
        v
 repositories.fixture.list_upcoming(filters) -> SELECT
        |
        v
 opcional: repositories.prediction.find_current_for_fixtures(ids)
        |
        v
 schemas/fixtures.Outgoing -> JSON
        |
        v
 Dio response -> DTOs @freezed -> Riverpod state -> UI
```

---

## Flujo 6 — Pantalla Performance

```
Flutter -> GET /api/v1/performance?model_version_id=...
Backend -> PerformanceService.metrics(filters)
   -> repos.prediction_outcomes.aggregate(filters) (GROUP BY queries sobre
      prediction_outcomes JOIN predictions JOIN fixtures JOIN seasons)
   -> calcula calibration buckets in Python (pandas small)
   -> 200 JSON
Flutter -> fl_chart: gráfico de calibration + barras por competencia
```

---

## Diagrama de tiempos (typical 24h)

```
T+00:00  [cron upcoming 60m]   sync fixtures próximos
T+00:05  [cron upcoming end]   generate predictions para fixtures sin predicción
T+00:15  [cron finished 15m]   actualiza resultados -> evaluate outcomes
T+00:30  [cron evaluate 30m]   re-intenta evaluar predicciones sin outcome
T+06:00  [cron stats 6h]       team/player statistics + injuries + lineups
```

Los offsets (5min, 30s, etc.) entre jobs evitan solapamientos y respetan
rate limits de API-Football (30 req/min en plan base).

---

## Prevención de data leakage — checklist aplicada en runtime

- `features.build_snapshot` une `team_statistics` con
  `WHERE as_of_date < DATE(kickoff_time)`.
- `fixtures` terminados: `WHERE status='finished' AND kickoff_time < kickoff`.
- `injuries`: `WHERE start_date < kickoff_time`.
- `lineups`: solo si `updated_external_at < kickoff_time` (las alineaciones
  confirmadas se publican ~1h antes; si se actualizan después del kick-off,
  NO se usan para esa predicción).
- `h2h`: `WHERE kickoff_time < fixture.kickoff_time`.
- Backtesting usa walk-forward con splits temporales (sin shuffle, sin
  fixtures futuros en ventanas de entrenamiento).

---

## Rate limiting / retries / observabilidad

| Cliente | Limite | Política de retries |
|---|---|---|
| API-Football | 30 req/min, 300-1000 req/día (plan) | `tenacity`: 3 retries, exp backoff 2s + jitter, solo en 429/5xx |
| GLM | 10 req/min | `tenacity`: 3 retries, exp backoff 1s + jitter, solo en 429/5xx |
| Frontend -> Backend | sin limite en app (throttle server side opcional) | Dio interceptor: 2 retries en 5xx |

Logs estructurados (`loguru`) por cada llamada a provider/GLM con
`request_id` (context var), endpoint, latency, status. Métricas de
`evaluate_outcomes` publicadas en log (en v1 sin Prometheus).
