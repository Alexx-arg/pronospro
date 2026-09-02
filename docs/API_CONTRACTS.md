# Contratos de API (v1)

Base URL: `http://<host>:8000/api/v1`
Content-Type: `application/json`. Todas las rutas devuelven errores con el
formato:

```json
{ "error": { "code": "NOT_FOUND", "message": "...", "details": {} } }
```

Errores estándar: `400 BAD_REQUEST`, `401 UNAUTHORIZED`, `404 NOT_FOUND`,
`409 CONFLICT` (p.ej. predecir un fixture ya jugado o demasiado tarde),
`429 TOO_MANY_REQUESTS`, `422 UNPROCESSABLE_ENTITY`, `500 INTERNAL`.

Paginación uniforme: `limit` (default 20, max 100), `offset` (default 0).
Respuesta envuelta:

```json
{ "items": [ ... ], "limit": 20, "offset": 0, "total": 137 }
```

---

## 1. `GET /fixtures/upcoming`

Próximos partidos (status `scheduled`/`in_play`), ordenados por `kickoff_time`.

Query params (todos opcionales):
| param | tipo | default | descripción |
|---|---|---|---|
| `competition_id` | int | – | filtra liga |
| `season_id` | int | – | filtra temporada |
| `team_id` | int | – |filtra por equipo local o visitante |
| `from` | ISO date | hoy | kickoff ≥ from |
| `to` | ISO date | +7d | kickoff ≤ to |
| `limit` | int | 20 | |
| `offset` | int | 0 | |
| `include_prediction` | bool | false | incluye predicción vigente embebida |

Response 200:
```json
{
  "items": [
    {
      "id": 1024,
      "external_id": 878921,
      "competition": { "id": 39, "name": "Premier League", "logo": "..." },
      "season": { "id": 12, "year": 2025 },
      "home_team": { "id": 50, "name": "Arsenal", "logo": "..." },
      "away_team": { "id": 49, "name": "Chelsea", "logo": "..." },
      "kickoff_time": "2026-08-15T19:00:00Z",
      "status": "scheduled",
      "venue": "Emirates Stadium",
      "prediction": null
    }
  ],
  "limit": 20, "offset": 0, "total": 137
}
```

## 2. `GET /fixtures/{fixture_id}`

Detalle completo con estadísticas, forma reciente, H2H y predicción vigente
(explicación incluida si existe).

Response 200 (resumido):
```json
{
  "fixture": { "id": 1024, "kickoff_time": "...", "status": "scheduled", ... },
  "competition": { ... },
  "season": { ... },
  "home_team": { ... },
  "away_team": { ... },
  "home_stats":   { "form": "WWDLW", "xg": 1.7, "xga": 0.9, "goals_for": 2.1, ... },
  "away_stats":   { ... },
  "h2h": [
    { "date": "2026-02-10", "home_goals": 2, "away_goals": 1, "competition": "Premier League" }
  ],
  "injuries": [
    { "player": "M. Saliba", "team_id": 50, "status": "doubtful", "type": "Hamstring" }
  ],
  "lineup": null,
  "prediction": {
    "id": 998,
    "model_version": { "name": "elo", "version": "v1.0.0" },
    "home_probability": 0.46, "draw_probability": 0.27, "away_probability": 0.27,
    "expected_home_goals": 1.62, "expected_away_goals": 1.05,
    "confidence": 0.72,
    "created_at": "2026-08-14T10:00:00Z",
    "explanation": {
      "summary": "Arsenal favoured due to home form and xG superiority.",
      "main_factors": ["Home advantage", "xG 1.7 vs 1.1", "Clean form WWDLW"],
      "risk_factors": ["Saliba doubtful", "Chelsea H2H parity last 5"],
      "confidence_explanation": "Confidence 0.72 reflects balanced midfield but stable home record."
    }
  }
}
```
404 si no existe.

## 3. `GET /teams/{team_id}`

```json
{
  "id": 50, "name": "Arsenal", "short_name": "ARS", "code": "ARS",
  "country": "England", "logo": "...", "venue": "Emirates Stadium",
  "founded": 1886
}
```

## 4. `GET /teams/{team_id}/stats`

Query params: `competition_id` (optional), `season_id` (optional, default
current season of the competition).

```json
{
  "team_id": 50,
  "competition_id": 39, "season_id": 12, "as_of_date": "2026-08-15",
  "fixtures_played": 3, "wins": 2, "draws": 1, "losses": 0,
  "goals_for": 6, "goals_against": 2, "clean_sheets": 2, "failed_to_score": 0,
  "form": "WWDLW",
  "shots_total": 47, "shots_on_target": 18, "possession_avg": 58.4,
  "xg": 1.7, "xga": 0.9,
  "yellow_cards": 5, "red_cards": 0
}
```

## 5. `GET /predictions`

Predicciones históricas y vigentes.

Query params:
| param | tipo | default | descripción |
|---|---|---|---|
| `model_version_id` | int | – | filtra por versión de modelo |
| `fixture_id` | int | – | */
| `competition_id` | int | – | |
| `season_id` | int | – | |
| `status` | enum | – | `pending` (sin outcome) / `evaluated` |
| `correct` | bool | – | solo acertadas / falladas |
| `from` | ISO date | – | kickoff ≥ |
| `to` | ISO date | – | kickoff ≤ |
| `limit` | int | 20 | |
| `offset` | int | 0 | |

Response 200 (resumen por item):
```json
{
  "items": [
    {
      "id": 998,
      "fixture": { "id": 1024, "kickoff_time": "...",
                   "home_team": { "name": "Arsenal" },
                   "away_team": { "name": "Chelsea" } },
      "model_version": { "name": "elo", "version": "v1.0.0" },
      "home_probability": 0.46, "draw_probability": 0.27, "away_probability": 0.27,
      "expected_home_goals": 1.62, "expected_away_goals": 1.05,
      "confidence": 0.72,
      "created_at": "...",
      "outcome": null
    }
  ],
  "limit": 20, "offset": 0, "total": 412
}
```

`outcome` presente solo si fixture `finished` y ya evaluado:
```json
"outcome": {
  "actual_home_goals": 2, "actual_away_goals": 1, "actual_result": "home",
  "predicted_result": "home", "correct": true,
  "brier_score": 0.142, "log_loss": 0.331, "evaluated_at": "..."
}
```

## 6. `GET /predictions/{prediction_id}`

Response 200 incluye `features_snapshot` (JSONB crudo del snapshot de
features) y `explanation`, además de los campos del item de listado.
404 si no existe.

## 7. `GET /performance`

Métricas agregadas de predicciones ya evaluadas.

Query params:
| param | tipo | default |
|---|---|---|
| `model_version_id` | int | todos |
| `competition_id` | int | todas |
| `season_id` | int | todas |
| `from` / `to` | ISO date | todo |
| `confidence_buckets` | bool | true |

Response 200:
```json
{
  "filters": { "model_version_id": null, "competition_id": null, "from": null, "to": null },
  "prediction_count": 412,
  "accuracy": 0.546,
  "brier_score": 0.198,
  "log_loss": 0.612,
  "calibration": [
    { "bucket": [0.0, 0.3),  "count": 41,  "empirical_acc": 0.293, "avg_confidence": 0.21 },
    { "bucket": [0.3, 0.5),  "count": 88,  "empirical_acc": 0.432, "avg_confidence": 0.42 },
    { "bucket": [0.5, 0.7),  "count": 121, "empirical_acc": 0.611, "avg_confidence": 0.61 },
    { "bucket": [0.7, 0.85), "count": 99,  "empirical_acc": 0.757, "avg_confidence": 0.78 },
    { "bucket": [0.85, 1.0], "count": 63,  "empirical_acc": 0.857, "avg_confidence": 0.92 }
  ],
  "accuracy_by_confidence": [ ... mismo bucketing ... ],
  "accuracy_by_competition": [
    { "competition_id": 39, "name": "Premier League", "count": 212, "accuracy": 0.561 }
  ],
  "accuracy_by_season": [
    { "season_id": 12, "year": 2025, "count": 380, "accuracy": 0.545 }
  ]
}
```

## 8. `GET /models`

```json
{
  "items": [
    { "id": 1, "name": "elo", "version": "v1.0.0", "is_active": true,
      "description": "Elo ratings diferencia", "parameters": {} },
    { "id": 2, "name": "poisson", "version": "v1.0.0", "is_active": true,
      "description": "Poisson xG", "parameters": {"decay": 0.92} }
  ]
}
```

---

## Endpoint administrativos (no expuestos a clientes móviles; protegidos por API key interna)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/admin/sync/upcoming` | dispara sincronización de próximos partidos |
| `POST` | `/admin/sync/finished` | dispara actualización de partidos terminados |
| `POST` | `/admin/sync/stats` | dispara sincronización de estadísticas |
| `POST` | `/admin/evaluate` | dispara evaluación de predicciones vencidas |
| `POST` | `/admin/predict` | genera predicciones para fixtures pendientes |

Normalmente invocados por el `worker` (APScheduler). Se exponen como HTTP
para debugging/operación manual. No se incluyen en el cliente Flutter.

## Versionado y compatibilidad

- El prefijo `/api/v1/` es obligatorio. Cambios breaking conllevan `/v2`.
- DTOs son campos opcionales-additivos (nunca se quita un campo de una
  response del mismo minor). El frontend usa `freezed` para tolerar
  unknown keys (no rompen).
