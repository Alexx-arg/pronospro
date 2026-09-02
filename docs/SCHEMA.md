# Esquema de PostgreSQL

Convenciones del DDL:
- `BIGSERIAL` PKs con `id` propio + `external_id` UNIQUE para el ID de
  API-Football (desacopla nuestra identidad del proveedor).
- `TIMESTAMPTZ` para todos los tiempos; fechas como `DATE`.
- `JSONB` para features snapshot, parámetros de modelo y explicación GLM.
- `NUMERIC(6,5)` para probabilidades (3 decimales usados; 1 de margen).
- `VARCHAR + CHECK` en vez de `ENUM` nativo para no romper `alembic`.
- Timestamps automáticos via trigger `updated_at` (configurable por tabla).
- Schema default `public`. Search path default.

Índices: enumerados al final de cada tabla. Tabla central es `predictions`,
**inmutable** (ver §10).

---

## 1. Competitions

```sql
CREATE TABLE competitions (
    id          BIGSERIAL PRIMARY KEY,
    external_id INTEGER UNIQUE NOT NULL,
    name        VARCHAR(200) NOT NULL,
    country     VARCHAR(100),
    logo        TEXT,
    type        VARCHAR(50) NOT NULL
                CHECK (type IN ('league','cup','playoff','super_cup')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_competitions_country ON competitions(country);
```

## 2. Seasons

```sql
CREATE TABLE seasons (
    id             BIGSERIAL PRIMARY KEY,
    competition_id BIGINT NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
    external_id    INTEGER NOT NULL,
    year           INTEGER NOT NULL,
    start_date     DATE,
    end_date       DATE,
    is_current     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (competition_id, year),
    UNIQUE (external_id, year)
);
CREATE INDEX idx_seasons_current ON seasons(is_current) WHERE is_current;
```

## 3. Teams

```sql
CREATE TABLE teams (
    id          BIGSERIAL PRIMARY KEY,
    external_id INTEGER UNIQUE NOT NULL,
    name        VARCHAR(200) NOT NULL,
    short_name  VARCHAR(50),
    code        VARCHAR(10),
    country     VARCHAR(100),
    logo        TEXT,
    venue       VARCHAR(200),
    founded     INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_teams_name_lower ON teams (LOWER(name));
```

## 4. Players

```sql
CREATE TABLE players (
    id          BIGSERIAL PRIMARY KEY,
    external_id INTEGER UNIQUE NOT NULL,
    name        VARCHAR(200) NOT NULL,
    photo       TEXT,
    nationality VARCHAR(100),
    birth_date  DATE,
    height_cm   INTEGER,
    weight_kg   INTEGER,
    position    VARCHAR(20) CHECK (position IN ('GK','DF','MF','FW')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Asociación jugador + equipo por competicion/temporada (puede cambiar)
CREATE TABLE player_team_seasons (
    id             BIGSERIAL PRIMARY KEY,
    player_id      BIGINT NOT NULL REFERENCES players(id)   ON DELETE CASCADE,
    team_id        BIGINT NOT NULL REFERENCES teams(id)     ON DELETE CASCADE,
    competition_id BIGINT NOT NULL REFERENCES competitions(id),
    season_id     BIGINT NOT NULL REFERENCES seasons(id)   ON DELETE CASCADE,
    is_on_loan     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (player_id, season_id, competition_id)
);
CREATE INDEX idx_pts_team  ON player_team_seasons(team_id, season_id);
CREATE INDEX idx_pts_team ON player_team_seasons(player_id, season_id);
```

## 5. Fixtures

```sql
CREATE TABLE fixtures (
    id             BIGSERIAL PRIMARY KEY,
    external_id    INTEGER UNIQUE NOT NULL,
    competition_id BIGINT NOT NULL REFERENCES competitions(id),
    season_id      BIGINT NOT NULL REFERENCES seasons(id)  ON DELETE CASCADE,
    home_team_id   BIGINT NOT NULL REFERENCES teams(id),
    away_team_id   BIGINT NOT NULL REFERENCES teams(id),
    matchday       INTEGER,
    round          VARCHAR(50),
    venue          VARCHAR(200),
    kickoff_time   TIMESTAMPTZ NOT NULL,
    status         VARCHAR(20) NOT NULL DEFAULT 'scheduled'
                   CHECK (status IN ('scheduled','in_play',
                                     'finished','postponed',
                                     'cancelled','suspended')),
    status_short    VARCHAR(5),
    home_goals      INTEGER,
    away_goals      INTEGER,
    finished_at     TIMESTAMPTZ,
    synced_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (home_goals IS NULL AND away_goals IS NULL)
        OR
        (home_goals IS NOT NULL AND away_goals IS NOT NULL)
    )
);
CREATE INDEX idx_fixtures_kickoff    ON fixtures(kickoff_time);
CREATE INDEX idx_fixtures_status    ON fixtures(status, kickoff_time);
CREATE INDEX idx_fixtures_comp_season ON fixtures(competition_id, season_id);
CREATE INDEX idx_fixtures_home       ON fixtures(home_team_id, kickoff_time);
CREATE INDEX idx_fixtures_away       ON fixtures(away_team_id, kickoff_time);
```

## 6. Team statistics

```sql
CREATE TABLE team_statistics (
    id               BIGSERIAL PRIMARY KEY,
    team_id          BIGINT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    competition_id   BIGINT NOT NULL REFERENCES competitions(id),
    season_id        BIGINT NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    as_of_date       DATE NOT NULL,

    fixtures_played  INTEGER NOT NULL DEFAULT 0,
    wins             INTEGER NOT NULL DEFAULT 0,
    draws            INTEGER NOT NULL DEFAULT 0,
    losses           INTEGER NOT NULL DEFAULT 0,
    goals_for        INTEGER NOT NULL DEFAULT 0,
    goals_against    INTEGER NOT NULL DEFAULT 0,
    clean_sheets     INTEGER NOT NULL DEFAULT 0,
    failed_to_score  INTEGER NOT NULL DEFAULT 0,
    form             VARCHAR(20),                -- p.ej. "WWDLW"
    shots_total       INTEGER,
    shots_on_target   INTEGER,
    shots_inside_box  INTEGER,
    shots_outside_box INTEGER,
    fouls             INTEGER,
    corners           INTEGER,
    offsides          INTEGER,
    possession_avg    NUMERIC(5,2),
    yellow_cards      INTEGER,
    red_cards         INTEGER,
    passes_total      INTEGER,
    passes_accuracy    NUMERIC(5,2),
    xg                 NUMERIC(6,3),
    xga                NUMERIC(6,3),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (team_id, competition_id, season_id, as_of_date)
);
CREATE INDEX idx_team_stats_lkp ON team_statistics(season_id, team_id, as_of_date);
```

## 7. Player statistics

```sql
CREATE TABLE player_statistics (
    id              BIGSERIAL PRIMARY KEY,
    player_id       BIGINT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    team_id         BIGINT NOT NULL REFERENCES teams(id)   ON DELETE CASCADE,
    competition_id  BIGINT NOT NULL REFERENCES competitions(id),
    season_id       BIGINT NOT NULL REFERENCES seasons(id)  ON DELETE CASCADE,
    appearances     INTEGER NOT NULL DEFAULT 0,
    starts          INTEGER NOT NULL DEFAULT 0,
    minutes_played  INTEGER NOT NULL DEFAULT 0,
    goals           INTEGER NOT NULL DEFAULT 0,
    assists         INTEGER NOT NULL DEFAULT 0,
    yellow_cards    INTEGER NOT NULL DEFAULT 0,
    red_cards       INTEGER NOT NULL DEFAULT 0,
    rating          NUMERIC(4,2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (player_id, season_id, competition_id)
);
CREATE INDEX idx_player_stats_team ON player_statistics(team_id, season_id);
```

## 8. Injuries

```sql
CREATE TABLE injuries (
    id                   BIGSERIAL PRIMARY KEY,
    external_id          INTEGER UNIQUE NOT NULL,
    player_id            BIGINT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    team_id              BIGINT NOT NULL REFERENCES teams(id)   ON DELETE CASCADE,
    competition_id       BIGINT REFERENCES competitions(id),
    fixture_id           BIGINT REFERENCES fixtures(id),
    type                 VARCHAR(100),
    reason               VARCHAR(255),
    status               VARCHAR(20) NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active','doubtful','recovered','suspended')),
    start_date           DATE NOT NULL,
    end_date             DATE,
    updated_external_at  TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_injuries_active   ON injuries(status, end_date) WHERE status IN ('active','doubtful');
CREATE INDEX idx_injuries_player   ON injuries(player_id, start_date);
CREATE INDEX idx_injuries_fixture ON injuries(fixture_id);
```

## 9. Lineups

```sql
CREATE TABLE lineups (
    id                   BIGSERIAL PRIMARY KEY,
    fixture_id           BIGINT NOT NULL REFERENCES fixtures(id)  ON DELETE CASCADE,
    team_id              BIGINT NOT NULL REFERENCES teams(id)     ON DELETE CASCADE,
    is_home              BOOLEAN NOT NULL,
    formation            VARCHAR(10),
    coach                VARCHAR(200),
    updated_external_at  TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (fixture_id, team_id)
);

CREATE TABLE lineup_players (
    id           BIGSERIAL PRIMARY KEY,
    lineup_id    BIGINT NOT NULL REFERENCES lineups(id) ON DELETE CASCADE,
    player_id    BIGINT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    position     VARCHAR(20) CHECK (position IN ('GK','DF','MF','FW')),
    position_x   INTEGER,
    position_y   INTEGER,
    shirt_number INTEGER,
    is_starter   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (lineup_id, player_id)
);
CREATE INDEX idx_lineup_players_lineup ON lineup_players(lineup_id, is_starter);
```

## 10. Model versions

```sql
CREATE TABLE model_versions (
    id           BIGSERIAL PRIMARY KEY,
    name         VARCHAR(50) NOT NULL
                 CHECK (name IN ('elo','poisson','gradient_boosting')),
    version      VARCHAR(50) NOT NULL,
    description  TEXT,
    parameters   JSONB NOT NULL DEFAULT '{}',
    is_active    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);
-- Solo una version activa por nombre de modelo
CREATE UNIQUE INDEX uq_model_active_per_name
    ON model_versions(name) WHERE is_active;
```

## 11. Predictions (INMUTABLE)

```sql
CREATE TABLE predictions (
    id                   BIGSERIAL PRIMARY KEY,
    fixture_id           BIGINT NOT NULL REFERENCES fixtures(id) ON DELETE RESTRICT,
    model_version_id     BIGINT NOT NULL REFERENCES model_versions(id),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kickoff_time         TIMESTAMPTZ NOT NULL,
    home_probability     NUMERIC(6,5) NOT NULL CHECK (home_probability  BETWEEN 0 AND 1),
    draw_probability     NUMERIC(6,5) NOT NULL CHECK (draw_probability  BETWEEN 0 AND 1),
    away_probability     NUMERIC(6,5) NOT NULL CHECK (away_probability  BETWEEN 0 AND 1),
    expected_home_goals  NUMERIC(5,2) NOT NULL CHECK (expected_home_goals >= 0),
    expected_away_goals  NUMERIC(5,2) NOT NULL CHECK (expected_away_goals >= 0),
    confidence           NUMERIC(5,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    features_snapshot    JSONB NOT NULL,
    explanation          JSONB,
    CHECK (home_probability + draw_probability + away_probability
           BETWEEN 0.999 AND 1.001),            -- suma 1 (tolerancia 0.001)
    UNIQUE (fixture_id, model_version_id)       -- 1 prediccion por modelo por fixture
);
CREATE INDEX idx_predictions_model   ON predictions(model_version_id, created_at);
CREATE INDEX idx_predictions_fixture ON predictions(fixture_id);
CREATE INDEX idx_predictions_created USING GIN ON predictions(created_at);
```

### 11.1 Inmutabilidad (defense in depth)

```sql
-- (a) Permisos: la app solo recibe SELECT/INSERT en esta tabla.
REVOKE UPDATE, DELETE ON predictions FROM PUBLIC;
-- GRANT SELECT, INSERT ON predictions TO <app_role>;

-- (b) Trigger que bloquea cualquier UPDATE aunque un rol con permisos lo intente.
CREATE OR REPLACE FUNCTION prevent_prediction_update()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'predictions is immutable: cannot UPDATE row id=%', OLD.id;
END;
$$;

CREATE TRIGGER trg_no_update_predictions
    BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION prevent_prediction_update();
```

Consecuencias operativas:
- Corregir un modelo = crear `model_version` nueva y nuevas `predictions`.
- `explanation` se setea al momento del INSERT (incluso si GLM tarda, se
  insertan los números sólidos y se hace `UPDATE` solo de la columna
  `explanation`... eso violaria el trigger. Solución: insertar con
  `explanation=NULL` y nunca actualizarla; si la explicación llega después,
  se persiste en una tabla auxiliar `prediction_explanations` con FK 1:1 —
  ver subsection 11.2). 

### 11.2 Explicaciones postergadas (sin mutar predictions)

```sql
CREATE TABLE prediction_explanations (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   BIGINT UNIQUE NOT NULL REFERENCES predictions(id) ON DELETE RESTRICT,
    summary         TEXT NOT NULL,
    main_factors    JSONB NOT NULL,         -- array de strings
    risk_factors    JSONB NOT NULL,         -- array de strings
    confidence_explanation TEXT NOT NULL,
    model_used      VARCHAR(100),           -- p.ej. "glm-5.2"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```
La mayoría de las predicciones tendrán explicación al insertar; esta tabla
solo se usa si GLM no respondió a tiempo y la predicción ya fue persistida
(numbers-first: nunca esperamos al LLM para guardar los números).

---

## 12. Prediction outcomes

```sql
CREATE TABLE prediction_outcomes (
    id                       BIGSERIAL PRIMARY KEY,
    prediction_id            BIGINT UNIQUE NOT NULL REFERENCES predictions(id)
                              ON DELETE RESTRICT,
    fixture_id               BIGINT NOT NULL REFERENCES fixtures(id)
                              ON DELETE RESTRICT,
    actual_home_goals        INTEGER NOT NULL CHECK (actual_home_goals >= 0),
    actual_away_goals        INTEGER NOT NULL CHECK (actual_away_goals >= 0),
    actual_result            VARCHAR(5) NOT NULL
                              CHECK (actual_result IN ('home','draw','away')),
    predicted_result         VARCHAR(5) NOT NULL
                              CHECK (predicted_result IN ('home','draw','away')),
    correct                  BOOLEAN NOT NULL,
    predicted_correct_prob   NUMERIC(6,5) NOT NULL,
    brier_score              NUMERIC(8,5) NOT NULL,
    log_loss                 NUMERIC(8,5) NOT NULL,
    evaluated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_outcomes_model     ON prediction_outcomes(fixture_id);
CREATE INDEX idx_outcomes_correct  ON prediction_outcomes(correct);
```

## 13. Métricas (vista auxiliar)

Agregación para `/api/v1/performance`. No se materializa; se deriva via repositorio
uniendo `prediction_outcomes → predictions → fixtures → seasons/competitions`:

```
accuracy                = mean(correct)
brier_score_avg         = mean(brier_score)
log_loss_avg            = mean(log_loss)
calibration_curve       = bucketiza por confidence [0-0.2),(0.2-0.4)...
                          y calcula |empirical_freq - confidence| por bucket
accuracy_by_confidence  = mean(correct) GROUP BY bucket(confidence)
accuracy_by_competition = mean(correct) GROUP BY competition_id
accuracy_by_season      = mean(correct) GROUP BY season_id
prediction_count        = count(*)
```

Buckets de confianza (por defecto): `[0.0,0.3), [0.3,0.5), [0.5,0.7),
[0.7,0.85), [0.85,1.0]`. Configurables en `.env` (`CONFIDENCE_BUCKETS`).

---

## 14. Mantenimiento automatizado

- Trigger genérico `updated_at` aplicado a todas las tablas mutables
  (NO a `predictions` ni `prediction_explanations`).
- `PARTITION BY RANGE (kickoff_time)` en `fixtures` si el volumen crece
  (futuro); no se incluye en la v1.
- Limpieza: jobs de VACUUM/ANALYZE periódicos fuera de la app.

## 15. Roles PostgreSQL (al final del setup)

- `app_user`: SELECT/INSERT en casi todo. UPDATE en tablas operativas
  (`fixtures`, `team_statistics`, `player_statistics`, `injuries`, `lineups`).
  **Nunca** UPDATE/DELETE en `predictions` / `prediction_outcomes`.
- `migrator`: rol para `alembic upgrade head` (DDL).
- `read_only`: para dashboards/BI.

Esto obliga que incluso una caída de seguridad en la app no permita mutar
el historial de predicciones.
