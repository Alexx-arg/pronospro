"""Application settings loaded from environment variables / .env file.

Only settings relevant to the persistence layer (Phase 2) are defined here.
Provider, GLM and scheduler settings will be added in later phases; their
placeholders are kept so that the .env.example contract stays backward
compatible.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration.

    Environment variables are loaded from the process environment and, when
    available, from a local ``.env`` file.  Field names map 1:1 to the
    variable names defined in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- App -----
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    backend_api_prefix: str = Field(default="/api/v1", alias="BACKEND_API_PREFIX")
    admin_api_key: str = Field(default="change-me", alias="ADMIN_API_KEY")
    # Sprint 6.3 — Security / CORS (D10.1: no hardcode, env-driven)
    api_secret_key: str | None = Field(default=None, alias="API_SECRET_KEY")
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

    # ----- PostgreSQL -----
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    # If DATABASE_URL is provided it takes precedence over the assembled URL.
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # DB connection behaviour
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=5, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: float = Field(default=30.0, alias="DB_POOL_TIMEOUT")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # ----- Prediction policy (used by later phases, kept here for parity) -----
    min_hours_before_kickoff: int = Field(
        default=1, alias="MIN_HOURS_BEFORE_KICKOFF_FOR_PREDICTION"
    )
    confidence_buckets: str = Field(
        default="0.0,0.3,0.5,0.7,0.85,1.0", alias="CONFIDENCE_BUCKETS"
    )

    # ----- Roles (for the init.sql bootstrap script, Phase 2 use) -----
    db_app_role: str = Field(default="app_user", alias="DB_APP_ROLE")
    db_app_role_password: str = Field(default="changeme", alias="DB_APP_ROLE_PASSWORD")

    # ---- kept for forward-compatibility with later phases (not used here) ----
    data_provider: str = Field(default="api_football", alias="DATA_PROVIDER")

    # ----- NVIDIA NIM Explanation Service -----
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    nvidia_model: str = Field(default="meta/llama-3.2-11b-vision-instruct", alias="NVIDIA_MODEL")

    # ----- API-Football provider (legacy, no longer usado — Sprint 7.x usa Bzzoiro) -----
    # API key MUST come from the environment. The variable name is the
    # canonical one required by the user spec: API_FOOTBALL_KEY.
    api_football_key: str = Field(default="", alias="API_FOOTBALL_KEY")
    api_football_base_url: str = Field(
        default="https://v3.football.api-sports.io",
        alias="API_FOOTBALL_BASE_URL",
    )
    api_football_timeout_seconds: float = Field(
        default=30.0, alias="API_FOOTBALL_TIMEOUT_SECONDS"
    )
    api_football_max_retries: int = Field(default=3, alias="API_FOOTBALL_MAX_RETRIES")
    api_football_rate_per_minute: int = Field(
        default=30, alias="API_FOOTBALL_RATE_LIMIT_PER_MIN"
    )
    api_football_rate_per_day: int | None = Field(
        default=None, alias="API_FOOTBALL_RATE_LIMIT_PER_DAY"
    )
    api_football_user_agent: str = Field(
        default="football-prediction-app/0.1", alias="API_FOOTBALL_USER_AGENT"
    )
    # Network / httpx pool tuning
    api_football_max_connections: int = Field(
        default=20, alias="API_FOOTBALL_MAX_CONNECTIONS"
    )
    api_football_max_keepalive_connections: int = Field(
        default=10, alias="API_FOOTBALL_MAX_KEEPALIVE_CONNECTIONS"
    )

    # ----- Sync configuration -----
    # Each job can be toggled independent of the scheduler (jobs are still
    # invocable manually via the admin endpoint / CLI even when disabled).
    sync_upcoming_enabled: bool = Field(default=True, alias="SYNC_UPCOMING_ENABLED")
    sync_upcoming_days: int = Field(default=7, alias="SYNC_UPCOMING_DAYS")
    sync_upcoming_interval_minutes: int = Field(
        default=60, alias="SYNC_UPCOMING_INTERVAL_MINUTES"
    )

    sync_finished_enabled: bool = Field(default=True, alias="SYNC_FINISHED_ENABLED")
    sync_finished_window_hours: int = Field(
        default=6, alias="SYNC_FINISHED_WINDOW_HOURS"
    )
    sync_finished_interval_minutes: int = Field(
        default=15, alias="SYNC_FINISHED_INTERVAL_MINUTES"
    )

    sync_teams_enabled: bool = Field(default=True, alias="SYNC_TEAMS_ENABLED")
    sync_teams_interval_minutes: int = Field(
        default=720, alias="SYNC_TEAMS_INTERVAL_MINUTES"
    )

    sync_team_statistics_enabled: bool = Field(
        default=True, alias="SYNC_TEAM_STATISTICS_ENABLED"
    )
    sync_team_statistics_interval_minutes: int = Field(
        default=360, alias="SYNC_TEAM_STATISTICS_INTERVAL_MINUTES"
    )

    sync_player_statistics_enabled: bool = Field(
        default=True, alias="SYNC_PLAYER_STATISTICS_ENABLED"
    )
    sync_player_statistics_interval_minutes: int = Field(
        default=360, alias="SYNC_PLAYER_STATISTICS_INTERVAL_MINUTES"
    )

    sync_injuries_enabled: bool = Field(default=True, alias="SYNC_INJURIES_ENABLED")
    sync_injuries_interval_minutes: int = Field(
        default=180, alias="SYNC_INJURIES_INTERVAL_MINUTES"
    )

    sync_lineups_enabled: bool = Field(default=True, alias="SYNC_LINEUPS_ENABLED")
    sync_lineups_interval_minutes: int = Field(
        default=120, alias="SYNC_LINEUPS_INTERVAL_MINUTES"
    )

    # Lightweight scheduler orchestrator (APScheduler)
    scheduler_enabled: bool = Field(default=False, alias="SCHEDULER_ENABLED")

    # Which leagues should the scheduler sync by default (comma-separated IDs).
    # 39=Premier League, 140=La Liga, 61=Serie A, 135=Bundesliga, 78=Ligue 1.
    current_leagues: str = Field(
        default="39,140,61,135,78", alias="CURRENT_LEAGUES"
    )

    # Year of the "current" season per league. Same value applied to all
    # leagues listed above (the provider actually expects the season start
    # year, e.g. 2024 for the 2024/2025 season in European leagues).
    current_season_year: int = Field(
        default=2024, alias="CURRENT_SEASON_YEAR"
    )

    def async_database_url(self) -> str:
        """Return the async SQLAlchemy URL (asyncpg driver)."""
        if self.database_url:
            url = self.database_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql+psycopg://"):
                url = url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
            return url
        return (
            "postgresql+asyncpg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def current_league_ids(self) -> list[int]:
        """Return the parsed list of leagues from ``current_leagues``."""
        return [int(tok) for tok in self.current_leagues.split(",") if tok.strip()]

    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS env var.

        Empty string → no CORS (restrictive). "*" → allow all (only in debug).
        Comma-separated list otherwise.
        """
        raw = self.cors_origins.strip()
        if not raw:
            return []
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (used by the session module and later layers)."""
    return Settings()  # type: ignore[call-arg]
