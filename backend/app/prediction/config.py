"""Prediction-engine configuration (Sprint 5.0).

``PredictionSettings`` is a deliberately small, standalone
``pydantic-settings`` config. It is NOT merged into
:mod:`app.config.Settings`: Phase 5 is offline-only and so its
configuration namespace is kept isolated, following the same "Phase
settings live near the phase code" pattern as Phase 3 jobs.

Variables load from the environment with the ``PREDICTION_`` prefix.
Field aliases match the variable names in ``.env.example`` (see
``docs/PHASE_5.md`` Appendix A).

Validation rules implemented here are minimal-but-real: data-shape
invariants (non-negative integers, ratios in ``(0, 1)``) that would
otherwise silently corrupt a walk-forward run.

See ``docs/PHASE_5.md`` §18 for the canonical field list.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prediction.contracts import NanPolicy, WalkForwardMode


class PredictionSettings(BaseSettings):
    """All knobs that influence walk-forward + comparison in Phase 5.

    Two runtime groups are exposed:

    * **Walk-forward iterator** — controls how
      :class:`app.prediction.backtesting.iterator.WalkForwardIterator`
      (Sprint 5.1) slices the dataset into folds.
    * **Calibration / metrics** — controls bins count for ECE/MCE and
      confidence buckets for ``per_confidence`` reports.

    Model-specific knobs (e.g. Poisson ``regularization_l2``, GB
    ``n_estimators``, Elo baseline ``beta_1`` grid) are NOT part of
    this settings object: they belong to each
      :class:`app.prediction.training.Trainer`'s default hyperparameter
      dict, and are pinned through the ``hyperparameters`` field of
      :class:`app.prediction.artifacts.ModelArtifact`.
    """

    model_config = SettingsConfigDict(
        env_prefix="PREDICTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Dataset selection ---
    dataset_version: str = Field(default="v001")

    # --- Walk-forward iterator ---
    walk_forward_min_train_size: int = Field(default=200, ge=1)
    walk_forward_val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    walk_forward_test_size: int = Field(default=50, ge=1)
    walk_forward_gap_days: int = Field(default=1, ge=0)
    walk_forward_mode: WalkForwardMode = Field(default=WalkForwardMode.EXPANDING)

    # --- Calibration / metrics ---
    calibration_bins: int = Field(default=10, ge=2, le=100)
    # Same CSV-of-floats encoding as ``app.config.Settings.confidence_buckets``
    # to keep .env.example visually consistent across the app.
    confidence_buckets: str = Field(
        default="0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )

    # --- Reproducibility ---
    random_seed: int = Field(default=42, ge=0)

    # --- Model-specific defaults shared across models ---
    # ``max_goals`` is a parameter shared by Elo baseline and Poisson
    # (the Poisson matrix cap). Lives here rather than on each trainer
    # so the configurability story is uniform across models and so the
    # contractual default is documented in .env.example.
    max_goals: int = Field(default=10, ge=1, le=30)

    # ``nan_policy`` is the *fallback* policy used by Poisson (which
    # cannot consume NaN natively). Gradient Boosting is exempted
    # (LightGBM / XGBoost consume NaN). Elo baseline does not use
    # any NaN-eligible feature. See PHASE_5.md §13.
    nan_policy: NanPolicy = Field(default=NanPolicy.DROP_ROW)

    @field_validator("walk_forward_min_train_size")
    @classmethod
    def _min_train_size_nonzero(cls, v: int) -> int:
        # ``ge=1`` already rejects 0; we keep an explicit validator
        # so the failure message is direct (Phase 5's value is
        # being misconfigured by humans here, not silently).
        if v <= 0:
            raise ValueError("walk_forward_min_train_size must be > 0")
        return v

    @field_validator("walk_forward_val_ratio")
    @classmethod
    def _val_ratio_strictly_in_unit_interval(cls, v: float) -> float:
        # ``gt=0, lt=1`` already covers it; explicit check for clarity
        # in error messages (also guards against ``v == 0``/1 subtleties
        # with floating-point equality at the bounds).
        if not (0.0 < v < 1.0):
            raise ValueError(
                "walk_forward_val_ratio must satisfy 0.0 < ratio < 1.0"
            )
        return v

    @field_validator("confidence_buckets")
    @classmethod
    def _confidence_buckets_sorted_and_increasing(cls, v: str) -> str:
        """``"b1,b2,...,bN"`` — must be strictly increasing and in ``[0, 1]``.

        The first bucked ("left edge") may be > 0; the last must be ≤ 1
        (the canonical value is ``1.0`` — the right-inclusive edge of
        the highest confidence bucket).
        """
        tokens = [tok.strip() for tok in v.split(",") if tok.strip()]
        if len(tokens) < 2:
            raise ValueError(
                "confidence_buckets must have at least 2 values, "
                f"got {tokens!r}"
            )
        try:
            bounds = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(
                f"confidence_buckets must be numeric, got {tokens!r}"
            ) from exc
        for b in bounds:
            if not (0.0 <= b <= 1.0):
                raise ValueError(
                    f"confidence_buckets entries must be in [0, 1], "
                    f"got {b!r}"
                )
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if not (a < b):
                raise ValueError(
                    f"confidence_buckets must be strictly increasing, "
                    f"got {bounds!r}"
                )
        return v

    def confidence_bucket_list(self) -> list[float]:
        """Parse ``confidence_buckets`` into a sorted strictly-increasing
        ``list[float]`` (eager; called once by the metrics runtime).
        """
        return [
            float(tok.strip())
            for tok in self.confidence_buckets.split(",")
            if tok.strip()
        ]


@lru_cache
def get_prediction_settings() -> PredictionSettings:
    """Cached accessor — mirrors :func:`app.config.get_settings`. Use this
    rather than instantiating ``PredictionSettings()`` ad-hoc so that
    tests monkeypatching the settings object see a single instance."""
    return PredictionSettings()


__all__ = ["PredictionSettings", "get_prediction_settings"]
