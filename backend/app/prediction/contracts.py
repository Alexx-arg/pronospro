"""Prediction-engine contracts (Sprint 5.0 — foundations).

These are the **type-level** contracts every Sprint 5.x component must
honour. They contain **no behaviour**: they declare the shapes that
trainers, predictors, calibrators and walk-forward folds must satisfy.

Spacing convention:
* ``NamedTuple``s for lightweight value tuples (zero runtime overhead,
  fully type-checked, immutable, hashable).
* ``@dataclass(frozen=True, slots=True)`` for richer payloads with
  defaults (used in :mod:`app.prediction.artifacts`).
* ``Protocol`` for behaviour contracts (predictors/trainers/
  calibrators) — supports structural typing without inheritance.

Key invariants enforced at this layer:
* :class:`MatchProbabilities` represents a point on the 2-simplex —
  the multiclass probability simplex for 3 classes. The invariant
  ``0 ≤ p_k ≤ 1`` and ``p_home + p_draw + p_away == 1`` is **not**
  enforced at construction time (a NamedTuple has no validator hook);
  it is enforced by the producers and asserted by tests.
* :class:`FixtureFeatures` is the **only** input format a predictor
  accepts. Predictors cannot read the dataset, the database, or
  recompute features — the ``feature_names`` tuple is carried
  alongside the vector so a predictor can defensively validate the
  shape it expects.

See ``docs/PHASE_5.md`` §3 for the design rationale.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

# numpy is already a declared dependency (numpy==2.1.1) used by Phase 4
# tests and by sklearn/statsmodels transitively. Importing it at the
# type level here stays within the "no new dependencies" Sprint 5.0
# constraint — it's not *used* in this module, only imported for the
# ndarray type alias below.
#
# If numpy were ever moved out, this module would fall back to the
# ``numpy.typing.NDArray`` import being replaceable by a Protocol —
# that decision would be a Phase-5 design change requiring reapproval.
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Avoid a runtime import cycle: artifacts.py imports contracts.py
    # for the dataclass field types (ModelName etc.). The Trainer
    # protocol's ``train`` return type is the only place contracts.py
    # needs to *name* ModelArtifact, and that's a string under
    # ``from __future__ import annotations``. Under ``mypy --strict``
    # the name must resolve at type-check time; importing under
    # TYPE_CHECKING satisfies the checker without ever importing
    # :mod:`app.prediction.artifacts` at runtime in this module.
    from app.prediction.artifacts import ModelArtifact


class ModelName(StrEnum):
    """The three canonical models to be compared in Phase 5.

    Membership is exhaustive — adding a new model is a deliberate
    change tracked in :file:`docs/PHASE_5.md` §11 and in the structural
    test ``test_model_name_contains_three_models``.
    """

    ELO_BASELINE = "elo_baseline"
    POISSON = "poisson"
    GRADIENT_BOOSTING = "gradient_boosting"


class MatchProbabilities(NamedTuple):
    """Output distribution over the 1X2 simplex.

    Represents the probability that the home team wins, the match
    ends in a draw, and the away team wins respectively.

    Contract (enforced by producers, asserted by tests):
    * ``0.0 ≤ p_home_win ≤ 1.0``
    * ``0.0 ≤ p_draw    ≤ 1.0``
    * ``0.0 ≤ p_away_win ≤ 1.0``
    * ``p_home_win + p_draw + p_away == 1.0`` (within rtol=1e-9).

    Optional per-goal distributions: when the underlying model
    produces them (Poisson, Elo-baseline via the shared
    ``poisson_to_1x2`` conversor), ``p_home_goals[k]`` is the
    probability the home team scores exactly ``k`` goals; ``None``
    otherwise (Gradient Boosting multiclass in the v1 iteration does
    not produce them).
    """

    p_home_win: float
    p_draw: float
    p_away_win: float
    p_home_goals: dict[int, float] | None = None
    p_away_goals: dict[int, float] | None = None


class FixtureFeatures(NamedTuple):
    """Input vector fed to a predictor for a single fixture.

    ``feature_vector`` is a 1-D array of shape ``(n_features,)`` aligned
    1:1 with ``feature_names``. Missing values from the Phase 4 CSV are
    represented as ``np.nan`` (NEVER coerced to ``0`` — see
    ``docs/PHASE_5.md`` §13). Predictors read this structure, not the
    dataset, not the DB.

    ``feature_names`` is the canonical tuple copied from
    :data:`app.features.example.FEATURE_NAMES` (or a subset thereof —
    e.g. Elo baseline only consumes the 3 Elo features). The
    ``ModelArtifact.inputs`` field declares what each artifact
    actually uses; a predictor MUST match its artifact's expected
    tuple before predicting.
    """

    fixture_id: int
    kickoff: datetime
    feature_vector: NDArray[Any]  # type: ignore[valid-type]
    feature_names: tuple[str, ...]


@runtime_checkable
class Predictor(Protocol):
    """Behavioural contract: ``predict`` maps a row to a distribution.

    Implementations MUST be deterministic given the same
    :class:`FixtureFeatures` and the same loaded :class:`ModelArtifact`
    (random seeds that produce variance belong in the trainer, not the
    predictor — once an artifact is sealed, ``predict`` is pure).
    """

    def predict(self, x: FixtureFeatures) -> MatchProbabilities: ...


@runtime_checkable
class Trainer(Protocol):
    """Behavioural contract: a trainer fits a model and seals an artifact.

    ``train`` receives only its designated slice of the dataset (the
    train block yielded by the walk-forward iterator); it MUST NOT
    retain references to the full dataset. The returned artifact must
    already carry ``payload_sha256`` and ``hyperparameters`` frozen —
    ``Trainer.train`` is the seal point.
    """

    name: ModelName

    def train(
        self,
        train_block: Sequence[FixtureFeatures],
        train_targets: Sequence[Sequence[int]],
        hyperparameters: dict[str, Any],
        seed: int | None,
    ) -> ModelArtifact: ...


class CalibratorKind(StrEnum):
    """Approved calibration methods.

    See ``docs/PHASE_5.md`` §7. ``ISOTONIC`` is intentionally omitted
    from defaults for multiclass (high overfit risk in K=3; does not
    preserve simplex by construction).
    """

    IDENTITY = "identity"
    TEMPERATURE = "temperature"
    DIRICHLET = "dirichlet"


@runtime_checkable
class Calibrator(Protocol):
    """Behavioural contract: simplex-preserving probability rectifier.

    ``fit`` is called on the validation block ONLY (never train, never
    test — defences enforced structurally by the runner in Sprint 5.8).
    ``transform`` maps raw probabilities to calibrated probabilities;
    producers guarantee ``Σ p == 1`` post-transform (enforced by
    math: temperature uses softmax, Dirichlet uses softmax; identity is
    trivially safe).
    """

    kind: CalibratorKind

    def fit(
        self,
        raw_probs: Sequence[MatchProbabilities],
        targets: Sequence[Sequence[int]],
    ) -> Calibrator: ...
    def transform(self, probs: Sequence[MatchProbabilities]) -> Sequence[MatchProbabilities]: ...


class WalkForwardMode(StrEnum):
    """Window-expansion policy of the walk-forward iterator.

    * ``EXPANDING``: the train window grows monotonically (each fold
      retains all previous examples + new ones). Default.
    * ``SLIDING``: the train window keeps a fixed length; the oldest
      examples are discarded as the fold advances. Useful for
      studying drift sensitivity; not the default because it discards
      potentially valuable historical signal.
    """

    EXPANDING = "expanding"
    SLIDING = "sliding"


class NanPolicy(StrEnum):
    """Missing-feature policy for models that cannot consume ``NaN``
    natively (Poisson GLM; binary classifiers without NaN handling).

    * ``DROP_ROW``: drop the fixture from the train block if any
      required feature is NaN; count and report as
      ``train_block_dropouts``.
    * ``IMPUTE_TRAIN_MEAN``: in test block, replace NaN with the
      per-feature mean from the train block (NEVER 0 — preserves the
      Phase 4 missing/zero distinction). For train block, drop rows
      regardless (imputing on the same data you fit on is leakage).

    Gradient Boosting (LightGBM candidate) consumes NaN natively and
    is exempted — it does NOT use this policy: NaN passes through.
    See ``docs/PHASE_5.md`` §13.
    """

    DROP_ROW = "drop_row"
    IMPUTE_TRAIN_MEAN = "impute_train_mean"


__all__ = [
    "Calibrator",
    "CalibratorKind",
    "FixtureFeatures",
    "MatchProbabilities",
    "ModelName",
    "NanPolicy",
    "Predictor",
    "Trainer",
    "WalkForwardMode",
]
