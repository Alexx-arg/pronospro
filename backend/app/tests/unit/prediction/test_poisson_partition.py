"""Partition verification for Poisson (Sprint 5.6 Option A)."""

from __future__ import annotations

from app.features.example import FEATURE_NAMES, TARGET_NAMES
from app.prediction.models.poisson import (
    HEAD_AWAY_FEATURES,
    HEAD_HOME_FEATURES,
    SHARED_FEATURES,
)


def test_partition_covers_all_66() -> None:
    union = set(HEAD_HOME_FEATURES) | set(HEAD_AWAY_FEATURES)
    assert len(union) == 66
    assert union == set(FEATURE_NAMES)
    assert len(HEAD_HOME_FEATURES) == 37
    assert len(HEAD_AWAY_FEATURES) == 37


def test_partition_is_disjoint_except_shared() -> None:
    overlap = set(HEAD_HOME_FEATURES) & set(HEAD_AWAY_FEATURES)
    assert overlap == set(SHARED_FEATURES)
    assert len(overlap) == 8
    assert set(SHARED_FEATURES) == {
        "h2h_home_wins",
        "h2h_away_wins",
        "h2h_draws",
        "h2h_total_goals_mean",
        "h2h_sample_size",
        "elo_difference",
        "table_points_difference",
        "table_goal_difference_difference",
    }


def test_no_target_features_in_either_head() -> None:
    for name in HEAD_HOME_FEATURES + HEAD_AWAY_FEATURES:
        assert name not in set(TARGET_NAMES), f"target {name} in partition"


def test_no_duplicate_within_head() -> None:
    assert len(HEAD_HOME_FEATURES) == len(set(HEAD_HOME_FEATURES))
    assert len(HEAD_AWAY_FEATURES) == len(set(HEAD_AWAY_FEATURES))


def test_all_features_are_from_feature_names() -> None:
    feat_set = set(FEATURE_NAMES)
    for name in HEAD_HOME_FEATURES:
        assert name in feat_set
    for name in HEAD_AWAY_FEATURES:
        assert name in feat_set


def test_shared_features_in_both() -> None:
    for name in SHARED_FEATURES:
        assert name in HEAD_HOME_FEATURES
        assert name in HEAD_AWAY_FEATURES
