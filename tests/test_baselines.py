from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import TARGET_COLUMN, TIMESTAMP_COLUMN
from src.data.split import (
    FINAL_TEST_FOLD_ID,
    KIND_DEVELOPMENT,
    KIND_FINAL_TEST,
    ROLE_EMBARGO,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models.baselines import (
    BASELINE_PERSISTENCE,
    BASELINE_TRAINING_MEAN,
    INTERPOLATED_COLUMN,
    RESULT_COLUMNS,
    compute_metrics,
    evaluate_baselines,
    evaluate_fold,
    get_fold_frames,
    mae,
    predict_persistence,
    predict_training_mean,
    r2,
    rmse,
    summarize_development,
    validate_evaluation,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(targets, segments=None, interpolated=None, start="2020-01-01"):
    """Build a minimal hourly table with the columns the baselines need."""
    n = len(targets)
    timestamps = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            TARGET_COLUMN: [float(v) for v in targets],
            SEGMENT_COLUMN: [0] * n if segments is None else segments,
            SENSOR_ELIGIBLE_COLUMN: True,
            INTERPOLATED_COLUMN: [False] * n if interpolated is None else interpolated,
        }
    )


def make_assignment(hourly, train_idx, embargo_idx, validation_idx, test_idx=()):
    """Assign hourly rows to roles for a single development fold (id 1)."""
    rows = []
    for role, indices in (
        (ROLE_TRAIN, train_idx),
        (ROLE_EMBARGO, embargo_idx),
        (ROLE_VALIDATION, validation_idx),
    ):
        for i in indices:
            rows.append(
                {
                    TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].iloc[i],
                    "fold_id": 1,
                    "fold_kind": KIND_DEVELOPMENT,
                    "role": role,
                }
            )
    for i in test_idx:
        rows.append(
            {
                TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].iloc[i],
                "fold_id": FINAL_TEST_FOLD_ID,
                "fold_kind": KIND_FINAL_TEST,
                "role": ROLE_TEST,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 8. Metric calculations against known synthetic examples
# ---------------------------------------------------------------------


def test_metrics_match_hand_computed_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 4.0, 6.0])
    # errors: 0, 0, -1, -2 -> squared 0,0,1,4 -> mean 1.25 -> rmse sqrt(1.25)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1.25))
    assert mae(y_true, y_pred) == pytest.approx(0.75)

    # ss_res = 5, ss_tot = sum((y - 2.5)^2) = 2.25+0.25+0.25+2.25 = 5
    assert r2(y_true, y_pred) == pytest.approx(0.0)


def test_r2_of_perfect_and_mean_predictors():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y_true, y_true) == pytest.approx(1.0)
    # Predicting the mean of y_true scores exactly zero by definition.
    assert r2(y_true, np.full_like(y_true, y_true.mean())) == pytest.approx(0.0)
    assert rmse(y_true, y_true) == pytest.approx(0.0)


def test_metrics_reject_degenerate_or_invalid_input():
    with pytest.raises(ValueError, match="zero variance"):
        r2(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="does not match"):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="zero observations"):
        mae(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        rmse(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


def test_compute_metrics_returns_all_three():
    result = compute_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 2.5])
    assert set(result) == {"rmse", "mae", "r2"}
    assert all(np.isfinite(v) for v in result.values())


# ---------------------------------------------------------------------
# 1 & 2. Training mean uses training rows only
# ---------------------------------------------------------------------


def test_training_mean_uses_training_rows_only():
    hourly = make_hourly([1.0, 2.0, 3.0, 100.0, 200.0])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[3:5]

    predicted = predict_training_mean(train, validation)
    assert predicted["prediction"].nunique() == 1
    assert predicted["prediction"].iloc[0] == pytest.approx(2.0)  # mean of 1,2,3 only
    assert len(predicted) == len(validation)


def test_changing_validation_target_does_not_change_training_mean():
    base = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0])
    altered = make_hourly([1.0, 2.0, 3.0, 999.0, -999.0])

    first = predict_training_mean(base.iloc[0:3], base.iloc[3:5])
    second = predict_training_mean(altered.iloc[0:3], altered.iloc[3:5])

    assert first["prediction"].iloc[0] == pytest.approx(second["prediction"].iloc[0])


# ---------------------------------------------------------------------
# 3, 4, 5. Persistence behaviour
# ---------------------------------------------------------------------


def test_persistence_first_validation_hour_uses_last_training_target():
    # rows 0-2 train, row 3 embargo, rows 4-6 validation
    hourly = make_hourly([1.0, 2.0, 7.0, 555.0, 10.0, 20.0, 30.0])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[4:7]

    predicted = predict_persistence(train, validation)

    # First validation hour reaches back past the embargo to the last
    # training observation (7.0), never the embargo value (555.0).
    assert predicted["prediction"].iloc[0] == pytest.approx(7.0)
    assert predicted["source_timestamp"].iloc[0] == hourly[TIMESTAMP_COLUMN].iloc[2]


def test_persistence_never_uses_the_current_target():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[3:6]

    predicted = predict_persistence(train, validation)
    observed = validation[TARGET_COLUMN].to_numpy()

    # No prediction may equal its own hour's target here, since all values differ.
    assert not np.any(predicted["prediction"].to_numpy() == observed)
    # Every source strictly precedes the hour it predicts.
    assert (predicted["source_timestamp"] < predicted[TIMESTAMP_COLUMN]).all()


def test_persistence_walks_forward_through_validation():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[3:6]

    predicted = predict_persistence(train, validation)
    # 3.0 carried from training, then each earlier validation target.
    assert predicted["prediction"].tolist() == pytest.approx([3.0, 10.0, 20.0])


def test_persistence_ignores_embargo_target_values():
    # Embargo row carries an extreme value that must never be used.
    hourly = make_hourly([5.0, 6.0, 9999.0, 10.0, 20.0])
    train = hourly.iloc[0:2]
    validation = hourly.iloc[3:5]

    predicted = predict_persistence(train, validation)

    assert 9999.0 not in set(predicted["prediction"])
    assert predicted["prediction"].iloc[0] == pytest.approx(6.0)
    embargo_ts = hourly[TIMESTAMP_COLUMN].iloc[2]
    assert embargo_ts not in set(predicted["source_timestamp"].dropna())


def test_persistence_frozen_variant_uses_only_training_history():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[3:6]

    predicted = predict_persistence(train, validation, use_validation_history=False)
    # Every hour receives the last training target; the state never updates.
    assert predicted["prediction"].tolist() == pytest.approx([3.0, 3.0, 3.0])


def test_persistence_ignores_interpolated_targets():
    hourly = make_hourly(
        [1.0, 2.0, 777.0, 10.0, 20.0],
        interpolated=[False, False, True, False, False],
    )
    train = hourly.iloc[0:3]  # includes the interpolated hour
    validation = hourly.iloc[3:5]

    predicted = predict_persistence(train, validation)
    assert 777.0 not in set(predicted["prediction"])
    assert predicted["prediction"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------
# 6 & 7. Segment reset and unavailable counting
# ---------------------------------------------------------------------


def test_persistence_resets_across_temporal_segments():
    # Training sits entirely in segment 0; validation entirely in segment 1.
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0], segments=[0, 0, 0, 1, 1])
    train = hourly.iloc[0:3]
    validation = hourly.iloc[3:5]

    predicted = predict_persistence(train, validation)

    # The first segment 1 hour has no earlier same-segment observation.
    assert pd.isna(predicted["prediction"].iloc[0])
    # The second one may use the first validation hour, same segment.
    assert predicted["prediction"].iloc[1] == pytest.approx(10.0)
    # No training value crossed the gap.
    assert 3.0 not in set(predicted["prediction"].dropna())


def test_unavailable_predictions_are_counted_not_imputed():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], segments=[0, 0, 0, 1, 1, 1])
    assignment = make_assignment(
        hourly, train_idx=[0, 1, 2], embargo_idx=[], validation_idx=[3, 4, 5]
    )
    frames = get_fold_frames(hourly, assignment, 1)
    results = evaluate_fold(frames)

    persistence = results[results["baseline"] == BASELINE_PERSISTENCE].iloc[0]
    assert persistence["n_validation"] == 3
    assert persistence["n_unavailable"] == 1  # first segment 1 hour
    assert persistence["n_scored"] == 2
    assert persistence["n_scored"] + persistence["n_unavailable"] == persistence["n_validation"]

    # The training mean always covers every validation hour.
    mean_row = results[results["baseline"] == BASELINE_TRAINING_MEAN].iloc[0]
    assert mean_row["n_unavailable"] == 0
    assert mean_row["n_scored"] == 3


# ---------------------------------------------------------------------
# 9. Final test rows are never evaluated
# ---------------------------------------------------------------------


def test_final_test_rows_are_not_evaluated():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 50.0, 60.0])
    assignment = make_assignment(
        hourly, train_idx=[0, 1, 2], embargo_idx=[3], validation_idx=[4], test_idx=[5, 6]
    )
    frames = get_fold_frames(hourly, assignment, 1)

    test_ts = set(hourly[TIMESTAMP_COLUMN].iloc[[5, 6]])
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(test_ts)
    assert not set(frames.validation[TIMESTAMP_COLUMN]).intersection(test_ts)

    predicted = predict_persistence(frames.train, frames.validation)
    assert not set(predicted["source_timestamp"].dropna()).intersection(test_ts)


def test_requesting_the_final_test_fold_is_rejected():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0])
    assignment = make_assignment(hourly, [0, 1], [], [3], test_idx=[2])
    with pytest.raises(ValueError, match="final test fold must not be evaluated"):
        get_fold_frames(hourly, assignment, FINAL_TEST_FOLD_ID)


# ---------------------------------------------------------------------
# 10 & 11. Schema and determinism
# ---------------------------------------------------------------------


def test_result_schema_contains_expected_columns():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    assignment = make_assignment(hourly, [0, 1, 2], [], [3, 4, 5])
    results = evaluate_baselines(hourly, assignment, fold_ids=(1,))

    assert list(results.columns) == RESULT_COLUMNS
    assert set(results["baseline"]) == {BASELINE_TRAINING_MEAN, BASELINE_PERSISTENCE}
    assert len(results) == 2


def test_results_are_deterministic_across_runs():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    assignment = make_assignment(hourly, [0, 1, 2], [], [3, 4, 5])

    first = evaluate_baselines(hourly, assignment, fold_ids=(1,))
    second = evaluate_baselines(hourly, assignment, fold_ids=(1,))
    pd.testing.assert_frame_equal(first, second)

    # Input row order must not change the outcome.
    shuffled = hourly.sample(frac=1.0, random_state=0).reset_index(drop=True)
    third = evaluate_baselines(shuffled, assignment, fold_ids=(1,))
    pd.testing.assert_frame_equal(first, third)


def test_summary_reports_mean_and_spread():
    hourly = make_hourly([float(i) for i in range(40)])
    assignment = pd.concat(
        [
            make_assignment(hourly, list(range(0, 10)), [], list(range(10, 20))),
            make_assignment(hourly, list(range(0, 20)), [], list(range(20, 30))).assign(fold_id=2),
            make_assignment(hourly, list(range(0, 30)), [], list(range(30, 40))).assign(fold_id=3),
        ],
        ignore_index=True,
    )
    results = evaluate_baselines(hourly, assignment)
    summary = summarize_development(results)

    assert set(summary["baseline"]) == {BASELINE_TRAINING_MEAN, BASELINE_PERSISTENCE}
    for column in ("rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"):
        assert column in summary.columns
    assert (summary["n_folds"] == 3).all()


# ---------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------


def test_validate_evaluation_flags_bad_counts():
    hourly = make_hourly([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    assignment = make_assignment(hourly, [0, 1, 2], [], [3, 4, 5])
    results = evaluate_baselines(hourly, assignment, fold_ids=(1,))

    corrupted = results.copy()
    corrupted.loc[0, "n_scored"] = 999
    with pytest.raises(ValueError, match="scored"):
        validate_evaluation(corrupted, hourly, assignment)


# ---------------------------------------------------------------------
# Real data integration (skips cleanly if artifacts are absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_HOURLY.exists() and REAL_SPLITS.exists()),
    reason="processed artifacts not available locally",
)
def test_real_data_baselines():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)

    results = evaluate_baselines(hourly, assignment)
    validate_evaluation(results, hourly, assignment)

    # Three folds x two baselines.
    assert len(results) == 6
    assert sorted(results["fold_id"].unique()) == [1, 2, 3]
    assert set(results["baseline"]) == {BASELINE_TRAINING_MEAN, BASELINE_PERSISTENCE}

    # Validation counts must match the committed split assignment exactly.
    committed = (
        assignment[
            (assignment["role"] == ROLE_VALIDATION)
            & (assignment["fold_kind"] == KIND_DEVELOPMENT)
        ]
        .groupby("fold_id")
        .size()
    )
    for _, row in results.iterrows():
        assert row["n_validation"] == committed.loc[row["fold_id"]]

    # No final test observation is scored.
    test_ts = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    for fold_id in (1, 2, 3):
        frames = get_fold_frames(hourly, assignment, fold_id)
        assert not set(frames.validation[TIMESTAMP_COLUMN]).intersection(test_ts)
        assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(test_ts)

    # No metric is infinite or missing.
    for metric in ("rmse", "mae", "r2"):
        assert np.isfinite(results[metric]).all()

    # Persistence covers the large majority of validation observations.
    persistence = results[results["baseline"] == BASELINE_PERSISTENCE]
    coverage = persistence["n_scored"].sum() / persistence["n_validation"].sum()
    assert coverage > 0.95


@pytest.mark.skipif(
    not (REAL_HOURLY.exists() and REAL_SPLITS.exists()),
    reason="processed artifacts not available locally",
)
def test_real_data_results_are_deterministic():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    first = evaluate_baselines(hourly, assignment)
    second = evaluate_baselines(hourly, assignment)
    pd.testing.assert_frame_equal(first, second)
