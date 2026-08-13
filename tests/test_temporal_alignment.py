from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (
    CORE_SENSOR_PREDICTOR_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
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
from src.models.baselines import INTERPOLATED_COLUMN
from src.models.linear_regression import build_spark_session
from src.models.random_forest import evaluate_model as random_forest_evaluate_model
from src.models.temporal_alignment import (
    ALIGNMENT_HOURS,
    ALIGNMENT_RESULT_COLUMNS,
    BASELINE_ALIGNMENT_HOURS,
    TARGET_TIMESTAMP_COLUMN,
    assess_consistency,
    best_supported_alignment,
    build_aligned_dataset,
    compare_with_baseline_alignment,
    evaluate_alignment,
    matched_comparison,
    matched_consistency,
    restrict_assignment,
    run_experiment,
    shift_target,
    summarize_alignments,
    validate_alignment,
    verify_reproduces_committed_benchmark,
    verify_reproducible,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_RANDOM_FOREST = REPO_ROOT / "data" / "processed" / "random_forest_results.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_SPLITS.exists()


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityTemporalAlignmentTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(n_rows: int = 150, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Hourly table with all 57 sensor predictors on a contiguous hourly grid.

    The target is a distinct value per hour, so a shifted target can be
    traced back to the exact hour it came from.
    """
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    frame[TARGET_COLUMN] = 2.0 + 1.5 * frame[predictors[0]] - 0.8 * frame[predictors[1]]
    # Make every target value unique and identifiable by hour.
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN] + np.arange(n_rows) * 1e-6
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    frame[SEGMENT_COLUMN] = 0
    frame[SENSOR_ELIGIBLE_COLUMN] = True
    frame[INTERPOLATED_COLUMN] = False
    for column in FEED_CONTEXT_PREDICTOR_COLUMNS:
        frame[column] = 50.0
    return frame


def make_assignment(hourly, train_idx, embargo_idx, validation_idx, test_idx=()):
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


def default_fixture():
    """Train 0..99, embargo 100..105, validation 106..149, on 150 contiguous hours."""
    hourly = make_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 100)),
        embargo_idx=list(range(100, 106)),
        validation_idx=list(range(106, 150)),
    )
    return hourly, assignment


def dropped(dataset, role: str) -> int:
    loss = dataset.row_loss
    return int(loss.loc[loss["role"] == role, "n_dropped"].iloc[0])


# ---------------------------------------------------------------------
# Correct target shifting
# ---------------------------------------------------------------------


@pytest.mark.parametrize("alignment_hours", ALIGNMENT_HOURS)
def test_shift_takes_the_target_from_the_later_hour(alignment_hours):
    hourly = make_hourly()
    shifted = shift_target(hourly, alignment_hours)

    lookup = hourly.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]
    lag = pd.Timedelta(alignment_hours, unit="h")

    assert (shifted[TARGET_TIMESTAMP_COLUMN] - shifted[TIMESTAMP_COLUMN] == lag).all()
    expected = shifted[TARGET_TIMESTAMP_COLUMN].map(lookup).to_numpy()
    assert np.array_equal(shifted[TARGET_COLUMN].to_numpy(), expected)

    # Spot check the first row against the raw table directly.
    first = shifted.iloc[0]
    assert first[TARGET_COLUMN] == pytest.approx(
        hourly[TARGET_COLUMN].iloc[alignment_hours], abs=0.0
    )


def test_zero_hour_shift_is_the_identity():
    hourly = make_hourly()
    shifted = shift_target(hourly, 0)

    assert len(shifted) == len(hourly)
    assert np.array_equal(shifted[TARGET_COLUMN].to_numpy(), hourly[TARGET_COLUMN].to_numpy())
    assert (shifted[TARGET_TIMESTAMP_COLUMN] == shifted[TIMESTAMP_COLUMN]).all()


def test_shift_leaves_every_predictor_untouched():
    """The features must stay at hour t. Moving them is what would import
    sensor readings recorded after the assay being predicted."""
    hourly = make_hourly()
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    for alignment_hours in ALIGNMENT_HOURS:
        shifted = shift_target(hourly, alignment_hours)
        original = hourly.set_index(TIMESTAMP_COLUMN).loc[shifted[TIMESTAMP_COLUMN], predictors]
        assert np.array_equal(
            shifted[predictors].to_numpy(dtype=float), original.to_numpy(dtype=float)
        )


def test_negative_alignment_is_rejected():
    hourly = make_hourly()
    with pytest.raises(ValueError, match="zero or a forward shift"):
        shift_target(hourly, -1)


def test_duplicate_timestamps_are_rejected():
    hourly = make_hourly(n_rows=10)
    duplicated = pd.concat([hourly, hourly.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        shift_target(duplicated, 1)


# ---------------------------------------------------------------------
# Chronological order
# ---------------------------------------------------------------------


@pytest.mark.parametrize("alignment_hours", ALIGNMENT_HOURS)
def test_shift_preserves_chronological_order(alignment_hours):
    hourly = make_hourly().sample(frac=1.0, random_state=7).reset_index(drop=True)
    shifted = shift_target(hourly, alignment_hours)

    assert shifted[TIMESTAMP_COLUMN].is_monotonic_increasing
    assert shifted[TARGET_TIMESTAMP_COLUMN].is_monotonic_increasing
    if alignment_hours > 0:
        assert (shifted[TARGET_TIMESTAMP_COLUMN] > shifted[TIMESTAMP_COLUMN]).all()


def test_restricted_assignment_stays_ordered_within_each_role():
    hourly, assignment = default_fixture()
    dataset = build_aligned_dataset(hourly, assignment, 2)

    for (_, _), group in dataset.assignment.groupby(["fold_id", "role"]):
        assert group[TIMESTAMP_COLUMN].is_monotonic_increasing


# ---------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------


def test_training_rows_never_take_a_target_from_the_embargo():
    """The last training hours are removed rather than allowed to read a
    target out of the embargo window the split deliberately withheld."""
    hourly, assignment = default_fixture()
    embargo_start = hourly[TIMESTAMP_COLUMN].iloc[100]

    for alignment_hours in (1, 2):
        dataset = build_aligned_dataset(hourly, assignment, alignment_hours)
        development = dataset.assignment[dataset.assignment["fold_kind"] == KIND_DEVELOPMENT]
        train = development[development["role"] == ROLE_TRAIN]
        target_hours = train[TIMESTAMP_COLUMN] + pd.Timedelta(alignment_hours, unit="h")
        assert (target_hours < embargo_start).all()


def test_validation_rows_never_take_a_target_from_beyond_the_window():
    hourly, assignment = default_fixture()
    validation_end = hourly[TIMESTAMP_COLUMN].iloc[149]

    for alignment_hours in (1, 2):
        dataset = build_aligned_dataset(hourly, assignment, alignment_hours)
        development = dataset.assignment[dataset.assignment["fold_kind"] == KIND_DEVELOPMENT]
        validation = development[development["role"] == ROLE_VALIDATION]
        target_hours = validation[TIMESTAMP_COLUMN] + pd.Timedelta(alignment_hours, unit="h")
        assert (target_hours <= validation_end).all()


def test_no_development_row_takes_a_target_from_the_final_test_period():
    hourly = make_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 90)),
        embargo_idx=list(range(90, 96)),
        validation_idx=list(range(96, 130)),
        test_idx=list(range(130, 150)),
    )
    test_hours = set(hourly[TIMESTAMP_COLUMN].iloc[130:150])

    for alignment_hours in (1, 2):
        dataset = build_aligned_dataset(hourly, assignment, alignment_hours)
        validate_alignment(dataset, hourly, assignment)

        development = dataset.assignment[dataset.assignment["fold_kind"] == KIND_DEVELOPMENT]
        target_hours = set(
            development[TIMESTAMP_COLUMN] + pd.Timedelta(alignment_hours, unit="h")
        )
        assert not target_hours.intersection(test_hours)

        # The test assignment itself must survive untouched, so the
        # downstream guards still have something to assert against.
        preserved = set(
            dataset.assignment.loc[dataset.assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
        )
        assert preserved == test_hours


def test_target_is_never_carried_across_a_temporal_discontinuity():
    hourly = make_hourly()
    hourly.loc[hourly.index >= 50, SEGMENT_COLUMN] = 1  # discontinuity between hour 49 and 50

    shifted = shift_target(hourly, 1)
    boundary_hour = hourly[TIMESTAMP_COLUMN].iloc[49]
    assert boundary_hour not in set(shifted[TIMESTAMP_COLUMN])

    lookup = hourly.set_index(TIMESTAMP_COLUMN)[SEGMENT_COLUMN]
    assert (
        shifted[TARGET_TIMESTAMP_COLUMN].map(lookup).to_numpy()
        == shifted[SEGMENT_COLUMN].to_numpy()
    ).all()


def test_missing_target_hour_removes_the_row_rather_than_imputing():
    hourly = make_hourly(n_rows=20).drop(index=[10]).reset_index(drop=True)
    shifted = shift_target(hourly, 1)

    missing_predecessor = hourly[TIMESTAMP_COLUMN].iloc[9]  # hour 9, whose t+1 was removed
    assert missing_predecessor not in set(shifted[TIMESTAMP_COLUMN])
    assert shifted[TARGET_COLUMN].notna().all()


def test_validate_alignment_detects_a_tampered_target():
    hourly, assignment = default_fixture()
    dataset = build_aligned_dataset(hourly, assignment, 1)
    validate_alignment(dataset, hourly, assignment)

    tampered = dataset.hourly.copy()
    tampered.loc[tampered.index[5], TARGET_COLUMN] = 999.0
    broken = type(dataset)(
        alignment_hours=dataset.alignment_hours,
        hourly=tampered,
        assignment=dataset.assignment,
        row_loss=dataset.row_loss,
    )
    with pytest.raises(ValueError, match="shifted target values are incorrect"):
        validate_alignment(broken, hourly, assignment)


def test_validate_alignment_detects_a_moved_predictor():
    hourly, assignment = default_fixture()
    dataset = build_aligned_dataset(hourly, assignment, 1)

    tampered = dataset.hourly.copy()
    first_predictor = CORE_SENSOR_PREDICTOR_COLUMNS[0]
    tampered[first_predictor] = tampered[first_predictor].shift(-1).bfill()
    broken = type(dataset)(
        alignment_hours=dataset.alignment_hours,
        hourly=tampered,
        assignment=dataset.assignment,
        row_loss=dataset.row_loss,
    )
    with pytest.raises(ValueError, match="predictor values changed"):
        validate_alignment(broken, hourly, assignment)


def test_restriction_cannot_invent_rows():
    hourly, assignment = default_fixture()
    retained = set(hourly[TIMESTAMP_COLUMN])
    restricted = restrict_assignment(assignment, retained, 1)

    committed = set(zip(assignment["fold_id"], assignment[TIMESTAMP_COLUMN], assignment["role"]))
    produced = set(zip(restricted["fold_id"], restricted[TIMESTAMP_COLUMN], restricted["role"]))
    assert produced.issubset(committed)


# ---------------------------------------------------------------------
# Expected row loss
# ---------------------------------------------------------------------


def test_zero_hour_alignment_removes_no_rows():
    hourly, assignment = default_fixture()
    dataset = build_aligned_dataset(hourly, assignment, 0)

    assert (dataset.row_loss["n_dropped"] == 0).all()
    assert len(dataset.assignment) == len(assignment)
    assert len(dataset.hourly) == len(hourly)


@pytest.mark.parametrize("alignment_hours", [1, 2])
def test_row_loss_matches_the_expected_boundary_rows(alignment_hours):
    """On a contiguous grid, a shift of k hours costs exactly k rows at the
    end of each role block: those are the hours whose target would fall
    into the next role, or off the end of the series."""
    hourly, assignment = default_fixture()
    dataset = build_aligned_dataset(hourly, assignment, alignment_hours)

    assert dropped(dataset, ROLE_TRAIN) == alignment_hours
    assert dropped(dataset, ROLE_VALIDATION) == alignment_hours
    assert dropped(dataset, ROLE_EMBARGO) == alignment_hours

    loss = dataset.row_loss
    assert (loss["n_retained"] == loss["n_committed"] - loss["n_dropped"]).all()
    assert (loss["n_dropped"] >= 0).all()

    # The hourly frame itself only loses the hours with no successor at all.
    assert len(dataset.hourly) == len(hourly) - alignment_hours


def test_row_loss_grows_with_the_shift():
    hourly, assignment = default_fixture()
    losses = [
        build_aligned_dataset(hourly, assignment, alignment_hours).row_loss["n_dropped"].sum()
        for alignment_hours in ALIGNMENT_HOURS
    ]
    assert losses == sorted(losses)
    assert losses[0] == 0


# ---------------------------------------------------------------------
# Comparison, aggregation, and decision rules
# ---------------------------------------------------------------------


def _results_fixture(rmse_by_alignment):
    rows = []
    for alignment_hours, values in rmse_by_alignment.items():
        for fold_id, rmse in zip((1, 2, 3), values):
            rows.append(
                {
                    "alignment_hours": alignment_hours,
                    "fold_id": fold_id,
                    "model": "random_forest",
                    "n_train": 1000,
                    "n_validation": 100,
                    "n_scored": 100,
                    "n_features": 57,
                    "rmse": rmse,
                    "mae": rmse * 0.8,
                    "r2": 1.0 - rmse,
                    "n_train_dropped": 0,
                    "n_validation_dropped": 0,
                }
            )
    return pd.DataFrame(rows, columns=ALIGNMENT_RESULT_COLUMNS)


def test_comparison_sign_convention():
    results = _results_fixture({0: [1.00, 1.00, 1.00], 1: [0.90, 1.05, 0.95]})
    comparison = compare_with_baseline_alignment(results)

    assert set(comparison["alignment_hours"]) == {1}
    by_fold = comparison.set_index("fold_id")
    assert by_fold.loc[1, "rmse_difference"] == pytest.approx(-0.10)  # shift better
    assert bool(by_fold.loc[1, "alignment_better"]) is True
    assert by_fold.loc[2, "rmse_difference"] == pytest.approx(0.05)  # shift worse
    assert bool(by_fold.loc[2, "alignment_better"]) is False


def test_aggregate_summary_reports_mean_and_spread():
    results = _results_fixture({0: [1.00, 1.10, 1.20], 1: [0.90, 0.90, 0.90]})
    summary = summarize_alignments(results).set_index("alignment_hours")

    assert summary.loc[0, "rmse_mean"] == pytest.approx(1.10)
    assert summary.loc[1, "rmse_mean"] == pytest.approx(0.90)
    assert summary.loc[1, "rmse_std"] == pytest.approx(0.0)
    assert (summary["n_folds"] == 3).all()


def test_consistency_requires_improvement_on_every_fold():
    """One good fold is not evidence. The rule asks for all three."""
    results = _results_fixture(
        {
            0: [1.00, 1.00, 1.00],
            1: [0.50, 1.05, 1.05],  # one big win, two losses
            2: [0.95, 0.95, 0.95],  # smaller but consistent
        }
    )
    consistency = assess_consistency(
        compare_with_baseline_alignment(results), meaningful_rmse=0.01
    ).set_index("alignment_hours")

    assert consistency.loc[1, "n_folds_improved"] == 1
    assert bool(consistency.loc[1, "improves_on_every_fold"]) is False
    assert bool(consistency.loc[1, "consistent_and_meaningful"]) is False
    # Its mean difference is favourable, which is exactly the trap the rule exists to catch.
    assert consistency.loc[1, "mean_rmse_difference"] < 0

    assert consistency.loc[2, "n_folds_improved"] == 3
    assert bool(consistency.loc[2, "consistent_and_meaningful"]) is True


def test_consistent_but_negligible_improvement_is_not_meaningful():
    results = _results_fixture({0: [1.00, 1.00, 1.00], 1: [0.999, 0.999, 0.999]})
    consistency = assess_consistency(
        compare_with_baseline_alignment(results), meaningful_rmse=0.01
    ).set_index("alignment_hours")

    assert bool(consistency.loc[1, "improves_on_every_fold"]) is True
    assert bool(consistency.loc[1, "consistent_and_meaningful"]) is False


def test_best_supported_alignment_defaults_to_the_current_one():
    results = _results_fixture({0: [1.00, 1.00, 1.00], 1: [1.05, 1.05, 1.05]})
    consistency = assess_consistency(compare_with_baseline_alignment(results))
    assert best_supported_alignment(consistency) == BASELINE_ALIGNMENT_HOURS

    results = _results_fixture({0: [1.00, 1.00, 1.00], 1: [0.90, 0.90, 0.90]})
    consistency = assess_consistency(compare_with_baseline_alignment(results))
    assert best_supported_alignment(consistency) == 1


def _matched_fixture(rmse_by_alignment):
    rows = []
    for alignment_hours, values in rmse_by_alignment.items():
        for fold_id, rmse in zip((1, 2, 3), values):
            rows.append(
                {
                    "fold_id": fold_id,
                    "alignment_hours": alignment_hours,
                    "n_common": 100,
                    "rmse": rmse,
                    "mae": rmse * 0.8,
                    "r2": 1.0 - rmse,
                    "rmse_difference": rmse - rmse_by_alignment[BASELINE_ALIGNMENT_HOURS][
                        fold_id - 1
                    ],
                }
            )
    return pd.DataFrame(rows)


def test_matched_consistency_excludes_the_reference_alignment():
    matched = _matched_fixture({0: [1.00, 1.00, 1.00], 1: [0.90, 0.90, 0.90]})
    rule = matched_consistency(matched)

    assert set(rule["alignment_hours"]) == {1}
    assert bool(rule.iloc[0]["consistent_and_meaningful"]) is True


def test_improvement_that_disappears_on_common_hours_is_not_acted_on():
    """A shift can look better simply by dropping the hours it finds hard.
    Equalizing the row sets is what separates that from a real gain."""
    headline = assess_consistency(
        compare_with_baseline_alignment(
            _results_fixture({0: [1.00, 1.00, 1.00], 1: [0.95, 0.95, 0.95]})
        )
    )
    assert bool(headline.iloc[0]["consistent_and_meaningful"]) is True

    # On the hours both alignments score, the shift wins one fold and loses two.
    matched = matched_consistency(
        _matched_fixture({0: [1.00, 1.00, 1.00], 1: [1.02, 0.95, 1.01]})
    )
    assert bool(matched.iloc[0]["consistent_and_meaningful"]) is False

    assert best_supported_alignment(headline, matched) == BASELINE_ALIGNMENT_HOURS
    # Without the matched evidence the headline alone would have selected it.
    assert best_supported_alignment(headline) == 1


def test_alignment_supported_by_both_rules_is_selected():
    headline = assess_consistency(
        compare_with_baseline_alignment(
            _results_fixture({0: [1.00, 1.00, 1.00], 1: [0.90, 0.90, 0.90]})
        )
    )
    matched = matched_consistency(
        _matched_fixture({0: [1.00, 1.00, 1.00], 1: [0.92, 0.91, 0.93]})
    )
    assert best_supported_alignment(headline, matched) == 1


def test_verify_reproduces_committed_benchmark_flags_a_drift():
    results = _results_fixture({0: [1.00, 1.00, 1.00]})
    committed = pd.DataFrame(
        {
            "fold_id": [1, 2, 3],
            "rmse": [1.00, 1.00, 1.00],
            "mae": [0.80, 0.80, 0.80],
            "r2": [0.0, 0.0, 0.0],
            "n_train": [1000, 1000, 1000],
            "n_validation": [100, 100, 100],
        }
    )
    verify_reproduces_committed_benchmark(results, committed)

    drifted = committed.copy()
    drifted.loc[0, "rmse"] = 0.95
    with pytest.raises(ValueError, match="committed Random Forest benchmark recorded"):
        verify_reproduces_committed_benchmark(results, drifted)


# ---------------------------------------------------------------------
# Evaluation on Spark
# ---------------------------------------------------------------------


def test_zero_hour_alignment_reproduces_a_direct_random_forest_run(spark):
    """The control arm must be the existing benchmark, not a near copy of it."""
    hourly, assignment = default_fixture()

    direct, _ = random_forest_evaluate_model(spark, hourly, assignment, fold_ids=(1,))
    aligned = evaluate_alignment(spark, hourly, assignment, 0, fold_ids=(1,))

    assert list(aligned.results.columns) == ALIGNMENT_RESULT_COLUMNS
    for metric in ("rmse", "mae", "r2"):
        assert float(aligned.results[metric].iloc[0]) == pytest.approx(
            float(direct[metric].iloc[0]), abs=1e-12
        )
    assert int(aligned.results["n_train"].iloc[0]) == int(direct["n_train"].iloc[0])
    assert int(aligned.results["n_validation"].iloc[0]) == int(direct["n_validation"].iloc[0])


def test_shifted_alignment_scores_fewer_rows_than_the_control(spark):
    hourly, assignment = default_fixture()
    control = evaluate_alignment(spark, hourly, assignment, 0, fold_ids=(1,))
    shifted = evaluate_alignment(spark, hourly, assignment, 2, fold_ids=(1,))

    assert int(shifted.results["n_validation"].iloc[0]) == int(
        control.results["n_validation"].iloc[0]
    ) - 2
    assert int(shifted.results["n_train"].iloc[0]) == int(control.results["n_train"].iloc[0]) - 2
    assert int(shifted.results["n_validation_dropped"].iloc[0]) == 2
    assert int(shifted.results["n_train_dropped"].iloc[0]) == 2
    assert int(shifted.results["n_scored"].iloc[0]) == int(shifted.results["n_validation"].iloc[0])


def test_evaluation_is_reproducible(spark):
    hourly, assignment = default_fixture()
    verify_reproducible(spark, hourly, assignment, 1, fold_id=1)

    first = evaluate_alignment(spark, hourly, assignment, 1, fold_ids=(1,))
    second = evaluate_alignment(spark, hourly, assignment, 1, fold_ids=(1,))
    for metric in ("rmse", "mae", "r2"):
        assert float(first.results[metric].iloc[0]) == pytest.approx(
            float(second.results[metric].iloc[0]), abs=1e-9
        )
    assert first.results["n_train"].tolist() == second.results["n_train"].tolist()


def test_matched_comparison_uses_only_common_hours(spark):
    hourly, assignment = default_fixture()
    results, evaluations = run_experiment(
        spark, hourly, assignment, alignments=(0, 2), fold_ids=(1,)
    )
    matched = matched_comparison(evaluations)

    assert set(matched["alignment_hours"]) == {0, 2}
    # The 2 hour arm scores two fewer hours, so the common set is its size.
    expected_common = int(
        results.loc[results["alignment_hours"] == 2, "n_scored"].iloc[0]
    )
    assert (matched["n_common"] == expected_common).all()
    reference = matched[matched["alignment_hours"] == 0].iloc[0]
    assert reference["rmse_difference"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------
# Real data integration (skips cleanly if artifacts are absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_alignment_datasets_are_leakage_free():
    """Runs the full guard set on the real split without fitting anything."""
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)

    committed_counts = {1: (1743, 482), 2: (2224, 483), 3: (2708, 481)}

    for alignment_hours in ALIGNMENT_HOURS:
        dataset = build_aligned_dataset(hourly, assignment, alignment_hours)
        validate_alignment(dataset, hourly, assignment)

        loss = dataset.row_loss.set_index(["fold_id", "role"])
        for fold_id, (n_train, n_validation) in committed_counts.items():
            assert int(loss.loc[(fold_id, ROLE_TRAIN), "n_committed"]) == n_train
            assert int(loss.loc[(fold_id, ROLE_VALIDATION), "n_committed"]) == n_validation
            if alignment_hours == 0:
                assert int(loss.loc[(fold_id, ROLE_TRAIN), "n_dropped"]) == 0
                assert int(loss.loc[(fold_id, ROLE_VALIDATION), "n_dropped"]) == 0
            else:
                # A shift can only cost rows, and never more than a few percent.
                assert 0 < int(loss.loc[(fold_id, ROLE_TRAIN), "n_dropped"]) < n_train * 0.05
                assert 0 < int(loss.loc[(fold_id, ROLE_VALIDATION), "n_dropped"]) < (
                    n_validation * 0.10
                )


@pytest.mark.skipif(
    not (REAL_ARTIFACTS and REAL_RANDOM_FOREST.exists()),
    reason="processed artifacts not available locally",
)
def test_real_data_zero_hour_arm_reproduces_the_committed_benchmark(spark):
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)

    evaluation = evaluate_alignment(spark, hourly, assignment, 0)
    verify_reproduces_committed_benchmark(
        evaluation.results, pd.read_parquet(REAL_RANDOM_FOREST)
    )

    assert evaluation.results["n_train"].tolist() == [1743, 2224, 2708]
    assert evaluation.results["n_validation"].tolist() == [482, 483, 481]
    assert (evaluation.results["n_features"] == 57).all()

    test_hours = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    for result in evaluation.fold_results:
        assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_hours)
