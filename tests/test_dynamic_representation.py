from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dynamic_features import (
    DYNAMIC_PREDICTOR_COLUMNS,
    HAS_CONTEXT_COLUMN,
    HIGH_FREQUENCY_COLUMNS,
    SEGMENT_COLUMN as DYNAMIC_SEGMENT_COLUMN,
    SENSOR_VALID_COLUMN,
    build_dynamic_features,
)
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
from src.models.dynamic_representation import (
    DYNAMIC_REPRESENTATION,
    MEANINGFUL_RMSE,
    RESULT_COLUMNS,
    STATIC_REPRESENTATION,
    SUPPORT_NONE,
    SUPPORT_STRONG,
    SUPPORT_WEAK,
    assert_features_are_backward_looking,
    assert_matched_rows,
    build_matched_dataset,
    classify_support,
    combine_results,
    compare_excursions,
    compare_representations,
    evaluate_fold,
    evaluate_representation,
    excursion_analysis,
    get_dynamic_predictors,
    get_static_predictors,
    join_dynamic_features,
    prediction_spread,
    restrict_assignment,
    subperiod_comparison,
    summarize_representations,
    summarize_row_loss,
    validate_evaluation,
    validate_representation_scope,
    verify_deterministic_evaluation,
    verify_reproduces_committed_benchmark,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_DYNAMIC = REPO_ROOT / "data" / "processed" / "dynamic_features.parquet"
REAL_RANDOM_FOREST = REPO_ROOT / "data" / "processed" / "random_forest_results.parquet"
REAL_ARTIFACTS = all(
    path.exists() for path in (REAL_HOURLY, REAL_SPLITS, REAL_DYNAMIC, REAL_RANDOM_FOREST)
)

SAMPLES_PER_HOUR = 180


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityDynamicRepresentationTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(n_rows: int = 200, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Hourly table with all 57 static sensor predictors on a contiguous grid."""
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    frame[TARGET_COLUMN] = (
        2.0
        + 1.5 * frame[predictors[0]]
        - 0.8 * frame[predictors[1]]
        + rng.normal(scale=0.2, size=n_rows)
    )
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    frame[SEGMENT_COLUMN] = 0
    frame[SENSOR_ELIGIBLE_COLUMN] = True
    frame[INTERPOLATED_COLUMN] = False
    for column in FEED_CONTEXT_PREDICTOR_COLUMNS:
        frame[column] = 50.0
    return frame


def make_dynamic(hourly: pd.DataFrame, seed: int = 1, without_history=()) -> pd.DataFrame:
    """Dynamic feature table matching an hourly fixture hour for hour.

    `without_history` names positions whose 120 minute window is treated
    as unavailable, mirroring a segment opening or a frozen sensor hour.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(hourly)
    frame = pd.DataFrame({TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].to_numpy()})
    for column in DYNAMIC_PREDICTOR_COLUMNS:
        frame[column] = rng.normal(size=n_rows)

    has_context = np.ones(n_rows, dtype=bool)
    for position in without_history:
        has_context[position] = False
    frame[HAS_CONTEXT_COLUMN] = has_context
    frame.loc[~has_context, DYNAMIC_PREDICTOR_COLUMNS] = np.nan
    return frame


def make_assignment(hourly, train_idx, embargo_idx, validation_idx, test_idx=(), fold_id=1):
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
                    "fold_id": fold_id,
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


def default_fixture(without_history=(), n_rows: int = 200):
    """Train 0..119, embargo 120..129, validation 130..179, test 180..199."""
    hourly = make_hourly(n_rows=n_rows)
    dynamic = make_dynamic(hourly, without_history=without_history)
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 120)),
        embargo_idx=list(range(120, 130)),
        validation_idx=list(range(130, 180)),
        test_idx=list(range(180, n_rows)),
    )
    return hourly, dynamic, assignment


def results_frame(representation: str, rmse, mae=None, r2=None, n_features: int = 57):
    mae = rmse if mae is None else mae
    r2 = [0.0] * len(rmse) if r2 is None else r2
    return pd.DataFrame(
        {
            "representation": representation,
            "fold_id": [1, 2, 3],
            "n_train": [100, 200, 300],
            "n_validation": [50, 50, 50],
            "n_scored": [50, 50, 50],
            "n_features": n_features,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        columns=RESULT_COLUMNS,
    )


def comparison_for(static_rmse, dynamic_rmse):
    combined = pd.concat(
        [
            results_frame(STATIC_REPRESENTATION, static_rmse),
            results_frame(DYNAMIC_REPRESENTATION, dynamic_rmse, n_features=153),
        ],
        ignore_index=True,
    )
    return compare_representations(combined)


# ---------------------------------------------------------------------
# Predictor scope
# ---------------------------------------------------------------------


def test_static_arm_is_the_committed_fifty_seven_predictors():
    predictors = get_static_predictors()
    assert predictors == list(CORE_SENSOR_PREDICTOR_COLUMNS)
    assert len(predictors) == 57
    validate_representation_scope(STATIC_REPRESENTATION, predictors)


def test_dynamic_arm_is_the_static_set_followed_by_the_history_features():
    predictors = get_dynamic_predictors()
    assert len(predictors) == 153
    assert predictors[:57] == list(CORE_SENSOR_PREDICTOR_COLUMNS)
    assert predictors[57:] == list(DYNAMIC_PREDICTOR_COLUMNS)
    validate_representation_scope(DYNAMIC_REPRESENTATION, predictors)


@pytest.mark.parametrize(
    "extra",
    ["iron_feed", "silica_feed", "iron_concentrate", "month", "day_of_week", "hour_of_day"],
)
def test_scope_guard_rejects_forbidden_columns(extra):
    predictors = get_dynamic_predictors() + [extra]
    with pytest.raises(ValueError):
        validate_representation_scope(DYNAMIC_REPRESENTATION, predictors)


def test_scope_guard_rejects_duplicates():
    predictors = get_dynamic_predictors()
    predictors[10] = predictors[11]
    with pytest.raises(ValueError, match="duplicates"):
        validate_representation_scope(DYNAMIC_REPRESENTATION, predictors)


def test_scope_guard_rejects_a_missing_history_feature():
    predictors = get_dynamic_predictors()[:-1]
    with pytest.raises(ValueError, match="Missing"):
        validate_representation_scope(DYNAMIC_REPRESENTATION, predictors)


def test_scope_guard_rejects_the_history_features_on_the_static_arm():
    with pytest.raises(ValueError):
        validate_representation_scope(STATIC_REPRESENTATION, get_dynamic_predictors())


def test_scope_guard_rejects_an_unknown_representation():
    with pytest.raises(ValueError, match="Unknown representation"):
        validate_representation_scope("hybrid", get_static_predictors())


# ---------------------------------------------------------------------
# Joining and row matching
# ---------------------------------------------------------------------


def test_join_attaches_every_history_column_without_changing_the_row_count():
    hourly, dynamic, _ = default_fixture()
    joined = join_dynamic_features(hourly, dynamic)
    assert len(joined) == len(hourly)
    assert set(DYNAMIC_PREDICTOR_COLUMNS).issubset(joined.columns)
    assert joined[TIMESTAMP_COLUMN].is_monotonic_increasing


def test_join_pairs_each_hour_with_its_own_feature_vector():
    hourly, dynamic, _ = default_fixture()
    shuffled = dynamic.sample(frac=1.0, random_state=3).reset_index(drop=True)
    joined = join_dynamic_features(hourly, shuffled)

    reference = dynamic.set_index(TIMESTAMP_COLUMN)
    for timestamp in hourly[TIMESTAMP_COLUMN].iloc[[0, 5, 60, 199]]:
        row = joined[joined[TIMESTAMP_COLUMN] == timestamp].iloc[0]
        np.testing.assert_allclose(
            row[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
            reference.loc[timestamp, DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_join_rejects_an_incomplete_dynamic_table():
    hourly, dynamic, _ = default_fixture()
    with pytest.raises(ValueError, match="no dynamic feature record"):
        join_dynamic_features(hourly, dynamic.iloc[:-5])


def test_join_rejects_a_missing_history_column():
    hourly, dynamic, _ = default_fixture()
    with pytest.raises(ValueError, match="missing column"):
        join_dynamic_features(hourly, dynamic.drop(columns=[DYNAMIC_PREDICTOR_COLUMNS[0]]))


def test_restriction_drops_only_the_hours_without_history():
    hourly, dynamic, assignment = default_fixture(without_history=(3, 135))
    matched = build_matched_dataset(hourly, dynamic, assignment)

    dropped = set(assignment[TIMESTAMP_COLUMN]) - set(matched.assignment[TIMESTAMP_COLUMN])
    assert dropped == {
        hourly[TIMESTAMP_COLUMN].iloc[3],
        hourly[TIMESTAMP_COLUMN].iloc[135],
    }


def test_restriction_leaves_the_final_test_assignment_untouched():
    hourly, dynamic, assignment = default_fixture(without_history=(3, 185))
    matched = build_matched_dataset(hourly, dynamic, assignment)

    before = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    after = set(matched.assignment.loc[matched.assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    assert after == before


def test_restriction_never_invents_a_row():
    hourly, dynamic, assignment = default_fixture(without_history=(7,))
    usable = set(hourly[TIMESTAMP_COLUMN]) - {hourly[TIMESTAMP_COLUMN].iloc[7]}
    restricted = restrict_assignment(assignment, usable)

    committed = set(zip(assignment["fold_id"], assignment[TIMESTAMP_COLUMN], assignment["role"]))
    produced = set(
        zip(restricted["fold_id"], restricted[TIMESTAMP_COLUMN], restricted["role"])
    )
    assert produced.issubset(committed)
    assert len(restricted) < len(assignment)


def test_row_loss_summary_counts_the_dropped_rows_per_role():
    hourly, dynamic, assignment = default_fixture(without_history=(3, 4, 135))
    matched = build_matched_dataset(hourly, dynamic, assignment)
    loss = summarize_row_loss(assignment, matched.assignment)

    by_role = loss.set_index("role")["n_dropped"]
    assert int(by_role.loc[ROLE_TRAIN]) == 2
    assert int(by_role.loc[ROLE_VALIDATION]) == 1
    assert int(by_role.loc[ROLE_EMBARGO]) == 0


def test_matched_dataset_keeps_only_hours_with_history():
    hourly, dynamic, assignment = default_fixture(without_history=(3, 135))
    matched = build_matched_dataset(hourly, dynamic, assignment)

    development = matched.assignment[matched.assignment["fold_kind"] == KIND_DEVELOPMENT]
    context = matched.hourly.set_index(TIMESTAMP_COLUMN)[HAS_CONTEXT_COLUMN]
    assert bool(context.loc[development[TIMESTAMP_COLUMN]].all())


def test_assert_matched_rows_rejects_different_scored_hours():
    hourly, dynamic, assignment = default_fixture()
    matched = build_matched_dataset(hourly, dynamic, assignment)
    frames = matched.hourly

    from src.models.dynamic_representation import (
        RepresentationEvaluation,
        RepresentationFoldResult,
    )

    def evaluation(representation, timestamps):
        predictions = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: timestamps,
                TARGET_COLUMN: np.arange(len(timestamps), dtype=float),
                "prediction": 1.0,
            }
        )
        result = RepresentationFoldResult(
            representation=representation,
            fold_id=1,
            metrics={"rmse": 1.0, "mae": 1.0, "r2": 0.0},
            n_train=10,
            n_validation=len(timestamps),
            n_scored=len(timestamps),
            n_features=57,
            predictions=predictions,
        )
        return RepresentationEvaluation(
            representation=representation,
            predictors=[],
            results=pd.DataFrame(),
            fold_results=[result],
        )

    hours = frames[TIMESTAMP_COLUMN].iloc[130:180].to_numpy()
    with pytest.raises(ValueError, match="scored different hours"):
        assert_matched_rows(
            evaluation(STATIC_REPRESENTATION, hours),
            evaluation(DYNAMIC_REPRESENTATION, hours[:-1]),
        )


# ---------------------------------------------------------------------
# Backward looking guard
# ---------------------------------------------------------------------


def make_raw_and_hourly(n_hours: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A raw record and the minimal hourly table the feature builder reads."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    raw = pd.DataFrame({TIMESTAMP_COLUMN: np.repeat(hours.to_numpy(), SAMPLES_PER_HOUR)})
    for offset, column in enumerate(HIGH_FREQUENCY_COLUMNS):
        raw[column] = rng.normal(loc=100.0 + offset, scale=5.0, size=len(raw))

    grouped = raw.groupby(TIMESTAMP_COLUMN)
    hourly = pd.DataFrame(
        {f"{column}_mean": grouped[column].mean() for column in HIGH_FREQUENCY_COLUMNS}
    )
    hourly[DYNAMIC_SEGMENT_COLUMN] = 0
    hourly[SENSOR_VALID_COLUMN] = True
    hourly.index.name = TIMESTAMP_COLUMN
    return raw, hourly.reset_index()


def test_backward_looking_guard_accepts_the_real_construction():
    raw, hourly = make_raw_and_hourly()
    dynamic = build_dynamic_features(raw, hourly)
    assert_features_are_backward_looking(hourly, dynamic, raw)


def test_backward_looking_guard_rejects_a_forward_looking_table():
    raw, hourly = make_raw_and_hourly()
    dynamic = build_dynamic_features(raw, hourly)

    # A table whose values were shifted backward in time carries the next
    # hour's measurements, which is exactly what the guard must catch.
    forward = dynamic.copy()
    forward[DYNAMIC_PREDICTOR_COLUMNS] = (
        dynamic[DYNAMIC_PREDICTOR_COLUMNS].shift(-1).to_numpy()
    )
    with pytest.raises(ValueError, match="reads the future"):
        assert_features_are_backward_looking(hourly, forward, raw)


# ---------------------------------------------------------------------
# Comparison, classification, analysis
# ---------------------------------------------------------------------


def test_comparison_signs_favour_the_dynamic_arm_when_it_predicts_better():
    comparison = comparison_for([1.00, 1.00, 1.00], [0.90, 0.95, 0.99])
    assert comparison["rmse_difference"].tolist() == pytest.approx([-0.10, -0.05, -0.01])
    assert comparison["dynamic_better"].tolist() == [True, True, True]


def test_comparison_reports_a_worse_dynamic_arm_as_worse():
    comparison = comparison_for([1.00, 1.00, 1.00], [1.10, 1.05, 1.01])
    assert bool((comparison["rmse_difference"] > 0).all())
    assert comparison["dynamic_better"].tolist() == [False, False, False]


def test_summary_reports_the_feature_count_of_each_arm():
    combined = pd.concat(
        [
            results_frame(STATIC_REPRESENTATION, [1.0, 1.0, 1.0]),
            results_frame(DYNAMIC_REPRESENTATION, [0.9, 0.9, 0.9], n_features=153),
        ],
        ignore_index=True,
    )
    summary = summarize_representations(combined).set_index("representation")
    assert int(summary.loc[STATIC_REPRESENTATION, "n_features"]) == 57
    assert int(summary.loc[DYNAMIC_REPRESENTATION, "n_features"]) == 153


def test_strong_support_requires_every_fold_and_a_material_margin():
    support = classify_support(comparison_for([1.00, 1.00, 1.00], [0.90, 0.92, 0.95]))
    assert support["classification"] == SUPPORT_STRONG
    assert support["improves_on_every_fold"]
    assert support["n_folds_improved"] == 3


def test_an_improvement_on_every_fold_below_the_margin_is_not_strong():
    support = classify_support(comparison_for([1.00, 1.00, 1.00], [0.999, 0.998, 0.999]))
    assert support["classification"] == SUPPORT_WEAK
    assert support["improves_on_every_fold"]
    assert abs(support["mean_rmse_difference"]) < MEANINGFUL_RMSE


def test_an_inconsistent_improvement_is_mixed():
    support = classify_support(comparison_for([1.00, 1.00, 1.00], [0.90, 1.02, 0.96]))
    assert support["classification"] == SUPPORT_WEAK
    assert support["n_folds_improved"] == 2


def test_no_improvement_anywhere_is_no_support():
    support = classify_support(comparison_for([1.00, 1.00, 1.00], [1.05, 1.02, 1.01]))
    assert support["classification"] == SUPPORT_NONE
    assert support["n_folds_improved"] == 0


def test_a_single_fold_gain_swamped_by_losses_is_no_support():
    support = classify_support(comparison_for([1.00, 1.00, 1.00], [0.98, 1.10, 1.08]))
    assert support["classification"] == SUPPORT_NONE


def test_classification_rejects_an_empty_comparison():
    empty = comparison_for([1.0, 1.0, 1.0], [0.9, 0.9, 0.9]).iloc[:0]
    with pytest.raises(ValueError, match="empty comparison"):
        classify_support(empty)


def test_committed_benchmark_check_rejects_a_metric_mismatch():
    results = results_frame(STATIC_REPRESENTATION, [1.0, 1.0, 1.0])
    committed = results.rename(columns={"representation": "model"}).copy()
    committed.loc[1, "rmse"] = 1.5
    with pytest.raises(ValueError, match="committed Random Forest benchmark"):
        verify_reproduces_committed_benchmark(results, committed)


def test_committed_benchmark_check_rejects_a_row_count_mismatch():
    results = results_frame(STATIC_REPRESENTATION, [1.0, 1.0, 1.0])
    committed = results.copy()
    committed.loc[0, "n_train"] = 999
    with pytest.raises(ValueError, match="n_train"):
        verify_reproduces_committed_benchmark(results, committed)


# ---------------------------------------------------------------------
# Spark backed evaluation
# ---------------------------------------------------------------------


def matched_fixture(without_history=(3, 135)):
    hourly, dynamic, assignment = default_fixture(without_history=without_history)
    return build_matched_dataset(hourly, dynamic, assignment), assignment


def test_evaluate_fold_rejects_the_final_test_fold(spark):
    matched, _ = matched_fixture()
    with pytest.raises(ValueError, match="final test fold"):
        evaluate_fold(
            spark,
            matched.hourly,
            matched.assignment,
            FINAL_TEST_FOLD_ID,
            STATIC_REPRESENTATION,
        )


def test_both_arms_score_the_same_rows_with_different_feature_counts(spark):
    matched, committed = matched_fixture()
    static = evaluate_representation(
        spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION, fold_ids=(1,)
    )
    dynamic = evaluate_representation(
        spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION, fold_ids=(1,)
    )

    assert_matched_rows(static, dynamic)
    assert int(static.results["n_features"].iloc[0]) == 57
    assert int(dynamic.results["n_features"].iloc[0]) == 153
    assert int(static.results["n_train"].iloc[0]) == 119
    assert int(static.results["n_validation"].iloc[0]) == 49

    for evaluation in (static, dynamic):
        validate_evaluation(evaluation, matched.hourly, matched.assignment, committed)


def test_no_final_test_hour_is_fitted_on_or_scored(spark):
    matched, committed = matched_fixture()
    test_hours = set(committed.loc[committed["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    assert test_hours

    for representation in (STATIC_REPRESENTATION, DYNAMIC_REPRESENTATION):
        evaluation = evaluate_representation(
            spark, matched.hourly, matched.assignment, representation, fold_ids=(1,)
        )
        result = evaluation.fold_results[0]
        assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_hours)
        validate_evaluation(evaluation, matched.hourly, matched.assignment, committed)


def test_validation_rejects_a_row_that_was_never_committed_to_that_role(spark):
    matched, committed = matched_fixture()
    evaluation = evaluate_representation(
        spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION, fold_ids=(1,)
    )
    # Relabel one embargo hour as training, which the committed split
    # never authorized.
    tampered = matched.assignment.copy()
    embargo_row = tampered.index[tampered["role"] == ROLE_EMBARGO][0]
    tampered.loc[embargo_row, "role"] = ROLE_TRAIN

    with pytest.raises(ValueError, match="not committed to that role"):
        validate_evaluation(evaluation, matched.hourly, tampered, committed)


def test_evaluation_is_deterministic(spark):
    matched, _ = matched_fixture()
    verify_deterministic_evaluation(
        spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION, fold_id=1
    )

    first = evaluate_fold(
        spark, matched.hourly, matched.assignment, 1, DYNAMIC_REPRESENTATION
    )
    second = evaluate_fold(
        spark, matched.hourly, matched.assignment, 1, DYNAMIC_REPRESENTATION
    )
    np.testing.assert_allclose(
        first.predictions["prediction"].to_numpy(),
        second.predictions["prediction"].to_numpy(),
        atol=1e-10,
    )


def test_the_dynamic_arm_learns_a_signal_carried_only_by_process_history(spark):
    """Positive control: the harness can benefit from a history feature.

    The target is built from the history features alone, so the static
    arm has nothing to fit and the dynamic arm should be clearly better.
    Without this check, a null result could equally mean the wider
    feature vector never reaches the model.
    """
    hourly = make_hourly(n_rows=400, seed=5)
    dynamic = make_dynamic(hourly, seed=6)

    informative = [f"{column}_change_1h" for column in HIGH_FREQUENCY_COLUMNS]
    signal = dynamic[informative].to_numpy().sum(axis=1)
    hourly = hourly.copy()
    hourly[TARGET_COLUMN] = 2.0 + signal

    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 250)),
        embargo_idx=list(range(250, 260)),
        validation_idx=list(range(260, 400)),
    )
    matched = build_matched_dataset(hourly, dynamic, assignment)

    static = evaluate_representation(
        spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION, fold_ids=(1,)
    )
    dynamic_arm = evaluate_representation(
        spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION, fold_ids=(1,)
    )
    assert_matched_rows(static, dynamic_arm)

    comparison = compare_representations(combine_results(static, dynamic_arm))
    assert float(comparison["rmse_difference"].iloc[0]) < 0
    assert float(comparison["r2_difference"].iloc[0]) > 0


def test_analysis_tables_cover_every_fold_and_group(spark):
    matched, _ = matched_fixture()
    static = evaluate_representation(
        spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION, fold_ids=(1,)
    )
    dynamic = evaluate_representation(
        spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION, fold_ids=(1,)
    )

    excursions = excursion_analysis(static, dynamic, matched.hourly, matched.assignment)
    assert set(excursions["representation"]) == {STATIC_REPRESENTATION, DYNAMIC_REPRESENTATION}
    assert set(excursions["group"]) == {"excursion", "normal"}
    # The threshold comes from the training window, so both arms share it.
    assert excursions["threshold"].nunique() == 1

    comparison = compare_excursions(excursions)
    assert len(comparison) == 2

    spread = pd.concat([prediction_spread(static), prediction_spread(dynamic)])
    assert bool((spread["prediction_std"] >= 0).all())
    assert bool((spread["prediction_range"] >= 0).all())

    blocks = subperiod_comparison(static, dynamic)
    assert len(blocks) == 4
    assert int(blocks["n"].sum()) == int(static.results["n_validation"].iloc[0])


def test_excursion_threshold_uses_the_training_window_only(spark):
    matched, _ = matched_fixture()
    static = evaluate_representation(
        spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION, fold_ids=(1,)
    )
    excursions = excursion_analysis(static, static, matched.hourly, matched.assignment)

    from src.models.baselines import get_fold_frames

    frames = get_fold_frames(matched.hourly, matched.assignment, 1)
    expected = float(frames.train[TARGET_COLUMN].quantile(0.90))
    assert float(excursions["threshold"].iloc[0]) == pytest.approx(expected)


# ---------------------------------------------------------------------
# Real artifacts
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_matched_dataset_removes_only_a_handful_of_hours():
    hourly = pd.read_parquet(REAL_HOURLY)
    dynamic = pd.read_parquet(REAL_DYNAMIC)
    assignment = pd.read_parquet(REAL_SPLITS)

    matched = build_matched_dataset(hourly, dynamic, assignment)
    development = assignment[assignment["fold_kind"] == KIND_DEVELOPMENT]
    retained = matched.assignment[matched.assignment["fold_kind"] == KIND_DEVELOPMENT]

    assert len(retained) < len(development)
    assert len(retained) / len(development) > 0.99

    # Every retained development hour carries a complete history window.
    context = matched.hourly.set_index(TIMESTAMP_COLUMN)[HAS_CONTEXT_COLUMN]
    assert bool(context.loc[retained[TIMESTAMP_COLUMN]].all())


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_restriction_never_touches_the_final_test_period():
    hourly = pd.read_parquet(REAL_HOURLY)
    dynamic = pd.read_parquet(REAL_DYNAMIC)
    assignment = pd.read_parquet(REAL_SPLITS)

    matched = build_matched_dataset(hourly, dynamic, assignment)
    before = assignment[assignment["fold_kind"] == KIND_FINAL_TEST]
    after = matched.assignment[matched.assignment["fold_kind"] == KIND_FINAL_TEST]
    assert len(after) == len(before)

    test_hours = set(before.loc[before["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    development = matched.assignment[matched.assignment["fold_kind"] == KIND_DEVELOPMENT]
    assert not set(development[TIMESTAMP_COLUMN]).intersection(test_hours)


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_fold_training_always_precedes_its_validation():
    hourly = pd.read_parquet(REAL_HOURLY)
    dynamic = pd.read_parquet(REAL_DYNAMIC)
    assignment = pd.read_parquet(REAL_SPLITS)
    matched = build_matched_dataset(hourly, dynamic, assignment)

    from src.models.baselines import DEVELOPMENT_FOLD_IDS, get_fold_frames

    for fold_id in DEVELOPMENT_FOLD_IDS:
        frames = get_fold_frames(matched.hourly, matched.assignment, fold_id)
        assert frames.train[TIMESTAMP_COLUMN].max() < frames.validation[TIMESTAMP_COLUMN].min()
        assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(
            frames.validation[TIMESTAMP_COLUMN]
        )
