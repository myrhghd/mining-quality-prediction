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
    ALL_PREDICTOR_COLUMNS,
    CORE_SENSOR_PREDICTOR_COLUMNS,
    EXCLUDED_OUTCOME_COLUMN,
    FEED_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.data.split import (
    FEED_ELIGIBLE_COLUMN,
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
from src.models import random_forest
from src.models.baselines import INTERPOLATED_COLUMN
from src.models.linear_regression import build_spark_session
from src.models.feed_chemistry_value import (
    COLLAPSE_TOLERANCE_SD,
    FEED_ENHANCED,
    MEANINGFUL_RMSE,
    RESULT_COLUMNS,
    SENSOR_ONLY,
    VALUE_MODERATE,
    VALUE_NONE,
    VALUE_STRONG,
    assert_feed_columns_are_usable,
    assert_matched_rows,
    classify_feed_value,
    combine_results,
    compare_configurations,
    compare_excursions,
    describe_feed_columns,
    evaluate_configuration,
    evaluate_fold,
    excursion_analysis,
    feed_eligible_timestamps,
    feed_importance_table,
    get_feed_enhanced_predictors,
    get_sensor_only_predictors,
    prediction_spread,
    restrict_assignment,
    summarize_configurations,
    summarize_row_loss,
    top_importance_table,
    validate_configuration_scope,
    validate_evaluation,
    verify_deterministic_evaluation,
    verify_feed_values_match_the_raw_record,
    verify_random_forest_configuration_unchanged,
    verify_reproduces_committed_benchmark,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_RANDOM_FOREST = REPO_ROOT / "data" / "processed" / "random_forest_results.parquet"
REAL_RAW = REPO_ROOT / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"
REAL_ARTIFACTS = all(path.exists() for path in (REAL_HOURLY, REAL_SPLITS, REAL_RANDOM_FOREST))


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityFeedChemistryTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(
    n_rows: int = 200, seed: int = 0, start: str = "2020-01-01", feed_signal: float = 0.0
) -> pd.DataFrame:
    """Hourly table carrying the 57 sensor aggregates and both feed columns.

    `feed_signal` scales how much of the target comes from feed chemistry,
    so the same fixture serves both a null case and a positive control.
    """
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    frame["iron_feed"] = rng.normal(loc=56.0, scale=3.0, size=n_rows)
    frame["silica_feed"] = rng.normal(loc=14.0, scale=4.0, size=n_rows)

    frame[TARGET_COLUMN] = (
        2.0
        + 1.5 * frame[predictors[0]]
        - 0.8 * frame[predictors[1]]
        + feed_signal * (frame["silica_feed"] - 14.0)
        + rng.normal(scale=0.2, size=n_rows)
    )
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    frame[SEGMENT_COLUMN] = 0
    frame[SENSOR_ELIGIBLE_COLUMN] = True
    frame[FEED_ELIGIBLE_COLUMN] = True
    frame[INTERPOLATED_COLUMN] = False
    for column in FEED_COLUMNS:
        frame[f"{column}_inconsistent"] = False
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


def default_fixture(n_rows: int = 200, feed_signal: float = 0.0, feed_missing=()):
    """Train 0..119, embargo 120..129, validation 130..179, test 180..199."""
    hourly = make_hourly(n_rows=n_rows, feed_signal=feed_signal)
    for position in feed_missing:
        hourly.loc[position, FEED_ELIGIBLE_COLUMN] = False
        for column in FEED_COLUMNS:
            hourly.loc[position, column] = np.nan
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 120)),
        embargo_idx=list(range(120, 130)),
        validation_idx=list(range(130, 180)),
        test_idx=list(range(180, n_rows)),
    )
    return hourly, assignment


def matched_fixture(**kwargs):
    hourly, assignment = default_fixture(**kwargs)
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
    return hourly, restricted, assignment


def results_frame(configuration: str, rmse, mae=None, r2=None, n_features: int = 57):
    mae = rmse if mae is None else mae
    r2 = [0.0] * len(rmse) if r2 is None else r2
    return pd.DataFrame(
        {
            "configuration": configuration,
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


def comparison_for(sensor_rmse, feed_rmse):
    combined = pd.concat(
        [
            results_frame(SENSOR_ONLY, sensor_rmse),
            results_frame(FEED_ENHANCED, feed_rmse, n_features=59),
        ],
        ignore_index=True,
    )
    return compare_configurations(combined)


# ---------------------------------------------------------------------
# Predictor schemas
# ---------------------------------------------------------------------


def test_sensor_only_schema_is_exactly_fifty_seven():
    predictors = get_sensor_only_predictors()
    assert len(predictors) == 57
    assert predictors == list(CORE_SENSOR_PREDICTOR_COLUMNS)
    validate_configuration_scope(SENSOR_ONLY, predictors)


def test_feed_enhanced_schema_is_exactly_fifty_nine():
    predictors = get_feed_enhanced_predictors()
    assert len(predictors) == 59
    assert predictors[:57] == list(CORE_SENSOR_PREDICTOR_COLUMNS)
    assert predictors[57:] == list(FEED_CONTEXT_PREDICTOR_COLUMNS)
    assert predictors[57:] == ["iron_feed", "silica_feed"]
    assert predictors == list(ALL_PREDICTOR_COLUMNS)
    validate_configuration_scope(FEED_ENHANCED, predictors)


def test_the_two_schemas_differ_only_by_the_feed_columns():
    difference = set(get_feed_enhanced_predictors()) - set(get_sensor_only_predictors())
    assert difference == {"iron_feed", "silica_feed"}
    assert not set(get_sensor_only_predictors()) - set(get_feed_enhanced_predictors())


def test_sensor_only_arm_rejects_the_feed_columns():
    with pytest.raises(ValueError):
        validate_configuration_scope(SENSOR_ONLY, get_feed_enhanced_predictors())


def test_feed_enhanced_arm_rejects_a_missing_feed_column():
    with pytest.raises(ValueError, match="Missing"):
        validate_configuration_scope(FEED_ENHANCED, get_sensor_only_predictors() + ["iron_feed"])


@pytest.mark.parametrize("configuration", [SENSOR_ONLY, FEED_ENHANCED])
def test_iron_concentrate_is_rejected_by_both_arms(configuration):
    predictors = list(get_feed_enhanced_predictors())
    if configuration == SENSOR_ONLY:
        predictors = list(get_sensor_only_predictors())
    with pytest.raises(ValueError):
        validate_configuration_scope(configuration, predictors + [EXCLUDED_OUTCOME_COLUMN])


@pytest.mark.parametrize(
    "extra",
    [
        "iron_concentrate",
        "silica_concentrate_first",
        "target_run_id",
        "hours_since_target_change",
        "month",
        "day_of_week",
        "hour_of_day",
        "is_interpolated",
    ],
)
def test_forbidden_columns_are_rejected(extra):
    with pytest.raises(ValueError):
        validate_configuration_scope(FEED_ENHANCED, get_feed_enhanced_predictors() + [extra])


@pytest.mark.parametrize(
    "history_feature",
    [
        "starch_flow_change_1h",
        "amina_flow_trailing_120m_mean",
        "ore_pulp_ph_trailing_120m_std",
        "starch_flow_trailing_15m_mean",
        "ore_pulp_flow_trailing_30m_slope",
        "amina_flow_trailing_60m_max",
    ],
)
def test_dynamic_history_features_are_rejected(history_feature):
    with pytest.raises(ValueError, match="Dynamic sensor history"):
        validate_configuration_scope(FEED_ENHANCED, get_feed_enhanced_predictors() + [history_feature])


def test_duplicates_are_rejected():
    predictors = get_feed_enhanced_predictors()
    predictors[5] = predictors[6]
    with pytest.raises(ValueError, match="duplicates"):
        validate_configuration_scope(FEED_ENHANCED, predictors)


def test_an_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration"):
        validate_configuration_scope("feed_only", get_feed_enhanced_predictors())


# ---------------------------------------------------------------------
# Feed column provenance
# ---------------------------------------------------------------------


def test_feed_description_names_the_raw_source_columns():
    hourly, _ = default_fixture()
    described = describe_feed_columns(hourly).set_index("column")
    assert described.loc["iron_feed", "raw_source"] == "% Iron Feed"
    assert described.loc["silica_feed", "raw_source"] == "% Silica Feed"
    assert int(described.loc["iron_feed", "n_missing"]) == 0


def test_usability_guard_accepts_a_clean_table():
    hourly, _ = default_fixture()
    assert_feed_columns_are_usable(hourly)


def test_usability_guard_rejects_the_presence_of_iron_concentrate():
    hourly, _ = default_fixture()
    hourly[EXCLUDED_OUTCOME_COLUMN] = 66.0
    with pytest.raises(ValueError, match="must never be available"):
        assert_feed_columns_are_usable(hourly)


def test_usability_guard_rejects_an_inconsistent_hour():
    hourly, _ = default_fixture()
    hourly.loc[3, "iron_feed_inconsistent"] = True
    with pytest.raises(ValueError, match="more than one value"):
        assert_feed_columns_are_usable(hourly)


def test_usability_guard_rejects_a_missing_value_on_an_eligible_hour():
    hourly, _ = default_fixture()
    hourly.loc[3, "silica_feed"] = np.nan
    with pytest.raises(ValueError, match="missing on at least one feed eligible hour"):
        assert_feed_columns_are_usable(hourly)


def test_usability_guard_rejects_a_feed_column_copied_from_the_target():
    hourly, _ = default_fixture()
    hourly["silica_feed"] = hourly[TARGET_COLUMN]
    with pytest.raises(ValueError, match="identical to the target"):
        assert_feed_columns_are_usable(hourly)


def make_raw_for(hourly: pd.DataFrame, samples_per_hour: int = 4) -> pd.DataFrame:
    """A raw record whose feed values are constant within every hour."""
    repeated = hourly.loc[
        hourly.index.repeat(samples_per_hour), [TIMESTAMP_COLUMN, *FEED_COLUMNS]
    ]
    return repeated.reset_index(drop=True)


def test_raw_provenance_guard_accepts_unmodified_values():
    hourly, _ = default_fixture()
    verify_feed_values_match_the_raw_record(hourly, make_raw_for(hourly))


def test_raw_provenance_guard_rejects_a_transformed_hourly_value():
    hourly, _ = default_fixture()
    raw = make_raw_for(hourly)
    tampered = hourly.copy()
    tampered["silica_feed"] = tampered["silica_feed"] * 1.01
    with pytest.raises(ValueError, match="does not match the raw observation"):
        verify_feed_values_match_the_raw_record(tampered, raw)


def test_raw_provenance_guard_rejects_a_feed_value_that_varies_within_an_hour():
    hourly, _ = default_fixture()
    raw = make_raw_for(hourly)
    raw.loc[1, "iron_feed"] = raw.loc[1, "iron_feed"] + 5.0
    with pytest.raises(ValueError, match="varies within at least one recorded hour"):
        verify_feed_values_match_the_raw_record(hourly, raw)


# ---------------------------------------------------------------------
# Matched rows
# ---------------------------------------------------------------------


def test_no_rows_are_removed_when_every_hour_carries_feed_chemistry():
    hourly, assignment = default_fixture()
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
    assert len(restricted) == len(assignment)
    assert bool((summarize_row_loss(assignment, restricted)["n_dropped"] == 0).all())


def test_rows_without_feed_chemistry_are_removed_from_development():
    hourly, assignment = default_fixture(feed_missing=(5, 140))
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))

    dropped = set(assignment[TIMESTAMP_COLUMN]) - set(restricted[TIMESTAMP_COLUMN])
    assert dropped == {
        hourly[TIMESTAMP_COLUMN].iloc[5],
        hourly[TIMESTAMP_COLUMN].iloc[140],
    }

    loss = summarize_row_loss(assignment, restricted).set_index("role")["n_dropped"]
    assert int(loss.loc[ROLE_TRAIN]) == 1
    assert int(loss.loc[ROLE_VALIDATION]) == 1


def test_the_final_test_assignment_is_never_restricted():
    hourly, assignment = default_fixture(feed_missing=(5, 185))
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))

    before = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    after = set(restricted.loc[restricted["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    assert after == before


def test_restriction_never_invents_a_row():
    hourly, assignment = default_fixture(feed_missing=(7,))
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
    committed = set(zip(assignment["fold_id"], assignment[TIMESTAMP_COLUMN], assignment["role"]))
    produced = set(zip(restricted["fold_id"], restricted[TIMESTAMP_COLUMN], restricted["role"]))
    assert produced.issubset(committed)


# ---------------------------------------------------------------------
# Comparison and classification
# ---------------------------------------------------------------------


def test_comparison_signs_favour_feed_when_it_predicts_better():
    comparison = comparison_for([1.00, 1.00, 1.00], [0.90, 0.95, 0.99])
    assert comparison["rmse_difference"].tolist() == pytest.approx([-0.10, -0.05, -0.01])
    assert comparison["feed_better"].tolist() == [True, True, True]


def test_summary_reports_the_feature_count_of_each_arm():
    combined = pd.concat(
        [
            results_frame(SENSOR_ONLY, [1.0, 1.0, 1.0]),
            results_frame(FEED_ENHANCED, [0.9, 0.9, 0.9], n_features=59),
        ],
        ignore_index=True,
    )
    summary = summarize_configurations(combined).set_index("configuration")
    assert int(summary.loc[SENSOR_ONLY, "n_features"]) == 57
    assert int(summary.loc[FEED_ENHANCED, "n_features"]) == 59


def test_strong_value_requires_every_fold_and_a_material_margin():
    value = classify_feed_value(comparison_for([1.00, 1.00, 1.00], [0.90, 0.92, 0.95]))
    assert value["classification"] == VALUE_STRONG
    assert value["improves_on_every_fold"]


def test_an_improvement_below_the_margin_is_not_strong():
    value = classify_feed_value(comparison_for([1.00, 1.00, 1.00], [0.999, 0.998, 0.999]))
    assert value["classification"] == VALUE_MODERATE
    assert abs(value["mean_rmse_difference"]) < MEANINGFUL_RMSE


def test_an_inconsistent_improvement_is_moderate():
    value = classify_feed_value(comparison_for([1.00, 1.00, 1.00], [0.90, 1.02, 0.96]))
    assert value["classification"] == VALUE_MODERATE
    assert value["n_folds_improved"] == 2


def test_no_improvement_anywhere_is_little_or_none():
    value = classify_feed_value(comparison_for([1.00, 1.00, 1.00], [1.05, 1.02, 1.01]))
    assert value["classification"] == VALUE_NONE
    assert value["n_folds_improved"] == 0


def test_a_single_gain_swamped_by_losses_is_little_or_none():
    value = classify_feed_value(comparison_for([1.00, 1.00, 1.00], [0.98, 1.10, 1.08]))
    assert value["classification"] == VALUE_NONE


def test_classification_rejects_an_empty_comparison():
    with pytest.raises(ValueError, match="empty comparison"):
        classify_feed_value(comparison_for([1.0, 1.0, 1.0], [0.9, 0.9, 0.9]).iloc[:0])


def test_committed_benchmark_check_rejects_a_metric_mismatch():
    results = results_frame(SENSOR_ONLY, [1.0, 1.0, 1.0])
    committed = results.copy()
    committed.loc[1, "rmse"] = 1.5
    with pytest.raises(ValueError, match="committed Random Forest benchmark"):
        verify_reproduces_committed_benchmark(results, committed)


def test_committed_benchmark_check_rejects_a_row_count_mismatch():
    results = results_frame(SENSOR_ONLY, [1.0, 1.0, 1.0])
    committed = results.copy()
    committed.loc[0, "n_validation"] = 999
    with pytest.raises(ValueError, match="n_validation"):
        verify_reproduces_committed_benchmark(results, committed)


# ---------------------------------------------------------------------
# Spark backed evaluation
# ---------------------------------------------------------------------


def test_random_forest_configuration_is_unchanged(spark):
    observed = verify_random_forest_configuration_unchanged(get_feed_enhanced_predictors())
    assert observed == {
        "numTrees": random_forest.NUM_TREES,
        "maxDepth": random_forest.MAX_DEPTH,
        "minInstancesPerNode": random_forest.MIN_INSTANCES_PER_NODE,
        "featureSubsetStrategy": random_forest.FEATURE_SUBSET_STRATEGY,
        "subsamplingRate": random_forest.SUBSAMPLING_RATE,
        "seed": random_forest.SEED,
    }
    # The same configuration must be used by both arms.
    assert verify_random_forest_configuration_unchanged(get_sensor_only_predictors()) == observed


def test_evaluate_fold_rejects_the_final_test_fold(spark):
    hourly, restricted, _ = matched_fixture()
    with pytest.raises(ValueError, match="final test fold"):
        evaluate_fold(spark, hourly, restricted, FINAL_TEST_FOLD_ID, FEED_ENHANCED)


def test_both_arms_score_the_same_rows_with_the_expected_feature_counts(spark):
    hourly, restricted, committed = matched_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    assert_matched_rows(sensor, feed)
    assert int(sensor.results["n_features"].iloc[0]) == 57
    assert int(feed.results["n_features"].iloc[0]) == 59
    assert int(sensor.results["n_train"].iloc[0]) == 120
    assert int(sensor.results["n_validation"].iloc[0]) == 50

    for evaluation in (sensor, feed):
        validate_evaluation(evaluation, hourly, restricted, committed)
        assert len(evaluation.fold_results[0].feature_importances) == len(evaluation.predictors)


def test_matched_row_check_rejects_a_different_scored_population(spark):
    hourly, restricted, _ = matched_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))

    trimmed = restricted[
        restricted[TIMESTAMP_COLUMN] != hourly[TIMESTAMP_COLUMN].iloc[179]
    ]
    feed = evaluate_configuration(spark, hourly, trimmed, FEED_ENHANCED, fold_ids=(1,))
    with pytest.raises(ValueError, match="scored different hours"):
        assert_matched_rows(sensor, feed)


def test_no_final_test_hour_is_fitted_on_or_scored(spark):
    hourly, restricted, committed = matched_fixture()
    test_hours = set(committed.loc[committed["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    assert test_hours

    for configuration in (SENSOR_ONLY, FEED_ENHANCED):
        evaluation = evaluate_configuration(
            spark, hourly, restricted, configuration, fold_ids=(1,)
        )
        scored = set(evaluation.fold_results[0].predictions[TIMESTAMP_COLUMN])
        assert not scored.intersection(test_hours)
        validate_evaluation(evaluation, hourly, restricted, committed)


def test_no_embargo_hour_is_fitted_on_or_scored(spark):
    hourly, restricted, committed = matched_fixture()
    embargo_hours = set(
        restricted.loc[restricted["role"] == ROLE_EMBARGO, TIMESTAMP_COLUMN]
    )
    assert len(embargo_hours) == 10

    evaluation = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))
    scored = set(evaluation.fold_results[0].predictions[TIMESTAMP_COLUMN])
    assert not scored.intersection(embargo_hours)

    from src.models.baselines import get_fold_frames

    frames = get_fold_frames(hourly, restricted, 1)
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(embargo_hours)
    validate_evaluation(evaluation, hourly, restricted, committed)


def test_validation_rejects_an_embargo_hour_relabelled_as_training(spark):
    hourly, restricted, committed = matched_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    tampered = restricted.copy()
    embargo_row = tampered.index[tampered["role"] == ROLE_EMBARGO][0]
    tampered.loc[embargo_row, "role"] = ROLE_TRAIN
    with pytest.raises(ValueError, match="not committed to that role"):
        validate_evaluation(evaluation, hourly, tampered, committed)


def test_validation_rejects_a_feed_ineligible_row_in_the_feed_arm(spark):
    hourly, restricted, committed = matched_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    # Mark a scored hour feed ineligible after the fact; the guard must
    # notice that the feed arm used a row without feed chemistry.
    tampered = hourly.copy()
    tampered.loc[135, FEED_ELIGIBLE_COLUMN] = False
    with pytest.raises(ValueError, match="feed ineligible"):
        validate_evaluation(evaluation, tampered, restricted, committed)


def test_validation_rejects_an_interpolated_target_row(spark):
    hourly, restricted, committed = matched_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))

    tampered = hourly.copy()
    tampered.loc[140, INTERPOLATED_COLUMN] = True
    with pytest.raises(ValueError, match="interpolated target row"):
        validate_evaluation(evaluation, tampered, restricted, committed)


def test_evaluation_is_deterministic(spark):
    hourly, restricted, _ = matched_fixture()
    verify_deterministic_evaluation(spark, hourly, restricted, FEED_ENHANCED, fold_id=1)

    first = evaluate_fold(spark, hourly, restricted, 1, FEED_ENHANCED)
    second = evaluate_fold(spark, hourly, restricted, 1, FEED_ENHANCED)
    np.testing.assert_allclose(
        first.predictions["prediction"].to_numpy(),
        second.predictions["prediction"].to_numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        first.feature_importances, second.feature_importances, atol=1e-10
    )


def test_feed_chemistry_is_learned_when_the_target_depends_on_it(spark):
    """Positive control: the harness can benefit from feed chemistry.

    Without this check, a null result could equally mean the two extra
    columns never reached the model.
    """
    hourly, restricted, _ = matched_fixture(n_rows=400, feed_signal=1.2)
    restricted = restrict_assignment(
        make_assignment(
            hourly,
            train_idx=list(range(0, 250)),
            embargo_idx=list(range(250, 260)),
            validation_idx=list(range(260, 400)),
        ),
        feed_eligible_timestamps(hourly),
    )

    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))
    assert_matched_rows(sensor, feed)

    comparison = compare_configurations(combine_results(sensor, feed))
    assert float(comparison["rmse_difference"].iloc[0]) < 0
    assert float(comparison["r2_difference"].iloc[0]) > 0

    importance = feed_importance_table(feed).set_index("feature")
    assert float(importance.loc["silica_feed", "importance"]) > 0
    assert int(importance.loc["silica_feed", "n_features"]) == 59


def test_feed_importance_table_reports_a_rank_within_the_full_predictor_set(spark):
    hourly, restricted, _ = matched_fixture()
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    table = feed_importance_table(feed)
    assert set(table["feature"]) == {"iron_feed", "silica_feed"}
    assert bool((table["rank"] >= 1).all())
    assert bool((table["rank"] <= 59).all())
    assert bool((table["n_features"] == 59).all())

    top = top_importance_table(feed, top_n=5)
    assert len(top) == 5
    assert top["rank"].tolist() == [1, 2, 3, 4, 5]
    assert top["importance"].is_monotonic_decreasing


def test_feed_importance_is_empty_for_the_sensor_only_arm(spark):
    hourly, restricted, _ = matched_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    assert feed_importance_table(sensor).empty


def test_analysis_tables_cover_every_group(spark):
    hourly, restricted, _ = matched_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    excursions = excursion_analysis(sensor, feed, hourly, restricted)
    assert set(excursions["configuration"]) == {SENSOR_ONLY, FEED_ENHANCED}
    assert set(excursions["group"]) == {"excursion", "normal"}
    # Both arms share the training derived threshold.
    assert excursions["threshold"].nunique() == 1

    from src.models.baselines import get_fold_frames

    frames = get_fold_frames(hourly, restricted, 1)
    assert float(excursions["threshold"].iloc[0]) == pytest.approx(
        float(frames.train[TARGET_COLUMN].quantile(0.90))
    )

    assert len(compare_excursions(excursions)) == 2

    spread = pd.concat(
        [
            prediction_spread(sensor, hourly, restricted),
            prediction_spread(feed, hourly, restricted),
        ]
    )
    assert bool((spread["prediction_std"] >= 0).all())
    assert bool((spread["share_within_quarter_sd_of_train_mean"].between(0.0, 1.0)).all())
    assert bool((spread["rmse_over_constant_rmse"] > 0).all())


# ---------------------------------------------------------------------
# Real artifacts
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_feed_columns_pass_the_usability_guard():
    hourly = pd.read_parquet(REAL_HOURLY)
    assert_feed_columns_are_usable(hourly)
    assert EXCLUDED_OUTCOME_COLUMN not in hourly.columns

    described = describe_feed_columns(hourly)
    assert int(described["n_missing"].sum()) == 0
    assert int(described["n_hours_inconsistent"].sum()) == 0


@pytest.mark.skipif(
    not (REAL_ARTIFACTS and REAL_RAW.exists()), reason="raw dataset not available locally"
)
def test_real_feed_values_match_the_raw_record():
    from src.data.preprocess import load_raw

    hourly = pd.read_parquet(REAL_HOURLY)
    verify_feed_values_match_the_raw_record(hourly, load_raw(REAL_RAW))


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_feed_eligibility_matches_sensor_eligibility():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)

    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
    loss = summarize_row_loss(assignment, restricted)
    assert int(loss["n_dropped"].sum()) == 0

    development = assignment[assignment["fold_kind"] == KIND_DEVELOPMENT]
    feed_ok = feed_eligible_timestamps(hourly)
    assert set(development[TIMESTAMP_COLUMN]).issubset(feed_ok)


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_development_rows_never_touch_the_final_test_period():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))

    test_hours = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    development = restricted[restricted["fold_kind"] == KIND_DEVELOPMENT]
    assert not set(development[TIMESTAMP_COLUMN]).intersection(test_hours)

    before = assignment[assignment["fold_kind"] == KIND_FINAL_TEST]
    after = restricted[restricted["fold_kind"] == KIND_FINAL_TEST]
    assert len(after) == len(before)
