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
from src.models.feed_chemistry_value import (
    FEED_ENHANCED,
    SENSOR_ONLY,
    feed_eligible_timestamps,
    restrict_assignment,
)
from src.models.linear_regression import build_spark_session
from src.models.excursion_classifier import (
    DECISION_THRESHOLD,
    EXCURSION_QUANTILE,
    IMPURITY,
    LABEL_COLUMN,
    POSITIVE_PROBABILITY_COLUMN,
    RESULT_COLUMNS,
    OUTCOME_NONE,
    OUTCOME_USEFUL,
    OUTCOME_WEAK,
    apply_excursion_label,
    assert_matched_rows_and_labels,
    average_precision,
    average_ranks,
    baseline_comparison,
    baseline_table,
    build_labelled_fold,
    classification_metrics,
    classify_outcome,
    combine_results,
    compare_configurations,
    confusion_counts,
    evaluate_configuration,
    evaluate_fold,
    feed_importance_table,
    probability_distributions,
    probability_separation,
    roc_auc,
    summarize_configurations,
    top_importance_table,
    training_excursion_threshold,
    validate_evaluation,
    verify_classifier_configuration,
    verify_deterministic_evaluation,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_SPLITS.exists()


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityExcursionClassifierTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(
    n_rows: int = 400, seed: int = 0, start: str = "2020-01-01", feed_signal: float = 0.0
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
        + 0.6 * frame[predictors[0]]
        + feed_signal * (frame["silica_feed"] - 14.0)
        + rng.normal(scale=0.3, size=n_rows)
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


def default_fixture(n_rows: int = 400, feed_signal: float = 0.0):
    """Train 0..249, embargo 250..259, validation 260..359, test 360..399."""
    hourly = make_hourly(n_rows=n_rows, feed_signal=feed_signal)
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 250)),
        embargo_idx=list(range(250, 260)),
        validation_idx=list(range(260, 360)),
        test_idx=list(range(360, n_rows)),
    )
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
    return hourly, restricted, assignment


# ---------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------


def test_threshold_comes_from_the_training_window_only():
    hourly, restricted, _ = default_fixture()
    labelled = build_labelled_fold(hourly, restricted, 1)

    from src.models.baselines import get_fold_frames

    frames = get_fold_frames(hourly, restricted, 1)
    assert labelled.threshold == pytest.approx(
        float(frames.train[TARGET_COLUMN].quantile(EXCURSION_QUANTILE))
    )


def test_changing_only_validation_targets_does_not_move_the_threshold():
    hourly, restricted, _ = default_fixture()
    baseline = build_labelled_fold(hourly, restricted, 1).threshold

    perturbed = hourly.copy()
    perturbed.loc[260:359, TARGET_COLUMN] = perturbed.loc[260:359, TARGET_COLUMN] * 10.0 + 50.0
    assert build_labelled_fold(perturbed, restricted, 1).threshold == pytest.approx(baseline)


def test_changing_only_final_test_targets_does_not_move_the_threshold():
    hourly, restricted, _ = default_fixture()
    baseline = build_labelled_fold(hourly, restricted, 1).threshold

    perturbed = hourly.copy()
    perturbed.loc[360:, TARGET_COLUMN] = 99.0
    assert build_labelled_fold(perturbed, restricted, 1).threshold == pytest.approx(baseline)


def test_changing_only_embargo_targets_does_not_move_the_threshold():
    hourly, restricted, _ = default_fixture()
    baseline = build_labelled_fold(hourly, restricted, 1).threshold

    perturbed = hourly.copy()
    perturbed.loc[250:259, TARGET_COLUMN] = 99.0
    assert build_labelled_fold(perturbed, restricted, 1).threshold == pytest.approx(baseline)


def test_validation_labels_use_the_training_threshold_unchanged():
    hourly, restricted, _ = default_fixture()
    labelled = build_labelled_fold(hourly, restricted, 1)

    expected = (
        labelled.validation[TARGET_COLUMN].to_numpy() >= labelled.threshold
    ).astype(int)
    np.testing.assert_array_equal(labelled.validation[LABEL_COLUMN].to_numpy(), expected)

    # The validation positive rate is free to differ from 10 percent; only
    # the training window is pinned by construction.
    train_rate = labelled.train[LABEL_COLUMN].mean()
    assert train_rate == pytest.approx(1.0 - EXCURSION_QUANTILE, abs=0.02)


def test_labels_are_inclusive_at_the_threshold():
    frame = pd.DataFrame({TARGET_COLUMN: [1.0, 2.0, 3.0]})
    labelled = apply_excursion_label(frame, threshold=2.0)
    assert labelled[LABEL_COLUMN].tolist() == [0, 1, 1]


def test_threshold_rejects_an_empty_training_window():
    with pytest.raises(ValueError, match="empty training window"):
        training_excursion_threshold(pd.DataFrame({TARGET_COLUMN: []}))


def test_threshold_rejects_an_out_of_range_quantile():
    frame = pd.DataFrame({TARGET_COLUMN: [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        training_excursion_threshold(frame, quantile=1.0)


# ---------------------------------------------------------------------
# Metric correctness on known examples
# ---------------------------------------------------------------------


def test_confusion_counts_on_a_hand_worked_example():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.4, 0.7, 0.1, 0.5, 0.2])
    counts = confusion_counts(labels, scores)
    assert counts == {
        "true_positives": 2,  # 0.9 and 0.5
        "false_negatives": 1,  # 0.4
        "false_positives": 1,  # 0.7
        "true_negatives": 2,  # 0.1 and 0.2
    }
    assert sum(counts.values()) == labels.size


def test_the_decision_threshold_is_inclusive_at_exactly_one_half():
    counts = confusion_counts(np.array([1, 0]), np.array([0.5, 0.5]))
    assert counts["true_positives"] == 1
    assert counts["false_positives"] == 1


def test_classification_metrics_on_a_hand_worked_example():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.4, 0.7, 0.1, 0.5, 0.2])
    metrics = classification_metrics(labels, scores)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_negative_rate"] == pytest.approx(1 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_perfect_ranking_scores_one_on_both_curves():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_reversed_ranking_scores_zero_roc_auc():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_constant_scores_give_a_coin_flip_roc_and_prevalence_pr():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = np.full(labels.size, 0.3)
    assert roc_auc(labels, scores) == pytest.approx(0.5)
    assert average_precision(labels, scores) == pytest.approx(labels.mean())


def test_average_precision_on_a_hand_worked_example():
    # Ranked: 0.9(pos), 0.8(neg), 0.7(pos), 0.6(neg)
    # Precision at each positive: 1/1 and 2/3; each adds 0.5 recall.
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert average_precision(labels, scores) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_roc_auc_on_a_hand_worked_example():
    # Positives 0.9 and 0.7; negatives 0.8 and 0.6.
    # Pairs won: (0.9>0.8), (0.9>0.6), (0.7>0.6). Lost: (0.7<0.8). 3/4.
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_tied_scores_count_as_half_a_pair():
    labels = np.array([1, 0])
    scores = np.array([0.5, 0.5])
    assert roc_auc(labels, scores) == pytest.approx(0.5)


def test_average_ranks_share_ties():
    np.testing.assert_allclose(average_ranks([10.0, 20.0, 20.0, 40.0]), [1.0, 2.5, 2.5, 4.0])
    np.testing.assert_allclose(average_ranks([5.0, 5.0, 5.0]), [2.0, 2.0, 2.0])


def test_precision_is_zero_when_nothing_is_flagged():
    labels = np.array([1, 0, 0, 0])
    scores = np.array([0.4, 0.1, 0.2, 0.3])
    metrics = classification_metrics(labels, scores)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_negative_rate"] == 1.0


def test_metrics_reject_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classification_metrics(np.array([1, 0]), np.array([1.5, 0.2]))


def test_metrics_reject_a_non_binary_label():
    with pytest.raises(ValueError, match="Labels must be 0 or 1"):
        classification_metrics(np.array([2, 0]), np.array([0.5, 0.2]))


def test_ranking_metrics_reject_a_single_class_window():
    with pytest.raises(ValueError, match="only one class"):
        roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="without a positive observation"):
        average_precision(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------
# Predictor schemas
# ---------------------------------------------------------------------


def test_sensor_schema_is_exactly_fifty_seven():
    from src.models.feed_chemistry_value import get_sensor_only_predictors

    predictors = get_sensor_only_predictors()
    assert len(predictors) == 57
    assert predictors == list(CORE_SENSOR_PREDICTOR_COLUMNS)


def test_feed_schema_is_exactly_fifty_nine():
    from src.models.feed_chemistry_value import get_feed_enhanced_predictors

    predictors = get_feed_enhanced_predictors()
    assert len(predictors) == 59
    assert predictors == list(ALL_PREDICTOR_COLUMNS)
    assert predictors[57:] == ["iron_feed", "silica_feed"]


@pytest.mark.parametrize("configuration", [SENSOR_ONLY, FEED_ENHANCED])
def test_iron_concentrate_is_rejected(configuration):
    from src.models.feed_chemistry_value import (
        get_feed_enhanced_predictors,
        get_sensor_only_predictors,
        validate_configuration_scope,
    )

    predictors = (
        get_sensor_only_predictors()
        if configuration == SENSOR_ONLY
        else get_feed_enhanced_predictors()
    )
    with pytest.raises(ValueError):
        validate_configuration_scope(configuration, predictors + [EXCLUDED_OUTCOME_COLUMN])


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
    from src.models.feed_chemistry_value import (
        get_feed_enhanced_predictors,
        validate_configuration_scope,
    )

    with pytest.raises(ValueError, match="Dynamic sensor history"):
        validate_configuration_scope(
            FEED_ENHANCED, get_feed_enhanced_predictors() + [history_feature]
        )


# ---------------------------------------------------------------------
# Baselines and outcome classification
# ---------------------------------------------------------------------


def fake_evaluation(configuration, per_fold):
    """Build an evaluation from explicit label and score arrays per fold."""
    from src.models.excursion_classifier import ClassifierEvaluation, ClassifierFoldResult

    fold_results = []
    for fold_id, (labels, scores) in per_fold.items():
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        predictions = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: pd.date_range("2020-01-01", periods=labels.size, freq="h"),
                TARGET_COLUMN: labels.astype(float),
                LABEL_COLUMN: labels,
                POSITIVE_PROBABILITY_COLUMN: scores,
            }
        )
        fold_results.append(
            ClassifierFoldResult(
                configuration=configuration,
                fold_id=fold_id,
                threshold=1.0,
                metrics=classification_metrics(labels, scores),
                n_train=1000,
                n_validation=labels.size,
                n_features=59,
                n_train_positive=100,
                n_validation_positive=int(labels.sum()),
                feature_importances=np.zeros(59),
                predictions=predictions,
            )
        )
    return ClassifierEvaluation(
        configuration=configuration,
        predictors=list(ALL_PREDICTOR_COLUMNS),
        results=pd.DataFrame(),
        fold_results=fold_results,
    )


def perfect_fold(n: int = 100, n_positive: int = 10):
    labels = np.array([1] * n_positive + [0] * (n - n_positive))
    scores = np.array([0.9] * n_positive + [0.1] * (n - n_positive))
    return labels, scores


def no_skill_fold(n: int = 100, n_positive: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_positive + [0] * (n - n_positive))
    return labels, rng.random(n) * 0.2


def test_always_normal_baseline_flags_nothing():
    evaluation = fake_evaluation(SENSOR_ONLY, {1: no_skill_fold()})
    table = baseline_table(evaluation)
    row = table[table["baseline"] == "always_normal"].iloc[0]
    assert row["true_positives"] == 0
    assert row["false_positives"] == 0
    assert row["recall"] == 0.0
    assert row["false_negative_rate"] == 1.0
    assert row["accuracy"] == pytest.approx(0.90)
    assert row["pr_auc"] == pytest.approx(0.10)
    assert row["roc_auc"] == pytest.approx(0.5)


def test_training_majority_baseline_reports_the_majority_class():
    evaluation = fake_evaluation(SENSOR_ONLY, {1: no_skill_fold()})
    table = baseline_table(evaluation)
    majority = table[table["baseline"].str.startswith("training_majority")].iloc[0]
    # 100 of 1000 training rows are positive, so the majority class is 0.
    assert "class 0" in majority["baseline"]
    assert majority["true_positives"] == 0
    assert majority["recall"] == 0.0


def test_baseline_counts_sum_to_the_validation_size():
    evaluation = fake_evaluation(SENSOR_ONLY, {1: no_skill_fold()})
    table = baseline_table(evaluation)
    for _, row in table[table["baseline"] != "no_skill"].iterrows():
        total = (
            row["true_negatives"]
            + row["false_positives"]
            + row["false_negatives"]
            + row["true_positives"]
        )
        assert total == row["n_validation"]


def test_no_skill_pr_auc_equals_prevalence():
    evaluation = fake_evaluation(SENSOR_ONLY, {1: no_skill_fold(n=200, n_positive=30)})
    row = baseline_table(evaluation)
    no_skill = row[row["baseline"] == "no_skill"].iloc[0]
    assert no_skill["pr_auc"] == pytest.approx(30 / 200)
    assert no_skill["roc_auc"] == pytest.approx(0.5)


def test_a_perfect_classifier_is_reported_as_useful():
    evaluation = fake_evaluation(FEED_ENHANCED, {1: perfect_fold(), 2: perfect_fold(), 3: perfect_fold()})
    outcome = classify_outcome(evaluation)
    assert outcome["classification"] == OUTCOME_USEFUL
    assert outcome["n_folds_operationally_useful"] == 3


def test_a_no_skill_classifier_is_reported_as_useless():
    evaluation = fake_evaluation(
        SENSOR_ONLY,
        {1: no_skill_fold(seed=1), 2: no_skill_fold(seed=2), 3: no_skill_fold(seed=3)},
    )
    outcome = classify_outcome(evaluation)
    assert outcome["classification"] == OUTCOME_NONE
    assert outcome["n_folds_beating_no_skill_pr"] == 0


def test_a_classifier_that_ranks_well_but_never_flags_is_weak():
    """Good ranking with every probability below 0.50 is not operational."""
    labels = np.array([1] * 10 + [0] * 90)
    scores = np.array([0.45] * 10 + [0.05] * 90)
    evaluation = fake_evaluation(
        FEED_ENHANCED, {1: (labels, scores), 2: (labels, scores), 3: (labels, scores)}
    )
    outcome = classify_outcome(evaluation)
    assert outcome["classification"] == OUTCOME_WEAK
    assert outcome["n_folds_beating_no_skill_pr"] == 3
    assert outcome["n_folds_operationally_useful"] == 0


def test_baseline_comparison_reports_lift_over_prevalence():
    evaluation = fake_evaluation(FEED_ENHANCED, {1: perfect_fold()})
    comparison = baseline_comparison(evaluation).iloc[0]
    assert comparison["prevalence"] == pytest.approx(0.10)
    assert comparison["pr_auc"] == pytest.approx(1.0)
    assert comparison["pr_auc_lift"] == pytest.approx(0.90)
    assert comparison["roc_auc_lift"] == pytest.approx(0.50)


def test_probability_separation_is_zero_when_the_classes_overlap_completely():
    labels = np.array([1] * 10 + [0] * 90)
    scores = np.full(100, 0.1)
    evaluation = fake_evaluation(SENSOR_ONLY, {1: (labels, scores)})
    separation = probability_separation(evaluation).iloc[0]
    assert separation["mean_difference"] == pytest.approx(0.0)
    assert separation["ks_distance"] == pytest.approx(0.0)


def test_probability_distributions_cover_both_true_classes():
    evaluation = fake_evaluation(SENSOR_ONLY, {1: perfect_fold()})
    table = probability_distributions(evaluation)
    assert set(table["true_class"]) == {"positive", "negative"}
    assert int(table["n"].sum()) == 100


def test_comparison_reports_the_direction_of_each_metric():
    sensor = fake_evaluation(SENSOR_ONLY, {1: no_skill_fold(seed=1)})
    feed = fake_evaluation(FEED_ENHANCED, {1: perfect_fold()})
    sensor_results = pd.DataFrame(
        [
            {
                "configuration": SENSOR_ONLY,
                "fold_id": 1,
                **{key: sensor.fold_results[0].metrics[key] for key in
                   ("recall", "false_negative_rate", "f1", "pr_auc", "roc_auc", "precision")},
            }
        ]
    )
    feed_results = pd.DataFrame(
        [
            {
                "configuration": FEED_ENHANCED,
                "fold_id": 1,
                **{key: feed.fold_results[0].metrics[key] for key in
                   ("recall", "false_negative_rate", "f1", "pr_auc", "roc_auc", "precision")},
            }
        ]
    )
    comparison = compare_configurations(pd.concat([sensor_results, feed_results]))
    row = comparison.iloc[0]
    assert bool(row["feed_improves_recall"])
    assert bool(row["feed_improves_pr_auc"])
    assert bool(row["feed_reduces_false_negative_rate"])
    assert row["false_negative_rate_difference"] < 0


# ---------------------------------------------------------------------
# Spark backed evaluation
# ---------------------------------------------------------------------


def test_classifier_configuration_is_fixed_and_documented(spark):
    from src.models.feed_chemistry_value import (
        get_feed_enhanced_predictors,
        get_sensor_only_predictors,
    )

    observed = verify_classifier_configuration(get_feed_enhanced_predictors())
    assert observed["numTrees"] == random_forest.NUM_TREES
    assert observed["maxDepth"] == random_forest.MAX_DEPTH
    assert observed["minInstancesPerNode"] == random_forest.MIN_INSTANCES_PER_NODE
    assert observed["featureSubsetStrategy"] == random_forest.FEATURE_SUBSET_STRATEGY
    assert observed["subsamplingRate"] == random_forest.SUBSAMPLING_RATE
    assert observed["seed"] == random_forest.SEED
    # Variance splitting is undefined for a class label, so this one value
    # necessarily differs from the regression benchmark.
    assert observed["impurity"] == IMPURITY == "gini"
    assert observed["impurity"] != "variance"
    assert verify_classifier_configuration(get_sensor_only_predictors()) == observed


def test_evaluate_fold_rejects_the_final_test_fold(spark):
    hourly, restricted, _ = default_fixture()
    with pytest.raises(ValueError, match="final test fold"):
        evaluate_fold(spark, hourly, restricted, FINAL_TEST_FOLD_ID, FEED_ENHANCED)


def test_probabilities_are_valid_and_counts_add_up(spark):
    hourly, restricted, committed = default_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    result = evaluation.fold_results[0]

    probabilities = result.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy()
    assert np.isfinite(probabilities).all()
    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    counts = sum(
        result.metrics[key]
        for key in ("true_negatives", "false_positives", "false_negatives", "true_positives")
    )
    assert int(counts) == result.n_validation == 100
    validate_evaluation(evaluation, hourly, restricted, committed)


def test_both_arms_use_matched_rows_labels_and_thresholds(spark):
    hourly, restricted, committed = default_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    assert_matched_rows_and_labels(sensor, feed)
    assert int(sensor.results["n_features"].iloc[0]) == 57
    assert int(feed.results["n_features"].iloc[0]) == 59
    assert sensor.fold_results[0].threshold == pytest.approx(feed.fold_results[0].threshold)

    for evaluation in (sensor, feed):
        validate_evaluation(evaluation, hourly, restricted, committed)


def test_matched_check_rejects_different_labels(spark):
    hourly, restricted, _ = default_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    feed.fold_results[0].predictions.loc[0, LABEL_COLUMN] = (
        1 - feed.fold_results[0].predictions.loc[0, LABEL_COLUMN]
    )
    with pytest.raises(ValueError, match="different labels"):
        assert_matched_rows_and_labels(sensor, feed)


def test_no_final_test_hour_is_fitted_on_or_scored(spark):
    hourly, restricted, committed = default_fixture()
    test_hours = set(committed.loc[committed["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    assert test_hours

    for configuration in (SENSOR_ONLY, FEED_ENHANCED):
        evaluation = evaluate_configuration(
            spark, hourly, restricted, configuration, fold_ids=(1,)
        )
        scored = set(evaluation.fold_results[0].predictions[TIMESTAMP_COLUMN])
        assert not scored.intersection(test_hours)
        validate_evaluation(evaluation, hourly, restricted, committed)


def test_final_test_assignment_is_untouched_by_the_restriction():
    hourly, restricted, committed = default_fixture()
    before = committed[committed["fold_kind"] == KIND_FINAL_TEST]
    after = restricted[restricted["fold_kind"] == KIND_FINAL_TEST]
    assert len(after) == len(before)
    assert set(after[TIMESTAMP_COLUMN]) == set(before[TIMESTAMP_COLUMN])


def test_no_embargo_hour_is_fitted_on_or_scored(spark):
    hourly, restricted, committed = default_fixture()
    embargo_hours = set(restricted.loc[restricted["role"] == ROLE_EMBARGO, TIMESTAMP_COLUMN])
    assert len(embargo_hours) == 10

    evaluation = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))
    scored = set(evaluation.fold_results[0].predictions[TIMESTAMP_COLUMN])
    assert not scored.intersection(embargo_hours)

    labelled = build_labelled_fold(hourly, restricted, 1)
    assert not set(labelled.train[TIMESTAMP_COLUMN]).intersection(embargo_hours)
    validate_evaluation(evaluation, hourly, restricted, committed)


def test_validation_rejects_an_embargo_hour_relabelled_as_training(spark):
    hourly, restricted, committed = default_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    tampered = restricted.copy()
    embargo_row = tampered.index[tampered["role"] == ROLE_EMBARGO][0]
    tampered.loc[embargo_row, "role"] = ROLE_TRAIN
    with pytest.raises(ValueError, match="not committed to that role"):
        validate_evaluation(evaluation, hourly, tampered, committed)


def test_validation_rejects_an_interpolated_target_row(spark):
    hourly, restricted, committed = default_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))

    tampered = hourly.copy()
    tampered.loc[300, INTERPOLATED_COLUMN] = True
    with pytest.raises(ValueError, match="interpolated target row"):
        validate_evaluation(evaluation, tampered, restricted, committed)


def test_validation_rejects_labels_that_ignore_the_training_threshold(spark):
    hourly, restricted, committed = default_fixture()
    evaluation = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))

    predictions = evaluation.fold_results[0].predictions
    predictions.loc[0, LABEL_COLUMN] = 1 - predictions.loc[0, LABEL_COLUMN]
    with pytest.raises(ValueError, match="do not follow the training derived threshold"):
        validate_evaluation(evaluation, hourly, restricted, committed)


def test_evaluation_is_deterministic(spark):
    hourly, restricted, _ = default_fixture()
    verify_deterministic_evaluation(spark, hourly, restricted, FEED_ENHANCED, fold_id=1)

    first = evaluate_fold(spark, hourly, restricted, 1, FEED_ENHANCED)
    second = evaluate_fold(spark, hourly, restricted, 1, FEED_ENHANCED)
    np.testing.assert_allclose(
        first.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(),
        second.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(first.feature_importances, second.feature_importances, atol=1e-12)


def test_feed_chemistry_is_learned_when_it_determines_the_label(spark):
    """Positive control: the classifier can exploit feed chemistry.

    The target is driven by `silica_feed`, so the label is largely a
    function of it. Without this check, a null result could equally mean
    the two extra columns never reached the model.
    """
    hourly, restricted, _ = default_fixture(feed_signal=3.0)

    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))
    assert_matched_rows_and_labels(sensor, feed)

    assert feed.fold_results[0].metrics["pr_auc"] > sensor.fold_results[0].metrics["pr_auc"]
    assert feed.fold_results[0].metrics["roc_auc"] > sensor.fold_results[0].metrics["roc_auc"]
    assert feed.fold_results[0].metrics["roc_auc"] > 0.80

    importance = feed_importance_table(feed).set_index("feature")
    assert float(importance.loc["silica_feed", "importance"]) > 0
    assert int(importance.loc["silica_feed", "rank"]) == 1
    assert int(importance.loc["silica_feed", "n_features"]) == 59


def test_feed_importance_reports_a_rank_within_all_fifty_nine(spark):
    hourly, restricted, _ = default_fixture()
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    table = feed_importance_table(feed)
    assert set(table["feature"]) == {"iron_feed", "silica_feed"}
    assert bool(table["rank"].between(1, 59).all())

    top = top_importance_table(feed, top_n=5)
    assert top["rank"].tolist() == [1, 2, 3, 4, 5]
    assert top["importance"].is_monotonic_decreasing


def test_feed_importance_is_empty_for_the_sensor_only_arm(spark):
    hourly, restricted, _ = default_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    assert feed_importance_table(sensor).empty


def test_result_schema_and_summary(spark):
    hourly, restricted, _ = default_fixture()
    sensor = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY, fold_ids=(1,))
    feed = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED, fold_ids=(1,))

    assert list(sensor.results.columns) == RESULT_COLUMNS
    combined = combine_results(sensor, feed)
    summary = summarize_configurations(combined).set_index("configuration")
    assert int(summary.loc[SENSOR_ONLY, "n_features"]) == 57
    assert int(summary.loc[FEED_ENHANCED, "n_features"]) == 59

    row = sensor.results.iloc[0]
    assert row["n_train_positive"] > 0
    assert row["train_positive_rate"] == pytest.approx(
        row["n_train_positive"] / row["n_train"]
    )


# ---------------------------------------------------------------------
# Real artifacts
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_thresholds_come_from_training_windows_only():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))

    from src.models.baselines import DEVELOPMENT_FOLD_IDS, get_fold_frames

    for fold_id in DEVELOPMENT_FOLD_IDS:
        labelled = build_labelled_fold(hourly, restricted, fold_id)
        frames = get_fold_frames(hourly, restricted, fold_id)
        assert labelled.threshold == pytest.approx(
            float(frames.train[TARGET_COLUMN].quantile(EXCURSION_QUANTILE))
        )
        # Both classes must be present for the ranking metrics to exist.
        assert labelled.train[LABEL_COLUMN].nunique() == 2
        assert labelled.validation[LABEL_COLUMN].nunique() == 2
        assert labelled.train[LABEL_COLUMN].mean() == pytest.approx(0.10, abs=0.01)


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_development_rows_never_touch_the_final_test_period():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))

    test_hours = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    development = restricted[restricted["fold_kind"] == KIND_DEVELOPMENT]
    assert not set(development[TIMESTAMP_COLUMN]).intersection(test_hours)

    from src.models.baselines import DEVELOPMENT_FOLD_IDS

    for fold_id in DEVELOPMENT_FOLD_IDS:
        labelled = build_labelled_fold(hourly, restricted, fold_id)
        assert not set(labelled.train[TIMESTAMP_COLUMN]).intersection(test_hours)
        assert not set(labelled.validation[TIMESTAMP_COLUMN]).intersection(test_hours)
        assert (
            labelled.train[TIMESTAMP_COLUMN].max() < labelled.validation[TIMESTAMP_COLUMN].min()
        )
