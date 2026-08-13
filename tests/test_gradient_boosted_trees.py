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
from src.models.baselines import INTERPOLATED_COLUMN, compute_metrics, get_fold_frames
from src.models.gradient_boosted_trees import (
    FEATURES_COLUMN,
    FIXED_CONFIGURATION,
    LOSS_TYPE,
    MAX_BINS,
    MAX_DEPTH,
    MAX_ITER,
    MIN_INSTANCES_PER_NODE,
    MODEL_NAME,
    PREDICTION_COLUMN,
    SEED,
    STEP_SIZE,
    SUBSAMPLING_RATE,
    build_comparators,
    build_pipeline,
    evaluate_fold,
    evaluate_model,
    feature_importance_table,
    get_sensor_predictors,
    importance_stability,
    training_diagnostics,
    validate_evaluation,
    validate_fixed_configuration,
    validate_predictor_scope,
    verify_deterministic_within_tolerance,
)
from src.models.linear_regression import build_spark_session, compare_with_baselines, to_spark

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_SPLITS.exists()


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityGradientBoostedTreesTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_linear_hourly(n_rows: int = 150, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Hourly table with all 57 sensor predictors and a linear target,
    used where the specific functional form does not matter."""
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    frame[TARGET_COLUMN] = (
        2.0
        + 1.5 * frame[predictors[0]]
        - 0.8 * frame[predictors[1]]
        + rng.normal(scale=0.1, size=n_rows)
    )
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    frame[SEGMENT_COLUMN] = 0
    frame[SENSOR_ELIGIBLE_COLUMN] = True
    frame[INTERPOLATED_COLUMN] = False
    for column in FEED_CONTEXT_PREDICTOR_COLUMNS:
        frame[column] = 50.0
    return frame


def make_nonlinear_frame(n_rows: int = 600, seed: int = 0, start: str = "2020-01-01"):
    """A small, deliberately narrow frame (four predictors, not the full 57)
    where the target is an interaction of two of them and the other two are
    pure noise.

    Each informative input's marginal correlation with the target is
    approximately zero by construction, because the target depends only on
    whether the two informative inputs share a sign, not on either one
    alone. A boosted tree ensemble can split on both variables directly
    and recover it.

    Deliberately independent of `CORE_SENSOR_PREDICTOR_COLUMNS`: this
    tests whether the boosting mechanism can learn a nonlinear interaction
    at all, not whether the fixed configuration can find two informative
    columns among 57 production predictors, which is a different and much
    harder question this milestone makes no claim about.
    """
    rng = np.random.default_rng(seed)
    predictors = ["p0", "p1", "p2", "p3"]

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    x0, x1 = frame["p0"], frame["p1"]
    interaction = ((x0 > 0) != (x1 > 0)).astype(float)  # linearly inseparable
    frame[TARGET_COLUMN] = 2.0 + 3.0 * interaction + rng.normal(scale=0.05, size=n_rows)
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    return frame, predictors


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
    hourly = make_linear_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 100)),
        embargo_idx=list(range(100, 106)),
        validation_idx=list(range(106, 150)),
    )
    return hourly, assignment


@pytest.fixture(scope="module")
def fitted_fold(spark):
    """One fitted fold reused across assertions; boosting 100 iterations is
    the slowest step in this module."""
    hourly, assignment = default_fixture()
    return hourly, assignment, evaluate_fold(spark, hourly, assignment, 1)


# ---------------------------------------------------------------------
# 1, 2, 3. Predictor scope
# ---------------------------------------------------------------------


def test_exactly_57_approved_predictors_are_selected():
    predictors = get_sensor_predictors()
    assert len(predictors) == 57
    assert predictors == list(CORE_SENSOR_PREDICTOR_COLUMNS)


def test_feed_predictors_are_excluded():
    predictors = get_sensor_predictors()
    assert not set(FEED_CONTEXT_PREDICTOR_COLUMNS).intersection(predictors)
    with pytest.raises(ValueError, match="Forbidden columns"):
        validate_predictor_scope(predictors[:56] + ["iron_feed"])
    with pytest.raises(ValueError, match="Forbidden columns"):
        validate_predictor_scope(predictors[:56] + ["silica_feed"])


def test_target_and_metadata_predictors_are_excluded():
    predictors = get_sensor_predictors()
    for column in (
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        SEGMENT_COLUMN,
        SENSOR_ELIGIBLE_COLUMN,
        INTERPOLATED_COLUMN,
        "iron_concentrate",
        "target_run_id",
        "target_run_length",
        "hours_since_target_change",
    ):
        assert column not in predictors
    with pytest.raises(ValueError, match="Forbidden columns"):
        validate_predictor_scope(predictors[:56] + ["target_run_length"])


# ---------------------------------------------------------------------
# 4, 5, 6, 7. Pipeline composition and fixed configuration
# ---------------------------------------------------------------------


def test_pipeline_contains_assembler_and_gbt_regressor(spark):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

    pipeline = build_pipeline(get_sensor_predictors())
    assert isinstance(pipeline, Pipeline)

    stages = pipeline.getStages()
    assert len(stages) == 2
    assert isinstance(stages[0], VectorAssembler)
    assert isinstance(stages[1], GBTRegressor)

    assert stages[0].getOutputCol() == FEATURES_COLUMN
    assert len(stages[0].getInputCols()) == 57
    assert stages[1].getFeaturesCol() == FEATURES_COLUMN
    assert stages[1].getLabelCol() == TARGET_COLUMN
    assert stages[1].getPredictionCol() == PREDICTION_COLUMN


def test_no_standard_scaler_in_pipeline(spark):
    from pyspark.ml.feature import StandardScaler

    pipeline = build_pipeline(get_sensor_predictors())
    assert not any(isinstance(stage, StandardScaler) for stage in pipeline.getStages())


def test_fixed_configuration_matches_specification(spark):
    assert MAX_ITER == 100
    assert MAX_DEPTH == 5
    assert MAX_BINS == 32
    assert MIN_INSTANCES_PER_NODE == 5
    assert STEP_SIZE == pytest.approx(0.05)
    assert SUBSAMPLING_RATE == pytest.approx(0.8)
    assert LOSS_TYPE == "squared"
    assert SEED == 42

    assert FIXED_CONFIGURATION == {
        "maxIter": 100,
        "maxDepth": 5,
        "maxBins": 32,
        "minInstancesPerNode": 5,
        "stepSize": 0.05,
        "subsamplingRate": 0.8,
        "lossType": "squared",
        "seed": 42,
    }

    booster = build_pipeline(get_sensor_predictors()).getStages()[1]
    assert booster.getMaxIter() == 100
    assert booster.getMaxDepth() == 5
    assert booster.getMaxBins() == 32
    assert booster.getMinInstancesPerNode() == 5
    assert booster.getStepSize() == pytest.approx(0.05)
    assert booster.getSubsamplingRate() == pytest.approx(0.8)
    assert booster.getLossType() == "squared"
    assert booster.getSeed() == 42

    validate_fixed_configuration(get_sensor_predictors())


def test_configuration_guard_rejects_a_changed_parameter(spark, monkeypatch):
    import src.models.gradient_boosted_trees as gbt

    # The guard compares the built pipeline against the specification, so
    # moving the specification must make it fail. monkeypatch restores the
    # original value when the test ends.
    monkeypatch.setitem(gbt.FIXED_CONFIGURATION, "maxDepth", 6)
    with pytest.raises(ValueError, match="Fixed configuration mismatch for maxDepth"):
        validate_fixed_configuration(get_sensor_predictors())


def test_configuration_guard_rejects_an_overridden_pipeline(spark):
    predictors = get_sensor_predictors()
    overridden = build_pipeline(predictors, max_depth=6)

    with pytest.raises(ValueError, match="Fixed configuration mismatch for maxDepth"):
        validate_fixed_configuration(predictors, pipeline=overridden)


# ---------------------------------------------------------------------
# 8, 9, 10, 11. Split assignments, embargo, final test
# ---------------------------------------------------------------------


def test_train_assignment_is_respected(fitted_fold):
    hourly, assignment, result = fitted_fold
    frames = get_fold_frames(hourly, assignment, 1)
    assert result.n_train == len(frames.train) == 100


def test_validation_assignment_is_respected(fitted_fold):
    hourly, assignment, result = fitted_fold
    frames = get_fold_frames(hourly, assignment, 1)
    assert result.n_validation == len(frames.validation) == 44
    assert set(result.predictions[TIMESTAMP_COLUMN]) == set(frames.validation[TIMESTAMP_COLUMN])


def test_embargo_rows_are_excluded(fitted_fold):
    hourly, assignment, result = fitted_fold
    frames = get_fold_frames(hourly, assignment, 1)
    embargo_ts = set(frames.embargo[TIMESTAMP_COLUMN])
    assert len(embargo_ts) == 6
    assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(embargo_ts)
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(embargo_ts)


def test_final_test_rows_are_excluded(spark):
    hourly = make_linear_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 90)),
        embargo_idx=list(range(90, 96)),
        validation_idx=list(range(96, 130)),
        test_idx=list(range(130, 150)),
    )
    result = evaluate_fold(spark, hourly, assignment, 1)

    test_ts = set(hourly[TIMESTAMP_COLUMN].iloc[130:150])
    assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_ts)

    frames = get_fold_frames(hourly, assignment, 1)
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(test_ts)


def test_final_test_fold_cannot_be_evaluated(spark):
    hourly, assignment = default_fixture()
    with pytest.raises(ValueError, match="final test fold must not be evaluated"):
        evaluate_fold(spark, hourly, assignment, FINAL_TEST_FOLD_ID)


# ---------------------------------------------------------------------
# 12, 13. Prediction count and metric correctness
# ---------------------------------------------------------------------


def test_prediction_count_equals_validation_count(fitted_fold):
    _, _, result = fitted_fold
    assert result.n_scored == result.n_validation == len(result.predictions)
    assert np.isfinite(result.predictions[PREDICTION_COLUMN]).all()


def test_metrics_match_independent_calculation(fitted_fold):
    _, _, result = fitted_fold

    y_true = result.predictions[TARGET_COLUMN].to_numpy()
    y_pred = result.predictions[PREDICTION_COLUMN].to_numpy()

    manual_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    manual_mae = float(np.mean(np.abs(y_true - y_pred)))
    manual_r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    assert result.validation_metrics["rmse"] == pytest.approx(manual_rmse, abs=1e-9)
    assert result.validation_metrics["mae"] == pytest.approx(manual_mae, abs=1e-9)
    assert result.validation_metrics["r2"] == pytest.approx(manual_r2, abs=1e-9)


# ---------------------------------------------------------------------
# 14. Training diagnostics
# ---------------------------------------------------------------------


def test_training_diagnostics_are_calculated_correctly(fitted_fold):
    _, _, result = fitted_fold
    diagnostics = training_diagnostics([result])

    assert len(diagnostics) == 1
    row = diagnostics.iloc[0]
    assert row["fold_id"] == 1
    assert row["train_rmse"] == pytest.approx(result.train_metrics["rmse"])
    assert row["train_mae"] == pytest.approx(result.train_metrics["mae"])
    assert row["train_r2"] == pytest.approx(result.train_metrics["r2"])
    assert row["validation_rmse"] == pytest.approx(result.validation_metrics["rmse"])
    assert row["validation_mae"] == pytest.approx(result.validation_metrics["mae"])
    assert row["validation_r2"] == pytest.approx(result.validation_metrics["r2"])
    assert row["rmse_generalization_gap"] == pytest.approx(
        result.validation_metrics["rmse"] - result.train_metrics["rmse"]
    )

    # An ensemble fitted on its own training rows should fit them at least
    # as well as it predicts forward.
    assert row["train_rmse"] <= row["validation_rmse"] + 1e-6


# ---------------------------------------------------------------------
# 15, 16, 17, 18. Feature importances
# ---------------------------------------------------------------------


def test_feature_importance_count_is_57(fitted_fold):
    _, _, result = fitted_fold
    assert len(result.feature_importances) == 57


def test_feature_importances_are_finite(fitted_fold):
    _, _, result = fitted_fold
    assert np.isfinite(result.feature_importances).all()


def test_feature_importances_are_nonnegative(fitted_fold):
    _, _, result = fitted_fold
    assert (result.feature_importances >= 0).all()


def test_feature_importances_sum_to_approximately_one(fitted_fold):
    _, _, result = fitted_fold
    assert result.feature_importances.sum() == pytest.approx(1.0, abs=1e-4)


def test_importance_diagnostic_tables(fitted_fold):
    _, _, result = fitted_fold
    predictors = get_sensor_predictors()

    importance_table = feature_importance_table([result], predictors)
    assert len(importance_table) == 1
    row = importance_table.iloc[0]
    assert row["n_features"] == 57
    assert row["importance_sum"] == pytest.approx(1.0, abs=1e-4)
    assert len(row["top_features"]) == 10
    assert len(row["top_importances"]) == 10
    assert list(row["top_importances"]) == sorted(row["top_importances"], reverse=True)


def test_importance_stability_table_shape_and_values(fitted_fold):
    _, _, result = fitted_fold
    predictors = get_sensor_predictors()

    stability = importance_stability([result], predictors, top_n=10)
    assert len(stability) == 57
    assert set(stability.columns) == {
        "feature",
        "mean_importance",
        "std_importance",
        "n_folds_in_top10",
    }
    assert stability["n_folds_in_top10"].max() <= 1  # only one fold evaluated here
    assert stability["n_folds_in_top10"].sum() == 10
    means = stability["mean_importance"].to_numpy()
    assert (means[:-1] >= means[1:]).all()


# ---------------------------------------------------------------------
# 19, 20, 21. Comparison sign conventions
# ---------------------------------------------------------------------


def _comparison_fixture():
    gbt_results = pd.DataFrame({"fold_id": [1, 2], "rmse": [0.80, 1.10]})
    baseline_results = pd.DataFrame(
        {
            "fold_id": [1, 2, 1, 2],
            "baseline": ["persistence", "persistence", "training_mean", "training_mean"],
            "rmse": [0.75, 0.75, 1.00, 1.00],
        }
    )
    lr_results = pd.DataFrame({"fold_id": [1, 2], "rmse": [0.85, 0.90]})
    rf_results = pd.DataFrame({"fold_id": [1, 2], "rmse": [0.82, 1.30]})

    comparators = build_comparators(baseline_results, lr_results, rf_results)
    return gbt_results, comparators, compare_with_baselines(gbt_results, comparators)


def test_comparators_cover_every_existing_model():
    _, comparators, _ = _comparison_fixture()
    assert set(comparators["baseline"]) == {
        "persistence",
        "training_mean",
        "linear_regression",
        "random_forest",
    }


def test_baseline_comparison_sign_convention():
    _, _, comparison = _comparison_fixture()

    persistence = comparison[comparison["baseline"] == "persistence"].set_index("fold_id")
    assert persistence.loc[1, "rmse_difference"] == pytest.approx(0.05)  # GBT worse
    assert bool(persistence.loc[1, "model_better"]) is False

    training_mean = comparison[comparison["baseline"] == "training_mean"].set_index("fold_id")
    assert training_mean.loc[1, "rmse_difference"] == pytest.approx(-0.20)  # GBT better
    assert bool(training_mean.loc[1, "model_better"]) is True


def test_linear_regression_comparison_sign_convention():
    _, _, comparison = _comparison_fixture()
    linear = comparison[comparison["baseline"] == "linear_regression"].set_index("fold_id")
    assert linear.loc[1, "rmse_difference"] == pytest.approx(-0.05)  # GBT better
    assert bool(linear.loc[1, "model_better"]) is True
    assert linear.loc[2, "rmse_difference"] == pytest.approx(0.20)  # GBT worse
    assert bool(linear.loc[2, "model_better"]) is False


def test_random_forest_comparison_sign_convention():
    _, _, comparison = _comparison_fixture()
    forest = comparison[comparison["baseline"] == "random_forest"].set_index("fold_id")
    assert forest.loc[1, "rmse_difference"] == pytest.approx(-0.02)  # GBT better
    assert bool(forest.loc[1, "model_better"]) is True
    assert forest.loc[2, "rmse_difference"] == pytest.approx(-0.20)  # GBT better
    assert bool(forest.loc[2, "model_better"]) is True
    assert forest["rmse_difference"].mean() == pytest.approx(-0.11)


# ---------------------------------------------------------------------
# 22. Determinism
# ---------------------------------------------------------------------


def test_results_are_deterministic_within_tolerance(spark):
    hourly, assignment = default_fixture()
    predictors = get_sensor_predictors()
    verify_deterministic_within_tolerance(spark, hourly, assignment, predictors, fold_id=1)


# ---------------------------------------------------------------------
# 23. Synthetic nonlinear signal
# ---------------------------------------------------------------------


def test_model_learns_nonlinear_interaction_signal(spark):
    """Target is an interaction with no useful linear projection.

    Exercises `build_pipeline` directly, bypassing the 57 column
    production scope guard in `evaluate_fold`, on a small controlled
    frame. The test isolates whether the assemble and boost mechanism can
    recover a genuine interaction; it makes no claim about finding a two
    way interaction inside 57 candidate sensor columns.
    """
    frame, predictors = make_nonlinear_frame(n_rows=600)
    train, validation = frame.iloc[:450], frame.iloc[460:600]

    for column in ("p0", "p1"):
        marginal_correlation = np.corrcoef(frame[column], frame[TARGET_COLUMN])[0, 1]
        assert abs(marginal_correlation) < 0.15  # linear signal absent by construction

    fitted = build_pipeline(predictors).fit(to_spark(spark, train, predictors))
    scored = fitted.transform(to_spark(spark, validation, predictors))
    predictions = scored.select(TIMESTAMP_COLUMN, TARGET_COLUMN, PREDICTION_COLUMN).toPandas()

    metrics = compute_metrics(
        predictions[TARGET_COLUMN].to_numpy(), predictions[PREDICTION_COLUMN].to_numpy()
    )
    assert metrics["r2"] > 0.5

    importances = np.asarray(fitted.stages[-1].featureImportances.toArray())
    order = np.argsort(importances)[::-1][:2]
    assert set(order) == {0, 1}  # p0 and p1, the two interacting features, dominate


# ---------------------------------------------------------------------
# Validation guard behaviour
# ---------------------------------------------------------------------


def test_validate_evaluation_passes_on_a_clean_fit(spark):
    hourly, assignment = default_fixture()
    results, fold_results = evaluate_model(spark, hourly, assignment, fold_ids=(1,))
    validate_evaluation(results, fold_results, hourly, assignment, get_sensor_predictors())


def test_validate_evaluation_flags_bad_schema(spark):
    hourly, assignment = default_fixture()
    results, fold_results = evaluate_model(spark, hourly, assignment, fold_ids=(1,))
    bad_results = results.rename(columns={"rmse": "wrong_name"})
    with pytest.raises(ValueError, match="schema"):
        validate_evaluation(bad_results, fold_results, hourly, assignment, get_sensor_predictors())


# ---------------------------------------------------------------------
# Real data integration (skips cleanly if artifacts are absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_gradient_boosted_trees(spark):
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    predictors = get_sensor_predictors()

    results, fold_results = evaluate_model(spark, hourly, assignment)
    validate_evaluation(results, fold_results, hourly, assignment, predictors)

    assert len(results) == 3
    assert sorted(results["fold_id"]) == [1, 2, 3]
    assert (results["model"] == MODEL_NAME).all()

    assert results["n_train"].tolist() == [1743, 2224, 2708]
    assert results["n_validation"].tolist() == [482, 483, 481]
    assert results["n_scored"].tolist() == [482, 483, 481]
    assert (results["n_features"] == 57).all()

    test_ts = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    for result in fold_results:
        frames = get_fold_frames(hourly, assignment, result.fold_id)
        assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(test_ts)
        assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_ts)
        assert len(result.feature_importances) == 57
        assert np.isfinite(result.predictions[PREDICTION_COLUMN]).all()

    for metric in ("rmse", "mae", "r2"):
        assert np.isfinite(results[metric]).all()
