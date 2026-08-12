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
from src.models.linear_regression import (
    DEFAULT_REG_PARAM,
    FEATURES_RAW_COLUMN,
    FEATURES_SCALED_COLUMN,
    MODEL_NAME,
    NUMERICAL_TOLERANCE,
    PREDICTION_COLUMN,
    RESULT_COLUMNS,
    build_pipeline,
    build_spark_session,
    coefficient_diagnostics,
    compare_with_baselines,
    evaluate_fold,
    evaluate_model,
    get_sensor_predictors,
    summarize_development,
    to_spark,
    validate_evaluation,
    validate_predictor_scope,
    verify_scaler_is_fold_local,
)

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_SPLITS.exists()


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole module; starting one is expensive."""
    session = build_spark_session("MiningQualityLinearRegressionTests")
    yield session
    session.stop()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(n_rows: int = 120, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Build a synthetic hourly table with all 57 sensor predictors.

    The target depends linearly on a few predictors plus noise, so a
    linear model has genuine signal to recover.
    """
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    data = {column: rng.normal(size=n_rows) for column in predictors}
    frame = pd.DataFrame(data)
    frame[TARGET_COLUMN] = (
        2.0
        + 1.5 * frame[predictors[0]]
        - 0.8 * frame[predictors[1]]
        + 0.4 * frame[predictors[2]]
        + rng.normal(scale=0.1, size=n_rows)
    )
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
    hourly = make_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 80)),
        embargo_idx=list(range(80, 85)),
        validation_idx=list(range(85, 120)),
    )
    return hourly, assignment


# ---------------------------------------------------------------------
# 1, 2, 3. Predictor scope
# ---------------------------------------------------------------------


def test_exactly_57_sensor_predictors_are_selected():
    predictors = get_sensor_predictors()
    assert len(predictors) == 57
    assert predictors == list(CORE_SENSOR_PREDICTOR_COLUMNS)
    assert len(set(predictors)) == 57


def test_feed_predictors_are_excluded():
    predictors = get_sensor_predictors()
    for column in FEED_CONTEXT_PREDICTOR_COLUMNS:
        assert column not in predictors
    assert "iron_feed" not in predictors
    assert "silica_feed" not in predictors

    with pytest.raises(ValueError, match="Forbidden columns"):
        validate_predictor_scope(predictors[:56] + ["iron_feed"])


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
        "n_frozen_sensors",
    ):
        assert column not in predictors

    with pytest.raises(ValueError, match="Forbidden columns"):
        validate_predictor_scope(predictors[:56] + ["target_run_length"])


def test_wrong_predictor_count_is_rejected():
    predictors = get_sensor_predictors()
    with pytest.raises(ValueError, match="Expected 57"):
        validate_predictor_scope(predictors[:30])


def test_unexpected_predictor_is_rejected_even_when_count_is_57():
    predictors = get_sensor_predictors()
    altered = predictors[:-1] + ["unexpected_feature"]
    with pytest.raises(ValueError, match="exactly match"):
        validate_predictor_scope(altered)


# ---------------------------------------------------------------------
# 4. Pipeline composition
# ---------------------------------------------------------------------


def test_pipeline_contains_required_spark_stages(spark):
    # Constructing a Spark ML estimator requires an active SparkContext,
    # so this test needs the session fixture even though it never runs a job.
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.ml.regression import LinearRegression

    pipeline = build_pipeline(get_sensor_predictors())
    assert isinstance(pipeline, Pipeline)

    stages = pipeline.getStages()
    assert len(stages) == 3
    assert isinstance(stages[0], VectorAssembler)
    assert isinstance(stages[1], StandardScaler)
    assert isinstance(stages[2], LinearRegression)

    assert stages[0].getOutputCol() == FEATURES_RAW_COLUMN
    assert stages[1].getInputCol() == FEATURES_RAW_COLUMN
    assert stages[1].getOutputCol() == FEATURES_SCALED_COLUMN
    assert stages[2].getFeaturesCol() == FEATURES_SCALED_COLUMN
    assert stages[2].getLabelCol() == TARGET_COLUMN
    assert stages[2].getPredictionCol() == PREDICTION_COLUMN

    # Ridge: L2 only, and Spark's internal standardization is off because
    # the pipeline already scales.
    assert stages[2].getElasticNetParam() == 0.0
    assert stages[2].getRegParam() == DEFAULT_REG_PARAM
    assert stages[2].getStandardization() is False
    assert len(stages[0].getInputCols()) == 57


# ---------------------------------------------------------------------
# 5, 6, 7. Split assignments, embargo, final test
# ---------------------------------------------------------------------


def test_train_and_validation_assignments_are_respected(spark):
    hourly, assignment = default_fixture()
    result = evaluate_fold(spark, hourly, assignment, 1)

    frames = get_fold_frames(hourly, assignment, 1)
    assert result.n_train == len(frames.train) == 80
    assert result.n_validation == len(frames.validation) == 35
    assert set(result.predictions[TIMESTAMP_COLUMN]) == set(frames.validation[TIMESTAMP_COLUMN])


def test_embargo_rows_are_excluded(spark):
    hourly, assignment = default_fixture()
    result = evaluate_fold(spark, hourly, assignment, 1)

    frames = get_fold_frames(hourly, assignment, 1)
    embargo_ts = set(frames.embargo[TIMESTAMP_COLUMN])
    assert len(embargo_ts) == 5
    assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(embargo_ts)
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(embargo_ts)


def test_final_test_rows_are_excluded(spark):
    hourly = make_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 70)),
        embargo_idx=list(range(70, 75)),
        validation_idx=list(range(75, 100)),
        test_idx=list(range(100, 120)),
    )
    result = evaluate_fold(spark, hourly, assignment, 1)

    test_ts = set(hourly[TIMESTAMP_COLUMN].iloc[100:120])
    assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_ts)

    frames = get_fold_frames(hourly, assignment, 1)
    assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(test_ts)
    validate_evaluation(
        pd.DataFrame(
            [
                {
                    "fold_id": 1,
                    "model": MODEL_NAME,
                    "n_train": result.n_train,
                    "n_validation": result.n_validation,
                    "n_scored": result.n_scored,
                    "n_features": result.n_features,
                    **result.metrics,
                }
            ],
            columns=RESULT_COLUMNS,
        ),
        [result],
        hourly,
        assignment,
        get_sensor_predictors(),
    )


def test_final_test_fold_cannot_be_evaluated(spark):
    hourly, assignment = default_fixture()
    with pytest.raises(ValueError, match="final test fold must not be evaluated"):
        evaluate_fold(spark, hourly, assignment, FINAL_TEST_FOLD_ID)


# ---------------------------------------------------------------------
# 8. Scaler is fitted inside the fold pipeline, not globally
# ---------------------------------------------------------------------


def test_scaler_is_fitted_from_training_rows_only(spark):
    hourly, assignment = default_fixture()
    frames = get_fold_frames(hourly, assignment, 1)
    predictors = get_sensor_predictors()

    fitted = build_pipeline(predictors).fit(to_spark(spark, frames.train, predictors))
    scaler_model = fitted.stages[1]
    learned_mean = np.asarray(scaler_model.mean.toArray(), dtype=float)

    train_mean = frames.train[predictors].to_numpy(float).mean(axis=0)
    global_mean = hourly[predictors].to_numpy(float).mean(axis=0)

    assert np.allclose(learned_mean, train_mean, rtol=1e-6, atol=1e-8)
    # The training window is a strict subset, so a globally fitted scaler
    # would have produced different statistics.
    assert not np.allclose(learned_mean, global_mean, rtol=1e-6, atol=1e-8)


def test_verify_scaler_helper_passes_on_fold_local_fit(spark):
    hourly, assignment = default_fixture()
    verify_scaler_is_fold_local(spark, hourly, assignment, get_sensor_predictors())


def test_validation_is_transformed_by_the_fitted_pipeline(spark):
    hourly, assignment = default_fixture()
    frames = get_fold_frames(hourly, assignment, 1)
    predictors = get_sensor_predictors()

    fitted = build_pipeline(predictors).fit(to_spark(spark, frames.train, predictors))
    scaled = fitted.transform(to_spark(spark, frames.validation, predictors))

    assert FEATURES_RAW_COLUMN in scaled.columns
    assert FEATURES_SCALED_COLUMN in scaled.columns
    assert PREDICTION_COLUMN in scaled.columns

    # Validation is standardized by the training statistics, so its own
    # scaled mean is not forced to zero.
    scaled_mean = np.mean(
        np.vstack([row[FEATURES_SCALED_COLUMN].toArray() for row in scaled.collect()]), axis=0
    )
    assert not np.allclose(scaled_mean, np.zeros_like(scaled_mean), atol=1e-8)


# ---------------------------------------------------------------------
# 9, 10. Prediction count and metric correctness
# ---------------------------------------------------------------------


def test_prediction_count_equals_validation_count(spark):
    hourly, assignment = default_fixture()
    result = evaluate_fold(spark, hourly, assignment, 1)
    assert result.n_scored == result.n_validation == len(result.predictions)
    assert np.isfinite(result.predictions[PREDICTION_COLUMN]).all()


def test_metrics_match_independent_calculation(spark):
    hourly, assignment = default_fixture()
    result = evaluate_fold(spark, hourly, assignment, 1)

    y_true = result.predictions[TARGET_COLUMN].to_numpy()
    y_pred = result.predictions[PREDICTION_COLUMN].to_numpy()

    manual_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    manual_mae = float(np.mean(np.abs(y_true - y_pred)))
    manual_r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    assert result.metrics["rmse"] == pytest.approx(manual_rmse, abs=1e-12)
    assert result.metrics["mae"] == pytest.approx(manual_mae, abs=1e-12)
    assert result.metrics["r2"] == pytest.approx(manual_r2, abs=1e-12)


def test_pipeline_mechanics_match_independent_least_squares(spark):
    """The assemble, scale, and fit path must reproduce plain least squares.

    Checked with the penalty switched off, where Spark's solution is
    uniquely determined and independent of its internal regularization
    scaling. This validates the pipeline mechanics exactly rather than
    approximately: the scaler statistics, the feature ordering, and the
    intercept handling all have to be right for this to hold.
    """
    hourly, assignment = default_fixture()
    frames = get_fold_frames(hourly, assignment, 1)
    predictors = get_sensor_predictors()

    fitted = build_pipeline(predictors, reg_param=0.0).fit(
        to_spark(spark, frames.train, predictors)
    )
    scaler_model, regression_model = fitted.stages[1], fitted.stages[2]

    x_train = frames.train[predictors].to_numpy(float)
    y_train = frames.train[TARGET_COLUMN].to_numpy(float)

    mean = np.asarray(scaler_model.mean.toArray(), dtype=float)
    std = np.asarray(scaler_model.std.toArray(), dtype=float)
    # Spark's StandardScaler uses the sample standard deviation.
    assert np.allclose(std, x_train.std(axis=0, ddof=1))

    z_train = (x_train - mean) / std
    expected = np.linalg.lstsq(z_train, y_train - y_train.mean(), rcond=None)[0]

    assert np.allclose(
        np.asarray(regression_model.coefficients.toArray()), expected, atol=1e-9
    )
    assert float(regression_model.intercept) == pytest.approx(float(y_train.mean()), abs=1e-9)


def test_regularization_shrinks_the_coefficient_norm(spark):
    """Ridge must actually penalize, and in the shrinking direction.

    The exact penalized solution is not duplicated here because the
    purpose of this test is narrower: verify that enabling L2
    regularization shrinks the fitted coefficients relative to the
    otherwise identical unpenalized Spark pipeline.
    """
    hourly, assignment = default_fixture()
    frames = get_fold_frames(hourly, assignment, 1)
    predictors = get_sensor_predictors()
    train_spark = to_spark(spark, frames.train, predictors)

    unpenalized = build_pipeline(predictors, reg_param=0.0).fit(train_spark).stages[-1]
    penalized = build_pipeline(predictors, reg_param=DEFAULT_REG_PARAM).fit(train_spark).stages[-1]

    unpenalized_norm = float(np.linalg.norm(unpenalized.coefficients.toArray()))
    penalized_norm = float(np.linalg.norm(penalized.coefficients.toArray()))

    assert penalized_norm < unpenalized_norm
    # A stronger penalty must shrink further still.
    stronger = build_pipeline(predictors, reg_param=1.0).fit(train_spark).stages[-1]
    assert float(np.linalg.norm(stronger.coefficients.toArray())) < penalized_norm


def test_model_recovers_known_signal(spark):
    """A linear model must beat a constant on data that is genuinely linear."""
    hourly, assignment = default_fixture()
    result = evaluate_fold(spark, hourly, assignment, 1)
    assert result.metrics["r2"] > 0.8

    frames = get_fold_frames(hourly, assignment, 1)
    predictors = get_sensor_predictors()
    # The three generating predictors should carry the largest coefficients.
    largest = {predictors[i] for i in np.argsort(np.abs(result.coefficients))[::-1][:3]}
    assert largest == set(predictors[:3])


# ---------------------------------------------------------------------
# 11, 12. Schema and determinism
# ---------------------------------------------------------------------


def test_result_schema_is_correct(spark):
    hourly, assignment = default_fixture()
    results, fold_results = evaluate_model(spark, hourly, assignment, fold_ids=(1,))

    assert list(results.columns) == RESULT_COLUMNS
    assert len(results) == 1
    assert results["model"].iloc[0] == MODEL_NAME
    assert results["n_features"].iloc[0] == 57
    assert len(fold_results) == 1


def test_results_are_deterministic_within_tolerance(spark):
    hourly, assignment = default_fixture()
    first = evaluate_fold(spark, hourly, assignment, 1)
    second = evaluate_fold(spark, hourly, assignment, 1)

    for metric in ("rmse", "mae", "r2"):
        assert first.metrics[metric] == pytest.approx(
            second.metrics[metric], abs=NUMERICAL_TOLERANCE
        )
    assert np.allclose(first.coefficients, second.coefficients, atol=NUMERICAL_TOLERANCE)
    assert first.intercept == pytest.approx(second.intercept, abs=NUMERICAL_TOLERANCE)

    # Input row order must not change the outcome.
    shuffled = hourly.sample(frac=1.0, random_state=7).reset_index(drop=True)
    third = evaluate_fold(spark, shuffled, assignment, 1)
    assert third.metrics["rmse"] == pytest.approx(first.metrics["rmse"], abs=1e-4)


# ---------------------------------------------------------------------
# 13. Baseline comparison sign convention
# ---------------------------------------------------------------------


def test_baseline_comparison_signs_are_correct():
    results = pd.DataFrame(
        {"fold_id": [1, 2], "model": [MODEL_NAME] * 2, "rmse": [0.80, 1.20]}
    )
    baselines = pd.DataFrame(
        {
            "fold_id": [1, 2, 1, 2],
            "baseline": ["persistence", "persistence", "training_mean", "training_mean"],
            "rmse": [1.00, 1.00, 0.50, 0.50],
        }
    )
    comparison = compare_with_baselines(results, baselines)

    persistence = comparison[comparison["baseline"] == "persistence"].set_index("fold_id")
    # Model better than persistence in both folds -> negative difference.
    assert persistence.loc[1, "rmse_difference"] == pytest.approx(-0.20)
    assert bool(persistence.loc[1, "model_better"]) is True
    assert persistence.loc[2, "rmse_difference"] == pytest.approx(0.20)
    assert bool(persistence.loc[2, "model_better"]) is False

    training_mean = comparison[comparison["baseline"] == "training_mean"].set_index("fold_id")
    assert training_mean.loc[1, "rmse_difference"] == pytest.approx(0.30)
    assert bool(training_mean.loc[1, "model_better"]) is False


def test_summary_and_diagnostics_shape(spark):
    hourly, assignment = default_fixture()
    results, fold_results = evaluate_model(spark, hourly, assignment, fold_ids=(1,))

    summary = summarize_development(results)
    for column in ("rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"):
        assert column in summary.columns

    diagnostics = coefficient_diagnostics(fold_results, get_sensor_predictors())
    assert len(diagnostics) == 1
    for column in (
        "intercept",
        "n_coefficients",
        "coefficient_l2_norm",
        "max_abs_coefficient",
        "largest_coefficients",
    ):
        assert column in diagnostics.columns
    assert diagnostics["n_coefficients"].iloc[0] == 57


# ---------------------------------------------------------------------
# Real data integration (skips cleanly if artifacts are absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_linear_regression(spark):
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
        assert not set(result.predictions[TIMESTAMP_COLUMN]).intersection(test_ts)

    for metric in ("rmse", "mae", "r2"):
        assert np.isfinite(results[metric]).all()
    assert (results["rmse"] > 0).all()


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_scaler_is_fold_local(spark):
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    verify_scaler_is_fold_local(spark, hourly, assignment, get_sensor_predictors())
