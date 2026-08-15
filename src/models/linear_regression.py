"""Spark MLlib Linear Regression benchmark for hourly `% Silica Concentrate`.

Completed sensor based regression benchmark, evaluated on the three
development validation folds only. The final test period is never loaded
into a scoring population here.

The benchmark uses Spark MLlib end to end: a Spark
`Pipeline` of `VectorAssembler` then `StandardScaler` then
`LinearRegression`, fitted independently for every fold. Metrics reuse
the baseline module's implementations so model and baseline figures are
directly comparable rather than merely similar.

Scaling and leakage
-------------------
The scaler is a pipeline stage, not a preprocessing step. Each fold
fits its own `StandardScaler` on that fold's training rows only, and
validation rows are transformed by the already fitted pipeline. Nothing
is scaled before the chronological split, so no validation statistic can
reach the training procedure.

Predictor scope
---------------
Only the 57 core sensor aggregates are used. Feed chemistry is excluded
because its historical operational availability is unresolved. Target
derived metadata is excluded because it exists to protect split
boundaries, not to predict.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    CORE_SENSOR_PREDICTOR_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    find_repo_root,
)
from src.data.split import (
    FINAL_TEST_FOLD_ID,
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
    DEVELOPMENT_FOLD_IDS,
    INTERPOLATED_COLUMN,
    compute_metrics,
    get_fold_frames,
    load_inputs,
)

MODEL_NAME = "linear_regression"

FEATURES_RAW_COLUMN = "features_raw"
FEATURES_SCALED_COLUMN = "features_scaled"
PREDICTION_COLUMN = "prediction"

RESULT_COLUMNS = [
    "fold_id",
    "model",
    "n_train",
    "n_validation",
    "n_scored",
    "n_features",
    "rmse",
    "mae",
    "r2",
]

# ---------------------------------------------------------------------
# Fixed model configuration
# ---------------------------------------------------------------------
#
# One configuration is used unchanged across all three folds. It was
# chosen a priori rather than from validation performance, and no
# parameter search was performed.
#
# `regParam=0.1` with `elasticNetParam=0.0` is ridge regression. Pure L2
# is the appropriate first choice here because the 57 sensor aggregates
# contain strongly collinear groups: the seven flotation column air flow
# aggregates move together, as do the seven level aggregates, and some
# level / variability / slope summaries remain correlated within process
# variables. Ordinary least squares on collinear predictors produces unstable, large
# magnitude coefficients that swing between folds, which would make the
# coefficient diagnostics below meaningless. Ridge shrinks correlated
# coefficients toward each other instead of arbitrarily selecting among
# them.
#
# L1 is deliberately avoided because this benchmark does not perform
# feature selection. It would zero out members of collinear groups in a
# way that is unstable across folds.
#
# `standardization=False` because the pipeline's own `StandardScaler`
# stage has already standardized the features. Leaving Spark's internal
# standardization enabled would rescale twice and make the reported
# regularization strength difficult to interpret.
#
# Spark documents `regParam` as lambda in its regularized linear
# regression objective. Because this pipeline explicitly standardizes
# the features and disables LinearRegression's internal feature
# standardization, the chosen value should be interpreted in Spark's
# objective on the already standardized feature vectors. Regularization
# parameters from other libraries should not be compared numerically
# unless their objective scaling and feature preprocessing are matched.
DEFAULT_REG_PARAM = 0.1
DEFAULT_ELASTIC_NET_PARAM = 0.0
DEFAULT_MAX_ITER = 100
DEFAULT_TOLERANCE = 1e-6
DEFAULT_SOLVER = "normal"

# Metrics are compared across independent Spark fits, so an exact float
# match is not a reasonable expectation. This tolerance is used by the
# determinism guard and tests.
NUMERICAL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FoldModelResult:
    fold_id: int
    metrics: dict[str, float]
    n_train: int
    n_validation: int
    n_scored: int
    n_features: int
    intercept: float
    coefficients: np.ndarray
    predictions: pd.DataFrame


# ---------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------


def configure_spark_environment() -> None:
    """Point Spark at this interpreter and at a compatible JDK.

    Spark launches Python workers as separate processes and resolves both
    the interpreter and the JVM from the environment. On a machine with
    several conda environments and an older system Java, the defaults
    resolve to the wrong ones: workers pick up whichever Python is first
    on PATH, and Spark 4 rejects the Java 8 that ships with the system.

    Both variables are set from the running interpreter's own prefix and
    only when not already set, so an explicit choice by the caller is
    respected. Nothing outside this process is modified.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    java_home = Path(sys.prefix) / "lib" / "jvm"
    if java_home.exists():
        os.environ.setdefault("JAVA_HOME", str(java_home))


def build_spark_session(app_name: str = "MiningQualityLinearRegression"):
    """Create a small local Spark session suited to this dataset.

    The hourly table is a few thousand rows, so the defaults tuned for a
    cluster only add overhead. Shuffle partitions are reduced from the
    default 200 accordingly, and the web UI is disabled because nothing
    here needs it.
    """
    configure_spark_environment()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.driver.memory", "2g")
        # Arrow cannot convert this table's timestamp column and falls back
        # to the standard path with a warning on every conversion. The data
        # is small enough that Arrow buys nothing, so it is disabled rather
        # than left to fail and retry.
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------
# Predictor selection
# ---------------------------------------------------------------------


def get_sensor_predictors() -> list[str]:
    """Return the 57 core sensor predictors, verified before use."""
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)
    if len(predictors) != 57:
        raise ValueError(f"Expected 57 core sensor predictors, found {len(predictors)}.")
    return predictors


def validate_predictor_scope(predictors: list[str]) -> None:
    """Fail if anything outside the sensor aggregate scope is present."""
    if len(predictors) != 57:
        raise ValueError(f"Expected 57 sensor predictors, got {len(predictors)}.")
    if len(set(predictors)) != len(predictors):
        raise ValueError("Predictor list contains duplicates.")

    forbidden = set(FEED_CONTEXT_PREDICTOR_COLUMNS) | {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        SEGMENT_COLUMN,
        SENSOR_ELIGIBLE_COLUMN,
        INTERPOLATED_COLUMN,
        "iron_concentrate",
        "is_feed_model_eligible",
        "target_run_id",
        "target_run_length",
        "hours_since_target_change",
        "n_samples",
        "n_frozen_sensors",
        "is_sensor_valid",
        "silica_concentrate_first",
        "silica_concentrate_last",
        "silica_concentrate_range",
    }
    leaked = forbidden.intersection(predictors)
    if leaked:
        raise ValueError(f"Forbidden columns present in the predictor set: {sorted(leaked)}")

    expected = list(CORE_SENSOR_PREDICTOR_COLUMNS)
    if predictors != expected:
        missing = sorted(set(expected) - set(predictors))
        unexpected = sorted(set(predictors) - set(expected))
        raise ValueError(
            "Predictor set must exactly match the 57 approved core sensor predictors. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )


# ---------------------------------------------------------------------
# Spark pipeline
# ---------------------------------------------------------------------


def build_pipeline(
    predictors: list[str],
    reg_param: float = DEFAULT_REG_PARAM,
    elastic_net_param: float = DEFAULT_ELASTIC_NET_PARAM,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
    solver: str = DEFAULT_SOLVER,
):
    """Assemble the Spark ML pipeline.

    Returned unfitted so that every fold fits its own instance; sharing a
    fitted pipeline across folds would leak one fold's scaling statistics
    into another.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.ml.regression import LinearRegression

    assembler = VectorAssembler(
        inputCols=predictors, outputCol=FEATURES_RAW_COLUMN, handleInvalid="error"
    )
    scaler = StandardScaler(
        inputCol=FEATURES_RAW_COLUMN,
        outputCol=FEATURES_SCALED_COLUMN,
        withMean=True,
        withStd=True,
    )
    regression = LinearRegression(
        featuresCol=FEATURES_SCALED_COLUMN,
        labelCol=TARGET_COLUMN,
        predictionCol=PREDICTION_COLUMN,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
        maxIter=max_iter,
        tol=tolerance,
        solver=solver,
        standardization=False,
    )
    return Pipeline(stages=[assembler, scaler, regression])


def to_spark(spark, frame: pd.DataFrame, predictors: list[str]):
    """Convert a pandas fold frame to Spark with only the needed columns."""
    columns = [TIMESTAMP_COLUMN, TARGET_COLUMN] + predictors
    subset = frame[columns].copy()
    for column in [TARGET_COLUMN] + predictors:
        subset[column] = subset[column].astype(float)
    return spark.createDataFrame(subset)


# ---------------------------------------------------------------------
# Fold training and scoring
# ---------------------------------------------------------------------


def evaluate_fold(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_id: int,
    predictors: list[str] | None = None,
    **pipeline_kwargs,
) -> FoldModelResult:
    """Fit the pipeline on one fold's training rows and score its validation rows."""
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this milestone.")

    predictors = get_sensor_predictors() if predictors is None else list(predictors)
    validate_predictor_scope(predictors)

    frames = get_fold_frames(hourly, assignment, fold_id)

    train_spark = to_spark(spark, frames.train, predictors)
    validation_spark = to_spark(spark, frames.validation, predictors)

    # Row counts must survive the conversion; a silent drop would quietly
    # change what the model is fitted on.
    if train_spark.count() != len(frames.train):
        raise ValueError(f"Fold {fold_id}: training row count changed during Spark conversion.")
    if validation_spark.count() != len(frames.validation):
        raise ValueError(f"Fold {fold_id}: validation row count changed during Spark conversion.")

    pipeline = build_pipeline(predictors, **pipeline_kwargs)
    fitted = pipeline.fit(train_spark)

    scored = fitted.transform(validation_spark)
    predictions = (
        scored.select(TIMESTAMP_COLUMN, TARGET_COLUMN, PREDICTION_COLUMN)
        .toPandas()
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )

    if len(predictions) != len(frames.validation):
        raise ValueError(
            f"Fold {fold_id}: produced {len(predictions)} predictions for "
            f"{len(frames.validation)} validation rows."
        )
    if not np.isfinite(predictions[PREDICTION_COLUMN]).all():
        raise ValueError(f"Fold {fold_id}: predictions contain non-finite values.")

    metrics = compute_metrics(
        predictions[TARGET_COLUMN].to_numpy(), predictions[PREDICTION_COLUMN].to_numpy()
    )

    regression_model = fitted.stages[-1]
    coefficients = np.asarray(regression_model.coefficients.toArray(), dtype=float)

    return FoldModelResult(
        fold_id=fold_id,
        metrics=metrics,
        n_train=len(frames.train),
        n_validation=len(frames.validation),
        n_scored=len(predictions),
        n_features=len(predictors),
        intercept=float(regression_model.intercept),
        coefficients=coefficients,
        predictions=predictions,
    )


def evaluate_model(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
    **pipeline_kwargs,
) -> tuple[pd.DataFrame, list[FoldModelResult]]:
    """Evaluate the model across every development fold."""
    fold_results = [
        evaluate_fold(spark, hourly, assignment, fold_id, **pipeline_kwargs)
        for fold_id in fold_ids
    ]

    results = pd.DataFrame(
        [
            {
                "fold_id": result.fold_id,
                "model": MODEL_NAME,
                "n_train": result.n_train,
                "n_validation": result.n_validation,
                "n_scored": result.n_scored,
                "n_features": result.n_features,
                **result.metrics,
            }
            for result in fold_results
        ],
        columns=RESULT_COLUMNS,
    )
    return results, fold_results


# ---------------------------------------------------------------------
# Summary, comparison, diagnostics
# ---------------------------------------------------------------------


def summarize_development(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds."""
    return (
        results.groupby("model")
        .agg(
            n_folds=("fold_id", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )


def compare_with_baselines(results: pd.DataFrame, baseline_results: pd.DataFrame) -> pd.DataFrame:
    """Per fold RMSE difference against each baseline.

    A negative `rmse_difference` means Linear Regression is better.
    """
    merged = results[["fold_id", "rmse"]].merge(
        baseline_results[["fold_id", "baseline", "rmse"]],
        on="fold_id",
        suffixes=("_model", "_baseline"),
    )
    merged["rmse_difference"] = merged["rmse_model"] - merged["rmse_baseline"]
    merged["model_better"] = merged["rmse_difference"] < 0
    return merged.sort_values(["baseline", "fold_id"], kind="mergesort").reset_index(drop=True)


def coefficient_diagnostics(
    fold_results: list[FoldModelResult], predictors: list[str], top_n: int = 8
) -> pd.DataFrame:
    """Per fold intercept, coefficient count, L2 norm, and largest terms.

    Because the pipeline standardizes features before fitting, coefficient
    magnitudes are on a comparable scale and can be read as relative
    influence. They are reported as a diagnostic only; no feature
    selection is performed from them in this benchmark.
    """
    rows = []
    for result in fold_results:
        coefficients = result.coefficients
        order = np.argsort(np.abs(coefficients))[::-1][:top_n]
        largest = "; ".join(
            f"{predictors[index]}={coefficients[index]:+.4f}" for index in order
        )
        rows.append(
            {
                "fold_id": result.fold_id,
                "intercept": result.intercept,
                "n_coefficients": len(coefficients),
                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
                "max_abs_coefficient": float(np.max(np.abs(coefficients))),
                "largest_coefficients": largest,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    results: pd.DataFrame,
    fold_results: list[FoldModelResult],
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    predictors: list[str],
) -> None:
    """Verify every structural guard, raising a clear error on violation."""
    if list(results.columns) != RESULT_COLUMNS:
        raise ValueError(
            f"Result schema mismatch. Expected {RESULT_COLUMNS}, got {list(results.columns)}"
        )

    validate_predictor_scope(predictors)

    eligible = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    final_test = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])

    for result in fold_results:
        fold_id = result.fold_id
        frames = get_fold_frames(hourly, assignment, fold_id)

        train_timestamps = set(frames.train[TIMESTAMP_COLUMN])
        validation_timestamps = set(frames.validation[TIMESTAMP_COLUMN])
        embargo_timestamps = set(frames.embargo[TIMESTAMP_COLUMN])

        # Timestamps must match the committed assignment exactly.
        committed = assignment[
            (assignment["fold_id"] == fold_id) & (assignment["fold_kind"] == "development")
        ]
        expected_train = set(committed.loc[committed["role"] == ROLE_TRAIN, TIMESTAMP_COLUMN])
        expected_validation = set(
            committed.loc[committed["role"] == ROLE_VALIDATION, TIMESTAMP_COLUMN]
        )
        if train_timestamps != expected_train:
            raise ValueError(f"Fold {fold_id}: training rows differ from the committed split.")
        if validation_timestamps != expected_validation:
            raise ValueError(f"Fold {fold_id}: validation rows differ from the committed split.")

        # Eligibility, embargo exclusion, final test exclusion.
        if not train_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: training contains ineligible hours.")
        if not validation_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: validation contains ineligible hours.")

        scored_timestamps = set(result.predictions[TIMESTAMP_COLUMN])
        if scored_timestamps.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was scored.")
        if scored_timestamps.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: a final test hour was scored.")
        if train_timestamps.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: training overlaps the final test period.")
        if scored_timestamps != validation_timestamps:
            raise ValueError(f"Fold {fold_id}: scored hours differ from the validation window.")

        if result.n_scored != result.n_validation:
            raise ValueError(
                f"Fold {fold_id}: scored {result.n_scored} of {result.n_validation} rows."
            )
        if result.n_features != 57:
            raise ValueError(f"Fold {fold_id}: model used {result.n_features} features, not 57.")
        if not np.isfinite(result.predictions[PREDICTION_COLUMN]).all():
            raise ValueError(f"Fold {fold_id}: predictions contain non-finite values.")
        if not np.isfinite(result.coefficients).all():
            raise ValueError(f"Fold {fold_id}: coefficients contain non-finite values.")
        if not np.isfinite(result.intercept):
            raise ValueError(f"Fold {fold_id}: intercept is not finite.")
        for metric, value in result.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"Fold {fold_id}: {metric} is not finite.")


def verify_scaler_is_fold_local(
    spark, hourly: pd.DataFrame, assignment: pd.DataFrame, predictors: list[str]
) -> None:
    """Confirm each fold's scaler is fitted from that fold's training rows.

    The scaler's learned mean is compared against the mean of the fold's
    own training data and against the mean of the full eligible
    population. Matching the former and differing from the latter is what
    distinguishes fold local fitting from global preprocessing.
    """
    frames = get_fold_frames(hourly, assignment, DEVELOPMENT_FOLD_IDS[0])
    train_spark = to_spark(spark, frames.train, predictors)

    pipeline = build_pipeline(predictors)
    fitted = pipeline.fit(train_spark)
    scaler_model = fitted.stages[1]

    learned_mean = np.asarray(scaler_model.mean.toArray(), dtype=float)
    fold_train_mean = frames.train[predictors].to_numpy(dtype=float).mean(axis=0)
    global_mean = (
        hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], predictors].to_numpy(dtype=float).mean(axis=0)
    )

    if not np.allclose(learned_mean, fold_train_mean, rtol=1e-6, atol=1e-8):
        raise ValueError("Scaler statistics do not match the fold's own training rows.")
    if np.allclose(learned_mean, global_mean, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "Scaler statistics match the full eligible population, which indicates "
            "global scaling rather than fold local fitting."
        )


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_baseline_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "baseline_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "linear_regression_results.parquet"


def format_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    lines = ["Linear Regression evaluation (development folds only)", ""]
    lines.append(
        f"{'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'scored':>6}  "
        f"{'feats':>5}  {'rmse':>7}  {'mae':>7}  {'r2':>8}"
    )
    for _, row in results.iterrows():
        lines.append(
            f"{int(row['fold_id']):>4}  {int(row['n_train']):>7,}  {int(row['n_validation']):>5,}  "
            f"{int(row['n_scored']):>6,}  {int(row['n_features']):>5}  {row['rmse']:>7.4f}  "
            f"{row['mae']:>7.4f}  {row['r2']:>8.4f}"
        )

    lines.extend(["", "Development summary across folds", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"  {row['model']:<18}  RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(
        ["", "Baseline comparison (negative difference means Linear Regression is better)", ""]
    )
    for baseline in sorted(comparison["baseline"].unique()):
        subset = comparison[comparison["baseline"] == baseline]
        lines.append(f"  vs {baseline}")
        for _, row in subset.iterrows():
            lines.append(
                f"    fold {int(row['fold_id'])}: {row['rmse_model']:.4f} vs "
                f"{row['rmse_baseline']:.4f}  difference {row['rmse_difference']:+.4f}"
            )
        lines.append(f"    aggregate difference {subset['rmse_difference'].mean():+.4f}")

    lines.extend(["", "Coefficient diagnostics", ""])
    for _, row in diagnostics.iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: intercept {row['intercept']:+.4f}, "
            f"{int(row['n_coefficients'])} coefficients, "
            f"L2 norm {row['coefficient_l2_norm']:.4f}, "
            f"max |coef| {row['max_abs_coefficient']:.4f}"
        )
        lines.append(f"    largest: {row['largest_coefficients']}")

    return "\n".join(lines)


def run(
    hourly_path: Path,
    splits_path: Path,
    baseline_path: Path,
    results_path: Path,
    spark=None,
):
    """Evaluate the model, validate the guards, compare, and write results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session()

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)
        predictors = get_sensor_predictors()

        results, fold_results = evaluate_model(spark, hourly, assignment)
        validate_evaluation(results, fold_results, hourly, assignment, predictors)
        verify_scaler_is_fold_local(spark, hourly, assignment, predictors)

        summary = summarize_development(results)
        diagnostics = coefficient_diagnostics(fold_results, predictors)

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline results not found at: {baseline_path}. Run the baseline module first."
            )
        baseline_results = pd.read_parquet(baseline_path)
        comparison = compare_with_baselines(results, baseline_results)

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(results_path, index=False)

        return results, summary, comparison, diagnostics
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and evaluating Linear Regression...")
    results, summary, comparison, diagnostics = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_baseline_path(repo_root),
        results_path,
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(results, summary, comparison, diagnostics))


if __name__ == "__main__":
    sys.exit(main())
