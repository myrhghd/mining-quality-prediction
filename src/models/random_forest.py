"""Spark MLlib Random Forest benchmark for hourly `% Silica Concentrate`.

Second sensor based regression model for the project. The question this
milestone asks is narrow: do nonlinear relationships and interactions
among the same 57 sensor predictors improve forward chronological
validation performance relative to Linear Regression? It does not ask
whether Random Forest is the right eventual model, and a positive or
negative result here should not be read as a verdict on that broader
question.

Evaluated on the three development validation folds only, using the
predictor scope, split assignments, and Spark environment handling
already established by `src.models.linear_regression` and
`src.models.baselines`, imported rather than reimplemented. The final
test period is never loaded into a scoring population here.

No `StandardScaler` stage is used. Tree splits are threshold comparisons
on one feature at a time, so a monotonic rescaling of any feature cannot
change which split a tree chooses; scaling would only relabel the
predictor axis, not change the model.

This is a fixed, untuned nonlinear benchmark. The configuration was set
before any Random Forest validation result was observed, and it is not
adjusted based on the results below. It should not be read as a
production configuration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import TARGET_COLUMN, TIMESTAMP_COLUMN, find_repo_root
from src.data.split import (
    FINAL_TEST_FOLD_ID,
    ROLE_EMBARGO,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models.baselines import DEVELOPMENT_FOLD_IDS, compute_metrics, get_fold_frames, load_inputs
from src.models.linear_regression import (
    NUMERICAL_TOLERANCE,
    RESULT_COLUMNS,
    build_spark_session,
    compare_with_baselines,
    get_sensor_predictors,
    summarize_development,
    to_spark,
    validate_predictor_scope,
)

MODEL_NAME = "random_forest"

FEATURES_COLUMN = "features"
PREDICTION_COLUMN = "prediction"

# ---------------------------------------------------------------------
# Fixed model configuration
# ---------------------------------------------------------------------
#
# One configuration is used unchanged across all three folds, chosen
# before viewing any Random Forest validation result. This is an initial
# nonlinear benchmark, not a tuned production model, and no parameter
# search is performed.
#
# numTrees=200        A reasonably stable ensemble average without
#                      excessive local runtime on a laptop scale Spark
#                      session; variance across trees falls off with
#                      diminishing returns well before this point.
# maxDepth=8           Deep enough to capture interactions between the 57
#                      sensor aggregates, shallow enough to limit how far
#                      individual trees can overfit given only a few
#                      thousand training hours per fold.
# minInstancesPerNode=5  A floor on how specific a split can be, to
#                      reduce leaves fitted to a handful of hours.
# featureSubsetStrategy="sqrt"  Samples roughly 8 of 57 features per
#                      split, decorrelating trees that would otherwise
#                      all lean on the same dominant sensor group.
# subsamplingRate=0.8  Each tree trains on 80 percent of the fold's rows,
#                      adding ensemble diversity beyond feature sampling
#                      alone.
# seed=42              Fixes the row and feature sampling so the
#                      experiment is reproducible.
#
# None of these values are claimed to be optimal; they are a defensible
# starting point for a first nonlinear pass.
NUM_TREES = 200
MAX_DEPTH = 8
MIN_INSTANCES_PER_NODE = 5
FEATURE_SUBSET_STRATEGY = "sqrt"
SUBSAMPLING_RATE = 0.8
SEED = 42


@dataclass(frozen=True)
class RandomForestFoldResult:
    fold_id: int
    validation_metrics: dict[str, float]
    train_metrics: dict[str, float]
    n_train: int
    n_validation: int
    n_scored: int
    n_features: int
    feature_importances: np.ndarray
    predictions: pd.DataFrame


# ---------------------------------------------------------------------
# Spark pipeline
# ---------------------------------------------------------------------


def build_pipeline(
    predictors: list[str],
    num_trees: int = NUM_TREES,
    max_depth: int = MAX_DEPTH,
    min_instances_per_node: int = MIN_INSTANCES_PER_NODE,
    feature_subset_strategy: str = FEATURE_SUBSET_STRATEGY,
    subsampling_rate: float = SUBSAMPLING_RATE,
    seed: int = SEED,
):
    """Assemble the Spark ML pipeline: `VectorAssembler` then
    `RandomForestRegressor`, with no scaling stage.

    Returned unfitted so that every fold fits its own instance.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor

    assembler = VectorAssembler(
        inputCols=predictors, outputCol=FEATURES_COLUMN, handleInvalid="error"
    )
    forest = RandomForestRegressor(
        featuresCol=FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        predictionCol=PREDICTION_COLUMN,
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances_per_node,
        featureSubsetStrategy=feature_subset_strategy,
        subsamplingRate=subsampling_rate,
        seed=seed,
    )
    return Pipeline(stages=[assembler, forest])


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
) -> RandomForestFoldResult:
    """Fit the pipeline on one fold's training rows and score its validation rows.

    Training set predictions are also produced, purely as an overfitting
    diagnostic; they play no role in model ranking.
    """
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this milestone.")

    predictors = get_sensor_predictors() if predictors is None else list(predictors)
    validate_predictor_scope(predictors)

    frames = get_fold_frames(hourly, assignment, fold_id)

    train_spark = to_spark(spark, frames.train, predictors)
    validation_spark = to_spark(spark, frames.validation, predictors)

    if train_spark.count() != len(frames.train):
        raise ValueError(f"Fold {fold_id}: training row count changed during Spark conversion.")
    if validation_spark.count() != len(frames.validation):
        raise ValueError(f"Fold {fold_id}: validation row count changed during Spark conversion.")

    pipeline = build_pipeline(predictors, **pipeline_kwargs)
    fitted = pipeline.fit(train_spark)

    def score(spark_frame: "pyspark.sql.DataFrame", expected_rows: int) -> pd.DataFrame:  # noqa: F821
        scored = fitted.transform(spark_frame)
        collected = (
            scored.select(TIMESTAMP_COLUMN, TARGET_COLUMN, PREDICTION_COLUMN)
            .toPandas()
            .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
            .reset_index(drop=True)
        )
        if len(collected) != expected_rows:
            raise ValueError(
                f"Fold {fold_id}: produced {len(collected)} predictions for {expected_rows} rows."
            )
        if not np.isfinite(collected[PREDICTION_COLUMN]).all():
            raise ValueError(f"Fold {fold_id}: predictions contain non-finite values.")
        return collected

    validation_predictions = score(validation_spark, len(frames.validation))
    train_predictions = score(train_spark, len(frames.train))

    validation_metrics = compute_metrics(
        validation_predictions[TARGET_COLUMN].to_numpy(),
        validation_predictions[PREDICTION_COLUMN].to_numpy(),
    )
    train_metrics = compute_metrics(
        train_predictions[TARGET_COLUMN].to_numpy(),
        train_predictions[PREDICTION_COLUMN].to_numpy(),
    )

    forest_model = fitted.stages[-1]
    importances = np.asarray(forest_model.featureImportances.toArray(), dtype=float)

    return RandomForestFoldResult(
        fold_id=fold_id,
        validation_metrics=validation_metrics,
        train_metrics=train_metrics,
        n_train=len(frames.train),
        n_validation=len(frames.validation),
        n_scored=len(validation_predictions),
        n_features=len(predictors),
        feature_importances=importances,
        predictions=validation_predictions,
    )


def evaluate_model(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
    **pipeline_kwargs,
) -> tuple[pd.DataFrame, list[RandomForestFoldResult]]:
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
                **result.validation_metrics,
            }
            for result in fold_results
        ],
        columns=RESULT_COLUMNS,
    )
    return results, fold_results


# ---------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------


def training_diagnostics(fold_results: list[RandomForestFoldResult]) -> pd.DataFrame:
    """Train versus validation metrics per fold, as an overfitting diagnostic only.

    `rmse_generalization_gap = validation_rmse - train_rmse`. A large
    positive gap indicates the ensemble fits its own training rows far
    better than it predicts forward in time. This table is never used to
    rank models; chronological validation performance alone does that.
    """
    rows = []
    for result in fold_results:
        rows.append(
            {
                "fold_id": result.fold_id,
                "train_rmse": result.train_metrics["rmse"],
                "train_mae": result.train_metrics["mae"],
                "train_r2": result.train_metrics["r2"],
                "validation_rmse": result.validation_metrics["rmse"],
                "validation_mae": result.validation_metrics["mae"],
                "validation_r2": result.validation_metrics["r2"],
                "rmse_generalization_gap": (
                    result.validation_metrics["rmse"] - result.train_metrics["rmse"]
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Feature importance diagnostics
# ---------------------------------------------------------------------


def feature_importance_table(
    fold_results: list[RandomForestFoldResult], predictors: list[str], top_n: int = 10
) -> pd.DataFrame:
    """Per fold importance summary: feature count, importance sum, and the
    top `top_n` features by importance."""
    rows = []
    for result in fold_results:
        importances = result.feature_importances
        order = np.argsort(importances)[::-1][:top_n]
        rows.append(
            {
                "fold_id": result.fold_id,
                "n_features": len(importances),
                "importance_sum": float(importances.sum()),
                "top_features": [predictors[i] for i in order],
                "top_importances": [float(importances[i]) for i in order],
            }
        )
    return pd.DataFrame(rows)


def importance_stability(
    fold_results: list[RandomForestFoldResult], predictors: list[str], top_n: int = 10
) -> pd.DataFrame:
    """Cross fold importance stability, descriptive only.

    For every feature: its mean importance across folds, the standard
    deviation of that importance, and how many folds place it in the top
    `top_n`. No feature is removed based on this table.
    """
    matrix = np.vstack([result.feature_importances for result in fold_results])
    mean_importance = matrix.mean(axis=0)
    std_importance = matrix.std(axis=0, ddof=0)

    top_counts = np.zeros(len(predictors), dtype=int)
    for result in fold_results:
        order = np.argsort(result.feature_importances)[::-1][:top_n]
        top_counts[order] += 1

    table = pd.DataFrame(
        {
            "feature": predictors,
            "mean_importance": mean_importance,
            "std_importance": std_importance,
            "n_folds_in_top10": top_counts,
        }
    )
    return table.sort_values("mean_importance", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------


def build_comparators(
    baseline_results: pd.DataFrame, linear_regression_results: pd.DataFrame
) -> pd.DataFrame:
    """Combine the baseline and Linear Regression results into one long
    table so `compare_with_baselines` can compare Random Forest against
    all three references with a single reused call.
    """
    lr_as_comparator = linear_regression_results[["fold_id", "rmse"]].copy()
    lr_as_comparator["baseline"] = "linear_regression"
    return pd.concat(
        [baseline_results[["fold_id", "baseline", "rmse"]], lr_as_comparator],
        ignore_index=True,
    )


COMPARATOR_LABELS = {
    "training_mean": "training mean baseline",
    "persistence": (
        "optimistic walk forward temporal reference (laboratory reporting "
        "latency is unavailable, so this is not a verified operator baseline)"
    ),
    "linear_regression": "Linear Regression",
}


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    results: pd.DataFrame,
    fold_results: list[RandomForestFoldResult],
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

        if not train_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: training contains ineligible hours.")
        if not validation_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: validation contains ineligible hours.")

        scored_timestamps = set(result.predictions[TIMESTAMP_COLUMN])
        if scored_timestamps.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was scored.")
        if scored_timestamps.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: a final test hour was scored.")
        if train_timestamps.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was used for fitting.")
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
        for label, metrics in (
            ("validation", result.validation_metrics),
            ("training diagnostic", result.train_metrics),
        ):
            for metric, value in metrics.items():
                if not np.isfinite(value):
                    raise ValueError(f"Fold {fold_id}: {label} {metric} is not finite.")

        importances = result.feature_importances
        if len(importances) != 57:
            raise ValueError(
                f"Fold {fold_id}: feature importance vector has {len(importances)} entries, not 57."
            )
        if not np.isfinite(importances).all():
            raise ValueError(f"Fold {fold_id}: feature importances contain non-finite values.")
        if (importances < 0).any():
            raise ValueError(f"Fold {fold_id}: a feature importance is negative.")
        importance_sum = float(importances.sum())
        if not np.isclose(importance_sum, 1.0, atol=1e-4):
            raise ValueError(
                f"Fold {fold_id}: feature importances sum to {importance_sum:.6f}, expected ~1.0."
            )


def verify_deterministic_within_tolerance(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    predictors: list[str],
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Fit the same fold twice and confirm the metrics agree within tolerance.

    A fixed seed makes Spark's Random Forest reproducible in this local,
    single node setup; this is confirmed empirically here rather than
    assumed, since determinism across distributed aggregation is a
    property of the specific execution, not a guarantee of the API.
    """
    first = evaluate_fold(spark, hourly, assignment, fold_id, predictors=predictors)
    second = evaluate_fold(spark, hourly, assignment, fold_id, predictors=predictors)

    for metric in ("rmse", "mae", "r2"):
        if abs(first.validation_metrics[metric] - second.validation_metrics[metric]) > tolerance:
            raise ValueError(
                f"Fold {fold_id}: {metric} is not deterministic within tolerance "
                f"({first.validation_metrics[metric]} vs {second.validation_metrics[metric]})."
            )
    if not np.allclose(
        first.feature_importances, second.feature_importances, atol=tolerance
    ):
        raise ValueError(f"Fold {fold_id}: feature importances are not deterministic.")


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_baseline_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "baseline_results.parquet"


def default_linear_regression_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "linear_regression_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def format_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    importance_table: pd.DataFrame,
    stability_table: pd.DataFrame,
) -> str:
    lines = ["Random Forest evaluation (development folds only)", ""]
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
            f"  {row['model']:<16}  RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(["", "Train versus validation diagnostics (diagnostic only, not for ranking)", ""])
    for _, row in diagnostics.iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: train RMSE {row['train_rmse']:.4f} (R2 {row['train_r2']:.4f}) "
            f"-> validation RMSE {row['validation_rmse']:.4f} (R2 {row['validation_r2']:.4f})  "
            f"gap {row['rmse_generalization_gap']:+.4f}"
        )

    lines.extend(["", "Comparison with existing models (negative difference means Random Forest is better)", ""])
    for comparator in sorted(comparison["baseline"].unique()):
        subset = comparison[comparison["baseline"] == comparator]
        label = COMPARATOR_LABELS.get(comparator, comparator)
        lines.append(f"  vs {label}")
        for _, row in subset.iterrows():
            lines.append(
                f"    fold {int(row['fold_id'])}: {row['rmse_model']:.4f} vs "
                f"{row['rmse_baseline']:.4f}  difference {row['rmse_difference']:+.4f}"
            )
        lines.append(f"    aggregate difference {subset['rmse_difference'].mean():+.4f}")

    lines.extend(["", "Feature importance diagnostics", ""])
    for _, row in importance_table.iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: {row['n_features']} features, "
            f"importance sum {row['importance_sum']:.6f}"
        )
        top_pairs = ", ".join(
            f"{name}={value:.4f}"
            for name, value in zip(row["top_features"][:5], row["top_importances"][:5])
        )
        lines.append(f"    top 5: {top_pairs}")

    lines.extend(["", "Cross fold importance stability (top 10 by mean importance)", ""])
    for _, row in stability_table.head(10).iterrows():
        lines.append(
            f"  {row['feature']:<38} mean {row['mean_importance']:.4f}  "
            f"sd {row['std_importance']:.4f}  in top10: {int(row['n_folds_in_top10'])}/3 folds"
        )

    return "\n".join(lines)


def run(
    hourly_path: Path,
    splits_path: Path,
    baseline_path: Path,
    linear_regression_path: Path,
    results_path: Path,
    spark=None,
):
    """Evaluate the model, validate the guards, compare, and write results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityRandomForest")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)
        predictors = get_sensor_predictors()

        results, fold_results = evaluate_model(spark, hourly, assignment)
        validate_evaluation(results, fold_results, hourly, assignment, predictors)
        verify_deterministic_within_tolerance(spark, hourly, assignment, predictors)

        summary = summarize_development(results)
        diagnostics = training_diagnostics(fold_results)
        importance_table = feature_importance_table(fold_results, predictors)
        stability_table = importance_stability(fold_results, predictors)

        for path, description in (
            (baseline_path, "baseline results"),
            (linear_regression_path, "Linear Regression results"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {description} not found at: {path}. Run that module first."
                )
        baseline_results = pd.read_parquet(baseline_path)
        linear_regression_results = pd.read_parquet(linear_regression_path)
        comparators = build_comparators(baseline_results, linear_regression_results)
        comparison = compare_with_baselines(results, comparators)

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(results_path, index=False)

        return results, summary, diagnostics, comparison, importance_table, stability_table
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and evaluating Random Forest...")
    results, summary, diagnostics, comparison, importance_table, stability_table = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_baseline_path(repo_root),
        default_linear_regression_path(repo_root),
        results_path,
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(
        format_report(
            results, summary, diagnostics, comparison, importance_table, stability_table
        )
    )


if __name__ == "__main__":
    sys.exit(main())
