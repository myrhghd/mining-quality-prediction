"""Spark MLlib Gradient Boosted Trees benchmark for hourly `% Silica Concentrate`.

Third sensor based regression model for the project. The question this
milestone asks is narrow: does sequential boosting capture useful
nonlinear structure beyond the bagged Random Forest, while the data, the
57 sensor predictors, and the chronological validation design are held
constant? It does not ask whether boosting is the right eventual model
family, and neither a positive nor a negative result here settles that
broader question.

Evaluated on the three development validation folds only, using the
predictor scope, split assignments, and Spark environment handling
already established by `src.models.linear_regression`,
`src.models.random_forest`, and `src.models.baselines`, imported rather
than reimplemented. The final test period is never loaded into a
scoring population here.

No `StandardScaler` stage is used. Boosted trees split on threshold
comparisons against one feature at a time, so a monotonic rescaling of
any feature cannot change which split a tree chooses; scaling would only
relabel the predictor axis.

This is a fixed, untuned benchmark. The configuration was set before any
Gradient Boosted Trees validation result was observed, and it is not
adjusted based on the results below. It is not claimed to be optimal and
should not be read as a production configuration.
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

MODEL_NAME = "gradient_boosted_trees"

FEATURES_COLUMN = "features"
PREDICTION_COLUMN = "prediction"

EXPECTED_PREDICTOR_COUNT = 57

# ---------------------------------------------------------------------
# Fixed model configuration
# ---------------------------------------------------------------------
#
# One configuration is used unchanged across all three folds, chosen
# before viewing any Gradient Boosted Trees validation result. No
# parameter search is performed at this milestone.
#
# maxIter=100          One hundred boosting iterations give the ensemble
#                      enough sequential learners for an initial
#                      benchmark at this learning rate.
# maxDepth=5           Allows interactions between the sensor aggregates
#                      while limiting how complex any individual tree in
#                      the sequence becomes.
# maxBins=32           Spark's default discretization of continuous
#                      splits, left unchanged so the benchmark does not
#                      quietly depend on a nonstandard binning choice.
# minInstancesPerNode=5  A floor on how specific a split can be, which
#                      reduces leaves fitted to a handful of hours.
# stepSize=0.05        A conservative learning rate, so each tree makes a
#                      small correction rather than a large one.
# subsamplingRate=0.8  Each tree is fitted on 80 percent of the fold's
#                      rows, introducing stochasticity that may reduce
#                      overfitting.
# lossType="squared"   Squared error is appropriate for this continuous
#                      regression target and keeps the training objective
#                      consistent with the reported RMSE.
# seed=42              Fixes the row sampling so the experiment is
#                      reproducible.
#
# None of these values are claimed to be optimal; they are a defensible
# starting point for a first boosting pass.
MAX_ITER = 100
MAX_DEPTH = 5
MAX_BINS = 32
MIN_INSTANCES_PER_NODE = 5
STEP_SIZE = 0.05
SUBSAMPLING_RATE = 0.8
LOSS_TYPE = "squared"
SEED = 42

# The specification this milestone must satisfy, kept separate from the
# constants above so a guard can compare a built pipeline against it
# rather than against itself.
FIXED_CONFIGURATION = {
    "maxIter": 100,
    "maxDepth": 5,
    "maxBins": 32,
    "minInstancesPerNode": 5,
    "stepSize": 0.05,
    "subsamplingRate": 0.8,
    "lossType": "squared",
    "seed": 42,
}


@dataclass(frozen=True)
class GradientBoostedTreesFoldResult:
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
    max_iter: int = MAX_ITER,
    max_depth: int = MAX_DEPTH,
    max_bins: int = MAX_BINS,
    min_instances_per_node: int = MIN_INSTANCES_PER_NODE,
    step_size: float = STEP_SIZE,
    subsampling_rate: float = SUBSAMPLING_RATE,
    loss_type: str = LOSS_TYPE,
    seed: int = SEED,
):
    """Assemble the Spark ML pipeline: `VectorAssembler` then
    `GBTRegressor`, with no scaling stage.

    Returned unfitted so that every fold fits its own instance.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

    assembler = VectorAssembler(
        inputCols=predictors, outputCol=FEATURES_COLUMN, handleInvalid="error"
    )
    booster = GBTRegressor(
        featuresCol=FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        predictionCol=PREDICTION_COLUMN,
        maxIter=max_iter,
        maxDepth=max_depth,
        maxBins=max_bins,
        minInstancesPerNode=min_instances_per_node,
        stepSize=step_size,
        subsamplingRate=subsampling_rate,
        lossType=loss_type,
        seed=seed,
    )
    return Pipeline(stages=[assembler, booster])


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
) -> GradientBoostedTreesFoldResult:
    """Fit the pipeline on one fold's training rows and score its validation rows.

    Training set predictions are also produced, purely as an overfitting
    diagnostic; they play no role in model ranking.
    """
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this milestone.")

    predictors = get_sensor_predictors() if predictors is None else list(predictors)
    validate_predictor_scope(predictors)

    # Validate the exact pipeline instance that will be fitted. This
    # prevents a caller from overriding one of the fixed benchmark
    # parameters through ``pipeline_kwargs`` while still passing a guard
    # that only inspected the default builder configuration.
    pipeline = build_pipeline(predictors, **pipeline_kwargs)
    validate_fixed_configuration(predictors, pipeline=pipeline)

    frames = get_fold_frames(hourly, assignment, fold_id)

    train_spark = to_spark(spark, frames.train, predictors)
    validation_spark = to_spark(spark, frames.validation, predictors)

    if train_spark.count() != len(frames.train):
        raise ValueError(f"Fold {fold_id}: training row count changed during Spark conversion.")
    if validation_spark.count() != len(frames.validation):
        raise ValueError(f"Fold {fold_id}: validation row count changed during Spark conversion.")

    fitted = pipeline.fit(train_spark)

    def score(spark_frame, expected_rows: int) -> pd.DataFrame:
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

    booster_model = fitted.stages[-1]
    importances = np.asarray(booster_model.featureImportances.toArray(), dtype=float)

    return GradientBoostedTreesFoldResult(
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
) -> tuple[pd.DataFrame, list[GradientBoostedTreesFoldResult]]:
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


def training_diagnostics(fold_results: list[GradientBoostedTreesFoldResult]) -> pd.DataFrame:
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
    fold_results: list[GradientBoostedTreesFoldResult],
    predictors: list[str],
    top_n: int = 10,
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
    fold_results: list[GradientBoostedTreesFoldResult],
    predictors: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Cross fold importance stability, descriptive only.

    For every feature: its mean importance across folds, the standard
    deviation of that importance, and how many folds place it in the top
    `top_n`. No feature is removed based on this table, and a high
    importance describes how often the ensemble split on a variable, not
    that the variable causes the outcome.
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


def compare_importance_with_random_forest(
    stability_table: pd.DataFrame,
    random_forest_stability: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Descriptive overlap between the two ensembles' leading features.

    Reports each model's mean importance rank for the union of their top
    `top_n` features. This is a description of where two fitted models
    placed their splits, not evidence about the process itself.
    """
    gbt = stability_table.set_index("feature")["mean_importance"]
    forest = random_forest_stability.set_index("feature")["mean_importance"]

    union = list(dict.fromkeys(list(gbt.index[:top_n]) + list(forest.index[:top_n])))
    gbt_rank = {feature: rank for rank, feature in enumerate(gbt.index, start=1)}
    forest_rank = {feature: rank for rank, feature in enumerate(forest.index, start=1)}

    return pd.DataFrame(
        {
            "feature": union,
            "gbt_mean_importance": [float(gbt.get(feature, np.nan)) for feature in union],
            "gbt_rank": [gbt_rank.get(feature) for feature in union],
            "random_forest_mean_importance": [
                float(forest.get(feature, np.nan)) for feature in union
            ],
            "random_forest_rank": [forest_rank.get(feature) for feature in union],
        }
    )


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------


def build_comparators(
    baseline_results: pd.DataFrame,
    linear_regression_results: pd.DataFrame,
    random_forest_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every existing model's fold results into one long table so
    `compare_with_baselines` can compare Gradient Boosted Trees against
    all four references with a single reused call.
    """
    frames = [baseline_results[["fold_id", "baseline", "rmse"]]]
    for results, label in (
        (linear_regression_results, "linear_regression"),
        (random_forest_results, "random_forest"),
    ):
        comparator = results[["fold_id", "rmse"]].copy()
        comparator["baseline"] = label
        frames.append(comparator)
    return pd.concat(frames, ignore_index=True)


COMPARATOR_LABELS = {
    "training_mean": "training mean baseline",
    "persistence": (
        "optimistic walk forward temporal reference because laboratory reporting "
        "latency is unavailable"
    ),
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
}


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_fixed_configuration(predictors: list[str], pipeline=None) -> None:
    """Confirm the exact pipeline to be fitted carries the fixed parameters.

    When ``pipeline`` is omitted, validate a freshly built default
    pipeline. Passing the actual pipeline instance lets the production
    evaluation path detect parameter overrides before fitting.
    """
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

    pipeline = build_pipeline(predictors) if pipeline is None else pipeline
    stages = pipeline.getStages()
    if len(stages) != 2:
        raise ValueError(f"Pipeline must have exactly 2 stages, found {len(stages)}.")
    if not isinstance(stages[0], VectorAssembler):
        raise ValueError("The first pipeline stage must be a VectorAssembler.")
    if not isinstance(stages[1], GBTRegressor):
        raise ValueError("The second pipeline stage must be a GBTRegressor.")

    booster = stages[1]
    actual = {
        "maxIter": booster.getMaxIter(),
        "maxDepth": booster.getMaxDepth(),
        "maxBins": booster.getMaxBins(),
        "minInstancesPerNode": booster.getMinInstancesPerNode(),
        "stepSize": booster.getStepSize(),
        "subsamplingRate": booster.getSubsamplingRate(),
        "lossType": booster.getLossType(),
        "seed": booster.getSeed(),
    }
    for name, expected in FIXED_CONFIGURATION.items():
        if isinstance(expected, float):
            matches = np.isclose(actual[name], expected)
        else:
            matches = actual[name] == expected
        if not matches:
            raise ValueError(
                f"Fixed configuration mismatch for {name}: expected {expected}, got {actual[name]}."
            )

    if booster.getLabelCol() != TARGET_COLUMN:
        raise ValueError(f"Label column must be {TARGET_COLUMN}.")
    if booster.getFeaturesCol() != FEATURES_COLUMN:
        raise ValueError(f"Features column must be {FEATURES_COLUMN}.")
    if booster.getPredictionCol() != PREDICTION_COLUMN:
        raise ValueError(f"Prediction column must be {PREDICTION_COLUMN}.")


def validate_evaluation(
    results: pd.DataFrame,
    fold_results: list[GradientBoostedTreesFoldResult],
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
        if result.n_features != EXPECTED_PREDICTOR_COUNT:
            raise ValueError(
                f"Fold {fold_id}: model used {result.n_features} features, "
                f"not {EXPECTED_PREDICTOR_COUNT}."
            )
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
        if len(importances) != EXPECTED_PREDICTOR_COUNT:
            raise ValueError(
                f"Fold {fold_id}: feature importance vector has {len(importances)} entries, "
                f"not {EXPECTED_PREDICTOR_COUNT}."
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

    # The actual fitted pipeline is validated inside ``evaluate_fold``.
    # Rechecking the default builder here also protects the module-level
    # benchmark specification from accidental drift.
    validate_fixed_configuration(predictors)


def verify_deterministic_within_tolerance(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    predictors: list[str],
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Fit the same fold twice and confirm the metrics agree within tolerance.

    A fixed seed makes Spark's boosting reproducible in this local, single
    node setup; this is confirmed empirically here rather than assumed,
    since determinism across distributed aggregation is a property of the
    specific execution, not a guarantee of the API.
    """
    first = evaluate_fold(spark, hourly, assignment, fold_id, predictors=predictors)
    second = evaluate_fold(spark, hourly, assignment, fold_id, predictors=predictors)

    for metric in ("rmse", "mae", "r2"):
        if abs(first.validation_metrics[metric] - second.validation_metrics[metric]) > tolerance:
            raise ValueError(
                f"Fold {fold_id}: {metric} is not deterministic within tolerance "
                f"({first.validation_metrics[metric]} vs {second.validation_metrics[metric]})."
            )
    if not np.allclose(first.feature_importances, second.feature_importances, atol=tolerance):
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


def default_random_forest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "gradient_boosted_trees_results.parquet"


def format_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    importance_table: pd.DataFrame,
    stability_table: pd.DataFrame,
) -> str:
    lines = ["Gradient Boosted Trees evaluation (development folds only)", ""]
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
            f"  {row['model']:<24}  RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(
        ["", "Train versus validation diagnostics (diagnostic only, not for ranking)", ""]
    )
    for _, row in diagnostics.iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: train RMSE {row['train_rmse']:.4f} "
            f"(R2 {row['train_r2']:.4f}) -> validation RMSE {row['validation_rmse']:.4f} "
            f"(R2 {row['validation_r2']:.4f})  gap {row['rmse_generalization_gap']:+.4f}"
        )

    lines.extend(
        [
            "",
            "Comparison with existing models "
            "(negative difference means Gradient Boosted Trees is better)",
            "",
        ]
    )
    for comparator in sorted(comparison["baseline"].unique()):
        subset = comparison[comparison["baseline"] == comparator]
        label = COMPARATOR_LABELS.get(comparator, comparator)
        lines.append(f"  vs {label}")
        for _, row in subset.iterrows():
            lines.append(
                f"    fold {int(row['fold_id'])}: {row['rmse_model']:.4f} vs "
                f"{row['rmse_baseline']:.4f}  difference {row['rmse_difference']:+.4f}"
            )
        lines.append(f"    mean difference {subset['rmse_difference'].mean():+.4f}")

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
    random_forest_path: Path,
    results_path: Path,
    spark=None,
):
    """Evaluate the model, validate the guards, compare, and write results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityGradientBoostedTrees")

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
            (random_forest_path, "Random Forest results"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {description} not found at: {path.name}. Run that module first."
                )
        comparators = build_comparators(
            pd.read_parquet(baseline_path),
            pd.read_parquet(linear_regression_path),
            pd.read_parquet(random_forest_path),
        )
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

    print("Starting Spark session and evaluating Gradient Boosted Trees...")
    results, summary, diagnostics, comparison, importance_table, stability_table = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_baseline_path(repo_root),
        default_linear_regression_path(repo_root),
        default_random_forest_path(repo_root),
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
