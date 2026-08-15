"""Information value of incoming feed chemistry for `% Silica Concentrate`.

Two model families and a wider dynamic sensor representation have now
been evaluated on the same chronological folds, and none of them
generalizes forward from process sensors alone. Every one of those
experiments varied how the plant's own operation was described. This one
varies what the plant is being fed.

Two predictor configurations are evaluated:

* the sensor only control, the existing 57 core sensor aggregates
* the feed enhanced configuration, those same 57 aggregates plus
  `iron_feed` and `silica_feed`

Everything else is held constant. The target, the 0 hour alignment, the
committed development folds, the embargo, the chronological validation
structure, the Random Forest configuration and its hyperparameters, and
the metric implementations are all imported unchanged. The presence of
feed chemistry is the only variable.

What this experiment does and does not establish
------------------------------------------------
This is an information value scenario. It asks how much additional
predictive information incoming ore composition would provide to the
model if reliable feed composition were available at prediction time.

It does not establish that these historical values were real time
measurements, and it does not establish that they were available to an
operator at the hour they describe. The raw dataset records one feed
composition value per hour with no sampling or reporting time, so the
operational availability of those numbers remains unresolved. Any
architecture conclusion drawn from this experiment depends on feed
composition being measured continuously, which is a property of a
proposed system rather than a property of this dataset.

Excluded information
--------------------
`% Iron Concentrate` is the other outcome of the same separation step and
is never a predictor. Preprocessing never aggregates it into the hourly
table at all, so it cannot appear here by accident; the guard below
confirms that rather than assuming it. Calendar values, target derived
metadata, the dynamic process history features, and interpolated target
rows are all outside this experiment. The final test period is never
fitted on, never scored, and never used to interpret a result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    ALL_PREDICTOR_COLUMNS,
    CORE_SENSOR_PREDICTOR_COLUMNS,
    EXCLUDED_OUTCOME_COLUMN,
    FEED_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    RAW_TO_STANDARD,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    find_repo_root,
)
from src.data.split import (
    FEED_ELIGIBLE_COLUMN,
    FINAL_TEST_FOLD_ID,
    KIND_DEVELOPMENT,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models.baselines import (
    DEVELOPMENT_FOLD_IDS,
    INTERPOLATED_COLUMN,
    compute_metrics,
    get_fold_frames,
    load_inputs,
)
from src.models.linear_regression import (
    NUMERICAL_TOLERANCE,
    build_spark_session,
    get_sensor_predictors,
    to_spark,
    validate_predictor_scope,
)
from src.models.random_forest import PREDICTION_COLUMN, build_pipeline

SENSOR_ONLY = "sensor_only"
FEED_ENHANCED = "feed_enhanced"
CONFIGURATIONS = (SENSOR_ONLY, FEED_ENHANCED)

RESULT_COLUMNS = [
    "configuration",
    "fold_id",
    "n_train",
    "n_validation",
    "n_scored",
    "n_features",
    "rmse",
    "mae",
    "r2",
]

# Quantile of a fold's own training target above which a validation hour
# counts as a high silica excursion. Derived from the training window
# only, so the grouping uses nothing a model would not have had. It is a
# reporting cut applied after measurement, never a value any model is
# fitted against.
EXCURSION_QUANTILE = 0.90

# Mean RMSE improvement below which a gain is treated as too small to act
# on. The same threshold the alignment and dynamic representation
# experiments use, so all three are read on one scale.
MEANINGFUL_RMSE = 0.01

# A prediction sitting within this many observed standard deviations of
# the fold's training mean is counted as having collapsed toward that
# mean rather than tracking the hour.
COLLAPSE_TOLERANCE_SD = 0.25

VALUE_STRONG = "strong feed chemistry value"
VALUE_MODERATE = "moderate or mixed feed chemistry value"
VALUE_NONE = "little or no feed chemistry value"


@dataclass(frozen=True)
class FeedFoldResult:
    configuration: str
    fold_id: int
    metrics: dict[str, float]
    n_train: int
    n_validation: int
    n_scored: int
    n_features: int
    feature_importances: np.ndarray
    predictions: pd.DataFrame


@dataclass(frozen=True)
class FeedEvaluation:
    configuration: str
    predictors: list[str]
    results: pd.DataFrame
    fold_results: list[FeedFoldResult]


# ---------------------------------------------------------------------
# Predictor sets
# ---------------------------------------------------------------------


def get_sensor_only_predictors() -> list[str]:
    """Return the 57 core sensor aggregates, the unchanged control."""
    return get_sensor_predictors()


def get_feed_enhanced_predictors() -> list[str]:
    """Return the 57 sensor aggregates followed by the two feed columns."""
    predictors = get_sensor_predictors() + list(FEED_CONTEXT_PREDICTOR_COLUMNS)
    if predictors != list(ALL_PREDICTOR_COLUMNS):
        raise ValueError(
            "The feed enhanced set must match the predictor schema declared by preprocessing."
        )
    if len(predictors) != 59:
        raise ValueError(f"Expected 59 feed enhanced predictors, found {len(predictors)}.")
    return predictors


def get_predictors(configuration: str) -> list[str]:
    if configuration == SENSOR_ONLY:
        return get_sensor_only_predictors()
    if configuration == FEED_ENHANCED:
        return get_feed_enhanced_predictors()
    raise ValueError(f"Unknown configuration: {configuration!r}")


# Names ruled out of both configurations. `iron_concentrate` is the other
# outcome of the same separation step; the rest are target derived
# metadata, split protection metadata, row quality flags, or calendar
# values this experiment does not use.
FORBIDDEN_PREDICTORS: set[str] = {
    EXCLUDED_OUTCOME_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
    FEED_ELIGIBLE_COLUMN,
    INTERPOLATED_COLUMN,
    "silica_concentrate_first",
    "silica_concentrate_last",
    "silica_concentrate_range",
    "target_run_id",
    "target_run_length",
    "hours_since_target_change",
    "n_samples",
    "n_frozen_sensors",
    "is_sensor_valid",
    "iron_feed_inconsistent",
    "silica_feed_inconsistent",
    "date",
    "month",
    "day_of_week",
    "hour_of_day",
    "year",
    "week_of_year",
}


def validate_configuration_scope(configuration: str, predictors: list[str]) -> None:
    """Fail if a predictor set contains anything outside its declared scope.

    The sensor only arm is delegated to the existing guard, so the control
    is checked by exactly the rule the committed benchmark uses. The feed
    enhanced arm must be that same list plus the two feed columns and
    nothing else, which is what keeps feed chemistry the only variable.
    """
    if len(set(predictors)) != len(predictors):
        raise ValueError("Predictor list contains duplicates.")

    # Checked before the exact list comparison so a history feature is
    # reported for what it is rather than as an unrecognized name.
    history = [
        column
        for column in predictors
        if column.endswith(("_change_1h", "_trailing_120m_mean", "_trailing_120m_std"))
        or "_trailing_15m" in column
        or "_trailing_30m" in column
        or "_trailing_60m" in column
    ]
    if history:
        raise ValueError(f"Dynamic sensor history features present in the predictor set: {history}")

    if configuration == SENSOR_ONLY:
        validate_predictor_scope(predictors)
        expected_forbidden = FORBIDDEN_PREDICTORS | set(FEED_CONTEXT_PREDICTOR_COLUMNS)
    elif configuration == FEED_ENHANCED:
        expected = list(ALL_PREDICTOR_COLUMNS)
        if predictors != expected:
            missing = sorted(set(expected) - set(predictors))
            unexpected = sorted(set(predictors) - set(expected))
            raise ValueError(
                "The feed enhanced predictor set must be the 57 approved sensor aggregates "
                f"followed by {FEED_CONTEXT_PREDICTOR_COLUMNS}. "
                f"Missing: {missing}; unexpected: {unexpected}"
            )
        expected_forbidden = FORBIDDEN_PREDICTORS
    else:
        raise ValueError(f"Unknown configuration: {configuration!r}")

    leaked = expected_forbidden.intersection(predictors)
    if leaked:
        raise ValueError(f"Forbidden columns present in the predictor set: {sorted(leaked)}")


# ---------------------------------------------------------------------
# Feed column provenance
# ---------------------------------------------------------------------


def describe_feed_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    """Summarize how the two feed columns are represented in the hourly table."""
    rows = []
    for column in FEED_COLUMNS:
        values = hourly[column]
        rows.append(
            {
                "column": column,
                "raw_source": next(
                    raw for raw, standard in RAW_TO_STANDARD.items() if standard == column
                ),
                "n_missing": int(values.isna().sum()),
                "n_distinct": int(values.nunique()),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "n_hours_inconsistent": int(hourly[f"{column}_inconsistent"].sum()),
            }
        )
    return pd.DataFrame(rows)


def assert_feed_columns_are_usable(hourly: pd.DataFrame) -> None:
    """Verify the feed columns are upstream inputs held apart from the outcome.

    Four properties are checked rather than assumed: the excluded
    concentrate outcome is absent from the table entirely, the feed
    columns are not copies of the target, no hour disagreed with itself
    when the hourly value was formed, and no value is missing on any hour
    the experiment can use.
    """
    if EXCLUDED_OUTCOME_COLUMN in hourly.columns:
        raise ValueError(
            f"{EXCLUDED_OUTCOME_COLUMN!r} is present in the hourly table and must never be "
            "available to this experiment."
        )

    missing = [column for column in FEED_COLUMNS if column not in hourly.columns]
    if missing:
        raise ValueError(f"Hourly table is missing feed column(s): {missing}")

    usable = hourly[hourly[FEED_ELIGIBLE_COLUMN]]
    for column in FEED_COLUMNS:
        if hourly[f"{column}_inconsistent"].any():
            raise ValueError(
                f"{column} holds more than one value within at least one hour, so the hourly "
                "value is not a faithful representative."
            )
        if usable[column].isna().any():
            raise ValueError(f"{column} is missing on at least one feed eligible hour.")
        if usable[column].equals(usable[TARGET_COLUMN]):
            raise ValueError(f"{column} is identical to the target and cannot be a predictor.")


def verify_feed_values_match_the_raw_record(hourly: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Confirm each hourly feed value is the unmodified raw observation.

    Preprocessing reduces each feed column to one representative value per
    hour. This reads the raw record back and confirms that value is the
    observation the plant actually logged, unchanged: no scaling, no
    smoothing, and nothing derived from the target. It also confirms the
    value is constant across the hour, which is what makes a single
    representative meaningful.
    """
    for column in FEED_COLUMNS:
        if column not in raw.columns:
            raise ValueError(f"Raw record is missing the standardized feed column {column!r}.")

    grouped = raw.groupby(TIMESTAMP_COLUMN)
    indexed = hourly.set_index(TIMESTAMP_COLUMN)

    for column in FEED_COLUMNS:
        distinct = grouped[column].nunique()
        if bool((distinct > 1).any()):
            raise ValueError(f"{column} varies within at least one recorded hour in the raw file.")

        observed = grouped[column].first().reindex(indexed.index)
        stored = indexed[column]
        if not np.array_equal(stored.to_numpy(dtype=float), observed.to_numpy(dtype=float)):
            raise ValueError(
                f"{column} in the hourly table does not match the raw observation for that hour."
            )

        # A feed value that moved with the target would be the signature of
        # a leakage transformation. The raw value is upstream of the assay
        # and must not equal it.
        if np.array_equal(stored.to_numpy(dtype=float), indexed[TARGET_COLUMN].to_numpy(dtype=float)):
            raise ValueError(f"{column} equals the target, which indicates a leakage transformation.")


# ---------------------------------------------------------------------
# Matched rows
# ---------------------------------------------------------------------


def feed_eligible_timestamps(hourly: pd.DataFrame) -> set:
    """Hours where both the sensor aggregates and the feed values are usable."""
    return set(hourly.loc[hourly[FEED_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])


def restrict_assignment(assignment: pd.DataFrame, usable_timestamps: set) -> pd.DataFrame:
    """Drop development rows where feed chemistry is unavailable.

    Both arms are then trained and scored on the identical row set, which
    is what makes any difference attributable to feed chemistry. On the
    committed dataset this removes nothing, because every sensor eligible
    hour is also feed eligible, but the restriction is applied rather than
    assumed so the comparison stays matched if that ever changes.

    Final test rows are carried through untouched. They are never fitted
    on or scored; they are retained so the leakage guards still have the
    committed test period to check against.
    """
    development = assignment[assignment["fold_kind"] == KIND_DEVELOPMENT]
    other = assignment[assignment["fold_kind"] != KIND_DEVELOPMENT]

    kept = development[development[TIMESTAMP_COLUMN].isin(usable_timestamps)]
    parts = [frame for frame in (kept, other) if not frame.empty]
    restricted = pd.concat(parts, ignore_index=True) if parts else assignment.iloc[:0].copy()
    return restricted.sort_values(
        ["fold_id", "role", TIMESTAMP_COLUMN], kind="mergesort"
    ).reset_index(drop=True)


def summarize_row_loss(assignment: pd.DataFrame, restricted: pd.DataFrame) -> pd.DataFrame:
    """Rows removed per fold and role because feed chemistry is unavailable."""

    def counts(frame: pd.DataFrame) -> pd.Series:
        development = frame[frame["fold_kind"] == KIND_DEVELOPMENT]
        return development.groupby(["fold_id", "role"]).size()

    before = counts(assignment)
    after = counts(restricted).reindex(before.index, fill_value=0)

    table = pd.DataFrame(
        {"n_committed": before, "n_retained": after, "n_dropped": before - after}
    ).reset_index()
    return table.sort_values(["fold_id", "role"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------
# Fold training and scoring
# ---------------------------------------------------------------------


def evaluate_fold(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_id: int,
    configuration: str,
    predictors: list[str] | None = None,
) -> FeedFoldResult:
    """Fit the unchanged Random Forest on one fold and score its validation rows.

    The pipeline and every hyperparameter come from
    `src.models.random_forest.build_pipeline` with its defaults, so the
    model is identical to the committed benchmark in both arms and no
    tuning is performed here.
    """
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this experiment.")

    predictors = get_predictors(configuration) if predictors is None else list(predictors)
    validate_configuration_scope(configuration, predictors)

    frames = get_fold_frames(hourly, assignment, fold_id)

    train_spark = to_spark(spark, frames.train, predictors)
    validation_spark = to_spark(spark, frames.validation, predictors)

    if train_spark.count() != len(frames.train):
        raise ValueError(f"Fold {fold_id}: training row count changed during Spark conversion.")
    if validation_spark.count() != len(frames.validation):
        raise ValueError(f"Fold {fold_id}: validation row count changed during Spark conversion.")

    fitted = build_pipeline(predictors).fit(train_spark)

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
    importances = np.asarray(fitted.stages[-1].featureImportances.toArray(), dtype=float)

    return FeedFoldResult(
        configuration=configuration,
        fold_id=fold_id,
        metrics=metrics,
        n_train=len(frames.train),
        n_validation=len(frames.validation),
        n_scored=len(predictions),
        n_features=len(predictors),
        feature_importances=importances,
        predictions=predictions,
    )


def evaluate_configuration(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: str,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> FeedEvaluation:
    """Evaluate one predictor configuration across every development fold."""
    predictors = get_predictors(configuration)
    fold_results = [
        evaluate_fold(spark, hourly, assignment, fold_id, configuration, predictors=predictors)
        for fold_id in fold_ids
    ]

    results = pd.DataFrame(
        [
            {
                "configuration": result.configuration,
                "fold_id": result.fold_id,
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
    return FeedEvaluation(
        configuration=configuration,
        predictors=predictors,
        results=results,
        fold_results=fold_results,
    )


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    evaluation: FeedEvaluation,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    committed_assignment: pd.DataFrame,
) -> None:
    """Verify every structural guard, raising a clear error on violation."""
    if list(evaluation.results.columns) != RESULT_COLUMNS:
        raise ValueError(
            f"Result schema mismatch. Expected {RESULT_COLUMNS}, "
            f"got {list(evaluation.results.columns)}"
        )

    validate_configuration_scope(evaluation.configuration, evaluation.predictors)
    expected_features = len(evaluation.predictors)
    uses_feed = evaluation.configuration == FEED_ENHANCED

    eligible = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    feed_eligible = feed_eligible_timestamps(hourly)
    interpolated = set(hourly.loc[hourly[INTERPOLATED_COLUMN].astype(bool), TIMESTAMP_COLUMN])
    final_test = set(
        committed_assignment.loc[committed_assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    )
    committed_pairs = set(
        zip(
            committed_assignment["fold_id"],
            committed_assignment[TIMESTAMP_COLUMN],
            committed_assignment["role"],
        )
    )

    for result in evaluation.fold_results:
        fold_id = result.fold_id
        frames = get_fold_frames(hourly, assignment, fold_id)

        train_timestamps = set(frames.train[TIMESTAMP_COLUMN])
        validation_timestamps = set(frames.validation[TIMESTAMP_COLUMN])
        embargo_timestamps = set(frames.embargo[TIMESTAMP_COLUMN])

        # A restriction may only remove rows. Every surviving row must hold
        # the same role in the same fold as the committed split.
        for role, timestamps in (
            (ROLE_TRAIN, train_timestamps),
            (ROLE_VALIDATION, validation_timestamps),
        ):
            invented = {
                timestamp
                for timestamp in timestamps
                if (fold_id, timestamp, role) not in committed_pairs
            }
            if invented:
                raise ValueError(
                    f"Fold {fold_id}: {len(invented)} {role} row(s) were not committed to that role."
                )

        if not train_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: training contains ineligible hours.")
        if not validation_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: validation contains ineligible hours.")
        if uses_feed:
            if not train_timestamps.issubset(feed_eligible):
                raise ValueError(f"Fold {fold_id}: training contains feed ineligible hours.")
            if not validation_timestamps.issubset(feed_eligible):
                raise ValueError(f"Fold {fold_id}: validation contains feed ineligible hours.")

        if train_timestamps.intersection(interpolated):
            raise ValueError(f"Fold {fold_id}: training contains an interpolated target row.")
        if validation_timestamps.intersection(interpolated):
            raise ValueError(f"Fold {fold_id}: validation contains an interpolated target row.")

        if train_timestamps.intersection(validation_timestamps):
            raise ValueError(f"Fold {fold_id}: training and validation timestamps overlap.")
        if train_timestamps.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was used for fitting.")
        if train_timestamps.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: training overlaps the final test period.")
        if max(train_timestamps) >= min(validation_timestamps):
            raise ValueError(f"Fold {fold_id}: training is not strictly earlier than validation.")

        scored = set(result.predictions[TIMESTAMP_COLUMN])
        if scored != validation_timestamps:
            raise ValueError(f"Fold {fold_id}: scored hours differ from the validation window.")
        if scored.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was scored.")
        if scored.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: a final test hour was scored.")
        if not result.predictions[TIMESTAMP_COLUMN].is_monotonic_increasing:
            raise ValueError(f"Fold {fold_id}: predictions are not in chronological order.")

        if result.n_scored != result.n_validation:
            raise ValueError(
                f"Fold {fold_id}: scored {result.n_scored} of {result.n_validation} rows."
            )
        if result.n_features != expected_features:
            raise ValueError(
                f"Fold {fold_id}: model used {result.n_features} features, not {expected_features}."
            )
        for metric, value in result.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"Fold {fold_id}: {metric} is not finite.")

        importances = result.feature_importances
        if len(importances) != expected_features:
            raise ValueError(
                f"Fold {fold_id}: importance vector has {len(importances)} entries, "
                f"not {expected_features}."
            )
        if not np.isfinite(importances).all():
            raise ValueError(f"Fold {fold_id}: feature importances contain non-finite values.")
        if (importances < 0).any():
            raise ValueError(f"Fold {fold_id}: a feature importance is negative.")
        if not np.isclose(float(importances.sum()), 1.0, atol=1e-4):
            raise ValueError(
                f"Fold {fold_id}: feature importances sum to {importances.sum():.6f}, expected ~1.0."
            )


def assert_matched_rows(sensor_only: FeedEvaluation, feed_enhanced: FeedEvaluation) -> None:
    """Confirm both arms were fitted and scored on identical rows."""
    left_by_fold = {result.fold_id: result for result in sensor_only.fold_results}
    right_by_fold = {result.fold_id: result for result in feed_enhanced.fold_results}

    if set(left_by_fold) != set(right_by_fold):
        raise ValueError("The two configurations do not cover the same folds.")

    for fold_id in sorted(left_by_fold):
        left, right = left_by_fold[fold_id], right_by_fold[fold_id]
        if left.n_train != right.n_train:
            raise ValueError(
                f"Fold {fold_id}: training rows differ between configurations "
                f"({left.n_train} against {right.n_train})."
            )
        if set(left.predictions[TIMESTAMP_COLUMN]) != set(right.predictions[TIMESTAMP_COLUMN]):
            raise ValueError(f"Fold {fold_id}: the two configurations scored different hours.")
        if not np.allclose(
            left.predictions[TARGET_COLUMN].to_numpy(),
            right.predictions[TARGET_COLUMN].to_numpy(),
        ):
            raise ValueError(f"Fold {fold_id}: the two configurations saw different targets.")


def verify_reproduces_committed_benchmark(
    results: pd.DataFrame,
    random_forest_results: pd.DataFrame,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Confirm the sensor only arm reproduces the committed benchmark.

    The sensor only arm changes nothing about the model or the data, so it
    must reproduce the existing Random Forest figures. If it does not,
    this harness has altered something it was supposed to hold constant
    and no comparison built on it can be trusted.
    """
    merged = results.merge(
        random_forest_results[["fold_id", "rmse", "mae", "r2", "n_train", "n_validation"]],
        on="fold_id",
        suffixes=("", "_committed"),
    )
    if len(merged) != len(results):
        raise ValueError(
            "The sensor only arm does not cover the same folds as the committed benchmark."
        )

    for _, row in merged.iterrows():
        for column in ("n_train", "n_validation"):
            if int(row[column]) != int(row[f"{column}_committed"]):
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: sensor only {column} is {int(row[column])}, but "
                    f"the committed benchmark used {int(row[f'{column}_committed'])}."
                )
        for metric in ("rmse", "mae", "r2"):
            if abs(float(row[metric]) - float(row[f"{metric}_committed"])) > tolerance:
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: sensor only {metric} is {row[metric]:.10f}, but "
                    f"the committed Random Forest benchmark recorded "
                    f"{row[f'{metric}_committed']:.10f}."
                )


def verify_random_forest_configuration_unchanged(predictors: list[str]) -> dict:
    """Confirm the pipeline still carries the committed benchmark settings.

    The experiment claims the model is held constant, so the fitted
    estimator's own parameters are read back from a freshly built pipeline
    and compared with the values the benchmark module declares.
    """
    from src.models import random_forest

    forest = build_pipeline(predictors).getStages()[-1]
    observed = {
        "numTrees": forest.getNumTrees(),
        "maxDepth": forest.getMaxDepth(),
        "minInstancesPerNode": forest.getMinInstancesPerNode(),
        "featureSubsetStrategy": forest.getFeatureSubsetStrategy(),
        "subsamplingRate": forest.getSubsamplingRate(),
        "seed": forest.getSeed(),
    }
    expected = {
        "numTrees": random_forest.NUM_TREES,
        "maxDepth": random_forest.MAX_DEPTH,
        "minInstancesPerNode": random_forest.MIN_INSTANCES_PER_NODE,
        "featureSubsetStrategy": random_forest.FEATURE_SUBSET_STRATEGY,
        "subsamplingRate": random_forest.SUBSAMPLING_RATE,
        "seed": random_forest.SEED,
    }
    if observed != expected:
        raise ValueError(
            f"The Random Forest configuration has changed. Expected {expected}, got {observed}."
        )
    return observed


def verify_deterministic_evaluation(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: str,
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Fit one fold twice and confirm the metrics and importances agree."""
    first = evaluate_fold(spark, hourly, assignment, fold_id, configuration)
    second = evaluate_fold(spark, hourly, assignment, fold_id, configuration)

    for metric in ("rmse", "mae", "r2"):
        if abs(first.metrics[metric] - second.metrics[metric]) > tolerance:
            raise ValueError(
                f"{configuration} fold {fold_id}: {metric} is not reproducible "
                f"({first.metrics[metric]} against {second.metrics[metric]})."
            )
    if not np.allclose(first.feature_importances, second.feature_importances, atol=tolerance):
        raise ValueError(f"{configuration} fold {fold_id}: feature importances are not reproducible.")


# ---------------------------------------------------------------------
# Comparison and analysis
# ---------------------------------------------------------------------


def combine_results(sensor_only: FeedEvaluation, feed_enhanced: FeedEvaluation) -> pd.DataFrame:
    """Fold level results for both arms in one table."""
    combined = pd.concat([sensor_only.results, feed_enhanced.results], ignore_index=True)
    return combined.sort_values(
        ["configuration", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


def compare_configurations(combined: pd.DataFrame) -> pd.DataFrame:
    """Per fold difference between the feed enhanced and sensor only arms.

    A negative RMSE or MAE difference and a positive R squared difference
    mean feed chemistry predicts better.
    """
    sensor = combined[combined["configuration"] == SENSOR_ONLY]
    feed = combined[combined["configuration"] == FEED_ENHANCED]

    merged = feed.merge(
        sensor[["fold_id", "rmse", "mae", "r2"]], on="fold_id", suffixes=("", "_sensor_only")
    )
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_sensor_only"]
    merged["mae_difference"] = merged["mae"] - merged["mae_sensor_only"]
    merged["r2_difference"] = merged["r2"] - merged["r2_sensor_only"]
    merged["feed_better"] = merged["rmse_difference"] < 0

    columns = [
        "fold_id",
        "n_train",
        "n_validation",
        "rmse",
        "rmse_sensor_only",
        "rmse_difference",
        "mae_difference",
        "r2_difference",
        "feed_better",
    ]
    return merged[columns].sort_values("fold_id", kind="mergesort").reset_index(drop=True)


def summarize_configurations(combined: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds."""
    summary = (
        combined.groupby("configuration")
        .agg(
            n_folds=("fold_id", "nunique"),
            n_features=("n_features", "max"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )
    return summary.sort_values("configuration", kind="mergesort").reset_index(drop=True)


def classify_feed_value(comparison: pd.DataFrame, meaningful_rmse: float = MEANINGFUL_RMSE) -> dict:
    """Apply the experiment's decision rule to the matched comparison.

    Strong value requires an improvement on every development fold and a
    mean improvement of at least `meaningful_rmse`. Little or no value
    means the aggregate gain does not reach that margin and at most one
    fold improves at all. Anything between the two is moderate or mixed.
    """
    n_folds = len(comparison)
    if n_folds == 0:
        raise ValueError("Cannot classify feed value from an empty comparison.")

    n_improved = int((comparison["rmse_difference"] < 0).sum())
    mean_difference = float(comparison["rmse_difference"].mean())
    improves_everywhere = n_improved == n_folds

    if improves_everywhere and mean_difference <= -meaningful_rmse:
        classification = VALUE_STRONG
    elif mean_difference > -meaningful_rmse and n_improved <= 1:
        classification = VALUE_NONE
    else:
        classification = VALUE_MODERATE

    return {
        "n_folds": n_folds,
        "n_folds_improved": n_improved,
        "improves_on_every_fold": improves_everywhere,
        "mean_rmse_difference": mean_difference,
        "mean_mae_difference": float(comparison["mae_difference"].mean()),
        "mean_r2_difference": float(comparison["r2_difference"].mean()),
        "worst_fold_rmse_difference": float(comparison["rmse_difference"].max()),
        "best_fold_rmse_difference": float(comparison["rmse_difference"].min()),
        "classification": classification,
    }


def excursion_analysis(
    sensor_only: FeedEvaluation,
    feed_enhanced: FeedEvaluation,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    quantile: float = EXCURSION_QUANTILE,
) -> pd.DataFrame:
    """Split each fold's error by whether the hour is a high silica excursion.

    The threshold is the fold's own training target quantile, so it uses
    only information a model would have had. Both arms are measured on the
    same hours within each group.
    """
    rows = []
    sensor_by_fold = {result.fold_id: result for result in sensor_only.fold_results}
    feed_by_fold = {result.fold_id: result for result in feed_enhanced.fold_results}

    for fold_id in sorted(sensor_by_fold):
        frames = get_fold_frames(hourly, assignment, fold_id)
        threshold = float(frames.train[TARGET_COLUMN].quantile(quantile))

        for configuration, result in (
            (SENSOR_ONLY, sensor_by_fold[fold_id]),
            (FEED_ENHANCED, feed_by_fold[fold_id]),
        ):
            predictions = result.predictions
            is_excursion = predictions[TARGET_COLUMN] >= threshold
            for label, subset in (
                ("excursion", predictions[is_excursion]),
                ("normal", predictions[~is_excursion]),
            ):
                if subset.empty:
                    continue
                observed = subset[TARGET_COLUMN].to_numpy(dtype=float)
                predicted = subset[PREDICTION_COLUMN].to_numpy(dtype=float)
                residual = observed - predicted
                rows.append(
                    {
                        "fold_id": fold_id,
                        "configuration": configuration,
                        "group": label,
                        "threshold": threshold,
                        "n": len(subset),
                        "observed_mean": float(observed.mean()),
                        "predicted_mean": float(predicted.mean()),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "mae": float(np.mean(np.abs(residual))),
                        "residual_mean": float(residual.mean()),
                    }
                )

    table = pd.DataFrame(rows)
    return table.sort_values(
        ["fold_id", "group", "configuration"], kind="mergesort"
    ).reset_index(drop=True)


def compare_excursions(excursions: pd.DataFrame) -> pd.DataFrame:
    """Feed enhanced against sensor only error within each fold and group."""
    sensor = excursions[excursions["configuration"] == SENSOR_ONLY]
    feed = excursions[excursions["configuration"] == FEED_ENHANCED]
    merged = feed.merge(
        sensor[["fold_id", "group", "rmse", "mae", "predicted_mean"]],
        on=["fold_id", "group"],
        suffixes=("", "_sensor_only"),
    )
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_sensor_only"]
    merged["mae_difference"] = merged["mae"] - merged["mae_sensor_only"]
    columns = [
        "fold_id",
        "group",
        "n",
        "threshold",
        "observed_mean",
        "predicted_mean_sensor_only",
        "predicted_mean",
        "rmse_sensor_only",
        "rmse",
        "rmse_difference",
        "mae_difference",
    ]
    return merged[columns].sort_values(["group", "fold_id"], kind="mergesort").reset_index(drop=True)


def prediction_spread(
    evaluation: FeedEvaluation, hourly: pd.DataFrame, assignment: pd.DataFrame
) -> pd.DataFrame:
    """Describe how wide each arm's predictions are and how far they collapse.

    A model that emits a nearly constant number can post a respectable
    RMSE while carrying no operational information. Spread is reported
    next to the fold's training mean, because collapsing onto that
    constant is the specific failure the earlier diagnostics found.
    """
    rows = []
    for result in evaluation.fold_results:
        frames = get_fold_frames(hourly, assignment, result.fold_id)
        train_mean = float(frames.train[TARGET_COLUMN].mean())
        train_std = float(frames.train[TARGET_COLUMN].std(ddof=0))

        predictions = result.predictions
        predicted = predictions[PREDICTION_COLUMN].to_numpy(dtype=float)
        observed = predictions[TARGET_COLUMN].to_numpy(dtype=float)
        observed_std = float(observed.std(ddof=0))

        constant_rmse = float(np.sqrt(np.mean((observed - train_mean) ** 2)))
        rows.append(
            {
                "configuration": evaluation.configuration,
                "fold_id": result.fold_id,
                "n": len(predictions),
                "prediction_std": float(predicted.std(ddof=0)),
                "prediction_iqr": float(
                    np.quantile(predicted, 0.75) - np.quantile(predicted, 0.25)
                ),
                "prediction_min": float(predicted.min()),
                "prediction_max": float(predicted.max()),
                "prediction_range": float(predicted.max() - predicted.min()),
                "observed_std": observed_std,
                "observed_range": float(observed.max() - observed.min()),
                "spread_ratio": float(predicted.std(ddof=0) / observed_std)
                if observed_std > 0
                else float("nan"),
                "train_mean": train_mean,
                "train_std": train_std,
                "predicted_mean_minus_train_mean": float(predicted.mean() - train_mean),
                "share_within_quarter_sd_of_train_mean": float(
                    np.mean(np.abs(predicted - train_mean) <= COLLAPSE_TOLERANCE_SD * observed_std)
                ),
                "training_mean_constant_rmse": constant_rmse,
                "rmse_over_constant_rmse": float(
                    np.sqrt(np.mean((observed - predicted) ** 2)) / constant_rmse
                ),
                "observed_vs_predicted_spearman": float(
                    predictions[TARGET_COLUMN].corr(
                        predictions[PREDICTION_COLUMN], method="spearman"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def feed_importance_table(evaluation: FeedEvaluation) -> pd.DataFrame:
    """Importance and rank of each feed column within the fitted forest.

    Rank 1 is the largest importance. Importance describes how often and
    how usefully the ensemble split on a feature given everything else it
    had available. It is not a causal effect and is not evidence that
    changing feed chemistry would change the outcome by any amount.
    """
    predictors = evaluation.predictors
    rows = []
    for result in evaluation.fold_results:
        importances = result.feature_importances
        order = np.argsort(-importances, kind="mergesort")
        rank_by_index = {int(index): position + 1 for position, index in enumerate(order)}

        for column in FEED_COLUMNS:
            if column not in predictors:
                continue
            index = predictors.index(column)
            rows.append(
                {
                    "configuration": evaluation.configuration,
                    "fold_id": result.fold_id,
                    "feature": column,
                    "importance": float(importances[index]),
                    "rank": rank_by_index[index],
                    "n_features": len(predictors),
                    "share_of_total_importance": float(importances[index] / importances.sum()),
                }
            )
    return pd.DataFrame(rows)


def top_importance_table(evaluation: FeedEvaluation, top_n: int = 8) -> pd.DataFrame:
    """The largest importances per fold, for context around the feed ranks."""
    predictors = evaluation.predictors
    rows = []
    for result in evaluation.fold_results:
        importances = result.feature_importances
        order = np.argsort(-importances, kind="mergesort")[:top_n]
        for position, index in enumerate(order, start=1):
            rows.append(
                {
                    "configuration": evaluation.configuration,
                    "fold_id": result.fold_id,
                    "rank": position,
                    "feature": predictors[int(index)],
                    "importance": float(importances[int(index)]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Paths, report, CLI
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_random_forest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def default_raw_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "feed_chemistry_results.parquet"


def format_report(report: dict) -> str:
    lines = ["Feed chemistry information value (development folds only)", ""]

    lines.append("Feed column representation in the hourly dataset")
    for _, row in report["feed_columns"].iterrows():
        lines.append(
            f"  {row['column']:<12} from {row['raw_source']!r}: {int(row['n_distinct'])} distinct "
            f"values, range {row['min']:.2f} to {row['max']:.2f}, mean {row['mean']:.2f}, "
            f"{int(row['n_missing'])} missing, {int(row['n_hours_inconsistent'])} inconsistent hours"
        )
    lines.append(f"  raw provenance check: {report['provenance']}")

    lines.extend(["", "Original sensor only benchmark on the full committed assignment", ""])
    lines.append(
        f"  {'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'feats':>5}  {'rmse':>7}  "
        f"{'mae':>7}  {'r2':>8}"
    )
    for _, row in report["unrestricted_sensor_only"].iterrows():
        lines.append(
            f"  {int(row['fold_id']):>4}  {int(row['n_train']):>7,}  "
            f"{int(row['n_validation']):>5,}  {int(row['n_features']):>5}  {row['rmse']:>7.4f}  "
            f"{row['mae']:>7.4f}  {row['r2']:>8.4f}"
        )

    lines.extend(["", "Rows removed because feed chemistry is unavailable", ""])
    dropped = report["row_loss"][report["row_loss"]["n_dropped"] > 0]
    if dropped.empty:
        lines.append("  none: every sensor eligible hour also carries feed chemistry")
    else:
        for _, row in dropped.iterrows():
            lines.append(
                f"  fold {int(row['fold_id'])} {row['role']}: {int(row['n_dropped'])} of "
                f"{int(row['n_committed'])}"
            )

    lines.extend(["", "Matched comparison on identical rows", ""])
    lines.append(
        f"  {'configuration':<14}  {'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'feats':>5}  "
        f"{'rmse':>7}  {'mae':>7}  {'r2':>8}"
    )
    for _, row in report["combined"].iterrows():
        lines.append(
            f"  {row['configuration']:<14}  {int(row['fold_id']):>4}  "
            f"{int(row['n_train']):>7,}  {int(row['n_validation']):>5,}  "
            f"{int(row['n_features']):>5}  {row['rmse']:>7.4f}  {row['mae']:>7.4f}  "
            f"{row['r2']:>8.4f}"
        )

    lines.extend(["", "Aggregate across folds", ""])
    for _, row in report["summary"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} ({int(row['n_features'])} features)  "
            f"RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(
        ["", "Feed enhanced against sensor only (negative RMSE difference favours feed)", ""]
    )
    for _, row in report["comparison"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: {row['rmse']:.4f} against "
            f"{row['rmse_sensor_only']:.4f}  RMSE {row['rmse_difference']:+.4f}  "
            f"MAE {row['mae_difference']:+.4f}  R2 {row['r2_difference']:+.4f}"
        )
    value = report["value"]
    lines.append(
        f"  improved on {value['n_folds_improved']}/{value['n_folds']} folds, "
        f"mean RMSE difference {value['mean_rmse_difference']:+.4f}, "
        f"mean MAE difference {value['mean_mae_difference']:+.4f}, "
        f"mean R2 difference {value['mean_r2_difference']:+.4f}"
    )

    lines.extend(["", "High silica excursion hours against the rest of the window", ""])
    for _, row in report["excursion_comparison"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['group']:<9} (n {int(row['n']):>4}, "
            f"threshold {row['threshold']:.3f}, observed mean {row['observed_mean']:.3f}): "
            f"sensor only {row['rmse_sensor_only']:.4f} -> feed {row['rmse']:.4f}  "
            f"({row['rmse_difference']:+.4f})"
        )

    lines.extend(["", "Prediction distribution and collapse toward the training mean", ""])
    for _, row in report["spread"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} fold {int(row['fold_id'])}: sd {row['prediction_std']:.4f} "
            f"(observed sd {row['observed_std']:.4f}, ratio {row['spread_ratio']:.3f}), "
            f"range {row['prediction_range']:.4f} (observed {row['observed_range']:.4f}), "
            f"rank agreement {row['observed_vs_predicted_spearman']:+.3f}"
        )
        lines.append(
            f"    {row['share_within_quarter_sd_of_train_mean']:.1%} of predictions sit within "
            f"{COLLAPSE_TOLERANCE_SD} observed sd of the training mean "
            f"({row['train_mean']:.3f}); model RMSE is "
            f"{row['rmse_over_constant_rmse']:.3f} of the constant baseline RMSE"
        )

    lines.extend(["", "Feed chemistry importance in the feed enhanced forest", ""])
    for _, row in report["feed_importance"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['feature']:<12}: importance "
            f"{row['importance']:.4f}, rank {int(row['rank'])} of {int(row['n_features'])}"
        )
    lines.append("  Importance describes ensemble splitting behaviour, not a causal effect.")

    lines.extend(["", "Largest importances per fold, for context", ""])
    for fold_id in sorted(report["top_importance"]["fold_id"].unique()):
        subset = report["top_importance"][report["top_importance"]["fold_id"] == fold_id]
        pairs = ", ".join(
            f"{row['feature']}={row['importance']:.4f}" for _, row in subset.head(5).iterrows()
        )
        lines.append(f"  fold {int(fold_id)}: {pairs}")

    lines.extend(["", f"Classification: {value['classification']}"])
    return "\n".join(lines)


def run(
    hourly_path: Path,
    splits_path: Path,
    random_forest_path: Path,
    results_path: Path,
    raw_path: Path | None = None,
    spark=None,
) -> dict:
    """Run both arms, validate every guard, compare, and write the results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityFeedChemistryValue")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)

        if not random_forest_path.exists():
            raise FileNotFoundError(
                f"Random Forest benchmark results not found at: {random_forest_path}. "
                "Run that module first."
            )
        committed_random_forest = pd.read_parquet(random_forest_path)

        assert_feed_columns_are_usable(hourly)
        provenance = "skipped, raw file not available"
        if raw_path is not None and raw_path.exists():
            from src.data.preprocess import load_raw

            verify_feed_values_match_the_raw_record(hourly, load_raw(raw_path))
            provenance = "hourly values match the unmodified raw observations"

        verify_random_forest_configuration_unchanged(get_feed_enhanced_predictors())

        restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
        row_loss = summarize_row_loss(assignment, restricted)

        # The sensor only arm on the untouched assignment must reproduce
        # the committed benchmark before anything else is read from it.
        unrestricted = evaluate_configuration(spark, hourly, assignment, SENSOR_ONLY)
        validate_evaluation(unrestricted, hourly, assignment, assignment)
        verify_reproduces_committed_benchmark(unrestricted.results, committed_random_forest)

        sensor_only = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY)
        feed_enhanced = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED)

        for evaluation in (sensor_only, feed_enhanced):
            validate_evaluation(evaluation, hourly, restricted, assignment)
        assert_matched_rows(sensor_only, feed_enhanced)
        verify_deterministic_evaluation(spark, hourly, restricted, FEED_ENHANCED)

        combined = combine_results(sensor_only, feed_enhanced)
        comparison = compare_configurations(combined)
        excursions = excursion_analysis(sensor_only, feed_enhanced, hourly, restricted)

        report = {
            "feed_columns": describe_feed_columns(hourly),
            "provenance": provenance,
            "unrestricted_sensor_only": unrestricted.results,
            "row_loss": row_loss,
            "combined": combined,
            "summary": summarize_configurations(combined),
            "comparison": comparison,
            "value": classify_feed_value(comparison),
            "excursions": excursions,
            "excursion_comparison": compare_excursions(excursions),
            "spread": pd.concat(
                [
                    prediction_spread(sensor_only, hourly, restricted),
                    prediction_spread(feed_enhanced, hourly, restricted),
                ],
                ignore_index=True,
            ),
            "feed_importance": feed_importance_table(feed_enhanced),
            "top_importance": top_importance_table(feed_enhanced),
        }

        results_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(results_path, index=False)

        return report
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and evaluating feed chemistry information value...")
    report = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_random_forest_path(repo_root),
        results_path,
        raw_path=default_raw_path(repo_root),
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(report))


if __name__ == "__main__":
    sys.exit(main())
