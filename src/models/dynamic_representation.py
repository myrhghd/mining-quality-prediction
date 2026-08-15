"""Static against dynamic sensor representation experiment.

The generalization diagnostics established that the 57 static hourly
sensor aggregates carry weak and unstable forward information about
`% Silica Concentrate`. One explanation those diagnostics could not test
is that a single hour, summarized in isolation, simply discards the
process history an operator would actually read: where inside the hour
the plant ended up, how far it has moved since the previous hour, and
how steady it has been over the last two hours.

This module tests that directly. Two feature representations are
evaluated:

* the static control, the existing 57 core sensor aggregates
* the dynamic representation, those same 57 aggregates plus the 96
  backward looking process history features built by
  `src.data.dynamic_features`

Everything else is held constant. The target, the 0 hour alignment, the
committed development folds, the embargo, the final test methodology,
the Random Forest configuration and its hyperparameters, and the metric
implementations are all imported unchanged. Feature representation is
the only variable.

Matched rows
------------
Six hours cannot be given a full 120 minute window, because they open a
temporal segment or follow a frozen sensor hour. Those hours are removed
from the development assignment, and the static control is refitted and
rescored on exactly the surviving rows. The headline static benchmark on
the full committed assignment is reported alongside for context, but the
decision is taken from the matched comparison, where the only difference
between the two arms is the feature vector.

Model scope
-----------
`src.models.random_forest` validates that its predictor set is exactly
the 57 approved sensor aggregates, so its fold runner cannot accept a
wider vector. Its pipeline builder and its fixed hyperparameters are
imported and used unchanged here, and this module supplies its own fold
runner and its own scope guard. No Random Forest parameter is tuned.

The final test period is never fitted on, never scored, and never used
to choose a representation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dynamic_features import (
    DYNAMIC_PREDICTOR_COLUMNS,
    HAS_CONTEXT_COLUMN,
    get_dynamic_predictor_columns,
)
from src.data.preprocess import (
    CORE_SENSOR_PREDICTOR_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    find_repo_root,
)
from src.data.split import (
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
from src.models.generalization_diagnostics import residual_subperiods
from src.models.linear_regression import (
    NUMERICAL_TOLERANCE,
    build_spark_session,
    get_sensor_predictors,
    to_spark,
    validate_predictor_scope,
)
from src.models.random_forest import PREDICTION_COLUMN, build_pipeline

STATIC_REPRESENTATION = "static"
DYNAMIC_REPRESENTATION = "dynamic"
REPRESENTATIONS = (STATIC_REPRESENTATION, DYNAMIC_REPRESENTATION)

RESULT_COLUMNS = [
    "representation",
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
# is treated as a high silica excursion. The threshold comes from the
# training window only, so the grouping uses nothing that would be
# unavailable at prediction time. It is a reporting cut applied after
# measurement, never a value any model is fitted against.
EXCURSION_QUANTILE = 0.90

# Mean RMSE improvement below which a gain is treated as too small to
# act on. Matches the reporting threshold the alignment experiment
# already uses, so the two experiments are read on the same scale.
MEANINGFUL_RMSE = 0.01

SUPPORT_STRONG = "strong support for temporal modeling"
SUPPORT_WEAK = "weak or mixed support"
SUPPORT_NONE = "no support"


@dataclass(frozen=True)
class RepresentationFoldResult:
    representation: str
    fold_id: int
    metrics: dict[str, float]
    n_train: int
    n_validation: int
    n_scored: int
    n_features: int
    predictions: pd.DataFrame


@dataclass(frozen=True)
class RepresentationEvaluation:
    representation: str
    predictors: list[str]
    results: pd.DataFrame
    fold_results: list[RepresentationFoldResult]


# ---------------------------------------------------------------------
# Predictor sets
# ---------------------------------------------------------------------


def get_static_predictors() -> list[str]:
    """Return the 57 core sensor aggregates, the unchanged control."""
    return get_sensor_predictors()


def get_dynamic_predictors() -> list[str]:
    """Return the 57 static aggregates followed by the 96 history features."""
    return get_sensor_predictors() + get_dynamic_predictor_columns()


def get_predictors(representation: str) -> list[str]:
    if representation == STATIC_REPRESENTATION:
        return get_static_predictors()
    if representation == DYNAMIC_REPRESENTATION:
        return get_dynamic_predictors()
    raise ValueError(f"Unknown representation: {representation!r}")


def validate_representation_scope(representation: str, predictors: list[str]) -> None:
    """Fail if a predictor set contains anything outside its declared scope.

    The static arm is delegated to the existing guard so the control is
    verified by exactly the check the committed benchmark uses. The
    dynamic arm is checked against the same forbidden set, extended with
    every calendar derived name this experiment rules out.
    """
    if len(set(predictors)) != len(predictors):
        raise ValueError("Predictor list contains duplicates.")

    if representation == STATIC_REPRESENTATION:
        validate_predictor_scope(predictors)
        return

    if representation != DYNAMIC_REPRESENTATION:
        raise ValueError(f"Unknown representation: {representation!r}")

    expected = list(CORE_SENSOR_PREDICTOR_COLUMNS) + list(DYNAMIC_PREDICTOR_COLUMNS)
    if predictors != expected:
        missing = sorted(set(expected) - set(predictors))
        unexpected = sorted(set(predictors) - set(expected))
        raise ValueError(
            "The dynamic predictor set must be the 57 static aggregates followed by the "
            f"96 history features. Missing: {missing}; unexpected: {unexpected}"
        )

    forbidden = set(FEED_CONTEXT_PREDICTOR_COLUMNS) | {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        SEGMENT_COLUMN,
        SENSOR_ELIGIBLE_COLUMN,
        INTERPOLATED_COLUMN,
        HAS_CONTEXT_COLUMN,
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
        # Calendar derived values are ruled out for this experiment.
        "date",
        "month",
        "day_of_week",
        "hour_of_day",
        "year",
        "week_of_year",
    }
    leaked = forbidden.intersection(predictors)
    if leaked:
        raise ValueError(f"Forbidden columns present in the predictor set: {sorted(leaked)}")


# ---------------------------------------------------------------------
# Matched dataset construction
# ---------------------------------------------------------------------


def join_dynamic_features(hourly: pd.DataFrame, dynamic: pd.DataFrame) -> pd.DataFrame:
    """Attach the dynamic feature columns to the hourly table.

    The join is on the hour alone, so no row can pick up a feature vector
    built for a different hour. Every hourly row must be matched exactly
    once; a partial join would silently change which rows a fold holds.
    """
    required = [TIMESTAMP_COLUMN, *DYNAMIC_PREDICTOR_COLUMNS, HAS_CONTEXT_COLUMN]
    missing = [column for column in required if column not in dynamic.columns]
    if missing:
        raise ValueError(f"Dynamic feature table is missing column(s): {missing}")

    overlap = set(hourly.columns).intersection(DYNAMIC_PREDICTOR_COLUMNS)
    if overlap:
        raise ValueError(
            f"The hourly table already defines dynamic feature name(s): {sorted(overlap)}"
        )

    joined = hourly.merge(dynamic[required], on=TIMESTAMP_COLUMN, how="left", validate="one_to_one")
    if len(joined) != len(hourly):
        raise ValueError("Joining the dynamic features changed the hourly row count.")
    if joined[HAS_CONTEXT_COLUMN].isna().any():
        raise ValueError("An hourly row has no dynamic feature record.")

    joined[HAS_CONTEXT_COLUMN] = joined[HAS_CONTEXT_COLUMN].astype(bool)
    return joined.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)


def restrict_assignment(assignment: pd.DataFrame, usable_timestamps: set) -> pd.DataFrame:
    """Drop development rows whose dynamic feature window is incomplete.

    Both arms are then trained and scored on the identical row set, which
    is what makes the comparison attributable to the feature vector.

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
    """Rows removed per fold and role because the history window is incomplete."""

    def counts(frame: pd.DataFrame) -> pd.Series:
        development = frame[frame["fold_kind"] == KIND_DEVELOPMENT]
        return development.groupby(["fold_id", "role"]).size()

    before = counts(assignment)
    after = counts(restricted).reindex(before.index, fill_value=0)

    table = pd.DataFrame(
        {"n_committed": before, "n_retained": after, "n_dropped": before - after}
    ).reset_index()
    return table.sort_values(["fold_id", "role"], kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class MatchedDataset:
    hourly: pd.DataFrame
    assignment: pd.DataFrame
    row_loss: pd.DataFrame


def build_matched_dataset(
    hourly: pd.DataFrame, dynamic: pd.DataFrame, assignment: pd.DataFrame
) -> MatchedDataset:
    """Join the history features and restrict the folds to usable hours."""
    joined = join_dynamic_features(hourly, dynamic)
    usable = set(joined.loc[joined[HAS_CONTEXT_COLUMN], TIMESTAMP_COLUMN])
    restricted = restrict_assignment(assignment, usable)
    return MatchedDataset(
        hourly=joined,
        assignment=restricted,
        row_loss=summarize_row_loss(assignment, restricted),
    )


# ---------------------------------------------------------------------
# Leakage guards on the feature construction itself
# ---------------------------------------------------------------------


def assert_features_are_backward_looking(
    hourly: pd.DataFrame, dynamic: pd.DataFrame, raw: pd.DataFrame
) -> None:
    """Confirm no dynamic feature changes when future hours are removed.

    Rebuilding the table from a raw record truncated at hour `t` must
    reproduce the values already computed for every hour up to `t`. A
    feature that read anything from a later hour would move. This is a
    direct behavioural test of the availability claim rather than an
    inspection of the formulas.
    """
    from src.data.dynamic_features import build_dynamic_features

    ordered_hours = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort")[TIMESTAMP_COLUMN]
    cutoff = pd.Timestamp(ordered_hours.iloc[len(ordered_hours) // 2])

    truncated_raw = raw[raw[TIMESTAMP_COLUMN] <= cutoff]
    truncated_hourly = hourly[hourly[TIMESTAMP_COLUMN] <= cutoff]

    rebuilt = build_dynamic_features(truncated_raw, truncated_hourly)
    reference = dynamic[dynamic[TIMESTAMP_COLUMN] <= cutoff].reset_index(drop=True)

    if len(rebuilt) != len(reference):
        raise ValueError("Truncating the future changed how many hours carry dynamic features.")
    if not rebuilt[TIMESTAMP_COLUMN].equals(reference[TIMESTAMP_COLUMN]):
        raise ValueError("Truncating the future changed the dynamic feature chronology.")

    left = rebuilt[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float)
    right = reference[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float)
    if not np.array_equal(left, right, equal_nan=True):
        raise ValueError(
            "A dynamic feature changed when later hours were removed, so it reads the future."
        )


# ---------------------------------------------------------------------
# Fold training and scoring
# ---------------------------------------------------------------------


def evaluate_fold(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_id: int,
    representation: str,
    predictors: list[str] | None = None,
) -> RepresentationFoldResult:
    """Fit the unchanged Random Forest on one fold and score its validation rows.

    The pipeline and every hyperparameter come from
    `src.models.random_forest.build_pipeline` with its defaults, so the
    model is identical to the committed benchmark in both arms.
    """
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this experiment.")

    predictors = get_predictors(representation) if predictors is None else list(predictors)
    validate_representation_scope(representation, predictors)

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

    return RepresentationFoldResult(
        representation=representation,
        fold_id=fold_id,
        metrics=metrics,
        n_train=len(frames.train),
        n_validation=len(frames.validation),
        n_scored=len(predictions),
        n_features=len(predictors),
        predictions=predictions,
    )


def evaluate_representation(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    representation: str,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> RepresentationEvaluation:
    """Evaluate one feature representation across every development fold."""
    predictors = get_predictors(representation)
    fold_results = [
        evaluate_fold(spark, hourly, assignment, fold_id, representation, predictors=predictors)
        for fold_id in fold_ids
    ]

    results = pd.DataFrame(
        [
            {
                "representation": result.representation,
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
    return RepresentationEvaluation(
        representation=representation,
        predictors=predictors,
        results=results,
        fold_results=fold_results,
    )


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    evaluation: RepresentationEvaluation,
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

    validate_representation_scope(evaluation.representation, evaluation.predictors)
    expected_features = len(evaluation.predictors)

    eligible = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    final_test = set(committed_assignment.loc[committed_assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])

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

        # The restriction may only remove rows. Every surviving row must
        # hold the same role in the same fold as the committed split.
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


def assert_matched_rows(
    static: RepresentationEvaluation, dynamic: RepresentationEvaluation
) -> None:
    """Confirm both arms were fitted and scored on identical rows."""
    static_by_fold = {result.fold_id: result for result in static.fold_results}
    dynamic_by_fold = {result.fold_id: result for result in dynamic.fold_results}

    if set(static_by_fold) != set(dynamic_by_fold):
        raise ValueError("The two representations do not cover the same folds.")

    for fold_id in sorted(static_by_fold):
        left, right = static_by_fold[fold_id], dynamic_by_fold[fold_id]
        if left.n_train != right.n_train:
            raise ValueError(
                f"Fold {fold_id}: training rows differ between representations "
                f"({left.n_train} against {right.n_train})."
            )
        if set(left.predictions[TIMESTAMP_COLUMN]) != set(right.predictions[TIMESTAMP_COLUMN]):
            raise ValueError(f"Fold {fold_id}: the two representations scored different hours.")
        if not np.allclose(
            left.predictions[TARGET_COLUMN].to_numpy(),
            right.predictions[TARGET_COLUMN].to_numpy(),
        ):
            raise ValueError(f"Fold {fold_id}: the two representations saw different targets.")


def verify_reproduces_committed_benchmark(
    results: pd.DataFrame,
    random_forest_results: pd.DataFrame,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Confirm the unrestricted static arm reproduces the committed benchmark.

    The static arm on the full committed assignment changes nothing about
    the model or the data, so it must reproduce the existing Random
    Forest figures. If it does not, this harness has altered something it
    was supposed to hold constant and no comparison built on it can be
    trusted.
    """
    merged = results.merge(
        random_forest_results[["fold_id", "rmse", "mae", "r2", "n_train", "n_validation"]],
        on="fold_id",
        suffixes=("", "_committed"),
    )
    if len(merged) != len(results):
        raise ValueError("The static arm does not cover the same folds as the committed benchmark.")

    for _, row in merged.iterrows():
        for column in ("n_train", "n_validation"):
            if int(row[column]) != int(row[f"{column}_committed"]):
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: static {column} is {int(row[column])}, but the "
                    f"committed benchmark used {int(row[f'{column}_committed'])}."
                )
        for metric in ("rmse", "mae", "r2"):
            if abs(float(row[metric]) - float(row[f"{metric}_committed"])) > tolerance:
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: static {metric} is {row[metric]:.10f}, but the "
                    f"committed Random Forest benchmark recorded "
                    f"{row[f'{metric}_committed']:.10f}."
                )


def verify_deterministic_features(raw: pd.DataFrame, hourly: pd.DataFrame) -> None:
    """Confirm the dynamic feature table is identical when rebuilt."""
    from src.data.dynamic_features import build_dynamic_features

    first = build_dynamic_features(raw, hourly)
    second = build_dynamic_features(raw, hourly)
    if not first.equals(second):
        raise ValueError("Dynamic feature construction is not deterministic.")


def verify_deterministic_evaluation(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    representation: str,
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Fit one fold twice and confirm the metrics agree within tolerance."""
    first = evaluate_fold(spark, hourly, assignment, fold_id, representation)
    second = evaluate_fold(spark, hourly, assignment, fold_id, representation)
    for metric in ("rmse", "mae", "r2"):
        if abs(first.metrics[metric] - second.metrics[metric]) > tolerance:
            raise ValueError(
                f"{representation} fold {fold_id}: {metric} is not reproducible "
                f"({first.metrics[metric]} against {second.metrics[metric]})."
            )


# ---------------------------------------------------------------------
# Comparison and analysis
# ---------------------------------------------------------------------


def combine_results(
    static: RepresentationEvaluation, dynamic: RepresentationEvaluation
) -> pd.DataFrame:
    """Fold level results for both arms in one table."""
    combined = pd.concat([static.results, dynamic.results], ignore_index=True)
    return combined.sort_values(
        ["representation", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


def compare_representations(combined: pd.DataFrame) -> pd.DataFrame:
    """Per fold difference between the dynamic and static arms.

    A negative RMSE or MAE difference and a positive R squared difference
    mean the dynamic representation predicts better.
    """
    static = combined[combined["representation"] == STATIC_REPRESENTATION]
    dynamic = combined[combined["representation"] == DYNAMIC_REPRESENTATION]

    merged = dynamic.merge(
        static[["fold_id", "rmse", "mae", "r2"]], on="fold_id", suffixes=("", "_static")
    )
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_static"]
    merged["mae_difference"] = merged["mae"] - merged["mae_static"]
    merged["r2_difference"] = merged["r2"] - merged["r2_static"]
    merged["dynamic_better"] = merged["rmse_difference"] < 0

    columns = [
        "fold_id",
        "n_train",
        "n_validation",
        "rmse",
        "rmse_static",
        "rmse_difference",
        "mae_difference",
        "r2_difference",
        "dynamic_better",
    ]
    return merged[columns].sort_values("fold_id", kind="mergesort").reset_index(drop=True)


def summarize_representations(combined: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds."""
    summary = (
        combined.groupby("representation")
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
    return summary.sort_values("representation", kind="mergesort").reset_index(drop=True)


def classify_support(
    comparison: pd.DataFrame, meaningful_rmse: float = MEANINGFUL_RMSE
) -> dict:
    """Apply the experiment's decision rule to the matched comparison.

    Strong support requires an improvement on every development fold and
    a mean improvement of at least `meaningful_rmse`. No support means
    the aggregate gain does not reach that margin and at most one fold
    improves at all. Anything between the two is mixed, which is a
    finding about which folds or regimes carry the gain rather than a
    verdict on temporal modeling.
    """
    n_folds = len(comparison)
    if n_folds == 0:
        raise ValueError("Cannot classify support from an empty comparison.")

    n_improved = int((comparison["rmse_difference"] < 0).sum())
    mean_difference = float(comparison["rmse_difference"].mean())
    worst = float(comparison["rmse_difference"].max())
    best = float(comparison["rmse_difference"].min())

    improves_everywhere = n_improved == n_folds
    if improves_everywhere and mean_difference <= -meaningful_rmse:
        classification = SUPPORT_STRONG
    elif mean_difference > -meaningful_rmse and n_improved <= 1:
        classification = SUPPORT_NONE
    else:
        classification = SUPPORT_WEAK

    return {
        "n_folds": n_folds,
        "n_folds_improved": n_improved,
        "improves_on_every_fold": improves_everywhere,
        "mean_rmse_difference": mean_difference,
        "worst_fold_rmse_difference": worst,
        "best_fold_rmse_difference": best,
        "mean_r2_difference": float(comparison["r2_difference"].mean()),
        "classification": classification,
    }


def excursion_analysis(
    static: RepresentationEvaluation,
    dynamic: RepresentationEvaluation,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    quantile: float = EXCURSION_QUANTILE,
) -> pd.DataFrame:
    """Split each fold's error by whether the hour is a high silica excursion.

    The threshold is the fold's own training target quantile, so the
    grouping uses only information a model would have had. Both arms are
    measured on the same hours in each group, so the difference between
    them is comparable within a group even though the two groups are not
    comparable with each other.
    """
    rows = []
    static_by_fold = {result.fold_id: result for result in static.fold_results}
    dynamic_by_fold = {result.fold_id: result for result in dynamic.fold_results}

    for fold_id in sorted(static_by_fold):
        frames = get_fold_frames(hourly, assignment, fold_id)
        threshold = float(frames.train[TARGET_COLUMN].quantile(quantile))

        for representation, result in (
            (STATIC_REPRESENTATION, static_by_fold[fold_id]),
            (DYNAMIC_REPRESENTATION, dynamic_by_fold[fold_id]),
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
                        "representation": representation,
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
        ["fold_id", "group", "representation"], kind="mergesort"
    ).reset_index(drop=True)


def compare_excursions(excursions: pd.DataFrame) -> pd.DataFrame:
    """Dynamic against static error within each fold and error group."""
    static = excursions[excursions["representation"] == STATIC_REPRESENTATION]
    dynamic = excursions[excursions["representation"] == DYNAMIC_REPRESENTATION]
    merged = dynamic.merge(
        static[["fold_id", "group", "rmse", "mae"]],
        on=["fold_id", "group"],
        suffixes=("", "_static"),
    )
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_static"]
    merged["mae_difference"] = merged["mae"] - merged["mae_static"]
    columns = [
        "fold_id",
        "group",
        "n",
        "threshold",
        "observed_mean",
        "rmse_static",
        "rmse",
        "rmse_difference",
        "mae_difference",
    ]
    return merged[columns].sort_values(
        ["group", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


def prediction_spread(evaluation: RepresentationEvaluation) -> pd.DataFrame:
    """Describe how wide each arm's predictions actually are.

    A model that emits a nearly constant number can post a respectable
    RMSE while carrying no operational information, so the width of the
    prediction distribution is reported next to its accuracy, together
    with the rank agreement between prediction and observation.
    """
    rows = []
    for result in evaluation.fold_results:
        predictions = result.predictions
        predicted = predictions[PREDICTION_COLUMN].to_numpy(dtype=float)
        observed = predictions[TARGET_COLUMN].to_numpy(dtype=float)
        observed_std = float(observed.std(ddof=0))
        rows.append(
            {
                "representation": evaluation.representation,
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
                "observed_vs_predicted_spearman": float(
                    predictions[TARGET_COLUMN].corr(
                        predictions[PREDICTION_COLUMN], method="spearman"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def subperiod_comparison(
    static: RepresentationEvaluation, dynamic: RepresentationEvaluation
) -> pd.DataFrame:
    """Error by chronological block, to see whether a gain is localized.

    Blocks are cut by position within each validation window, reusing the
    generalization diagnostics implementation so the block definition is
    identical to the one already reported for this benchmark.
    """
    frames = []
    for evaluation in (static, dynamic):
        parts = []
        for result in evaluation.fold_results:
            frame = result.predictions.copy()
            frame["fold_id"] = result.fold_id
            frame["residual"] = frame[TARGET_COLUMN] - frame[PREDICTION_COLUMN]
            parts.append(frame)
        residuals = pd.concat(parts, ignore_index=True)
        blocks = residual_subperiods(residuals)
        blocks.insert(0, "representation", evaluation.representation)
        frames.append(blocks)

    combined = pd.concat(frames, ignore_index=True)
    static_blocks = combined[combined["representation"] == STATIC_REPRESENTATION]
    dynamic_blocks = combined[combined["representation"] == DYNAMIC_REPRESENTATION]

    merged = dynamic_blocks.merge(
        static_blocks[["fold_id", "subperiod", "rmse"]],
        on=["fold_id", "subperiod"],
        suffixes=("", "_static"),
    )
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_static"]
    columns = [
        "fold_id",
        "subperiod",
        "n",
        "start",
        "end",
        "observed_mean",
        "rmse_static",
        "rmse",
        "rmse_difference",
    ]
    return merged[columns].sort_values(
        ["fold_id", "subperiod"], kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Paths, report, CLI
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_dynamic_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "dynamic_features.parquet"


def default_random_forest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "dynamic_representation_results.parquet"


def format_report(report: dict) -> str:
    lines = ["Static against dynamic sensor representation (development folds only)", ""]

    lines.append("Original static benchmark on the full committed assignment")
    lines.append(
        f"  {'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'feats':>5}  {'rmse':>7}  "
        f"{'mae':>7}  {'r2':>8}"
    )
    for _, row in report["unrestricted_static"].iterrows():
        lines.append(
            f"  {int(row['fold_id']):>4}  {int(row['n_train']):>7,}  "
            f"{int(row['n_validation']):>5,}  {int(row['n_features']):>5}  {row['rmse']:>7.4f}  "
            f"{row['mae']:>7.4f}  {row['r2']:>8.4f}"
        )

    lines.extend(["", "Rows removed because the 120 minute window is incomplete", ""])
    dropped = report["row_loss"][report["row_loss"]["n_dropped"] > 0]
    if dropped.empty:
        lines.append("  none")
    else:
        for _, row in dropped.iterrows():
            lines.append(
                f"  fold {int(row['fold_id'])} {row['role']}: {int(row['n_dropped'])} of "
                f"{int(row['n_committed'])}"
            )

    lines.extend(["", "Matched comparison on identical rows", ""])
    lines.append(
        f"  {'representation':<14}  {'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'feats':>5}  "
        f"{'rmse':>7}  {'mae':>7}  {'r2':>8}"
    )
    for _, row in report["combined"].iterrows():
        lines.append(
            f"  {row['representation']:<14}  {int(row['fold_id']):>4}  "
            f"{int(row['n_train']):>7,}  {int(row['n_validation']):>5,}  "
            f"{int(row['n_features']):>5}  {row['rmse']:>7.4f}  {row['mae']:>7.4f}  "
            f"{row['r2']:>8.4f}"
        )

    lines.extend(["", "Aggregate across folds", ""])
    for _, row in report["summary"].iterrows():
        lines.append(
            f"  {row['representation']:<8} ({int(row['n_features'])} features)  "
            f"RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(
        ["", "Dynamic against static (negative RMSE difference means dynamic predicts better)", ""]
    )
    for _, row in report["comparison"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: {row['rmse']:.4f} against {row['rmse_static']:.4f}  "
            f"RMSE {row['rmse_difference']:+.4f}  MAE {row['mae_difference']:+.4f}  "
            f"R2 {row['r2_difference']:+.4f}"
        )
    support = report["support"]
    lines.append(
        f"  improved on {support['n_folds_improved']}/{support['n_folds']} folds, "
        f"mean RMSE difference {support['mean_rmse_difference']:+.4f}, "
        f"worst fold {support['worst_fold_rmse_difference']:+.4f}, "
        f"best fold {support['best_fold_rmse_difference']:+.4f}"
    )

    lines.extend(["", "Error by chronological block within each validation window", ""])
    for _, row in report["subperiods"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} block {int(row['subperiod'])} "
            f"({int(row['n']):>3} h, {row['start'].date()} to {row['end'].date()}, "
            f"observed mean {row['observed_mean']:.3f}): static {row['rmse_static']:.4f} -> "
            f"dynamic {row['rmse']:.4f}  ({row['rmse_difference']:+.4f})"
        )

    lines.extend(["", "High silica excursion hours against the rest of the window", ""])
    for _, row in report["excursion_comparison"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['group']:<9} (n {int(row['n']):>4}, "
            f"threshold {row['threshold']:.3f}, observed mean {row['observed_mean']:.3f}): "
            f"static {row['rmse_static']:.4f} -> dynamic {row['rmse']:.4f}  "
            f"({row['rmse_difference']:+.4f})"
        )

    lines.extend(["", "Prediction distribution", ""])
    for _, row in report["spread"].iterrows():
        lines.append(
            f"  {row['representation']:<8} fold {int(row['fold_id'])}: sd {row['prediction_std']:.4f} "
            f"(observed sd {row['observed_std']:.4f}, ratio {row['spread_ratio']:.3f}), "
            f"IQR {row['prediction_iqr']:.4f}, range {row['prediction_range']:.4f} "
            f"(observed range {row['observed_range']:.4f}), "
            f"rank agreement {row['observed_vs_predicted_spearman']:+.3f}"
        )

    lines.extend(["", f"Classification: {support['classification']}"])
    return "\n".join(lines)


def run(
    hourly_path: Path,
    splits_path: Path,
    dynamic_path: Path,
    random_forest_path: Path,
    results_path: Path,
    spark=None,
) -> dict:
    """Run both arms, validate every guard, compare, and write the results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityDynamicRepresentation")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)

        for path, description in (
            (dynamic_path, "dynamic feature"),
            (random_forest_path, "Random Forest benchmark"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {description} dataset not found at: {path}. Run that module first."
                )
        dynamic_features = pd.read_parquet(dynamic_path)
        committed_random_forest = pd.read_parquet(random_forest_path)

        matched = build_matched_dataset(hourly, dynamic_features, assignment)

        # The static arm on the untouched assignment must reproduce the
        # committed benchmark before anything else is read from it.
        unrestricted_static = evaluate_representation(
            spark, matched.hourly, assignment, STATIC_REPRESENTATION
        )
        validate_evaluation(unrestricted_static, matched.hourly, assignment, assignment)
        verify_reproduces_committed_benchmark(
            unrestricted_static.results, committed_random_forest
        )

        static = evaluate_representation(
            spark, matched.hourly, matched.assignment, STATIC_REPRESENTATION
        )
        dynamic = evaluate_representation(
            spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION
        )

        for evaluation in (static, dynamic):
            validate_evaluation(evaluation, matched.hourly, matched.assignment, assignment)
        assert_matched_rows(static, dynamic)
        verify_deterministic_evaluation(
            spark, matched.hourly, matched.assignment, DYNAMIC_REPRESENTATION
        )

        combined = combine_results(static, dynamic)
        comparison = compare_representations(combined)
        excursions = excursion_analysis(
            static, dynamic, matched.hourly, matched.assignment
        )

        report = {
            "unrestricted_static": unrestricted_static.results,
            "row_loss": matched.row_loss,
            "combined": combined,
            "summary": summarize_representations(combined),
            "comparison": comparison,
            "support": classify_support(comparison),
            "excursions": excursions,
            "excursion_comparison": compare_excursions(excursions),
            "spread": pd.concat(
                [prediction_spread(static), prediction_spread(dynamic)], ignore_index=True
            ),
            "subperiods": subperiod_comparison(static, dynamic),
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

    print("Starting Spark session and comparing sensor representations...")
    report = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_dynamic_path(repo_root),
        default_random_forest_path(repo_root),
        results_path,
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(report))


if __name__ == "__main__":
    sys.exit(main())
