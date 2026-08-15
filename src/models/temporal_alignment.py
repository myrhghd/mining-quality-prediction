"""Sensor to target temporal alignment experiment.

Random Forest was the strongest conventional regression benchmark, but
its forward R squared remained negative in two of three development
folds. One possible explanation was target alignment: the hourly sensor
aggregates and hourly assay are joined at the same hour, but flotation is
a residence time process, so the assay reported for hour `t` may reflect
pulp that passed the sensors earlier.

This module tests that directly. Sensor features stay at hour `t`. The
target is taken from hour `t`, `t + 1`, or `t + 2`. Everything else is
held constant: the same 57 core sensor predictors, the same
preprocessing, the same committed development folds, the same Random
Forest benchmark configuration imported unchanged from
`src.models.random_forest`, the same metrics, and the same leakage
guards. Alignment is the only variable.

Direction of the shift
----------------------
The target moves forward and the features do not move at all. A row
predicts an assay that is reported after its own sensor hour, so no
future sensor information can enter a predictor. The reverse shift,
which would let a row read sensors from after the assay it predicts, is
never constructed.

Role containment
----------------
A row is retained only when the hour supplying its target carries the
same role in the same fold as the hour supplying its features. A
training row therefore cannot read a target out of the embargo or the
validation window, and a validation row cannot read a target from beyond
its own window. This is stricter than dropping only the final rows of
the series, and it is what keeps the embargo meaningful once the target
moves.

Rows whose target hour is unavailable under that rule are removed rather
than imputed, so each alignment is evaluated on slightly fewer hours than
the one before it. Fold level counts and removed row counts are reported
alongside every metric. The runtime report also computes a matched
comparison over hours scored by all three alignments, because RMSE over
different row sets is not a like for like comparison.

Headline mean RMSE decreased at the 1 and 2 hour shifts, but the shifted
arms scored different row populations and mean R squared worsened. The
stored `temporal_alignment_results.parquet` artifact does not contain the
common hour table required for a direct like for like assessment. The
recorded experiment decision therefore retained the 0 hour alignment.

The final test period is never fitted on, never scored, and never used to
select an alignment.
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
    KIND_DEVELOPMENT,
    ROLE_EMBARGO,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models.baselines import DEVELOPMENT_FOLD_IDS, compute_metrics, load_inputs
from src.models.linear_regression import (
    NUMERICAL_TOLERANCE,
    build_spark_session,
    get_sensor_predictors,
    validate_predictor_scope,
)
from src.models.random_forest import (
    MODEL_NAME,
    PREDICTION_COLUMN,
    evaluate_model,
)
from src.models.random_forest import validate_evaluation as validate_random_forest_evaluation

# The three alignments under test, in hours between the sensor hour and
# the hour whose assay is predicted.
ALIGNMENT_HOURS = (0, 1, 2)

# The alignment the project currently uses, and the reference every other
# alignment is compared against.
BASELINE_ALIGNMENT_HOURS = 0

# Column recording which hour supplied each row's target. Kept in the
# aligned frame for auditing; it is never a predictor.
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

ALIGNMENT_RESULT_COLUMNS = [
    "alignment_hours",
    "fold_id",
    "model",
    "n_train",
    "n_validation",
    "n_scored",
    "n_features",
    "rmse",
    "mae",
    "r2",
    "n_train_dropped",
    "n_validation_dropped",
]


@dataclass(frozen=True)
class AlignedDataset:
    """One alignment's view of the hourly table and the split assignment."""

    alignment_hours: int
    hourly: pd.DataFrame
    assignment: pd.DataFrame
    row_loss: pd.DataFrame


@dataclass(frozen=True)
class AlignmentEvaluation:
    alignment_hours: int
    results: pd.DataFrame
    fold_results: list
    dataset: AlignedDataset


# ---------------------------------------------------------------------
# Target shifting
# ---------------------------------------------------------------------


def shift_target(hourly: pd.DataFrame, alignment_hours: int) -> pd.DataFrame:
    """Return the hourly table with the target taken from `alignment_hours` later.

    Predictor columns are untouched: every retained row keeps the sensor
    aggregates measured during its own hour. Only `TARGET_COLUMN` is
    replaced, with the value observed at `timestamp + alignment_hours`,
    and `TARGET_TIMESTAMP_COLUMN` records where that value came from.

    A row is retained only when the target hour exists in the hourly
    chronology and lies in the same temporal segment, so a target is
    never carried across a data discontinuity. Rows whose target hour is
    missing are removed, never imputed.
    """
    if alignment_hours < 0:
        raise ValueError(
            f"Alignment must be zero or a forward shift, got {alignment_hours} hours. "
            "A negative shift would let a row read sensors recorded after the assay "
            "it predicts."
        )

    ordered = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    if ordered[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError("The hourly table contains duplicate timestamps.")

    lag = pd.Timedelta(alignment_hours, unit="h")
    target_timestamp = ordered[TIMESTAMP_COLUMN] + lag

    targets = ordered.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]
    segments = ordered.set_index(TIMESTAMP_COLUMN)[SEGMENT_COLUMN]

    shifted_target = target_timestamp.map(targets)
    shifted_segment = target_timestamp.map(segments)

    available = shifted_target.notna() & (shifted_segment == ordered[SEGMENT_COLUMN])

    aligned = ordered.loc[available].copy()
    aligned[TARGET_COLUMN] = shifted_target.loc[available].to_numpy()
    aligned[TARGET_TIMESTAMP_COLUMN] = target_timestamp.loc[available].to_numpy()
    return aligned.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)


def restrict_assignment(
    assignment: pd.DataFrame, retained_timestamps: set, alignment_hours: int
) -> pd.DataFrame:
    """Restrict the committed assignment to rows whose target stays in role.

    A development row survives only when the hour `alignment_hours` later
    is assigned the same role within the same fold. That single rule
    keeps a training row from reading a target out of the embargo or the
    validation window, and keeps a validation row from reading a target
    from beyond its own window.

    Final test rows are carried through unchanged. They are never fitted
    on or scored; they are retained so the leakage guards can still assert
    that the test period was left alone.
    """
    development = assignment[assignment["fold_kind"] == KIND_DEVELOPMENT].copy()
    other = assignment[assignment["fold_kind"] != KIND_DEVELOPMENT]

    lag = pd.Timedelta(alignment_hours, unit="h")
    development[TARGET_TIMESTAMP_COLUMN] = development[TIMESTAMP_COLUMN] + lag

    roles_at_target = development[["fold_id", TIMESTAMP_COLUMN, "role"]].rename(
        columns={TIMESTAMP_COLUMN: TARGET_TIMESTAMP_COLUMN, "role": "target_role"}
    )
    merged = development.merge(roles_at_target, on=["fold_id", TARGET_TIMESTAMP_COLUMN], how="left")

    keeps_role = merged["target_role"] == merged["role"]
    has_features = merged[TIMESTAMP_COLUMN].isin(retained_timestamps)
    kept = merged.loc[keeps_role & has_features, assignment.columns]

    # Empty frames are dropped before concatenation: pandas materializes a
    # placeholder value for every column of an empty block, which is both
    # wasted work and a deprecated conversion for datetime columns.
    parts = [frame for frame in (kept, other) if not frame.empty]
    restricted = pd.concat(parts, ignore_index=True) if parts else assignment.iloc[:0].copy()
    return restricted.sort_values(
        ["fold_id", "role", TIMESTAMP_COLUMN], kind="mergesort"
    ).reset_index(drop=True)


def summarize_row_loss(
    assignment: pd.DataFrame, restricted: pd.DataFrame, alignment_hours: int
) -> pd.DataFrame:
    """Rows removed per fold and role because the target hour is unavailable."""
    def counts(frame: pd.DataFrame) -> pd.Series:
        development = frame[frame["fold_kind"] == KIND_DEVELOPMENT]
        return development.groupby(["fold_id", "role"]).size()

    before = counts(assignment)
    after = counts(restricted).reindex(before.index, fill_value=0)

    table = pd.DataFrame(
        {
            "alignment_hours": alignment_hours,
            "n_committed": before,
            "n_retained": after,
            "n_dropped": before - after,
        }
    ).reset_index()
    return table.sort_values(["fold_id", "role"], kind="mergesort").reset_index(drop=True)


def build_aligned_dataset(
    hourly: pd.DataFrame, assignment: pd.DataFrame, alignment_hours: int
) -> AlignedDataset:
    """Build one alignment's hourly frame, assignment, and row loss table."""
    aligned_hourly = shift_target(hourly, alignment_hours)
    retained = set(aligned_hourly[TIMESTAMP_COLUMN])
    restricted = restrict_assignment(assignment, retained, alignment_hours)
    row_loss = summarize_row_loss(assignment, restricted, alignment_hours)

    return AlignedDataset(
        alignment_hours=alignment_hours,
        hourly=aligned_hourly,
        assignment=restricted,
        row_loss=row_loss,
    )


# ---------------------------------------------------------------------
# Leakage and integrity guards
# ---------------------------------------------------------------------


def validate_alignment(
    dataset: AlignedDataset, hourly: pd.DataFrame, assignment: pd.DataFrame
) -> None:
    """Verify the shift is correct, chronological, and leakage free."""
    alignment_hours = dataset.alignment_hours
    lag = pd.Timedelta(alignment_hours, unit="h")
    aligned = dataset.hourly

    if not aligned[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(f"{alignment_hours}h alignment: hourly rows are not in chronological order.")
    if aligned[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(f"{alignment_hours}h alignment: duplicate feature timestamps.")

    # The target must come from exactly `alignment_hours` after the sensor
    # hour, and never from before it.
    offsets = aligned[TARGET_TIMESTAMP_COLUMN] - aligned[TIMESTAMP_COLUMN]
    if not bool((offsets == lag).all()):
        raise ValueError(f"{alignment_hours}h alignment: a target hour is not exactly {lag} later.")
    if alignment_hours > 0 and not bool(
        (aligned[TARGET_TIMESTAMP_COLUMN] > aligned[TIMESTAMP_COLUMN]).all()
    ):
        raise ValueError(f"{alignment_hours}h alignment: a target hour is not strictly later.")

    original = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").set_index(TIMESTAMP_COLUMN)

    # The shifted target must equal the observed target at the target hour.
    expected_target = aligned[TARGET_TIMESTAMP_COLUMN].map(original[TARGET_COLUMN]).to_numpy()
    if not np.allclose(aligned[TARGET_COLUMN].to_numpy(), expected_target, equal_nan=False):
        raise ValueError(f"{alignment_hours}h alignment: shifted target values are incorrect.")

    # Predictors must be untouched. Shifting the label must never move a
    # feature, because that is what would import future sensor readings.
    predictors = get_sensor_predictors()
    validate_predictor_scope(predictors)
    unchanged = original.loc[aligned[TIMESTAMP_COLUMN], predictors].to_numpy(dtype=float)
    if not np.array_equal(aligned[predictors].to_numpy(dtype=float), unchanged):
        raise ValueError(
            f"{alignment_hours}h alignment: predictor values changed during the target shift."
        )

    # A target must never be carried across a temporal discontinuity.
    target_segments = aligned[TARGET_TIMESTAMP_COLUMN].map(original[SEGMENT_COLUMN]).to_numpy()
    if not bool((target_segments == aligned[SEGMENT_COLUMN].to_numpy()).all()):
        raise ValueError(
            f"{alignment_hours}h alignment: a target was taken across a temporal segment boundary."
        )

    # Every retained row must have been committed to the same role, and the
    # hour supplying its target must hold that same role in that same fold.
    committed = assignment[assignment["fold_kind"] == KIND_DEVELOPMENT]
    restricted = dataset.assignment[dataset.assignment["fold_kind"] == KIND_DEVELOPMENT]

    committed_pairs = set(zip(committed["fold_id"], committed[TIMESTAMP_COLUMN], committed["role"]))
    restricted_pairs = set(
        zip(restricted["fold_id"], restricted[TIMESTAMP_COLUMN], restricted["role"])
    )
    if not restricted_pairs.issubset(committed_pairs):
        raise ValueError(
            f"{alignment_hours}h alignment: the restricted assignment invented rows that were "
            "not committed."
        )

    role_by_key = {
        (fold_id, timestamp): role
        for fold_id, timestamp, role in committed_pairs
    }
    for fold_id, timestamp, role in restricted_pairs:
        target_role = role_by_key.get((fold_id, timestamp + lag))
        if target_role != role:
            raise ValueError(
                f"{alignment_hours}h alignment fold {fold_id}: the hour supplying the target for "
                f"{timestamp} is assigned {target_role!r}, not {role!r}."
            )

    # The final test period must survive the restriction untouched, so the
    # downstream guards still have something to check against.
    original_test = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    restricted_test = set(
        dataset.assignment.loc[dataset.assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    )
    if restricted_test != original_test:
        raise ValueError(f"{alignment_hours}h alignment: the final test assignment was altered.")

    # No development row may draw its target from an embargo hour or from
    # the final test period.
    target_hours = {timestamp + lag for _, timestamp, _ in restricted_pairs}
    if target_hours.intersection(original_test):
        raise ValueError(
            f"{alignment_hours}h alignment: a development row draws its target from the "
            "final test period."
        )

    embargo_rows = committed[committed["role"] == ROLE_EMBARGO]
    embargo_by_fold = {
        fold_id: set(group[TIMESTAMP_COLUMN])
        for fold_id, group in embargo_rows.groupby("fold_id")
    }
    for fold_id, timestamp, role in restricted_pairs:
        if role not in (ROLE_TRAIN, ROLE_VALIDATION):
            continue
        if (timestamp + lag) in embargo_by_fold.get(fold_id, ()):
            raise ValueError(
                f"{alignment_hours}h alignment fold {fold_id}: the row at {timestamp} draws "
                "its target from an embargo hour."
            )

    # Retained hours must remain eligible under the unchanged flag.
    eligible = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    assigned = {timestamp for _, timestamp, _ in restricted_pairs}
    if not assigned.issubset(eligible):
        raise ValueError(f"{alignment_hours}h alignment: an ineligible hour survived restriction.")

    # A shift can only remove rows, never add them.
    loss = dataset.row_loss
    if bool((loss["n_dropped"] < 0).any()):
        raise ValueError(f"{alignment_hours}h alignment: row counts grew after shifting.")
    if alignment_hours == 0 and bool((loss["n_dropped"] != 0).any()):
        raise ValueError("The 0 hour alignment must not remove any committed row.")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_alignment(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    alignment_hours: int,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> AlignmentEvaluation:
    """Evaluate the unchanged Random Forest benchmark at one alignment.

    The model, its hyperparameters, its pipeline, and its own validation
    guards are imported from `src.models.random_forest` and used as they
    are. Only the data handed to them differs.
    """
    dataset = build_aligned_dataset(hourly, assignment, alignment_hours)
    validate_alignment(dataset, hourly, assignment)

    results, fold_results = evaluate_model(
        spark, dataset.hourly, dataset.assignment, fold_ids=fold_ids
    )
    validate_random_forest_evaluation(
        results, fold_results, dataset.hourly, dataset.assignment, get_sensor_predictors()
    )

    dropped = dataset.row_loss.set_index(["fold_id", "role"])["n_dropped"]
    results = results.copy()
    results.insert(0, "alignment_hours", alignment_hours)
    results["n_train_dropped"] = [
        int(dropped.get((fold_id, ROLE_TRAIN), 0)) for fold_id in results["fold_id"]
    ]
    results["n_validation_dropped"] = [
        int(dropped.get((fold_id, ROLE_VALIDATION), 0)) for fold_id in results["fold_id"]
    ]
    results = results[ALIGNMENT_RESULT_COLUMNS]

    return AlignmentEvaluation(
        alignment_hours=alignment_hours,
        results=results,
        fold_results=fold_results,
        dataset=dataset,
    )


def run_experiment(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    alignments: tuple[int, ...] = ALIGNMENT_HOURS,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> tuple[pd.DataFrame, dict[int, AlignmentEvaluation]]:
    """Evaluate every alignment and return the combined fold level results."""
    evaluations = {
        alignment_hours: evaluate_alignment(
            spark, hourly, assignment, alignment_hours, fold_ids=fold_ids
        )
        for alignment_hours in alignments
    }
    results = pd.concat(
        [evaluation.results for evaluation in evaluations.values()], ignore_index=True
    )
    results = results.sort_values(
        ["alignment_hours", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    return results, evaluations


# ---------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------


def summarize_alignments(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds."""
    summary = (
        results.groupby("alignment_hours")
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
    return summary.sort_values("alignment_hours", kind="mergesort").reset_index(drop=True)


def compare_with_baseline_alignment(
    results: pd.DataFrame, baseline_alignment: int = BASELINE_ALIGNMENT_HOURS
) -> pd.DataFrame:
    """Per fold RMSE and MAE difference against the current 0 hour alignment.

    A negative difference means the shifted alignment predicts better.
    """
    if baseline_alignment not in set(results["alignment_hours"]):
        raise ValueError(f"No results for the {baseline_alignment} hour reference alignment.")

    reference = results[results["alignment_hours"] == baseline_alignment][
        ["fold_id", "rmse", "mae", "r2"]
    ]
    shifted = results[results["alignment_hours"] != baseline_alignment]

    merged = shifted.merge(reference, on="fold_id", suffixes=("", "_reference"))
    merged["rmse_difference"] = merged["rmse"] - merged["rmse_reference"]
    merged["mae_difference"] = merged["mae"] - merged["mae_reference"]
    merged["r2_difference"] = merged["r2"] - merged["r2_reference"]
    merged["alignment_better"] = merged["rmse_difference"] < 0

    columns = [
        "alignment_hours",
        "fold_id",
        "rmse",
        "rmse_reference",
        "rmse_difference",
        "mae_difference",
        "r2_difference",
        "alignment_better",
    ]
    return merged[columns].sort_values(
        ["alignment_hours", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


def matched_comparison(
    evaluations: dict[int, AlignmentEvaluation],
    baseline_alignment: int = BASELINE_ALIGNMENT_HOURS,
) -> pd.DataFrame:
    """Recompute metrics on the hours every alignment scores in common.

    Each alignment loses a different set of rows, so its headline RMSE is
    measured over a slightly different validation population. Restricting
    every alignment to the intersection of scored sensor hours removes
    that difference, leaving the alignment itself as the only thing that
    varies. The observed target still differs between alignments, because
    that is the experiment.
    """
    rows = []
    fold_ids = sorted(
        {
            int(result.fold_id)
            for evaluation in evaluations.values()
            for result in evaluation.fold_results
        }
    )

    for fold_id in fold_ids:
        per_alignment = {}
        for alignment_hours, evaluation in evaluations.items():
            for result in evaluation.fold_results:
                if result.fold_id == fold_id:
                    per_alignment[alignment_hours] = result.predictions
        if len(per_alignment) != len(evaluations):
            continue

        common = set.intersection(
            *(set(frame[TIMESTAMP_COLUMN]) for frame in per_alignment.values())
        )
        if not common:
            continue

        reference_metrics = None
        for alignment_hours in sorted(per_alignment):
            frame = per_alignment[alignment_hours]
            subset = frame[frame[TIMESTAMP_COLUMN].isin(common)]
            metrics = compute_metrics(
                subset[TARGET_COLUMN].to_numpy(), subset[PREDICTION_COLUMN].to_numpy()
            )
            if alignment_hours == baseline_alignment:
                reference_metrics = metrics
            rows.append(
                {
                    "fold_id": fold_id,
                    "alignment_hours": alignment_hours,
                    "n_common": len(common),
                    **metrics,
                }
            )

        if reference_metrics is not None:
            for row in rows:
                if row["fold_id"] == fold_id:
                    row["rmse_difference"] = row["rmse"] - reference_metrics["rmse"]

    return pd.DataFrame(rows).sort_values(
        ["alignment_hours", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


def assess_consistency(
    comparison: pd.DataFrame, meaningful_rmse: float = 0.01
) -> pd.DataFrame:
    """Decide whether a shifted alignment improves consistently, not once.

    An alignment counts as consistently better only when it lowers RMSE on
    every development fold. `meaningful_rmse` is the margin below which a
    mean improvement is treated as too small to act on; it is a reporting
    threshold applied after the fact, not a tuning parameter, and the fold
    level numbers are reported regardless so the reader can apply their
    own.
    """
    rows = []
    for alignment_hours, subset in comparison.groupby("alignment_hours"):
        improved = int((subset["rmse_difference"] < 0).sum())
        n_folds = len(subset)
        mean_difference = float(subset["rmse_difference"].mean())
        rows.append(
            {
                "alignment_hours": int(alignment_hours),
                "n_folds": n_folds,
                "n_folds_improved": improved,
                "improves_on_every_fold": improved == n_folds,
                "mean_rmse_difference": mean_difference,
                "worst_fold_rmse_difference": float(subset["rmse_difference"].max()),
                "best_fold_rmse_difference": float(subset["rmse_difference"].min()),
                "consistent_and_meaningful": bool(
                    improved == n_folds and mean_difference <= -meaningful_rmse
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("alignment_hours", kind="mergesort").reset_index(
        drop=True
    )


def matched_consistency(
    matched: pd.DataFrame,
    baseline_alignment: int = BASELINE_ALIGNMENT_HOURS,
    meaningful_rmse: float = 0.01,
) -> pd.DataFrame:
    """Apply the consistency rule to the matched row comparison.

    The headline comparison scores each alignment on its own surviving
    hours, so a shift that happens to drop the hardest hours can look
    better without predicting anything better. This applies the same rule
    to the hours every alignment scores in common, where that particular
    advantage is unavailable.
    """
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "alignment_hours",
                "n_folds",
                "n_folds_improved",
                "improves_on_every_fold",
                "mean_rmse_difference",
                "worst_fold_rmse_difference",
                "best_fold_rmse_difference",
                "consistent_and_meaningful",
            ]
        )
    shifted = matched[matched["alignment_hours"] != baseline_alignment]
    return assess_consistency(shifted, meaningful_rmse=meaningful_rmse)


def best_supported_alignment(
    consistency: pd.DataFrame, matched: pd.DataFrame | None = None
) -> int:
    """The alignment the evidence supports, defaulting to the current one.

    A shifted alignment is selected only when it improves on every
    development fold by a meaningful margin, and, when a matched
    comparison is supplied, only when it still does so on the hours every
    alignment scores in common. An improvement that survives one test but
    not the other is evidence about which rows each alignment kept, not
    about alignment, so it is not acted on.

    Where nothing clears that bar the current 0 hour alignment stands,
    because the experiment then provides no reason to change it.
    """
    qualifying = consistency[consistency["consistent_and_meaningful"]]
    if matched is not None and not matched.empty:
        supported = set(
            matched.loc[matched["consistent_and_meaningful"], "alignment_hours"].astype(int)
        )
        qualifying = qualifying[qualifying["alignment_hours"].astype(int).isin(supported)]
    if qualifying.empty:
        return BASELINE_ALIGNMENT_HOURS
    best = qualifying.sort_values("mean_rmse_difference", kind="mergesort").iloc[0]
    return int(best["alignment_hours"])


# ---------------------------------------------------------------------
# Reproducibility guard
# ---------------------------------------------------------------------


def verify_reproducible(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    alignment_hours: int,
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Evaluate one alignment twice and confirm the metrics agree."""
    first = evaluate_alignment(spark, hourly, assignment, alignment_hours, fold_ids=(fold_id,))
    second = evaluate_alignment(spark, hourly, assignment, alignment_hours, fold_ids=(fold_id,))

    for metric in ("rmse", "mae", "r2"):
        left = float(first.results[metric].iloc[0])
        right = float(second.results[metric].iloc[0])
        if abs(left - right) > tolerance:
            raise ValueError(
                f"{alignment_hours}h alignment fold {fold_id}: {metric} is not reproducible "
                f"({left} vs {right})."
            )


def verify_reproduces_committed_benchmark(
    results: pd.DataFrame,
    random_forest_results: pd.DataFrame,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Confirm the 0 hour arm reproduces the committed Random Forest results.

    The 0 hour alignment removes no rows and changes no target, so it must
    reproduce the existing benchmark. If it does not, the harness has
    altered something it was supposed to hold constant, and no comparison
    built on it can be trusted. Row counts must match exactly; metrics are
    compared at the tolerance the project already uses for independent
    Spark fits, since the committed figures came from a separate run.
    """
    baseline = results[results["alignment_hours"] == BASELINE_ALIGNMENT_HOURS]
    merged = baseline.merge(
        random_forest_results[["fold_id", "rmse", "mae", "r2", "n_train", "n_validation"]],
        on="fold_id",
        suffixes=("", "_committed"),
    )
    if len(merged) != len(baseline):
        raise ValueError("The 0 hour arm does not cover the same folds as the committed benchmark.")

    for _, row in merged.iterrows():
        for column in ("n_train", "n_validation"):
            if int(row[column]) != int(row[f"{column}_committed"]):
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: 0 hour {column} is {int(row[column])}, "
                    f"but the committed benchmark used {int(row[f'{column}_committed'])}."
                )
        for metric in ("rmse", "mae", "r2"):
            if abs(float(row[metric]) - float(row[f"{metric}_committed"])) > tolerance:
                raise ValueError(
                    f"Fold {int(row['fold_id'])}: 0 hour {metric} is {row[metric]:.10f}, but the "
                    f"committed Random Forest benchmark recorded {row[f'{metric}_committed']:.10f}."
                )


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_random_forest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "temporal_alignment_results.parquet"


def format_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    matched: pd.DataFrame,
    consistency: pd.DataFrame,
    row_loss: pd.DataFrame,
) -> str:
    lines = ["Sensor to target alignment experiment (development folds only)", ""]
    lines.append(
        f"{'align':>5}  {'fold':>4}  {'n_train':>7}  {'n_val':>5}  {'dropped':>7}  "
        f"{'rmse':>7}  {'mae':>7}  {'r2':>8}"
    )
    for _, row in results.iterrows():
        dropped = int(row["n_train_dropped"]) + int(row["n_validation_dropped"])
        lines.append(
            f"{int(row['alignment_hours']):>4}h  {int(row['fold_id']):>4}  "
            f"{int(row['n_train']):>7,}  {int(row['n_validation']):>5,}  {dropped:>7,}  "
            f"{row['rmse']:>7.4f}  {row['mae']:>7.4f}  {row['r2']:>8.4f}"
        )

    lines.extend(["", "Aggregate across folds", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"  {int(row['alignment_hours'])}h alignment  "
            f"RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )

    lines.extend(
        ["", "Against the 0 hour alignment (negative means the shift predicts better)", ""]
    )
    for alignment_hours in sorted(comparison["alignment_hours"].unique()):
        subset = comparison[comparison["alignment_hours"] == alignment_hours]
        lines.append(f"  {int(alignment_hours)}h alignment")
        for _, row in subset.iterrows():
            lines.append(
                f"    fold {int(row['fold_id'])}: {row['rmse']:.4f} vs "
                f"{row['rmse_reference']:.4f}  RMSE {row['rmse_difference']:+.4f}  "
                f"MAE {row['mae_difference']:+.4f}  R2 {row['r2_difference']:+.4f}"
            )
        lines.append(f"    mean RMSE difference {subset['rmse_difference'].mean():+.4f}")

    if not matched.empty:
        lines.extend(
            ["", "Matched comparison on the hours every alignment scores in common", ""]
        )
        for fold_id in sorted(matched["fold_id"].unique()):
            subset = matched[matched["fold_id"] == fold_id]
            common = int(subset["n_common"].iloc[0])
            parts = ", ".join(
                f"{int(row['alignment_hours'])}h {row['rmse']:.4f} "
                f"({row.get('rmse_difference', float('nan')):+.4f})"
                for _, row in subset.iterrows()
            )
            lines.append(f"  fold {fold_id} ({common:,} common hours): {parts}")

    matched_rule = matched_consistency(matched)

    lines.extend(["", "Consistency of improvement", ""])
    for label, table in (
        ("on each alignment's own scored hours", consistency),
        ("on the common hours", matched_rule),
    ):
        lines.append(f"  {label}")
        for _, row in table.iterrows():
            lines.append(
                f"    {int(row['alignment_hours'])}h alignment: improved on "
                f"{int(row['n_folds_improved'])}/{int(row['n_folds'])} folds, "
                f"mean RMSE difference {row['mean_rmse_difference']:+.4f}, "
                f"worst fold {row['worst_fold_rmse_difference']:+.4f}, "
                f"consistent and meaningful: {bool(row['consistent_and_meaningful'])}"
            )

    lines.extend(["", "Rows removed because the target hour is unavailable", ""])
    for alignment_hours in sorted(row_loss["alignment_hours"].unique()):
        subset = row_loss[
            (row_loss["alignment_hours"] == alignment_hours) & (row_loss["n_dropped"] > 0)
        ]
        if subset.empty:
            lines.append(f"  {int(alignment_hours)}h alignment: no rows removed")
            continue
        detail = ", ".join(
            f"fold {int(row['fold_id'])} {row['role']} {int(row['n_dropped'])}"
            f"/{int(row['n_committed'])}"
            for _, row in subset.iterrows()
        )
        lines.append(f"  {int(alignment_hours)}h alignment: {detail}")

    best = best_supported_alignment(consistency, matched_rule)
    lines.extend(
        [
            "",
            f"Best supported alignment: {best}h "
            "(a shift must improve on every fold under both rules to be selected)",
        ]
    )
    return "\n".join(lines)


def run(
    hourly_path: Path,
    splits_path: Path,
    random_forest_path: Path,
    results_path: Path,
    spark=None,
):
    """Run every alignment, validate the guards, compare, and write results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityTemporalAlignment")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)

        results, evaluations = run_experiment(spark, hourly, assignment)

        if not random_forest_path.exists():
            raise FileNotFoundError(
                f"Random Forest benchmark results not found at: {random_forest_path.name}. "
                "Run that module first."
            )
        verify_reproduces_committed_benchmark(results, pd.read_parquet(random_forest_path))
        verify_reproducible(spark, hourly, assignment, ALIGNMENT_HOURS[-1])

        summary = summarize_alignments(results)
        comparison = compare_with_baseline_alignment(results)
        matched = matched_comparison(evaluations)
        consistency = assess_consistency(comparison)
        row_loss = pd.concat(
            [evaluation.dataset.row_loss for evaluation in evaluations.values()],
            ignore_index=True,
        )

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(results_path, index=False)

        return results, summary, comparison, matched, consistency, row_loss
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and evaluating sensor to target alignments...")
    results, summary, comparison, matched, consistency, row_loss = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_random_forest_path(repo_root),
        results_path,
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(results, summary, comparison, matched, consistency, row_loss))


if __name__ == "__main__":
    sys.exit(main())
