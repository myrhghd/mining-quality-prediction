"""Baseline models for hourly `% Silica Concentrate` prediction.

Establishes reference performance for later regression models,
evaluated on the three development validation folds only. The final test
period is never loaded into a scoring population here.

Two baselines are evaluated:

* `training_mean` predicts a single constant, the mean target of that
  fold's training rows. It is the reference point for R squared: a model
  that cannot beat it has learned nothing useful.
* `persistence` predicts the most recent earlier observed target under
  a walk forward availability assumption. It measures how strong a naive
  autoregressive reference can be when earlier noninterpolated labels are
  treated as available; it is not a verified operator baseline.

Assay availability limitation
-----------------------------
The raw dataset records hourly target timestamps but no laboratory
reporting time, so there is no way to know when a result actually became
available to an operator. The persistence baseline therefore uses the
most recent earlier noninterpolated target among the assigned modeling
observations. No assay turnaround time is assumed or invented.

When validation history is enabled, the implementation assumes an earlier
validation target is available before the next scored timestamp. That is
a useful optimistic walk forward reference for temporal autocorrelation,
but the dataset does not establish that the assumption is operationally
true. The result must therefore not be presented as a verified operator
baseline or as a definitive deployment threshold.

Embargo treatment
-----------------
Embargo hours are excluded from training and from scoring, and they are
also excluded from the persistence state. A validation timestamp reaches
back past the embargo into the training side rather than carrying a
target forward from an embargo hour. This is deliberately conservative:
the embargo exists to withhold information from the model, so allowing
the baseline to read target values from it would hand the baseline an
advantage the validation design intends to remove.

Temporal segments
-----------------
Persistence never bridges a temporal discontinuity. A prediction is only
formed from an earlier observation in the same `temporal_segment_id`;
where no such observation exists the prediction is left unavailable and
counted, never imputed.
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

INTERPOLATED_COLUMN = "is_interpolated"

BASELINE_TRAINING_MEAN = "training_mean"
BASELINE_PERSISTENCE = "persistence"

DEVELOPMENT_FOLD_IDS = (1, 2, 3)

RESULT_COLUMNS = [
    "fold_id",
    "baseline",
    "n_train",
    "n_validation",
    "n_scored",
    "n_unavailable",
    "rmse",
    "mae",
    "r2",
]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = _as_paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = _as_paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, measured against the variance of
    `y_true` itself (the conventional definition).

    A constant predictor equal to the mean of `y_true` scores exactly 0.
    A predictor worse than that mean scores below 0, which is expected
    for a training mean evaluated on a later period.
    """
    y_true, y_pred = _as_paired_arrays(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        raise ValueError(
            "Cannot compute R squared: the observed target has zero variance "
            "over the scored observations."
        )
    return 1.0 - ss_res / ss_tot


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return RMSE, MAE, and R squared for one set of predictions.

    MAPE is deliberately not reported. `% Silica Concentrate` reaches
    values low enough that a percentage error becomes unstable and would
    exaggerate errors on the smallest observations.
    """
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def _as_paired_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Prediction shape {y_pred.shape} does not match observation shape {y_true.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute a metric over zero observations.")
    if not np.isfinite(y_true).all():
        raise ValueError("Observed targets contain non-finite values.")
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain non-finite values.")
    return y_true, y_pred


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FoldFrames:
    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame
    embargo: pd.DataFrame


def load_inputs(hourly_path: Path, splits_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the hourly feature table and the committed split assignments."""
    for path, description in ((hourly_path, "hourly feature"), (splits_path, "split assignment")):
        if not path.exists():
            raise FileNotFoundError(
                f"Required {description} dataset not found at: {path}. "
                "Run the preprocessing and split modules first."
            )

    hourly = pd.read_parquet(hourly_path)
    assignment = pd.read_parquet(splits_path)

    required_hourly = [
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEGMENT_COLUMN,
        SENSOR_ELIGIBLE_COLUMN,
        INTERPOLATED_COLUMN,
    ]
    missing = [column for column in required_hourly if column not in hourly.columns]
    if missing:
        raise ValueError(f"Hourly dataset is missing required column(s): {missing}")

    required_assignment = [TIMESTAMP_COLUMN, "fold_id", "fold_kind", "role"]
    missing = [column for column in required_assignment if column not in assignment.columns]
    if missing:
        raise ValueError(f"Split assignment is missing required column(s): {missing}")

    hourly = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    return hourly, assignment


def get_fold_frames(hourly: pd.DataFrame, assignment: pd.DataFrame, fold_id: int) -> FoldFrames:
    """Join one development fold's assignment back onto the hourly table.

    Only development folds may be requested. The final test fold is
    rejected outright so it cannot be scored by accident.
    """
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError(
            "The final test fold must not be evaluated in the baseline milestone."
        )
    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise ValueError(f"Unknown development fold id: {fold_id}")

    fold_rows = assignment[
        (assignment["fold_id"] == fold_id) & (assignment["fold_kind"] == KIND_DEVELOPMENT)
    ]
    if fold_rows.empty:
        raise ValueError(f"No development assignment rows found for fold {fold_id}.")

    indexed = hourly.set_index(TIMESTAMP_COLUMN)

    def frame_for(role: str) -> pd.DataFrame:
        timestamps = fold_rows.loc[fold_rows["role"] == role, TIMESTAMP_COLUMN]
        selected = indexed.loc[timestamps].reset_index()
        return selected.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    return FoldFrames(
        fold_id=fold_id,
        train=frame_for(ROLE_TRAIN),
        validation=frame_for(ROLE_VALIDATION),
        embargo=frame_for(ROLE_EMBARGO),
    )


# ---------------------------------------------------------------------
# Baseline A: training mean
# ---------------------------------------------------------------------


def predict_training_mean(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Predict the mean training target for every validation observation.

    The constant is computed from training rows only; nothing from the
    validation window contributes to it.
    """
    if train.empty:
        raise ValueError("Cannot compute a training mean from an empty training set.")

    constant = float(train[TARGET_COLUMN].mean())
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: validation[TIMESTAMP_COLUMN].to_numpy(),
            "prediction": constant,
            "source_timestamp": pd.NaT,
        }
    )


# ---------------------------------------------------------------------
# Baseline B: persistence / last known assay
# ---------------------------------------------------------------------


def predict_persistence(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    use_validation_history: bool = True,
) -> pd.DataFrame:
    """Predict the most recent earlier observed target value.

    The state pool is built from training rows and, by default, from
    earlier rows of the validation window itself. Walking the state
    forward through validation represents an explicit optimistic
    availability assumption: an earlier validation target is treated as
    available before the next scored timestamp. The raw dataset does not
    contain reporting timestamps, so this assumption cannot be verified.
    The first validation hour has no predecessor inside the window, so it
    reaches back past the embargo into training history rather than being
    left undefined.

    Embargo rows are never part of the pool. Interpolated hours are
    excluded because they are not observed assays. Matching is restricted
    to the same temporal segment, so a discontinuity resets the state
    instead of carrying a stale value across the gap.

    Set `use_validation_history=False` for the stricter variant in which
    the state freezes at the last training observation and every
    validation hour receives that same value.
    """
    pool_parts = [train]
    if use_validation_history:
        pool_parts.append(validation)

    pool = pd.concat(pool_parts, ignore_index=True)
    # Defensive: assigned rows are already eligible and noninterpolated,
    # but an interpolated hour must never act as a known assay.
    pool = pool[~pool[INTERPOLATED_COLUMN].astype(bool)]
    pool = pool[[TIMESTAMP_COLUMN, TARGET_COLUMN, SEGMENT_COLUMN]].copy()
    pool["source_timestamp"] = pool[TIMESTAMP_COLUMN]
    pool = pool.rename(columns={TARGET_COLUMN: "prediction"})
    pool = pool.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    targets = (
        validation[[TIMESTAMP_COLUMN, SEGMENT_COLUMN]]
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )

    if pool.empty:
        return pd.DataFrame(
            {
                TIMESTAMP_COLUMN: targets[TIMESTAMP_COLUMN].to_numpy(),
                "prediction": np.nan,
                "source_timestamp": pd.NaT,
            }
        )

    # `allow_exact_matches=False` enforces strictly earlier information.
    # `by=SEGMENT_COLUMN` prevents any match across a temporal gap.
    merged = pd.merge_asof(
        targets,
        pool,
        on=TIMESTAMP_COLUMN,
        by=SEGMENT_COLUMN,
        direction="backward",
        allow_exact_matches=False,
    )

    return merged[[TIMESTAMP_COLUMN, "prediction", "source_timestamp"]]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_fold(frames: FoldFrames, use_validation_history: bool = True) -> pd.DataFrame:
    """Evaluate both baselines on one development fold."""
    validation = frames.validation
    observed = validation.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]

    rows = []
    predictions = {
        BASELINE_TRAINING_MEAN: predict_training_mean(frames.train, validation),
        BASELINE_PERSISTENCE: predict_persistence(
            frames.train, validation, use_validation_history=use_validation_history
        ),
    }

    for baseline, predicted in predictions.items():
        available = predicted[predicted["prediction"].notna()]
        n_unavailable = len(predicted) - len(available)

        if available.empty:
            raise ValueError(
                f"Fold {frames.fold_id} baseline {baseline} produced no usable predictions."
            )

        y_true = observed.loc[available[TIMESTAMP_COLUMN]].to_numpy()
        y_pred = available["prediction"].to_numpy()
        metrics = compute_metrics(y_true, y_pred)

        rows.append(
            {
                "fold_id": frames.fold_id,
                "baseline": baseline,
                "n_train": len(frames.train),
                "n_validation": len(validation),
                "n_scored": len(available),
                "n_unavailable": n_unavailable,
                **metrics,
            }
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def evaluate_baselines(
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
    use_validation_history: bool = True,
) -> pd.DataFrame:
    """Evaluate both baselines across every development fold."""
    results = [
        evaluate_fold(
            get_fold_frames(hourly, assignment, fold_id),
            use_validation_history=use_validation_history,
        )
        for fold_id in fold_ids
    ]
    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values(["baseline", "fold_id"], kind="mergesort").reset_index(drop=True)


def summarize_development(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds.

    Fold level results are reported alongside this summary rather than
    being replaced by it: three folds is too few for an average to stand
    on its own, and the spread across folds carries the more useful
    information about stability.
    """
    summary = (
        results.groupby("baseline")
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
    return summary.sort_values("baseline", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    results: pd.DataFrame,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    use_validation_history: bool = True,
) -> None:
    """Verify every structural guard, raising a clear error on violation."""
    if list(results.columns) != RESULT_COLUMNS:
        raise ValueError(
            f"Result schema mismatch. Expected {RESULT_COLUMNS}, got {list(results.columns)}"
        )

    final_test_timestamps = set(
        assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    )
    eligible_timestamps = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])

    for _, row in results.iterrows():
        if row["n_scored"] > row["n_validation"]:
            raise ValueError(
                f"Fold {row['fold_id']} baseline {row['baseline']}: scored "
                f"{row['n_scored']} rows but only {row['n_validation']} validation rows exist."
            )
        if row["n_scored"] + row["n_unavailable"] != row["n_validation"]:
            raise ValueError(
                f"Fold {row['fold_id']} baseline {row['baseline']}: scored plus unavailable "
                "does not equal the validation count."
            )
        for metric in ("rmse", "mae", "r2"):
            if not np.isfinite(row[metric]):
                raise ValueError(
                    f"Fold {row['fold_id']} baseline {row['baseline']}: {metric} is not finite."
                )
        if row["rmse"] < 0 or row["mae"] < 0:
            raise ValueError(
                f"Fold {row['fold_id']} baseline {row['baseline']}: negative error metric."
            )

    for fold_id in sorted(results["fold_id"].unique()):
        frames = get_fold_frames(hourly, assignment, int(fold_id))
        train_timestamps = set(frames.train[TIMESTAMP_COLUMN])
        validation_timestamps = set(frames.validation[TIMESTAMP_COLUMN])
        embargo_timestamps = set(frames.embargo[TIMESTAMP_COLUMN])

        # Only development validation rows may be scored, and every one of
        # them must belong to the eligible sensor population.
        if not validation_timestamps.issubset(eligible_timestamps):
            raise ValueError(f"Fold {fold_id}: validation contains ineligible hours.")
        if not train_timestamps.issubset(eligible_timestamps):
            raise ValueError(f"Fold {fold_id}: training contains ineligible hours.")

        # The final test period must be untouched.
        for name, timestamps in (
            ("training", train_timestamps),
            ("validation", validation_timestamps),
        ):
            overlap = timestamps.intersection(final_test_timestamps)
            if overlap:
                raise ValueError(
                    f"Fold {fold_id}: {name} overlaps the final test period "
                    f"({len(overlap)} hours)."
                )

        # No validation target may enter the training mean.
        if train_timestamps.intersection(validation_timestamps):
            raise ValueError(f"Fold {fold_id}: training and validation timestamps overlap.")
        recomputed = float(frames.train[TARGET_COLUMN].mean())
        prediction = predict_training_mean(frames.train, frames.validation)
        if not np.isclose(prediction["prediction"].iloc[0], recomputed):
            raise ValueError(f"Fold {fold_id}: training mean does not match training rows.")

        # Persistence guards, checked against the emitted source timestamps.
        persistence = predict_persistence(
            frames.train, frames.validation, use_validation_history=use_validation_history
        )
        matched = persistence[persistence["source_timestamp"].notna()]

        if not (matched["source_timestamp"] < matched[TIMESTAMP_COLUMN]).all():
            raise ValueError(
                f"Fold {fold_id}: persistence used a target at or after the predicted hour."
            )
        used_sources = set(matched["source_timestamp"])
        if used_sources.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: persistence used an embargo target value.")
        if used_sources.intersection(final_test_timestamps):
            raise ValueError(f"Fold {fold_id}: persistence used a final test target value.")

        allowed_sources = train_timestamps | (
            validation_timestamps if use_validation_history else set()
        )
        if not used_sources.issubset(allowed_sources):
            raise ValueError(f"Fold {fold_id}: persistence used a target outside the allowed pool.")

        segments = hourly.set_index(TIMESTAMP_COLUMN)[SEGMENT_COLUMN]
        source_segments = segments.loc[matched["source_timestamp"]].to_numpy()
        target_segments = segments.loc[matched[TIMESTAMP_COLUMN]].to_numpy()
        if not (source_segments == target_segments).all():
            raise ValueError(f"Fold {fold_id}: persistence crossed a temporal segment boundary.")

        # Interpolated hours are not observed assays.
        interpolated = set(hourly.loc[hourly[INTERPOLATED_COLUMN].astype(bool), TIMESTAMP_COLUMN])
        if used_sources.intersection(interpolated):
            raise ValueError(f"Fold {fold_id}: persistence used an interpolated target value.")


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "baseline_results.parquet"


def run(
    hourly_path: Path, splits_path: Path, results_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both baselines, validate the guards, and write the results."""
    hourly, assignment = load_inputs(hourly_path, splits_path)
    results = evaluate_baselines(hourly, assignment)
    validate_evaluation(results, hourly, assignment)
    summary = summarize_development(results)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path, index=False)

    return results, summary


def format_summary(results: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render the fold level results and the development summary."""
    lines = ["Baseline evaluation (development folds only)", ""]
    lines.append(
        f"{'fold':>4}  {'baseline':<14}  {'n_train':>7}  {'n_val':>5}  "
        f"{'scored':>6}  {'unavail':>7}  {'rmse':>7}  {'mae':>7}  {'r2':>8}"
    )
    for _, row in results.iterrows():
        lines.append(
            f"{int(row['fold_id']):>4}  {row['baseline']:<14}  {int(row['n_train']):>7,}  "
            f"{int(row['n_validation']):>5,}  {int(row['n_scored']):>6,}  "
            f"{int(row['n_unavailable']):>7,}  {row['rmse']:>7.4f}  {row['mae']:>7.4f}  "
            f"{row['r2']:>8.4f}"
        )

    lines.extend(["", "Development summary across folds", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"  {row['baseline']:<14}  "
            f"RMSE {row['rmse_mean']:.4f} (sd {row['rmse_std']:.4f})   "
            f"MAE {row['mae_mean']:.4f} (sd {row['mae_std']:.4f})   "
            f"R2 {row['r2_mean']:.4f} (sd {row['r2_std']:.4f})"
        )
    return "\n".join(lines)


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    hourly_path = default_hourly_path(repo_root)
    splits_path = default_splits_path(repo_root)
    results_path = default_results_path(repo_root)

    print("Evaluating baselines...")
    results, summary = run(hourly_path, splits_path, results_path)
    print(f"Baseline results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_summary(results, summary))


if __name__ == "__main__":
    sys.exit(main())
