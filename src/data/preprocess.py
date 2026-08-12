"""Hourly preprocessing layer for the mining flotation raw dataset.

Converts the raw 20 second process log into one row per recorded hour.
Each high frequency process variable is summarized by its within hour
mean, standard deviation, and slope against normalized within hour
position. Slow changing feed variables are carried forward as a single
representative value per hour.

The Silica Concentrate target is inspected for two known raw data
conditions, both established during exploratory analysis and detected
here programmatically rather than by hardcoded timestamps:

* Interpolated hours, where the target follows a near exact linear
  progression across the hour instead of a constant hourly target value.
* Frozen sensor hours, where most or all high frequency sensors do not
  change at all within the hour.

Every hourly row is kept in the output table. Eligibility for primary
modeling is expressed through boolean and diagnostic columns rather
than by discarding rows, so the full table remains available for audit
and sensitivity analysis.

Target derived columns (the target itself, interpolation diagnostics,
and target run metadata) are kept in explicitly separate lists from
the predictor schema so they cannot be pulled into a feature vector by
accident.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Raw to standardized column mapping
# ---------------------------------------------------------------------

RAW_TO_STANDARD: dict[str, str] = {
    "date": "timestamp",
    "% Iron Feed": "iron_feed",
    "% Silica Feed": "silica_feed",
    "Starch Flow": "starch_flow",
    "Amina Flow": "amina_flow",
    "Ore Pulp Flow": "ore_pulp_flow",
    "Ore Pulp pH": "ore_pulp_ph",
    "Ore Pulp Density": "ore_pulp_density",
    "% Iron Concentrate": "iron_concentrate",
    "% Silica Concentrate": "silica_concentrate",
}
for _i in range(1, 8):
    RAW_TO_STANDARD[f"Flotation Column {_i:02d} Air Flow"] = f"flotation_column_{_i:02d}_air_flow"
    RAW_TO_STANDARD[f"Flotation Column {_i:02d} Level"] = f"flotation_column_{_i:02d}_level"

EXPECTED_RAW_COLUMNS = set(RAW_TO_STANDARD)

# ---------------------------------------------------------------------
# Variable groups (standardized names)
# ---------------------------------------------------------------------

TIMESTAMP_COLUMN = "timestamp"

TARGET_COLUMN = "silica_concentrate"
# Tracked only for raw schema validation and for asserting it never enters
# the predictor schema. It is deliberately never aggregated into the
# hourly dataset, so it cannot appear as a column at all downstream.
EXCLUDED_OUTCOME_COLUMN = "iron_concentrate"

FEED_COLUMNS = ["iron_feed", "silica_feed"]

HIGH_FREQUENCY_COLUMNS = (
    ["starch_flow", "amina_flow", "ore_pulp_flow", "ore_pulp_ph", "ore_pulp_density"]
    + [f"flotation_column_{i:02d}_air_flow" for i in range(1, 8)]
    + [f"flotation_column_{i:02d}_level" for i in range(1, 8)]
)
assert len(HIGH_FREQUENCY_COLUMNS) == 19

# Deterministic interpolation detection threshold.
#
# During exploratory analysis, every non-constant hour in the raw file
# fit a straight line against normalized within hour position with a
# maximum residual, normalized by the hour's value range, no larger
# than roughly 1.1e-8. That magnitude is consistent with floating
# point round trip noise rather than independent measurement. A
# threshold of 1e-4 leaves several orders of magnitude of margin above
# that observed noise floor while remaining far tighter than any
# plausible genuine measurement variation.
INTERPOLATION_RESIDUAL_THRESHOLD = 1e-4

# An hour is excluded from primary modeling when more than half of the
# 19 high frequency sensors are constant (frozen) within that hour.
SENSOR_FREEZE_MAJORITY_THRESHOLD = len(HIGH_FREQUENCY_COLUMNS) // 2  # 9 -> invalid at 10+


# ---------------------------------------------------------------------
# Predictor schema
# ---------------------------------------------------------------------

CORE_SENSOR_PREDICTOR_COLUMNS: list[str] = [
    f"{column}_{stat}" for column in HIGH_FREQUENCY_COLUMNS for stat in ("mean", "std", "slope")
]
assert len(CORE_SENSOR_PREDICTOR_COLUMNS) == 57

FEED_CONTEXT_PREDICTOR_COLUMNS: list[str] = list(FEED_COLUMNS)

ALL_PREDICTOR_COLUMNS: list[str] = CORE_SENSOR_PREDICTOR_COLUMNS + FEED_CONTEXT_PREDICTOR_COLUMNS

# Columns that must never appear in a predictor list: the target
# itself, its diagnostics, the excluded outcome, target run metadata,
# and row level quality/segment metadata.
NON_PREDICTOR_COLUMNS: set[str] = {
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
    "silica_concentrate_first",
    "silica_concentrate_last",
    "silica_concentrate_range",
    EXCLUDED_OUTCOME_COLUMN,
    "is_interpolated",
    "target_run_id",
    "target_run_length",
    "hours_since_target_change",
    "n_samples",
    "n_frozen_sensors",
    "is_sensor_valid",
    "is_sensor_model_eligible",
    "is_feed_model_eligible",
    "iron_feed_inconsistent",
    "silica_feed_inconsistent",
    "temporal_segment_id",
}


def get_predictor_columns() -> tuple[list[str], list[str], list[str]]:
    """Return (core sensor predictors, feed context predictors, all predictors)."""
    return list(CORE_SENSOR_PREDICTOR_COLUMNS), list(FEED_CONTEXT_PREDICTOR_COLUMNS), list(ALL_PREDICTOR_COLUMNS)


# ---------------------------------------------------------------------
# Raw ingestion
# ---------------------------------------------------------------------


def validate_raw_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if any expected raw column is missing."""
    missing = EXPECTED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw dataset is missing {len(missing)} expected column(s): {sorted(missing)}"
        )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and rename raw columns to machine friendly standardized names."""
    validate_raw_columns(df)
    return df.rename(columns=RAW_TO_STANDARD)


def load_raw(raw_csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV without modifying it on disk.

    Handles the comma decimal formatting used by the raw export and
    parses the hourly timestamp column.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path, decimal=",", parse_dates=["date"])
    df = standardize_columns(df)
    df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Within hour normalized position
# ---------------------------------------------------------------------


def add_normalized_position(df: pd.DataFrame, group_col: str = TIMESTAMP_COLUMN) -> pd.DataFrame:
    """Add a `_normalized_position` column mapping each row's position within
    its group onto [0, 1], regardless of how many rows the group has.

    This keeps slope calculations comparable across hours with 174, 179,
    or 180 raw observations. Position reflects preserved row order within
    each recorded hour; the raw file does not provide exact subhour
    timestamps for individual rows.
    """
    df = df.copy()
    group_sizes = df.groupby(group_col)[group_col].transform("size")
    position = df.groupby(group_col).cumcount()
    denominator = (group_sizes - 1).where(group_sizes > 1, other=1)
    df["_normalized_position"] = position / denominator
    return df


# ---------------------------------------------------------------------
# High frequency sensor aggregation (mean, std, slope)
# ---------------------------------------------------------------------


def _grouped_ols_slope(
    df: pd.DataFrame, value_col: str, x_col: str, group_col: str
) -> pd.Series:
    """Closed form OLS slope of `value_col` against `x_col`, per group.

    Uses the centered sum formula slope = sum((x - xbar)(y - ybar)) /
    sum((x - xbar)^2), computed with vectorized groupby operations
    (no per group Python loop) for determinism and speed.
    """
    x_mean = df.groupby(group_col)[x_col].transform("mean")
    y_mean = df.groupby(group_col)[value_col].transform("mean")
    x_centered = df[x_col] - x_mean
    y_centered = df[value_col] - y_mean

    numerator = (x_centered * y_centered).groupby(df[group_col]).sum()
    denominator = (x_centered * x_centered).groupby(df[group_col]).sum()

    slope = numerator / denominator.replace(0.0, np.nan)
    return slope.fillna(0.0)


def compute_high_frequency_aggregates(
    df: pd.DataFrame, columns: list[str], group_col: str = TIMESTAMP_COLUMN
) -> pd.DataFrame:
    """For each column, compute per hour mean, standard deviation, and
    normalized-position slope."""
    if "_normalized_position" not in df.columns:
        df = add_normalized_position(df, group_col)

    grouped = df.groupby(group_col)
    frames = []
    for column in columns:
        mean = grouped[column].mean().rename(f"{column}_mean")
        std = grouped[column].std(ddof=1).fillna(0.0).rename(f"{column}_std")
        slope = _grouped_ols_slope(df, column, "_normalized_position", group_col).rename(
            f"{column}_slope"
        )
        frames.extend([mean, std, slope])

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------
# Slow changing feed variables
# ---------------------------------------------------------------------


def compute_feed_context(
    df: pd.DataFrame, columns: list[str] = FEED_COLUMNS, group_col: str = TIMESTAMP_COLUMN
) -> pd.DataFrame:
    """Reduce each feed column to one representative value per hour.

    Each feed column is expected to be constant within every hour. If a
    hour unexpectedly contains more than one distinct value, that
    condition is flagged in a companion `<column>_inconsistent` column
    rather than silently ignored, and the first observed value (in
    chronological row order) is used as the deterministic representative.
    """
    grouped = df.groupby(group_col)
    frames = []
    for column in columns:
        representative = grouped[column].first().rename(column)
        inconsistent = (grouped[column].nunique() > 1).rename(f"{column}_inconsistent")
        frames.extend([representative, inconsistent])
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------
# Target construction and interpolation detection
# ---------------------------------------------------------------------


def detect_interpolated_hours(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = TIMESTAMP_COLUMN,
    threshold: float = INTERPOLATION_RESIDUAL_THRESHOLD,
) -> pd.DataFrame:
    """Detect hours whose target values follow a near exact linear
    progression against normalized within hour position, rather than a
    constant hourly target value.

    This is a deterministic statistical test, not a lookup of known
    timestamps: a straight line is fit to each hour's target sequence,
    and the hour is flagged when the largest fitting residual, relative
    to that hour's value range, falls below `threshold`. A hour with a
    single unique target value is never flagged (there is nothing to
    interpolate). The flag indicates that the target matches the
    deterministic linear progression pattern; it does not assert a
    known cause for why the pattern exists in the raw file.
    """
    if "_normalized_position" not in df.columns:
        df = add_normalized_position(df, group_col)

    grouped_target = df.groupby(group_col)[target_col]
    n_unique = grouped_target.nunique()
    value_range = grouped_target.transform("max") - grouped_target.transform("min")

    x_mean = df.groupby(group_col)["_normalized_position"].transform("mean")
    y_mean = grouped_target.transform("mean")
    x_centered = df["_normalized_position"] - x_mean
    y_centered = df[target_col] - y_mean

    slope_by_group = _grouped_ols_slope(df, target_col, "_normalized_position", group_col)
    slope_row = df[group_col].map(slope_by_group)

    fitted = y_mean + slope_row * x_centered
    residual = (df[target_col] - fitted).abs()
    normalized_residual = residual / value_range.replace(0.0, np.nan)

    max_normalized_residual = normalized_residual.groupby(df[group_col]).max()

    is_interpolated = (n_unique > 1) & (max_normalized_residual < threshold)
    is_interpolated = is_interpolated.fillna(False)

    result = pd.DataFrame(
        {
            "is_interpolated": is_interpolated,
            "_max_normalized_residual": max_normalized_residual,
        }
    )
    return result


def compute_target_summary(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN, group_col: str = TIMESTAMP_COLUMN
) -> pd.DataFrame:
    """Compute the primary hourly target value plus diagnostic summaries.

    The primary target column (named after `TARGET_COLUMN`) is the
    within hour mean. For a constant hour every raw value is identical,
    so the mean equals that single observed value exactly. For an
    interpolated hour, the mean is retained as a representative number
    together with first, last, and range diagnostics; such hours are
    expected to be excluded from primary modeling via `is_interpolated`.
    """
    grouped = df.groupby(group_col)[target_col]
    summary = pd.DataFrame(
        {
            target_col: grouped.mean(),
            f"{target_col}_first": grouped.first(),
            f"{target_col}_last": grouped.last(),
            f"{target_col}_range": grouped.max() - grouped.min(),
        }
    )
    return summary


# ---------------------------------------------------------------------
# Sensor freeze detection
# ---------------------------------------------------------------------


def compute_sensor_freeze(
    df: pd.DataFrame,
    columns: list[str] = HIGH_FREQUENCY_COLUMNS,
    group_col: str = TIMESTAMP_COLUMN,
    majority_threshold: int = SENSOR_FREEZE_MAJORITY_THRESHOLD,
) -> pd.DataFrame:
    """Count, per hour, how many high frequency sensors are constant
    (frozen) within that hour, and flag hours where more than half are.
    """
    grouped = df.groupby(group_col)
    frozen_flags = pd.concat(
        [(grouped[column].nunique() == 1).rename(column) for column in columns], axis=1
    )
    n_frozen_sensors = frozen_flags.sum(axis=1).rename("n_frozen_sensors")
    is_sensor_valid = (n_frozen_sensors <= majority_threshold).rename("is_sensor_valid")
    return pd.concat([n_frozen_sensors, is_sensor_valid], axis=1)


# ---------------------------------------------------------------------
# Temporal segments
# ---------------------------------------------------------------------


def assign_temporal_segments(
    hourly_df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COLUMN
) -> pd.Series:
    """Assign an integer temporal segment id based on discontinuities in
    the hourly timestamp sequence.

    A new segment starts whenever the gap to the previous hourly
    timestamp exceeds the modal (most common) gap observed in the
    sequence. No specific date or gap duration is hardcoded: both the
    normal interval and any anomalous gaps are derived from the data.
    """
    ordered = hourly_df[timestamp_col].sort_values()
    gaps = ordered.diff()
    modal_gap = gaps.mode().iloc[0]
    is_new_segment = gaps > modal_gap
    segment_id = is_new_segment.cumsum()
    segment_id.index = ordered.index
    return segment_id.reindex(hourly_df.index).rename("temporal_segment_id")


# ---------------------------------------------------------------------
# Target run metadata
# ---------------------------------------------------------------------


def compute_target_run_metadata(
    hourly_df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Compute target run metadata within each temporal segment.

    A run restarts when the target changes or when a new temporal segment
    begins. Resetting at segment boundaries prevents a large data gap from
    being treated as continuous target history merely because the values on
    either side happen to match. This metadata is for split protection and
    sensitivity analysis only, never for model predictors.
    """
    ordered = hourly_df.sort_values(TIMESTAMP_COLUMN)

    target_changed = ordered[target_col].ne(ordered[target_col].shift())
    if "temporal_segment_id" in ordered.columns:
        segment_changed = ordered["temporal_segment_id"].ne(
            ordered["temporal_segment_id"].shift()
        )
    else:
        segment_changed = pd.Series(False, index=ordered.index)
        if len(segment_changed):
            segment_changed.iloc[0] = True

    run_id = (target_changed | segment_changed).cumsum().rename("target_run_id")
    run_length = run_id.groupby(run_id).transform("size").rename("target_run_length")
    hours_since_change = run_id.groupby(run_id).cumcount().rename("hours_since_target_change")

    result = pd.concat([run_id, run_length, hours_since_change], axis=1)
    return result.reindex(hourly_df.index)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingSummary:
    raw_row_count: int
    hourly_row_count: int
    interpolated_hours: int
    sensor_invalid_hours: int
    sensor_model_eligible_hours: int
    feed_model_eligible_hours: int
    temporal_segments: int


def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full hourly analytical dataset from a standardized raw
    dataframe (as returned by `load_raw`).

    Every recorded hour is kept as one row. Diagnostic and eligibility
    columns describe data quality; no row is discarded here.
    """
    df = add_normalized_position(df, TIMESTAMP_COLUMN)

    n_samples = df.groupby(TIMESTAMP_COLUMN).size().rename("n_samples")

    sensor_aggregates = compute_high_frequency_aggregates(df, HIGH_FREQUENCY_COLUMNS)
    feed_context = compute_feed_context(df, FEED_COLUMNS)
    target_summary = compute_target_summary(df, TARGET_COLUMN)
    interpolation = detect_interpolated_hours(df, TARGET_COLUMN)
    sensor_freeze = compute_sensor_freeze(df, HIGH_FREQUENCY_COLUMNS)

    hourly = pd.concat(
        [n_samples, sensor_aggregates, feed_context, target_summary, interpolation, sensor_freeze],
        axis=1,
    )
    hourly.index.name = TIMESTAMP_COLUMN
    hourly = hourly.reset_index()
    hourly = hourly.drop(columns=["_max_normalized_residual"])

    hourly["temporal_segment_id"] = assign_temporal_segments(hourly)

    run_metadata = compute_target_run_metadata(hourly, TARGET_COLUMN)
    hourly = pd.concat([hourly, run_metadata], axis=1)

    core_predictors, feed_predictors, _ = get_predictor_columns()
    core_predictors_available = hourly[core_predictors].notna().all(axis=1)
    feed_predictors_available = hourly[feed_predictors].notna().all(axis=1)

    base_quality_eligible = (~hourly["is_interpolated"]) & hourly["is_sensor_valid"]

    # Primary configuration: process sensors only. Feed chemistry is kept
    # separate because its real-time operational availability is unresolved.
    hourly["is_sensor_model_eligible"] = base_quality_eligible & core_predictors_available

    # Secondary configuration: the same eligible sensor rows plus available
    # feed context. This allows a later sensor-only vs feed-enhanced comparison.
    hourly["is_feed_model_eligible"] = (
        hourly["is_sensor_model_eligible"] & feed_predictors_available
    )

    hourly = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    return hourly


def summarize(hourly: pd.DataFrame, raw_row_count: int) -> PreprocessingSummary:
    return PreprocessingSummary(
        raw_row_count=raw_row_count,
        hourly_row_count=len(hourly),
        interpolated_hours=int(hourly["is_interpolated"].sum()),
        sensor_invalid_hours=int((~hourly["is_sensor_valid"]).sum()),
        sensor_model_eligible_hours=int(hourly["is_sensor_model_eligible"].sum()),
        feed_model_eligible_hours=int(hourly["is_feed_model_eligible"].sum()),
        temporal_segments=int(hourly["temporal_segment_id"].nunique()),
    )


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing
    environment.yml is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "environment.yml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root (no environment.yml found in any parent "
        "directory). Run this module from within the mining-quality-prediction repository."
    )


def default_raw_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"


def default_output_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def run(raw_csv_path: Path, output_path: Path) -> PreprocessingSummary:
    """Read raw data, validate it, build the hourly dataset, and write it
    to `output_path` as Parquet. Returns a summary of the result."""
    raw_df = load_raw(raw_csv_path)
    hourly = build_hourly_dataset(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False)

    return summarize(hourly, raw_row_count=len(raw_df))


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    raw_path = default_raw_path(repo_root)
    output_path = default_output_path(repo_root)

    print("Loading raw mining dataset...")
    summary = run(raw_path, output_path)
    print("Hourly feature dataset written successfully.")

    core_predictors, feed_predictors, all_predictors = get_predictor_columns()

    print()
    print("Preprocessing summary")
    print("-" * 40)
    print(f"Raw rows:                  {summary.raw_row_count:,}")
    print(f"Hourly rows:                {summary.hourly_row_count:,}")
    print(f"Interpolated hours:         {summary.interpolated_hours:,}")
    print(f"Sensor invalid hours:       {summary.sensor_invalid_hours:,}")
    print(f"Sensor-model eligible hours:{summary.sensor_model_eligible_hours:>10,}")
    print(f"Feed-model eligible hours:  {summary.feed_model_eligible_hours:>10,}")
    print(f"Temporal segments:          {summary.temporal_segments:,}")
    print(f"Core sensor predictors:     {len(core_predictors)}")
    print(f"Feed context predictors:    {len(feed_predictors)}")
    print(f"Total predictors:           {len(all_predictors)}")


if __name__ == "__main__":
    sys.exit(main())
