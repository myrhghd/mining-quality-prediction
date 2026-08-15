"""Backward looking process history features for the hourly mining dataset.

The 57 static hourly sensor aggregates describe one hour in isolation:
each process variable contributes its within hour mean, standard
deviation, and slope. They carry no information about what the plant was
doing before that hour, and no information about where inside the hour
the process ended up.

This module adds that information. Every feature summarizes raw 20 second
observations drawn from a window that ends with the hour being predicted
and extends backward from it. Nothing is read from a later hour, and
nothing is read from a later position than the end of the predicted hour.

Availability at prediction time
-------------------------------
The information boundary is exactly the one the existing static features
already assume: an hourly row labelled `t` may use every raw observation
recorded during hour `t` and during any earlier hour. The project's
committed 0 hour alignment already pairs hour `t` sensor aggregates with
the hour `t` assay, so these features add process history without
widening that assumption. No feature here reaches into hour `t + 1` or
beyond.

Sub hour windows
----------------
The raw file stamps every observation with its hour and records 180
observations for almost every hour, which is a 20 second sampling
interval. It does not carry an exact sub hour timestamp per row, so a
window of 15 or 30 minutes is taken as the corresponding trailing
fraction of that hour's preserved row order. This is the same assumption
the existing within hour slope already makes: rows arrive in order and
are evenly spaced across the hour.

Cross hour windows
------------------
A 120 minute window pools the raw observations of hour `t - 1` and hour
`t`. It is built only when the previous hour is present in the raw
record, sits in the same temporal segment, and is not a frozen sensor
hour. A frozen hour would contribute a block of unchanging readings and
make the plant look artificially steady, so it is treated as missing
context rather than as data. Rows without full context are flagged
through `HAS_CONTEXT_COLUMN` and are never given imputed values.

Feature scope
-------------
Statistics are not generated for every window and every variable. The
recency and change features apply to all 19 high frequency process
variables. The extreme, stability, and short term slope features apply
only to the five independently manipulated process variables, because
the 14 flotation column channels are seven near duplicate pairs of the
same two measurements and repeating those statistics seven times would
add width without adding information.

Feed chemistry, the iron concentrate outcome, and any calendar derived
value are outside this module entirely.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    HIGH_FREQUENCY_COLUMNS,
    TIMESTAMP_COLUMN,
    _grouped_ols_slope,
    find_repo_root,
)

# ---------------------------------------------------------------------
# Window and variable scope
# ---------------------------------------------------------------------

# Nominal minutes covered by one recorded hour. Used to convert a
# trailing window length into a fraction of that hour's rows.
MINUTES_PER_HOUR = 60

SHORT_WINDOW_MINUTES = 15
SLOPE_WINDOW_MINUTES = 30
HOUR_WINDOW_MINUTES = 60
LONG_WINDOW_MINUTES = 120

# The five process variables an operator sets or controls directly. The
# extreme, stability, and short term slope features are restricted to
# these because the seven air flow channels and the seven level channels
# are near duplicates of one another.
PRIMARY_PROCESS_COLUMNS: list[str] = [
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_ph",
    "ore_pulp_density",
]
assert set(PRIMARY_PROCESS_COLUMNS).issubset(HIGH_FREQUENCY_COLUMNS)

SEGMENT_COLUMN = "temporal_segment_id"
SENSOR_VALID_COLUMN = "is_sensor_valid"

# Marks hours whose full 120 minute window could be built. Diagnostic
# metadata, never a predictor.
HAS_CONTEXT_COLUMN = "has_dynamic_context"

# Internal helper columns, removed before the table is returned.
POSITION_COLUMN = "_within_hour_position"
GROUP_SIZE_COLUMN = "_within_hour_size"
WINDOW_MINUTE_COLUMN = "_window_minute"


# ---------------------------------------------------------------------
# Predictor schema
# ---------------------------------------------------------------------

# Recency and change, for all 19 high frequency process variables.
#
# trailing_15m_mean          Where the process actually sat in the final
#                            quarter of the hour, which the whole hour
#                            mean averages away.
# trailing_120m_mean         The level the plant has been holding over
#                            the last two hours.
# trailing_15m_minus_120m_mean  Divergence between the most recent
#                            reading and that longer trailing level. A
#                            tree cannot form this difference from the
#                            two levels on its own, because a split
#                            examines one feature at a time.
# change_1h                  Movement from the previous hour's mean to
#                            this hour's mean.
RECENCY_FEATURE_SUFFIXES: list[str] = [
    "trailing_15m_mean",
    "trailing_120m_mean",
    "trailing_15m_minus_120m_mean",
    "change_1h",
]

# Extremes, stability, and short term direction, for the five primary
# process variables.
#
# trailing_30m_slope         Direction and rate of movement over the
#                            final half hour, in units per minute.
# trailing_120m_std          Variability across the full two hour
#                            window, which spans more than the single
#                            hour the static standard deviation covers.
# trailing_60m_min / _max    The extremes reached during the hour. A
#                            mean and a standard deviation do not
#                            recover a brief excursion.
PRIMARY_FEATURE_SUFFIXES: list[str] = [
    "trailing_30m_slope",
    "trailing_120m_std",
    "trailing_60m_min",
    "trailing_60m_max",
]

DYNAMIC_PREDICTOR_COLUMNS: list[str] = [
    f"{column}_{suffix}"
    for column in HIGH_FREQUENCY_COLUMNS
    for suffix in RECENCY_FEATURE_SUFFIXES
] + [
    f"{column}_{suffix}"
    for column in PRIMARY_PROCESS_COLUMNS
    for suffix in PRIMARY_FEATURE_SUFFIXES
]
assert len(DYNAMIC_PREDICTOR_COLUMNS) == 96
assert len(set(DYNAMIC_PREDICTOR_COLUMNS)) == 96


def get_dynamic_predictor_columns() -> list[str]:
    """Return the 96 dynamic process history predictors."""
    return list(DYNAMIC_PREDICTOR_COLUMNS)


# ---------------------------------------------------------------------
# Within hour position
# ---------------------------------------------------------------------


def add_within_hour_index(
    raw: pd.DataFrame, group_col: str = TIMESTAMP_COLUMN
) -> pd.DataFrame:
    """Add each raw row's ordinal position within its hour and that hour's size.

    Position reflects preserved row order. The raw file does not provide
    a sub hour timestamp, so order is the only available notion of when
    inside the hour an observation was taken.
    """
    df = raw.copy()
    df[POSITION_COLUMN] = df.groupby(group_col).cumcount()
    df[GROUP_SIZE_COLUMN] = df.groupby(group_col)[group_col].transform("size")
    return df


def window_row_count(group_size: pd.Series, minutes: int) -> pd.Series:
    """Number of trailing rows that cover `minutes` of a recorded hour.

    Rounded up so a window never covers less time than requested, and
    capped at the hour's own row count. Deriving the count from each
    hour's own size keeps the window length correct for the two hours
    that hold fewer than 180 observations.
    """
    if not 0 < minutes <= MINUTES_PER_HOUR:
        raise ValueError(
            f"A sub hour window must be between 1 and {MINUTES_PER_HOUR} minutes, got {minutes}."
        )
    count = np.ceil(group_size.to_numpy(dtype=float) * minutes / MINUTES_PER_HOUR)
    count = np.minimum(count, group_size.to_numpy(dtype=float))
    return pd.Series(count.astype(int), index=group_size.index)


def trailing_window_mask(indexed_raw: pd.DataFrame, minutes: int) -> pd.Series:
    """Select the rows in the final `minutes` of each recorded hour.

    The mask is anchored to the end of the hour, so it can only ever move
    backward from the prediction boundary.
    """
    rows = window_row_count(indexed_raw[GROUP_SIZE_COLUMN], minutes)
    first_kept = indexed_raw[GROUP_SIZE_COLUMN] - rows
    return indexed_raw[POSITION_COLUMN] >= first_kept


# ---------------------------------------------------------------------
# Sub hour trailing statistics
# ---------------------------------------------------------------------


def compute_trailing_window_stats(
    indexed_raw: pd.DataFrame,
    columns: list[str],
    minutes: int,
    statistics: tuple[str, ...],
    group_col: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Summarize the final `minutes` of every recorded hour.

    Only `mean`, `min`, and `max` are supported here; the slope has its
    own function because it needs an explicit time axis.
    """
    unsupported = set(statistics) - {"mean", "min", "max"}
    if unsupported:
        raise ValueError(f"Unsupported trailing statistic(s): {sorted(unsupported)}")

    window = indexed_raw[trailing_window_mask(indexed_raw, minutes)]
    grouped = window.groupby(group_col)

    frames = []
    for column in columns:
        for statistic in statistics:
            values = getattr(grouped[column], statistic)()
            frames.append(values.rename(f"{column}_trailing_{minutes}m_{statistic}"))
    return pd.concat(frames, axis=1)


def compute_trailing_slope(
    indexed_raw: pd.DataFrame,
    columns: list[str],
    minutes: int,
    group_col: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Ordinary least squares slope over the final `minutes` of each hour.

    The time axis is minutes elapsed inside the window, derived from each
    hour's own row spacing, so the slope is a rate per minute and stays
    comparable across hours that hold different row counts.
    """
    window = indexed_raw[trailing_window_mask(indexed_raw, minutes)].copy()

    rows = window_row_count(window[GROUP_SIZE_COLUMN], minutes)
    first_kept = window[GROUP_SIZE_COLUMN] - rows
    minutes_per_row = MINUTES_PER_HOUR / window[GROUP_SIZE_COLUMN]
    window[WINDOW_MINUTE_COLUMN] = (window[POSITION_COLUMN] - first_kept) * minutes_per_row

    frames = []
    for column in columns:
        slope = _grouped_ols_slope(window, column, WINDOW_MINUTE_COLUMN, group_col)
        frames.append(slope.rename(f"{column}_trailing_{minutes}m_slope"))
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------
# Cross hour trailing statistics
# ---------------------------------------------------------------------


def compute_hourly_sufficient_statistics(
    indexed_raw: pd.DataFrame, columns: list[str], group_col: str = TIMESTAMP_COLUMN
) -> pd.DataFrame:
    """Per hour count, sum, and sum of squares for each column.

    A window spanning two hours is then assembled by adding the two
    hours' statistics, which gives exactly the same mean and standard
    deviation as pooling the raw observations without scanning them
    twice.
    """
    grouped = indexed_raw.groupby(group_col)
    frames = [grouped.size().rename("_count")]
    for column in columns:
        frames.append(grouped[column].sum().rename(f"{column}_sum"))
        frames.append((indexed_raw[column] ** 2).groupby(indexed_raw[group_col]).sum().rename(f"{column}_sumsq"))
    return pd.concat(frames, axis=1)


def previous_hour_available(
    hourly: pd.DataFrame,
    recorded_hours: pd.Index,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> pd.Series:
    """Flag hours whose immediately preceding hour can supply history.

    The previous hour must be present in the raw record, belong to the
    same temporal segment, and not be a frozen sensor hour. Requiring the
    same segment stops a window from bridging a recording gap; excluding
    frozen hours stops a stalled instrument from being read as a steady
    process.
    """
    ordered = hourly.sort_values(timestamp_col, kind="mergesort")
    previous = ordered[timestamp_col] - pd.Timedelta(1, unit="h")

    segment = ordered.set_index(timestamp_col)[SEGMENT_COLUMN]
    valid = ordered.set_index(timestamp_col)[SENSOR_VALID_COLUMN]

    is_recorded = previous.isin(set(recorded_hours))
    same_segment = previous.map(segment).eq(ordered[SEGMENT_COLUMN])
    # `eq(True)` rather than a fill, so an hour with no predecessor at all
    # resolves to False without a dtype conversion.
    previous_valid = previous.map(valid).eq(True)

    available = (is_recorded & same_segment & previous_valid).to_numpy()
    return pd.Series(available, index=ordered[timestamp_col], name=HAS_CONTEXT_COLUMN)


def compute_long_window_stats(
    sufficient: pd.DataFrame,
    columns: list[str],
    mean_columns: list[str],
    std_columns: list[str],
    available: pd.Series,
    minutes: int = LONG_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Mean and standard deviation over the window ending at each hour.

    The window pools hour `t - 1` and hour `t`. Hours whose previous hour
    cannot supply history receive missing values rather than a window
    silently shortened to one hour, so an incomplete window is never
    presented as a complete one.

    The standard deviation uses the sample convention `ddof=1`, matching
    the existing within hour standard deviation.
    """
    ordered = sufficient.sort_index()
    shifted = ordered.shift(1)

    # A shift is only meaningful where the previous row really is the
    # previous hour; `available` already encodes that condition.
    mask = available.reindex(ordered.index).fillna(False).astype(bool).to_numpy()

    count = (ordered["_count"] + shifted["_count"]).to_numpy(dtype=float)

    frames = []
    for column, mean_name, std_name in zip(columns, mean_columns, std_columns):
        total = (ordered[f"{column}_sum"] + shifted[f"{column}_sum"]).to_numpy(dtype=float)
        total_squares = (
            ordered[f"{column}_sumsq"] + shifted[f"{column}_sumsq"]
        ).to_numpy(dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
            # Numerical noise can push a variance fractionally below zero
            # when a variable holds an almost constant value across the
            # window, so it is floored before the square root.
            variance = np.maximum((total_squares - count * mean**2) / (count - 1.0), 0.0)

        frames.append(pd.Series(np.where(mask, mean, np.nan), index=ordered.index, name=mean_name))
        if std_name is not None:
            frames.append(
                pd.Series(
                    np.where(mask, np.sqrt(variance), np.nan), index=ordered.index, name=std_name
                )
            )

    result = pd.concat(frames, axis=1)
    result.index.name = TIMESTAMP_COLUMN
    return result


def compute_hourly_change(
    hourly: pd.DataFrame,
    columns: list[str],
    available: pd.Series,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Difference between this hour's mean and the previous hour's mean.

    Computed from the committed static hourly means, so the change
    feature and the level it is derived from cannot disagree. Hours
    without usable previous history receive missing values.
    """
    ordered = hourly.sort_values(timestamp_col, kind="mergesort").set_index(timestamp_col)
    mask = available.reindex(ordered.index).fillna(False).astype(bool).to_numpy()

    frames = []
    for column in columns:
        means = ordered[f"{column}_mean"]
        change = (means - means.shift(1)).to_numpy(dtype=float)
        frames.append(
            pd.Series(np.where(mask, change, np.nan), index=ordered.index, name=f"{column}_change_1h")
        )
    result = pd.concat(frames, axis=1)
    result.index.name = TIMESTAMP_COLUMN
    return result


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicFeatureSummary:
    raw_row_count: int
    hourly_row_count: int
    hours_with_context: int
    hours_without_context: int
    n_dynamic_features: int


def build_dynamic_features(raw: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Build the dynamic process history table, one row per recorded hour.

    `raw` is the standardized 20 second record returned by
    `src.data.preprocess.load_raw`. `hourly` is the committed hourly
    table, used for its static means, its temporal segments, and its
    sensor freeze flag. No row is discarded; hours without a full 120
    minute window carry missing values and a false `HAS_CONTEXT_COLUMN`.
    """
    indexed = add_within_hour_index(raw)

    short_means = compute_trailing_window_stats(
        indexed, HIGH_FREQUENCY_COLUMNS, SHORT_WINDOW_MINUTES, ("mean",)
    )
    hour_extremes = compute_trailing_window_stats(
        indexed, PRIMARY_PROCESS_COLUMNS, HOUR_WINDOW_MINUTES, ("min", "max")
    )
    slopes = compute_trailing_slope(indexed, PRIMARY_PROCESS_COLUMNS, SLOPE_WINDOW_MINUTES)

    sufficient = compute_hourly_sufficient_statistics(indexed, HIGH_FREQUENCY_COLUMNS)
    available = previous_hour_available(hourly, sufficient.index)

    long_means = compute_long_window_stats(
        sufficient,
        HIGH_FREQUENCY_COLUMNS,
        mean_columns=[f"{column}_trailing_120m_mean" for column in HIGH_FREQUENCY_COLUMNS],
        std_columns=[
            f"{column}_trailing_120m_std" if column in PRIMARY_PROCESS_COLUMNS else None
            for column in HIGH_FREQUENCY_COLUMNS
        ],
        available=available,
    )
    changes = compute_hourly_change(hourly, HIGH_FREQUENCY_COLUMNS, available)

    features = pd.concat(
        [short_means, hour_extremes, slopes, long_means, changes], axis=1
    )

    # Short against long divergence. Both operands are trailing means
    # ending at the same hour, so the difference is backward looking by
    # construction.
    for column in HIGH_FREQUENCY_COLUMNS:
        features[f"{column}_trailing_15m_minus_120m_mean"] = (
            features[f"{column}_trailing_15m_mean"] - features[f"{column}_trailing_120m_mean"]
        )

    features[HAS_CONTEXT_COLUMN] = available.reindex(features.index).fillna(False).astype(bool)

    features.index.name = TIMESTAMP_COLUMN
    features = features.reset_index()
    features = features[[TIMESTAMP_COLUMN, *DYNAMIC_PREDICTOR_COLUMNS, HAS_CONTEXT_COLUMN]]
    return features.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)


def validate_dynamic_features(features: pd.DataFrame, hourly: pd.DataFrame) -> None:
    """Verify the structural guarantees of the dynamic feature table."""
    expected_columns = [TIMESTAMP_COLUMN, *DYNAMIC_PREDICTOR_COLUMNS, HAS_CONTEXT_COLUMN]
    if list(features.columns) != expected_columns:
        raise ValueError("Dynamic feature table does not match the declared schema.")

    if not features[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError("Dynamic feature rows are not in chronological order.")
    if features[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError("Dynamic feature table contains duplicate hours.")

    hourly_hours = set(hourly[TIMESTAMP_COLUMN])
    if set(features[TIMESTAMP_COLUMN]) != hourly_hours:
        raise ValueError("Dynamic feature hours differ from the committed hourly chronology.")

    with_context = features[features[HAS_CONTEXT_COLUMN]]
    values = with_context[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A dynamic feature is missing or non finite on an hour marked usable.")

    # Every feature that needs the previous hour must be absent exactly
    # where the context is absent, so an incomplete window can never be
    # mistaken for a complete one.
    cross_hour = [
        column
        for column in DYNAMIC_PREDICTOR_COLUMNS
        if column.endswith(("trailing_120m_mean", "trailing_120m_std", "change_1h", "trailing_15m_minus_120m_mean"))
    ]
    without_context = features[~features[HAS_CONTEXT_COLUMN]]
    if not without_context.empty:
        if without_context[cross_hour].notna().to_numpy().any():
            raise ValueError("An hour without history carries a cross hour feature value.")

    forbidden = {"iron_feed", "silica_feed", "iron_concentrate", "silica_concentrate"}
    leaked = forbidden.intersection(DYNAMIC_PREDICTOR_COLUMNS)
    if leaked:
        raise ValueError(f"Forbidden columns present in the dynamic schema: {sorted(leaked)}")


def summarize(
    features: pd.DataFrame, raw_row_count: int
) -> DynamicFeatureSummary:
    return DynamicFeatureSummary(
        raw_row_count=raw_row_count,
        hourly_row_count=len(features),
        hours_with_context=int(features[HAS_CONTEXT_COLUMN].sum()),
        hours_without_context=int((~features[HAS_CONTEXT_COLUMN]).sum()),
        n_dynamic_features=len(DYNAMIC_PREDICTOR_COLUMNS),
    )


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_raw_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_output_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "dynamic_features.parquet"


def run(raw_csv_path: Path, hourly_path: Path, output_path: Path) -> DynamicFeatureSummary:
    """Build the dynamic feature table, validate it, and write it to Parquet."""
    from src.data.preprocess import load_raw

    if not hourly_path.exists():
        raise FileNotFoundError(
            f"Hourly feature dataset not found at: {hourly_path}. "
            "Run the preprocessing module first."
        )

    raw = load_raw(raw_csv_path)
    hourly = pd.read_parquet(hourly_path)

    features = build_dynamic_features(raw, hourly)
    validate_dynamic_features(features, hourly)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return summarize(features, raw_row_count=len(raw))


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    output_path = default_output_path(repo_root)

    print("Building dynamic process history features...")
    summary = run(
        default_raw_path(repo_root), default_hourly_path(repo_root), output_path
    )
    print(f"Dynamic features written to {output_path.relative_to(repo_root)}")
    print()
    print("Dynamic feature summary")
    print("-" * 40)
    print(f"Raw rows:                   {summary.raw_row_count:,}")
    print(f"Hourly rows:                {summary.hourly_row_count:,}")
    print(f"Hours with full history:    {summary.hours_with_context:,}")
    print(f"Hours without history:      {summary.hours_without_context:,}")
    print(f"Dynamic features:           {summary.n_dynamic_features}")


if __name__ == "__main__":
    sys.exit(main())
