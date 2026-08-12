"""Chronological validation split layer for the hourly mining dataset.

Consumes the hourly analytical table produced by `src.data.preprocess`
and produces a deterministic, leakage aware assignment of hours to
three expanding development folds plus one untouched final test period.

Design constraints enforced here:

* Strict chronology. Training always precedes validation, and every
  validation window precedes the final test period. Nothing is shuffled
  and no random state is used.
* Temporal segments are respected. A validation or test window never
  spans a data discontinuity, and the early isolated segment is kept as
  training history only.
* A timestamp based embargo separates training from every validation
  window and from the final test period. The embargo is expressed as a
  duration rather than a row count, because eligible hours are missing
  wherever preprocessing marked an hour interpolated or sensor invalid.
* Target holding runs are never split across a partition boundary. A
  candidate boundary is moved to the nearest safe timestamp, and the
  embargo start is extended backward so that no run has members in both
  training and embargo.

Target run metadata is used only to place boundaries. It never enters a
predictor matrix; the predictor schema is imported unchanged from
`src.data.preprocess` rather than redefined here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.preprocess import (
    TIMESTAMP_COLUMN,
    find_repo_root,
    get_predictor_columns,
)

# ---------------------------------------------------------------------
# Column names used by this module
# ---------------------------------------------------------------------

SENSOR_ELIGIBLE_COLUMN = "is_sensor_model_eligible"
FEED_ELIGIBLE_COLUMN = "is_feed_model_eligible"
SEGMENT_COLUMN = "temporal_segment_id"
RUN_ID_COLUMN = "target_run_id"
RUN_LENGTH_COLUMN = "target_run_length"
HOURS_SINCE_CHANGE_COLUMN = "hours_since_target_change"

REQUIRED_INPUT_COLUMNS = [
    TIMESTAMP_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
    FEED_ELIGIBLE_COLUMN,
    SEGMENT_COLUMN,
    RUN_ID_COLUMN,
    RUN_LENGTH_COLUMN,
    HOURS_SINCE_CHANGE_COLUMN,
]

# ---------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------

# Embargo duration. Configurable so a 48 hour sensitivity variant can be
# evaluated later without redesigning this module.
DEFAULT_EMBARGO = pd.Timedelta(24, unit="h")

# Fraction of the final temporal segment's eligible hours reserved as the
# untouched final test period.
DEFAULT_TEST_FRACTION = 0.15

# Chronological positions, as fractions of the development period, at
# which each fold's validation window begins. Training expands to each
# successive boundary; validation windows are of similar size and move
# forward in time.
DEFAULT_FOLD_START_FRACTIONS = (0.55, 0.70, 0.85)

# How far a candidate boundary may be moved, in hours, while searching
# for a timestamp that does not split a target holding run.
DEFAULT_BOUNDARY_SEARCH_HOURS = 72

ROLE_TRAIN = "train"
ROLE_EMBARGO = "embargo"
ROLE_VALIDATION = "validation"
ROLE_TEST = "test"

KIND_DEVELOPMENT = "development"
KIND_FINAL_TEST = "final_test"

FINAL_TEST_FOLD_ID = 0


# ---------------------------------------------------------------------
# Metadata containers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FoldMetadata:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_count: int
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    embargo_count: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_count: int


@dataclass(frozen=True)
class SplitMetadata:
    embargo: pd.Timedelta
    development_count: int
    development_start: pd.Timestamp
    development_end: pd.Timestamp
    final_test_start: pd.Timestamp
    final_test_end: pd.Timestamp
    final_test_count: int
    final_test_embargo_start: pd.Timestamp
    final_test_embargo_end: pd.Timestamp
    final_test_embargo_count: int
    folds: list[FoldMetadata] = field(default_factory=list)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def validate_input_columns(hourly: pd.DataFrame) -> None:
    """Raise a clear error if the hourly table lacks a required column."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in hourly.columns]
    if missing:
        raise ValueError(
            f"Hourly dataset is missing {len(missing)} required column(s) for splitting: {missing}"
        )


def load_hourly(hourly_path: Path) -> pd.DataFrame:
    """Load the hourly feature table and validate the columns needed here."""
    if not hourly_path.exists():
        raise FileNotFoundError(
            f"Hourly feature dataset not found at: {hourly_path}. "
            "Run the preprocessing module first."
        )
    hourly = pd.read_parquet(hourly_path)
    validate_input_columns(hourly)
    return hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)


def validate_split_configuration(
    embargo: pd.Timedelta,
    test_fraction: float,
    fold_start_fractions: tuple[float, ...],
    search_hours: int,
) -> None:
    """Validate the fixed chronological split design parameters."""
    if embargo <= pd.Timedelta(0):
        raise ValueError("Embargo duration must be positive.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("Test fraction must be strictly between 0 and 1.")
    if len(fold_start_fractions) != 3:
        raise ValueError("Exactly three development fold start fractions are required.")
    if any(not 0.0 < fraction < 1.0 for fraction in fold_start_fractions):
        raise ValueError("Every fold start fraction must be strictly between 0 and 1.")
    if list(fold_start_fractions) != sorted(fold_start_fractions) or len(set(fold_start_fractions)) != 3:
        raise ValueError("Fold start fractions must be unique and strictly increasing.")
    if search_hours < 0:
        raise ValueError("Boundary search hours must be nonnegative.")


# ---------------------------------------------------------------------
# Boundary placement
# ---------------------------------------------------------------------


def is_safe_boundary(hourly: pd.DataFrame, timestamp: pd.Timestamp) -> bool:
    """A boundary is safe when the hour at `timestamp` begins a target run.

    Placing a partition edge at the first hour of a run guarantees that no
    run has members on both sides of that edge.
    """
    row = hourly.loc[hourly[TIMESTAMP_COLUMN] == timestamp]
    if row.empty:
        return False
    return bool(row.iloc[0][HOURS_SINCE_CHANGE_COLUMN] == 0)


def find_safe_boundary(
    hourly: pd.DataFrame,
    candidate_timestamps: pd.Series,
    candidate: pd.Timestamp,
    search_hours: int = DEFAULT_BOUNDARY_SEARCH_HOURS,
) -> pd.Timestamp:
    """Move `candidate` to the nearest timestamp that does not split a run.

    Only timestamps present in `candidate_timestamps` (the eligible
    chronology) are considered, so a boundary always lands on a usable
    hour. The nearest run start is selected. If two safe run starts are
    equally distant, a length 1 run is preferred, then the earlier
    timestamp is used as the deterministic tie break.

    Distance is the primary criterion so target run structure cannot move
    a boundary farther than necessary.
    """
    window = pd.Timedelta(search_hours, unit="h")
    in_range = candidate_timestamps[
        (candidate_timestamps >= candidate - window)
        & (candidate_timestamps <= candidate + window)
    ]
    if in_range.empty:
        raise ValueError(
            f"No candidate timestamps within {search_hours} hours of {candidate}; "
            "cannot place a safe boundary."
        )

    meta = hourly.set_index(TIMESTAMP_COLUMN)
    options = pd.DataFrame({TIMESTAMP_COLUMN: in_range.to_numpy()})
    options["hours_since"] = options[TIMESTAMP_COLUMN].map(meta[HOURS_SINCE_CHANGE_COLUMN])
    options["run_length"] = options[TIMESTAMP_COLUMN].map(meta[RUN_LENGTH_COLUMN])
    options["distance"] = (options[TIMESTAMP_COLUMN] - candidate).abs()

    run_starts = options[options["hours_since"] == 0]
    if run_starts.empty:
        raise ValueError(
            f"No safe (run start) boundary within {search_hours} hours of {candidate}."
        )

    run_starts = run_starts.copy()
    run_starts["singleton_priority"] = (run_starts["run_length"] == 1).astype(int)
    run_starts = run_starts.sort_values(
        ["distance", "singleton_priority", TIMESTAMP_COLUMN],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return pd.Timestamp(run_starts.iloc[0][TIMESTAMP_COLUMN])


def effective_embargo_start(
    hourly: pd.DataFrame, raw_start: pd.Timestamp, boundary: pd.Timestamp
) -> pd.Timestamp:
    """Extend an embargo start backward so no target run straddles it.

    If a run has members both inside the embargo window and before it,
    the embargo is widened to that run's first hour. Those hours are then
    withheld from training rather than being split across the boundary.
    """
    in_window = hourly[
        (hourly[TIMESTAMP_COLUMN] >= raw_start) & (hourly[TIMESTAMP_COLUMN] < boundary)
    ]
    if in_window.empty:
        return raw_start

    touched_runs = in_window[RUN_ID_COLUMN].unique()
    run_first_hour = hourly.loc[
        hourly[RUN_ID_COLUMN].isin(touched_runs), TIMESTAMP_COLUMN
    ].min()
    return min(raw_start, pd.Timestamp(run_first_hour))


# ---------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------


def build_split(
    hourly: pd.DataFrame,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    fold_start_fractions: tuple[float, ...] = DEFAULT_FOLD_START_FRACTIONS,
    search_hours: int = DEFAULT_BOUNDARY_SEARCH_HOURS,
) -> tuple[pd.DataFrame, SplitMetadata]:
    """Build the chronological split assignment table and its metadata.

    Returns a long assignment table with one row per (fold, role,
    timestamp) and a `SplitMetadata` describing every boundary.

    The assignment covers sensor model eligible hours only. Feed enhanced
    models reuse these same boundaries, restricted further to hours where
    the feed eligibility flag is also set; no separate chronology is
    produced for them.
    """
    validate_input_columns(hourly)
    validate_split_configuration(embargo, test_fraction, fold_start_fractions, search_hours)
    hourly = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    eligible = hourly[hourly[SENSOR_ELIGIBLE_COLUMN]].copy()
    if eligible.empty:
        raise ValueError("No sensor model eligible hours available to split.")

    eligible_timestamps = eligible[TIMESTAMP_COLUMN]

    # The final test period is drawn from the latest temporal segment, so
    # it is never separated from its training history by a data gap.
    final_segment = eligible[SEGMENT_COLUMN].max()
    final_segment_rows = eligible[eligible[SEGMENT_COLUMN] == final_segment]
    if len(final_segment_rows) < 2:
        raise ValueError(
            f"Final temporal segment {final_segment} has too few eligible hours to "
            "reserve a test period."
        )

    # 1. Approximate chronological boundary, then move it somewhere safe.
    test_index = int(len(final_segment_rows) * (1.0 - test_fraction))
    test_index = min(max(test_index, 1), len(final_segment_rows) - 1)
    test_candidate = pd.Timestamp(final_segment_rows.iloc[test_index][TIMESTAMP_COLUMN])
    final_test_start = find_safe_boundary(
        hourly, final_segment_rows[TIMESTAMP_COLUMN], test_candidate, search_hours
    )

    # 2. Embargo immediately before the test period, widened if it would
    #    otherwise cut through a target run.
    test_embargo_raw = final_test_start - embargo
    test_embargo_start = effective_embargo_start(hourly, test_embargo_raw, final_test_start)

    development = eligible[eligible[TIMESTAMP_COLUMN] < test_embargo_start].copy()
    test_embargo_rows = eligible[
        (eligible[TIMESTAMP_COLUMN] >= test_embargo_start)
        & (eligible[TIMESTAMP_COLUMN] < final_test_start)
    ]
    final_test_rows = eligible[eligible[TIMESTAMP_COLUMN] >= final_test_start]

    if development.empty:
        raise ValueError("Final test boundary leaves no development observations.")
    if final_test_rows.empty:
        raise ValueError("Final test boundary leaves no test observations.")

    # 3. Development fold boundaries, each moved to a safe timestamp.
    development_timestamps = development[TIMESTAMP_COLUMN]
    validation_starts: list[pd.Timestamp] = []
    for fraction in fold_start_fractions:
        index = int(len(development) * fraction)
        index = min(max(index, 1), len(development) - 1)
        candidate = pd.Timestamp(development.iloc[index][TIMESTAMP_COLUMN])
        validation_starts.append(
            find_safe_boundary(hourly, development_timestamps, candidate, search_hours)
        )

    if len(set(validation_starts)) != len(validation_starts):
        raise ValueError(
            "Fold boundaries collapsed onto the same timestamp after safe boundary "
            "adjustment; widen the fold spacing or narrow the search radius."
        )
    if validation_starts != sorted(validation_starts):
        raise ValueError("Fold boundaries are not chronologically increasing after adjustment.")

    development_end = pd.Timestamp(development[TIMESTAMP_COLUMN].max())

    assignments: list[pd.DataFrame] = []
    folds: list[FoldMetadata] = []

    for position, validation_start in enumerate(validation_starts):
        fold_id = position + 1
        is_last_fold = position == len(validation_starts) - 1
        validation_stop = None if is_last_fold else validation_starts[position + 1]

        embargo_raw = validation_start - embargo
        embargo_start = effective_embargo_start(hourly, embargo_raw, validation_start)

        train_rows = development[development[TIMESTAMP_COLUMN] < embargo_start]
        embargo_rows = development[
            (development[TIMESTAMP_COLUMN] >= embargo_start)
            & (development[TIMESTAMP_COLUMN] < validation_start)
        ]
        if is_last_fold:
            validation_rows = development[development[TIMESTAMP_COLUMN] >= validation_start]
        else:
            validation_rows = development[
                (development[TIMESTAMP_COLUMN] >= validation_start)
                & (development[TIMESTAMP_COLUMN] < validation_stop)
            ]

        if train_rows.empty:
            raise ValueError(f"Fold {fold_id} has no training observations.")
        if validation_rows.empty:
            raise ValueError(f"Fold {fold_id} has no validation observations.")

        for role, rows in (
            (ROLE_TRAIN, train_rows),
            (ROLE_EMBARGO, embargo_rows),
            (ROLE_VALIDATION, validation_rows),
        ):
            if rows.empty:
                continue
            assignments.append(
                pd.DataFrame(
                    {
                        TIMESTAMP_COLUMN: rows[TIMESTAMP_COLUMN].to_numpy(),
                        "fold_id": fold_id,
                        "fold_kind": KIND_DEVELOPMENT,
                        "role": role,
                    }
                )
            )

        folds.append(
            FoldMetadata(
                fold_id=fold_id,
                train_start=pd.Timestamp(train_rows[TIMESTAMP_COLUMN].min()),
                train_end=pd.Timestamp(train_rows[TIMESTAMP_COLUMN].max()),
                train_count=len(train_rows),
                embargo_start=embargo_start,
                embargo_end=validation_start,
                embargo_count=len(embargo_rows),
                validation_start=validation_start,
                validation_end=pd.Timestamp(validation_rows[TIMESTAMP_COLUMN].max()),
                validation_count=len(validation_rows),
            )
        )

    # 4. Final test assignment. The training side is the whole development
    #    period, so the table fully describes the final evaluation split.
    for role, rows in (
        (ROLE_TRAIN, development),
        (ROLE_EMBARGO, test_embargo_rows),
        (ROLE_TEST, final_test_rows),
    ):
        if rows.empty:
            continue
        assignments.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: rows[TIMESTAMP_COLUMN].to_numpy(),
                    "fold_id": FINAL_TEST_FOLD_ID,
                    "fold_kind": KIND_FINAL_TEST,
                    "role": role,
                }
            )
        )

    assignment = pd.concat(assignments, ignore_index=True)
    assignment = assignment.sort_values(
        ["fold_id", "role", TIMESTAMP_COLUMN], kind="mergesort"
    ).reset_index(drop=True)

    metadata = SplitMetadata(
        embargo=embargo,
        development_count=len(development),
        development_start=pd.Timestamp(development[TIMESTAMP_COLUMN].min()),
        development_end=development_end,
        final_test_start=final_test_start,
        final_test_end=pd.Timestamp(final_test_rows[TIMESTAMP_COLUMN].max()),
        final_test_count=len(final_test_rows),
        final_test_embargo_start=test_embargo_start,
        final_test_embargo_end=final_test_start,
        final_test_embargo_count=len(test_embargo_rows),
        folds=folds,
    )

    return assignment, metadata


# ---------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------


def validate_split(
    assignment: pd.DataFrame, metadata: SplitMetadata, hourly: pd.DataFrame
) -> None:
    """Verify every structural invariant, raising a clear error on failure."""
    hourly_indexed = hourly.set_index(TIMESTAMP_COLUMN)

    # Only sensor model eligible hours may be assigned.
    assigned = assignment[TIMESTAMP_COLUMN].unique()
    eligibility = hourly_indexed.loc[assigned, SENSOR_ELIGIBLE_COLUMN]
    if not bool(eligibility.all()):
        raise ValueError("Assignment contains hours that are not sensor model eligible.")

    # A timestamp may appear in multiple development folds, but it may
    # belong to only one role within a given fold.
    duplicated = assignment.duplicated(subset=["fold_id", TIMESTAMP_COLUMN], keep=False)
    if bool(duplicated.any()):
        raise ValueError(
            "Assignment contains timestamp(s) assigned to more than one role "
            "within the same fold."
        )

    expected_development_fold_ids = {fold.fold_id for fold in metadata.folds}
    actual_development_fold_ids = set(
        assignment.loc[assignment["fold_kind"] == KIND_DEVELOPMENT, "fold_id"]
    )
    if actual_development_fold_ids != expected_development_fold_ids:
        raise ValueError("Development fold ids do not match split metadata.")

    test_role_kinds = set(assignment.loc[assignment["role"] == ROLE_TEST, "fold_kind"])
    if test_role_kinds != {KIND_FINAL_TEST}:
        raise ValueError("Test role must appear only in the final test assignment.")

    # Target run metadata must not leak into the predictor schema.
    _, _, all_predictors = get_predictor_columns()
    forbidden = {RUN_ID_COLUMN, RUN_LENGTH_COLUMN, HOURS_SINCE_CHANGE_COLUMN, SEGMENT_COLUMN}
    leaked = forbidden.intersection(all_predictors)
    if leaked:
        raise ValueError(f"Target run metadata leaked into the predictor schema: {sorted(leaked)}")

    for fold in metadata.folds:
        prefix = f"Fold {fold.fold_id}"

        if not fold.train_end < fold.validation_start:
            raise ValueError(f"{prefix}: training does not precede validation.")
        if not fold.embargo_start <= fold.embargo_end:
            raise ValueError(f"{prefix}: embargo window is inverted.")
        if not fold.validation_start <= fold.validation_end:
            raise ValueError(f"{prefix}: validation window is inverted.")
        if fold.embargo_end != fold.validation_start:
            raise ValueError(f"{prefix}: embargo does not end where validation begins.")
        if fold.validation_start - fold.embargo_start < metadata.embargo:
            raise ValueError(
                f"{prefix}: embargo is shorter than the configured {metadata.embargo}."
            )

        fold_rows = assignment[assignment["fold_id"] == fold.fold_id]
        train_ts = fold_rows.loc[fold_rows["role"] == ROLE_TRAIN, TIMESTAMP_COLUMN]
        embargo_ts = fold_rows.loc[fold_rows["role"] == ROLE_EMBARGO, TIMESTAMP_COLUMN]
        validation_ts = fold_rows.loc[fold_rows["role"] == ROLE_VALIDATION, TIMESTAMP_COLUMN]

        if len(train_ts) != fold.train_count:
            raise ValueError(f"{prefix}: training assignment count does not match metadata.")
        if len(embargo_ts) != fold.embargo_count:
            raise ValueError(f"{prefix}: embargo assignment count does not match metadata.")
        if len(validation_ts) != fold.validation_count:
            raise ValueError(f"{prefix}: validation assignment count does not match metadata.")

        if not bool((train_ts < fold.embargo_start).all()):
            raise ValueError(f"{prefix}: a training hour falls at or after the embargo start.")
        if not bool(
            ((embargo_ts >= fold.embargo_start) & (embargo_ts < fold.embargo_end)).all()
        ):
            raise ValueError(f"{prefix}: an embargo assignment falls outside the embargo interval.")
        if not bool((validation_ts >= fold.validation_start).all()):
            raise ValueError(f"{prefix}: a validation hour occurs before validation start.")
        if not bool((validation_ts <= fold.validation_end).all()):
            raise ValueError(f"{prefix}: a validation hour occurs after validation end.")

        if not bool((train_ts < validation_ts.min()).all()):
            raise ValueError(f"{prefix}: a training hour is not earlier than every validation hour.")

        # Validation must precede the final test period.
        if not bool((validation_ts < metadata.final_test_start).all()):
            raise ValueError(f"{prefix}: validation overlaps the final test period.")

        # A validation window must not span a temporal discontinuity.
        validation_segment_values = hourly_indexed.loc[validation_ts, SEGMENT_COLUMN]
        validation_segments = validation_segment_values.nunique()
        if validation_segments != 1:
            raise ValueError(
                f"{prefix}: validation window spans {validation_segments} temporal segments."
            )
        final_segment = hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], SEGMENT_COLUMN].max()
        if validation_segment_values.iloc[0] != final_segment:
            raise ValueError(f"{prefix}: validation is not in the latest temporal segment.")

        _assert_runs_not_split(
            hourly, fold.embargo_start, f"{prefix} train/embargo boundary"
        )
        _assert_runs_not_split(
            hourly, fold.validation_start, f"{prefix} embargo/validation boundary"
        )

    # Expanding training windows and forward moving validation windows.
    for earlier, later in zip(metadata.folds, metadata.folds[1:]):
        if later.train_count <= earlier.train_count:
            raise ValueError(
                f"Training window did not expand from fold {earlier.fold_id} to {later.fold_id}."
            )
        if later.validation_start <= earlier.validation_start:
            raise ValueError(
                f"Validation window did not move forward from fold {earlier.fold_id} "
                f"to {later.fold_id}."
            )

    # Final test invariants.
    if metadata.development_end >= metadata.final_test_start:
        raise ValueError("Development data is not strictly earlier than the final test period.")
    if metadata.final_test_embargo_end != metadata.final_test_start:
        raise ValueError("Final test embargo does not end where the test period begins.")
    if metadata.final_test_start - metadata.final_test_embargo_start < metadata.embargo:
        raise ValueError("Final test embargo is shorter than the configured duration.")

    final_rows = assignment[
        (assignment["fold_id"] == FINAL_TEST_FOLD_ID)
        & (assignment["fold_kind"] == KIND_FINAL_TEST)
    ]
    final_train_ts = final_rows.loc[final_rows["role"] == ROLE_TRAIN, TIMESTAMP_COLUMN]
    final_embargo_ts = final_rows.loc[final_rows["role"] == ROLE_EMBARGO, TIMESTAMP_COLUMN]
    test_ts = final_rows.loc[final_rows["role"] == ROLE_TEST, TIMESTAMP_COLUMN]

    if len(final_train_ts) != metadata.development_count:
        raise ValueError("Final evaluation training count does not match development metadata.")
    if len(final_embargo_ts) != metadata.final_test_embargo_count:
        raise ValueError("Final test embargo count does not match metadata.")
    if len(test_ts) != metadata.final_test_count:
        raise ValueError("Final test count does not match metadata.")

    if not bool((final_train_ts < metadata.final_test_embargo_start).all()):
        raise ValueError("Final evaluation training includes an hour inside the test embargo.")
    if not bool(
        (
            (final_embargo_ts >= metadata.final_test_embargo_start)
            & (final_embargo_ts < metadata.final_test_embargo_end)
        ).all()
    ):
        raise ValueError("A final test embargo assignment falls outside the embargo interval.")
    if not bool((test_ts >= metadata.final_test_start).all()):
        raise ValueError("A final test assignment occurs before the test start.")

    eligible_ts = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    final_assigned_ts = set(final_rows[TIMESTAMP_COLUMN])
    if final_assigned_ts != eligible_ts:
        raise ValueError("Final evaluation assignment does not cover every eligible hour exactly once.")

    test_segments = hourly_indexed.loc[test_ts, SEGMENT_COLUMN]
    if test_segments.nunique() != 1:
        raise ValueError("Final test period spans multiple temporal segments.")
    final_segment = hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], SEGMENT_COLUMN].max()
    if test_segments.iloc[0] != final_segment:
        raise ValueError("Final test is not in the latest temporal segment.")

    for fold in metadata.folds:
        if not fold.validation_end < metadata.final_test_start:
            raise ValueError(
                f"Fold {fold.fold_id} validation is not earlier than the final test period."
            )

    _assert_runs_not_split(
        hourly, metadata.final_test_embargo_start, "development/final test embargo boundary"
    )
    _assert_runs_not_split(hourly, metadata.final_test_start, "final test boundary")

    # Global chronological ordering within each fold and role.
    for (fold_id, role), group in assignment.groupby(["fold_id", "role"]):
        timestamps = group[TIMESTAMP_COLUMN]
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"Fold {fold_id} role {role} is not chronologically ordered.")


def _assert_runs_not_split(
    hourly: pd.DataFrame, boundary: pd.Timestamp, description: str
) -> None:
    """Fail if any target run has hours on both sides of `boundary`."""
    before = set(hourly.loc[hourly[TIMESTAMP_COLUMN] < boundary, RUN_ID_COLUMN])
    at_or_after = set(hourly.loc[hourly[TIMESTAMP_COLUMN] >= boundary, RUN_ID_COLUMN])
    straddling = before.intersection(at_or_after)
    if straddling:
        raise ValueError(
            f"{description} at {boundary} splits {len(straddling)} target run(s): "
            f"{sorted(straddling)[:5]}"
        )


# ---------------------------------------------------------------------
# Paths and CLI entry point
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_split_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def run(hourly_path: Path, output_path: Path, embargo: pd.Timedelta = DEFAULT_EMBARGO):
    """Read hourly features, build and validate the split, and write it."""
    hourly = load_hourly(hourly_path)
    assignment, metadata = build_split(hourly, embargo=embargo)
    validate_split(assignment, metadata, hourly)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_parquet(output_path, index=False)

    return assignment, metadata


def format_summary(metadata: SplitMetadata) -> str:
    """Render the split summary using calculated values only."""
    embargo_hours = int(metadata.embargo.total_seconds() // 3600)
    lines = [
        "Chronological split summary",
        f"Development observations: {metadata.development_count:,}",
        f"Final test observations:  {metadata.final_test_count:,}",
        f"Embargo:                  {embargo_hours} hours",
    ]
    for fold in metadata.folds:
        lines.extend(
            [
                "",
                f"Fold {fold.fold_id}",
                f"Train:      {fold.train_start} -> {fold.train_end}  ({fold.train_count:,} hours)",
                f"Validation: {fold.validation_start} -> {fold.validation_end}  "
                f"({fold.validation_count:,} hours)",
                f"Boundary:   {fold.validation_start}  "
                f"(embargo [{fold.embargo_start}, {fold.embargo_end}), "
                f"{fold.embargo_count:,} hours withheld)",
            ]
        )
    lines.extend(
        [
            "",
            f"Final test start: {metadata.final_test_start}",
            f"Final test end:   {metadata.final_test_end}",
            f"Final test embargo: [{metadata.final_test_embargo_start}, "
            f"{metadata.final_test_embargo_end}) "
            f"({metadata.final_test_embargo_count:,} hours withheld)",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    hourly_path = default_hourly_path(repo_root)
    output_path = default_split_path(repo_root)

    print("Building chronological split...")
    assignment, metadata = run(hourly_path, output_path)
    print(f"Split assignment written to {output_path.relative_to(repo_root)}")
    print()
    print(format_summary(metadata))


if __name__ == "__main__":
    sys.exit(main())
