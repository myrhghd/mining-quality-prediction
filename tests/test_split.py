from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import TIMESTAMP_COLUMN, get_predictor_columns
from src.data.split import (
    DEFAULT_EMBARGO,
    FINAL_TEST_FOLD_ID,
    HOURS_SINCE_CHANGE_COLUMN,
    KIND_FINAL_TEST,
    ROLE_EMBARGO,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    RUN_ID_COLUMN,
    RUN_LENGTH_COLUMN,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
    build_split,
    find_safe_boundary,
    load_hourly,
    run,
    validate_input_columns,
    validate_split,
    validate_split_configuration,
)

REAL_HOURLY_PATH = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"


# ---------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------


def make_hourly(
    n_hours: int = 400,
    start: str = "2020-01-01",
    target_values=None,
    eligible_mask=None,
    feed_eligible_mask=None,
    segment_break_at: int | None = None,
    segment_gap_hours: int = 24 * 10,
) -> pd.DataFrame:
    """Build a small synthetic hourly table with the columns the splitter needs.

    By default every hour has a distinct target (all runs length 1) and is
    eligible, which keeps boundary placement unconstrained unless a test
    deliberately introduces runs, gaps, or ineligible hours.
    """
    timestamps = list(pd.date_range(start, periods=n_hours, freq="h"))
    segments = [0] * n_hours
    if segment_break_at is not None:
        shift = pd.Timedelta(segment_gap_hours, unit="h")
        for i in range(segment_break_at, n_hours):
            timestamps[i] = timestamps[i] + shift
            segments[i] = 1

    if target_values is None:
        target_values = [float(i) for i in range(n_hours)]

    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "silica_concentrate": target_values,
            SEGMENT_COLUMN: segments,
        }
    )

    # Runs restart when the target changes or a new segment begins, matching
    # the preprocessing definition.
    target_changed = frame["silica_concentrate"].ne(frame["silica_concentrate"].shift())
    segment_changed = frame[SEGMENT_COLUMN].ne(frame[SEGMENT_COLUMN].shift())
    run_id = (target_changed | segment_changed).cumsum()
    frame[RUN_ID_COLUMN] = run_id
    frame[RUN_LENGTH_COLUMN] = run_id.groupby(run_id).transform("size")
    frame[HOURS_SINCE_CHANGE_COLUMN] = run_id.groupby(run_id).cumcount()

    frame[SENSOR_ELIGIBLE_COLUMN] = True if eligible_mask is None else eligible_mask
    frame["is_feed_model_eligible"] = (
        frame[SENSOR_ELIGIBLE_COLUMN] if feed_eligible_mask is None else feed_eligible_mask
    )
    return frame


# ---------------------------------------------------------------------
# 1. Chronological train / embargo / validation ordering
# ---------------------------------------------------------------------


def test_fold_ordering_is_chronological():
    hourly = make_hourly()
    assignment, metadata = build_split(hourly)

    assert len(metadata.folds) == 3
    for fold in metadata.folds:
        assert fold.train_end < fold.embargo_start or fold.embargo_count == 0
        assert fold.train_end < fold.validation_start
        assert fold.embargo_end == fold.validation_start
        assert fold.validation_start <= fold.validation_end

        rows = assignment[assignment["fold_id"] == fold.fold_id]
        train_ts = rows.loc[rows["role"] == ROLE_TRAIN, TIMESTAMP_COLUMN]
        val_ts = rows.loc[rows["role"] == ROLE_VALIDATION, TIMESTAMP_COLUMN]
        assert train_ts.max() < val_ts.min()


# ---------------------------------------------------------------------
# 2. Embargo is timestamp based, not row count based
# ---------------------------------------------------------------------


def test_embargo_uses_timestamps_not_row_count():
    # Make hours sparse around the middle by marking a stretch ineligible.
    # A row-count embargo would span a much longer wall-clock period; a
    # timestamp embargo must still cover exactly the configured duration.
    n = 400
    eligible = [True] * n
    for i in range(200, 220):
        eligible[i] = False
    hourly = make_hourly(n_hours=n, eligible_mask=eligible)

    assignment, metadata = build_split(hourly)

    for fold in metadata.folds:
        assert fold.validation_start - fold.embargo_start >= DEFAULT_EMBARGO
        rows = assignment[assignment["fold_id"] == fold.fold_id]
        train_ts = rows.loc[rows["role"] == ROLE_TRAIN, TIMESTAMP_COLUMN]
        # No training hour may sit inside the embargo interval.
        assert not ((train_ts >= fold.embargo_start) & (train_ts < fold.embargo_end)).any()


def test_embargo_duration_is_configurable():
    hourly = make_hourly()
    _, short = build_split(hourly, embargo=pd.Timedelta(24, unit="h"))
    _, long = build_split(hourly, embargo=pd.Timedelta(48, unit="h"))

    assert short.embargo == pd.Timedelta(24, unit="h")
    assert long.embargo == pd.Timedelta(48, unit="h")
    for fold in long.folds:
        assert fold.validation_start - fold.embargo_start >= pd.Timedelta(48, unit="h")
    # A longer embargo withholds at least as much training history.
    assert long.folds[0].train_count <= short.folds[0].train_count


# ---------------------------------------------------------------------
# 3. Safe boundary moves away from the middle of a target run
# ---------------------------------------------------------------------


def test_safe_boundary_avoids_splitting_a_run():
    # Hours 100..119 all share one target value, forming a 20 hour run.
    values = [float(i) for i in range(400)]
    for i in range(100, 120):
        values[i] = 999.0
    hourly = make_hourly(target_values=values)

    mid_run = hourly.iloc[110][TIMESTAMP_COLUMN]
    assert hourly.iloc[110][HOURS_SINCE_CHANGE_COLUMN] != 0  # candidate is mid-run

    boundary = find_safe_boundary(hourly, hourly[TIMESTAMP_COLUMN], mid_run)

    row = hourly.loc[hourly[TIMESTAMP_COLUMN] == boundary].iloc[0]
    assert row[HOURS_SINCE_CHANGE_COLUMN] == 0
    assert boundary != mid_run


def test_safe_boundary_chooses_nearest_run_start_before_singleton_preference():
    # Candidate lies inside a 3 hour run. Its run start is 1 hour earlier,
    # while the nearest singleton run start is 2 hours away. Distance must
    # win so target structure cannot move the split farther than necessary.
    values = [float(i) for i in range(200)]
    for i in range(99, 102):
        values[i] = 999.0
    hourly = make_hourly(n_hours=200, target_values=values)

    candidate = hourly.iloc[100][TIMESTAMP_COLUMN]
    boundary = find_safe_boundary(hourly, hourly[TIMESTAMP_COLUMN], candidate)

    assert boundary == hourly.iloc[99][TIMESTAMP_COLUMN]
    assert hourly.loc[
        hourly[TIMESTAMP_COLUMN] == boundary, HOURS_SINCE_CHANGE_COLUMN
    ].iloc[0] == 0


def test_no_target_run_is_split_across_any_boundary():
    values = [float(i) for i in range(400)]
    # Long runs placed near the default fold boundary fractions.
    for i in range(215, 235):
        values[i] = 777.0
    for i in range(270, 290):
        values[i] = 888.0
    hourly = make_hourly(target_values=values)

    assignment, metadata = build_split(hourly)
    validate_split(assignment, metadata, hourly)  # raises if a run is split

    boundaries = [metadata.final_test_start, metadata.final_test_embargo_start]
    for fold in metadata.folds:
        boundaries.extend([fold.embargo_start, fold.validation_start])

    for boundary in boundaries:
        before = set(hourly.loc[hourly[TIMESTAMP_COLUMN] < boundary, RUN_ID_COLUMN])
        after = set(hourly.loc[hourly[TIMESTAMP_COLUMN] >= boundary, RUN_ID_COLUMN])
        assert not before.intersection(after)


# ---------------------------------------------------------------------
# 4 & 5. Expanding training windows, forward moving validation
# ---------------------------------------------------------------------


def test_training_windows_expand_and_validation_moves_forward():
    hourly = make_hourly()
    _, metadata = build_split(hourly)

    train_counts = [f.train_count for f in metadata.folds]
    assert train_counts == sorted(train_counts)
    assert len(set(train_counts)) == len(train_counts)

    val_starts = [f.validation_start for f in metadata.folds]
    assert val_starts == sorted(val_starts)
    assert len(set(val_starts)) == len(val_starts)

    # Every fold shares the same training origin; only the end expands.
    assert len({f.train_start for f in metadata.folds}) == 1


# ---------------------------------------------------------------------
# 6. Final test is later than every validation fold
# ---------------------------------------------------------------------


def test_final_test_is_chronologically_last():
    hourly = make_hourly()
    assignment, metadata = build_split(hourly)

    for fold in metadata.folds:
        assert fold.validation_end < metadata.final_test_start

    assert metadata.development_end < metadata.final_test_start

    test_ts = assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    dev_ts = assignment.loc[
        assignment["fold_kind"] != KIND_FINAL_TEST, TIMESTAMP_COLUMN
    ]
    assert test_ts.min() > dev_ts.max()
    assert metadata.final_test_count > 0


def test_final_test_assignment_is_distinguishable():
    hourly = make_hourly()
    assignment, _ = build_split(hourly)

    final = assignment[assignment["fold_kind"] == KIND_FINAL_TEST]
    assert set(final["fold_id"].unique()) == {FINAL_TEST_FOLD_ID}
    assert ROLE_TEST in set(final["role"])
    # role 'test' must appear only in the final test assignment
    assert set(assignment.loc[assignment["role"] == ROLE_TEST, "fold_kind"]) == {KIND_FINAL_TEST}


# ---------------------------------------------------------------------
# 7. Temporal segment gaps are respected
# ---------------------------------------------------------------------


def test_temporal_segment_gap_is_respected():
    # Early isolated segment (60 hours) then a large later segment.
    hourly = make_hourly(n_hours=460, segment_break_at=60)

    assignment, metadata = build_split(hourly)
    validate_split(assignment, metadata, hourly)

    hourly_indexed = hourly.set_index(TIMESTAMP_COLUMN)

    # Validation and test windows must live entirely in the later segment.
    for fold in metadata.folds:
        rows = assignment[assignment["fold_id"] == fold.fold_id]
        val_ts = rows.loc[rows["role"] == ROLE_VALIDATION, TIMESTAMP_COLUMN]
        assert hourly_indexed.loc[val_ts, SEGMENT_COLUMN].nunique() == 1
        assert hourly_indexed.loc[val_ts, SEGMENT_COLUMN].iloc[0] == 1

    test_ts = assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    assert hourly_indexed.loc[test_ts, SEGMENT_COLUMN].nunique() == 1

    # The early segment is retained as training history rather than discarded.
    fold1 = assignment[(assignment["fold_id"] == 1) & (assignment["role"] == ROLE_TRAIN)]
    train_segments = hourly_indexed.loc[fold1[TIMESTAMP_COLUMN], SEGMENT_COLUMN]
    assert 0 in set(train_segments)


# ---------------------------------------------------------------------
# 8. Ineligible rows are excluded
# ---------------------------------------------------------------------


def test_ineligible_rows_are_never_assigned():
    n = 400
    eligible = [True] * n
    for i in range(150, 180):
        eligible[i] = False
    hourly = make_hourly(n_hours=n, eligible_mask=eligible)

    assignment, metadata = build_split(hourly)
    validate_split(assignment, metadata, hourly)

    ineligible_ts = set(hourly.loc[~hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    assigned_ts = set(assignment[TIMESTAMP_COLUMN])
    assert not ineligible_ts.intersection(assigned_ts)


# ---------------------------------------------------------------------
# 9. Feed eligibility does not create different temporal boundaries
# ---------------------------------------------------------------------


def test_feed_eligibility_does_not_change_boundaries():
    n = 400
    sensor_eligible = [True] * n

    feed_all_available = [True] * n
    feed_missing = [True] * n
    for i in range(100, 260):
        feed_missing[i] = False

    hourly_full_feed = make_hourly(
        n_hours=n,
        eligible_mask=sensor_eligible,
        feed_eligible_mask=feed_all_available,
    )
    hourly_missing_feed = make_hourly(
        n_hours=n,
        eligible_mask=sensor_eligible,
        feed_eligible_mask=feed_missing,
    )

    _, full_meta = build_split(hourly_full_feed)
    _, missing_meta = build_split(hourly_missing_feed)

    # The only difference between these inputs is feed eligibility. Temporal
    # boundaries must therefore remain identical.
    assert missing_meta.final_test_start == full_meta.final_test_start
    assert [f.validation_start for f in missing_meta.folds] == [
        f.validation_start for f in full_meta.folds
    ]


# ---------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------


def test_split_is_deterministic():
    hourly = make_hourly()

    first_assignment, first_meta = build_split(hourly)
    second_assignment, second_meta = build_split(hourly)

    pd.testing.assert_frame_equal(first_assignment, second_assignment)
    assert first_meta.final_test_start == second_meta.final_test_start
    assert first_meta.development_count == second_meta.development_count
    assert [f.train_count for f in first_meta.folds] == [f.train_count for f in second_meta.folds]

    # Row order of the input must not change the result.
    shuffled = hourly.sample(frac=1.0, random_state=0).reset_index(drop=True)
    third_assignment, third_meta = build_split(shuffled)
    pd.testing.assert_frame_equal(first_assignment, third_assignment)
    assert third_meta.final_test_start == first_meta.final_test_start


# ---------------------------------------------------------------------
# 11. Target metadata never enters predictor definitions
# ---------------------------------------------------------------------


def test_target_metadata_absent_from_predictors():
    _, _, all_predictors = get_predictor_columns()
    for column in (
        RUN_ID_COLUMN,
        RUN_LENGTH_COLUMN,
        HOURS_SINCE_CHANGE_COLUMN,
        SEGMENT_COLUMN,
        SENSOR_ELIGIBLE_COLUMN,
        "silica_concentrate",
    ):
        assert column not in all_predictors


def test_split_assignment_carries_no_predictor_or_target_columns():
    hourly = make_hourly()
    assignment, _ = build_split(hourly)
    assert set(assignment.columns) == {TIMESTAMP_COLUMN, "fold_id", "fold_kind", "role"}


# ---------------------------------------------------------------------
# Split configuration validation
# ---------------------------------------------------------------------


def test_split_configuration_rejects_invalid_design_parameters():
    with pytest.raises(ValueError, match="three"):
        validate_split_configuration(
            DEFAULT_EMBARGO,
            0.15,
            (0.55, 0.70),
            72,
        )

    with pytest.raises(ValueError, match="increasing"):
        validate_split_configuration(
            DEFAULT_EMBARGO,
            0.15,
            (0.70, 0.55, 0.85),
            72,
        )

    with pytest.raises(ValueError, match="Test fraction"):
        validate_split_configuration(
            DEFAULT_EMBARGO,
            1.0,
            (0.55, 0.70, 0.85),
            72,
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_validate_input_columns_reports_missing():
    frame = pd.DataFrame({TIMESTAMP_COLUMN: pd.to_datetime(["2020-01-01"])})
    with pytest.raises(ValueError, match="missing"):
        validate_input_columns(frame)


def test_invariant_rejects_same_timestamp_in_multiple_roles():
    hourly = make_hourly()
    assignment, metadata = build_split(hourly)

    fold = metadata.folds[0]
    bad = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: [fold.validation_start],
            "fold_id": [fold.fold_id],
            "fold_kind": ["development"],
            "role": [ROLE_TRAIN],
        }
    )
    corrupted = pd.concat([assignment, bad], ignore_index=True)

    with pytest.raises(ValueError, match="more than one role"):
        validate_split(corrupted, metadata, hourly)


def test_invariant_failure_is_raised():
    hourly = make_hourly()
    assignment, metadata = build_split(hourly)

    # Inject a training hour that sits inside fold 1's embargo.
    fold = metadata.folds[0]
    bad = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: [fold.embargo_start],
            "fold_id": [fold.fold_id],
            "fold_kind": ["development"],
            "role": [ROLE_TRAIN],
        }
    )
    corrupted = pd.concat([assignment, bad], ignore_index=True).sort_values(
        ["fold_id", "role", TIMESTAMP_COLUMN]
    )
    with pytest.raises(ValueError, match="more than one role|embargo"):
        validate_split(corrupted, metadata, hourly)


# ---------------------------------------------------------------------
# Real data integration test (skips cleanly if the artifact is absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_HOURLY_PATH.exists(), reason="hourly feature dataset not available locally"
)
def test_real_dataset_split(tmp_path):
    hourly = load_hourly(REAL_HOURLY_PATH)
    output_path = tmp_path / "hourly_splits.parquet"
    assignment, metadata = run(REAL_HOURLY_PATH, output_path)

    # Three development folds and exactly one final test period.
    assert len(metadata.folds) == 3
    development_folds = assignment.loc[
        assignment["fold_kind"] == "development", "fold_id"
    ].unique()
    assert sorted(development_folds) == [1, 2, 3]
    assert assignment.loc[assignment["role"] == ROLE_TEST, "fold_id"].nunique() == 1

    # Final test is chronologically last.
    test_ts = assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    val_ts = assignment.loc[assignment["role"] == ROLE_VALIDATION, TIMESTAMP_COLUMN]
    assert test_ts.min() > val_ts.max()

    # Every assigned hour is sensor model eligible.
    eligible_ts = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    assert set(assignment[TIMESTAMP_COLUMN]).issubset(eligible_ts)

    # Embargo hours are withheld from both training and scoring.
    for fold in metadata.folds:
        rows = assignment[assignment["fold_id"] == fold.fold_id]
        embargo_ts = set(rows.loc[rows["role"] == ROLE_EMBARGO, TIMESTAMP_COLUMN])
        scored_ts = set(
            rows.loc[rows["role"].isin([ROLE_TRAIN, ROLE_VALIDATION]), TIMESTAMP_COLUMN]
        )
        assert not embargo_ts.intersection(scored_ts)

    # All invariants hold on the real data.
    validate_split(assignment, metadata, hourly)
    assert output_path.exists()

    # Feed eligible hours are a subset of the assigned population, using the
    # same boundaries rather than a separate chronology.
    feed_ts = set(hourly.loc[hourly["is_feed_model_eligible"], TIMESTAMP_COLUMN])
    assert feed_ts.issubset(eligible_ts)


@pytest.mark.skipif(
    not REAL_HOURLY_PATH.exists(), reason="hourly feature dataset not available locally"
)
def test_real_dataset_split_is_deterministic():
    hourly = load_hourly(REAL_HOURLY_PATH)
    first, first_meta = build_split(hourly)
    second, second_meta = build_split(hourly)
    pd.testing.assert_frame_equal(first, second)
    assert first_meta.final_test_start == second_meta.final_test_start
