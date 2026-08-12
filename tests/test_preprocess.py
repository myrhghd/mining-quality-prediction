from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (
    ALL_PREDICTOR_COLUMNS,
    CORE_SENSOR_PREDICTOR_COLUMNS,
    EXCLUDED_OUTCOME_COLUMN,
    FEED_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    HIGH_FREQUENCY_COLUMNS,
    NON_PREDICTOR_COLUMNS,
    RAW_TO_STANDARD,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    add_normalized_position,
    assign_temporal_segments,
    build_hourly_dataset,
    compute_feed_context,
    compute_high_frequency_aggregates,
    compute_sensor_freeze,
    compute_target_run_metadata,
    detect_interpolated_hours,
    get_predictor_columns,
    run,
    standardize_columns,
    validate_raw_columns,
)

REAL_RAW_PATH = REPO_ROOT / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


def _make_raw_like_row(timestamp, target, **overrides) -> dict:
    """Build one standardized 'raw' row with sane defaults for every
    high frequency column, target, and feed column."""
    row = {TIMESTAMP_COLUMN: timestamp, TARGET_COLUMN: target, EXCLUDED_OUTCOME_COLUMN: 65.0}
    for col in HIGH_FREQUENCY_COLUMNS:
        row[col] = 100.0
    for col in FEED_COLUMNS:
        row[col] = 50.0
    row.update(overrides)
    return row


# ---------------------------------------------------------------------
# 1. Standardized schema
# ---------------------------------------------------------------------


def test_standardize_columns_maps_expected_names():
    raw = pd.DataFrame({col: [1.0] for col in RAW_TO_STANDARD})
    standardized = standardize_columns(raw)
    assert set(standardized.columns) == set(RAW_TO_STANDARD.values())
    assert TARGET_COLUMN in standardized.columns
    assert EXCLUDED_OUTCOME_COLUMN in standardized.columns
    assert len(HIGH_FREQUENCY_COLUMNS) == 19


def test_validate_raw_columns_fails_clearly_when_missing():
    raw = pd.DataFrame({"date": [1], "% Iron Feed": [1]})
    with pytest.raises(ValueError, match="missing"):
        validate_raw_columns(raw)


# ---------------------------------------------------------------------
# 2. Normalized within hour position
# ---------------------------------------------------------------------


def test_normalized_position_maps_zero_to_one_regardless_of_group_size():
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["h1", "h1", "h1", "h2"],
            "value": [1, 2, 3, 4],
        }
    )
    result = add_normalized_position(df, TIMESTAMP_COLUMN)
    h1 = result.loc[result[TIMESTAMP_COLUMN] == "h1", "_normalized_position"].tolist()
    h2 = result.loc[result[TIMESTAMP_COLUMN] == "h2", "_normalized_position"].tolist()
    assert h1 == pytest.approx([0.0, 0.5, 1.0])
    # a group of size 1 must not divide by zero
    assert h2 == pytest.approx([0.0])


# ---------------------------------------------------------------------
# 3. Normalized slope calculation
# ---------------------------------------------------------------------


def test_slope_matches_known_linear_relationship():
    # normalized positions 0, 0.5, 1.0 with values 10, 15, 20 -> slope 10
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["h1", "h1", "h1"],
            "sensor": [10.0, 15.0, 20.0],
        }
    )
    result = compute_high_frequency_aggregates(df, ["sensor"])
    assert result.loc["h1", "sensor_slope"] == pytest.approx(10.0, abs=1e-9)
    assert result.loc["h1", "sensor_mean"] == pytest.approx(15.0)


def test_slope_is_comparable_across_different_group_sizes():
    # same underlying linear relationship (slope 10 per unit of normalized
    # position) expressed with 3 rows and with 5 rows
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["h_small"] * 3 + ["h_large"] * 5,
            "sensor": [10.0, 15.0, 20.0, 10.0, 12.5, 15.0, 17.5, 20.0],
        }
    )
    result = compute_high_frequency_aggregates(df, ["sensor"])
    assert result.loc["h_small", "sensor_slope"] == pytest.approx(10.0, abs=1e-9)
    assert result.loc["h_large", "sensor_slope"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------
# 4. Interpolation detection on known synthetic examples
# ---------------------------------------------------------------------


def test_interpolation_detection_on_synthetic_hours():
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["constant"] * 3 + ["ramp"] * 4 + ["noisy"] * 4,
            TARGET_COLUMN: [
                2.5, 2.5, 2.5,               # constant hour: no interpolation
                2.50, 2.60, 2.70, 2.80,      # perfect linear ramp
                2.50, 2.90, 2.40, 2.60,      # varies, but not on a line
            ],
        }
    )
    result = detect_interpolated_hours(df)
    assert result.loc["constant", "is_interpolated"] == False  # noqa: E712
    assert result.loc["ramp", "is_interpolated"] == True  # noqa: E712
    assert result.loc["noisy", "is_interpolated"] == False  # noqa: E712


# ---------------------------------------------------------------------
# 5. Sensor freeze detection
# ---------------------------------------------------------------------


def test_sensor_freeze_detection_counts_and_flags():
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["frozen"] * 3 + ["varying"] * 3 + ["partial"] * 3,
            "a": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            "b": [2.0, 2.0, 2.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0],
            "c": [3.0, 3.0, 3.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0],
        }
    )
    result = compute_sensor_freeze(df, columns=["a", "b", "c"], majority_threshold=1)
    assert result.loc["frozen", "n_frozen_sensors"] == 3
    assert result.loc["frozen", "is_sensor_valid"] == False  # noqa: E712 (3 > majority_threshold=1)
    assert result.loc["varying", "n_frozen_sensors"] == 0
    assert result.loc["varying", "is_sensor_valid"] == True  # noqa: E712
    assert result.loc["partial", "n_frozen_sensors"] == 1
    assert result.loc["partial", "is_sensor_valid"] == True  # noqa: E712 (1 <= threshold 1)


# ---------------------------------------------------------------------
# 6. Temporal segment creation
# ---------------------------------------------------------------------


def test_temporal_segments_split_on_gap_not_hardcoded_date():
    timestamps = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
            "2020-01-01 02:00:00",
            "2020-02-10 09:00:00",  # large, arbitrary gap
            "2020-02-10 10:00:00",
        ]
    )
    hourly = pd.DataFrame({TIMESTAMP_COLUMN: timestamps})
    segments = assign_temporal_segments(hourly)
    assert segments.tolist() == [0, 0, 0, 1, 1]


def test_temporal_segments_handle_multiple_gaps():
    timestamps = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
            "2020-03-01 00:00:00",
            "2020-05-01 00:00:00",
            "2020-05-01 01:00:00",
        ]
    )
    hourly = pd.DataFrame({TIMESTAMP_COLUMN: timestamps})
    segments = assign_temporal_segments(hourly)
    assert segments.nunique() == 3


# ---------------------------------------------------------------------
# 7. Feed variable consistency
# ---------------------------------------------------------------------


def test_feed_context_flags_within_hour_inconsistency():
    df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["consistent"] * 3 + ["inconsistent"] * 3,
            "iron_feed": [50.0, 50.0, 50.0, 50.0, 52.0, 50.0],
        }
    )
    result = compute_feed_context(df, columns=["iron_feed"])
    assert result.loc["consistent", "iron_feed"] == 50.0
    assert result.loc["consistent", "iron_feed_inconsistent"] == False  # noqa: E712
    assert result.loc["inconsistent", "iron_feed"] == 50.0  # deterministic: first value
    assert result.loc["inconsistent", "iron_feed_inconsistent"] == True  # noqa: E712


# ---------------------------------------------------------------------
# 8. Target run metadata
# ---------------------------------------------------------------------


def test_target_run_metadata_tracks_holding_blocks():
    hourly = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range("2020-01-01", periods=6, freq="h"),
            TARGET_COLUMN: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
        }
    )
    result = compute_target_run_metadata(hourly)
    assert result["target_run_length"].tolist() == [2, 2, 3, 3, 3, 1]
    assert result["hours_since_target_change"].tolist() == [0, 1, 0, 1, 2, 0]
    assert result["target_run_id"].nunique() == 3


def test_target_run_metadata_resets_at_temporal_segment_boundary():
    hourly = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-02-01 00:00:00",
                    "2020-02-01 01:00:00",
                ]
            ),
            TARGET_COLUMN: [2.0, 2.0, 2.0, 2.0],
            "temporal_segment_id": [0, 0, 1, 1],
        }
    )
    result = compute_target_run_metadata(hourly)
    assert result["target_run_length"].tolist() == [2, 2, 2, 2]
    assert result["hours_since_target_change"].tolist() == [0, 1, 0, 1]
    assert result["target_run_id"].nunique() == 2


# ---------------------------------------------------------------------
# 9. Predictor schema excludes target and target derived metadata
# ---------------------------------------------------------------------


def test_predictor_columns_exclude_target_and_metadata():
    core, feed, all_predictors = get_predictor_columns()
    assert len(core) == 57
    assert len(feed) == 2
    assert len(all_predictors) == 59

    assert TARGET_COLUMN not in all_predictors
    assert EXCLUDED_OUTCOME_COLUMN not in all_predictors
    assert "is_interpolated" not in all_predictors
    assert "target_run_id" not in all_predictors
    assert "target_run_length" not in all_predictors
    assert "hours_since_target_change" not in all_predictors
    assert "temporal_segment_id" not in all_predictors
    assert "n_frozen_sensors" not in all_predictors
    assert "is_sensor_valid" not in all_predictors
    assert "is_sensor_model_eligible" not in all_predictors
    assert "is_feed_model_eligible" not in all_predictors
    assert "iron_feed_inconsistent" not in all_predictors
    assert "silica_feed_inconsistent" not in all_predictors

    assert set(all_predictors).isdisjoint(NON_PREDICTOR_COLUMNS)
    assert {"iron_feed_inconsistent", "silica_feed_inconsistent"}.issubset(
        NON_PREDICTOR_COLUMNS
    )
    assert set(CORE_SENSOR_PREDICTOR_COLUMNS) == {
        f"{c}_{stat}" for c in HIGH_FREQUENCY_COLUMNS for stat in ("mean", "std", "slope")
    }
    assert FEED_CONTEXT_PREDICTOR_COLUMNS == FEED_COLUMNS
    assert set(ALL_PREDICTOR_COLUMNS) == set(all_predictors)


# ---------------------------------------------------------------------
# 10. Output eligibility logic (end to end on a small synthetic dataset)
# ---------------------------------------------------------------------


def test_build_hourly_dataset_eligibility_end_to_end():
    rows = []

    # Hour 1: clean, eligible for both sensor-only and feed-enhanced models.
    varying_row_a = {col: 100.0 for col in HIGH_FREQUENCY_COLUMNS}
    varying_row_b = {col: 110.0 for col in HIGH_FREQUENCY_COLUMNS}
    rows.append(_make_raw_like_row("2020-01-01 00:00:00", target=2.0, **varying_row_a))
    rows.append(_make_raw_like_row("2020-01-01 00:00:00", target=2.0, **varying_row_b))

    # Hour 2: all sensors vary, so interpolation is the only exclusion reason.
    for idx, target in enumerate([2.00, 2.10, 2.20]):
        sensor_values = {col: 100.0 + idx * 10.0 for col in HIGH_FREQUENCY_COLUMNS}
        rows.append(
            _make_raw_like_row(
                "2020-01-01 01:00:00",
                target=target,
                **sensor_values,
            )
        )

    # Hour 3: fully frozen sensors -> excluded from both eligibility variants.
    frozen_kwargs = {col: 5.0 for col in HIGH_FREQUENCY_COLUMNS}
    rows.append(_make_raw_like_row("2020-01-01 02:00:00", target=2.3, **frozen_kwargs))
    rows.append(_make_raw_like_row("2020-01-01 02:00:00", target=2.3, **frozen_kwargs))

    df = pd.DataFrame(rows)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    hourly = build_hourly_dataset(df)
    assert len(hourly) == 3

    by_hour = hourly.set_index(TIMESTAMP_COLUMN)
    h1 = pd.Timestamp("2020-01-01 00:00:00")
    h2 = pd.Timestamp("2020-01-01 01:00:00")
    h3 = pd.Timestamp("2020-01-01 02:00:00")

    assert bool(by_hour.loc[h1, "is_sensor_model_eligible"]) is True
    assert bool(by_hour.loc[h1, "is_feed_model_eligible"]) is True
    assert bool(by_hour.loc[h1, "is_interpolated"]) is False

    assert bool(by_hour.loc[h2, "is_sensor_valid"]) is True
    assert bool(by_hour.loc[h2, "is_interpolated"]) is True
    assert bool(by_hour.loc[h2, "is_sensor_model_eligible"]) is False
    assert bool(by_hour.loc[h2, "is_feed_model_eligible"]) is False

    assert int(by_hour.loc[h3, "n_frozen_sensors"]) == len(HIGH_FREQUENCY_COLUMNS)
    assert bool(by_hour.loc[h3, "is_sensor_valid"]) is False
    assert bool(by_hour.loc[h3, "is_sensor_model_eligible"]) is False
    assert bool(by_hour.loc[h3, "is_feed_model_eligible"]) is False

    _, _, all_predictors = get_predictor_columns()
    assert hourly[all_predictors].isna().sum().sum() == 0
    assert EXCLUDED_OUTCOME_COLUMN not in hourly.columns
    assert EXCLUDED_OUTCOME_COLUMN not in all_predictors


# ---------------------------------------------------------------------
# 11. Lightweight integration test against the real raw file (skips if absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_RAW_PATH.exists(), reason="raw dataset not available locally")
def test_real_dataset_produces_expected_structure(tmp_path):
    output_path = tmp_path / "hourly_features.parquet"
    summary = run(REAL_RAW_PATH, output_path)

    assert summary.raw_row_count == 737_453
    assert summary.hourly_row_count == 4_097
    assert summary.interpolated_hours == 310
    assert summary.sensor_invalid_hours == 4
    assert summary.sensor_model_eligible_hours == 3_783
    assert summary.feed_model_eligible_hours == 3_783
    assert summary.temporal_segments == 2
    assert output_path.exists()

    hourly = pd.read_parquet(output_path)
    assert len(hourly) == 4_097
    _, _, all_predictors = get_predictor_columns()
    for col in all_predictors:
        assert col in hourly.columns
    assert EXCLUDED_OUTCOME_COLUMN not in all_predictors
