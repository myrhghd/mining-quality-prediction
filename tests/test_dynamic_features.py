from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dynamic_features import (
    DYNAMIC_PREDICTOR_COLUMNS,
    HAS_CONTEXT_COLUMN,
    HIGH_FREQUENCY_COLUMNS,
    LONG_WINDOW_MINUTES,
    PRIMARY_PROCESS_COLUMNS,
    SEGMENT_COLUMN,
    SENSOR_VALID_COLUMN,
    SHORT_WINDOW_MINUTES,
    SLOPE_WINDOW_MINUTES,
    add_within_hour_index,
    build_dynamic_features,
    get_dynamic_predictor_columns,
    previous_hour_available,
    trailing_window_mask,
    validate_dynamic_features,
    window_row_count,
)
from src.data.preprocess import TIMESTAMP_COLUMN

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_DYNAMIC = REPO_ROOT / "data" / "processed" / "dynamic_features.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_DYNAMIC.exists()

SAMPLES_PER_HOUR = 180


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_raw(
    n_hours: int = 12,
    samples_per_hour: int = SAMPLES_PER_HOUR,
    seed: int = 0,
    start: str = "2020-01-01 00:00:00",
) -> pd.DataFrame:
    """A raw 20 second record with all 19 high frequency process columns.

    Values are drawn independently per row so that a window boundary
    error changes the resulting statistic; a smooth series would hide
    an off by one window.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=n_hours, freq="h")
    timestamps = np.repeat(hours.to_numpy(), samples_per_hour)

    frame = pd.DataFrame({TIMESTAMP_COLUMN: timestamps})
    for offset, column in enumerate(HIGH_FREQUENCY_COLUMNS):
        frame[column] = rng.normal(loc=100.0 + offset, scale=5.0, size=len(frame))
    return frame


def make_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """The minimal hourly table `build_dynamic_features` reads."""
    grouped = raw.groupby(TIMESTAMP_COLUMN)
    hourly = pd.DataFrame(
        {f"{column}_mean": grouped[column].mean() for column in HIGH_FREQUENCY_COLUMNS}
    )
    hourly[SEGMENT_COLUMN] = 0
    hourly[SENSOR_VALID_COLUMN] = True
    hourly.index.name = TIMESTAMP_COLUMN
    return hourly.reset_index()


def default_fixture(**kwargs):
    raw = make_raw(**kwargs)
    return raw, make_hourly(raw)


def hour_rows(raw: pd.DataFrame, timestamp) -> pd.DataFrame:
    return raw[raw[TIMESTAMP_COLUMN] == timestamp]


def feature_row(features: pd.DataFrame, timestamp) -> pd.Series:
    return features[features[TIMESTAMP_COLUMN] == timestamp].iloc[0]


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------


def test_schema_declares_ninety_six_unique_features():
    columns = get_dynamic_predictor_columns()
    assert len(columns) == 96
    assert len(set(columns)) == 96


def test_schema_covers_the_declared_variable_scope():
    columns = set(get_dynamic_predictor_columns())
    for variable in HIGH_FREQUENCY_COLUMNS:
        for suffix in (
            "trailing_15m_mean",
            "trailing_120m_mean",
            "trailing_15m_minus_120m_mean",
            "change_1h",
        ):
            assert f"{variable}_{suffix}" in columns
    for variable in PRIMARY_PROCESS_COLUMNS:
        for suffix in (
            "trailing_30m_slope",
            "trailing_120m_std",
            "trailing_60m_min",
            "trailing_60m_max",
        ):
            assert f"{variable}_{suffix}" in columns


def test_schema_excludes_feed_chemistry_the_outcome_and_calendar_values():
    columns = set(get_dynamic_predictor_columns())
    forbidden = {
        "iron_feed",
        "silica_feed",
        "iron_concentrate",
        "silica_concentrate",
        "date",
        "month",
        "day_of_week",
        "hour_of_day",
    }
    assert not columns.intersection(forbidden)
    for column in columns:
        assert not any(
            token in column for token in ("iron", "silica_feed", "month", "weekday", "hour_of_day")
        )


def test_built_table_matches_the_declared_schema():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)
    assert list(features.columns) == [
        TIMESTAMP_COLUMN,
        *DYNAMIC_PREDICTOR_COLUMNS,
        HAS_CONTEXT_COLUMN,
    ]
    validate_dynamic_features(features, hourly)


# ---------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------


def test_window_row_count_rounds_up_and_never_exceeds_the_hour():
    sizes = pd.Series([180, 174, 179, 3])
    assert window_row_count(sizes, 15).tolist() == [45, 44, 45, 1]
    assert window_row_count(sizes, 30).tolist() == [90, 87, 90, 2]
    assert window_row_count(sizes, 60).tolist() == [180, 174, 179, 3]


def test_window_row_count_rejects_a_window_longer_than_an_hour():
    with pytest.raises(ValueError, match="between 1 and 60"):
        window_row_count(pd.Series([180]), 120)


def test_trailing_window_mask_selects_only_the_end_of_each_hour():
    raw, _ = default_fixture(n_hours=3)
    indexed = add_within_hour_index(raw)
    mask = trailing_window_mask(indexed, SHORT_WINDOW_MINUTES)

    selected = indexed[mask]
    assert len(selected) == 3 * 45
    for _, group in selected.groupby(TIMESTAMP_COLUMN):
        positions = group["_within_hour_position"].to_numpy()
        assert positions.min() == SAMPLES_PER_HOUR - 45
        assert positions.max() == SAMPLES_PER_HOUR - 1


def test_trailing_window_of_an_hour_covers_every_row():
    raw, _ = default_fixture(n_hours=2)
    indexed = add_within_hour_index(raw)
    assert bool(trailing_window_mask(indexed, 60).all())


def test_window_length_adapts_to_a_short_hour():
    raw, _ = default_fixture(n_hours=3)
    # Drop rows so the middle hour holds 120 observations instead of 180.
    hours = raw[TIMESTAMP_COLUMN].unique()
    short_hour = hours[1]
    keep = ~(raw[TIMESTAMP_COLUMN] == short_hour) | (
        raw.groupby(TIMESTAMP_COLUMN).cumcount() < 120
    )
    trimmed = raw[keep].reset_index(drop=True)

    indexed = add_within_hour_index(trimmed)
    selected = indexed[trailing_window_mask(indexed, SHORT_WINDOW_MINUTES)]
    counts = selected.groupby(TIMESTAMP_COLUMN).size()
    assert int(counts.loc[short_hour]) == 30
    assert int(counts.loc[hours[0]]) == 45


# ---------------------------------------------------------------------
# Feature values
# ---------------------------------------------------------------------


def test_trailing_15m_mean_matches_a_direct_recomputation():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)

    for timestamp in hourly[TIMESTAMP_COLUMN]:
        rows = hour_rows(raw, timestamp)
        tail = rows.iloc[len(rows) - 45 :]
        row = feature_row(features, timestamp)
        for column in ("starch_flow", "flotation_column_04_level"):
            assert row[f"{column}_trailing_15m_mean"] == pytest.approx(
                tail[column].mean(), abs=1e-12
            )


def test_trailing_60m_extremes_match_the_whole_hour():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)

    for timestamp in hourly[TIMESTAMP_COLUMN]:
        rows = hour_rows(raw, timestamp)
        row = feature_row(features, timestamp)
        for column in PRIMARY_PROCESS_COLUMNS:
            assert row[f"{column}_trailing_60m_min"] == pytest.approx(rows[column].min())
            assert row[f"{column}_trailing_60m_max"] == pytest.approx(rows[column].max())


def test_trailing_120m_statistics_match_pooling_two_hours():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)
    hours = list(hourly[TIMESTAMP_COLUMN])

    for timestamp in hours[1:]:
        previous = timestamp - pd.Timedelta(1, unit="h")
        pooled = pd.concat([hour_rows(raw, previous), hour_rows(raw, timestamp)])
        assert len(pooled) == 2 * SAMPLES_PER_HOUR
        row = feature_row(features, timestamp)
        for column in PRIMARY_PROCESS_COLUMNS:
            assert row[f"{column}_trailing_120m_mean"] == pytest.approx(
                pooled[column].mean(), abs=1e-10
            )
            assert row[f"{column}_trailing_120m_std"] == pytest.approx(
                pooled[column].std(ddof=1), abs=1e-8
            )


def test_change_1h_matches_the_difference_of_the_hourly_means():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)
    indexed = hourly.set_index(TIMESTAMP_COLUMN)
    hours = list(hourly[TIMESTAMP_COLUMN])

    for timestamp in hours[1:]:
        previous = timestamp - pd.Timedelta(1, unit="h")
        row = feature_row(features, timestamp)
        for column in HIGH_FREQUENCY_COLUMNS:
            expected = (
                indexed.loc[timestamp, f"{column}_mean"]
                - indexed.loc[previous, f"{column}_mean"]
            )
            assert row[f"{column}_change_1h"] == pytest.approx(expected, abs=1e-10)


def test_short_minus_long_is_exactly_the_difference_of_its_operands():
    raw, hourly = default_fixture()
    features = build_dynamic_features(raw, hourly)
    usable = features[features[HAS_CONTEXT_COLUMN]]

    for column in HIGH_FREQUENCY_COLUMNS:
        expected = (
            usable[f"{column}_trailing_15m_mean"] - usable[f"{column}_trailing_120m_mean"]
        )
        np.testing.assert_allclose(
            usable[f"{column}_trailing_15m_minus_120m_mean"].to_numpy(),
            expected.to_numpy(),
            atol=1e-12,
        )


def test_trailing_30m_slope_recovers_a_known_ramp_in_units_per_minute():
    raw, _ = default_fixture(n_hours=4)
    minutes_per_row = 60.0 / SAMPLES_PER_HOUR
    position = raw.groupby(TIMESTAMP_COLUMN).cumcount().to_numpy()
    rate = 0.25  # units per minute
    raw = raw.copy()
    raw["ore_pulp_ph"] = 7.0 + rate * position * minutes_per_row

    hourly = make_hourly(raw)
    features = build_dynamic_features(raw, hourly)
    slopes = features["ore_pulp_ph_trailing_30m_slope"].to_numpy()
    np.testing.assert_allclose(slopes, rate, atol=1e-9)


def test_trailing_30m_slope_is_zero_for_a_flat_variable():
    raw, _ = default_fixture(n_hours=3)
    raw = raw.copy()
    raw["amina_flow"] = 500.0
    hourly = make_hourly(raw)
    features = build_dynamic_features(raw, hourly)
    np.testing.assert_allclose(
        features["amina_flow_trailing_30m_slope"].to_numpy(), 0.0, atol=1e-12
    )


def test_the_slope_window_covers_half_an_hour_not_the_whole_hour():
    """A ramp confined to the final half hour must register on the slope."""
    raw, _ = default_fixture(n_hours=3)
    raw = raw.copy()
    position = raw.groupby(TIMESTAMP_COLUMN).cumcount().to_numpy()
    minutes_per_row = 60.0 / SAMPLES_PER_HOUR
    first_kept = SAMPLES_PER_HOUR - int(np.ceil(SAMPLES_PER_HOUR * SLOPE_WINDOW_MINUTES / 60))
    inside_window = position >= first_kept
    raw["starch_flow"] = np.where(
        inside_window, 3000.0 + (position - first_kept) * minutes_per_row, 3000.0
    )

    hourly = make_hourly(raw)
    features = build_dynamic_features(raw, hourly)
    np.testing.assert_allclose(
        features["starch_flow_trailing_30m_slope"].to_numpy(), 1.0, atol=1e-9
    )


# ---------------------------------------------------------------------
# Backward looking guarantees
# ---------------------------------------------------------------------


def test_no_feature_changes_when_later_hours_are_removed():
    raw, hourly = default_fixture(n_hours=10)
    full = build_dynamic_features(raw, hourly)

    cutoff = hourly[TIMESTAMP_COLUMN].iloc[5]
    truncated = build_dynamic_features(
        raw[raw[TIMESTAMP_COLUMN] <= cutoff], hourly[hourly[TIMESTAMP_COLUMN] <= cutoff]
    )
    reference = full[full[TIMESTAMP_COLUMN] <= cutoff].reset_index(drop=True)

    assert truncated[TIMESTAMP_COLUMN].equals(reference[TIMESTAMP_COLUMN])
    np.testing.assert_array_equal(
        truncated[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
        reference[DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
    )


def test_no_future_observation_enters_a_feature():
    """Rewriting every later observation must leave earlier hours untouched."""
    raw, hourly = default_fixture(n_hours=8)
    original = build_dynamic_features(raw, hourly)

    boundary = hourly[TIMESTAMP_COLUMN].iloc[4]
    perturbed_raw = raw.copy()
    later = perturbed_raw[TIMESTAMP_COLUMN] > boundary
    for column in HIGH_FREQUENCY_COLUMNS:
        perturbed_raw.loc[later, column] = perturbed_raw.loc[later, column] * 3.0 + 17.0
    perturbed_hourly = make_hourly(perturbed_raw)

    perturbed = build_dynamic_features(perturbed_raw, perturbed_hourly)

    unchanged = original[TIMESTAMP_COLUMN] <= boundary
    np.testing.assert_array_equal(
        perturbed.loc[unchanged.to_numpy(), DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
        original.loc[unchanged.to_numpy(), DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
    )
    # The perturbation must actually have moved the later hours, or the
    # comparison above would prove nothing.
    moved = ~unchanged
    assert not np.array_equal(
        perturbed.loc[moved.to_numpy(), DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
        original.loc[moved.to_numpy(), DYNAMIC_PREDICTOR_COLUMNS].to_numpy(dtype=float),
    )


def test_a_within_hour_feature_ignores_the_start_of_the_hour():
    """The 15 minute mean must not move when earlier parts of the hour change."""
    raw, _ = default_fixture(n_hours=3)
    perturbed = raw.copy()
    position = perturbed.groupby(TIMESTAMP_COLUMN).cumcount().to_numpy()
    early = position < SAMPLES_PER_HOUR - 45
    perturbed.loc[early, "ore_pulp_flow"] = -999.0

    baseline = build_dynamic_features(raw, make_hourly(raw))
    changed = build_dynamic_features(perturbed, make_hourly(perturbed))

    np.testing.assert_allclose(
        changed["ore_pulp_flow_trailing_15m_mean"].to_numpy(),
        baseline["ore_pulp_flow_trailing_15m_mean"].to_numpy(),
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Context availability
# ---------------------------------------------------------------------


def test_the_first_hour_has_no_history():
    raw, hourly = default_fixture(n_hours=5)
    features = build_dynamic_features(raw, hourly)
    first = features.iloc[0]
    assert not bool(first[HAS_CONTEXT_COLUMN])
    assert pd.isna(first["starch_flow_trailing_120m_mean"])
    assert pd.isna(first["starch_flow_change_1h"])
    assert bool(features.iloc[1:][HAS_CONTEXT_COLUMN].all())


def test_an_hour_after_a_frozen_sensor_hour_has_no_history():
    raw, hourly = default_fixture(n_hours=6)
    frozen_hour = hourly[TIMESTAMP_COLUMN].iloc[2]
    hourly = hourly.copy()
    hourly.loc[hourly[TIMESTAMP_COLUMN] == frozen_hour, SENSOR_VALID_COLUMN] = False

    features = build_dynamic_features(raw, hourly)
    context = features.set_index(TIMESTAMP_COLUMN)[HAS_CONTEXT_COLUMN]

    assert not bool(context.loc[frozen_hour + pd.Timedelta(1, unit="h")])
    # The frozen hour itself still has history: its own predecessor is fine.
    assert bool(context.loc[frozen_hour])


def test_an_hour_after_a_segment_boundary_has_no_history():
    raw, hourly = default_fixture(n_hours=6)
    hourly = hourly.copy()
    boundary = hourly[TIMESTAMP_COLUMN].iloc[3]
    hourly.loc[hourly[TIMESTAMP_COLUMN] >= boundary, SEGMENT_COLUMN] = 1

    features = build_dynamic_features(raw, hourly)
    context = features.set_index(TIMESTAMP_COLUMN)[HAS_CONTEXT_COLUMN]
    assert not bool(context.loc[boundary])
    assert bool(context.loc[boundary + pd.Timedelta(1, unit="h")])


def test_a_recording_gap_stops_history_from_bridging_it():
    raw, hourly = default_fixture(n_hours=6)
    missing = hourly[TIMESTAMP_COLUMN].iloc[2]
    raw = raw[raw[TIMESTAMP_COLUMN] != missing].reset_index(drop=True)
    hourly = hourly[hourly[TIMESTAMP_COLUMN] != missing].reset_index(drop=True)

    features = build_dynamic_features(raw, hourly)
    context = features.set_index(TIMESTAMP_COLUMN)[HAS_CONTEXT_COLUMN]
    assert not bool(context.loc[missing + pd.Timedelta(1, unit="h")])


def test_previous_hour_availability_is_indexed_by_hour():
    _, hourly = default_fixture(n_hours=4)
    available = previous_hour_available(hourly, pd.Index(hourly[TIMESTAMP_COLUMN]))
    assert list(available.index) == list(hourly[TIMESTAMP_COLUMN])
    assert available.tolist() == [False, True, True, True]


def test_cross_hour_features_are_absent_exactly_where_history_is():
    raw, hourly = default_fixture(n_hours=6)
    hourly = hourly.copy()
    hourly.loc[hourly[TIMESTAMP_COLUMN] == hourly[TIMESTAMP_COLUMN].iloc[3], SENSOR_VALID_COLUMN] = False

    features = build_dynamic_features(raw, hourly)
    cross_hour = [
        column
        for column in DYNAMIC_PREDICTOR_COLUMNS
        if column.endswith(
            ("trailing_120m_mean", "trailing_120m_std", "change_1h", "trailing_15m_minus_120m_mean")
        )
    ]
    without = features[~features[HAS_CONTEXT_COLUMN]]
    assert len(without) == 2
    assert without[cross_hour].isna().to_numpy().all()

    with_context = features[features[HAS_CONTEXT_COLUMN]]
    assert with_context[DYNAMIC_PREDICTOR_COLUMNS].notna().to_numpy().all()


# ---------------------------------------------------------------------
# Ordering, determinism, validation
# ---------------------------------------------------------------------


def test_rows_are_chronologically_ordered_regardless_of_input_order():
    raw, hourly = default_fixture(n_hours=8)
    shuffled_raw = raw.sample(frac=1.0, random_state=7)
    shuffled_hourly = hourly.sample(frac=1.0, random_state=11)

    features = build_dynamic_features(shuffled_raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort"), shuffled_hourly)
    assert features[TIMESTAMP_COLUMN].is_monotonic_increasing
    assert not features[TIMESTAMP_COLUMN].duplicated().any()


def test_construction_is_deterministic():
    raw, hourly = default_fixture(n_hours=9)
    first = build_dynamic_features(raw, hourly)
    second = build_dynamic_features(raw, hourly)
    assert first.equals(second)


def test_validation_rejects_a_cross_hour_value_without_history():
    raw, hourly = default_fixture(n_hours=5)
    features = build_dynamic_features(raw, hourly)
    features.loc[0, "starch_flow_change_1h"] = 0.0
    with pytest.raises(ValueError, match="without history"):
        validate_dynamic_features(features, hourly)


def test_validation_rejects_a_missing_value_on_a_usable_hour():
    raw, hourly = default_fixture(n_hours=5)
    features = build_dynamic_features(raw, hourly)
    usable = features.index[features[HAS_CONTEXT_COLUMN]][0]
    features.loc[usable, "starch_flow_trailing_15m_mean"] = np.nan
    with pytest.raises(ValueError, match="non finite"):
        validate_dynamic_features(features, hourly)


def test_validation_rejects_a_chronology_that_does_not_match_the_hourly_table():
    raw, hourly = default_fixture(n_hours=5)
    features = build_dynamic_features(raw, hourly)
    with pytest.raises(ValueError, match="differ from the committed hourly chronology"):
        validate_dynamic_features(features.iloc[:-1], hourly)


def test_validation_rejects_rows_out_of_order():
    raw, hourly = default_fixture(n_hours=5)
    features = build_dynamic_features(raw, hourly)
    reversed_rows = features.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="chronological order"):
        validate_dynamic_features(reversed_rows, hourly)


# ---------------------------------------------------------------------
# Real artifacts
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_dynamic_features_satisfy_every_structural_guard():
    hourly = pd.read_parquet(REAL_HOURLY)
    features = pd.read_parquet(REAL_DYNAMIC)
    validate_dynamic_features(features, hourly)

    assert len(features) == len(hourly)
    assert features[TIMESTAMP_COLUMN].is_monotonic_increasing
    # Every hour that is not a segment opening or preceded by a frozen
    # sensor hour should carry a full window.
    assert features[HAS_CONTEXT_COLUMN].mean() > 0.99


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_hours_without_history_are_explained_by_the_data():
    hourly = pd.read_parquet(REAL_HOURLY).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    features = pd.read_parquet(REAL_DYNAMIC)

    without = features.loc[~features[HAS_CONTEXT_COLUMN], TIMESTAMP_COLUMN]
    recorded = set(hourly[TIMESTAMP_COLUMN])
    segment = hourly.set_index(TIMESTAMP_COLUMN)[SEGMENT_COLUMN]
    valid = hourly.set_index(TIMESTAMP_COLUMN)[SENSOR_VALID_COLUMN]

    for timestamp in without:
        previous = timestamp - pd.Timedelta(LONG_WINDOW_MINUTES - 60, unit="m")
        explained = (
            previous not in recorded
            or segment.loc[previous] != segment.loc[timestamp]
            or not bool(valid.loc[previous])
        )
        assert explained, f"{timestamp} has no history for no discoverable reason"
