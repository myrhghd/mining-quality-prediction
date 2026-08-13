from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (
    CORE_SENSOR_PREDICTOR_COLUMNS,
    FEED_CONTEXT_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.data.split import (
    FINAL_TEST_FOLD_ID,
    KIND_DEVELOPMENT,
    KIND_FINAL_TEST,
    ROLE_EMBARGO,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SEGMENT_COLUMN,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models.baselines import INTERPOLATED_COLUMN, get_fold_frames
from src.models.generalization_diagnostics import (
    HIGH_ERROR_QUANTILE,
    annotate_residual_context,
    assay_gap_context,
    assert_no_final_test_contamination,
    baseline_decomposition,
    classify_relationship_stability,
    describe_sample,
    excess_kurtosis,
    ks_statistic,
    label_high_error,
    operating_regime_profile,
    population_stability_index,
    predictor_diagnostics,
    rank_operating_regime,
    rank_predictor_drift,
    residual_subperiods,
    residual_summary,
    residual_trend,
    skewness,
    standardized_mean_difference,
    target_distributions,
    target_drift,
    variance_ratio,
)
from src.models.random_forest import PREDICTION_COLUMN

REAL_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_features.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "processed" / "hourly_splits.parquet"
REAL_ARTIFACTS = REAL_HOURLY.exists() and REAL_SPLITS.exists()


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def make_hourly(n_rows: int = 200, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Hourly table with all 57 predictors, contiguous hours, one segment."""
    rng = np.random.default_rng(seed)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in predictors})
    frame[TARGET_COLUMN] = 2.0 + 1.5 * frame[predictors[0]] + rng.normal(scale=0.2, size=n_rows)
    frame[TIMESTAMP_COLUMN] = pd.date_range(start, periods=n_rows, freq="h")
    frame[SEGMENT_COLUMN] = 0
    frame[SENSOR_ELIGIBLE_COLUMN] = True
    frame[INTERPOLATED_COLUMN] = False
    for column in FEED_CONTEXT_PREDICTOR_COLUMNS:
        frame[column] = 50.0
    return frame


def make_assignment(hourly, train_idx, embargo_idx, validation_idx, test_idx=(), fold_id=1):
    rows = []
    for role, indices in (
        (ROLE_TRAIN, train_idx),
        (ROLE_EMBARGO, embargo_idx),
        (ROLE_VALIDATION, validation_idx),
    ):
        for i in indices:
            rows.append(
                {
                    TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].iloc[i],
                    "fold_id": fold_id,
                    "fold_kind": KIND_DEVELOPMENT,
                    "role": role,
                }
            )
    for i in test_idx:
        rows.append(
            {
                TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].iloc[i],
                "fold_id": FINAL_TEST_FOLD_ID,
                "fold_kind": KIND_FINAL_TEST,
                "role": ROLE_TEST,
            }
        )
    return pd.DataFrame(rows)


def default_fixture():
    """Train 0..119, embargo 120..129, validation 130..179, test 180..199."""
    hourly = make_hourly()
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 120)),
        embargo_idx=list(range(120, 130)),
        validation_idx=list(range(130, 180)),
        test_idx=list(range(180, 200)),
    )
    return hourly, assignment


def make_residuals(hourly, assignment, fold_ids=(1,), seed: int = 3) -> pd.DataFrame:
    """Residuals shaped like the Random Forest output, without fitting anything."""
    rng = np.random.default_rng(seed)
    frames = []
    for fold_id in fold_ids:
        validation = get_fold_frames(hourly, assignment, fold_id).validation
        observed = validation[TARGET_COLUMN].to_numpy(dtype=float)
        predicted = observed + rng.normal(scale=0.5, size=observed.size)
        frame = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: validation[TIMESTAMP_COLUMN].to_numpy(),
                TARGET_COLUMN: observed,
                PREDICTION_COLUMN: predicted,
                "fold_id": fold_id,
            }
        )
        frame["residual"] = frame[TARGET_COLUMN] - frame[PREDICTION_COLUMN]
        frame["absolute_residual"] = frame["residual"].abs()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------


def test_describe_sample_matches_direct_calculation():
    values = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
    described = describe_sample(values)

    assert described["n"] == 5
    assert described["mean"] == pytest.approx(3.6)
    assert described["median"] == pytest.approx(2.0)
    assert described["std"] == pytest.approx(values.std(ddof=0))
    assert described["min"] == pytest.approx(1.0)
    assert described["max"] == pytest.approx(10.0)
    assert described["q25"] == pytest.approx(np.quantile(values, 0.25))
    assert described["q75"] == pytest.approx(np.quantile(values, 0.75))
    assert described["iqr"] == pytest.approx(
        np.quantile(values, 0.75) - np.quantile(values, 0.25)
    )


def test_describe_sample_rejects_empty_and_non_finite():
    with pytest.raises(ValueError, match="empty sample"):
        describe_sample([])
    with pytest.raises(ValueError, match="non-finite"):
        describe_sample([1.0, np.nan])


def test_skewness_and_kurtosis_on_known_shapes():
    symmetric = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert skewness(symmetric) == pytest.approx(0.0, abs=1e-12)

    right_tailed = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    assert skewness(right_tailed) > 1.0

    constant = np.full(10, 4.0)
    assert skewness(constant) == 0.0
    assert excess_kurtosis(constant) == 0.0

    rng = np.random.default_rng(0)
    normal = rng.normal(size=200_000)
    assert excess_kurtosis(normal) == pytest.approx(0.0, abs=0.1)


def test_standardized_mean_difference_is_a_scale_free_effect_size():
    rng = np.random.default_rng(1)
    train = rng.normal(loc=0.0, scale=2.0, size=500)
    shift = 2.0
    validation = train + shift

    # A pure translation leaves both variances equal, so the pooled spread is
    # the sample's own standard deviation and the measure is the shift in
    # units of it.
    expected = shift / float(train.std(ddof=0))
    assert standardized_mean_difference(train, validation) == pytest.approx(expected, abs=1e-9)
    assert standardized_mean_difference(train, train) == pytest.approx(0.0, abs=1e-12)
    # Sign follows the direction of the shift.
    assert standardized_mean_difference(validation, train) == pytest.approx(-expected, abs=1e-9)


def test_standardized_mean_difference_does_not_grow_with_sample_size():
    """The measure must describe the shift, not the confidence in it. This is
    the property a significance test would not have."""
    rng = np.random.default_rng(2)
    small = standardized_mean_difference(
        rng.normal(size=100), rng.normal(loc=0.5, size=100)
    )
    large = standardized_mean_difference(
        rng.normal(size=100_000), rng.normal(loc=0.5, size=100_000)
    )
    assert abs(large - 0.5) < 0.05
    assert abs(small - large) < 0.25


def test_variance_ratio_and_constant_handling():
    rng = np.random.default_rng(3)
    train = rng.normal(scale=2.0, size=2000)
    assert variance_ratio(train, train) == pytest.approx(1.0)
    assert variance_ratio(train, train * 2.0) == pytest.approx(4.0, rel=1e-9)
    assert np.isnan(variance_ratio(np.ones(10), np.arange(10.0)))


def test_ks_statistic_bounds_and_known_values():
    a = np.arange(100.0)
    assert ks_statistic(a, a) == pytest.approx(0.0)
    # Fully separated samples give the maximum distance.
    assert ks_statistic(np.zeros(50), np.ones(50)) == pytest.approx(1.0)
    shifted = ks_statistic(a, a + 50.0)
    assert 0.0 < shifted < 1.0


def test_population_stability_index_behaviour():
    rng = np.random.default_rng(4)
    train = rng.normal(size=5000)

    assert population_stability_index(train, train) == pytest.approx(0.0, abs=1e-12)
    moved = population_stability_index(train, train + 1.0)
    assert moved > 0.25  # a full standard deviation is a major shift by convention
    assert population_stability_index(train, rng.normal(size=5000)) < 0.1

    # Too few distinct training values to form bins.
    assert np.isnan(population_stability_index(np.ones(50), np.arange(50.0)))


def test_drift_statistics_are_deterministic():
    rng = np.random.default_rng(5)
    train, validation = rng.normal(size=800), rng.normal(loc=0.4, size=400)

    for statistic in (
        standardized_mean_difference,
        variance_ratio,
        ks_statistic,
        population_stability_index,
    ):
        assert statistic(train, validation) == statistic(train, validation)


# ---------------------------------------------------------------------
# Target drift
# ---------------------------------------------------------------------


def test_target_distributions_use_the_committed_fold_rows():
    hourly, assignment = default_fixture()
    distributions = target_distributions(hourly, assignment, fold_ids=(1,))
    frames = get_fold_frames(hourly, assignment, 1)

    by_split = distributions.set_index("split")
    assert int(by_split.loc["train", "n"]) == len(frames.train) == 120
    assert int(by_split.loc["validation", "n"]) == len(frames.validation) == 50
    assert by_split.loc["train", "mean"] == pytest.approx(frames.train[TARGET_COLUMN].mean())
    assert by_split.loc["validation", "mean"] == pytest.approx(
        frames.validation[TARGET_COLUMN].mean()
    )


def test_target_drift_detects_an_injected_level_shift():
    hourly, assignment = default_fixture()
    validation_hours = hourly[TIMESTAMP_COLUMN].iloc[130:180]

    shifted = hourly.copy()
    mask = shifted[TIMESTAMP_COLUMN].isin(validation_hours)
    train_std = float(hourly[TARGET_COLUMN].iloc[0:120].std(ddof=0))
    shifted.loc[mask, TARGET_COLUMN] = shifted.loc[mask, TARGET_COLUMN] + train_std

    before = target_drift(hourly, assignment, fold_ids=(1,)).iloc[0]
    after = target_drift(shifted, assignment, fold_ids=(1,)).iloc[0]

    assert abs(before["standardized_mean_difference"]) < 0.3
    assert after["standardized_mean_difference"] > 0.8
    # The injected shift is recovered as the change in the measured drift.
    assert after["mean_change"] - before["mean_change"] == pytest.approx(train_std, abs=1e-9)
    # A pure level shift leaves the spread alone.
    assert after["variance_ratio"] == pytest.approx(before["variance_ratio"], abs=1e-9)


def test_target_drift_detects_variance_compression():
    hourly, assignment = default_fixture()
    validation_hours = set(hourly[TIMESTAMP_COLUMN].iloc[130:180])

    compressed = hourly.copy()
    mask = compressed[TIMESTAMP_COLUMN].isin(validation_hours)
    centre = compressed.loc[mask, TARGET_COLUMN].mean()
    compressed.loc[mask, TARGET_COLUMN] = (
        centre + (compressed.loc[mask, TARGET_COLUMN] - centre) * 0.5
    )

    drift = target_drift(compressed, assignment, fold_ids=(1,)).iloc[0]
    baseline = target_drift(hourly, assignment, fold_ids=(1,)).iloc[0]
    assert drift["variance_ratio"] == pytest.approx(baseline["variance_ratio"] * 0.25, rel=1e-9)


# ---------------------------------------------------------------------
# Predictor drift and relationship stability
# ---------------------------------------------------------------------


def test_predictor_diagnostics_cover_every_predictor_and_fold():
    hourly, assignment = default_fixture()
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)
    table = predictor_diagnostics(hourly, assignment, predictors, fold_ids=(1,))

    assert len(table) == 57
    assert set(table["predictor"]) == set(predictors)
    assert (table["fold_id"] == 1).all()
    assert table.notna().all().all()


def test_predictor_diagnostics_compare_train_against_validation():
    """The reported means must be the fold's own training and validation
    means, not the whole population's."""
    hourly, assignment = default_fixture()
    predictor = CORE_SENSOR_PREDICTOR_COLUMNS[0]
    table = predictor_diagnostics(hourly, assignment, [predictor], fold_ids=(1,)).iloc[0]
    frames = get_fold_frames(hourly, assignment, 1)

    assert table["train_mean"] == pytest.approx(frames.train[predictor].mean())
    assert table["validation_mean"] == pytest.approx(frames.validation[predictor].mean())
    assert table["train_mean"] != pytest.approx(hourly[predictor].mean())


def test_predictor_diagnostics_recover_an_injected_correlation_reversal():
    hourly, assignment = default_fixture()
    predictor = CORE_SENSOR_PREDICTOR_COLUMNS[0]

    reversed_frame = hourly.copy()
    validation_mask = reversed_frame[TIMESTAMP_COLUMN].isin(
        hourly[TIMESTAMP_COLUMN].iloc[130:180]
    )
    reversed_frame.loc[validation_mask, predictor] = -reversed_frame.loc[
        validation_mask, predictor
    ]

    row = predictor_diagnostics(reversed_frame, assignment, [predictor], fold_ids=(1,)).iloc[0]
    assert row["train_spearman"] > 0.5
    assert row["validation_spearman"] < -0.5
    assert bool(row["spearman_sign_reversed"]) is True
    assert row["spearman_change"] == pytest.approx(
        row["validation_spearman"] - row["train_spearman"]
    )


def test_rank_predictor_drift_orders_by_magnitude_and_flags_consistency():
    diagnostics = pd.DataFrame(
        {
            "fold_id": [1, 2, 3, 1, 2, 3],
            "predictor": ["steady"] * 3 + ["moving"] * 3,
            "standardized_mean_difference": [0.01, -0.02, 0.01, 0.8, 0.7, 0.9],
            "ks_statistic": [0.01, 0.02, 0.01, 0.4, 0.35, 0.45],
            "psi": [0.001, 0.002, 0.001, 0.6, 0.5, 0.7],
            "variance_ratio": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        }
    )
    ranking = rank_predictor_drift(diagnostics, smd_material=0.25)

    assert list(ranking["predictor"]) == ["moving", "steady"]
    moving = ranking.set_index("predictor").loc["moving"]
    assert moving["mean_absolute_smd"] == pytest.approx(0.8)
    assert int(moving["n_folds_material_smd"]) == 3
    assert bool(moving["same_direction_in_all_folds"]) is True

    steady = ranking.set_index("predictor").loc["steady"]
    assert int(steady["n_folds_material_smd"]) == 0
    assert bool(steady["same_direction_in_all_folds"]) is False


def test_relationship_stability_classification():
    diagnostics = pd.DataFrame(
        {
            "fold_id": [1, 2, 3] * 3,
            "predictor": ["stays"] * 3 + ["flips"] * 3 + ["fades"] * 3,
            "train_spearman": [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
            "validation_spearman": [0.5, 0.5, 0.5, -0.3, 0.4, 0.4, 0.2, 0.2, 0.2],
            "spearman_change": [0.0, 0.0, 0.0, -0.7, 0.0, 0.0, -0.4, -0.4, -0.4],
        }
    )
    stability = classify_relationship_stability(diagnostics).set_index("predictor")

    assert stability.loc["stays", "classification"] == "stable"
    assert stability.loc["flips", "classification"] == "reverses"
    assert int(stability.loc["flips", "n_folds_reversed"]) == 1
    assert stability.loc["fades", "classification"] == "weakens"
    assert int(stability.loc["fades", "n_folds_weakened"]) == 3


# ---------------------------------------------------------------------
# Residuals: fold isolation, chronology, contamination
# ---------------------------------------------------------------------


def test_no_final_test_contamination_passes_and_fails_correctly():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    assert_no_final_test_contamination(residuals, assignment, hourly)

    contaminated = pd.concat(
        [
            residuals,
            residuals.head(1).assign(
                **{TIMESTAMP_COLUMN: hourly[TIMESTAMP_COLUMN].iloc[180]}
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="final test hour"):
        assert_no_final_test_contamination(contaminated, assignment, hourly)


def test_contamination_guard_detects_a_missing_validation_hour():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    with pytest.raises(ValueError, match="differ from the committed validation window"):
        assert_no_final_test_contamination(residuals.iloc[1:], assignment, hourly)


def test_residual_summary_matches_direct_calculation():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    summary = residual_summary(residuals).iloc[0]

    observed = residuals[TARGET_COLUMN].to_numpy()
    predicted = residuals[PREDICTION_COLUMN].to_numpy()
    residual = observed - predicted

    assert summary["n"] == len(residuals)
    assert summary["residual_mean"] == pytest.approx(residual.mean())
    assert summary["residual_std"] == pytest.approx(residual.std(ddof=0))
    assert summary["mae"] == pytest.approx(np.mean(np.abs(residual)))
    assert summary["rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))


def test_residual_summary_isolates_folds():
    """Two folds must be summarized independently, never pooled."""
    hourly = make_hourly(n_rows=260)
    assignment = pd.concat(
        [
            make_assignment(
                hourly,
                train_idx=list(range(0, 100)),
                embargo_idx=list(range(100, 110)),
                validation_idx=list(range(110, 160)),
                fold_id=1,
            ),
            make_assignment(
                hourly,
                train_idx=list(range(0, 160)),
                embargo_idx=list(range(160, 170)),
                validation_idx=list(range(170, 220)),
                fold_id=2,
            ),
        ],
        ignore_index=True,
    )
    residuals = make_residuals(hourly, assignment, fold_ids=(1, 2))

    summary = residual_summary(residuals).set_index("fold_id")
    assert list(summary.index) == [1, 2]
    assert (summary["n"] == 50).all()

    for fold_id in (1, 2):
        only = residual_summary(residuals[residuals["fold_id"] == fold_id]).iloc[0]
        assert summary.loc[fold_id, "rmse"] == pytest.approx(only["rmse"])


def test_residual_subperiods_are_chronological_and_partition_the_window():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    blocks = residual_subperiods(residuals, n_subperiods=4)

    assert list(blocks["subperiod"]) == [1, 2, 3, 4]
    assert blocks["n"].sum() == len(residuals)
    # Blocks follow one another in time and do not overlap.
    assert blocks["start"].is_monotonic_increasing
    assert blocks["end"].is_monotonic_increasing
    assert (blocks["start"].iloc[1:].to_numpy() > blocks["end"].iloc[:-1].to_numpy()).all()

    # Each block's RMSE is computed from its own rows only.
    first = residuals.sort_values(TIMESTAMP_COLUMN).head(int(blocks["n"].iloc[0]))
    expected = float(np.sqrt(np.mean(first["residual"].to_numpy() ** 2)))
    assert blocks["rmse"].iloc[0] == pytest.approx(expected)


def test_residual_subperiods_respect_an_unsorted_input():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    shuffled = residuals.sample(frac=1.0, random_state=11).reset_index(drop=True)

    assert residual_subperiods(shuffled).equals(residual_subperiods(residuals))


def test_residual_trend_detects_growing_error():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    ordered = residuals.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    ordered["absolute_residual"] = np.arange(len(ordered), dtype=float)
    ordered["residual"] = ordered["absolute_residual"]

    trend = residual_trend(ordered).iloc[0]
    assert trend["position_vs_absolute_residual_spearman"] == pytest.approx(1.0)


def test_residual_diagnostics_are_deterministic():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)

    assert residual_summary(residuals).equals(residual_summary(residuals))
    assert residual_subperiods(residuals).equals(residual_subperiods(residuals))
    assert residual_trend(residuals).equals(residual_trend(residuals))


# ---------------------------------------------------------------------
# Context annotation, operating regime, assay gaps
# ---------------------------------------------------------------------


def test_annotation_measures_gaps_and_distance_to_interpolated_hours():
    hourly, assignment = default_fixture()
    hourly.loc[hourly.index == 135, INTERPOLATED_COLUMN] = True
    residuals = make_residuals(hourly, assignment)

    annotated = annotate_residual_context(residuals, hourly)
    indexed = annotated.set_index(TIMESTAMP_COLUMN)["hours_to_nearest_interpolated"]

    assert indexed.loc[hourly[TIMESTAMP_COLUMN].iloc[135]] == pytest.approx(0.0)
    assert indexed.loc[hourly[TIMESTAMP_COLUMN].iloc[137]] == pytest.approx(2.0)
    assert indexed.loc[hourly[TIMESTAMP_COLUMN].iloc[130]] == pytest.approx(5.0)
    # A contiguous grid means every hour follows the previous one.
    assert (annotated["hours_since_previous_row"] == 1.0).all()


def test_annotation_detects_a_recording_gap():
    hourly = make_hourly()
    hourly = hourly.drop(index=[140, 141]).reset_index(drop=True)
    assignment = make_assignment(
        hourly,
        train_idx=list(range(0, 120)),
        embargo_idx=list(range(120, 130)),
        validation_idx=list(range(130, 178)),
    )
    residuals = make_residuals(hourly, assignment)
    annotated = annotate_residual_context(residuals, hourly)

    assert annotated["hours_since_previous_row"].max() == pytest.approx(3.0)


def test_high_error_labelling_is_per_fold():
    hourly = make_hourly(n_rows=260)
    assignment = pd.concat(
        [
            make_assignment(
                hourly,
                train_idx=list(range(0, 100)),
                embargo_idx=list(range(100, 110)),
                validation_idx=list(range(110, 160)),
                fold_id=1,
            ),
            make_assignment(
                hourly,
                train_idx=list(range(0, 160)),
                embargo_idx=list(range(160, 170)),
                validation_idx=list(range(170, 220)),
                fold_id=2,
            ),
        ],
        ignore_index=True,
    )
    residuals = make_residuals(hourly, assignment, fold_ids=(1, 2))
    # Make fold 2 uniformly worse; the cut must still select within each fold.
    residuals.loc[residuals["fold_id"] == 2, "absolute_residual"] += 10.0

    labelled = label_high_error(residuals, quantile=HIGH_ERROR_QUANTILE)
    counts = labelled.groupby("fold_id")["is_high_error"].sum()
    assert counts.loc[1] > 0
    assert counts.loc[2] > 0
    assert abs(int(counts.loc[1]) - int(counts.loc[2])) <= 1


def test_operating_regime_profile_recovers_an_injected_condition():
    hourly, assignment = default_fixture()
    predictor = CORE_SENSOR_PREDICTOR_COLUMNS[0]
    residuals = make_residuals(hourly, assignment)

    # Force the largest errors onto hours where one predictor runs high.
    validation_hours = residuals[TIMESTAMP_COLUMN].to_numpy()
    raised = set(validation_hours[:5])
    hourly.loc[hourly[TIMESTAMP_COLUMN].isin(raised), predictor] = 20.0
    residuals.loc[residuals[TIMESTAMP_COLUMN].isin(raised), "absolute_residual"] = 99.0

    labelled = label_high_error(residuals)
    profile = operating_regime_profile(
        labelled, hourly, list(CORE_SENSOR_PREDICTOR_COLUMNS)
    )
    row = profile[profile["predictor"] == predictor].iloc[0]

    assert row["high_error_mean"] > row["normal_mean"]
    assert row["standardized_difference"] > 1.0

    ranking = rank_operating_regime(profile)
    assert ranking["predictor"].iloc[0] == predictor


def test_assay_gap_context_separates_near_and_far_hours():
    hourly, assignment = default_fixture()
    near_index = [131, 132]
    hourly.loc[hourly.index == 130, INTERPOLATED_COLUMN] = True
    residuals = make_residuals(hourly, assignment)
    annotated = annotate_residual_context(residuals, hourly)

    context = assay_gap_context(
        annotated, hourly, list(CORE_SENSOR_PREDICTOR_COLUMNS)
    ).set_index("group")

    # Hours 130, 131 and 132 sit within two hours of the interpolated hour.
    assert int(context.loc["near_gap", "n"]) == 1 + len(near_index)
    assert int(context.loc["far_from_gap", "n"]) == len(residuals) - 3
    assert int(context["n"].sum()) == len(residuals)


# ---------------------------------------------------------------------
# Baseline decomposition
# ---------------------------------------------------------------------


def test_baseline_decomposition_reproduces_r_squared_from_its_parts():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    decomposition = baseline_decomposition(residuals, hourly, assignment).iloc[0]

    observed = residuals[TARGET_COLUMN].to_numpy()
    residual = residuals["residual"].to_numpy()
    n = observed.size
    sst = float(np.sum((observed - observed.mean()) ** 2))
    sse = float(np.sum(residual**2))

    assert decomposition["n"] == n
    assert decomposition["total_sum_of_squares"] == pytest.approx(sst)
    assert decomposition["model_sse"] == pytest.approx(sse)
    assert decomposition["r2_model"] == pytest.approx(1.0 - sse / sst)
    # The identity the report relies on: R2 is one minus the squared ratio
    # of model RMSE to the validation standard deviation.
    assert decomposition["r2_model"] == pytest.approx(
        1.0 - decomposition["rmse_over_validation_std"] ** 2
    )


def test_baseline_decomposition_separates_drift_from_scatter():
    """The training mean baseline's error must equal the validation spread
    plus a penalty that is exactly the squared mean drift."""
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    decomposition = baseline_decomposition(residuals, hourly, assignment).iloc[0]

    assert decomposition["training_mean_sse"] == pytest.approx(
        decomposition["total_sum_of_squares"] + decomposition["training_mean_drift_penalty"]
    )
    assert decomposition["training_mean_drift_penalty"] == pytest.approx(
        decomposition["n"] * decomposition["mean_drift"] ** 2
    )
    # Model MSE splits into squared bias plus residual variance.
    residual = residuals["residual"].to_numpy()
    assert decomposition["model_mse"] == pytest.approx(
        residual.mean() ** 2 + residual.var(ddof=0)
    )


def test_negative_r_squared_follows_from_error_exceeding_validation_spread():
    """A constructed case: a model whose error is larger than the validation
    standard deviation scores below zero, regardless of any mean drift."""
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)

    noisy = residuals.copy()
    rng = np.random.default_rng(9)
    observed = noisy[TARGET_COLUMN].to_numpy()
    noisy[PREDICTION_COLUMN] = observed.mean() + rng.normal(
        scale=observed.std(ddof=0) * 2.0, size=observed.size
    )
    noisy["residual"] = noisy[TARGET_COLUMN] - noisy[PREDICTION_COLUMN]

    decomposition = baseline_decomposition(noisy, hourly, assignment).iloc[0]
    assert decomposition["rmse_over_validation_std"] > 1.0
    assert decomposition["r2_model"] < 0.0


def test_baseline_decomposition_is_deterministic():
    hourly, assignment = default_fixture()
    residuals = make_residuals(hourly, assignment)
    assert baseline_decomposition(residuals, hourly, assignment).equals(
        baseline_decomposition(residuals, hourly, assignment)
    )


# ---------------------------------------------------------------------
# Real data integration (skips cleanly if artifacts are absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_diagnostics_use_only_development_folds():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    distributions = target_distributions(hourly, assignment)
    assert sorted(distributions["fold_id"].unique()) == [1, 2, 3]
    assert list(
        distributions.loc[distributions["split"] == "train", "n"]
    ) == [1743, 2224, 2708]
    assert list(
        distributions.loc[distributions["split"] == "validation", "n"]
    ) == [482, 483, 481]

    table = predictor_diagnostics(hourly, assignment, predictors)
    assert len(table) == 3 * 57
    assert table[["standardized_mean_difference", "ks_statistic"]].notna().all().all()

    # No development row may come from the final test period.
    final_test = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    for fold_id in (1, 2, 3):
        frames = get_fold_frames(hourly, assignment, fold_id)
        assert not set(frames.train[TIMESTAMP_COLUMN]).intersection(final_test)
        assert not set(frames.validation[TIMESTAMP_COLUMN]).intersection(final_test)


@pytest.mark.skipif(not REAL_ARTIFACTS, reason="processed artifacts not available locally")
def test_real_data_drift_tables_are_deterministic():
    hourly = pd.read_parquet(REAL_HOURLY)
    assignment = pd.read_parquet(REAL_SPLITS)
    predictors = list(CORE_SENSOR_PREDICTOR_COLUMNS)

    assert target_drift(hourly, assignment).equals(target_drift(hourly, assignment))

    first = predictor_diagnostics(hourly, assignment, predictors)
    second = predictor_diagnostics(hourly, assignment, predictors)
    assert first.equals(second)
    assert rank_predictor_drift(first).equals(rank_predictor_drift(second))
