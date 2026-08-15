"""Diagnosis of why the sensor models do not generalize forward.

Three conventional regression model families were evaluated on the same
chronological folds. Random Forest improved RMSE relative to the training
mean baseline, but forward R squared remained negative in two of three
development folds. Predictor to target relationships were weak and
unstable. The 1 and 2 hour target shifts did not establish reliable
forward generalization. This module diagnoses those results without
proposing another model family.

It measures five things and keeps them separate:

* whether the target itself moves between a fold's training period and
  its validation period
* whether the 57 sensor predictors move
* whether the relationship between each predictor and the target holds
  between the two periods
* where the Random Forest errors fall in time, and in what process
  conditions
* what actually produces a negative R squared, arithmetically

Nothing here fits a new model, selects features, or touches the final
test period. The Random Forest benchmark is refitted once, unchanged,
purely to recover its validation predictions, which the benchmark module
does not persist; the refit is checked against the committed results
before any residual is analyzed.

Effect sizes, not significance tests
------------------------------------
Drift is reported as standardized mean difference, variance ratio,
Kolmogorov Smirnov distance, and population stability index. All four are
effect sizes that do not grow with sample size. No p value is reported,
because with a few thousand hours per fold almost any difference is
statistically significant and that fact carries no information about
whether a difference matters.

Correlation measure
-------------------
Predictor to target association is measured with Spearman rank
correlation. The sensor aggregates include standard deviation and slope
summaries with heavy tails, and the target is bounded below and skewed,
so a rank measure describes the association more honestly than Pearson.
Pearson is reported alongside it for reference, and the two are compared
rather than one being assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import TARGET_COLUMN, TIMESTAMP_COLUMN, find_repo_root
from src.data.split import ROLE_TEST, SEGMENT_COLUMN
from src.models.baselines import (
    DEVELOPMENT_FOLD_IDS,
    INTERPOLATED_COLUMN,
    compute_metrics,
    get_fold_frames,
    load_inputs,
)
from src.models.linear_regression import build_spark_session, get_sensor_predictors
from src.models.random_forest import PREDICTION_COLUMN, evaluate_model

# Number of chronological blocks each validation window is divided into
# for the temporal error profile. Four keeps roughly 120 hours per block
# at the current fold sizes, enough for a stable RMSE per block.
N_SUBPERIODS = 4

# Validation rows whose absolute residual sits at or above this quantile
# of their own fold are treated as the high error group for the operating
# regime comparison. It is a descriptive cut, not a threshold the models
# are tuned against.
HIGH_ERROR_QUANTILE = 0.90

# Bins for the population stability index. Ten equal frequency training
# bins is the conventional choice and keeps roughly 170 or more training
# hours per bin at the smallest fold size.
PSI_BINS = 10

# Conventional reading points for the drift measures. They are reporting
# thresholds applied after measurement, never inputs to any model.
SMD_MATERIAL = 0.25
PSI_MATERIAL = 0.10
CORRELATION_MATERIAL = 0.10


# ---------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------


def _as_float_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sample.")
    if not np.isfinite(array).all():
        raise ValueError("Sample contains non-finite values.")
    return array


def describe_sample(values) -> dict[str, float]:
    """Count, location, spread, extremes, quantiles, and shape."""
    array = _as_float_array(values)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "skewness": skewness(array),
        "excess_kurtosis": excess_kurtosis(array),
    }


def skewness(values) -> float:
    """Population skewness. Zero for a symmetric sample."""
    array = _as_float_array(values)
    deviation = array - array.mean()
    spread = array.std(ddof=0)
    if spread == 0.0:
        return 0.0
    return float(np.mean(deviation**3) / spread**3)


def excess_kurtosis(values) -> float:
    """Population excess kurtosis. Zero for a normal sample."""
    array = _as_float_array(values)
    deviation = array - array.mean()
    spread = array.std(ddof=0)
    if spread == 0.0:
        return 0.0
    return float(np.mean(deviation**4) / spread**4 - 3.0)


def standardized_mean_difference(train, validation) -> float:
    """Mean shift expressed in pooled standard deviations.

    Independent of sample size, so a large fold cannot manufacture an
    apparent shift the way a significance test would. A positive value
    means the validation period sits higher than the training period.
    """
    a = _as_float_array(train)
    b = _as_float_array(validation)
    pooled = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
    difference = float(b.mean() - a.mean())
    if pooled == 0.0:
        return 0.0 if difference == 0.0 else float("nan")
    return difference / float(pooled)


def variance_ratio(train, validation) -> float:
    """Validation variance divided by training variance.

    Below 1 means the validation window is quieter than the period the
    model was fitted on, which matters here because R squared is measured
    against the validation window's own variance.
    """
    a = _as_float_array(train)
    b = _as_float_array(validation)
    denominator = a.var(ddof=0)
    if denominator == 0.0:
        return float("nan")
    return float(b.var(ddof=0) / denominator)


def ks_statistic(train, validation) -> float:
    """Largest gap between the two empirical distribution functions.

    Reported as a distance in [0, 1], never as a test result. The
    statistic itself is an effect size; its p value would be driven by
    the sample size and is deliberately not computed.
    """
    a = np.sort(_as_float_array(train))
    b = np.sort(_as_float_array(validation))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def population_stability_index(train, validation, n_bins: int = PSI_BINS) -> float:
    """Population stability index over equal frequency training bins.

    Bin edges come from the training sample only, so the measure asks how
    the validation period redistributes itself across the shape the model
    was fitted on. Returns NaN when the training sample has too few
    distinct values to form at least two bins.
    """
    a = _as_float_array(train)
    b = _as_float_array(validation)

    edges = np.unique(np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return float("nan")
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    train_share = np.histogram(a, bins=edges)[0] / a.size
    validation_share = np.histogram(b, bins=edges)[0] / b.size

    # An empty validation bin would send the logarithm to negative
    # infinity. Flooring both shares keeps the index finite and bounds the
    # contribution of a bin that simply was not visited.
    floor = 1.0 / max(a.size, b.size)
    train_share = np.clip(train_share, floor, None)
    validation_share = np.clip(validation_share, floor, None)

    return float(np.sum((validation_share - train_share) * np.log(validation_share / train_share)))


# ---------------------------------------------------------------------
# Inputs, including one controlled Random Forest refit
# ---------------------------------------------------------------------


def collect_residuals(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
    committed_results: pd.DataFrame | None = None,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Recover the Random Forest validation predictions and residuals.

    The benchmark module writes metrics but not predictions, so the
    unchanged benchmark is refitted once here. When the committed results
    are supplied the refit is verified against them before anything is
    read from it, so a residual analysis can never be built on a model
    that differs from the one being diagnosed.

    Residual convention: observed minus predicted. A positive residual
    means the model predicted too low.
    """
    results, fold_results = evaluate_model(spark, hourly, assignment, fold_ids=fold_ids)

    if committed_results is not None:
        merged = results.merge(
            committed_results[["fold_id", "rmse", "mae", "r2"]],
            on="fold_id",
            suffixes=("", "_committed"),
        )
        if len(merged) != len(results):
            raise ValueError("The refit does not cover the same folds as the committed benchmark.")
        for _, row in merged.iterrows():
            for metric in ("rmse", "mae", "r2"):
                if abs(float(row[metric]) - float(row[f"{metric}_committed"])) > tolerance:
                    raise ValueError(
                        f"Fold {int(row['fold_id'])}: refitted {metric} is {row[metric]:.10f} but "
                        f"the committed benchmark recorded {row[f'{metric}_committed']:.10f}. "
                        "The residuals would not describe the benchmarked model."
                    )

    frames = []
    for result in fold_results:
        frame = result.predictions.copy()
        frame["fold_id"] = result.fold_id
        frame["residual"] = frame[TARGET_COLUMN] - frame[PREDICTION_COLUMN]
        frame["absolute_residual"] = frame["residual"].abs()
        frames.append(frame)

    residuals = pd.concat(frames, ignore_index=True)
    return residuals.sort_values(["fold_id", TIMESTAMP_COLUMN], kind="mergesort").reset_index(
        drop=True
    )


def assert_no_final_test_contamination(
    residuals: pd.DataFrame, assignment: pd.DataFrame, hourly: pd.DataFrame
) -> None:
    """Confirm nothing under diagnosis comes from the final test period."""
    final_test = set(assignment.loc[assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN])
    scored = set(residuals[TIMESTAMP_COLUMN])
    if scored.intersection(final_test):
        raise ValueError("A final test hour appears in the diagnostic residuals.")

    for fold_id in sorted(residuals["fold_id"].unique()):
        frames = get_fold_frames(hourly, assignment, int(fold_id))
        expected = set(frames.validation[TIMESTAMP_COLUMN])
        actual = set(residuals.loc[residuals["fold_id"] == fold_id, TIMESTAMP_COLUMN])
        if actual != expected:
            raise ValueError(
                f"Fold {int(fold_id)}: residual hours differ from the committed validation window."
            )
        if set(frames.train[TIMESTAMP_COLUMN]).intersection(final_test):
            raise ValueError(f"Fold {int(fold_id)}: training overlaps the final test period.")


# ---------------------------------------------------------------------
# 1. Target drift
# ---------------------------------------------------------------------


def target_distributions(
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> pd.DataFrame:
    """Training and validation target distributions, one row per fold and split."""
    rows = []
    for fold_id in fold_ids:
        frames = get_fold_frames(hourly, assignment, fold_id)
        for split, frame in (("train", frames.train), ("validation", frames.validation)):
            rows.append(
                {"fold_id": fold_id, "split": split, **describe_sample(frame[TARGET_COLUMN])}
            )
    return pd.DataFrame(rows)


def target_drift(
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> pd.DataFrame:
    """Change in the target's level, spread, and shape between the two periods."""
    rows = []
    for fold_id in fold_ids:
        frames = get_fold_frames(hourly, assignment, fold_id)
        train = frames.train[TARGET_COLUMN].to_numpy(dtype=float)
        validation = frames.validation[TARGET_COLUMN].to_numpy(dtype=float)

        rows.append(
            {
                "fold_id": fold_id,
                "train_mean": float(train.mean()),
                "validation_mean": float(validation.mean()),
                "mean_change": float(validation.mean() - train.mean()),
                "standardized_mean_difference": standardized_mean_difference(train, validation),
                "train_std": float(train.std(ddof=0)),
                "validation_std": float(validation.std(ddof=0)),
                "variance_ratio": variance_ratio(train, validation),
                "iqr_ratio": float(
                    (np.quantile(validation, 0.75) - np.quantile(validation, 0.25))
                    / (np.quantile(train, 0.75) - np.quantile(train, 0.25))
                ),
                "ks_statistic": ks_statistic(train, validation),
                "psi": population_stability_index(train, validation),
                "skewness_change": skewness(validation) - skewness(train),
                "kurtosis_change": excess_kurtosis(validation) - excess_kurtosis(train),
                "validation_within_train_range": bool(
                    validation.min() >= train.min() and validation.max() <= train.max()
                ),
                "share_below_train_q05": float(
                    np.mean(validation < np.quantile(train, 0.05))
                ),
                "share_above_train_q95": float(
                    np.mean(validation > np.quantile(train, 0.95))
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Predictor drift and 3. relationship stability
# ---------------------------------------------------------------------


def predictor_diagnostics(
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    predictors: list[str],
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> pd.DataFrame:
    """Per fold, per predictor drift and predictor to target association.

    Drift and relationship stability are computed together because they
    answer different questions about the same pair of samples and are read
    against each other: a predictor can move without its relationship to
    the target changing, and a relationship can invert without the
    predictor moving at all.
    """
    rows = []
    for fold_id in fold_ids:
        frames = get_fold_frames(hourly, assignment, fold_id)
        train, validation = frames.train, frames.validation
        train_target = train[TARGET_COLUMN]
        validation_target = validation[TARGET_COLUMN]

        for predictor in predictors:
            train_values = train[predictor].to_numpy(dtype=float)
            validation_values = validation[predictor].to_numpy(dtype=float)

            train_spearman = float(train[predictor].corr(train_target, method="spearman"))
            validation_spearman = float(
                validation[predictor].corr(validation_target, method="spearman")
            )
            train_pearson = float(train[predictor].corr(train_target, method="pearson"))
            validation_pearson = float(
                validation[predictor].corr(validation_target, method="pearson")
            )

            rows.append(
                {
                    "fold_id": fold_id,
                    "predictor": predictor,
                    "train_mean": float(train_values.mean()),
                    "validation_mean": float(validation_values.mean()),
                    "standardized_mean_difference": standardized_mean_difference(
                        train_values, validation_values
                    ),
                    "variance_ratio": variance_ratio(train_values, validation_values),
                    "ks_statistic": ks_statistic(train_values, validation_values),
                    "psi": population_stability_index(train_values, validation_values),
                    "train_spearman": train_spearman,
                    "validation_spearman": validation_spearman,
                    "spearman_change": validation_spearman - train_spearman,
                    "spearman_sign_reversed": bool(
                        train_spearman * validation_spearman < 0
                    ),
                    "train_pearson": train_pearson,
                    "validation_pearson": validation_pearson,
                    "pearson_change": validation_pearson - train_pearson,
                }
            )
    return pd.DataFrame(rows)


def rank_predictor_drift(
    diagnostics: pd.DataFrame, smd_material: float = SMD_MATERIAL
) -> pd.DataFrame:
    """Rank predictors by how far they move and how consistently.

    Magnitude is the mean absolute standardized mean difference across
    folds. Consistency is how many folds move the predictor in the same
    direction, and how many exceed the reporting threshold. A predictor
    that moves hard in one fold and not at all in the others is a
    different finding from one that moves in all three, and the table
    keeps them distinguishable.
    """
    grouped = diagnostics.groupby("predictor")
    table = pd.DataFrame(
        {
            "mean_absolute_smd": grouped["standardized_mean_difference"].apply(
                lambda values: float(values.abs().mean())
            ),
            "max_absolute_smd": grouped["standardized_mean_difference"].apply(
                lambda values: float(values.abs().max())
            ),
            "mean_smd": grouped["standardized_mean_difference"].mean(),
            "n_folds_material_smd": grouped["standardized_mean_difference"].apply(
                lambda values: int((values.abs() >= smd_material).sum())
            ),
            "same_direction_in_all_folds": grouped["standardized_mean_difference"].apply(
                lambda values: bool(np.all(values > 0) or np.all(values < 0))
            ),
            "mean_ks": grouped["ks_statistic"].mean(),
            "mean_psi": grouped["psi"].mean(),
            "mean_variance_ratio": grouped["variance_ratio"].mean(),
        }
    ).reset_index()

    return table.sort_values(
        ["mean_absolute_smd", "mean_psi"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def classify_relationship_stability(
    diagnostics: pd.DataFrame, material: float = CORRELATION_MATERIAL
) -> pd.DataFrame:
    """Summarize how each predictor's association with the target holds up.

    A predictor is classified from its behaviour across all three folds,
    not from any single one. `reverses` means the rank correlation changes
    sign in at least one fold while being non trivial on the training
    side; `weakens` means the association is materially smaller in
    validation without changing sign; `fold_specific` means the training
    side association is itself inconsistent across folds.
    """
    rows = []
    for predictor, group in diagnostics.groupby("predictor"):
        train_values = group["train_spearman"].to_numpy(dtype=float)
        validation_values = group["validation_spearman"].to_numpy(dtype=float)
        change = group["spearman_change"].to_numpy(dtype=float)

        reversals = int(
            np.sum((train_values * validation_values < 0) & (np.abs(train_values) >= material))
        )
        weakened = int(
            np.sum(
                (np.abs(validation_values) < np.abs(train_values) - material)
                & (train_values * validation_values >= 0)
            )
        )
        train_consistent = bool(np.all(train_values > 0) or np.all(train_values < 0))

        if reversals > 0:
            classification = "reverses"
        elif not train_consistent and float(np.abs(train_values).max()) >= material:
            classification = "fold_specific"
        elif weakened >= 2:
            classification = "weakens"
        else:
            classification = "stable"

        rows.append(
            {
                "predictor": predictor,
                "mean_train_spearman": float(train_values.mean()),
                "mean_validation_spearman": float(validation_values.mean()),
                "mean_absolute_change": float(np.abs(change).mean()),
                "n_folds_reversed": reversals,
                "n_folds_weakened": weakened,
                "train_sign_consistent": train_consistent,
                "classification": classification,
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values("mean_absolute_change", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# 4. Temporal error analysis
# ---------------------------------------------------------------------


def residual_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    """Per fold residual level, spread, and error magnitude."""
    rows = []
    for fold_id, group in residuals.groupby("fold_id"):
        observed = group[TARGET_COLUMN].to_numpy(dtype=float)
        predicted = group[PREDICTION_COLUMN].to_numpy(dtype=float)
        residual = group["residual"].to_numpy(dtype=float)
        metrics = compute_metrics(observed, predicted)
        rows.append(
            {
                "fold_id": int(fold_id),
                "n": len(group),
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std(ddof=0)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "prediction_std": float(predicted.std(ddof=0)),
                "observed_std": float(observed.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def residual_subperiods(
    residuals: pd.DataFrame, n_subperiods: int = N_SUBPERIODS
) -> pd.DataFrame:
    """Error profile across equal sized chronological blocks of each window.

    Blocks are cut by position in the ordered validation window rather
    than by calendar date, so every block holds a comparable number of
    hours and the RMSE values are equally reliable.
    """
    rows = []
    for fold_id, group in residuals.groupby("fold_id"):
        ordered = group.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
        blocks = np.array_split(np.arange(len(ordered)), n_subperiods)
        for position, index in enumerate(blocks, start=1):
            if index.size == 0:
                continue
            block = ordered.iloc[index]
            residual = block["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "subperiod": position,
                    "n": len(block),
                    "start": block[TIMESTAMP_COLUMN].min(),
                    "end": block[TIMESTAMP_COLUMN].max(),
                    "residual_mean": float(residual.mean()),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "observed_mean": float(block[TARGET_COLUMN].mean()),
                    "predicted_mean": float(block[PREDICTION_COLUMN].mean()),
                }
            )
    return pd.DataFrame(rows)


def residual_trend(residuals: pd.DataFrame) -> pd.DataFrame:
    """Whether error grows or shrinks through each validation window.

    Reported as the Spearman correlation between chronological position
    and absolute residual. A rank measure is used because a few very large
    errors would otherwise decide the slope.
    """
    rows = []
    for fold_id, group in residuals.groupby("fold_id"):
        ordered = group.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
        position = pd.Series(np.arange(len(ordered), dtype=float))
        rows.append(
            {
                "fold_id": int(fold_id),
                "position_vs_absolute_residual_spearman": float(
                    position.corr(ordered["absolute_residual"], method="spearman")
                ),
                "position_vs_residual_spearman": float(
                    position.corr(ordered["residual"], method="spearman")
                ),
                # Whether error tracks the level of the target itself, which
                # separates "errors happen at certain times" from "errors
                # happen at certain target values".
                "observed_vs_absolute_residual_spearman": float(
                    ordered[TARGET_COLUMN].corr(ordered["absolute_residual"], method="spearman")
                ),
                "observed_vs_predicted_spearman": float(
                    ordered[TARGET_COLUMN].corr(ordered[PREDICTION_COLUMN], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def annotate_residual_context(
    residuals: pd.DataFrame, hourly: pd.DataFrame
) -> pd.DataFrame:
    """Attach temporal gap and assay reporting context to every residual.

    `hours_since_previous_row` measures the distance to the previous hour
    present in the table at all, so a value above 1 marks a recording gap.
    `hours_to_nearest_interpolated` measures the distance to the nearest
    hour whose assay was interpolated rather than observed. Both are
    diagnostic annotations; no row is removed on the strength of them.
    """
    ordered = hourly.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    gap = ordered[TIMESTAMP_COLUMN].diff().dt.total_seconds() / 3600.0
    gap_by_hour = pd.Series(gap.to_numpy(), index=ordered[TIMESTAMP_COLUMN])

    interpolated_hours = ordered.loc[
        ordered[INTERPOLATED_COLUMN].astype(bool), TIMESTAMP_COLUMN
    ].to_numpy()

    annotated = residuals.copy()
    annotated["hours_since_previous_row"] = annotated[TIMESTAMP_COLUMN].map(gap_by_hour)

    if interpolated_hours.size == 0:
        annotated["hours_to_nearest_interpolated"] = np.inf
    else:
        target = annotated[TIMESTAMP_COLUMN].to_numpy()
        distances = np.abs(
            target[:, None].astype("datetime64[ns]") - interpolated_hours[None, :]
        )
        annotated["hours_to_nearest_interpolated"] = (
            distances.min(axis=1) / np.timedelta64(1, "h")
        ).astype(float)

    annotated[SEGMENT_COLUMN] = annotated[TIMESTAMP_COLUMN].map(
        ordered.set_index(TIMESTAMP_COLUMN)[SEGMENT_COLUMN]
    )
    return annotated


def residuals_near_gaps(annotated: pd.DataFrame, gap_hours: float = 1.0) -> pd.DataFrame:
    """Error on hours that follow a recording gap versus contiguous hours."""
    rows = []
    for fold_id, group in annotated.groupby("fold_id"):
        after_gap = group["hours_since_previous_row"] > gap_hours
        for label, subset in (("after_gap", group[after_gap]), ("contiguous", group[~after_gap])):
            if subset.empty:
                continue
            residual = subset["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "group": label,
                    "n": len(subset),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "observed_mean": float(subset[TARGET_COLUMN].mean()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 5. Operating regime analysis
# ---------------------------------------------------------------------


def label_high_error(
    annotated: pd.DataFrame, quantile: float = HIGH_ERROR_QUANTILE
) -> pd.DataFrame:
    """Flag the largest absolute residuals within each fold.

    The cut is applied per fold so that a fold with generally larger
    errors does not dominate the comparison.
    """
    labelled = annotated.copy()
    threshold = labelled.groupby("fold_id")["absolute_residual"].transform(
        lambda values: values.quantile(quantile)
    )
    labelled["is_high_error"] = labelled["absolute_residual"] >= threshold
    return labelled


def operating_regime_profile(
    labelled: pd.DataFrame, hourly: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    """Describe how sensor conditions differ on the highest error hours.

    For every predictor, the mean on high error hours is compared with the
    mean on the remaining validation hours, standardized by the fold's own
    validation spread. This is a description of where the errors fall, not
    a claim that any variable causes them.
    """
    indexed = hourly.set_index(TIMESTAMP_COLUMN)
    rows = []
    for fold_id, group in labelled.groupby("fold_id"):
        features = indexed.loc[group[TIMESTAMP_COLUMN], predictors]
        high = features[group["is_high_error"].to_numpy()]
        normal = features[~group["is_high_error"].to_numpy()]
        spread = features.std(ddof=0)

        for predictor in predictors:
            denominator = float(spread[predictor])
            difference = float(high[predictor].mean() - normal[predictor].mean())
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "predictor": predictor,
                    "high_error_mean": float(high[predictor].mean()),
                    "normal_mean": float(normal[predictor].mean()),
                    "standardized_difference": (
                        difference / denominator if denominator > 0 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def rank_operating_regime(profile: pd.DataFrame) -> pd.DataFrame:
    """Predictors that most consistently separate high error hours."""
    grouped = profile.groupby("predictor")["standardized_difference"]
    table = pd.DataFrame(
        {
            "mean_standardized_difference": grouped.mean(),
            "mean_absolute_difference": grouped.apply(lambda values: float(values.abs().mean())),
            "same_direction_in_all_folds": grouped.apply(
                lambda values: bool(np.all(values > 0) or np.all(values < 0))
            ),
        }
    ).reset_index()
    return table.sort_values(
        "mean_absolute_difference", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def high_error_target_profile(labelled: pd.DataFrame) -> pd.DataFrame:
    """Target behaviour on high error hours against the rest of the window."""
    rows = []
    for fold_id, group in labelled.groupby("fold_id"):
        for label, subset in (
            ("high_error", group[group["is_high_error"]]),
            ("normal", group[~group["is_high_error"]]),
        ):
            observed = subset[TARGET_COLUMN].to_numpy(dtype=float)
            predicted = subset[PREDICTION_COLUMN].to_numpy(dtype=float)
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "group": label,
                    "n": len(subset),
                    "observed_mean": float(observed.mean()),
                    "observed_std": float(observed.std(ddof=0)),
                    "predicted_mean": float(predicted.mean()),
                    "predicted_std": float(predicted.std(ddof=0)),
                    "mae": float(np.mean(np.abs(subset["residual"].to_numpy(dtype=float)))),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 6. Baseline decomposition
# ---------------------------------------------------------------------


def baseline_decomposition(
    residuals: pd.DataFrame,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Show arithmetically what produces the reported R squared.

    R squared compares the model's sum of squared errors with the total
    sum of squares taken about the validation window's own mean. Writing
    both quantities out makes clear whether a negative value comes from
    the model erring badly, from the validation window being unusually
    quiet, or from the training period sitting at a different level.

    The model's error is also split into a bias part, the squared mean
    residual, and a scatter part, the residual variance, because a model
    that is simply offset is a different problem from one that is noisy.
    """
    rows = []
    for fold_id, group in residuals.groupby("fold_id"):
        frames = get_fold_frames(hourly, assignment, int(fold_id))
        train_mean = float(frames.train[TARGET_COLUMN].mean())

        observed = group[TARGET_COLUMN].to_numpy(dtype=float)
        residual = group["residual"].to_numpy(dtype=float)
        n = observed.size
        validation_mean = float(observed.mean())
        validation_variance = float(observed.var(ddof=0))

        total_sum_of_squares = float(np.sum((observed - validation_mean) ** 2))
        model_sse = float(np.sum(residual**2))
        training_mean_sse = float(np.sum((observed - train_mean) ** 2))

        rows.append(
            {
                "fold_id": int(fold_id),
                "n": n,
                "train_mean": train_mean,
                "validation_mean": validation_mean,
                "mean_drift": validation_mean - train_mean,
                "validation_variance": validation_variance,
                "validation_std": float(np.sqrt(validation_variance)),
                "total_sum_of_squares": total_sum_of_squares,
                "model_sse": model_sse,
                "model_mse": model_sse / n,
                "model_rmse": float(np.sqrt(model_sse / n)),
                "rmse_over_validation_std": float(
                    np.sqrt(model_sse / n) / np.sqrt(validation_variance)
                ),
                "residual_bias": float(residual.mean()),
                "bias_share_of_mse": float(residual.mean() ** 2 / (model_sse / n)),
                "training_mean_sse": training_mean_sse,
                "training_mean_drift_penalty": float(n * (validation_mean - train_mean) ** 2),
                "drift_share_of_training_mean_sse": float(
                    n * (validation_mean - train_mean) ** 2 / training_mean_sse
                ),
                "r2_model": 1.0 - model_sse / total_sum_of_squares,
                "r2_training_mean": 1.0 - training_mean_sse / total_sum_of_squares,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 7. Assay reporting gap context
# ---------------------------------------------------------------------


def assay_gap_context(
    annotated: pd.DataFrame, hourly: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    """Compare hours near an interpolated assay with hours far from one.

    The modeling population already excludes interpolated hours, so this
    asks a different question: whether the observed hours that sit next to
    a reporting gap behave differently from those that do not. Proximity
    is a diagnostic annotation only and no row is excluded from anything.

    `sensor_divergence` is the mean absolute standardized difference
    across all 57 predictors between the two groups, a single number
    summarizing how far apart the sensor conditions are.
    """
    indexed = hourly.set_index(TIMESTAMP_COLUMN)
    rows = []
    for fold_id, group in annotated.groupby("fold_id"):
        near = group["hours_to_nearest_interpolated"] <= 2.0
        features = indexed.loc[group[TIMESTAMP_COLUMN], predictors]
        spread = features.std(ddof=0)
        near_features = features[near.to_numpy()]
        far_features = features[~near.to_numpy()]

        if near_features.empty or far_features.empty:
            divergence = float("nan")
        else:
            difference = (near_features.mean() - far_features.mean()).abs()
            usable = spread > 0
            divergence = float((difference[usable] / spread[usable]).mean())

        for label, subset in (("near_gap", group[near]), ("far_from_gap", group[~near])):
            if subset.empty:
                continue
            residual = subset["residual"].to_numpy(dtype=float)
            observed = subset[TARGET_COLUMN].to_numpy(dtype=float)
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "group": label,
                    "n": len(subset),
                    "observed_mean": float(observed.mean()),
                    "observed_std": float(observed.std(ddof=0)),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "sensor_divergence": divergence,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Assembly, report, CLI
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_random_forest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "random_forest_results.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "generalization_diagnostics.parquet"


def format_report(diagnostics: dict) -> str:
    lines = ["Generalization diagnostics (development folds only)", ""]

    lines.append("Target distributions")
    distributions = diagnostics["target_distributions"]
    lines.append(
        f"  {'fold':>4}  {'split':<11}  {'n':>5}  {'mean':>6}  {'median':>6}  {'sd':>5}  "
        f"{'min':>5}  {'max':>5}  {'q05':>5}  {'q95':>5}"
    )
    for _, row in distributions.iterrows():
        lines.append(
            f"  {int(row['fold_id']):>4}  {row['split']:<11}  {int(row['n']):>5,}  "
            f"{row['mean']:>6.3f}  {row['median']:>6.3f}  {row['std']:>5.3f}  {row['min']:>5.2f}  "
            f"{row['max']:>5.2f}  {row['q05']:>5.2f}  {row['q95']:>5.2f}"
        )

    lines.extend(["", "Target drift", ""])
    for _, row in diagnostics["target_drift"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: mean {row['train_mean']:.3f} -> "
            f"{row['validation_mean']:.3f} ({row['mean_change']:+.3f}, "
            f"smd {row['standardized_mean_difference']:+.3f}), "
            f"variance ratio {row['variance_ratio']:.3f}, KS {row['ks_statistic']:.3f}, "
            f"PSI {row['psi']:.3f}, skew change {row['skewness_change']:+.3f}"
        )

    lines.extend(["", "Predictor drift, ranked by mean absolute standardized difference", ""])
    ranking = diagnostics["predictor_drift_ranking"]
    lines.append(
        f"  {'predictor':<38}  {'|smd|':>6}  {'smd':>7}  {'KS':>5}  {'PSI':>5}  "
        f"{'varratio':>8}  {'folds>=0.25':>11}  {'same dir':>8}"
    )
    for _, row in ranking.head(12).iterrows():
        lines.append(
            f"  {row['predictor']:<38}  {row['mean_absolute_smd']:>6.3f}  {row['mean_smd']:>+7.3f}  "
            f"{row['mean_ks']:>5.3f}  {row['mean_psi']:>5.3f}  {row['mean_variance_ratio']:>8.3f}  "
            f"{int(row['n_folds_material_smd']):>11}  "
            f"{str(bool(row['same_direction_in_all_folds'])):>8}"
        )
    material = ranking[ranking["mean_absolute_smd"] >= SMD_MATERIAL]
    lines.append(
        f"  {len(material)} of {len(ranking)} predictors reach a mean |smd| of {SMD_MATERIAL}"
    )

    lines.extend(["", "Predictor to target relationship stability", ""])
    stability = diagnostics["relationship_stability"]
    counts = stability["classification"].value_counts()
    lines.append(
        "  " + ", ".join(f"{label}: {int(count)}" for label, count in counts.items())
    )
    lines.append(
        f"  {'predictor':<38}  {'train rho':>9}  {'val rho':>8}  {'|change|':>8}  "
        f"{'reversed':>8}  {'class':<13}"
    )
    for _, row in stability.head(12).iterrows():
        lines.append(
            f"  {row['predictor']:<38}  {row['mean_train_spearman']:>+9.3f}  "
            f"{row['mean_validation_spearman']:>+8.3f}  {row['mean_absolute_change']:>8.3f}  "
            f"{int(row['n_folds_reversed']):>8}  {row['classification']:<13}"
        )

    lines.extend(["", "Residual summary", ""])
    for _, row in diagnostics["residual_summary"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: mean {row['residual_mean']:+.4f}, "
            f"sd {row['residual_std']:.4f}, MAE {row['mae']:.4f}, RMSE {row['rmse']:.4f}, "
            f"R2 {row['r2']:+.4f}  (prediction sd {row['prediction_std']:.4f} vs "
            f"observed sd {row['observed_std']:.4f})"
        )

    lines.extend(["", "Error by chronological subperiod", ""])
    for _, row in diagnostics["residual_subperiods"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} block {int(row['subperiod'])} "
            f"({int(row['n']):>3} h, {row['start'].date()} to {row['end'].date()}): "
            f"RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}, "
            f"residual mean {row['residual_mean']:+.4f}, observed mean {row['observed_mean']:.3f}"
        )

    lines.extend(["", "Error trend and what error tracks (Spearman)", ""])
    for _, row in diagnostics["residual_trend"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: position vs |residual| "
            f"{row['position_vs_absolute_residual_spearman']:+.3f}, position vs residual "
            f"{row['position_vs_residual_spearman']:+.3f}, observed vs |residual| "
            f"{row['observed_vs_absolute_residual_spearman']:+.3f}, observed vs predicted "
            f"{row['observed_vs_predicted_spearman']:+.3f}"
        )

    lines.extend(["", "Error near recording gaps", ""])
    for _, row in diagnostics["residuals_near_gaps"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['group']:<11}: n {int(row['n']):>4}, "
            f"RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}"
        )

    lines.extend(["", "High error hours versus the rest of the window", ""])
    for _, row in diagnostics["high_error_target_profile"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['group']:<10}: n {int(row['n']):>4}, "
            f"observed {row['observed_mean']:.3f} (sd {row['observed_std']:.3f}), "
            f"predicted {row['predicted_mean']:.3f} (sd {row['predicted_std']:.3f}), "
            f"MAE {row['mae']:.4f}"
        )

    lines.extend(["", "Sensor conditions on high error hours, ranked by mean magnitude", ""])
    regime = diagnostics["operating_regime_ranking"]
    lines.append(f"  {'predictor':<38}  {'mean |diff|':>11}  {'signed':>7}  {'same dir':>8}")
    for _, row in regime.head(10).iterrows():
        lines.append(
            f"  {row['predictor']:<38}  {row['mean_absolute_difference']:>11.3f}  "
            f"{row['mean_standardized_difference']:>+7.3f}  "
            f"{str(bool(row['same_direction_in_all_folds'])):>8}"
        )
    consistent = regime[regime["same_direction_in_all_folds"]]
    lines.append(
        f"  {len(consistent)} of {len(regime)} predictors separate high error hours in the same "
        f"direction in all folds; largest mean magnitude is "
        f"{regime['mean_absolute_difference'].max():.3f} standard deviations"
    )

    lines.extend(["", "Baseline decomposition", ""])
    for _, row in diagnostics["baseline_decomposition"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: validation sd {row['validation_std']:.4f}, "
            f"model RMSE {row['model_rmse']:.4f}, ratio "
            f"{row['rmse_over_validation_std']:.4f} -> R2 {row['r2_model']:+.4f}"
        )
        lines.append(
            f"    SST {row['total_sum_of_squares']:.1f}, model SSE {row['model_sse']:.1f}, "
            f"training mean SSE {row['training_mean_sse']:.1f} "
            f"(R2 {row['r2_training_mean']:+.4f})"
        )
        lines.append(
            f"    mean drift {row['mean_drift']:+.4f}, drift penalty "
            f"{row['training_mean_drift_penalty']:.1f} "
            f"({row['drift_share_of_training_mean_sse']:.1%} of the constant baseline error); "
            f"model bias {row['residual_bias']:+.4f} "
            f"({row['bias_share_of_mse']:.1%} of model MSE)"
        )

    lines.extend(["", "Hours near an interpolated assay versus hours away from one", ""])
    for _, row in diagnostics["assay_gap_context"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['group']:<13}: n {int(row['n']):>4}, "
            f"observed {row['observed_mean']:.3f} (sd {row['observed_std']:.3f}), "
            f"RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}, "
            f"sensor divergence {row['sensor_divergence']:.3f}"
        )

    return "\n".join(lines)


def run_diagnostics(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    committed_results: pd.DataFrame | None = None,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
) -> dict:
    """Produce every diagnostic table from one controlled refit."""
    predictors = get_sensor_predictors()

    residuals = collect_residuals(
        spark, hourly, assignment, fold_ids=fold_ids, committed_results=committed_results
    )
    assert_no_final_test_contamination(residuals, assignment, hourly)

    annotated = annotate_residual_context(residuals, hourly)
    labelled = label_high_error(annotated)
    predictor_table = predictor_diagnostics(hourly, assignment, predictors, fold_ids=fold_ids)
    regime_profile = operating_regime_profile(labelled, hourly, predictors)

    return {
        "predictors": predictors,
        "residuals": annotated,
        "target_distributions": target_distributions(hourly, assignment, fold_ids=fold_ids),
        "target_drift": target_drift(hourly, assignment, fold_ids=fold_ids),
        "predictor_diagnostics": predictor_table,
        "predictor_drift_ranking": rank_predictor_drift(predictor_table),
        "relationship_stability": classify_relationship_stability(predictor_table),
        "residual_summary": residual_summary(residuals),
        "residual_subperiods": residual_subperiods(residuals),
        "residual_trend": residual_trend(residuals),
        "residuals_near_gaps": residuals_near_gaps(annotated),
        "high_error_target_profile": high_error_target_profile(labelled),
        "operating_regime_profile": regime_profile,
        "operating_regime_ranking": rank_operating_regime(regime_profile),
        "baseline_decomposition": baseline_decomposition(residuals, hourly, assignment),
        "assay_gap_context": assay_gap_context(annotated, hourly, predictors),
    }


def run(
    hourly_path: Path,
    splits_path: Path,
    random_forest_path: Path,
    results_path: Path,
    spark=None,
) -> dict:
    """Run every diagnostic and write the per predictor artifact."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityGeneralizationDiagnostics")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)

        if not random_forest_path.exists():
            raise FileNotFoundError(
                f"Random Forest benchmark results not found at: {random_forest_path.name}. "
                "Run that module first."
            )
        committed = pd.read_parquet(random_forest_path)

        diagnostics = run_diagnostics(spark, hourly, assignment, committed_results=committed)

        # The per fold per predictor table is the one artifact worth
        # persisting: it is the largest, and the only one another analysis
        # would plausibly read rather than recompute.
        artifact = diagnostics["predictor_diagnostics"].merge(
            diagnostics["operating_regime_profile"].rename(
                columns={"standardized_difference": "high_error_standardized_difference"}
            ),
            on=["fold_id", "predictor"],
            how="left",
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        artifact.to_parquet(results_path, index=False)

        diagnostics["artifact"] = artifact
        return diagnostics
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and running generalization diagnostics...")
    diagnostics = run(
        default_hourly_path(repo_root),
        default_splits_path(repo_root),
        default_random_forest_path(repo_root),
        results_path,
    )
    print(f"Per predictor diagnostics written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(diagnostics))


if __name__ == "__main__":
    sys.exit(main())
