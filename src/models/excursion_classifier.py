"""High silica excursion classification under chronological validation.

The regression experiments did not support reliable forward prediction
of exact silica concentration. The persisted feed chemistry regression
results show that adding `iron_feed` and `silica_feed` increased RMSE in
all three development folds. The feed experiment also calculated a high
silica subgroup analysis at runtime, but that table is not preserved in
the committed result artifact and is not treated here as independently
reproducible evidence.

This module tests that reformulation directly. The target becomes a
binary label, high silica excursion or not, and the question becomes
whether the available inputs can tell those hours apart. Two predictor
configurations are compared:

* the sensor only classifier, the existing 57 core sensor aggregates
* the feed enhanced classifier, those same 57 plus `iron_feed` and
  `silica_feed`

The persisted classifier results show that feed chemistry improved ROC
AUC in all three folds and PR AUC in two. The result remained weak and
inconsistent and did not produce a usable fixed threshold warning rule.

Label construction
------------------
The threshold is the 90th percentile of the target within each fold's own
training window, and that same number is applied unchanged to the
validation window. Nothing about the label is derived from validation
data, from the final test period, or from the full series. The threshold
is a statistical description of that training period, not a documented
plant safety or quality specification, and it should never be presented
as one.

Metrics
-------
Roughly one hour in ten is positive by construction, so accuracy is not
informative and is reported only as secondary context. Precision, recall,
F1, and the confusion matrix are computed at a fixed decision threshold
of 0.50 on the predicted positive class probability. That threshold is
fixed in advance and is never optimized against validation data. Ranking
quality is reported separately as average precision and ROC AUC, both
computed from the probabilities rather than from the thresholded
decision, and both compared against the no skill references.

The final test period is never fitted on, never scored, and never used to
interpret a result.
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
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    SENSOR_ELIGIBLE_COLUMN,
)
from src.models import random_forest
from src.models.baselines import (
    DEVELOPMENT_FOLD_IDS,
    INTERPOLATED_COLUMN,
    get_fold_frames,
    load_inputs,
)
from src.models.feed_chemistry_value import (
    FEED_ENHANCED,
    SENSOR_ONLY,
    feed_eligible_timestamps,
    get_feed_enhanced_predictors,
    get_sensor_only_predictors,
    restrict_assignment,
    summarize_row_loss,
    validate_configuration_scope,
)
from src.models.generalization_diagnostics import ks_statistic
from src.models.linear_regression import NUMERICAL_TOLERANCE, build_spark_session

CONFIGURATIONS = (SENSOR_ONLY, FEED_ENHANCED)

LABEL_COLUMN = "is_high_silica"
PROBABILITY_COLUMN = "probability"
POSITIVE_PROBABILITY_COLUMN = "positive_probability"
FEATURES_COLUMN = "features"

# ---------------------------------------------------------------------
# Fixed label and decision configuration
# ---------------------------------------------------------------------

# Quantile of each fold's own training target that defines an excursion.
# Chosen before any classification result was observed and never adjusted.
EXCURSION_QUANTILE = 0.90

# Fixed decision threshold on the predicted positive class probability.
# An hour is flagged when that probability is at least this value. The
# threshold is set in advance and is never tuned against validation data.
DECISION_THRESHOLD = 0.50

# ---------------------------------------------------------------------
# Fixed classifier configuration
# ---------------------------------------------------------------------
#
# The ensemble settings are imported from the committed regression
# benchmark rather than restated, so the two cannot drift apart. Each one
# means the same thing for a classification forest as for a regression
# forest: how many trees to average, how deep each may grow, how few rows
# a split may isolate, how many features each split may consider, and how
# much of the training set each tree sees.
#
# `featureSubsetStrategy="sqrt"` deserves a note. Spark resolves its
# "auto" default to one third of the features for regression and to the
# square root for classification. The benchmark set "sqrt" explicitly, so
# carrying it across keeps the literal setting and also lands on the
# conventional classification choice.
NUM_TREES = random_forest.NUM_TREES
MAX_DEPTH = random_forest.MAX_DEPTH
MIN_INSTANCES_PER_NODE = random_forest.MIN_INSTANCES_PER_NODE
FEATURE_SUBSET_STRATEGY = random_forest.FEATURE_SUBSET_STRATEGY
SUBSAMPLING_RATE = random_forest.SUBSAMPLING_RATE
SEED = random_forest.SEED

# The one setting that cannot be carried across. The regression benchmark
# splits on variance reduction, which is undefined for a class label.
# Gini impurity is Spark's classification default and the conventional
# choice; it is declared here rather than left implicit. No other value
# was evaluated, because this experiment performs no search.
IMPURITY = "gini"

# No class weighting is applied. Spark's `weightCol` would let the rare
# positive class be upweighted, which changes the decision boundary and is
# a form of tuning. Leaving it unset keeps the classifier untouched and
# keeps the comparison between the two arms clean.
CLASS_WEIGHTING = None

# ---------------------------------------------------------------------
# Reporting thresholds for the decision framework
# ---------------------------------------------------------------------
#
# All three are applied after measurement and are reported alongside the
# raw numbers so a reader can substitute their own. None is an input to
# any model.

# How far average precision must sit above the no skill prevalence before
# the ranking is treated as carrying real information.
MEANINGFUL_PR_AUC_LIFT = 0.05

# How far ROC AUC must sit above the 0.5 coin flip reference.
MEANINGFUL_ROC_AUC = 0.60

# Operational floors for a warning system that an engineer would act on.
MINIMUM_USEFUL_RECALL = 0.50
MINIMUM_USEFUL_PRECISION = 0.30

OUTCOME_USEFUL = "useful forward excursion classifier"
OUTCOME_WEAK = "weak or inconsistent classifier"
OUTCOME_NONE = "no useful forward classifier"

RESULT_COLUMNS = [
    "configuration",
    "fold_id",
    "n_train",
    "n_validation",
    "n_features",
    "threshold",
    "n_train_positive",
    "train_positive_rate",
    "n_validation_positive",
    "validation_positive_rate",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "false_negative_rate",
    "false_positive_rate",
    "accuracy",
]


@dataclass(frozen=True)
class ClassifierFoldResult:
    configuration: str
    fold_id: int
    threshold: float
    metrics: dict[str, float]
    n_train: int
    n_validation: int
    n_features: int
    n_train_positive: int
    n_validation_positive: int
    feature_importances: np.ndarray
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ClassifierEvaluation:
    configuration: str
    predictors: list[str]
    results: pd.DataFrame
    fold_results: list[ClassifierFoldResult]


# ---------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------


def training_excursion_threshold(
    train: pd.DataFrame, quantile: float = EXCURSION_QUANTILE
) -> float:
    """The high silica threshold, taken from the training window only.

    Uses the linear interpolation quantile pandas applies by default,
    which is deterministic for a fixed sample. Nothing from validation or
    from the final test period contributes.
    """
    if train.empty:
        raise ValueError("Cannot derive an excursion threshold from an empty training window.")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"Excursion quantile must be strictly between 0 and 1, got {quantile}.")
    return float(train[TARGET_COLUMN].quantile(quantile))


def apply_excursion_label(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Attach the binary excursion label using a supplied threshold.

    An hour is positive when its observed target is at or above the
    threshold. The same threshold value is used for training and
    validation, so the label means the same thing in both windows.
    """
    labelled = frame.copy()
    labelled[LABEL_COLUMN] = (labelled[TARGET_COLUMN] >= threshold).astype(int)
    return labelled


@dataclass(frozen=True)
class LabelledFold:
    fold_id: int
    threshold: float
    train: pd.DataFrame
    validation: pd.DataFrame
    embargo: pd.DataFrame


def build_labelled_fold(
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_id: int,
    quantile: float = EXCURSION_QUANTILE,
) -> LabelledFold:
    """Label one development fold from its own training distribution."""
    frames = get_fold_frames(hourly, assignment, fold_id)
    threshold = training_excursion_threshold(frames.train, quantile)
    return LabelledFold(
        fold_id=fold_id,
        threshold=threshold,
        train=apply_excursion_label(frames.train, threshold),
        validation=apply_excursion_label(frames.validation, threshold),
        embargo=frames.embargo,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def _as_label_and_score(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"Score shape {scores.shape} does not match label shape {labels.shape}.")
    if labels.size == 0:
        raise ValueError("Cannot compute a metric over zero observations.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    return labels, scores


def confusion_counts(y_true, y_score, threshold: float = DECISION_THRESHOLD) -> dict[str, int]:
    """Confusion matrix counts at a fixed probability threshold.

    An hour is flagged when its predicted positive probability is at or
    above `threshold`, rather than by taking the larger of the two class
    probabilities, so the decision rule is explicit and a probability of
    exactly 0.5 resolves the same way every time.
    """
    labels, scores = _as_label_and_score(y_true, y_score)
    flagged = scores >= threshold
    positive = labels == 1
    return {
        "true_negatives": int(np.sum(~flagged & ~positive)),
        "false_positives": int(np.sum(flagged & ~positive)),
        "false_negatives": int(np.sum(~flagged & positive)),
        "true_positives": int(np.sum(flagged & positive)),
    }


def average_ranks(values) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing their average rank."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    ranks = np.empty(array.size, dtype=float)

    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve, computed from the rank statistic.

    Equivalent to the probability that a randomly chosen positive hour
    receives a higher score than a randomly chosen negative hour, with
    tied scores counting as half. Raises when either class is absent,
    because the quantity is undefined there rather than zero.
    """
    labels, scores = _as_label_and_score(y_true, y_score)
    n_positive = int(np.sum(labels == 1))
    n_negative = labels.size - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC AUC is undefined when the validation window holds only one class.")

    ranks = average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def average_precision(y_true, y_score) -> float:
    """Average precision, the standard summary of the precision recall curve.

    Computed as the sum of precision at each distinct score threshold,
    weighted by the increase in recall it produces. This step form is used
    rather than a trapezoidal integration because interpolating between
    precision recall points overstates performance. Tied scores are
    resolved together, so two hours with the same probability cannot be
    separated by ordering luck.
    """
    labels, scores = _as_label_and_score(y_true, y_score)
    n_positive = int(np.sum(labels == 1))
    if n_positive == 0:
        raise ValueError("Average precision is undefined without a positive observation.")

    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]

    true_positives = np.cumsum(ordered_labels == 1)
    predicted_positives = np.arange(1, labels.size + 1)

    # Keep only the last index of each run of equal scores: a threshold
    # cannot separate observations that share a score.
    distinct = np.append(ordered_scores[1:] != ordered_scores[:-1], True)

    precision = true_positives[distinct] / predicted_positives[distinct]
    recall = true_positives[distinct] / n_positive
    recall_gain = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_gain * precision))


def classification_metrics(
    y_true, y_score, threshold: float = DECISION_THRESHOLD
) -> dict[str, float]:
    """Every reported classification metric for one scored window.

    Precision is defined as 0.0 when nothing is flagged. That case is a
    real outcome for a rare positive class, and reporting it as zero keeps
    it visible instead of turning it into a missing value.
    """
    labels, scores = _as_label_and_score(y_true, y_score)
    counts = confusion_counts(labels, scores, threshold)

    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    true_negatives = counts["true_negatives"]

    flagged = true_positives + false_positives
    actual_positive = true_positives + false_negatives
    actual_negative = true_negatives + false_positives

    precision = float(true_positives / flagged) if flagged else 0.0
    recall = float(true_positives / actual_positive) if actual_positive else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        **{key: float(value) for key, value in counts.items()},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "false_negative_rate": float(false_negatives / actual_positive) if actual_positive else 0.0,
        "false_positive_rate": float(false_positives / actual_negative) if actual_negative else 0.0,
        "accuracy": float((true_positives + true_negatives) / labels.size),
    }


# ---------------------------------------------------------------------
# Spark pipeline
# ---------------------------------------------------------------------


def build_classifier_pipeline(
    predictors: list[str],
    num_trees: int = NUM_TREES,
    max_depth: int = MAX_DEPTH,
    min_instances_per_node: int = MIN_INSTANCES_PER_NODE,
    feature_subset_strategy: str = FEATURE_SUBSET_STRATEGY,
    subsampling_rate: float = SUBSAMPLING_RATE,
    impurity: str = IMPURITY,
    seed: int = SEED,
):
    """Assemble `VectorAssembler` then `RandomForestClassifier`, unfitted.

    No scaling stage, for the same reason the regression benchmark has
    none: a tree split is a threshold comparison on one feature, so a
    monotonic rescaling cannot change which split a tree chooses.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(
        inputCols=predictors, outputCol=FEATURES_COLUMN, handleInvalid="error"
    )
    forest = RandomForestClassifier(
        featuresCol=FEATURES_COLUMN,
        labelCol=LABEL_COLUMN,
        probabilityCol=PROBABILITY_COLUMN,
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances_per_node,
        featureSubsetStrategy=feature_subset_strategy,
        subsamplingRate=subsampling_rate,
        impurity=impurity,
        seed=seed,
    )
    return Pipeline(stages=[assembler, forest])


def to_labelled_spark(spark, frame: pd.DataFrame, predictors: list[str]):
    """Convert a labelled fold frame to Spark with only the needed columns."""
    columns = [TIMESTAMP_COLUMN, TARGET_COLUMN, LABEL_COLUMN] + predictors
    subset = frame[columns].copy()
    for column in [TARGET_COLUMN] + predictors:
        subset[column] = subset[column].astype(float)
    subset[LABEL_COLUMN] = subset[LABEL_COLUMN].astype(float)
    return spark.createDataFrame(subset)


# ---------------------------------------------------------------------
# Fold training and scoring
# ---------------------------------------------------------------------


def evaluate_fold(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    fold_id: int,
    configuration: str,
    predictors: list[str] | None = None,
    quantile: float = EXCURSION_QUANTILE,
    threshold: float = DECISION_THRESHOLD,
) -> ClassifierFoldResult:
    """Fit the classifier on one fold's training rows and score its validation rows."""
    if fold_id == FINAL_TEST_FOLD_ID:
        raise ValueError("The final test fold must not be evaluated in this experiment.")

    if predictors is None:
        predictors = (
            get_sensor_only_predictors()
            if configuration == SENSOR_ONLY
            else get_feed_enhanced_predictors()
        )
    else:
        predictors = list(predictors)
    validate_configuration_scope(configuration, predictors)

    labelled = build_labelled_fold(hourly, assignment, fold_id, quantile)

    if labelled.train[LABEL_COLUMN].nunique() < 2:
        raise ValueError(f"Fold {fold_id}: the training window holds only one class.")
    if labelled.validation[LABEL_COLUMN].nunique() < 2:
        raise ValueError(f"Fold {fold_id}: the validation window holds only one class.")

    train_spark = to_labelled_spark(spark, labelled.train, predictors)
    validation_spark = to_labelled_spark(spark, labelled.validation, predictors)

    if train_spark.count() != len(labelled.train):
        raise ValueError(f"Fold {fold_id}: training row count changed during Spark conversion.")
    if validation_spark.count() != len(labelled.validation):
        raise ValueError(f"Fold {fold_id}: validation row count changed during Spark conversion.")

    fitted = build_classifier_pipeline(predictors).fit(train_spark)

    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    scored = fitted.transform(validation_spark).withColumn(
        POSITIVE_PROBABILITY_COLUMN, vector_to_array(F.col(PROBABILITY_COLUMN))[1]
    )
    predictions = (
        scored.select(
            TIMESTAMP_COLUMN, TARGET_COLUMN, LABEL_COLUMN, POSITIVE_PROBABILITY_COLUMN
        )
        .toPandas()
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    predictions[LABEL_COLUMN] = predictions[LABEL_COLUMN].astype(int)

    if len(predictions) != len(labelled.validation):
        raise ValueError(
            f"Fold {fold_id}: produced {len(predictions)} scores for "
            f"{len(labelled.validation)} validation rows."
        )

    probabilities = predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Fold {fold_id}: predicted probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"Fold {fold_id}: predicted probabilities fall outside [0, 1].")

    metrics = classification_metrics(
        predictions[LABEL_COLUMN].to_numpy(), probabilities, threshold
    )
    importances = np.asarray(fitted.stages[-1].featureImportances.toArray(), dtype=float)

    return ClassifierFoldResult(
        configuration=configuration,
        fold_id=fold_id,
        threshold=labelled.threshold,
        metrics=metrics,
        n_train=len(labelled.train),
        n_validation=len(labelled.validation),
        n_features=len(predictors),
        n_train_positive=int(labelled.train[LABEL_COLUMN].sum()),
        n_validation_positive=int(labelled.validation[LABEL_COLUMN].sum()),
        feature_importances=importances,
        predictions=predictions,
    )


def evaluate_configuration(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: str,
    fold_ids: tuple[int, ...] = DEVELOPMENT_FOLD_IDS,
    quantile: float = EXCURSION_QUANTILE,
    threshold: float = DECISION_THRESHOLD,
) -> ClassifierEvaluation:
    """Evaluate one predictor configuration across every development fold."""
    predictors = (
        get_sensor_only_predictors()
        if configuration == SENSOR_ONLY
        else get_feed_enhanced_predictors()
    )
    fold_results = [
        evaluate_fold(
            spark,
            hourly,
            assignment,
            fold_id,
            configuration,
            predictors=predictors,
            quantile=quantile,
            threshold=threshold,
        )
        for fold_id in fold_ids
    ]

    results = pd.DataFrame(
        [
            {
                "configuration": result.configuration,
                "fold_id": result.fold_id,
                "n_train": result.n_train,
                "n_validation": result.n_validation,
                "n_features": result.n_features,
                "threshold": result.threshold,
                "n_train_positive": result.n_train_positive,
                "train_positive_rate": result.n_train_positive / result.n_train,
                "n_validation_positive": result.n_validation_positive,
                "validation_positive_rate": result.n_validation_positive / result.n_validation,
                **{
                    key: result.metrics[key]
                    for key in (
                        "precision",
                        "recall",
                        "f1",
                        "pr_auc",
                        "roc_auc",
                        "true_negatives",
                        "false_positives",
                        "false_negatives",
                        "true_positives",
                        "false_negative_rate",
                        "false_positive_rate",
                        "accuracy",
                    )
                },
            }
            for result in fold_results
        ],
        columns=RESULT_COLUMNS,
    )
    for column in (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ):
        results[column] = results[column].astype(int)

    return ClassifierEvaluation(
        configuration=configuration,
        predictors=predictors,
        results=results,
        fold_results=fold_results,
    )


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------


def validate_evaluation(
    evaluation: ClassifierEvaluation,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    committed_assignment: pd.DataFrame,
) -> None:
    """Verify every structural guard, raising a clear error on violation."""
    if list(evaluation.results.columns) != RESULT_COLUMNS:
        raise ValueError(
            f"Result schema mismatch. Expected {RESULT_COLUMNS}, "
            f"got {list(evaluation.results.columns)}"
        )

    validate_configuration_scope(evaluation.configuration, evaluation.predictors)
    expected_features = len(evaluation.predictors)
    uses_feed = evaluation.configuration == FEED_ENHANCED

    eligible = set(hourly.loc[hourly[SENSOR_ELIGIBLE_COLUMN], TIMESTAMP_COLUMN])
    feed_eligible = feed_eligible_timestamps(hourly)
    interpolated = set(hourly.loc[hourly[INTERPOLATED_COLUMN].astype(bool), TIMESTAMP_COLUMN])
    final_test = set(
        committed_assignment.loc[committed_assignment["role"] == ROLE_TEST, TIMESTAMP_COLUMN]
    )
    committed_pairs = set(
        zip(
            committed_assignment["fold_id"],
            committed_assignment[TIMESTAMP_COLUMN],
            committed_assignment["role"],
        )
    )

    for result in evaluation.fold_results:
        fold_id = result.fold_id
        labelled = build_labelled_fold(hourly, assignment, fold_id)

        train_timestamps = set(labelled.train[TIMESTAMP_COLUMN])
        validation_timestamps = set(labelled.validation[TIMESTAMP_COLUMN])
        embargo_timestamps = set(labelled.embargo[TIMESTAMP_COLUMN])

        for role, timestamps in (
            (ROLE_TRAIN, train_timestamps),
            (ROLE_VALIDATION, validation_timestamps),
        ):
            invented = {
                timestamp
                for timestamp in timestamps
                if (fold_id, timestamp, role) not in committed_pairs
            }
            if invented:
                raise ValueError(
                    f"Fold {fold_id}: {len(invented)} {role} row(s) were not committed to that role."
                )

        if not train_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: training contains ineligible hours.")
        if not validation_timestamps.issubset(eligible):
            raise ValueError(f"Fold {fold_id}: validation contains ineligible hours.")
        if uses_feed:
            if not train_timestamps.issubset(feed_eligible):
                raise ValueError(f"Fold {fold_id}: training contains feed ineligible hours.")
            if not validation_timestamps.issubset(feed_eligible):
                raise ValueError(f"Fold {fold_id}: validation contains feed ineligible hours.")

        if train_timestamps.intersection(interpolated):
            raise ValueError(f"Fold {fold_id}: training contains an interpolated target row.")
        if validation_timestamps.intersection(interpolated):
            raise ValueError(f"Fold {fold_id}: validation contains an interpolated target row.")

        if train_timestamps.intersection(validation_timestamps):
            raise ValueError(f"Fold {fold_id}: training and validation timestamps overlap.")
        if train_timestamps.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was used for fitting.")
        if train_timestamps.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: training overlaps the final test period.")
        if max(train_timestamps) >= min(validation_timestamps):
            raise ValueError(f"Fold {fold_id}: training is not strictly earlier than validation.")

        scored = set(result.predictions[TIMESTAMP_COLUMN])
        if scored != validation_timestamps:
            raise ValueError(f"Fold {fold_id}: scored hours differ from the validation window.")
        if scored.intersection(embargo_timestamps):
            raise ValueError(f"Fold {fold_id}: an embargo hour was scored.")
        if scored.intersection(final_test):
            raise ValueError(f"Fold {fold_id}: a final test hour was scored.")
        if not result.predictions[TIMESTAMP_COLUMN].is_monotonic_increasing:
            raise ValueError(f"Fold {fold_id}: scores are not in chronological order.")

        # The threshold must be reproducible from the training window alone.
        recomputed = training_excursion_threshold(labelled.train)
        if not np.isclose(recomputed, result.threshold):
            raise ValueError(
                f"Fold {fold_id}: the recorded threshold does not match the training window."
            )

        # The validation labels must follow that same threshold unchanged.
        expected_labels = (
            result.predictions[TARGET_COLUMN].to_numpy(dtype=float) >= result.threshold
        ).astype(int)
        if not np.array_equal(result.predictions[LABEL_COLUMN].to_numpy(), expected_labels):
            raise ValueError(
                f"Fold {fold_id}: validation labels do not follow the training derived threshold."
            )

        probabilities = result.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(dtype=float)
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError(f"Fold {fold_id}: a predicted probability falls outside [0, 1].")

        counts = sum(
            result.metrics[key]
            for key in ("true_negatives", "false_positives", "false_negatives", "true_positives")
        )
        if int(counts) != result.n_validation:
            raise ValueError(
                f"Fold {fold_id}: confusion matrix counts sum to {int(counts)}, "
                f"not {result.n_validation}."
            )

        if result.n_features != expected_features:
            raise ValueError(
                f"Fold {fold_id}: model used {result.n_features} features, not {expected_features}."
            )
        for metric, value in result.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"Fold {fold_id}: {metric} is not finite.")

        importances = result.feature_importances
        if len(importances) != expected_features:
            raise ValueError(
                f"Fold {fold_id}: importance vector has {len(importances)} entries, "
                f"not {expected_features}."
            )
        if (importances < 0).any() or not np.isfinite(importances).all():
            raise ValueError(f"Fold {fold_id}: feature importances are invalid.")


def assert_matched_rows_and_labels(
    sensor_only: ClassifierEvaluation, feed_enhanced: ClassifierEvaluation
) -> None:
    """Confirm both arms saw identical rows, labels, and thresholds."""
    left_by_fold = {result.fold_id: result for result in sensor_only.fold_results}
    right_by_fold = {result.fold_id: result for result in feed_enhanced.fold_results}

    if set(left_by_fold) != set(right_by_fold):
        raise ValueError("The two configurations do not cover the same folds.")

    for fold_id in sorted(left_by_fold):
        left, right = left_by_fold[fold_id], right_by_fold[fold_id]
        if left.n_train != right.n_train:
            raise ValueError(
                f"Fold {fold_id}: training rows differ between configurations "
                f"({left.n_train} against {right.n_train})."
            )
        if left.n_train_positive != right.n_train_positive:
            raise ValueError(f"Fold {fold_id}: training positive counts differ.")
        if not np.isclose(left.threshold, right.threshold):
            raise ValueError(f"Fold {fold_id}: the two configurations used different thresholds.")
        if set(left.predictions[TIMESTAMP_COLUMN]) != set(right.predictions[TIMESTAMP_COLUMN]):
            raise ValueError(f"Fold {fold_id}: the two configurations scored different hours.")
        if not np.array_equal(
            left.predictions[LABEL_COLUMN].to_numpy(), right.predictions[LABEL_COLUMN].to_numpy()
        ):
            raise ValueError(f"Fold {fold_id}: the two configurations saw different labels.")


def verify_classifier_configuration(predictors: list[str]) -> dict:
    """Read the fixed classifier settings back off a freshly built pipeline."""
    forest = build_classifier_pipeline(predictors).getStages()[-1]
    observed = {
        "numTrees": forest.getNumTrees(),
        "maxDepth": forest.getMaxDepth(),
        "minInstancesPerNode": forest.getMinInstancesPerNode(),
        "featureSubsetStrategy": forest.getFeatureSubsetStrategy(),
        "subsamplingRate": forest.getSubsamplingRate(),
        "impurity": forest.getImpurity(),
        "seed": forest.getSeed(),
    }
    expected = {
        "numTrees": NUM_TREES,
        "maxDepth": MAX_DEPTH,
        "minInstancesPerNode": MIN_INSTANCES_PER_NODE,
        "featureSubsetStrategy": FEATURE_SUBSET_STRATEGY,
        "subsamplingRate": SUBSAMPLING_RATE,
        "impurity": IMPURITY,
        "seed": SEED,
    }
    if observed != expected:
        raise ValueError(
            f"The classifier configuration has changed. Expected {expected}, got {observed}."
        )
    return observed


def verify_deterministic_evaluation(
    spark,
    hourly: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: str,
    fold_id: int = DEVELOPMENT_FOLD_IDS[0],
    tolerance: float = NUMERICAL_TOLERANCE,
) -> None:
    """Fit one fold twice and confirm the results agree within tolerance."""
    first = evaluate_fold(spark, hourly, assignment, fold_id, configuration)
    second = evaluate_fold(spark, hourly, assignment, fold_id, configuration)

    for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
        if abs(first.metrics[metric] - second.metrics[metric]) > tolerance:
            raise ValueError(
                f"{configuration} fold {fold_id}: {metric} is not reproducible "
                f"({first.metrics[metric]} against {second.metrics[metric]})."
            )
    if not np.allclose(
        first.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(),
        second.predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(),
        atol=tolerance,
    ):
        raise ValueError(f"{configuration} fold {fold_id}: probabilities are not reproducible.")
    if not np.allclose(first.feature_importances, second.feature_importances, atol=tolerance):
        raise ValueError(f"{configuration} fold {fold_id}: importances are not reproducible.")


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------


def baseline_table(evaluation: ClassifierEvaluation) -> pd.DataFrame:
    """Trivial references every classifier must beat to be worth anything.

    `always_normal` flags nothing. `training_majority` flags according to
    the majority class of the fold's own training labels, which is
    computed rather than assumed. `no_skill` describes what a random
    ranking achieves: average precision equal to the positive prevalence,
    and ROC AUC of one half.
    """
    rows = []
    for result in evaluation.fold_results:
        labels = result.predictions[LABEL_COLUMN].to_numpy(dtype=int)
        n = labels.size
        n_positive = int(labels.sum())
        prevalence = n_positive / n

        train_majority = 1 if result.n_train_positive * 2 > result.n_train else 0

        rows.append(
            {
                "fold_id": result.fold_id,
                "baseline": "always_normal",
                "n_validation": n,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": float((n - n_positive) / n),
                "true_negatives": n - n_positive,
                "false_positives": 0,
                "false_negatives": n_positive,
                "true_positives": 0,
                "false_negative_rate": 1.0,
                "false_positive_rate": 0.0,
                "pr_auc": prevalence,
                "roc_auc": 0.5,
            }
        )

        if train_majority == 1:
            majority_metrics = {
                "precision": prevalence,
                "recall": 1.0,
                "f1": float(2 * prevalence / (prevalence + 1.0)),
                "accuracy": prevalence,
                "true_negatives": 0,
                "false_positives": n - n_positive,
                "false_negatives": 0,
                "true_positives": n_positive,
                "false_negative_rate": 0.0,
                "false_positive_rate": 1.0,
            }
        else:
            majority_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": float((n - n_positive) / n),
                "true_negatives": n - n_positive,
                "false_positives": 0,
                "false_negatives": n_positive,
                "true_positives": 0,
                "false_negative_rate": 1.0,
                "false_positive_rate": 0.0,
            }
        rows.append(
            {
                "fold_id": result.fold_id,
                "baseline": f"training_majority (class {train_majority})",
                "n_validation": n,
                **majority_metrics,
                "pr_auc": prevalence,
                "roc_auc": 0.5,
            }
        )

        rows.append(
            {
                "fold_id": result.fold_id,
                "baseline": "no_skill",
                "n_validation": n,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "accuracy": float("nan"),
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 0,
                "false_negative_rate": float("nan"),
                "false_positive_rate": float("nan"),
                "pr_auc": prevalence,
                "roc_auc": 0.5,
            }
        )

    return pd.DataFrame(rows)


def baseline_comparison(evaluation: ClassifierEvaluation) -> pd.DataFrame:
    """Lift of the classifier's ranking over the no skill references."""
    rows = []
    for result in evaluation.fold_results:
        labels = result.predictions[LABEL_COLUMN].to_numpy(dtype=int)
        prevalence = float(labels.sum() / labels.size)
        rows.append(
            {
                "configuration": evaluation.configuration,
                "fold_id": result.fold_id,
                "prevalence": prevalence,
                "pr_auc": result.metrics["pr_auc"],
                "pr_auc_lift": result.metrics["pr_auc"] - prevalence,
                "roc_auc": result.metrics["roc_auc"],
                "roc_auc_lift": result.metrics["roc_auc"] - 0.5,
                "beats_no_skill_pr": result.metrics["pr_auc"] - prevalence
                >= MEANINGFUL_PR_AUC_LIFT,
                "beats_no_skill_roc": result.metrics["roc_auc"] >= MEANINGFUL_ROC_AUC,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Probability separation
# ---------------------------------------------------------------------


def probability_distributions(evaluation: ClassifierEvaluation) -> pd.DataFrame:
    """Predicted probability distribution for each true class, per fold."""
    rows = []
    for result in evaluation.fold_results:
        predictions = result.predictions
        for label, name in ((1, "positive"), (0, "negative")):
            subset = predictions[predictions[LABEL_COLUMN] == label]
            if subset.empty:
                continue
            values = subset[POSITIVE_PROBABILITY_COLUMN].to_numpy(dtype=float)
            rows.append(
                {
                    "configuration": evaluation.configuration,
                    "fold_id": result.fold_id,
                    "true_class": name,
                    "n": values.size,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                    "max": float(values.max()),
                    "share_at_or_above_decision_threshold": float(
                        np.mean(values >= DECISION_THRESHOLD)
                    ),
                }
            )
    return pd.DataFrame(rows)


def probability_separation(evaluation: ClassifierEvaluation) -> pd.DataFrame:
    """How far apart the two classes' predicted probabilities actually sit.

    Reported as the difference in means, the same difference standardized
    by the pooled spread, and the Kolmogorov Smirnov distance between the
    two distributions. All three are effect sizes that do not grow with
    sample size. No significance test is reported, because with several
    hundred hours per fold almost any difference would be significant and
    that fact carries no information about whether the separation is
    usable.
    """
    rows = []
    for result in evaluation.fold_results:
        predictions = result.predictions
        positive = predictions.loc[
            predictions[LABEL_COLUMN] == 1, POSITIVE_PROBABILITY_COLUMN
        ].to_numpy(dtype=float)
        negative = predictions.loc[
            predictions[LABEL_COLUMN] == 0, POSITIVE_PROBABILITY_COLUMN
        ].to_numpy(dtype=float)
        if positive.size == 0 or negative.size == 0:
            continue

        pooled = np.sqrt((positive.var(ddof=0) + negative.var(ddof=0)) / 2.0)
        difference = float(positive.mean() - negative.mean())
        rows.append(
            {
                "configuration": evaluation.configuration,
                "fold_id": result.fold_id,
                "positive_mean": float(positive.mean()),
                "negative_mean": float(negative.mean()),
                "mean_difference": difference,
                "standardized_difference": float(difference / pooled) if pooled > 0 else 0.0,
                "ks_distance": ks_statistic(negative, positive),
                "max_probability": float(predictions[POSITIVE_PROBABILITY_COLUMN].max()),
                "share_above_decision_threshold": float(
                    np.mean(
                        predictions[POSITIVE_PROBABILITY_COLUMN].to_numpy(dtype=float)
                        >= DECISION_THRESHOLD
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------


def feed_importance_table(evaluation: ClassifierEvaluation) -> pd.DataFrame:
    """Importance and rank of each feed column within the fitted classifier.

    Rank 1 is the largest importance. Importance describes how often and
    how usefully the ensemble split on a feature given everything else it
    had available. It is not a causal effect and is not evidence that
    changing feed chemistry would change the outcome.
    """
    predictors = evaluation.predictors
    rows = []
    for result in evaluation.fold_results:
        importances = result.feature_importances
        order = np.argsort(-importances, kind="mergesort")
        rank_by_index = {int(index): position + 1 for position, index in enumerate(order)}

        for column in ("iron_feed", "silica_feed"):
            if column not in predictors:
                continue
            index = predictors.index(column)
            rows.append(
                {
                    "configuration": evaluation.configuration,
                    "fold_id": result.fold_id,
                    "feature": column,
                    "importance": float(importances[index]),
                    "rank": rank_by_index[index],
                    "n_features": len(predictors),
                }
            )
    return pd.DataFrame(rows)


def top_importance_table(evaluation: ClassifierEvaluation, top_n: int = 8) -> pd.DataFrame:
    """The largest importances per fold, for context around the feed ranks."""
    predictors = evaluation.predictors
    rows = []
    for result in evaluation.fold_results:
        importances = result.feature_importances
        order = np.argsort(-importances, kind="mergesort")[:top_n]
        for position, index in enumerate(order, start=1):
            rows.append(
                {
                    "configuration": evaluation.configuration,
                    "fold_id": result.fold_id,
                    "rank": position,
                    "feature": predictors[int(index)],
                    "importance": float(importances[int(index)]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Comparison and outcome
# ---------------------------------------------------------------------


def combine_results(
    sensor_only: ClassifierEvaluation, feed_enhanced: ClassifierEvaluation
) -> pd.DataFrame:
    combined = pd.concat([sensor_only.results, feed_enhanced.results], ignore_index=True)
    return combined.sort_values(
        ["configuration", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)


COMPARED_METRICS = ("recall", "false_negative_rate", "f1", "pr_auc", "roc_auc", "precision")


def compare_configurations(combined: pd.DataFrame) -> pd.DataFrame:
    """Per fold difference between the feed enhanced and sensor only arms.

    Every metric except the false negative rate improves when it rises.
    The false negative rate improves when it falls, and it is reported
    with its own sign so a reader is not left inferring the direction.
    """
    sensor = combined[combined["configuration"] == SENSOR_ONLY]
    feed = combined[combined["configuration"] == FEED_ENHANCED]
    merged = feed.merge(
        sensor[["fold_id", *COMPARED_METRICS]], on="fold_id", suffixes=("", "_sensor_only")
    )

    for metric in COMPARED_METRICS:
        merged[f"{metric}_difference"] = merged[metric] - merged[f"{metric}_sensor_only"]
    merged["feed_improves_recall"] = merged["recall_difference"] > 0
    merged["feed_improves_pr_auc"] = merged["pr_auc_difference"] > 0
    merged["feed_reduces_false_negative_rate"] = merged["false_negative_rate_difference"] < 0

    columns = (
        ["fold_id"]
        + [f"{metric}_sensor_only" for metric in COMPARED_METRICS]
        + list(COMPARED_METRICS)
        + [f"{metric}_difference" for metric in COMPARED_METRICS]
        + ["feed_improves_recall", "feed_improves_pr_auc", "feed_reduces_false_negative_rate"]
    )
    return merged[columns].sort_values("fold_id", kind="mergesort").reset_index(drop=True)


def summarize_configurations(combined: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each headline metric across folds."""
    summary = (
        combined.groupby("configuration")
        .agg(
            n_folds=("fold_id", "nunique"),
            n_features=("n_features", "max"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            false_negative_rate_mean=("false_negative_rate", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("configuration", kind="mergesort").reset_index(drop=True)


def classify_outcome(evaluation: ClassifierEvaluation) -> dict:
    """Apply the experiment's decision framework to one configuration.

    A classifier counts as useful only when every development fold clears
    all four reporting thresholds. It counts as providing nothing when no
    fold clears either ranking threshold, which is the case where the
    probabilities carry no more information than a coin flip. Everything
    in between is weak or inconsistent, which is a statement about
    stability rather than about the absence of signal.
    """
    comparison = baseline_comparison(evaluation)
    n_folds = len(comparison)
    if n_folds == 0:
        raise ValueError("Cannot classify an outcome from an empty evaluation.")

    by_fold = {result.fold_id: result for result in evaluation.fold_results}
    operational = [
        by_fold[int(row["fold_id"])].metrics["recall"] >= MINIMUM_USEFUL_RECALL
        and by_fold[int(row["fold_id"])].metrics["precision"] >= MINIMUM_USEFUL_PRECISION
        for _, row in comparison.iterrows()
    ]

    n_pr = int(comparison["beats_no_skill_pr"].sum())
    n_roc = int(comparison["beats_no_skill_roc"].sum())
    n_operational = int(sum(operational))

    if n_pr == n_folds and n_roc == n_folds and n_operational == n_folds:
        classification = OUTCOME_USEFUL
    elif n_pr == 0 and n_roc == 0:
        classification = OUTCOME_NONE
    else:
        classification = OUTCOME_WEAK

    return {
        "configuration": evaluation.configuration,
        "n_folds": n_folds,
        "n_folds_beating_no_skill_pr": n_pr,
        "n_folds_beating_no_skill_roc": n_roc,
        "n_folds_operationally_useful": n_operational,
        "mean_pr_auc": float(comparison["pr_auc"].mean()),
        "mean_pr_auc_lift": float(comparison["pr_auc_lift"].mean()),
        "mean_roc_auc": float(comparison["roc_auc"].mean()),
        "classification": classification,
    }


# ---------------------------------------------------------------------
# Paths, report, CLI
# ---------------------------------------------------------------------


def default_hourly_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_features.parquet"


def default_splits_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "hourly_splits.parquet"


def default_results_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / "excursion_classifier_results.parquet"


def format_report(report: dict) -> str:
    lines = ["High silica excursion classification (development folds only)", ""]

    lines.append("Fixed classifier configuration")
    for key, value in report["configuration_settings"].items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  decision threshold on P(excursion): {DECISION_THRESHOLD}")
    lines.append(f"  label quantile of the training target: {EXCURSION_QUANTILE}")

    lines.extend(["", "Rows removed because feed chemistry is unavailable", ""])
    dropped = report["row_loss"][report["row_loss"]["n_dropped"] > 0]
    if dropped.empty:
        lines.append("  none: every sensor eligible hour also carries feed chemistry")
    else:
        for _, row in dropped.iterrows():
            lines.append(
                f"  fold {int(row['fold_id'])} {row['role']}: {int(row['n_dropped'])} of "
                f"{int(row['n_committed'])}"
            )

    lines.extend(["", "Label definition per fold", ""])
    for _, row in report["labels"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: threshold {row['threshold']:.4f} "
            f"(training 90th percentile), train {int(row['n_train_positive'])}/"
            f"{int(row['n_train'])} positive ({row['train_positive_rate']:.1%}), "
            f"validation {int(row['n_validation_positive'])}/{int(row['n_validation'])} "
            f"positive ({row['validation_positive_rate']:.1%})"
        )

    lines.extend(["", "Fold metrics", ""])
    lines.append(
        f"  {'configuration':<14} {'fold':>4} {'feats':>5} {'prec':>6} {'recall':>6} "
        f"{'F1':>6} {'PR AUC':>7} {'ROC AUC':>8} {'FNR':>6} {'FPR':>6} {'acc':>6}"
    )
    for _, row in report["combined"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} {int(row['fold_id']):>4} {int(row['n_features']):>5} "
            f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f} "
            f"{row['pr_auc']:>7.4f} {row['roc_auc']:>8.4f} {row['false_negative_rate']:>6.3f} "
            f"{row['false_positive_rate']:>6.3f} {row['accuracy']:>6.3f}"
        )

    lines.extend(["", "Confusion matrices at the fixed 0.50 threshold", ""])
    for _, row in report["combined"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} fold {int(row['fold_id'])}: "
            f"TN {int(row['true_negatives']):>4}  FP {int(row['false_positives']):>3}  "
            f"FN {int(row['false_negatives']):>3}  TP {int(row['true_positives']):>3}  "
            f"(sum {int(row['true_negatives'] + row['false_positives'] + row['false_negatives'] + row['true_positives'])} "
            f"of {int(row['n_validation'])})"
        )

    lines.extend(["", "Aggregate across folds", ""])
    for _, row in report["summary"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} ({int(row['n_features'])} features)  "
            f"PR AUC {row['pr_auc_mean']:.4f} (sd {row['pr_auc_std']:.4f})  "
            f"ROC AUC {row['roc_auc_mean']:.4f} (sd {row['roc_auc_std']:.4f})  "
            f"recall {row['recall_mean']:.3f}  precision {row['precision_mean']:.3f}  "
            f"F1 {row['f1_mean']:.3f}  FNR {row['false_negative_rate_mean']:.3f}  "
            f"accuracy {row['accuracy_mean']:.3f}"
        )

    lines.extend(["", "Feed enhanced against sensor only", ""])
    for _, row in report["comparison"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])}: recall {row['recall_sensor_only']:.3f} -> "
            f"{row['recall']:.3f} ({row['recall_difference']:+.3f}), "
            f"FNR {row['false_negative_rate_sensor_only']:.3f} -> "
            f"{row['false_negative_rate']:.3f} ({row['false_negative_rate_difference']:+.3f})"
        )
        lines.append(
            f"           PR AUC {row['pr_auc_sensor_only']:.4f} -> {row['pr_auc']:.4f} "
            f"({row['pr_auc_difference']:+.4f}), ROC AUC {row['roc_auc_sensor_only']:.4f} -> "
            f"{row['roc_auc']:.4f} ({row['roc_auc_difference']:+.4f}), "
            f"F1 {row['f1_difference']:+.3f}"
        )

    lines.extend(["", "Against trivial baselines", ""])
    for _, row in report["baselines"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['baseline']:<28}: "
            f"recall {row['recall']:.3f}  precision {row['precision']:.3f}  "
            f"F1 {row['f1']:.3f}  accuracy {row['accuracy']:.3f}  "
            f"PR AUC {row['pr_auc']:.4f}  ROC AUC {row['roc_auc']:.3f}"
        )

    lines.extend(["", "Ranking lift over the no skill references", ""])
    for _, row in report["baseline_comparison"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} fold {int(row['fold_id'])}: prevalence "
            f"{row['prevalence']:.4f}, PR AUC {row['pr_auc']:.4f} "
            f"(lift {row['pr_auc_lift']:+.4f}), ROC AUC {row['roc_auc']:.4f} "
            f"(lift {row['roc_auc_lift']:+.4f})"
        )

    lines.extend(["", "Predicted probability by true class", ""])
    for _, row in report["probability_distributions"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} fold {int(row['fold_id'])} {row['true_class']:<8} "
            f"(n {int(row['n']):>3}): mean {row['mean']:.4f}, median {row['median']:.4f}, "
            f"q75 {row['q75']:.4f}, q95 {row['q95']:.4f}, max {row['max']:.4f}, "
            f"{row['share_at_or_above_decision_threshold']:.1%} at or above 0.50"
        )

    lines.extend(["", "Probability separation between the classes", ""])
    for _, row in report["probability_separation"].iterrows():
        lines.append(
            f"  {row['configuration']:<14} fold {int(row['fold_id'])}: positive mean "
            f"{row['positive_mean']:.4f} against negative mean {row['negative_mean']:.4f} "
            f"(difference {row['mean_difference']:+.4f}, standardized "
            f"{row['standardized_difference']:+.3f}, KS {row['ks_distance']:.3f}); "
            f"highest probability anywhere {row['max_probability']:.4f}"
        )

    lines.extend(["", "Feed chemistry importance in the feed enhanced classifier", ""])
    for _, row in report["feed_importance"].iterrows():
        lines.append(
            f"  fold {int(row['fold_id'])} {row['feature']:<12}: importance "
            f"{row['importance']:.4f}, rank {int(row['rank'])} of {int(row['n_features'])}"
        )
    lines.append("  Importance describes ensemble splitting behaviour, not a causal effect.")

    lines.extend(["", "Largest importances per fold, for context", ""])
    for fold_id in sorted(report["top_importance"]["fold_id"].unique()):
        subset = report["top_importance"][report["top_importance"]["fold_id"] == fold_id]
        pairs = ", ".join(
            f"{row['feature']}={row['importance']:.4f}" for _, row in subset.head(5).iterrows()
        )
        lines.append(f"  fold {int(fold_id)}: {pairs}")

    lines.extend(["", "Outcome by configuration", ""])
    for outcome in report["outcomes"]:
        lines.append(
            f"  {outcome['configuration']:<14}: {outcome['classification']}  "
            f"(PR lift on {outcome['n_folds_beating_no_skill_pr']}/{outcome['n_folds']} folds, "
            f"ROC on {outcome['n_folds_beating_no_skill_roc']}/{outcome['n_folds']}, "
            f"operationally useful on {outcome['n_folds_operationally_useful']}/{outcome['n_folds']}; "
            f"mean PR AUC {outcome['mean_pr_auc']:.4f}, lift {outcome['mean_pr_auc_lift']:+.4f}, "
            f"mean ROC AUC {outcome['mean_roc_auc']:.4f})"
        )

    return "\n".join(lines)


def run(hourly_path: Path, splits_path: Path, results_path: Path, spark=None) -> dict:
    """Run both arms, validate every guard, compare, and write the results."""
    owns_spark = spark is None
    if owns_spark:
        spark = build_spark_session("MiningQualityExcursionClassifier")

    try:
        hourly, assignment = load_inputs(hourly_path, splits_path)

        configuration_settings = verify_classifier_configuration(get_feed_enhanced_predictors())

        restricted = restrict_assignment(assignment, feed_eligible_timestamps(hourly))
        row_loss = summarize_row_loss(assignment, restricted)

        sensor_only = evaluate_configuration(spark, hourly, restricted, SENSOR_ONLY)
        feed_enhanced = evaluate_configuration(spark, hourly, restricted, FEED_ENHANCED)

        for evaluation in (sensor_only, feed_enhanced):
            validate_evaluation(evaluation, hourly, restricted, assignment)
        assert_matched_rows_and_labels(sensor_only, feed_enhanced)
        verify_deterministic_evaluation(spark, hourly, restricted, FEED_ENHANCED)

        combined = combine_results(sensor_only, feed_enhanced)
        labels = sensor_only.results[
            [
                "fold_id",
                "threshold",
                "n_train",
                "n_train_positive",
                "train_positive_rate",
                "n_validation",
                "n_validation_positive",
                "validation_positive_rate",
            ]
        ]

        report = {
            "configuration_settings": configuration_settings,
            "row_loss": row_loss,
            "labels": labels,
            "combined": combined,
            "summary": summarize_configurations(combined),
            "comparison": compare_configurations(combined),
            "baselines": baseline_table(sensor_only),
            "baseline_comparison": pd.concat(
                [baseline_comparison(sensor_only), baseline_comparison(feed_enhanced)],
                ignore_index=True,
            ),
            "probability_distributions": pd.concat(
                [probability_distributions(sensor_only), probability_distributions(feed_enhanced)],
                ignore_index=True,
            ),
            "probability_separation": pd.concat(
                [probability_separation(sensor_only), probability_separation(feed_enhanced)],
                ignore_index=True,
            ),
            "feed_importance": feed_importance_table(feed_enhanced),
            "top_importance": top_importance_table(feed_enhanced),
            "outcomes": [classify_outcome(sensor_only), classify_outcome(feed_enhanced)],
        }

        results_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(results_path, index=False)

        return report
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    results_path = default_results_path(repo_root)

    print("Starting Spark session and evaluating high silica excursion classification...")
    report = run(
        default_hourly_path(repo_root), default_splits_path(repo_root), results_path
    )
    print(f"Results written to {results_path.relative_to(repo_root)}")
    print()
    print(format_report(report))


if __name__ == "__main__":
    sys.exit(main())
