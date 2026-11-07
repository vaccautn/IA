"""Canonical metrics derived from a categorical confusion matrix."""
from __future__ import annotations

import math
from typing import Any

from .constants import BCS_CLASS_SCORES, NUM_CLASSES, SCORE_STEP

METRICS_TOLERANCE = 1e-8
"""Maximum persisted floating-point disagreement from confusion-derived metrics."""


def derive_category_metrics(confusion_matrix: object) -> dict[str, Any]:
    """Derive every category metric from one integer 5x5 confusion matrix."""
    if (
        type(confusion_matrix) is not list
        or len(confusion_matrix) != NUM_CLASSES
        or any(
            type(row) is not list
            or len(row) != NUM_CLASSES
            or any(type(value) is not int or value < 0 for value in row)
            for row in confusion_matrix
        )
    ):
        raise ValueError("confusion matrix must be a non-negative 5x5 integer matrix")
    matrix = confusion_matrix
    support = [sum(row) for row in matrix]
    total = sum(support)
    if total <= 0:
        raise ValueError("confusion matrix must contain at least one sample")
    diagonal = sum(matrix[index][index] for index in range(NUM_CLASSES))
    within = sum(
        matrix[true_index][predicted_index]
        for true_index in range(NUM_CLASSES)
        for predicted_index in range(NUM_CLASSES)
        if abs(predicted_index - true_index) <= 1
    )
    error_ge_2 = sum(
        matrix[true_index][predicted_index]
        for true_index in range(NUM_CLASSES)
        for predicted_index in range(NUM_CLASSES)
        if abs(predicted_index - true_index) >= 2
    )
    absolute_error = sum(
        abs(predicted_index - true_index) * matrix[true_index][predicted_index]
        for true_index in range(NUM_CLASSES)
        for predicted_index in range(NUM_CLASSES)
    )
    precision: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}
    f1: dict[str, float | None] = {}
    within_one_by_class: dict[str, float | None] = {}
    error_ge_2_by_class: dict[str, float | None] = {}
    for index, class_name in enumerate(BCS_CLASS_SCORES):
        predicted = sum(matrix[row][index] for row in range(NUM_CLASSES))
        true_positive = matrix[index][index]
        precision_value = true_positive / predicted if predicted else None
        recall_value = true_positive / support[index] if support[index] else None
        precision[str(class_name)] = precision_value
        recall[str(class_name)] = recall_value
        f1[str(class_name)] = (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value is not None
            and recall_value is not None
            and precision_value + recall_value
            else None
        )
        within_one_by_class[str(class_name)] = (
            sum(
                matrix[index][predicted_index]
                for predicted_index in range(NUM_CLASSES)
                if abs(predicted_index - index) <= 1
            )
            / support[index]
            if support[index]
            else None
        )
        error_ge_2_by_class[str(class_name)] = (
            sum(
                matrix[index][predicted_index]
                for predicted_index in range(NUM_CLASSES)
                if abs(predicted_index - index) >= 2
            )
            / support[index]
            if support[index]
            else None
        )
    valid_f1 = [value for value in f1.values() if value is not None]
    valid_recall = [value for value in recall.values() if value is not None]
    return {
        "exact_acc": diagonal / total,
        "within_one": within / total,
        "mae": SCORE_STEP * absolute_error / total,
        "ordinal_mae": SCORE_STEP * absolute_error / total,
        "error_ge_2": error_ge_2 / total,
        "macro_f1": sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0,
        "balanced_accuracy": sum(valid_recall) / len(valid_recall) if valid_recall else 0.0,
        "within_one_by_class": within_one_by_class,
        "error_ge_2_by_class": error_ge_2_by_class,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "total": total,
    }


def assert_metrics_match_confusion(metrics: object) -> None:
    """Reject persisted metrics that disagree with their confusion matrix."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics payload must be an object")
    derived = derive_category_metrics(metrics.get("confusion_matrix"))
    scalar_fields = (
        "exact_acc", "within_one", "mae", "ordinal_mae", "error_ge_2",
        "macro_f1", "balanced_accuracy",
    )
    for field in scalar_fields:
        value = metrics.get(field)
        expected = derived[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0, abs_tol=METRICS_TOLERANCE)
        ):
            raise ValueError(f"metrics.{field} does not match the confusion matrix")
    for field in ("support", "total", "confusion_matrix"):
        if metrics.get(field) != derived[field]:
            raise ValueError(f"metrics.{field} does not match the confusion matrix")
    for field in ("within_one_by_class", "error_ge_2_by_class", "precision", "recall", "f1"):
        reported = metrics.get(field)
        expected = derived[field]
        if type(reported) is not dict or set(reported) != set(expected):
            raise ValueError(f"metrics.{field} does not match the confusion matrix")
        for class_name, expected_value in expected.items():
            value = reported[class_name]
            if expected_value is None:
                if value is not None:
                    raise ValueError(f"metrics.{field} does not match the confusion matrix")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(float(value), expected_value, rel_tol=0, abs_tol=METRICS_TOLERANCE)
            ):
                raise ValueError(f"metrics.{field} does not match the confusion matrix")
