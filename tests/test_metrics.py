# test_metrics.py — Verifies basic metric calculations.
# It checks known values so the formulas stay easy to trust.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)

BINARY_METRICS = [accuracy_score, precision_score, recall_score, f1_score]


def test_mean_squared_error_matches_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(5 / 3)


def test_mean_squared_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1.0, 2.0]), np.array([1.0]))


def test_accuracy_score_matches_known_value() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0])

    assert accuracy_score(y_true, y_pred) == pytest.approx(3 / 6)


def test_precision_score_matches_known_value() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0])

    assert precision_score(y_true, y_pred) == pytest.approx(1 / 2)


def test_recall_score_matches_known_value() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0])

    assert recall_score(y_true, y_pred) == pytest.approx(1 / 3)


def test_f1_score_matches_known_value() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0])

    assert f1_score(y_true, y_pred) == pytest.approx(2 / 5)


def test_binary_metrics_return_zero_with_no_predicted_positives() -> None:
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 0])

    assert precision_score(y_true, y_pred) == pytest.approx(0.0)
    assert f1_score(y_true, y_pred) == pytest.approx(0.0)


def test_binary_metrics_return_zero_with_no_actual_positives() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 0, 1])

    assert recall_score(y_true, y_pred) == pytest.approx(0.0)
    assert f1_score(y_true, y_pred) == pytest.approx(0.0)


@pytest.mark.parametrize("metric", BINARY_METRICS)
def test_binary_metrics_reject_non_vector_labels(metric) -> None:
    with pytest.raises(ValueError, match="shape"):
        metric(np.array([[0, 1]]), np.array([[0, 1]]))


@pytest.mark.parametrize("metric", BINARY_METRICS)
def test_binary_metrics_reject_mismatched_shapes(metric) -> None:
    with pytest.raises(ValueError, match="same shape"):
        metric(np.array([0, 1]), np.array([0]))


@pytest.mark.parametrize("metric", BINARY_METRICS)
@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (np.array([0, 2]), np.array([0, 1])),
        (np.array([0, 1]), np.array([0, 2])),
    ],
)
def test_binary_metrics_reject_labels_outside_zero_and_one(
    metric,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        metric(y_true, y_pred)
