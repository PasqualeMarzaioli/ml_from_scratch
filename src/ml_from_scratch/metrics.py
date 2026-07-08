# metrics.py — Implements basic model evaluation metrics.
# It keeps formulas explicit so metric values can be checked by hand.
# Author: Pasquale Marzaioli

import numpy as np


def _validate_binary_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated binary label arrays with shape (n_samples,)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Keep binary metric validation centralized so every formula has same checks.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both have shape (n_samples,).")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)) or np.any(
        (y_pred != 0) & (y_pred != 1),
    ):
        raise ValueError("y_true and y_pred must contain only 0 and 1.")

    return y_true, y_pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error between two one-dimensional arrays.

    Args:
        y_true: True target values with shape (n_samples,).
        y_pred: Predicted target values with shape (n_samples,).

    Returns:
        The average squared difference between true and predicted values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Validate shapes once at the metric boundary so callers get clear errors.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both have shape (n_samples,).")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return float(np.mean((y_pred - y_true) ** 2))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of matching binary labels.

    Args:
        y_true: True binary labels with shape (n_samples,).
        y_pred: Predicted binary labels with shape (n_samples,).

    Returns:
        Correct predictions divided by total predictions.
    """
    y_true, y_pred = _validate_binary_labels(y_true, y_pred)

    if y_true.size == 0:
        return 0.0

    return float(np.sum(y_true == y_pred) / y_true.size)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return binary precision for class 1 predictions.

    Args:
        y_true: True binary labels with shape (n_samples,).
        y_pred: Predicted binary labels with shape (n_samples,).

    Returns:
        True positives divided by predicted positives.
    """
    y_true, y_pred = _validate_binary_labels(y_true, y_pred)

    # Precision asks how many predicted positives were actually positive.
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0.0

    return float(true_positives / predicted_positives)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return binary recall for class 1 labels.

    Args:
        y_true: True binary labels with shape (n_samples,).
        y_pred: Predicted binary labels with shape (n_samples,).

    Returns:
        True positives divided by actual positives.
    """
    y_true, y_pred = _validate_binary_labels(y_true, y_pred)

    # Recall asks how many actual positives the model found.
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0.0

    return float(true_positives / actual_positives)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the binary F1 score for class 1 labels.

    Args:
        y_true: True binary labels with shape (n_samples,).
        y_pred: Predicted binary labels with shape (n_samples,).

    Returns:
        Harmonic mean of precision and recall.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))
