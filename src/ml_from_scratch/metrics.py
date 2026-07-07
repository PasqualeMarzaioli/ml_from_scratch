# metrics.py — Implements basic model evaluation metrics.
# It keeps formulas explicit so metric values can be checked by hand.
# Author: Pasquale Marzaioli

import numpy as np


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
