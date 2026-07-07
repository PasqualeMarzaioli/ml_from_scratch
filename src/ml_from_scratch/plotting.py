# plotting.py — Provides small plotting helpers for linear regression.
# It visualizes model fit and training loss without hiding the learning process.
# Author: Pasquale Marzaioli

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ml_from_scratch.linear_regression import LinearRegressionGD


def plot_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegressionGD,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one-feature training data and the fitted regression line.

    Args:
        X: Feature matrix with shape (n_samples, 1).
        y: Target vector with shape (n_samples,).
        model: Fitted linear regression model.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the plot.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # A 2D line plot can only show one input feature without extra choices.
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("plot_regression_fit expects X with shape (n_samples, 1).")
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("y must have shape (n_samples,) matching X.")

    if ax is None:
        _, ax = plt.subplots()

    sorted_indices = np.argsort(X[:, 0])
    sorted_X = X[sorted_indices]

    ax.scatter(X[:, 0], y, label="data", alpha=0.75)
    ax.plot(sorted_X[:, 0], model.predict(sorted_X), color="tab:red", label="model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear regression fit")
    ax.legend()

    return ax


def plot_loss_curve(
    loss_history: Sequence[float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot mean squared error over gradient descent iterations.

    Args:
        loss_history: Loss values stored during training.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the plot.
    """
    if len(loss_history) == 0:
        raise ValueError("loss_history must contain at least one value.")

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(np.arange(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared error")
    ax.set_title("Training loss")

    return ax
