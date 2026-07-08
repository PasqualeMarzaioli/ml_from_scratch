# plotting.py — Provides small plotting helpers for regression and classification.
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
    prediction_X: np.ndarray | None = None,
) -> plt.Axes:
    """Plot one-feature training data and the fitted regression line.

    Args:
        X: Feature matrix with shape (n_samples, 1).
        y: Target vector with shape (n_samples,).
        model: Fitted linear regression model.
        ax: Optional Matplotlib axes to draw on.
        prediction_X: Optional model input with shape (n_samples, n_features).

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
    if prediction_X is None:
        prediction_X = X
    else:
        prediction_X = np.asarray(prediction_X, dtype=float)
        if prediction_X.ndim != 2 or prediction_X.shape[0] != X.shape[0]:
            raise ValueError("prediction_X must have one row for each row in X.")

    if ax is None:
        _, ax = plt.subplots()

    sorted_indices = np.argsort(X[:, 0])
    sorted_X = X[sorted_indices]
    sorted_prediction_X = prediction_X[sorted_indices]

    ax.scatter(X[:, 0], y, label="data", alpha=0.75)
    ax.plot(
        sorted_X[:, 0],
        model.predict(sorted_prediction_X),
        color="tab:red",
        label="model",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear regression fit")
    ax.legend()

    return ax


def plot_loss_curve(
    loss_history: Sequence[float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot mean squared error over gradient descent iterations on a log y-axis.

    Args:
        loss_history: Loss values stored during training.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the plot.
    """
    losses = np.asarray(loss_history, dtype=float)

    if losses.size == 0:
        raise ValueError("loss_history must contain at least one value.")
    if losses.ndim != 1:
        raise ValueError("loss_history must be one-dimensional.")
    if np.any(losses <= 0):
        raise ValueError("loss_history must contain only positive values.")

    if ax is None:
        _, ax = plt.subplots()

    # Log scale makes early progress and later convergence visible together.
    ax.plot(np.arange(1, losses.size + 1), losses)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared error (log scale)")
    ax.set_title("Training loss")

    return ax


def plot_binary_cross_entropy_curve(
    loss_history: Sequence[float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot binary cross-entropy over gradient descent iterations on a log y-axis.

    Args:
        loss_history: Loss values stored during training.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the plot.
    """
    # Reuse the same loss plotting behavior, then relabel it for classification.
    ax = plot_loss_curve(loss_history, ax=ax)
    ax.set_ylabel("binary cross-entropy (log scale)")
    ax.set_title("Logistic regression training loss")
    return ax


def plot_logistic_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    probability_X: np.ndarray,
    probabilities: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one-feature binary data and a class-1 probability curve.

    Args:
        X: Feature matrix for displayed training data with shape (n_samples, 1).
        y: Binary target vector with shape (n_samples,).
        probability_X: X-axis values for the probability curve with shape (n_points, 1).
        probabilities: Class-1 probabilities with shape (n_points,).
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the plot.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    probability_X = np.asarray(probability_X, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    # A probability curve over x can only be drawn directly for one input feature.
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(
            "plot_logistic_regression_fit expects X with shape (n_samples, 1)."
        )
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("y must have shape (n_samples,) matching X.")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y must contain only 0 and 1.")
    if probability_X.ndim != 2 or probability_X.shape[1] != 1:
        raise ValueError("probability_X must have shape (n_points, 1).")
    if probabilities.ndim != 1 or probability_X.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "probabilities must have shape (n_points,) matching probability_X."
        )

    if ax is None:
        _, ax = plt.subplots()

    sorted_indices = np.argsort(probability_X[:, 0])
    sorted_probability_X = probability_X[sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]

    ax.scatter(X[:, 0], y, label="data", alpha=0.75)
    ax.plot(
        sorted_probability_X[:, 0],
        sorted_probabilities,
        color="tab:red",
        label="P(class 1)",
    )
    ax.axhline(0.5, color="tab:gray", linestyle="--", label="threshold")
    ax.set_xlabel("x")
    ax.set_ylabel("probability")
    ax.set_title("Logistic regression fit")
    ax.legend()

    return ax


def plot_loss_curve_zoom(
    loss_history: Sequence[float],
    skip_first: int = 5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot mean squared error after skipping the first training iterations.

    Args:
        loss_history: Loss values stored during training.
        skip_first: Number of initial iterations to exclude from the plot.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        The Matplotlib axes containing the zoomed plot.
    """
    losses = np.asarray(loss_history, dtype=float)

    if losses.size == 0:
        raise ValueError("loss_history must contain at least one value.")
    if losses.ndim != 1:
        raise ValueError("loss_history must be one-dimensional.")
    if skip_first < 0:
        raise ValueError("skip_first must be non-negative.")
    if skip_first >= losses.size:
        raise ValueError("skip_first must leave at least one loss value to plot.")

    if ax is None:
        _, ax = plt.subplots()

    # Dropping the first steep updates reveals the smaller convergence changes.
    iterations = np.arange(skip_first + 1, losses.size + 1)
    ax.plot(iterations, losses[skip_first:])
    if iterations.size > 1:
        ax.set_xlim(iterations[0], iterations[-1])
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared error")
    ax.set_title(f"Training loss after first {skip_first} iterations")

    return ax
