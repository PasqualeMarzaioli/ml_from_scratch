# test_plotting.py — Verifies plotting helper behavior.
# It checks plot configuration without depending on image pixel output.
# Author: Pasquale Marzaioli

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ml_from_scratch.linear_regression import LinearRegressionGD
from ml_from_scratch.plotting import (
    plot_loss_curve,
    plot_loss_curve_zoom,
    plot_regression_fit,
)
from ml_from_scratch.preprocessing import normalize_features


def test_plot_loss_curve_uses_log_y_axis() -> None:
    ax = plot_loss_curve([10.0, 1.0, 0.5])

    assert ax.get_yscale() == "log"

    plt.close(ax.figure)


def test_plot_loss_curve_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        plot_loss_curve([1.0, 0.0])


def test_plot_loss_curve_zoom_skips_first_iterations() -> None:
    ax = plot_loss_curve_zoom([10.0, 5.0, 2.0, 1.0], skip_first=2)

    line = ax.lines[0]
    assert line.get_xdata().tolist() == [3, 4]
    assert line.get_ydata().tolist() == [2.0, 1.0]
    assert ax.get_xlim() == pytest.approx((3.0, 4.0))
    assert ax.get_yscale() == "linear"

    plt.close(ax.figure)


def test_plot_loss_curve_zoom_requires_remaining_values() -> None:
    with pytest.raises(ValueError, match="leave at least one"):
        plot_loss_curve_zoom([1.0, 0.5], skip_first=2)


def test_plot_regression_fit_can_plot_original_x_with_normalized_model_input() -> None:
    X = np.array([[0.0], [10.0], [20.0]])
    y = np.array([1.0, 21.0, 41.0])
    X_normalized, _, _ = normalize_features(X)

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=200)
    model.fit(X_normalized, y)

    ax = plot_regression_fit(X, y, model, prediction_X=X_normalized)
    line = ax.lines[0]

    assert line.get_xdata().tolist() == [0.0, 10.0, 20.0]
    assert np.allclose(line.get_ydata(), model.predict(X_normalized))

    plt.close(ax.figure)
