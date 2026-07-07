# test_linear_regression.py — Verifies linear regression numerical behavior.
# It checks shapes, gradient updates, and convergence on simple data.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.linear_regression import LinearRegressionGD


def test_predict_returns_one_value_per_sample() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)

    assert model.predict(X).shape == (3,)


def test_first_gradient_step_matches_hand_calculation() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([3.0, 5.0])

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=1)
    model.fit(X, y)

    assert model.loss_history_[0] == pytest.approx(17.0)
    assert model.weights_[0] == pytest.approx(1.3)
    assert model.bias_ == pytest.approx(0.8)


def test_loss_decreases_during_training() -> None:
    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = 3 * X[:, 0] - 2

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=200)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]


def test_model_learns_simple_line() -> None:
    X = np.linspace(-1, 1, 80).reshape(-1, 1)
    y = 3 * X[:, 0] - 2

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)

    assert model.weights_[0] == pytest.approx(3.0, abs=0.01)
    assert model.bias_ == pytest.approx(-2.0, abs=0.01)
