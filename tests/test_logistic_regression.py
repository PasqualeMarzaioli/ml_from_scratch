# test_logistic_regression.py — Verifies binary logistic regression behavior.
# It checks probability shapes, label predictions, validation, and convergence.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.logistic_regression import (
    LogisticRegressionGD,
    binary_cross_entropy_loss,
    sigmoid,
)


def test_sigmoid_matches_known_values() -> None:
    z = np.array([-1.0, 0.0, 1.0])

    probabilities = sigmoid(z)

    assert probabilities == pytest.approx(
        np.array([0.26894142, 0.5, 0.73105858]),
    )


def test_binary_cross_entropy_loss_matches_known_value() -> None:
    y_true = np.array([1, 0])
    y_probability = np.array([0.5, 0.5])

    assert binary_cross_entropy_loss(y_true, y_probability) == pytest.approx(
        np.log(2),
    )


def test_predict_proba_returns_one_probability_per_sample() -> None:
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0, 0, 1])

    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=100)
    model.fit(X, y)

    assert model.predict_proba(X).shape == (3,)


def test_predict_returns_binary_labels() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=300)
    model.fit(X, y)

    assert set(model.predict(X).tolist()) <= {0, 1}


def test_loss_decreases_during_training() -> None:
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = (X[:, 0] >= 0).astype(int)

    model = LogisticRegressionGD(learning_rate=0.3, n_iterations=200)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]


def test_model_learns_simple_separable_dataset() -> None:
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=500)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_fit_rejects_labels_outside_zero_and_one() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])

    model = LogisticRegressionGD()

    with pytest.raises(ValueError, match="0 and 1"):
        model.fit(X, y)


def test_prediction_before_fit_raises_value_error() -> None:
    model = LogisticRegressionGD()

    with pytest.raises(ValueError, match="fitted"):
        model.predict(np.array([[0.0]]))
