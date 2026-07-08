# test_preprocessing.py — Verifies data splitting behavior.
# It keeps train and test row alignment reproducible with a fixed seed.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.preprocessing import normalize_features, train_test_split


def test_normalize_features_returns_zero_mean_unit_std_columns() -> None:
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    X_normalized, means, scales = normalize_features(X)

    assert np.allclose(means, np.array([2.0, 20.0]))
    assert np.allclose(scales, np.std(X, axis=0))
    assert np.allclose(np.mean(X_normalized, axis=0), np.array([0.0, 0.0]))
    assert np.allclose(np.std(X_normalized, axis=0), np.array([1.0, 1.0]))


def test_normalize_features_handles_constant_columns() -> None:
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

    X_normalized, means, scales = normalize_features(X)

    assert np.allclose(means, np.array([2.0, 5.0]))
    assert np.allclose(scales, np.array([np.std(X[:, 0]), 1.0]))
    assert np.allclose(X_normalized[:, 1], np.array([0.0, 0.0, 0.0]))


def test_normalize_features_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        normalize_features(np.array([1.0, 2.0, 3.0]))


def test_train_test_split_returns_expected_sizes() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1,
    )

    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_train_test_split_keeps_rows_aligned_without_shuffle() -> None:
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=2,
        shuffle=False,
    )

    assert np.array_equal(X_test, X[:2])
    assert np.array_equal(y_test, y[:2])
    assert np.array_equal(X_train, X[2:])
    assert np.array_equal(y_train, y[2:])
