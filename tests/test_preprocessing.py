# test_preprocessing.py — Verifies data splitting behavior.
# It keeps train and test row alignment reproducible with a fixed seed.
# Author: Pasquale Marzaioli

import numpy as np

from ml_from_scratch.preprocessing import train_test_split


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
