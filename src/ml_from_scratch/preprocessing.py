# preprocessing.py — Implements small data preparation helpers.
# It provides the minimum split utility needed by examples and tests.
# Author: Pasquale Marzaioli

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float | int = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and targets into train and test sets.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        y: Target vector with shape (n_samples,).
        test_size: Test fraction as a float, or exact test count as an int.
        random_state: Optional seed for reproducible shuffling.
        shuffle: Whether to shuffle rows before splitting.

    Returns:
        X_train, X_test, y_train, y_test in that order.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Keep shape rules strict so every downstream formula uses one convention.
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("y must have shape (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    n_samples = X.shape[0]
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        n_test = int(np.ceil(n_samples * test_size))
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            raise ValueError("Integer test_size must be between 1 and n_samples - 1.")
        n_test = test_size
    else:
        raise TypeError("test_size must be a float or an int.")

    if n_test >= n_samples:
        raise ValueError("test_size leaves no samples for training.")

    # NumPy indexing keeps rows aligned between X and y after optional shuffling.
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
