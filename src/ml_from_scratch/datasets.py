# datasets.py — Builds small synthetic datasets for examples and tests.
# It uses NumPy random generators so random_state reproduces every output.
# Author: Pasquale Marzaioli

import numpy as np


def _validate_dataset_parameters(
    n_samples: int,
    n_features: int,
    noise: float,
) -> None:
    """Validate shared synthetic dataset parameters."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError("n_samples must be a positive integer.")
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise ValueError("n_features must be a positive integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")


def _prepare_weights(
    weights: np.ndarray | None,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return explicit or generated weights with shape (n_features,)."""
    if weights is None:
        return rng.normal(size=n_features)

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_features,):
        raise ValueError("weights must have shape (n_features,).")

    return weights


def _prepare_polynomial_coefficients(
    coefficients: np.ndarray | None,
    degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return polynomial coefficients from degree 0 through degree n."""
    if coefficients is None:
        return rng.normal(size=degree + 1)

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (degree + 1,):
        raise ValueError("coefficients must have shape (degree + 1,).")

    return coefficients


def make_regression_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 1.0,
    weights: np.ndarray | None = None,
    bias: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic linear regression dataset.

    Args:
        n_samples: Number of rows in X and y.
        n_features: Number of feature columns in X.
        noise: Standard deviation of Gaussian target noise.
        weights: Optional true weights with shape (n_features,).
        bias: Scalar added to the linear target.
        random_state: Optional seed for reproducible features, weights, and noise.

    Returns:
        X with shape (n_samples, n_features) and y with shape (n_samples,).
    """
    _validate_dataset_parameters(n_samples, n_features, noise)

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = _prepare_weights(weights, n_features, rng)

    # Keep the target formula visible: linear signal plus Gaussian noise.
    y = X @ weights + bias + rng.normal(0, noise, size=n_samples)
    return X, y


def make_polynomial_regression_data(
    n_samples: int = 100,
    degree: int = 2,
    noise: float = 1.0,
    coefficients: np.ndarray | None = None,
    x_min: float = -3.0,
    x_max: float = 3.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a one-feature synthetic polynomial regression dataset.

    Args:
        n_samples: Number of rows in X and y.
        degree: Highest polynomial power in the target formula.
        noise: Standard deviation of Gaussian target noise.
        coefficients: Optional coefficients from degree 0 to degree n.
        x_min: First x value in the evenly spaced feature column.
        x_max: Last x value in the evenly spaced feature column.
        random_state: Optional seed for reproducible coefficients and noise.

    Returns:
        X with shape (n_samples, 1) and y with shape (n_samples,).
    """
    _validate_dataset_parameters(n_samples, 1, noise)
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise ValueError("degree must be a positive integer.")
    if degree <= 0:
        raise ValueError("degree must be positive.")
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max.")

    rng = np.random.default_rng(random_state)
    coefficients = _prepare_polynomial_coefficients(coefficients, degree, rng)
    X = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    x = X[:, 0]

    # Coefficients are ordered as bias, x, x^2, ... so the formula stays explicit.
    y = np.zeros(n_samples)
    for power, coefficient in enumerate(coefficients):
        y += coefficient * x**power
    y += rng.normal(0, noise, size=n_samples)

    return X, y


def make_binary_classification_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.5,
    weights: np.ndarray | None = None,
    bias: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic binary classification dataset.

    Args:
        n_samples: Number of rows in X and y.
        n_features: Number of feature columns in X.
        noise: Standard deviation of Gaussian score noise.
        weights: Optional true weights with shape (n_features,).
        bias: Scalar added to the linear score.
        random_state: Optional seed for reproducible features, weights, and noise.

    Returns:
        X with shape (n_samples, n_features) and binary y with shape (n_samples,).
    """
    _validate_dataset_parameters(n_samples, n_features, noise)

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = _prepare_weights(weights, n_features, rng)

    # Threshold noisy linear scores at zero to create labels 0 and 1.
    scores = X @ weights + bias + rng.normal(0, noise, size=n_samples)
    y = (scores >= 0).astype(int)
    return X, y
