# __init__.py — Exposes the public package API.
# It imports only the first educational building blocks implemented so far.
# Author: Pasquale Marzaioli

from ml_from_scratch.linear_regression import LinearRegressionGD
from ml_from_scratch.metrics import mean_squared_error
from ml_from_scratch.preprocessing import (
    normalize_features,
    polynomial_features,
    train_test_split,
    transform_features,
)

__all__ = [
    "LinearRegressionGD",
    "mean_squared_error",
    "normalize_features",
    "polynomial_features",
    "train_test_split",
    "transform_features",
]
