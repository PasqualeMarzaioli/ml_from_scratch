# __init__.py — Exposes the public package API.
# It imports only the educational building blocks implemented so far.
# Author: Pasquale Marzaioli

from ml_from_scratch.datasets import (
    make_binary_classification_data,
    make_polynomial_regression_data,
    make_regression_data,
)
from ml_from_scratch.linear_regression import LinearRegressionGD
from ml_from_scratch.logistic_regression import (
    LogisticRegressionGD,
    binary_cross_entropy_loss,
    sigmoid,
)
from ml_from_scratch.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from ml_from_scratch.preprocessing import (
    normalize_features,
    polynomial_features,
    train_test_split,
    transform_features,
)

__all__ = [
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "accuracy_score",
    "binary_cross_entropy_loss",
    "f1_score",
    "make_binary_classification_data",
    "make_polynomial_regression_data",
    "make_regression_data",
    "mean_squared_error",
    "normalize_features",
    "polynomial_features",
    "precision_score",
    "recall_score",
    "sigmoid",
    "train_test_split",
    "transform_features",
]
