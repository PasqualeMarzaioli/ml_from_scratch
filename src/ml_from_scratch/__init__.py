# __init__.py — Exposes the public package API.
# It imports only the first educational building blocks implemented so far.
# Author: Pasquale Marzaioli

from ml_from_scratch.linear_regression import LinearRegressionGD
from ml_from_scratch.metrics import mean_squared_error
from ml_from_scratch.preprocessing import train_test_split

__all__ = ["LinearRegressionGD", "mean_squared_error", "train_test_split"]
