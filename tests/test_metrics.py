# test_metrics.py — Verifies basic metric calculations.
# It checks known values so the formulas stay easy to trust.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.metrics import mean_squared_error


def test_mean_squared_error_matches_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(5 / 3)


def test_mean_squared_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1.0, 2.0]), np.array([1.0]))
