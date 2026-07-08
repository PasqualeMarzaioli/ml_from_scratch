# test_datasets.py — Verifies deterministic synthetic dataset helpers.
# It checks shapes, reproducibility, labels, custom weights, and validation.
# Author: Pasquale Marzaioli

import numpy as np
import pytest

from ml_from_scratch.datasets import (
    make_binary_classification_data,
    make_polynomial_regression_data,
    make_regression_data,
)


def test_regression_data_shape() -> None:
    X, y = make_regression_data(n_samples=12, n_features=3, random_state=1)

    assert X.shape == (12, 3)
    assert y.shape == (12,)


def test_classification_data_shape() -> None:
    X, y = make_binary_classification_data(n_samples=12, n_features=3, random_state=1)

    assert X.shape == (12, 3)
    assert y.shape == (12,)


def test_polynomial_regression_data_shape() -> None:
    X, y = make_polynomial_regression_data(n_samples=12, degree=3, random_state=1)

    assert X.shape == (12, 1)
    assert y.shape == (12,)


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_same_random_state_reproduces_data(make_data) -> None:
    first_X, first_y = make_data(n_samples=20, n_features=2, random_state=3)
    second_X, second_y = make_data(n_samples=20, n_features=2, random_state=3)

    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_different_random_state_changes_data(make_data) -> None:
    first_X, first_y = make_data(n_samples=20, n_features=2, random_state=3)
    second_X, second_y = make_data(n_samples=20, n_features=2, random_state=4)

    assert not np.array_equal(first_X, second_X)
    assert not np.array_equal(first_y, second_y)


def test_polynomial_regression_same_random_state_reproduces_data() -> None:
    first_X, first_y = make_polynomial_regression_data(random_state=3)
    second_X, second_y = make_polynomial_regression_data(random_state=3)

    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)


def test_polynomial_regression_different_random_state_changes_targets() -> None:
    first_X, first_y = make_polynomial_regression_data(random_state=3)
    second_X, second_y = make_polynomial_regression_data(random_state=4)

    assert np.array_equal(first_X, second_X)
    assert not np.array_equal(first_y, second_y)


def test_custom_regression_weights_produce_expected_noiseless_targets() -> None:
    weights = np.array([2.0, -1.0])
    X, y = make_regression_data(
        n_samples=8,
        n_features=2,
        noise=0.0,
        weights=weights,
        bias=0.5,
        random_state=5,
    )

    assert np.allclose(y, X @ weights + 0.5)


def test_custom_polynomial_coefficients_produce_expected_noiseless_targets() -> None:
    coefficients = np.array([1.0, -2.0, 0.5])
    X, y = make_polynomial_regression_data(
        n_samples=8,
        degree=2,
        noise=0.0,
        coefficients=coefficients,
        x_min=-1.0,
        x_max=1.0,
        random_state=5,
    )

    x = X[:, 0]
    assert np.allclose(
        y, coefficients[0] + coefficients[1] * x + coefficients[2] * x**2
    )


def test_custom_classification_weights_produce_binary_labels() -> None:
    _, y = make_binary_classification_data(
        n_samples=30,
        n_features=2,
        weights=np.array([1.0, -1.0]),
        random_state=6,
    )

    assert set(y.tolist()) <= {0, 1}


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_invalid_n_samples_is_rejected(make_data) -> None:
    with pytest.raises(ValueError, match="n_samples"):
        make_data(n_samples=0)


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_invalid_n_features_is_rejected(make_data) -> None:
    with pytest.raises(ValueError, match="n_features"):
        make_data(n_features=0)


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_invalid_noise_is_rejected(make_data) -> None:
    with pytest.raises(ValueError, match="noise"):
        make_data(noise=-0.1)


@pytest.mark.parametrize(
    "make_data",
    [make_regression_data, make_binary_classification_data],
)
def test_wrong_weight_shape_is_rejected(make_data) -> None:
    with pytest.raises(ValueError, match="weights"):
        make_data(n_features=2, weights=np.array([1.0]))


def test_invalid_polynomial_degree_is_rejected() -> None:
    with pytest.raises(ValueError, match="degree"):
        make_polynomial_regression_data(degree=0)


def test_wrong_polynomial_coefficient_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        make_polynomial_regression_data(degree=2, coefficients=np.array([1.0, 2.0]))


def test_invalid_polynomial_x_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="x_min"):
        make_polynomial_regression_data(x_min=1.0, x_max=1.0)
