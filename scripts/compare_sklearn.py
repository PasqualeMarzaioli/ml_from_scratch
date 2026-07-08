# compare_sklearn.py — Compares from-scratch models with scikit-learn models.
# It reuses the same synthetic data, splits, preprocessing, metrics, and plots.
# Author: Pasquale Marzaioli

import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ml_from_scratch.datasets import (
    make_binary_classification_data,
    make_polynomial_regression_data,
    make_regression_data,
)
from ml_from_scratch.linear_regression import LinearRegressionGD
from ml_from_scratch.logistic_regression import (
    LogisticRegressionGD,
    binary_cross_entropy_loss,
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


def raw_scale_parameters(
    weights: np.ndarray,
    bias: float,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert normalized-feature parameters back to the original feature scale.

    Args:
        weights: Learned weights for normalized features with shape (n_features,).
        bias: Learned scalar bias for normalized features.
        means: Training feature means with shape (n_features,).
        scales: Training feature scales with shape (n_features,).

    Returns:
        Raw-scale weights with shape (n_features,) and the raw-scale scalar bias.
    """
    raw_weights = weights / scales
    raw_bias = bias - float(np.sum(weights * means / scales))
    return raw_weights, raw_bias


def print_metric_block(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print binary classification metrics for labels with shape (n_samples,)."""
    print(f"{name} accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"{name} precision: {precision_score(y_true, y_pred):.3f}")
    print(f"{name} recall: {recall_score(y_true, y_pred):.3f}")
    print(f"{name} F1: {f1_score(y_true, y_pred):.3f}")


def compare_linear_regression() -> tuple[np.ndarray, np.ndarray]:
    """Compare linear regression models on the same train/test split."""
    X, y = make_regression_data(
        n_samples=100,
        weights=np.array([3.0]),
        bias=2.0,
        noise=1.0,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
    )

    # Normalize once and give both models the same transformed inputs.
    X_train_normalized, means, scales = normalize_features(X_train)
    X_test_normalized = transform_features(X_test, means, scales)

    scratch_model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    scratch_model.fit(X_train_normalized, y_train)

    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train_normalized, y_train)

    scratch_predictions = scratch_model.predict(X_test_normalized)
    sklearn_predictions = sklearn_model.predict(X_test_normalized)

    scratch_weights, scratch_bias = raw_scale_parameters(
        scratch_model.weights_,
        scratch_model.bias_,
        means,
        scales,
    )
    sklearn_weights, sklearn_intercept = raw_scale_parameters(
        sklearn_model.coef_,
        float(sklearn_model.intercept_),
        means,
        scales,
    )

    print("Linear regression")
    print(f"from-scratch MSE: {mean_squared_error(y_test, scratch_predictions):.3f}")
    print(f"scikit-learn MSE: {mean_squared_error(y_test, sklearn_predictions):.3f}")
    print(f"from-scratch raw-scale weights: {np.round(scratch_weights, 3)}")
    print(f"from-scratch raw-scale bias: {scratch_bias:.3f}")
    print(f"scikit-learn raw-scale weights: {np.round(sklearn_weights, 3)}")
    print(f"scikit-learn raw-scale intercept: {sklearn_intercept:.3f}")

    return sklearn_predictions, scratch_predictions


def compare_polynomial_regression() -> tuple[np.ndarray, np.ndarray]:
    """Compare polynomial regression after explicit feature expansion."""
    degree = 2
    X, y = make_polynomial_regression_data(
        n_samples=120,
        degree=degree,
        coefficients=np.array([1.0, -2.0, 0.5]),
        noise=0.6,
        random_state=11,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=11,
    )

    # Expand raw x into powers before normalization so both models see x and x^2.
    X_train_polynomial = polynomial_features(X_train, degree=degree)
    X_test_polynomial = polynomial_features(X_test, degree=degree)
    X_train_normalized, means, scales = normalize_features(X_train_polynomial)
    X_test_normalized = transform_features(X_test_polynomial, means, scales)

    scratch_model = LinearRegressionGD(learning_rate=0.05, n_iterations=2000)
    scratch_model.fit(X_train_normalized, y_train)

    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train_normalized, y_train)

    scratch_predictions = scratch_model.predict(X_test_normalized)
    sklearn_predictions = sklearn_model.predict(X_test_normalized)

    scratch_weights, scratch_bias = raw_scale_parameters(
        scratch_model.weights_,
        scratch_model.bias_,
        means,
        scales,
    )
    sklearn_weights, sklearn_intercept = raw_scale_parameters(
        sklearn_model.coef_,
        float(sklearn_model.intercept_),
        means,
        scales,
    )

    print("\nPolynomial regression")
    print(f"degree: {degree}")
    print(f"from-scratch MSE: {mean_squared_error(y_test, scratch_predictions):.3f}")
    print(f"scikit-learn MSE: {mean_squared_error(y_test, sklearn_predictions):.3f}")
    print(f"from-scratch raw-scale bias: {scratch_bias:.3f}")
    print(f"from-scratch raw-scale coefficients: {np.round(scratch_weights, 3)}")
    print(f"scikit-learn raw-scale intercept: {sklearn_intercept:.3f}")
    print(f"scikit-learn raw-scale coefficients: {np.round(sklearn_weights, 3)}")

    return sklearn_predictions, scratch_predictions


def compare_logistic_regression() -> tuple[np.ndarray, np.ndarray]:
    """Compare binary logistic regression models on normalized features."""
    X, y = make_binary_classification_data(
        n_samples=120,
        weights=np.array([1.0]),
        noise=0.6,
        random_state=19,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=19,
    )

    # Use identical normalization so metric differences come from the learners.
    X_train_normalized, means, scales = normalize_features(X_train)
    X_test_normalized = transform_features(X_test, means, scales)

    scratch_model = LogisticRegressionGD(learning_rate=0.2, n_iterations=1000)
    scratch_model.fit(X_train_normalized, y_train)

    # C=np.inf disables regularization, matching the from-scratch objective.
    sklearn_model = SklearnLogisticRegression(C=np.inf, max_iter=1000)
    sklearn_model.fit(X_train_normalized, y_train)

    scratch_probabilities = scratch_model.predict_proba(X_test_normalized)
    sklearn_probabilities = sklearn_model.predict_proba(X_test_normalized)[:, 1]
    scratch_predictions = scratch_model.predict(X_test_normalized)
    sklearn_predictions = sklearn_model.predict(X_test_normalized)

    scratch_weights, scratch_bias = raw_scale_parameters(
        scratch_model.weights_,
        scratch_model.bias_,
        means,
        scales,
    )
    sklearn_weights, sklearn_intercept = raw_scale_parameters(
        sklearn_model.coef_[0],
        float(sklearn_model.intercept_[0]),
        means,
        scales,
    )
    probability_difference = np.mean(
        np.abs(scratch_probabilities - sklearn_probabilities)
    )

    print("\nLogistic regression")
    print_metric_block("from-scratch", y_test, scratch_predictions)
    print(
        "from-scratch BCE: "
        f"{binary_cross_entropy_loss(y_test, scratch_probabilities):.3f}"
    )
    print(f"from-scratch raw-scale weights: {np.round(scratch_weights, 3)}")
    print(f"from-scratch raw-scale bias: {scratch_bias:.3f}")
    print_metric_block("scikit-learn", y_test, sklearn_predictions)
    print(
        "scikit-learn BCE: "
        f"{binary_cross_entropy_loss(y_test, sklearn_probabilities):.3f}"
    )
    print(f"scikit-learn raw-scale weights: {np.round(sklearn_weights, 3)}")
    print(f"scikit-learn raw-scale intercept: {sklearn_intercept:.3f}")
    print(f"mean absolute probability difference: {probability_difference:.3f}")
    print(
        f"from-scratch first 5 probabilities: {np.round(scratch_probabilities[:5], 3)}"
    )
    print(
        f"scikit-learn first 5 probabilities: {np.round(sklearn_probabilities[:5], 3)}"
    )

    return sklearn_probabilities, scratch_probabilities


def save_comparison_plot(
    linear_values: tuple[np.ndarray, np.ndarray],
    polynomial_values: tuple[np.ndarray, np.ndarray],
    logistic_values: tuple[np.ndarray, np.ndarray],
) -> Path:
    """Save parity plots comparing scikit-learn and from-scratch outputs.

    Args:
        linear_values: scikit-learn and from-scratch regression predictions.
        polynomial_values: scikit-learn and from-scratch polynomial predictions.
        logistic_values: scikit-learn and from-scratch class-1 probabilities.

    Returns:
        Path to the saved comparison image.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)

    # Import plotting after cache setup so Matplotlib works in locked-down shells.
    import matplotlib.pyplot as plt

    from ml_from_scratch.plotting import plot_model_agreement

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sklearn_comparison.png"

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    plot_model_agreement(
        linear_values[0],
        linear_values[1],
        title="Linear regression",
        value_label="prediction",
        ax=axes[0],
    )
    plot_model_agreement(
        polynomial_values[0],
        polynomial_values[1],
        title="Polynomial regression",
        value_label="prediction",
        ax=axes[1],
    )
    plot_model_agreement(
        logistic_values[0],
        logistic_values[1],
        title="Logistic regression",
        value_label="probability",
        ax=axes[2],
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    """Run all scikit-learn comparisons."""
    linear_values = compare_linear_regression()
    polynomial_values = compare_polynomial_regression()
    logistic_values = compare_logistic_regression()
    output_path = save_comparison_plot(
        linear_values,
        polynomial_values,
        logistic_values,
    )
    print(f"\nsaved comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
