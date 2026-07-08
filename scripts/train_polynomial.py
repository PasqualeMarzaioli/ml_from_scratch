# train_polynomial.py — Trains polynomial regression on synthetic data.
# It expands one input feature into powers, then reuses linear regression.
# Author: Pasquale Marzaioli

import os
from pathlib import Path


def main() -> None:
    """Train a quadratic model and save example figures under the plots directory."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)

    # Import plotting after cache setup so Matplotlib works in locked-down shells.
    import numpy as np

    from ml_from_scratch.linear_regression import LinearRegressionGD
    from ml_from_scratch.metrics import mean_squared_error
    from ml_from_scratch.plotting import (
        plot_loss_curve,
        plot_loss_curve_zoom,
        plot_regression_fit,
    )
    from ml_from_scratch.preprocessing import (
        normalize_features,
        polynomial_features,
        train_test_split,
        transform_features,
    )

    rng = np.random.default_rng(11)
    degree = 2
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = 1 - 2 * X[:, 0] + 0.5 * X[:, 0] ** 2 + rng.normal(0, 0.6, X.shape[0])

    # Split raw x values first, then apply the same feature expansion to each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=11,
    )

    X_train_polynomial = polynomial_features(X_train, degree=degree)
    X_test_polynomial = polynomial_features(X_test, degree=degree)
    X_train_normalized, means, scales = normalize_features(X_train_polynomial)
    X_test_normalized = transform_features(X_test_polynomial, means, scales)

    model = LinearRegressionGD(learning_rate=0.05, n_iterations=2000)
    model.fit(X_train_normalized, y_train)

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Plot original x values on the axis while the model predicts from x powers.
    fit_ax = plot_regression_fit(
        X_train,
        y_train,
        model,
        prediction_X=X_train_normalized,
    )
    fit_ax.set_title("Polynomial regression fit")
    fit_ax.figure.savefig(
        output_dir / "polynomial_regression_fit.png",
        dpi=150,
        bbox_inches="tight",
    )

    loss_ax = plot_loss_curve(model.loss_history_)
    loss_ax.figure.savefig(
        output_dir / "polynomial_regression_loss.png",
        dpi=150,
        bbox_inches="tight",
    )

    zoom_ax = plot_loss_curve_zoom(model.loss_history_, skip_first=5)
    zoom_ax.figure.savefig(
        output_dir / "polynomial_regression_loss_zoom.png",
        dpi=150,
        bbox_inches="tight",
    )

    # Convert normalized-space weights back to the original polynomial feature scale.
    raw_weights = model.weights_ / scales
    raw_bias = model.bias_ - float(np.sum(model.weights_ * means / scales))
    test_mse = mean_squared_error(y_test, model.predict(X_test_normalized))

    print(f"degree: {degree}")
    print(f"raw-scale bias: {raw_bias:.3f}")
    for power, weight in enumerate(raw_weights, start=1):
        print(f"raw-scale x^{power} weight: {weight:.3f}")
    print(f"test MSE: {test_mse:.3f}")
    print(f"saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
