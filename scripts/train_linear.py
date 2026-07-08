# train_linear.py — Trains linear regression on synthetic data and saves plots.
# It demonstrates the package API with a reproducible one-feature dataset.
# Author: Pasquale Marzaioli

import os
from pathlib import Path


def main() -> None:
    """Train the model and save example figures under the plots directory."""
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
        train_test_split,
        transform_features,
    )

    rng = np.random.default_rng(7)
    X = np.linspace(0, 10, 80).reshape(-1, 1)
    y = 3 * X[:, 0] + 2 + rng.normal(0, 1, size=X.shape[0])

    # Split before training so the printed error estimates unseen data behavior.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
    )

    X_train_normalized, means, scales = normalize_features(X_train)
    X_test_normalized = transform_features(X_test, means, scales)

    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train_normalized, y_train)

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Save one plot for the fitted line and one for optimization progress.
    fit_ax = plot_regression_fit(
        X_train,
        y_train,
        model,
        prediction_X=X_train_normalized,
    )
    fit_ax.figure.savefig(
        output_dir / "linear_regression_fit.png",
        dpi=150,
        bbox_inches="tight",
    )

    loss_ax = plot_loss_curve(model.loss_history_)
    loss_ax.figure.savefig(
        output_dir / "linear_regression_loss.png",
        dpi=150,
        bbox_inches="tight",
    )

    zoom_ax = plot_loss_curve_zoom(model.loss_history_, skip_first=175)
    zoom_ax.figure.savefig(
        output_dir / "linear_regression_loss_zoom.png",
        dpi=150,
        bbox_inches="tight",
    )

    # Convert normalized-space parameters back to the original x scale for display.
    raw_weights = model.weights_ / scales
    raw_bias = model.bias_ - float(np.sum(model.weights_ * means / scales))
    test_mse = mean_squared_error(y_test, model.predict(X_test_normalized))

    print(f"raw-scale weight: {raw_weights[0]:.3f}")
    print(f"raw-scale bias: {raw_bias:.3f}")
    print(f"test MSE: {test_mse:.3f}")
    print(f"saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
