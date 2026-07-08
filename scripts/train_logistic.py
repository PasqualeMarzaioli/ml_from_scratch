# train_logistic.py — Trains logistic regression and saves example plots.
# It demonstrates train/test splitting, normalization, accuracy, and probability curves.
# Author: Pasquale Marzaioli

import os
from pathlib import Path

import numpy as np

from ml_from_scratch.logistic_regression import LogisticRegressionGD
from ml_from_scratch.preprocessing import (
    normalize_features,
    train_test_split,
    transform_features,
)


def main() -> None:
    """Train the classifier, save plots, and print a compact accuracy summary."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)

    # Import plotting after cache setup so Matplotlib works in locked-down shells.
    import matplotlib.pyplot as plt

    from ml_from_scratch.plotting import (
        plot_binary_cross_entropy_curve,
        plot_logistic_regression_fit,
    )

    rng = np.random.default_rng(19)
    X = np.linspace(-4, 4, 120).reshape(-1, 1)
    y = (X[:, 0] + rng.normal(0, 0.6, size=X.shape[0]) >= 0).astype(int)

    # Split before normalization so test data stays unseen during preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=19,
    )

    X_train_normalized, means, scales = normalize_features(X_train)
    X_test_normalized = transform_features(X_test, means, scales)

    model = LogisticRegressionGD(learning_rate=0.2, n_iterations=1000)
    model.fit(X_train_normalized, y_train)

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Plot raw x values while the model predicts from normalized features.
    x_grid = np.linspace(float(np.min(X)), float(np.max(X)), 200).reshape(-1, 1)
    x_grid_normalized = transform_features(x_grid, means, scales)
    probabilities = model.predict_proba(x_grid_normalized)

    fit_ax = plot_logistic_regression_fit(X_train, y_train, x_grid, probabilities)
    fit_ax.figure.savefig(
        output_dir / "logistic_regression_fit.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fit_ax.figure)

    loss_ax = plot_binary_cross_entropy_curve(model.loss_history_)
    loss_ax.figure.savefig(
        output_dir / "logistic_regression_loss.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(loss_ax.figure)

    print(f"final loss: {model.loss_history_[-1]:.3f}")
    print(f"train accuracy: {model.score(X_train_normalized, y_train):.3f}")
    print(f"test accuracy: {model.score(X_test_normalized, y_test):.3f}")
    print(f"saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
