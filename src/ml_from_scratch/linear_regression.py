# linear_regression.py — Implements linear regression trained with gradient descent.
# It keeps the learning loop explicit so each mathematical step is visible.
# Author: Pasquale Marzaioli

import numpy as np

from ml_from_scratch.metrics import mean_squared_error


class LinearRegressionGD:
    """Linear regression model trained by batch gradient descent.

    Args:
        learning_rate: Step size used in each gradient descent update.
        n_iterations: Number of full-batch gradient descent updates.

    Attributes:
        weights_: Learned weights with shape (n_features,).
        bias_: Learned scalar bias.
        loss_history_: Mean squared error before each parameter update.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """Fit the model to feature matrix X and target vector y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target vector with shape (n_samples,).

        Returns:
            The fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._validate_training_data(X, y)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # Forward pass: y_hat = X @ w + b for every training row.
            predictions = X @ self.weights_ + self.bias_
            errors = predictions - y
            self.loss_history_.append(float(np.mean(errors**2)))

            # MSE gradients: dL/dw = 2/n X.T errors, dL/db = 2/n sum(errors).
            weights_gradient = (2 / n_samples) * (X.T @ errors)
            bias_gradient = (2 / n_samples) * np.sum(errors)

            # Gradient descent moves parameters opposite the loss gradient.
            self.weights_ -= self.learning_rate * weights_gradient
            self.bias_ -= self.learning_rate * bias_gradient

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for a feature matrix.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted values with shape (n_samples,).
        """
        self._require_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError("X must have the same number of features used in fit.")

        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return negative mean squared error on X and y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target vector with shape (n_samples,).

        Returns:
            Negative mean squared error. Higher values are better.
        """
        y = np.asarray(y, dtype=float)
        return -mean_squared_error(y, self.predict(X))

    @staticmethod
    def _validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
        """Validate the shared shape convention for training data."""
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")

    def _require_fitted(self) -> None:
        """Stop prediction when parameters have not been learned yet."""
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")
