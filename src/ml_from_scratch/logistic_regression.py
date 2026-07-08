# logistic_regression.py — Implements binary logistic regression with gradient descent.
# It keeps the sigmoid, loss, and update equations visible for learning.
# Author: Pasquale Marzaioli

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Map raw scores to probabilities with the logistic sigmoid.

    Args:
        z: Raw score array with any shape.

    Returns:
        Probabilities with the same shape as z.
    """
    z = np.asarray(z, dtype=float)

    # Clipping avoids exponential overflow while preserving practical probabilities.
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy_loss(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    """Return binary cross-entropy for true labels and predicted probabilities.

    Args:
        y_true: Binary target vector with shape (n_samples,).
        y_probability: Predicted probabilities with shape (n_samples,).

    Returns:
        Average binary cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)

    if y_true.ndim != 1 or y_probability.ndim != 1:
        raise ValueError("y_true and y_probability must both have shape (n_samples,).")
    if y_true.shape != y_probability.shape:
        raise ValueError("y_true and y_probability must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1.")

    # Probabilities exactly at 0 or 1 make log undefined, so clip only for loss.
    y_probability = np.clip(y_probability, 1e-15, 1 - 1e-15)
    loss = y_true * np.log(y_probability)
    loss += (1 - y_true) * np.log(1 - y_probability)
    return float(-np.mean(loss))


class LogisticRegressionGD:
    """Binary logistic regression model trained by batch gradient descent.

    Args:
        learning_rate: Step size used in each gradient descent update.
        n_iterations: Number of full-batch gradient descent updates.

    Attributes:
        weights_: Learned weights with shape (n_features,).
        bias_: Learned scalar bias.
        loss_history_: Binary cross-entropy before each parameter update.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        """Fit the model to feature matrix X and binary target vector y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary target vector with shape (n_samples,).

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
            # Forward pass: z = X @ w + b, then p = sigmoid(z).
            probabilities = sigmoid(X @ self.weights_ + self.bias_)
            errors = probabilities - y
            self.loss_history_.append(binary_cross_entropy_loss(y, probabilities))

            # BCE gradients for sigmoid output simplify to p - y.
            weights_gradient = (1 / n_samples) * (X.T @ errors)
            bias_gradient = (1 / n_samples) * np.sum(errors)

            # Move parameters opposite the gradient to reduce classification loss.
            self.weights_ -= self.learning_rate * weights_gradient
            self.bias_ -= self.learning_rate * bias_gradient

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class-1 probabilities for a feature matrix.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples,).
        """
        self._require_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError("X must have the same number of features used in fit.")

        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for a feature matrix.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted labels with shape (n_samples,).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on X and y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary target vector with shape (n_samples,).

        Returns:
            Fraction of labels predicted correctly.
        """
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,).")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("y must contain only 0 and 1.")

        predictions = self.predict(X)
        if y.shape != predictions.shape:
            raise ValueError("X and y must contain the same number of samples.")

        return float(np.mean(predictions == y))

    @staticmethod
    def _validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
        """Validate the shared shape convention and binary labels."""
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("y must contain only 0 and 1.")

    def _require_fitted(self) -> None:
        """Stop prediction when parameters have not been learned yet."""
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")
