<!-- README.md — Explains the project, linear regression math, and usage.
It connects the code to the formulas so the package is useful for learning.
Author: Pasquale Marzaioli
-->

# ML From Scratch

`ml_from_scratch` is a small educational Python package for learning supervised
machine learning by implementing the core ideas directly with NumPy.

This first version implements linear regression trained with batch gradient
descent.

## What Linear Regression Learns

Linear regression learns a straight-line relationship between input features
and a numeric target.

For one row of data:

```text
y_hat = X @ w + b
```

- `X` is the feature matrix with shape `(n_samples, n_features)`.
- `w` is the weight vector with shape `(n_features,)`.
- `b` is the scalar bias.
- `y_hat` is the prediction vector with shape `(n_samples,)`.

The model starts with weights set to zero and improves them one gradient descent
step at a time.

## Loss Function

The loss tells the model how wrong its predictions are. This package uses mean
squared error:

```text
loss = mean((y_hat - y) ** 2)
```

Squaring makes large errors count more than small errors, and averaging gives
one number for the whole training set.

## Gradients

Gradient descent needs the slope of the loss with respect to each parameter.

For `n` samples:

```text
errors = y_hat - y
dL/dw = (2 / n) * X.T @ errors
dL/db = (2 / n) * sum(errors)
```

These gradients say how much the loss changes if the weights or bias move a
little.

## Gradient Descent Update

Each training step moves parameters in the opposite direction of the gradient:

```text
w = w - learning_rate * dL/dw
b = b - learning_rate * dL/db
```

The `learning_rate` controls the step size. Too small learns slowly. Too large
can overshoot the best line.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Generate Example Plots

```bash
python scripts/train_linear.py
```

This saves:

- `plots/linear_regression_fit.png`
- `plots/linear_regression_loss.png`
- `plots/linear_regression_loss_zoom.png`

## Public API

```python
import numpy as np

from ml_from_scratch import LinearRegressionGD

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1.0, 3.0, 5.0])

model = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
model.fit(X, y)

predictions = model.predict(X)
```

## What This Project Does Not Cover Yet

- polynomial regression
- logistic regression
- feature normalization
- classification metrics
- scikit-learn comparison scripts
