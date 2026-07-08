<!-- README.md — Explains the project, linear regression math, and usage.
It connects the code to the formulas so the package is useful for learning.
Author: Pasquale Marzaioli
-->

# ML From Scratch

`ml_from_scratch` is a small educational Python package for learning
machine learning by implementing the core ideas directly with NumPy.

This version implements linear regression and logistic regression trained with
batch gradient descent, plus polynomial feature expansion for curved
one-feature examples.

## Synthetic Datasets

Synthetic datasets make examples repeatable and easy to inspect because the
true pattern is known before training. The `random_state` argument fixes the
NumPy generator seed, so features, generated weights, and noise are the same
each time the example runs.

```python
import numpy as np

from ml_from_scratch import make_regression_data

X, y = make_regression_data(
    n_samples=100,
    weights=np.array([3.0]),
    bias=2.0,
    noise=1.0,
    random_state=7,
)
```

```python
import numpy as np

from ml_from_scratch import make_binary_classification_data

X, y = make_binary_classification_data(
    n_samples=100,
    weights=np.array([1.0]),
    noise=0.5,
    random_state=19,
)
```

```python
import numpy as np

from ml_from_scratch import make_polynomial_regression_data

X, y = make_polynomial_regression_data(
    n_samples=120,
    degree=2,
    coefficients=np.array([1.0, -2.0, 0.5]),
    noise=0.6,
    random_state=11,
)
```

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

## Polynomial Regression

Polynomial regression fits a curve by adding powers of an input feature before
training the same linear regression model.

For one input feature `x` and degree `3`, the expanded features are:

```text
[x, x^2, x^3]
```

The prediction formula becomes:

```text
y_hat = b + w1*x + w2*x^2 + w3*x^3
```

This is still linear in the learned parameters because the model learns the
weights `w1`, `w2`, `w3`, and the bias `b`. The powers of `x` are fixed input
features created before training, not learned parameters.

## Logistic Regression

Logistic regression classifies each row into one of two classes: `0` or `1`.
It first computes a raw score with the same linear formula used by linear
regression:

```text
z = X @ w + b
```

Then it converts that score into a probability with the sigmoid function:

```text
p = sigmoid(z) = 1 / (1 + exp(-z))
```

The loss is binary cross-entropy:

```text
loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

For `n` samples, the gradients are:

```text
errors = p - y
dL/dw = (1 / n) * X.T @ errors
dL/db = (1 / n) * sum(errors)
```

Gradient descent uses the same update rule as linear regression:

```text
w = w - learning_rate * dL/dw
b = b - learning_rate * dL/db
```

After training, predicted probabilities become class labels with a fixed
threshold: `p >= 0.5` gives class `1`, otherwise class `0`.

## Classification Metrics

Binary classification metrics compare true labels and predicted labels, where
class `1` is treated as the positive class.

- True positive (`TP`): the true label is `1` and the prediction is `1`.
- False positive (`FP`): the true label is `0` but the prediction is `1`.
- False negative (`FN`): the true label is `1` but the prediction is `0`.
- Accuracy: correct predictions divided by all predictions.
- Precision: `TP / (TP + FP)`, so it answers: when the model predicts `1`, how
  often is it right?
- Recall: `TP / (TP + FN)`, so it answers: out of the real `1` labels, how many
  did the model find?
- F1: `2 * precision * recall / (precision + recall)`, a single score that is
  high only when both precision and recall are high.

When a denominator is zero, these metric functions return `0.0`.

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

## Feature Normalization

Gradient descent is easier to tune when features use similar scales. This
package normalizes each feature column with:

```text
X_normalized = (X - mean) / standard_deviation
```

After normalization, each non-constant feature has mean `0` and standard
deviation `1`. Constant features are converted to zeros because their standard
deviation is `0`. Reuse the returned means and scales to transform future data
with the same formula.

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

## Run Educational Notebooks

Install the optional notebook tools only when you want to view or execute the
walkthroughs:

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks
```

The notebooks are:

- `notebooks/01_linear_regression.ipynb`: linear regression, MSE, gradient
  descent, normalization, and fit diagnostics.
- `notebooks/02_logistic_regression.ipynb`: sigmoid probabilities, binary
  cross-entropy, thresholding, and classification metrics.
- `notebooks/03_overfitting.ipynb`: polynomial features, underfitting,
  overfitting, and why test error matters.

To execute one notebook from the terminal:

```bash
jupyter execute --kernel_name=python3 notebooks/01_linear_regression.ipynb
```

## Run Example Scripts

```bash
python scripts/train_linear.py
python scripts/train_polynomial.py
python scripts/train_logistic.py
```

The example scripts save:

- `plots/linear_regression_fit.png`
- `plots/linear_regression_loss.png`
- `plots/linear_regression_loss_zoom.png`
- `plots/polynomial_regression_fit.png`
- `plots/polynomial_regression_loss.png`
- `plots/polynomial_regression_loss_zoom.png`
- `plots/logistic_regression_fit.png`
- `plots/logistic_regression_loss.png`

The logistic script also prints final binary cross-entropy loss, train accuracy,
test accuracy, test precision, test recall, and test F1.

## Compare With Scikit-Learn

Scikit-learn is used only in `scripts/compare_sklearn.py` so the core
from-scratch implementations stay focused on learning the math. Install the
optional comparison dependency before running the script:

```bash
pip install -e ".[comparison]"
python scripts/compare_sklearn.py
```

The comparison script trains from-scratch and scikit-learn models on the same
synthetic datasets, train/test splits, and normalized features. Small numeric
differences are expected because gradient descent, closed-form solvers,
regularization, and library defaults do not all match exactly. For logistic
regression, scikit-learn regularization is disabled so probabilities and
coefficients are more directly comparable.

The script also saves a parity plot to `plots/sklearn_comparison.png`. Points
near the diagonal mean the from-scratch and scikit-learn outputs agree.

## Public API

```python
import numpy as np

from ml_from_scratch import (
    LinearRegressionGD,
    LogisticRegressionGD,
    accuracy_score,
    f1_score,
    make_binary_classification_data,
    make_polynomial_regression_data,
    make_regression_data,
    precision_score,
    recall_score,
)
from ml_from_scratch.preprocessing import (
    normalize_features,
    polynomial_features,
    transform_features,
)

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1.0, 3.0, 5.0])

X_polynomial = polynomial_features(X, degree=2)
X_normalized, means, scales = normalize_features(X_polynomial)

new_X = polynomial_features(np.array([[3.0]]), degree=2)
new_X_normalized = transform_features(new_X, means, scales)

model = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
model.fit(X_normalized, y)

predictions = model.predict(new_X_normalized)

classifier_X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
classifier_y = np.array([0, 0, 1, 1])

classifier = LogisticRegressionGD(learning_rate=0.5, n_iterations=300)
classifier.fit(classifier_X, classifier_y)

probabilities = classifier.predict_proba(classifier_X)
labels = classifier.predict(classifier_X)
accuracy = accuracy_score(classifier_y, labels)
precision = precision_score(classifier_y, labels)
recall = recall_score(classifier_y, labels)
f1 = f1_score(classifier_y, labels)
```

## What This Project Does Not Cover Yet

- multi-class classification
- advanced classification metrics such as ROC curves
