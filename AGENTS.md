# AGENTS.md

## Project: ML Playground From Scratch

This repository is an educational machine learning project. Its purpose is to teach the fundamentals of supervised learning by implementing core algorithms from scratch, with minimal dependencies and explicit mathematical explanations.

The primary goal is not performance. The primary goal is conceptual clarity, reproducibility, and readable code.

## Intended learning outcomes

The student should understand:

- What it means for a model to learn from data.
- How a loss function measures prediction error.
- How gradient descent updates parameters.
- Why train/test splits matter.
- How overfitting and underfitting appear in practice.
- How simple metrics such as MSE, accuracy, precision, recall, and F1 are computed.
- How a hand-written implementation compares with a library implementation.

## Scope

Implement the following from scratch using NumPy:

- Linear regression.
- Polynomial regression.
- Logistic regression for binary classification.
- Gradient descent.
- Train/test split.
- Feature normalization.
- Basic metrics.
- Simple plotting utilities.
- Unit tests for numerical correctness.

Scikit-learn may be used only for comparison scripts, not for the core implementation.

## Recommended repository structure

```text
ml-playground-from-scratch/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   └── ml_from_scratch/
│       ├── __init__.py
│       ├── datasets.py
│       ├── linear_regression.py
│       ├── logistic_regression.py
│       ├── metrics.py
│       ├── preprocessing.py
│       ├── plotting.py
│       └── optimization.py
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   └── 03_overfitting.ipynb
├── scripts/
│   ├── train_linear.py
│   ├── train_logistic.py
│   └── compare_sklearn.py
└── tests/
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_metrics.py
    └── test_preprocessing.py
```

## Implementation rules

- Use Python 3.11+.
- Use NumPy for numerical computation.
- Use Matplotlib for plots.
- Avoid hidden magic and excessive abstraction.
- Keep functions small and explicit.
- Every mathematical function should have a docstring explaining inputs, outputs, and shapes.
- Every model should expose at least:
  - `fit(X, y)`
  - `predict(X)`
  - `score(X, y)` where appropriate.
- Do not use scikit-learn inside the core model implementations.
- Do not implement neural networks in this project.

## Mathematical conventions

Use these shape conventions consistently:

```text
X: shape (n_samples, n_features)
y: shape (n_samples,)
weights: shape (n_features,)
bias: scalar
predictions: shape (n_samples,)
```

For linear regression:

```text
y_hat = X @ w + b
loss = mean((y_hat - y)^2)
```

For logistic regression:

```text
z = X @ w + b
p = sigmoid(z)
loss = binary_cross_entropy(y, p)
```

## Commands

Use these commands unless the project configuration changes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

If `pyproject.toml` does not exist yet, create it.

## Testing expectations

Tests should cover:

- Prediction shapes.
- Loss decreases during training on a simple synthetic dataset.
- Gradient computations for small examples.
- Metric values on known inputs.
- Normalization behavior.
- Edge cases such as constant features and mismatched shapes.

When adding functionality, add tests before or alongside implementation.

## README expectations

The README should explain:

- What the project teaches.
- The mathematical model.
- How gradient descent works.
- How to run training scripts.
- How to interpret plots.
- What the project intentionally does not cover.

## Codex workflow

When working on this repository:

1. Inspect the current file tree.
2. Read `README.md`, `pyproject.toml`, and existing tests before editing.
3. Propose a small implementation plan.
4. Modify one module at a time.
5. Add or update tests.
6. Run formatting and tests.
7. Report what changed and what remains incomplete.

## Definition of done

A task is complete when:

- The relevant code is implemented.
- Tests pass.
- The public API is documented.
- The README or notebook is updated if the concept is user-facing.
- The implementation is understandable to a beginner who knows basic Python and linear algebra.

## Avoid

- Do not introduce deep learning frameworks.
- Do not use scikit-learn for core algorithms.
- Do not optimize prematurely.
- Do not hide learning-relevant steps behind utilities.
- Do not produce code without tests.
