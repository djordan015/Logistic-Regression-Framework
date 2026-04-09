"""
logistic_regression.py

Python wrapper around the compiled C++ extension.

Accepts both numpy arrays and plain Python lists as inputs, converting to
``list[list[float]]`` / ``list[float]`` internally before handing off to C++.

Classes:
LogisticRegression
    The main user-facing class.  Mirrors the scikit-learn estimator interface:
    ``fit``, ``predict``, ``predict_proba``, ``score``.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    from logistic_regression_cpu import (
        GradientDescent,
        LogitClassifier as _LogitClassifier,
        Model as _Model,
        ModelSnapshot,
        Optimizer,
        SGD,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import the C++ extension 'logistic_regression_cpu'.\n"
        "Build it first:\n"
        "    mkdir build && cd build\n"
        "    cmake ..\n"
        "    make\n"
        "Then make sure the resulting .so / .pyd is on your PYTHONPATH."
    ) from exc


def _to_2d_list(X: Union[np.ndarray, list]) -> list:
    """Convert any 2-D array-like to ``list[list[float]]``."""
    if isinstance(X, np.ndarray):
        return X.tolist()
    return [[float(v) for v in row] for row in X]


def _to_1d_list(y: Union[np.ndarray, list]) -> list:
    """Convert any 1-D array-like to ``list[float]``."""
    if isinstance(y, np.ndarray):
        return y.tolist()
    return [float(v) for v in y]


def _make_optimizer(name: str) -> Optimizer:
    """Instantiate a C++ optimizer from its string name."""
    key = name.lower()
    if key == "sgd":
        return SGD()
    if key in ("gd", "gradient_descent", "gradientdescent"):
        return GradientDescent()
    raise ValueError(
        f"Unknown optimizer '{name}'. "
        "Choose 'sgd' for SGD or 'gd' / 'gradient_descent' for full-batch GD."
    )

class LogisticRegression:
    """
    Parameters:
    threshold : float, optional
    epochs : int, optional
    optimizer : str, optional
    debug : bool, optional
    """

    def __init__(
        self,
        threshold: float = 0.5,
        epochs: int = 1000,
        optimizer: str = "gd",
        debug: bool = False,
    ) -> None:
        self.threshold = threshold
        self.epochs = epochs
        self.debug = debug
        self._optimizer_name = optimizer

        # Underlying C++ objects
        self._model: _Model = _Model(threshold, epochs, debug)
        self._opt: Optimizer = _make_optimizer(optimizer)
        self._classifier: _LogitClassifier = _LogitClassifier()

        self._is_fitted: bool = False

    # Fitting 

    def fit(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
    ) -> "LogisticRegression":
        self._model.train(_to_2d_list(X), _to_1d_list(y), self._opt)
        self._is_fitted = True
        return self

    # Inference

    def predict_proba(
        self,
        X: Union[np.ndarray, list],
    ) -> np.ndarray:
        self._check_fitted()
        X_list   = _to_2d_list(X)
        weights  = self._model.get_weights()   # list[float] copy
        bias     = self._model.get_bias()
        probs    = self._classifier.forward_batch(X_list, weights, bias)
        return np.array(probs, dtype=float)

    def predict(
        self,
        X: Union[np.ndarray, list],
    ) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(float)

    def score(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
    ) -> float:
        self._check_fitted()
        return self._model.test(_to_2d_list(X), _to_1d_list(y))

    # Model persistence

    def get_snapshot(self) -> ModelSnapshot:
        self._check_fitted()
        return self._model.get_snapshot()

    def load_snapshot(self, snapshot: ModelSnapshot) -> "LogisticRegression":
        self._model.load_snapshot(snapshot.weights, snapshot.bias)
        self._is_fitted = True
        return self

    # Hyper-parameter setters

    def set_epochs(self, epochs: int) -> "LogisticRegression":
        self.epochs = epochs
        self._model.set_epochs(epochs)
        return self

    def set_threshold(self, threshold: float) -> "LogisticRegression":
        self.threshold = threshold
        self._model.set_threshold(threshold)
        return self

    # Properties (read-only after fit)

    @property
    def weights(self) -> np.ndarray:
        self._check_fitted()
        return np.array(self._model.get_weights(), dtype=float)

    @property
    def bias(self) -> float:
        self._check_fitted()
        return float(self._model.get_bias())

    @property
    def coef_(self) -> np.ndarray:
        return self.weights

    @property
    def intercept_(self) -> float:
        return self.bias

    # Dunder

    def _check_fitted(self) -> None:
        """Raise if the model has not been trained yet."""
        if not self._is_fitted:
            raise RuntimeError(
                "This LogisticRegression instance is not fitted yet. "
                "Call fit() with training data before using this method."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LogisticRegression("
            f"threshold={self.threshold}, "
            f"epochs={self.epochs}, "
            f"optimizer='{self._optimizer_name}', "
            f"debug={self.debug}) [{status}]"
        )
