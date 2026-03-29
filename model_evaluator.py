"""
model_evaluator.py — Trains and evaluates a GradientBoosting model on clean sensor data
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Uses sliding-window feature engineering to turn time-series sensor readings
into a supervised regression task, then checks RMSE / R² quality gates.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Any


class ModelEvaluator:
    """
    Trains a GradientBoostingRegressor on a cleaned dataset and evaluates
    it against configured quality gates.

    Parameters
    ----------
    df            : pd.DataFrame — the clean dataset
    target_column : str          — column to predict (auto-selected if not found)
    config        : module       — the config module
    """

    def __init__(self, df: pd.DataFrame, target_column: str = "temperature", config: Any = None):
        self.df     = df.copy()
        self.config = config
        # Auto-detect target if the requested column is absent
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.target  = target_column if target_column in df.columns else (
            numeric_cols[0] if numeric_cols else None
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Build features, train model, compute metrics, and check quality gates.

        Returns
        -------
        {
            "target"        : str,
            "rmse"          : float,
            "r2"            : float,
            "mae"           : float,
            "train_samples" : int,
            "test_samples"  : int,
            "passed"        : bool,
            "gates"         : dict,
            "predictions"   : {"actual": list, "predicted": list}   (first 50)
        }
        """
        if self.target is None:
            return self._error("No numeric target column found in dataset.")

        values = self.df[self.target].dropna().values
        window = self.config.WINDOW_SIZE if self.config else 10

        if len(values) < window + 20:
            return self._error(
                f"Not enough data for model evaluation "
                f"(need >{window + 20} rows, got {len(values)})."
            )

        X, y = self._build_features(values, window)
        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2   = float(r2_score(y_test, preds))
        mae  = float(mean_absolute_error(y_test, preds))

        threshold_rmse = self.config.RMSE_THRESHOLD if self.config else 5.0
        threshold_r2   = self.config.R2_THRESHOLD   if self.config else 0.5

        gates = {
            "rmse_pass": rmse < threshold_rmse,
            "r2_pass":   r2   > threshold_r2,
            "rmse_threshold": threshold_rmse,
            "r2_threshold":   threshold_r2,
        }

        return {
            "target":        self.target,
            "rmse":          round(rmse, 4),
            "r2":            round(r2,   4),
            "mae":           round(mae,  4),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "passed":        gates["rmse_pass"] and gates["r2_pass"],
            "gates":         gates,
            "predictions": {
                "actual":    y_test[:50].tolist(),
                "predicted": preds[:50].tolist(),
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_features(values: np.ndarray, window: int):
        """Create sliding-window feature matrix from a 1-D time series."""
        X, y = [], []
        for i in range(window, len(values)):
            X.append(values[i - window : i])
            y.append(values[i])
        return np.array(X), np.array(y)

    @staticmethod
    def _error(message: str) -> dict:
        """Return an error result dict without raising an exception."""
        return {
            "target":        None,
            "rmse":          None,
            "r2":            None,
            "mae":           None,
            "train_samples": 0,
            "test_samples":  0,
            "passed":        False,
            "gates":         {},
            "error":         message,
            "predictions":   {},
        }
