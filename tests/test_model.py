"""
tests/test_model.py — Model quality gate tests
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Trains a GradientBoostingRegressor on the sample data and asserts
that RMSE and R² meet the configured quality thresholds.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DATA_FILE = Path("data/sample_sensor_data.csv")


def load_data() -> pd.DataFrame:
    """Load the sample sensor dataset."""
    return pd.read_csv(DATA_FILE)


def _build_sliding_window(values: np.ndarray, window: int = 10):
    """Create supervised features from a 1-D time series via sliding window."""
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window: i])
        y.append(values[i])
    return np.array(X), np.array(y)


def test_model_meets_quality_gates():
    """
    A GradientBoostingRegressor trained on the first numeric column must
    satisfy both RMSE < RMSE_THRESHOLD and R² > R2_THRESHOLD.
    """
    df           = load_data()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        pytest.skip("No numeric columns found — skipping model test")

    target = "temperature" if "temperature" in numeric_cols else numeric_cols[0]
    values = df[target].dropna().values

    if len(values) < 50:
        pytest.skip(f"Not enough data for model test (found {len(values)} rows)")

    X, y  = _build_sliding_window(values, window=10)
    split = int(len(X) * 0.8)

    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, random_state=42
    )
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])

    rmse = float(np.sqrt(mean_squared_error(y[split:], preds)))
    r2   = float(r2_score(y[split:], preds))

    assert rmse < config.RMSE_THRESHOLD, (
        f"RMSE {rmse:.4f} exceeds threshold {config.RMSE_THRESHOLD}"
    )
    assert r2 > config.R2_THRESHOLD, (
        f"R² {r2:.4f} is below threshold {config.R2_THRESHOLD}"
    )


def test_model_predictions_in_reasonable_range():
    """
    Model predictions for temperature should stay within a physically
    plausible range even on unseen data.
    """
    df           = load_data()
    if "temperature" not in df.columns:
        pytest.skip("No temperature column")

    values = df["temperature"].dropna().values
    if len(values) < 50:
        pytest.skip("Not enough data")

    X, y  = _build_sliding_window(values, window=10)
    split = int(len(X) * 0.8)

    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])

    # Predictions should stay within ±3× the observed range of the training data
    obs_min, obs_max = values.min(), values.max()
    margin = (obs_max - obs_min) * 3
    assert preds.min() > obs_min - margin, "Predictions are unreasonably low"
    assert preds.max() < obs_max + margin, "Predictions are unreasonably high"
