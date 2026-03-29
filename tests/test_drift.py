"""
tests/test_drift.py — Distribution drift / stability tests
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Checks that sensor readings don't show sudden distribution shifts
that could indicate sensor malfunction or data corruption.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/sample_sensor_data.csv")


def load_data() -> pd.DataFrame:
    """Load the sample sensor dataset."""
    return pd.read_csv(DATA_FILE)


def test_temperature_distribution_stable():
    """
    Temperature mean must be within a physically reasonable range
    and standard deviation must not be extreme.
    """
    df = load_data()
    if "temperature" not in df.columns:
        pytest.skip("No temperature column")

    temps = df["temperature"].dropna()
    mean  = temps.mean()
    std   = temps.std()

    assert -30 < mean < 50, (
        f"Temperature mean {mean:.1f}°C seems unreasonable "
        f"(expected between -30 and 50)"
    )
    assert std < 30, (
        f"Temperature std {std:.1f}°C is too high — possible data corruption or drift"
    )


def test_no_sudden_value_jumps():
    """
    Consecutive values in numeric columns should not jump by more than
    50× the mean step size — extreme jumps indicate sensor faults or corruption.
    """
    df = load_data()
    for col in df.select_dtypes(include="number").columns:
        values = df[col].dropna().values
        if len(values) < 10:
            continue
        diffs    = np.abs(np.diff(values))
        mean_diff = diffs.mean()
        max_diff  = diffs.max()
        if mean_diff > 0:
            assert max_diff < mean_diff * 50, (
                f"Column '{col}' has a sudden jump of {max_diff:.2f} "
                f"(mean diff: {mean_diff:.4f}) — possible sensor fault"
            )


def test_first_half_vs_second_half_temperature():
    """
    The mean temperature in the first half of the dataset should be within
    10°C of the mean in the second half (no severe temporal drift).
    """
    df = load_data()
    if "temperature" not in df.columns:
        pytest.skip("No temperature column")

    temps  = df["temperature"].dropna().reset_index(drop=True)
    half   = len(temps) // 2
    first  = temps.iloc[:half]
    second = temps.iloc[half:]

    diff = abs(first.mean() - second.mean())
    assert diff < 10, (
        f"Temperature mean shifted by {diff:.2f}°C between first and second half "
        f"— possible drift or sensor recalibration event"
    )


def test_pressure_distribution_stable():
    """
    Pressure standard deviation should not exceed 50 hPa
    (physically, global surface pressure varies ~900–1100 hPa).
    """
    df = load_data()
    if "pressure" not in df.columns:
        pytest.skip("No pressure column")

    pressure = df["pressure"].dropna()
    assert pressure.std() < 50, (
        f"Pressure std = {pressure.std():.2f} hPa — unusually high, "
        f"possible corruption or outliers"
    )
