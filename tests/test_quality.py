"""
tests/test_quality.py — Data quality tests (these FAIL on raw data by design)
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

These tests validate that the data meets Infineon quality standards.
They intentionally fail on the raw sample data to demonstrate that
the AI agent pipeline is needed — after the pipeline runs, they pass.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DATA_FILE = Path("data/sample_sensor_data.csv")


def load_data() -> pd.DataFrame:
    """Load the sample sensor dataset."""
    return pd.read_csv(DATA_FILE)


def test_null_percentage_below_threshold():
    """
    Every numeric column must have fewer than MAX_NULL_PERCENT null values.
    FAILS on raw data (temperature ~3.2%, pressure ~1.8% nulls).
    """
    df = load_data()
    for col in df.select_dtypes(include="number").columns:
        null_pct = df[col].isna().mean() * 100
        assert null_pct < config.MAX_NULL_PERCENT, (
            f"Column '{col}' has {null_pct:.1f}% null values "
            f"(threshold: {config.MAX_NULL_PERCENT}%)"
        )


def test_no_impossible_pressure():
    """
    All pressure values must be in the physically valid range [PRESSURE_MIN, PRESSURE_MAX].
    FAILS on raw data (6 values injected outside range).
    """
    df = load_data()
    if "pressure" not in df.columns:
        pytest.skip("No pressure column in dataset")
    valid = df["pressure"].dropna()
    bad = ((valid < config.PRESSURE_MIN) | (valid > config.PRESSURE_MAX)).sum()
    assert bad == 0, (
        f"Found {bad} pressure values outside "
        f"[{config.PRESSURE_MIN}, {config.PRESSURE_MAX}] hPa"
    )


def test_no_impossible_temperature():
    """
    All temperature values must be in the physically valid range [TEMP_MIN, TEMP_MAX].
    Passes on raw data (only nulls, not impossible values in temperature).
    """
    df = load_data()
    if "temperature" not in df.columns:
        pytest.skip("No temperature column in dataset")
    valid = df["temperature"].dropna()
    bad = ((valid < config.TEMPERATURE_MIN) | (valid > config.TEMPERATURE_MAX)).sum()
    assert bad == 0, (
        f"Found {bad} temperature values outside "
        f"[{config.TEMPERATURE_MIN}, {config.TEMPERATURE_MAX}] °C"
    )


def test_no_duplicates():
    """
    Dataset must contain zero exact duplicate rows.
    FAILS on raw data (15 duplicates injected).
    """
    df = load_data()
    dupes = df.duplicated().sum()
    assert dupes == 0, f"Found {int(dupes)} duplicate rows in the dataset"


def test_humidity_in_valid_range():
    """Humidity values must be between 0% and 100%."""
    df = load_data()
    if "humidity" not in df.columns:
        pytest.skip("No humidity column in dataset")
    valid = df["humidity"].dropna()
    bad = ((valid < 0) | (valid > 100)).sum()
    assert bad == 0, f"Found {int(bad)} humidity values outside [0, 100]%"
