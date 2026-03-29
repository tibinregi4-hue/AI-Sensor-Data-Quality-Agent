"""
tests/test_schema.py — Schema validation tests
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Verifies that the dataset has the expected structure:
  - Required columns present
  - Numeric columns are numeric dtype
  - Dataset is not empty
  - No fully-empty columns
"""

import pytest
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/sample_sensor_data.csv")


def load_data() -> pd.DataFrame:
    """Load the sample sensor dataset."""
    return pd.read_csv(DATA_FILE)


def test_data_file_exists():
    """Confirm the data file exists before running any other test."""
    assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"


def test_has_required_columns():
    """All required sensor columns must be present in the dataset."""
    df = load_data()
    required = ["timestamp", "temperature", "pressure"]
    for col in required:
        assert col in df.columns, f"Missing required column: '{col}'"


def test_numeric_columns_are_numeric():
    """The dataset must have at least 2 numeric columns."""
    df = load_data()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    assert len(numeric_cols) >= 2, (
        f"Expected at least 2 numeric columns, found {len(numeric_cols)}: {numeric_cols}"
    )


def test_not_empty():
    """Dataset must have at least 100 rows for meaningful analysis."""
    df = load_data()
    assert len(df) > 0,   "Dataset is empty."
    assert len(df) >= 100, f"Dataset too small: {len(df)} rows (minimum: 100)"


def test_no_fully_empty_columns():
    """No column should be completely null."""
    df = load_data()
    for col in df.columns:
        non_null = df[col].notna().sum()
        assert non_null > 0, f"Column '{col}' is completely empty (all NaN)"


def test_column_count_reasonable():
    """Dataset should have between 2 and 100 columns."""
    df = load_data()
    assert 2 <= len(df.columns) <= 100, (
        f"Unexpected column count: {len(df.columns)}"
    )
