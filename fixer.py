"""
fixer.py — Data-cleaning tools invoked by the AI agent
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Provides a DataFixer class that applies structured fix plans returned
by the DataQualityAgent.  Every method logs what it changed so the
report generator can document the full provenance of the clean dataset.
"""

import numpy as np
import pandas as pd
from typing import Any


class DataFixer:
    """
    Applies data-cleaning operations to a DataFrame, logging every action.

    Parameters
    ----------
    df : pd.DataFrame — the raw dataset to clean (copied on init)
    """

    def __init__(self, df: pd.DataFrame):
        self.df_original     = df.copy()
        self.df              = df.copy()
        self.actions_taken:  list[dict] = []
        self._rows_before    = len(df)

    # ── Public fix operations ────────────────────────────────────────────────

    def fix_nulls(self, columns: list[str] | None = None):
        """
        Fill missing values in numeric columns with the column median.

        Parameters
        ----------
        columns : list of column names to impute; defaults to all numeric columns
        """
        if columns is None:
            columns = self.df.select_dtypes(include="number").columns.tolist()

        for col in columns:
            if col not in self.df.columns:
                continue
            n_missing = int(self.df[col].isna().sum())
            if n_missing == 0:
                continue
            median_val = float(self.df[col].median())
            self.df[col] = self.df[col].fillna(median_val)
            self.actions_taken.append({
                "action":  "fix_nulls",
                "column":  col,
                "filled":  n_missing,
                "value":   round(median_val, 4),
                "method":  "median_imputation",
            })

    def remove_outliers(self, columns: list[str] | None = None, sigma: float = 3.0):
        """
        Clip values beyond ±sigma standard deviations to the boundary value.

        Parameters
        ----------
        columns : columns to clip; defaults to all numeric columns
        sigma   : number of standard deviations defining the boundary
        """
        if columns is None:
            columns = self.df.select_dtypes(include="number").columns.tolist()

        for col in columns:
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            if len(series) < 10:
                continue
            mu, std = series.mean(), series.std()
            if std == 0:
                continue
            lo, hi = mu - sigma * std, mu + sigma * std
            n_clipped = int(((self.df[col] < lo) | (self.df[col] > hi)).sum())
            if n_clipped == 0:
                continue
            self.df[col] = self.df[col].clip(lower=lo, upper=hi)
            self.actions_taken.append({
                "action":   "remove_outliers",
                "column":   col,
                "clipped":  n_clipped,
                "lower":    round(lo, 4),
                "upper":    round(hi, 4),
                "sigma":    sigma,
            })

    def remove_duplicates(self):
        """Drop exact duplicate rows, keeping the first occurrence."""
        n_before  = len(self.df)
        self.df   = self.df.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(self.df)
        if n_removed > 0:
            self.actions_taken.append({
                "action":  "remove_duplicates",
                "removed": n_removed,
            })

    def fix_dtypes(self):
        """
        Attempt to coerce object columns that look numeric into float64.
        Silently skips columns that cannot be converted.
        """
        for col in self.df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")
            if converted.notna().sum() > 0.5 * len(self.df):
                self.df[col] = converted
                self.actions_taken.append({
                    "action": "fix_dtypes",
                    "column": col,
                    "new_dtype": "float64",
                })

    def fix_range_violations(self, column: str, lo: float, hi: float):
        """
        Replace values outside [lo, hi] with NaN, then impute with median.

        Parameters
        ----------
        column : name of the column to fix
        lo     : minimum physically valid value
        hi     : maximum physically valid value
        """
        if column not in self.df.columns:
            return
        bad_mask = (self.df[column] < lo) | (self.df[column] > hi)
        n_bad    = int(bad_mask.sum())
        if n_bad == 0:
            return
        self.df.loc[bad_mask, column] = np.nan
        median_val = float(self.df[column].median())
        self.df[column] = self.df[column].fillna(median_val)
        self.actions_taken.append({
            "action":  "fix_range_violations",
            "column":  column,
            "fixed":   n_bad,
            "range":   [lo, hi],
            "imputed_with": round(median_val, 4),
        })

    # ── Fix-plan executor ────────────────────────────────────────────────────

    def apply_fix_plan(self, fix_plan: list[dict]) -> pd.DataFrame:
        """
        Execute a structured fix plan as produced by DataQualityAgent.

        Each item in fix_plan is a dict with at least a ``"fix"`` key:
          - "remove_duplicates"
          - "fix_nulls"          (optional: "columns")
          - "remove_outliers"    (optional: "columns", "sigma")
          - "fix_range_violations" (requires: "column", "lo", "hi")
          - "fix_dtypes"

        Parameters
        ----------
        fix_plan : list[dict] — the agent's ordered list of fixes

        Returns
        -------
        pd.DataFrame — the cleaned DataFrame
        """
        for step in fix_plan:
            fix_type = step.get("fix", "")

            if fix_type == "remove_duplicates":
                self.remove_duplicates()

            elif fix_type == "fix_nulls":
                cols = step.get("columns", None)
                self.fix_nulls(columns=cols)

            elif fix_type == "remove_outliers":
                cols  = step.get("columns", None)
                sigma = step.get("sigma", 3.0)
                self.remove_outliers(columns=cols, sigma=sigma)

            elif fix_type == "fix_range_violations":
                col = step.get("column")
                lo  = step.get("lo")
                hi  = step.get("hi")
                if col and lo is not None and hi is not None:
                    self.fix_range_violations(col, lo, hi)

            elif fix_type == "fix_dtypes":
                self.fix_dtypes()

        return self.df

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_report(self) -> dict:
        """
        Return a summary of all fix operations applied, including row counts.

        Returns
        -------
        dict with keys: actions, rows_before, rows_after, rows_removed
        """
        return {
            "actions":      self.actions_taken,
            "rows_before":  self._rows_before,
            "rows_after":   len(self.df),
            "rows_removed": self._rows_before - len(self.df),
        }
