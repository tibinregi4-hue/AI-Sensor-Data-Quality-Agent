"""
validator.py — Core data validation engine
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Loads any CSV, auto-detects columns, and runs:
  - Schema validation
  - Null-percentage check
  - Physical range check
  - Duplicate detection
  - Outlier detection (3σ)
  - Distribution drift detection
"""

import numpy as np
import pandas as pd
from typing import Any


class DataValidator:
    """
    Validates a pandas DataFrame against data-quality rules defined in config.

    Parameters
    ----------
    df     : pd.DataFrame — the dataset to validate
    config : module       — the config module (or any object with the threshold attrs)
    """

    def __init__(self, df: pd.DataFrame, config: Any):
        self.df      = df.copy()
        self.config  = config
        self.issues: list[dict] = []
        self.summary: dict      = {}
        self._numeric_cols: list[str] = df.select_dtypes(include="number").columns.tolist()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_all(self) -> dict:
        """
        Run all validation checks in sequence and return a unified results dict.

        Returns
        -------
        {
            "issues"  : list of issue dicts,
            "summary" : per-check summary dict,
            "passed"  : True if no critical issues found,
            "shape"   : (rows, cols) of the input DataFrame
        }
        """
        self.issues  = []
        self.summary = {}

        self.check_schema()
        self.check_nulls()
        self.check_ranges()
        self.check_duplicates()
        self.check_outliers()
        self.check_drift()

        critical = [i for i in self.issues if i.get("severity") == "critical"]
        warnings = [i for i in self.issues if i.get("severity") == "warning"]

        return {
            "issues":           self.issues,
            "summary":          self.summary,
            "passed":           len(critical) == 0,
            "shape":            self.df.shape,
            "critical_count":   len(critical),
            "warning_count":    len(warnings),
        }

    # ── Individual checks ────────────────────────────────────────────────────

    def check_schema(self):
        """Verify expected columns exist and numeric columns have numeric dtypes."""
        expected = {"timestamp", "temperature", "pressure"}
        present  = set(self.df.columns)
        missing  = expected - present
        extra    = present - expected

        result = {
            "check":   "schema",
            "columns": list(self.df.columns),
            "missing": list(missing),
            "extra":   list(extra),
            "dtypes":  {c: str(self.df[c].dtype) for c in self.df.columns},
        }

        if missing:
            self._add_issue(
                "schema",
                f"Missing expected columns: {sorted(missing)}",
                severity="critical",
                detail=result,
                fix_hint="add_columns",
            )
        else:
            result["status"] = "pass"

        self.summary["schema"] = result

    def check_nulls(self):
        """Flag columns whose null percentage exceeds the configured threshold."""
        null_info: dict[str, float] = {}
        failing: list[str] = []

        for col in self.df.columns:
            pct = float(self.df[col].isna().mean() * 100)
            null_info[col] = round(pct, 2)
            if pct >= self.config.MAX_NULL_PERCENT:
                failing.append(col)
                self._add_issue(
                    "nulls",
                    f"Column '{col}' has {pct:.1f}% null values (threshold: {self.config.MAX_NULL_PERCENT}%)",
                    severity="critical",
                    detail={"column": col, "null_pct": pct, "null_count": int(self.df[col].isna().sum())},
                    fix_hint="fix_nulls",
                )

        self.summary["nulls"] = {
            "check":       "nulls",
            "null_pcts":   null_info,
            "failing_cols": failing,
            "status":      "pass" if not failing else "fail",
        }

    def check_ranges(self):
        """Check known sensor columns for physically impossible values."""
        range_rules = {}
        if "temperature" in self.df.columns:
            range_rules["temperature"] = (self.config.TEMPERATURE_MIN, self.config.TEMPERATURE_MAX)
        if "pressure" in self.df.columns:
            range_rules["pressure"] = (self.config.PRESSURE_MIN, self.config.PRESSURE_MAX)

        violations: dict[str, int] = {}
        for col, (lo, hi) in range_rules.items():
            valid = self.df[col].dropna()
            n_bad = int(((valid < lo) | (valid > hi)).sum())
            if n_bad > 0:
                violations[col] = n_bad
                self._add_issue(
                    "ranges",
                    f"Column '{col}' has {n_bad} values outside [{lo}, {hi}]",
                    severity="critical",
                    detail={"column": col, "min_allowed": lo, "max_allowed": hi,
                            "violations": n_bad,
                            "actual_min": float(valid.min()), "actual_max": float(valid.max())},
                    fix_hint="remove_outliers",
                )

        self.summary["ranges"] = {
            "check":      "ranges",
            "rules":      {c: list(r) for c, r in range_rules.items()},
            "violations": violations,
            "status":     "pass" if not violations else "fail",
        }

    def check_duplicates(self):
        """Count and flag exact duplicate rows."""
        n_dupes = int(self.df.duplicated().sum())
        if n_dupes > 0:
            self._add_issue(
                "duplicates",
                f"Found {n_dupes} exact duplicate rows",
                severity="critical",
                detail={"count": n_dupes},
                fix_hint="remove_duplicates",
            )

        self.summary["duplicates"] = {
            "check":  "duplicates",
            "count":  n_dupes,
            "status": "pass" if n_dupes == 0 else "fail",
        }

    def check_outliers(self):
        """Flag values that fall beyond ±3σ for each numeric column."""
        outlier_info: dict[str, int] = {}

        for col in self._numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 10:
                continue
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                continue
            n_out = int(((series - mu).abs() > self.config.OUTLIER_SIGMA * sigma).sum())
            outlier_info[col] = n_out
            if n_out > 0:
                self._add_issue(
                    "outliers",
                    f"Column '{col}' has {n_out} outliers beyond {self.config.OUTLIER_SIGMA}σ",
                    severity="warning",
                    detail={"column": col, "count": n_out,
                            "mean": round(float(mu), 4), "std": round(float(sigma), 4)},
                    fix_hint="remove_outliers",
                )

        self.summary["outliers"] = {
            "check":        "outliers",
            "counts":       outlier_info,
            "sigma":        self.config.OUTLIER_SIGMA,
            "status":       "pass" if all(v == 0 for v in outlier_info.values()) else "warning",
        }

    def check_drift(self):
        """
        Compare the first-half vs second-half distribution of numeric columns.
        Flags if the mean shifts by more than DRIFT_THRESHOLD standard deviations.
        """
        drift_info: dict[str, dict] = {}
        flagged: list[str] = []
        n = len(self.df)
        half = n // 2

        for col in self._numeric_cols:
            first  = self.df[col].iloc[:half].dropna()
            second = self.df[col].iloc[half:].dropna()
            if len(first) < 10 or len(second) < 10:
                continue

            mu1, sd1 = first.mean(), first.std()
            mu2, _   = second.mean(), second.std()

            if sd1 == 0:
                continue

            shift = abs(mu2 - mu1) / sd1
            drift_info[col] = {"mean_first": round(float(mu1), 4),
                                "mean_second": round(float(mu2), 4),
                                "shift_sigma": round(float(shift), 3)}

            if shift > self.config.DRIFT_THRESHOLD:
                flagged.append(col)
                self._add_issue(
                    "drift",
                    f"Column '{col}' shows distribution drift: mean shifted {shift:.2f}σ",
                    severity="warning",
                    detail=drift_info[col],
                    fix_hint="investigate_drift",
                )

        self.summary["drift"] = {
            "check":   "drift",
            "details": drift_info,
            "flagged": flagged,
            "status":  "pass" if not flagged else "warning",
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _add_issue(self, check: str, message: str, severity: str,
                   detail: dict, fix_hint: str):
        """Append a structured issue record to the issue list."""
        self.issues.append({
            "check":    check,
            "message":  message,
            "severity": severity,
            "detail":   detail,
            "fix_hint": fix_hint,
        })
