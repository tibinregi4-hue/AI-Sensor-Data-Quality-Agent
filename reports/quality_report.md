# Data Quality Report
> Generated: 2026-03-29T21:56:33  |  File: `data\sample_sensor_data.csv`
> Author: Tibin Regi  |  Project: Infineon AI & Data Engineering Internship

---

## 1. Initial Validation

| Check | Status | Detail |
|-------|--------|--------|
| Schema | ✅ PASS | 7 columns, all valid |
| Nulls | ❌ FAIL | temperature: 6.5% |
| Ranges | ❌ FAIL | pressure: 6 bad values |
| Duplicates | ❌ FAIL | 15 duplicate rows |
| Outliers | ⚠️ WARN | pressure: 6, voltage: 8 |
| Drift | ✅ PASS | distributions stable |

**Result:** ❌ FAILED  — 3 critical issues, 2 warnings

---

## 2. AI Agent Diagnosis

**Mode:** rule_based

**Severity:** HIGH

**Diagnosis:** High severity — identified issues: 15 duplicate rows; 6 bad pressure values; nulls in ['temperature']; 14 outliers across ['pressure', 'voltage'].

### Fix Plan

1. Remove 15 exact duplicate rows to prevent data leakage.
2. Fix 6 pressure values outside physical range [900, 1100] hPa.
3. Median-impute missing values (temperature: 6.5%).
4. Clip 14 outlier values beyond 3.0σ.

---

## 3. Fixes Applied

| Action | Column | Count | Detail |
|--------|--------|-------|--------|
| remove_duplicates | — | 15 | Exact duplicate rows removed |
| fix_range_violations | pressure | 6 | Range [900, 1100], imputed with 1013.125 |
| fix_nulls | temperature | 325 | Median = 23.38 |
| remove_outliers | pressure | 24 | Clipped to [983.65, 1042.55] |
| remove_outliers | voltage | 8 | Clipped to [2.81, 3.79] |

Rows before: **5015**  →  Rows after: **5000**

---

## 4. Post-Fix Validation

| Check | Status | Detail |
|-------|--------|--------|
| Schema | ✅ PASS | 7 columns, all valid |
| Nulls | ✅ PASS | 0 missing |
| Ranges | ✅ PASS | all values in range |
| Duplicates | ✅ PASS | 0 duplicates |
| Outliers | ⚠️ WARN | pressure: 24, voltage: 17 |
| Drift | ✅ PASS | distributions stable |

**Result:** ✅ ALL CHECKS PASSED

---

## 5. Model Evaluation

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| RMSE | 1.4936 | < 5.0 | ✅ PASS |
| R² | 0.9043 | > 0.5 | ✅ PASS |
| MAE | 0.826 | — | — |
| Train samples | 3992 | — | — |
| Test samples  | 998 | — | — |

---

*Built by Tibin Regi as a demonstration project for the Infineon AI & Data Engineering Internship.*