"""
report_generator.py — Generates JSON, Markdown, and terminal reports
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project
"""

import json
import datetime
from pathlib import Path
from typing import Any


# ── ANSI colour helpers ───────────────────────────────────────────────────────
class C:
    """Minimal ANSI colour codes for terminal output."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"

def _ok(text: str)   -> str: return f"{C.GREEN}{text}{C.RESET}"
def _fail(text: str) -> str: return f"{C.RED}{text}{C.RESET}"
def _warn(text: str) -> str: return f"{C.YELLOW}{text}{C.RESET}"
def _bold(text: str) -> str: return f"{C.BOLD}{text}{C.RESET}"
def _cyan(text: str) -> str: return f"{C.CYAN}{text}{C.RESET}"


class ReportGenerator:
    """
    Aggregates all pipeline results into structured reports.

    Parameters
    ----------
    validation_before : dict  — raw validation result (pre-fix)
    agent_diagnosis   : dict  — agent's diagnosis + fix plan
    fix_report        : dict  — fixer's action log
    validation_after  : dict  — re-validation result (post-fix)
    model_metrics     : dict  — ModelEvaluator results
    data_path         : str   — path to the input file (for metadata)
    """

    def __init__(
        self,
        validation_before: dict,
        agent_diagnosis:   dict,
        fix_report:        dict,
        validation_after:  dict,
        model_metrics:     dict,
        data_path:         str = "",
    ):
        self.before   = validation_before
        self.diagnosis = agent_diagnosis
        self.fixes     = fix_report
        self.after     = validation_after
        self.model     = model_metrics
        self.data_path = str(data_path)
        self.ts        = datetime.datetime.now().isoformat(timespec="seconds")
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_all(self) -> dict[str, Path]:
        """Save JSON and Markdown reports; return paths."""
        paths = {
            "json":     self._save_json(),
            "markdown": self._save_markdown(),
        }
        return paths

    def print_validation_summary(self, result: dict, label: str = ""):
        """Print a formatted validation summary to the terminal."""
        summary = result.get("summary", {})
        if label:
            print(f"\n        {'─'*40}")
            print(f"        {_bold(label)}")

        checks = [
            ("schema",     "Schema check     "),
            ("nulls",      "Null check       "),
            ("ranges",     "Range check      "),
            ("duplicates", "Duplicate check  "),
            ("outliers",   "Outlier check    "),
            ("drift",      "Drift check      "),
        ]
        for key, label_str in checks:
            chk = summary.get(key, {})
            status = chk.get("status", "unknown")
            if status == "pass":
                indicator = _ok("PASS")
            elif status == "fail":
                indicator = _fail("FAIL")
            elif status == "warning":
                indicator = _warn("WARNING")
            else:
                indicator = _warn("SKIP")

            detail = self._check_detail(key, chk)
            print(f"        {label_str}: {indicator}  {C.DIM}{detail}{C.RESET}")

        critical = result.get("critical_count", 0)
        warnings = result.get("warning_count", 0)
        if critical == 0 and warnings == 0:
            print(f"\n        {_ok(_bold('ALL CHECKS PASSED'))}")
        else:
            print(f"\n        Issues found: {_fail(str(critical) + ' critical')}, "
                  f"{_warn(str(warnings) + ' warning')}")

    def print_agent_summary(self):
        """Print the agent's diagnosis and fix plan."""
        d = self.diagnosis
        sev = d.get("severity", "unknown")
        mode = d.get("mode", "rule_based")
        sev_str = {"high": _fail(sev), "medium": _warn(sev),
                   "low": _ok(sev), "none": _ok(sev)}.get(sev, sev)

        print(f"        Diagnosis: {sev_str} severity — {d.get('diagnosis', '')}")
        print(f"        Mode: {_cyan(mode)}")
        plan = d.get("fix_plan", [])
        if plan:
            print("        Fix plan:")
            for i, step in enumerate(plan, 1):
                reason = step.get("reason", step.get("fix", ""))
                print(f"          {i}. {reason}")
        else:
            print("        No fixes required.")

    def print_fixer_summary(self):
        """Print what the fixer actually changed."""
        actions = self.fixes.get("actions", [])
        before  = self.fixes.get("rows_before", "?")
        after   = self.fixes.get("rows_after",  "?")

        if not actions:
            print("        No changes applied.")
            return

        for act in actions:
            fix = act.get("action", "")
            if fix == "remove_duplicates":
                print(f"        Removed {act['removed']} duplicate rows")
            elif fix == "fix_nulls":
                print(f"        Filled {act['filled']} missing values in '{act['column']}' "
                      f"(median: {act['value']})")
            elif fix == "remove_outliers":
                print(f"        Clipped {act['clipped']} outliers in '{act['column']}' "
                      f"[{act['lower']:.2f}, {act['upper']:.2f}]")
            elif fix == "fix_range_violations":
                print(f"        Fixed {act['fixed']} range violations in '{act['column']}'")
            elif fix == "fix_dtypes":
                print(f"        Converted '{act['column']}' dtype to {act['new_dtype']}")

        print(f"        Rows: {before} → {after}  "
              f"({_ok(str(before - after) + ' removed') if before > after else _ok('unchanged')})")

    def print_model_summary(self):
        """Print model evaluation metrics."""
        m = self.model
        if m.get("error"):
            print(f"        {_warn('Skipped:')} {m['error']}")
            return

        rmse = m.get("rmse", 0)
        r2   = m.get("r2",   0)
        mae  = m.get("mae",  0)
        gates = m.get("gates", {})

        rmse_ok = gates.get("rmse_pass", False)
        r2_ok   = gates.get("r2_pass",  False)

        print(f"        Training samples  : {m.get('train_samples', '?')}")
        print(f"        Testing samples   : {m.get('test_samples',  '?')}")
        print(f"        Target column     : {m.get('target', '?')}")
        print(f"        RMSE              : {rmse:.4f}  "
              f"(threshold: {gates.get('rmse_threshold', '?')})  "
              + (_ok("PASS") if rmse_ok else _fail("FAIL")))
        print(f"        R²                : {r2:.4f}  "
              f"(threshold: {gates.get('r2_threshold', '?')})  "
              + (_ok("PASS") if r2_ok else _fail("FAIL")))
        print(f"        MAE               : {mae:.4f}")

    # ── JSON report ───────────────────────────────────────────────────────────

    def _save_json(self) -> Path:
        """Write the full results as a structured JSON file."""
        payload = {
            "metadata": {
                "generated_at": self.ts,
                "data_path":    self.data_path,
                "author":       "Tibin Regi",
                "project":      "AI-Powered Data Quality Agent",
            },
            "validation_before": self.before,
            "agent_diagnosis":   self.diagnosis,
            "fix_report":        self.fixes,
            "validation_after":  self.after,
            "model_metrics":     self.model,
        }
        path = self.reports_dir / "quality_report.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    # ── Markdown report ───────────────────────────────────────────────────────

    def _save_markdown(self) -> Path:
        """Write a human-readable Markdown summary report."""
        lines = [
            "# Data Quality Report",
            f"> Generated: {self.ts}  |  File: `{self.data_path}`",
            f"> Author: Tibin Regi  |  Project: Infineon AI & Data Engineering Internship",
            "",
            "---",
            "",
            "## 1. Initial Validation",
            "",
            self._md_validation_table(self.before),
            "",
            f"**Result:** {'✅ PASSED' if self.before.get('passed') else '❌ FAILED'}  "
            f"— {self.before.get('critical_count', 0)} critical issues, "
            f"{self.before.get('warning_count', 0)} warnings",
            "",
            "---",
            "",
            "## 2. AI Agent Diagnosis",
            "",
            f"**Mode:** {self.diagnosis.get('mode', 'unknown')}",
            "",
            f"**Severity:** {self.diagnosis.get('severity', 'unknown').upper()}",
            "",
            f"**Diagnosis:** {self.diagnosis.get('diagnosis', '')}",
            "",
            "### Fix Plan",
            "",
        ]
        plan = self.diagnosis.get("fix_plan", [])
        if plan:
            for i, step in enumerate(plan, 1):
                reason = step.get("reason", step.get("fix", ""))
                lines.append(f"{i}. {reason}")
        else:
            lines.append("No fixes required.")

        lines += [
            "",
            "---",
            "",
            "## 3. Fixes Applied",
            "",
            self._md_fix_table(),
            "",
            f"Rows before: **{self.fixes.get('rows_before', '?')}**  "
            f"→  Rows after: **{self.fixes.get('rows_after', '?')}**",
            "",
            "---",
            "",
            "## 4. Post-Fix Validation",
            "",
            self._md_validation_table(self.after),
            "",
            f"**Result:** {'✅ ALL CHECKS PASSED' if self.after.get('passed') else '⚠️ Issues remain'}",
            "",
            "---",
            "",
            "## 5. Model Evaluation",
            "",
            self._md_model_table(),
            "",
            "---",
            "",
            "*Built by Tibin Regi as a demonstration project for the Infineon AI & Data Engineering Internship.*",
        ]

        path = self.reports_dir / "quality_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Markdown helpers ──────────────────────────────────────────────────────

    def _md_validation_table(self, result: dict) -> str:
        summary = result.get("summary", {})
        rows = [
            "| Check | Status | Detail |",
            "|-------|--------|--------|",
        ]
        for key in ("schema", "nulls", "ranges", "duplicates", "outliers", "drift"):
            chk    = summary.get(key, {})
            status = chk.get("status", "?")
            icon   = {"pass": "✅ PASS", "fail": "❌ FAIL",
                      "warning": "⚠️ WARN"}.get(status, status)
            detail = self._check_detail(key, chk)
            rows.append(f"| {key.capitalize()} | {icon} | {detail} |")
        return "\n".join(rows)

    def _md_fix_table(self) -> str:
        actions = self.fixes.get("actions", [])
        if not actions:
            return "_No fixes were applied._"
        rows = [
            "| Action | Column | Count | Detail |",
            "|--------|--------|-------|--------|",
        ]
        for act in actions:
            fix = act.get("action", "")
            if fix == "remove_duplicates":
                rows.append(f"| remove_duplicates | — | {act['removed']} | Exact duplicate rows removed |")
            elif fix == "fix_nulls":
                rows.append(f"| fix_nulls | {act['column']} | {act['filled']} | Median = {act['value']} |")
            elif fix == "remove_outliers":
                rows.append(f"| remove_outliers | {act['column']} | {act['clipped']} | "
                             f"Clipped to [{act['lower']:.2f}, {act['upper']:.2f}] |")
            elif fix == "fix_range_violations":
                rows.append(f"| fix_range_violations | {act['column']} | {act['fixed']} | "
                             f"Range {act['range']}, imputed with {act['imputed_with']} |")
            elif fix == "fix_dtypes":
                rows.append(f"| fix_dtypes | {act['column']} | — | Converted to {act['new_dtype']} |")
        return "\n".join(rows)

    def _md_model_table(self) -> str:
        m = self.model
        if m.get("error"):
            return f"_Skipped: {m['error']}_"
        gates = m.get("gates", {})
        return (
            f"| Metric | Value | Threshold | Status |\n"
            f"|--------|-------|-----------|--------|\n"
            f"| RMSE | {m.get('rmse', '?')} | < {gates.get('rmse_threshold', '?')} | "
            f"{'✅ PASS' if gates.get('rmse_pass') else '❌ FAIL'} |\n"
            f"| R² | {m.get('r2', '?')} | > {gates.get('r2_threshold', '?')} | "
            f"{'✅ PASS' if gates.get('r2_pass') else '❌ FAIL'} |\n"
            f"| MAE | {m.get('mae', '?')} | — | — |\n"
            f"| Train samples | {m.get('train_samples', '?')} | — | — |\n"
            f"| Test samples  | {m.get('test_samples', '?')} | — | — |"
        )

    # ── Detail strings ─────────────────────────────────────────────────────────

    @staticmethod
    def _check_detail(key: str, chk: dict) -> str:
        """Return a short human-readable detail string for a check result."""
        if key == "schema":
            missing = chk.get("missing", [])
            cols    = chk.get("columns", [])
            if missing:
                return f"Missing: {missing}"
            return f"{len(cols)} columns, all valid"
        elif key == "nulls":
            failing = chk.get("failing_cols", [])
            pcts    = chk.get("null_pcts", {})
            if failing:
                return ", ".join(f"{c}: {pcts.get(c,0):.1f}%" for c in failing)
            return "0 missing"
        elif key == "ranges":
            v = chk.get("violations", {})
            if v:
                return ", ".join(f"{c}: {n} bad values" for c, n in v.items())
            return "all values in range"
        elif key == "duplicates":
            n = chk.get("count", 0)
            return f"{n} duplicate rows" if n else "0 duplicates"
        elif key == "outliers":
            counts = {c: n for c, n in chk.get("counts", {}).items() if n > 0}
            if counts:
                return ", ".join(f"{c}: {n}" for c, n in counts.items())
            return "0 outliers"
        elif key == "drift":
            flagged = chk.get("flagged", [])
            return f"drift in {flagged}" if flagged else "distributions stable"
        return ""
