"""
agent.py — AI Data Quality Agent
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Analyses validation reports and produces structured fix plans.
Works in two modes:
  1. LLM mode   — calls Claude API for intelligent diagnosis
  2. Offline mode — deterministic rule-based fallback (works without API key)
"""

import json
from typing import Any


class DataQualityAgent:
    """
    Analyses a validation report and recommends a structured fix plan.

    Parameters
    ----------
    validation_report : dict  — output of DataValidator.validate_all()
    config            : module — the config module
    """

    def __init__(self, validation_report: dict, config: Any):
        self.report = validation_report
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        """
        Run analysis and return a diagnosis dict.

        Returns
        -------
        {
            "diagnosis" : str  — natural-language description of findings,
            "fix_plan"  : list[dict] — ordered list of fix operations,
            "severity"  : "high" | "medium" | "low" | "none",
            "mode"      : "llm" | "rule_based"
        }
        """
        if self.config.USE_LLM and self.config.CLAUDE_API_KEY:
            try:
                return self._llm_analysis()
            except Exception as exc:
                print(f"        [Agent] LLM call failed ({exc}), falling back to rule-based.")
                return self._rule_based_analysis(note=str(exc))
        else:
            return self._rule_based_analysis()

    # ── LLM analysis ──────────────────────────────────────────────────────────

    def _llm_analysis(self) -> dict:
        """Send the validation report to Claude and parse the structured response."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.CLAUDE_API_KEY)

        system_prompt = (
            "You are a senior data quality engineer at Infineon Technologies. "
            "You receive structured validation reports about industrial sensor datasets "
            "and produce concise, actionable fix plans.\n\n"
            "Respond ONLY with a JSON object (no markdown, no preamble) with this schema:\n"
            "{\n"
            '  "diagnosis": "<1-2 sentence summary of the data quality situation>",\n'
            '  "severity": "<high|medium|low|none>",\n'
            '  "fix_plan": [\n'
            '    {"fix": "<fix_type>", "columns": [...], "reason": "<why>"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Valid fix types: remove_duplicates, fix_nulls, remove_outliers, "
            "fix_range_violations (needs column/lo/hi keys), fix_dtypes.\n"
            "Order fixes logically: duplicates first, then ranges, then nulls, then outliers."
        )

        user_prompt = (
            "Here is the data validation report:\n\n"
            + json.dumps(self.report, indent=2, default=str)
            + "\n\nProvide your JSON diagnosis and fix plan."
        )

        message = client.messages.create(
            model=self.config.MODEL_NAME,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        return {
            "diagnosis": parsed.get("diagnosis", "LLM diagnosis unavailable."),
            "fix_plan":  parsed.get("fix_plan", []),
            "severity":  parsed.get("severity", "medium"),
            "mode":      "llm",
        }

    # ── Rule-based fallback ──────────────────────────────────────────────────

    def _rule_based_analysis(self, note: str = "") -> dict:
        """
        Deterministic fallback — maps each issue type to a pre-defined fix.
        Works completely offline; no API key required.
        """
        issues   = self.report.get("issues", [])
        summary  = self.report.get("summary", {})
        fix_plan: list[dict] = []
        notes:    list[str]  = []

        # ── 1. Duplicates (always first) ─────────────────────────────────────
        dup = summary.get("duplicates", {})
        if dup.get("count", 0) > 0:
            n = dup["count"]
            fix_plan.append({"fix": "remove_duplicates",
                              "reason": f"Remove {n} exact duplicate rows to prevent data leakage."})
            notes.append(f"{n} duplicate rows")

        # ── 2. Range violations ───────────────────────────────────────────────
        range_violations = summary.get("ranges", {}).get("violations", {})
        if "pressure" in range_violations:
            n = range_violations["pressure"]
            fix_plan.append({
                "fix":    "fix_range_violations",
                "column": "pressure",
                "lo":     self.config.PRESSURE_MIN,
                "hi":     self.config.PRESSURE_MAX,
                "reason": f"Fix {n} pressure values outside physical range "
                          f"[{self.config.PRESSURE_MIN}, {self.config.PRESSURE_MAX}] hPa."
            })
            notes.append(f"{n} bad pressure values")

        if "temperature" in range_violations:
            n = range_violations["temperature"]
            fix_plan.append({
                "fix":    "fix_range_violations",
                "column": "temperature",
                "lo":     self.config.TEMPERATURE_MIN,
                "hi":     self.config.TEMPERATURE_MAX,
                "reason": f"Fix {n} temperature values outside physical range."
            })
            notes.append(f"{n} bad temperature values")

        # ── 3. Null imputation ────────────────────────────────────────────────
        null_summary = summary.get("nulls", {})
        failing_cols = null_summary.get("failing_cols", [])
        if failing_cols:
            null_pcts = null_summary.get("null_pcts", {})
            detail = ", ".join(f"{c}: {null_pcts.get(c, 0):.1f}%" for c in failing_cols)
            fix_plan.append({
                "fix":     "fix_nulls",
                "columns": failing_cols,
                "reason":  f"Median-impute missing values ({detail})."
            })
            notes.append(f"nulls in {failing_cols}")

        # ── 4. Outlier clipping ───────────────────────────────────────────────
        outlier_counts = summary.get("outliers", {}).get("counts", {})
        outlier_cols   = [c for c, n in outlier_counts.items() if n > 0]
        if outlier_cols:
            total = sum(outlier_counts[c] for c in outlier_cols)
            fix_plan.append({
                "fix":     "remove_outliers",
                "columns": outlier_cols,
                "sigma":   self.config.OUTLIER_SIGMA,
                "reason":  f"Clip {total} outlier values beyond {self.config.OUTLIER_SIGMA}σ."
            })
            notes.append(f"{total} outliers across {outlier_cols}")

        # ── 5. Drift — no auto-fix, just flag ────────────────────────────────
        drift_flagged = summary.get("drift", {}).get("flagged", [])
        if drift_flagged:
            notes.append(f"drift detected in {drift_flagged} — manual investigation recommended")

        # ── Severity ──────────────────────────────────────────────────────────
        critical_count = self.report.get("critical_count", 0)
        if critical_count >= 3:
            severity = "high"
        elif critical_count >= 1:
            severity = "medium"
        elif self.report.get("warning_count", 0) > 0:
            severity = "low"
        else:
            severity = "none"

        if notes:
            diagnosis = (
                f"{severity.capitalize()} severity — identified issues: "
                + "; ".join(notes) + "."
            )
        else:
            diagnosis = "No significant data quality issues detected. Dataset is clean."

        if note:
            diagnosis += f" (Offline mode: {note})"

        return {
            "diagnosis": diagnosis,
            "fix_plan":  fix_plan,
            "severity":  severity,
            "mode":      "rule_based",
        }
