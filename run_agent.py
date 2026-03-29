"""
run_agent.py — Main entry point for the Infineon AI Data Quality Agent
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project

Usage:
    python run_agent.py                          # uses default sample data
    python run_agent.py data/your_file.csv       # use any CSV file

Pipeline:
    1. Load CSV
    2. Validate (schema, nulls, ranges, duplicates, outliers, drift)
    3. AI Agent diagnoses issues & builds fix plan
    4. Fixer applies the plan
    5. Re-validate cleaned data
    6. Train GradientBoosting model & evaluate quality gates
    7. Generate JSON + Markdown reports
"""

import sys
import pandas as pd
from pathlib import Path

from validator      import DataValidator
from agent          import DataQualityAgent
from fixer          import DataFixer
from model_evaluator import ModelEvaluator
from report_generator import ReportGenerator
import config


# ── Colour helpers (duplicated here to avoid circular import) ─────────────────
_RESET = "\033[0m"; _BOLD = "\033[1m"; _GREEN = "\033[92m"
_RED   = "\033[91m"; _CYAN = "\033[96m"; _WHITE = "\033[97m"

def _ok(t):   return f"{_GREEN}{t}{_RESET}"
def _fail(t): return f"{_RED}{t}{_RESET}"
def _bold(t): return f"{_BOLD}{t}{_RESET}"
def _cyan(t): return f"{_CYAN}{t}{_RESET}"


def main(data_path: str | None = None):
    """Run the full data quality pipeline on the given CSV file."""

    if data_path is None:
        data_path = config.DEFAULT_DATA_PATH

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"\n  {_fail('ERROR:')} File not found: {data_path}")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {_bold('Infineon AI Data Quality Agent')}")
    print(f"  {_bold('Built by Tibin Regi')}")
    print(f"  {'─' * 50}")
    print(f"  LLM mode: {_ok('ENABLED (Claude)') if config.USE_LLM else _fail('OFFLINE (rule-based fallback)')}")
    print("=" * 62)

    # ── Step 0: Load data ─────────────────────────────────────────────────────
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        print(f"\n  {_fail('ERROR loading CSV:')} {exc}")
        sys.exit(1)

    print(f"\n  Loaded: {_cyan(str(data_path))}  "
          f"({len(df):,} rows, {len(df.columns)} columns)")
    print(f"  Columns: {list(df.columns)}")

    # ── Step 1: Validate ──────────────────────────────────────────────────────
    print(f"\n  {_bold('[1/5] Validating data quality...')}")
    validator       = DataValidator(df, config)
    validation_before = validator.validate_all()
    
    reporter = ReportGenerator(
        validation_before=validation_before,
        agent_diagnosis={},
        fix_report={},
        validation_after={},
        model_metrics={},
        data_path=str(data_path),
    )
    reporter.print_validation_summary(validation_before)

    # ── Step 2: AI Agent ──────────────────────────────────────────────────────
    print(f"\n  {_bold('[2/5] AI Agent analyzing issues...')}")
    agent     = DataQualityAgent(validation_before, config)
    diagnosis = agent.analyze()
    reporter.diagnosis = diagnosis
    reporter.print_agent_summary()

    # ── Step 3: Apply fixes ───────────────────────────────────────────────────
    print(f"\n  {_bold('[3/5] Applying fixes...')}")
    fixer    = DataFixer(df)
    clean_df = fixer.apply_fix_plan(diagnosis["fix_plan"])
    fix_report = fixer.get_report()
    reporter.fixes = fix_report
    reporter.print_fixer_summary()

    # ── Step 4: Re-validate ───────────────────────────────────────────────────
    print(f"\n  {_bold('[4/5] Re-validating cleaned data...')}")
    validator2       = DataValidator(clean_df, config)
    validation_after = validator2.validate_all()
    reporter.after   = validation_after
    reporter.print_validation_summary(validation_after)

    # ── Step 5: Train & evaluate model ────────────────────────────────────────
    print(f"\n  {_bold('[5/5] Training AI model...')}")
    evaluator     = ModelEvaluator(clean_df, config=config)
    model_metrics = evaluator.evaluate()
    reporter.model = model_metrics
    reporter.print_model_summary()

    # ── Step 6: Save reports ──────────────────────────────────────────────────
    paths = reporter.save_all()
    print(f"\n  Reports saved:")
    for fmt, path in paths.items():
        print(f"    {_cyan(str(path))}")

    # ── Footer ────────────────────────────────────────────────────────────────
    all_passed = (
        validation_after.get("passed", False)
        and model_metrics.get("passed", False)
    )
    print("\n" + "=" * 62)
    if all_passed:
        print(f"  {_ok(_bold('PIPELINE COMPLETE — ALL QUALITY GATES PASSED ✓'))}")
    else:
        remaining = validation_after.get("critical_count", 0)
        model_ok  = model_metrics.get("passed", False)
        msg_parts = []
        if remaining:
            msg_parts.append(f"{remaining} validation issue(s) remain")
        if not model_ok and not model_metrics.get("error"):
            msg_parts.append("model quality gates not met")
        print(f"  {_fail(_bold('PIPELINE COMPLETE — ' + '; '.join(msg_parts).upper()))}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
