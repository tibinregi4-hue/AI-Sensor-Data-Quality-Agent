# Infineon AI Data Quality Agent

> **Built by Tibin Regi** — BSc Robotics & AI, University of Klagenfurt  
> Demonstration project for the **Infineon Technologies AI & Data Engineering Internship**

An end-to-end automated pipeline that validates any CSV sensor dataset, uses an AI agent to diagnose problems, applies targeted fixes, evaluates a predictive model, and generates full audit reports — all in a single command.

---

## What This Project Does

Drop any CSV file into the `data/` folder and run one command. The system automatically detects data quality issues (missing values, impossible sensor readings, duplicates, outliers, distribution drift), has an AI agent build a structured fix plan, applies the fixes, revalidates the cleaned data, trains a GradientBoosting model, evaluates it against quality gates, and writes a JSON + Markdown report. A GitHub Actions CI/CD pipeline runs everything automatically on every push.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          run_agent.py  (orchestrator)        │
                    └────────────────┬────────────────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────────┐
          │                                                          │
    ┌─────▼──────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐│
    │ validator  │───▶│   agent    │───▶│  fixer   │───▶│validator ││
    │            │    │            │    │          │    │ (pass 2) ││
    │ • schema   │    │ LLM mode:  │    │ • nulls  │    │          ││
    │ • nulls    │    │  Claude    │    │ • ranges │    │ all PASS ││
    │ • ranges   │    │  API       │    │ • dupes  │    └─────┬────┘│
    │ • dupes    │    │            │    │ • clips  │          │     │
    │ • outliers │    │ Offline:   │    └──────────┘          │     │
    │ • drift    │    │  rule-based│                          │     │
    └────────────┘    └────────────┘                          │     │
                                                              ▼     │
                                                    ┌─────────────┐ │
                                                    │model_       │ │
                                                    │evaluator    │ │
                                                    │             │ │
                                                    │ Gradient    │ │
                                                    │ Boosting    │ │
                                                    │ RMSE / R²   │ │
                                                    └──────┬──────┘ │
                                                           │        │
                                                    ┌──────▼──────┐ │
                                                    │report_      │ │
                                                    │generator    │ │
                                                    │             │ │
                                                    │ JSON report │ │
                                                    │ MD report   │ │
                                                    │ Terminal    │ │
                                                    └─────────────┘ │
                                                                     │
    ┌────────────────────────────────────────────────────────────────┘
    │
    ▼  GitHub Actions CI/CD (.github/workflows/validate.yml)
    ┌─────────────────────────────────────────────────────────┐
    │  on: push → install deps → pytest → run pipeline        │
    │              → upload reports as artifacts              │
    └─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/infineon-data-agent.git
cd infineon-data-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (uses built-in sample data)
python run_agent.py

# 4. Run with your own CSV file
python run_agent.py data/your_file.csv

# 5. Run tests only
pytest tests/ -v
```

**Windows (double-click):** `run.bat`  
**PowerShell:** `.\run.ps1`

---

## Using With Your Own Data

The pipeline works with **any CSV file** — no configuration needed:

```bash
python run_agent.py data/my_sensors.csv
```

The system auto-detects:
- Which columns are numeric (for range/outlier/drift checks)
- Whether `temperature` and `pressure` columns exist (for physical-range rules)
- The best target column for model training

To add custom range rules, edit `config.py`:

```python
TEMPERATURE_MIN = -50   # °C
TEMPERATURE_MAX = 60    # °C
PRESSURE_MIN    = 900   # hPa
PRESSURE_MAX    = 1100  # hPa
MAX_NULL_PERCENT = 5.0  # %
```

---

## Enabling AI Mode (Claude API)

By default the agent runs in offline rule-based mode — no internet needed.  
To enable LLM-powered diagnosis:

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...   # Linux / macOS
set ANTHROPIC_API_KEY=sk-ant-...      # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."   # PowerShell
```

The agent automatically detects the key and switches to LLM mode.

---

## How CI/CD Works

Every `git push` to `main` triggers `.github/workflows/validate.yml`:

1. **Checkout** — fresh clone on Ubuntu runner
2. **Install deps** — `pip install -r requirements.txt`
3. **Pytest** — runs all 4 test modules; fails the build if quality gates are not met
4. **Full pipeline** — `python run_agent.py` end-to-end
5. **Upload reports** — JSON + Markdown saved as GitHub Actions artifacts

The `ANTHROPIC_API_KEY` secret can be added in *Settings → Secrets → Actions* to enable LLM mode in CI.

---

## Sample Terminal Output

```
==============================================================
  Infineon AI Data Quality Agent
  Built by Tibin Regi
  ──────────────────────────────────────────────────────────
  LLM mode: OFFLINE (rule-based fallback)
==============================================================

  Loaded: data/sample_sensor_data.csv  (5,015 rows, 7 columns)
  Columns: ['timestamp', 'temperature', 'pressure', 'humidity', 'voltage', 'sensor_id', 'status']

  [1/5] Validating data quality...
        Schema check     : PASS   7 columns, all valid
        Null check       : FAIL   temperature: 3.2%, pressure: 1.8%
        Range check      : FAIL   pressure: 6 bad values
        Duplicate check  : FAIL   15 duplicate rows
        Outlier check    : WARNING  voltage: 8
        Drift check      : PASS   distributions stable

        Issues found: 3 critical, 1 warning

  [2/5] AI Agent analyzing issues...
        Diagnosis: medium severity — identified issues: ...
        Mode: rule_based
        Fix plan:
          1. Remove 15 exact duplicate rows
          2. Fix 6 pressure values outside physical range
          3. Median-impute missing values (temperature: 3.2%, pressure: 1.8%)
          4. Clip 8 outlier values beyond 3.0σ

  [3/5] Applying fixes...
        Removed 15 duplicate rows
        Fixed 6 range violations in 'pressure'
        Filled 160 missing values in 'temperature' (median: 20.03)
        Filled 90 missing values in 'pressure' (median: 1013.06)
        Clipped 8 outliers in 'voltage'
        Rows: 5015 → 5000  (15 removed)

  [4/5] Re-validating cleaned data...
        Schema check     : PASS   7 columns, all valid
        Null check       : PASS   0 missing
        Range check      : PASS   all values in range
        Duplicate check  : PASS   0 duplicates
        Outlier check    : PASS   0 outliers
        Drift check      : PASS   distributions stable

        ALL CHECKS PASSED

  [5/5] Training AI model...
        Training samples  : 3992
        Testing samples   : 998
        Target column     : temperature
        RMSE              : 0.0012  (threshold: 5.0)  PASS
        R²                : 1.0000  (threshold: 0.5)   PASS
        MAE               : 0.0008

  Reports saved:
    reports/quality_report.json
    reports/quality_report.md

==============================================================
  PIPELINE COMPLETE — ALL QUALITY GATES PASSED ✓
==============================================================
```

---

## Project Structure

```
infineon_data_agent/
├── data/
│   └── sample_sensor_data.csv    # 5,000 rows with realistic quality issues
├── tests/
│   ├── test_schema.py            # Column existence, dtypes, non-empty
│   ├── test_quality.py           # Nulls, ranges, duplicates (FAIL on raw data)
│   ├── test_drift.py             # Distribution stability, sudden jumps
│   └── test_model.py             # RMSE / R² quality gates
├── reports/                      # Auto-generated JSON + Markdown reports
├── .github/workflows/
│   └── validate.yml              # GitHub Actions CI/CD pipeline
├── validator.py                  # Core validation engine
├── agent.py                      # AI agent (LLM + offline fallback)
├── fixer.py                      # Data-cleaning tools
├── model_evaluator.py            # GradientBoosting training & evaluation
├── report_generator.py           # JSON, Markdown, terminal output
├── run_agent.py                  # Main entry point
├── config.py                     # All thresholds and settings
├── requirements.txt
├── run.bat                       # Windows double-click launcher
└── run.ps1                       # PowerShell launcher
```

---

## Skills Demonstrated

| Internship Requirement | Implementation |
|---|---|
| Data Engineering | `validator.py` — auto-detects schema, nulls, ranges, drift |
| AI Agents | `agent.py` — LLM diagnosis with tool-use pattern; offline fallback |
| Data Management | `fixer.py` — median imputation, outlier clipping, deduplication |
| Software Testing | `tests/` — 4 pytest modules, intentionally fail on raw data |
| CI/CD | `.github/workflows/validate.yml` — GitHub Actions full pipeline |
| ML Workflows | `model_evaluator.py` — sliding-window features, quality gates |
| Documentation | `report_generator.py` — JSON + Markdown audit reports |

---

*Built by **Tibin Regi** | BSc Robotics & AI, University of Klagenfurt*  
*Demonstration project for semi-conductor industries- AI & Data Engineering Internship*
