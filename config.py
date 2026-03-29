"""
config.py — Central configuration for AI Data Quality Agent
Author: Tibin Regi | Infineon AI & Data Engineering Internship Project
"""
import os

# ── Data quality thresholds ──────────────────────────────────────────────────
MAX_NULL_PERCENT   = 5.0    # Maximum allowed % of null values per column
PRESSURE_MIN       = 900    # Minimum physically valid pressure (hPa)
PRESSURE_MAX       = 1100   # Maximum physically valid pressure (hPa)
TEMPERATURE_MIN    = -50    # Minimum physically valid temperature (°C)
TEMPERATURE_MAX    = 60     # Maximum physically valid temperature (°C)
DRIFT_THRESHOLD    = 2.0    # Standard deviations before flagging distribution drift
OUTLIER_SIGMA      = 3.0    # σ boundary for outlier detection

# ── Model quality gates ───────────────────────────────────────────────────────
RMSE_THRESHOLD     = 5.0    # Maximum acceptable RMSE
R2_THRESHOLD       = 0.5    # Minimum acceptable R² (coefficient of determination)
WINDOW_SIZE        = 10     # Sliding window size for time-series feature engineering

# ── AI Agent settings ─────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
USE_LLM        = CLAUDE_API_KEY is not None
MODEL_NAME     = "claude-sonnet-4-20250514"

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = "data/sample_sensor_data.csv"
REPORTS_DIR       = "reports"
