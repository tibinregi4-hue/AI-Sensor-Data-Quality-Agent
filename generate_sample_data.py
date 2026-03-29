"""
Script to generate sample_sensor_data.csv with realistic problems built in.
Run once to regenerate the sample data.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 5000

timestamps = pd.date_range("2024-01-01", periods=N, freq="1min")
sensor_ids = np.random.choice(["SENS_001", "SENS_002", "SENS_003", "SENS_004"], N)
status = np.random.choice(["OK", "WARN", "OK", "OK", "OK"], N)

# Auto-correlated temperature: slow sinusoidal trend + AR(1) noise → model can predict it
t_idx = np.arange(N)
temperature = (
    20                                              # base
    + 8 * np.sin(2 * np.pi * t_idx / (60 * 24))   # daily cycle (period = 1440 min)
    + 3 * np.sin(2 * np.pi * t_idx / (60 * 24 * 7))  # weekly cycle
)
# add small AR(1) noise
ar_noise = np.zeros(N)
for i in range(1, N):
    ar_noise[i] = 0.9 * ar_noise[i-1] + np.random.normal(0, 0.5)
temperature = temperature + ar_noise

pressure    = np.random.normal(1013, 10, N)       # Normal: ~1013 hPa ±10
humidity    = np.random.uniform(30, 70, N)        # 30–70%
voltage     = np.random.normal(3.3, 0.1, N)      # 3.3V ±0.1

# ---- inject realistic problems ----

# 1) Missing values in temperature (~6.5%) and pressure (~3.2%) — enough to trigger MAX_NULL_PERCENT=5%
temp_null_idx = np.random.choice(N, 325, replace=False)
pressure_null_idx = np.random.choice(N, 160, replace=False)
temperature[temp_null_idx] = np.nan
pressure[pressure_null_idx] = np.nan

# 2) Impossible pressure values (6 values outside 900–1100)
bad_pressure_idx = np.random.choice(np.where(~np.isnan(pressure))[0], 6, replace=False)
pressure[bad_pressure_idx] = np.random.choice([200, 1500, -50, 2000, 100, 1800], 6)

# 3) Voltage outliers (8 values beyond 3σ)
outlier_idx = np.random.choice(N, 8, replace=False)
voltage[outlier_idx] = np.random.choice([0.1, 6.5, 7.2, 0.05, 5.9, 6.8, 0.08, 7.5], 8)

df = pd.DataFrame({
    "timestamp":   timestamps,
    "temperature": np.round(temperature, 2),
    "pressure":    np.round(pressure, 2),
    "humidity":    np.round(humidity, 2),
    "voltage":     np.round(voltage, 4),
    "sensor_id":   sensor_ids,
    "status":      status,
})

# 4) Add 15 exact duplicate rows (appended at end — do NOT shuffle so time-series model works)
dup_rows = df.iloc[100:115].copy()
df = pd.concat([df, dup_rows], ignore_index=True)

out = Path("data/sample_sensor_data.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Generated {len(df)} rows → {out}")
