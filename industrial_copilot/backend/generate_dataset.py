"""
generate_dataset.py — Batch mock dataset generator for Phase 1.

Generates a realistic 20,000-row sensor dataset with controlled
machine states (normal, machine_fault, sensor_freeze, sensor_drift, idle).

Output: data/mock_data/generated_sensor_data.csv

Usage:
    python generate_dataset.py
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    STATE_ROW_DISTRIBUTION,
    SENSOR_COLUMNS,
    MOCK_DATA_PATH,
)
from simulator.anomaly_injector import (
    normal_reading,
    machine_fault_reading,
    sensor_freeze_reading,
    sensor_drift_reading,
    idle_reading,
)


def generate_dataset(total_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic sensor dataset using block-based state injection.

    Instead of row-by-row random state (which looks unrealistic), states are
    injected in contiguous blocks — mimicking how real faults occur as events.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Build state sequence as contiguous blocks ──────────────────────────
    state_blocks = []
    for state, count in STATE_ROW_DISTRIBUTION.items():
        remaining = count
        while remaining > 0:
            block_size = min(random.randint(10, 120), remaining)
            state_blocks.append((state, block_size))
            remaining -= block_size

    random.shuffle(state_blocks)

    # ── Generate readings block by block ───────────────────────────────────
    rows = []
    drift_step = 0.0
    frozen_snapshot = None
    start_time = datetime(2026, 1, 1, 6, 0, 0)   # factory shift start

    tick = 0
    drift_step: float = 0.0
    for state, block_size in state_blocks:
        drift_step = 0.0          # reset drift each new drift block
        frozen_snapshot = None    # reset freeze snapshot

        for j in range(block_size):
            timestamp = start_time + timedelta(seconds=tick)

            if state == "normal":
                reading = normal_reading()

            elif state == "machine_fault":
                reading = machine_fault_reading()

            elif state == "sensor_freeze":
                if frozen_snapshot is None:
                    frozen_snapshot = normal_reading()
                reading = sensor_freeze_reading(frozen_snapshot)

            elif state == "sensor_drift":
                drift_step = drift_step + float(random.uniform(0.05, 0.3))
                reading = sensor_drift_reading(drift_step)

            else:  # idle
                reading = idle_reading()

            row = {
                "timestamp":     timestamp.isoformat(),
                "temperature":   round(reading["temperature"], 3),
                "motor_current": round(reading["motor_current"], 3),
                "vibration":     round(reading["vibration"], 4),
                "speed":         round(reading["speed"], 2),
                "pressure":      round(reading["pressure"], 3),
                "state":         state,
            }
            rows.append(row)
            tick += 1

    df = pd.DataFrame(rows)

    # Trim or pad to exact total_rows
    if len(df) > total_rows:
        df = df.iloc[:total_rows].copy()

    df = df.reset_index(drop=True)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Print a quick statistical summary of the generated dataset."""
    print("\n" + "═" * 55)
    print("  Dataset Summary")
    print("═" * 55)
    print(f"  Total rows   : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print()

    print("  State distribution:")
    counts = df["state"].value_counts()
    for state, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {state:<18s} {count:>6,}  ({pct:.1f}%)")

    print()
    print("  Sensor statistics (mean ± std):")
    for col in SENSOR_COLUMNS:
        m = df[col].mean()
        s = df[col].std()
        print(f"    {col:<18s}  {m:8.3f} ± {s:.3f}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    print("Generating mock sensor dataset …")
    df = generate_dataset(total_rows=20_000)
    validate_dataset(df)

    os.makedirs(os.path.dirname(MOCK_DATA_PATH), exist_ok=True)
    df.to_csv(MOCK_DATA_PATH, index=False)
    print(f"✓ Dataset saved → {MOCK_DATA_PATH}")
