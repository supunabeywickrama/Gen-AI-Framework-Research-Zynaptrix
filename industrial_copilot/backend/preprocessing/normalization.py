"""
normalization.py — Sensor data normalization utilities.

Uses StandardScaler fitted only on NORMAL operating state data
so anomalous readings remain detectable (not normalized away).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SENSOR_COLUMNS, PROCESSED_DATA_PATH, MOCK_DATA_PATH


def get_scaler_path(machine_id: str = "PUMP-001") -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", f"scaler_{machine_id}.pkl"
    )

def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on normal-state data only.
    """
    if "state" in df.columns:
        fit_df = df[df["state"] == "normal"][SENSOR_COLUMNS]
    else:
        fit_df = df[SENSOR_COLUMNS]

    scaler = StandardScaler()
    scaler.fit(fit_df)
    return scaler

def save_scaler(scaler: StandardScaler, machine_id: str = "PUMP-001") -> None:
    path = get_scaler_path(machine_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved → {path}")

def load_scaler(machine_id: str = "PUMP-001") -> StandardScaler:
    path = get_scaler_path(machine_id)
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of df with sensor columns standardized."""
    df = df.copy()
    df[SENSOR_COLUMNS] = scaler.transform(df[SENSOR_COLUMNS])
    return df

def fit_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Convenience: fit scaler then normalize."""
    scaler = fit_scaler(df)
    return normalize(df, scaler), scaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize sensor data")
    parser.add_argument("--machine_id", default="PUMP-001", help="Machine ID")
    parser.add_argument("--input", help="Custom input CSV")
    parser.add_argument("--output", help="Custom output CSV")
    args = parser.parse_args()

    input_path = args.input or MOCK_DATA_PATH.replace(".csv", f"_{args.machine_id}.csv")
    if not os.path.exists(input_path):
        input_path = MOCK_DATA_PATH # Fallback

    df = pd.read_csv(input_path)
    normalized_df, scaler = fit_and_normalize(df)

    output_path = args.output or PROCESSED_DATA_PATH.replace(".csv", f"_{args.machine_id}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalized_df.to_csv(output_path, index=False)
    save_scaler(scaler, args.machine_id)

    print(f"✓ Normalized data saved → {output_path}")
