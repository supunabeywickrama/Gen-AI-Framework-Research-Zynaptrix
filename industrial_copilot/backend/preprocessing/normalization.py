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

from config.settings import SENSOR_COLUMNS, PROCESSED_DATA_PATH


SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "scaler.pkl"
)


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on normal-state data only.
    If a 'state' column exists, uses only 'normal' rows for fitting.
    """
    if "state" in df.columns:
        fit_df = df[df["state"] == "normal"][SENSOR_COLUMNS]
    else:
        fit_df = df[SENSOR_COLUMNS]

    scaler = StandardScaler()
    scaler.fit(fit_df)
    return scaler


def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved → {path}")


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of df with sensor columns standardized."""
    df = df.copy()
    df[SENSOR_COLUMNS] = scaler.transform(df[SENSOR_COLUMNS])
    return df


def fit_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Convenience: fit scaler then normalize. Returns (normalized_df, scaler)."""
    scaler = fit_scaler(df)
    return normalize(df, scaler), scaler


if __name__ == "__main__":
    from config.settings import MOCK_DATA_PATH

    df = pd.read_csv(MOCK_DATA_PATH)
    normalized_df, scaler = fit_and_normalize(df)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    normalized_df.to_csv(PROCESSED_DATA_PATH, index=False)
    save_scaler(scaler)

    print(f"✓ Normalized data saved → {PROCESSED_DATA_PATH}")
