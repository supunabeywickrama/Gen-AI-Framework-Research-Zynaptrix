"""
detect_anomaly.py — Inference script for real-time and batch anomaly detection.

Loads:
  - Trained autoencoder model  (data/processed/autoencoder.keras)
  - Scaler                     (data/processed/scaler.pkl)
  - Thresholds                 (data/processed/thresholds.json)

Usage (batch):
    python models/detect_anomaly.py --input data/mock_data/generated_sensor_data.csv

Usage (live — single reading dict):
    from models.detect_anomaly import AnomalyDetector
    detector = AnomalyDetector()
    result = detector.detect({"temperature": 180.2, "motor_current": 4.3, ...})
"""

import sys
import os
import json
import pickle
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SENSOR_COLUMNS

log = logging.getLogger(__name__)

MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
DENSE_MODEL    = os.path.join(MODEL_DIR, "autoencoder.keras")
LSTM_MODEL     = os.path.join(MODEL_DIR, "lstm_autoencoder.keras")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "thresholds.json")
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")


class AnomalyDetector:
    """
    Lightweight wrapper around the trained Dense Autoencoder for inference.

    Example:
        detector = AnomalyDetector()
        result = detector.detect({"temperature": 175, "motor_current": 7.2, ...})
        print(result["is_anomaly"], result["score"])
    """

    def __init__(self, model_path: str = DENSE_MODEL,
                 scaler_path: str = SCALER_PATH,
                 threshold_key: str = "dense"):
        self._model    = None
        self._scaler   = None
        self._threshold = None
        self._model_path   = model_path
        self._scaler_path  = scaler_path
        self._threshold_key = threshold_key

    def _load(self):
        if self._model is not None:
            return

        # Lazy import TF so the file is importable without TF installed
        import tensorflow as tf

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                "Run 'python models/train_model.py' first."
            )
        self._model = tf.keras.models.load_model(self._model_path)

        with open(self._scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        with open(THRESHOLD_FILE) as f:
            thresholds = json.load(f)
        self._threshold = thresholds[self._threshold_key]
        log.info(f"AnomalyDetector loaded | threshold={self._threshold:.6f}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, reading: dict) -> dict:
        """
        Detect anomaly in a single sensor reading.

        Args:
            reading: dict with keys matching SENSOR_COLUMNS

        Returns:
            {
              "is_anomaly": bool,
              "score":      float  (reconstruction MSE),
              "threshold":  float,
              "sensors":    dict of raw input values,
            }
        """
        self._load()
        x = np.array([[reading[c] for c in SENSOR_COLUMNS]], dtype=np.float32)
        x_scaled = self._scaler.transform(x)
        x_pred   = self._model.predict(x_scaled, verbose=0)
        score    = float(np.mean(np.square(x_scaled - x_pred)))

        return {
            "is_anomaly": score > self._threshold,
            "score":      round(score, 6),
            "threshold":  round(self._threshold, 6),
            "sensors":    reading,
        }

    def detect_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies on an entire DataFrame.

        Returns the input DataFrame with two extra columns:
            recon_error  — MSE score per row
            is_anomaly   — bool flag
        """
        self._load()
        X = df[SENSOR_COLUMNS].values.astype(np.float32)
        X_scaled = self._scaler.transform(X)
        X_pred   = self._model.predict(X_scaled, verbose=0)
        errors   = np.mean(np.square(X_scaled - X_pred), axis=1)

        result = df.copy()
        result["recon_error"] = errors
        result["is_anomaly"]  = errors > self._threshold
        return result

    @property
    def threshold(self) -> float:
        self._load()
        return self._threshold


# ── CLI batch runner ──────────────────────────────────────────────────────────

def run_batch(csv_path: str):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    detector = AnomalyDetector()
    df = pd.read_csv(csv_path)
    log.info(f"Running batch detection on {len(df):,} rows …")

    result_df = detector.detect_batch(df)

    # Summary per state (if available)
    if "state" in result_df.columns:
        print("\nAnomaly detection results by state:")
        print("─" * 55)
        for state, group in result_df.groupby("state"):
            flagged = group["is_anomaly"].sum()
            total   = len(group)
            pct     = flagged / total * 100
            print(f"  {state:<18s}  {flagged:>5,}/{total:<6,}  ({pct:.1f}% flagged)")
        print("─" * 55)

    total_flagged = result_df["is_anomaly"].sum()
    print(f"\n  Total anomalies detected: {total_flagged:,} / {len(result_df):,} rows")

    # Save results
    out_path = csv_path.replace(".csv", "_anomaly_results.csv")
    result_df.to_csv(out_path, index=False)
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch anomaly detection")
    parser.add_argument("--input", required=True, help="Path to sensor CSV file")
    args = parser.parse_args()
    run_batch(args.input)
