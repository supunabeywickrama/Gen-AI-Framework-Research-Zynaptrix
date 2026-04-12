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


log = logging.getLogger(__name__)

MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
DENSE_MODEL    = os.path.join(MODEL_DIR, "autoencoder.keras")
LSTM_MODEL     = os.path.join(MODEL_DIR, "lstm_autoencoder.keras")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "thresholds.json")


class AnomalyDetector:
    """
    Inference wrapper that caches models/scalers per machine_id.
    """

    def __init__(self, threshold_type: str = "dense"):
        self._registry = {} # {machine_id: {"model": ..., "scaler": ..., "threshold": ...}}
        self._threshold_type = threshold_type

    def _get_assets(self, machine_id: str):
        if machine_id in self._registry:
            return self._registry[machine_id]

        import tensorflow as tf
        import requests
        import tempfile
        from unified_rag.db.database import SessionLocal
        from unified_rag.db.models import AnomalyThreshold, MachineAsset

        db = SessionLocal()
        try:
            # 1. Fetch Threshold from DB
            t_key = f"{machine_id}_{self._threshold_type}"
            t_record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == machine_id, 
                                                        AnomalyThreshold.threshold_type == self._threshold_type).first()
            if not t_record:
                log.warning(f"Threshold not found in DB for {t_key}, falling back to PUMP-001")
                t_record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == "PUMP-001", 
                                                            AnomalyThreshold.threshold_type == self._threshold_type).first()
            
            # 2. Fetch Assets (Model/Scaler) from DB
            asset_types = [f"model_{self._threshold_type}", "scaler"]
            asset_records = db.query(MachineAsset).filter(MachineAsset.machine_id == machine_id, 
                                                        MachineAsset.asset_type.in_(asset_types)).all()
            
            # Map assets
            assets_meta = {a.asset_type: a.url for a in asset_records}
            
            # Fallback to PUMP-001 if machine assets missing
            if len(assets_meta) < 2:
                log.warning(f"Assets not found in DB for {machine_id}, falling back to PUMP-001")
                asset_records = db.query(MachineAsset).filter(MachineAsset.machine_id == "PUMP-001", 
                                                            MachineAsset.asset_type.in_(asset_types)).all()
                assets_meta = {a.asset_type: a.url for a in asset_records}

            # Helper to load from URL or path
            def load_asset(url_or_path, is_model=True):
                if url_or_path.startswith("http"):
                    # Dynamic Cloudinary download
                    with tempfile.NamedTemporaryFile(suffix=".keras" if is_model else ".pkl", delete=False) as tmp:
                        r = requests.get(url_or_path)
                        tmp.write(r.content)
                        tmp_path = tmp.name
                    
                    if is_model:
                        res = tf.keras.models.load_model(tmp_path)
                    else:
                        with open(tmp_path, "rb") as f:
                            res = pickle.load(f)
                    
                    try: os.remove(tmp_path)
                    except: pass
                    return res
                else:
                    # Legacy local path
                    if is_model:
                        return tf.keras.models.load_model(url_or_path)
                    else:
                        with open(url_or_path, "rb") as f:
                            return pickle.load(f)

            # 3. Load actual objects
            model_url = assets_meta.get(f"model_{self._threshold_type}")
            scaler_url = assets_meta.get("scaler")
            
            if not model_url or not scaler_url:
                raise Exception(f"Failed to find assets (model/scaler) for {machine_id} in cloud registry")

            log.info(f"Loading cloud assets for {machine_id} ...")
            model = load_asset(model_url, is_model=True)
            scaler = load_asset(scaler_url, is_model=False)
            threshold = t_record.value if t_record else 0.5 

            assets = {"model": model, "scaler": scaler, "threshold": threshold}
            self._registry[machine_id] = assets
            return assets
        finally:
            db.close()

    def _calculate_health(self, score: float, threshold: float) -> int:
        """
        Calculates a 0-100% health score.
        100% = Perfect alignment with normal patterns.
        50%  = Threshold crossed (Anomaly starts).
        <50% = Significant degradation.
        """
        if score <= 0: return 100
        
        if score < threshold:
            # Linear decay from 100% to 50%
            health = 100 - (score / threshold) * 50
        else:
            # Exponential decay from 50% down to 0%
            # Half-life of decay is the threshold value again
            import math
            health = 50 * math.exp(-(score - threshold) / threshold)
            
        return int(max(0, min(100, health)))

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, reading: dict) -> dict:
        """
        Detect anomaly in a single sensor reading.
        Expects 'machine_id' in the reading dict.
        """
        machine_id = reading.get("machine_id", "PUMP-001")
        assets = self._get_assets(machine_id)
        
        from simulator.anomaly_injector import get_machine_config
        cfg = get_machine_config(machine_id)
        sensor_cols = list(cfg.keys())
        
        # Extract only sensor columns for model
        x = np.array([[float(reading.get(c, 0.0)) for c in sensor_cols]], dtype=np.float32)
        x_scaled = assets["scaler"].transform(x)
        x_pred   = assets["model"].predict(x_scaled, verbose=0)
        score    = float(np.mean(np.square(x_scaled - x_pred)))
        
        health_score = self._calculate_health(score, assets["threshold"])

        return {
            "is_anomaly": score > assets["threshold"],
            "score":      round(score, 6),
            "threshold":  round(assets["threshold"], 6),
            "health_score": health_score,
            "sensors":    reading,
            "machine_id": machine_id
        }

    def detect_batch(self, df: pd.DataFrame, machine_id: str = "PUMP-001") -> pd.DataFrame:
        """Detect anomalies on an entire DataFrame for one machine."""
        assets = self._get_assets(machine_id)
        from simulator.anomaly_injector import get_machine_config
        cfg = get_machine_config(machine_id)
        sensor_cols = list(cfg.keys())
        
        X = df[sensor_cols].values.astype(np.float32)
        X_scaled = assets["scaler"].transform(X)
        X_pred   = assets["model"].predict(X_scaled, verbose=0)
        errors   = np.mean(np.square(X_scaled - X_pred), axis=1)

        result = df.copy()
        result["recon_error"] = errors
        result["is_anomaly"]  = errors > assets["threshold"]
        return result


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
