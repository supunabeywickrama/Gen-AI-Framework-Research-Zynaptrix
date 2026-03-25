"""
train_model.py — Training pipeline for the Dense and LSTM Autoencoders.

Training rules:
  - ONLY normal-state rows are used for training
  - Validation split is taken from normal rows
  - Anomaly threshold is computed as:  mean + 2*std  of training reconstruction errors
  - Trained model and threshold are saved to disk

Usage:
    python models/train_model.py --model dense   (default)
    python models/train_model.py --model lstm
"""

import sys
import os
import argparse
import json
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROCESSED_DATA_PATH, SENSOR_COLUMNS
from models.autoencoder_model import (
    build_autoencoder,
    get_callbacks as dense_callbacks,
    reconstruction_error as dense_error,
)
from models.lstm_autoencoder import (
    build_lstm_autoencoder,
    create_sequences,
    get_callbacks as lstm_callbacks,
    reconstruction_error as lstm_error,
)
from preprocessing.normalization import get_scaler_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
DENSE_MODEL    = os.path.join(MODEL_DIR, "autoencoder.keras")
LSTM_MODEL     = os.path.join(MODEL_DIR, "lstm_autoencoder.keras")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "thresholds.json")

TIMESTEPS      = 30   # window length for LSTM


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_model_paths(machine_id: str = "PUMP-001"):
    return {
        "dense": os.path.join(MODEL_DIR, f"autoencoder_{machine_id}.keras"),
        "lstm":  os.path.join(MODEL_DIR, f"lstm_autoencoder_{machine_id}.keras"),
        "data":  PROCESSED_DATA_PATH.replace(".csv", f"_{machine_id}.csv")
    }

def load_normal_data(machine_id: str = "PUMP-001") -> np.ndarray:
    """Load normalized dataset and return only normal-state rows."""
    paths = get_model_paths(machine_id)
    if not os.path.exists(paths["data"]):
        log.warning(f"No processed data for {machine_id}, falling back to default.")
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        df = pd.read_csv(paths["data"])
        
    normal_df = df[df["state"] == "normal"][SENSOR_COLUMNS]
    log.info(f"[{machine_id}] Normal rows for training: {len(normal_df):,}")
    return normal_df.values.astype(np.float32)


def compute_threshold(errors: np.ndarray, n_sigma: float = 2.0) -> float:
    """Threshold = mean + n_sigma * std of training reconstruction errors."""
    threshold = float(np.mean(errors) + n_sigma * np.std(errors))
    log.info(f"Computed threshold: {threshold:.6f}  (mean={np.mean(errors):.6f}, std={np.std(errors):.6f})")
    return threshold


def save_threshold(machine_id: str, key: str, value: float) -> None:
    thresholds = {}
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            thresholds = json.load(f)
    
    # Store with machine prefix
    thresholds[f"{machine_id}_{key}"] = value
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Threshold for {machine_id}_{key} saved → {THRESHOLD_FILE}")


def plot_loss(history, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history.history["loss"],     label="Train loss", linewidth=2, color="#3498db")
    ax.plot(history.history["val_loss"], label="Val loss",   linewidth=2, color="#e74c3c", linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Loss curve saved → {out_path}")


def plot_error_distribution(errors: np.ndarray, threshold: float,
                             title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=80, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.axvline(threshold, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Error distribution saved → {out_path}")


# ── Training routines ─────────────────────────────────────────────────────────

def train_dense(machine_id: str = "PUMP-001"):
    log.info("═" * 50)
    log.info(f"Training Dense Autoencoder for {machine_id}")
    log.info("═" * 50)

    X = load_normal_data(machine_id)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    log.info(f"Train: {len(X_train):,}   Val: {len(X_val):,}")

    model = build_autoencoder(n_features=len(SENSOR_COLUMNS))
    model.summary()

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=64,
        callbacks=dense_callbacks(patience=12),
        verbose=2,
    )

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = get_model_paths(machine_id)
    model.save(paths["dense"])
    log.info(f"Model saved → {paths['dense']}")

    # Compute and save threshold
    errors = dense_error(model, X_train)
    threshold = compute_threshold(errors)
    save_threshold(machine_id, "dense", threshold)

    # Charts
    plot_loss(history, "Dense Autoencoder — Training Loss",
              os.path.join(MODEL_DIR, "dense_loss_curve.png"))
    plot_error_distribution(errors, threshold,
                            "Dense Autoencoder — Training Error Distribution",
                            os.path.join(MODEL_DIR, "dense_error_dist.png"))

    log.info("Dense autoencoder training complete ✓")
    return model, threshold


def train_lstm(machine_id: str = "PUMP-001"):
    log.info("═" * 50)
    log.info(f"Training LSTM Autoencoder for {machine_id}")
    log.info("═" * 50)

    X = load_normal_data(machine_id)
    X_seq = create_sequences(X, timesteps=TIMESTEPS)
    log.info(f"Sequences shape: {X_seq.shape}")

    split = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split], X_seq[split:]

    model = build_lstm_autoencoder(timesteps=TIMESTEPS, n_features=len(SENSOR_COLUMNS))
    model.summary()

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=60,
        batch_size=64,
        callbacks=lstm_callbacks(patience=8),
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = get_model_paths(machine_id)
    model.save(paths["lstm"])
    log.info(f"Model saved → {paths['lstm']}")

    errors = lstm_error(model, X_train)
    threshold = compute_threshold(errors)
    save_threshold(machine_id, "lstm", threshold)

    plot_loss(history, "LSTM Autoencoder — Training Loss",
              os.path.join(MODEL_DIR, "lstm_loss_curve.png"))
    plot_error_distribution(errors, threshold,
                            "LSTM Autoencoder — Training Error Distribution",
                            os.path.join(MODEL_DIR, "lstm_error_dist.png"))

    log.info("LSTM autoencoder training complete ✓")
    return model, threshold


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-machine Model Training")
    parser.add_argument("--model", choices=["dense", "lstm"], default="dense")
    parser.add_argument("--machine_id", default="PUMP-001", help="Machine to train for")
    args = parser.parse_args()

    if args.model == "dense":
        train_dense(args.machine_id)
    else:
        train_lstm(args.machine_id)
