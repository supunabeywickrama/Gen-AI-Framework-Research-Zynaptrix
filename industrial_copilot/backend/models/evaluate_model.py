"""
evaluate_model.py — Comprehensive evaluation pipeline for Autoencoder anomaly detection.

Computes 8 key metrics using the labeled dataset (state column as ground truth):
    1. Precision          — Of flagged anomalies, how many are real
    2. Recall/Sensitivity — Of real faults, how many were caught
    3. F1 Score           — Harmonic mean of precision & recall
    4. AUC-ROC            — Discrimination ability across thresholds
    5. False Positive Rate— Sensor glitches wrongly flagged
    6. False Negative Rate— Missed real faults (safety critical)
    7. Threshold Selection— Mean + 2σ justification via sweep
    8. MSE Distribution   — Normal vs fault separation histogram

Ground truth mapping:
    normal         → 0  (not anomaly)
    machine_fault  → 1  (anomaly)
    sensor_freeze  → 1  (anomaly)
    sensor_drift   → 1  (anomaly)
    idle           → 1  (anomaly)

Usage:
    python models/evaluate_model.py --machine_id PUMP-001 --model dense
    python models/evaluate_model.py --machine_id PUMP-001 --model lstm
    python models/evaluate_model.py --machine_id PUMP-001 --model both
    python models/evaluate_model.py --all                                # all machines, both models
"""

import sys
import os
import json
import pickle
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROCESSED_DATA_PATH, MOCK_DATA_PATH
from preprocessing.normalization import get_scaler_path
from models.autoencoder_model import reconstruction_error as dense_error
from models.lstm_autoencoder import (
    create_sequences,
    reconstruction_error as lstm_error,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
EVAL_DIR  = os.path.join(MODEL_DIR, "evaluation")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "thresholds.json")
TIMESTEPS = 30  # LSTM window size

# ── Ground truth: state → binary label ────────────────────────────────────────
STATE_TO_LABEL = {
    "normal":        0,
    "machine_fault": 1,
    "sensor_freeze": 1,
    "sensor_drift":  1,
    "idle":          1,
}

# ── Plot styling ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#3498db",
    "danger":     "#e74c3c",
    "success":    "#2ecc71",
    "warning":    "#f39c12",
    "purple":     "#9b59b6",
    "dark":       "#2c3e50",
    "light_gray": "#ecf0f1",
    "orange":     "#e67e22",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_full_dataset(machine_id: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the FULL normalized dataset (all states) for evaluation.
    Returns (DataFrame, sensor_columns).
    """
    processed_path = PROCESSED_DATA_PATH.replace(".csv", f"_{machine_id}.csv")
    if not os.path.exists(processed_path):
        log.warning(f"No processed data for {machine_id}, falling back to default.")
        processed_path = PROCESSED_DATA_PATH

    df = pd.read_csv(processed_path)
    sensor_cols = [c for c in df.columns if c not in ["timestamp", "machine_id", "state"]]
    log.info(f"[{machine_id}] Loaded {len(df):,} rows, {len(sensor_cols)} sensors, "
             f"states: {df['state'].value_counts().to_dict()}")
    return df, sensor_cols


def load_model_and_assets(machine_id: str, model_type: str = "dense"):
    """Load trained model, scaler, and threshold."""
    import tensorflow as tf

    if model_type == "dense":
        model_path = os.path.join(MODEL_DIR, f"autoencoder_{machine_id}.keras")
    else:
        model_path = os.path.join(MODEL_DIR, f"lstm_autoencoder_{machine_id}.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    scaler_path = get_scaler_path(machine_id)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(THRESHOLD_FILE) as f:
        thresholds = json.load(f)

    threshold_key = f"{machine_id}_{model_type}"
    if threshold_key not in thresholds:
        raise KeyError(f"Threshold not found for key: {threshold_key}")

    threshold = thresholds[threshold_key]

    log.info(f"[{machine_id}] Model: {model_type}, Threshold: {threshold:.6f}")
    return model, scaler, threshold


# ══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ground_truth(states: pd.Series) -> np.ndarray:
    """Map state strings to binary labels: 0=normal, 1=anomaly."""
    return states.map(STATE_TO_LABEL).values.astype(int)


def compute_mse_scores(model, X_scaled: np.ndarray, model_type: str = "dense") -> np.ndarray:
    """Compute reconstruction MSE for all samples."""
    if model_type == "dense":
        return dense_error(model, X_scaled)
    else:
        X_seq = create_sequences(X_scaled, timesteps=TIMESTEPS)
        return lstm_error(model, X_seq)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mse_scores: np.ndarray
) -> dict:
    """
    Compute all classification metrics from ground truth and predictions.

    Returns dict with precision, recall, f1, auc_roc, fpr, fnr,
    confusion matrix breakdown, and per-class report.
    """
    # Core metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC (uses continuous scores, not binary predictions)
    try:
        auc_roc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc_roc = 0.0
        log.warning("AUC-ROC could not be computed (single class in y_true)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Per-class report
    report = classification_report(y_true, y_pred,
                                   target_names=["Normal", "Anomaly"],
                                   output_dict=True, zero_division=0)

    return {
        "precision":           round(precision, 6),
        "recall":              round(recall, 6),
        "f1_score":            round(f1, 6),
        "auc_roc":             round(auc_roc, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "accuracy":            round((tp + tn) / (tp + tn + fp + fn), 6),
        "confusion_matrix": {
            "true_positive":  int(tp),
            "true_negative":  int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        },
        "per_class_report": report,
        "total_samples":     int(len(y_true)),
        "total_anomalies":   int(y_true.sum()),
        "total_normal":      int((y_true == 0).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve(y_true: np.ndarray, mse_scores: np.ndarray,
                   machine_id: str, model_type: str, out_dir: str) -> str:
    """Plot ROC curve with AUC annotation."""
    fpr_vals, tpr_vals, _ = roc_curve(y_true, mse_scores)
    auc_val = roc_auc_score(y_true, mse_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_vals, tpr_vals, color=COLORS["primary"], linewidth=2.5,
            label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["danger"], linewidth=1.5,
            linestyle="--", alpha=0.7, label="Random Classifier")

    # Fill area under curve
    ax.fill_between(fpr_vals, tpr_vals, alpha=0.15, color=COLORS["primary"])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — {machine_id} ({model_type.upper()} Autoencoder)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(out_dir, f"roc_curve_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ROC curve saved → {path}")
    return path


def plot_mse_distribution(normal_errors: np.ndarray, fault_errors: np.ndarray,
                          threshold: float, machine_id: str, model_type: str,
                          out_dir: str) -> str:
    """Overlapping histograms: normal (green) vs fault (red) with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute reasonable bins
    all_errors = np.concatenate([normal_errors, fault_errors])
    max_plot = min(np.percentile(all_errors, 99.5), threshold * 5)

    bins = np.linspace(0, max_plot, 100)

    ax.hist(normal_errors[normal_errors <= max_plot], bins=bins,
            color=COLORS["success"], alpha=0.65, edgecolor="white",
            label=f"Normal (n={len(normal_errors):,})", density=True)
    ax.hist(fault_errors[fault_errors <= max_plot], bins=bins,
            color=COLORS["danger"], alpha=0.55, edgecolor="white",
            label=f"Fault/Anomaly (n={len(fault_errors):,})", density=True)

    # Threshold line
    ax.axvline(threshold, color=COLORS["dark"], linewidth=2.5, linestyle="--",
               label=f"Threshold (mean+2σ) = {threshold:.4f}")

    # Mean annotations
    ax.axvline(np.mean(normal_errors), color=COLORS["success"], linewidth=1.5,
               linestyle=":", alpha=0.8)
    ax.axvline(np.mean(fault_errors), color=COLORS["danger"], linewidth=1.5,
               linestyle=":", alpha=0.8)

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"MSE Distribution — {machine_id} ({model_type.upper()} Autoencoder)\n"
                 f"Normal vs Fault Separation", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"mse_distribution_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  MSE distribution saved → {path}")
    return path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          machine_id: str, model_type: str, out_dir: str) -> str:
    """Heatmap confusion matrix with counts + percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)

    # Add text annotations
    labels = ["Normal", "Anomaly"]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 50 else COLORS["dark"]
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {machine_id} ({model_type.upper()})", fontsize=14)

    plt.colorbar(im, ax=ax, label="Percentage (%)", shrink=0.8)
    plt.tight_layout()

    path = os.path.join(out_dir, f"confusion_matrix_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix saved → {path}")
    return path


def plot_threshold_sweep(y_true: np.ndarray, mse_scores: np.ndarray,
                         current_threshold: float, machine_id: str,
                         model_type: str, out_dir: str) -> str:
    """
    F1, Precision, Recall vs threshold curve.
    Shows why mean+2σ is (or isn't) optimal.
    """
    mean_err = np.mean(mse_scores[y_true == 0])  # mean of normal errors
    std_err  = np.std(mse_scores[y_true == 0])   # std of normal errors

    # Sweep from mean to mean + 5σ
    thresholds = np.linspace(max(mean_err, 0.001), mean_err + 5 * std_err, 200)

    precisions = []
    recalls    = []
    f1s        = []
    fprs       = []

    for t in thresholds:
        preds = (mse_scores > t).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))
        cm = confusion_matrix(y_true, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        else:
            fprs.append(0)

    # Find optimal threshold (max F1)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Precision, Recall, F1 vs Threshold
    ax1.plot(thresholds, precisions, color=COLORS["primary"], linewidth=2,
             label="Precision", linestyle="-")
    ax1.plot(thresholds, recalls, color=COLORS["success"], linewidth=2,
             label="Recall", linestyle="-")
    ax1.plot(thresholds, f1s, color=COLORS["purple"], linewidth=2.5,
             label="F1 Score", linestyle="-")

    # Current threshold (mean+2σ)
    ax1.axvline(current_threshold, color=COLORS["danger"], linewidth=2.5,
                linestyle="--", alpha=0.9, label=f"Current (mean+2σ) = {current_threshold:.4f}")

    # Optimal threshold
    ax1.axvline(best_threshold, color=COLORS["warning"], linewidth=2,
                linestyle=":", alpha=0.9, label=f"Optimal F1 = {best_threshold:.4f}")

    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title(f"Threshold Sweep — {machine_id} ({model_type.upper()})", fontsize=14)
    ax1.legend(fontsize=9, loc="center right", framealpha=0.9)
    ax1.set_ylim([-0.05, 1.05])

    # Right plot: FPR vs Threshold
    ax2.plot(thresholds, fprs, color=COLORS["danger"], linewidth=2.5, label="FPR")
    ax2.axvline(current_threshold, color=COLORS["danger"], linewidth=2.5,
                linestyle="--", alpha=0.9, label=f"Current threshold")
    ax2.axhline(0.01, color=COLORS["warning"], linewidth=1.5, linestyle=":",
                alpha=0.7, label="FPR = 1% target")
    ax2.axhline(0.15, color=COLORS["light_gray"], linewidth=1.5, linestyle=":",
                alpha=0.7, label="Baseline FPR = 15%")

    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("False Positive Rate", fontsize=12)
    ax2.set_title(f"FPR vs Threshold — {machine_id}", fontsize=14)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.set_ylim([-0.02, max(0.2, max(fprs) * 1.1)])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    path = os.path.join(out_dir, f"threshold_sweep_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Threshold sweep saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_autoencoder(machine_id: str, model_type: str = "dense") -> dict:
    """
    Full evaluation pipeline for one machine and one model type.

    Steps:
        1. Load trained model, scaler, threshold
        2. Load FULL dataset (all states)
        3. Scale sensor data with fitted scaler
        4. Compute reconstruction errors for ALL rows
        5. Map state → binary ground truth
        6. Compute all classification metrics
        7. Generate 4 plots
        8. Save results JSON + return

    Returns:
        Complete metrics dict with plot paths.
    """
    log.info("═" * 60)
    log.info(f"  EVALUATING: {machine_id} — {model_type.upper()} Autoencoder")
    log.info("═" * 60)

    # ── 1. Load assets ────────────────────────────────────────────────────
    model, scaler, threshold = load_model_and_assets(machine_id, model_type)

    # ── 2. Load full dataset ──────────────────────────────────────────────
    df, sensor_cols = load_full_dataset(machine_id)

    if "state" not in df.columns:
        raise ValueError("Dataset must have a 'state' column for evaluation.")

    # ── 3. Prepare data ──────────────────────────────────────────────────
    X_raw = df[sensor_cols].values.astype(np.float32)

    # The data is already normalized (loaded from processed CSV), but we need
    # to ensure the same scaling. If loading raw data, scale it:
    # X_scaled = scaler.transform(X_raw)
    # Since processed data is already normalized, use it directly:
    X_scaled = X_raw

    # ── 4. Compute reconstruction errors ──────────────────────────────────
    if model_type == "dense":
        mse_scores = dense_error(model, X_scaled)
        y_true = compute_ground_truth(df["state"])
    else:
        # LSTM: creates sequences, so we lose first (TIMESTEPS-1) rows
        X_seq = create_sequences(X_scaled, timesteps=TIMESTEPS)
        mse_scores = lstm_error(model, X_seq)
        # Align ground truth: take state from last timestep of each window
        y_true = compute_ground_truth(df["state"].iloc[TIMESTEPS - 1:].reset_index(drop=True))
        # Ensure alignment
        y_true = y_true[:len(mse_scores)]

    log.info(f"  MSE scores: {len(mse_scores):,} samples")
    log.info(f"  Ground truth: {y_true.sum():,} anomalies, {(y_true == 0).sum():,} normal")

    # ── 5. Binary predictions ─────────────────────────────────────────────
    y_pred = (mse_scores > threshold).astype(int)

    # ── 6. Compute metrics ────────────────────────────────────────────────
    metrics = compute_classification_metrics(y_true, y_pred, mse_scores)

    # Add threshold info
    normal_mse = mse_scores[y_true == 0]
    fault_mse  = mse_scores[y_true == 1]

    metrics["threshold"] = round(threshold, 6)
    metrics["threshold_method"] = "mean + 2σ (training normal data)"
    metrics["mean_mse_normal"] = round(float(np.mean(normal_mse)), 6)
    metrics["std_mse_normal"]  = round(float(np.std(normal_mse)), 6)
    metrics["mean_mse_fault"]  = round(float(np.mean(fault_mse)), 6)
    metrics["std_mse_fault"]   = round(float(np.std(fault_mse)), 6)
    metrics["separation_ratio"] = round(
        float(np.mean(fault_mse)) / max(float(np.mean(normal_mse)), 1e-10), 4
    )

    # ── 7. Generate plots ─────────────────────────────────────────────────
    out_dir = os.path.join(EVAL_DIR, machine_id)
    os.makedirs(out_dir, exist_ok=True)

    plot_paths = {}
    try:
        plot_paths["roc_curve"] = plot_roc_curve(
            y_true, mse_scores, machine_id, model_type, out_dir)
    except Exception as e:
        log.error(f"  ROC curve failed: {e}")

    try:
        plot_paths["mse_distribution"] = plot_mse_distribution(
            normal_mse, fault_mse, threshold, machine_id, model_type, out_dir)
    except Exception as e:
        log.error(f"  MSE distribution failed: {e}")

    try:
        plot_paths["confusion_matrix"] = plot_confusion_matrix(
            y_true, y_pred, machine_id, model_type, out_dir)
    except Exception as e:
        log.error(f"  Confusion matrix failed: {e}")

    try:
        plot_paths["threshold_sweep"] = plot_threshold_sweep(
            y_true, mse_scores, threshold, machine_id, model_type, out_dir)
    except Exception as e:
        log.error(f"  Threshold sweep failed: {e}")

    # ── 8. Assemble result ────────────────────────────────────────────────
    result = {
        "machine_id":    machine_id,
        "model_type":    model_type,
        "metrics":       metrics,
        "plots":         {k: os.path.basename(v) for k, v in plot_paths.items()},
        "evaluated_at":  datetime.now(timezone.utc).isoformat(),
        "ground_truth_mapping": STATE_TO_LABEL,
    }

    # Save JSON
    json_path = os.path.join(out_dir, f"metrics_{model_type}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"  Metrics saved → {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(result)

    return result


def _print_summary(result: dict) -> None:
    """Pretty-print evaluation summary to console."""
    m = result["metrics"]
    mid = result["machine_id"]
    mt = result["model_type"].upper()

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS: {mid} -- {mt} Autoencoder")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<28s} {'Value':>12s}")
    print(f"  {'-' * 42}")
    print(f"  {'Accuracy':<28s} {m['accuracy']:>11.2%}")
    print(f"  {'Precision':<28s} {m['precision']:>11.2%}")
    print(f"  {'Recall / Sensitivity':<28s} {m['recall']:>11.2%}")
    print(f"  {'F1 Score':<28s} {m['f1_score']:>11.2%}")
    print(f"  {'AUC-ROC':<28s} {m['auc_roc']:>11.4f}")
    print(f"  {'False Positive Rate (FPR)':<28s} {m['false_positive_rate']:>11.2%}")
    print(f"  {'False Negative Rate (FNR)':<28s} {m['false_negative_rate']:>11.2%}")
    print(f"  {'-' * 42}")
    print(f"  {'Threshold (mean+2s)':<28s} {m['threshold']:>12.6f}")
    print(f"  {'Mean MSE (Normal)':<28s} {m['mean_mse_normal']:>12.6f}")
    print(f"  {'Mean MSE (Fault)':<28s} {m['mean_mse_fault']:>12.6f}")
    print(f"  {'Separation Ratio':<28s} {m['separation_ratio']:>12.4f}x")
    print(f"  {'-' * 42}")

    cm = m["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TP={cm['true_positive']:,}  FP={cm['false_positive']:,}")
    print(f"    FN={cm['false_negative']:,}  TN={cm['true_negative']:,}")
    print(f"  Total: {m['total_samples']:,} samples "
          f"({m['total_anomalies']:,} anomalies, {m['total_normal']:,} normal)")
    print(f"{'=' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-MACHINE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all_machines(model_type: str = "both") -> dict:
    """
    Run evaluation for all available machines.
    Returns combined report dict.
    """
    # Discover available machines from model files
    machines = []
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("autoencoder_") and fname.endswith(".keras"):
            mid = fname.replace("autoencoder_", "").replace(".keras", "")
            machines.append(mid)

    log.info(f"Found {len(machines)} machines to evaluate: {machines}")

    results = {}
    for mid in sorted(machines):
        model_types = ["dense", "lstm"] if model_type == "both" else [model_type]
        results[mid] = {}

        for mt in model_types:
            try:
                results[mid][mt] = evaluate_autoencoder(mid, mt)
            except FileNotFoundError as e:
                log.warning(f"  Skipping {mid}/{mt}: {e}")
            except Exception as e:
                log.error(f"  Error evaluating {mid}/{mt}: {e}")

    # Save combined summary
    summary = _build_summary_table(results)
    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Combined summary saved → {summary_path}")

    _print_comparison_table(summary)
    return results


def _build_summary_table(results: dict) -> dict:
    """Build a comparison table across all machines."""
    rows = []
    for mid, model_results in results.items():
        for mt, result in model_results.items():
            m = result["metrics"]
            rows.append({
                "machine_id":          mid,
                "model_type":          mt,
                "accuracy":            m["accuracy"],
                "precision":           m["precision"],
                "recall":              m["recall"],
                "f1_score":            m["f1_score"],
                "auc_roc":             m["auc_roc"],
                "fpr":                 m["false_positive_rate"],
                "fnr":                 m["false_negative_rate"],
                "threshold":           m["threshold"],
                "separation_ratio":    m["separation_ratio"],
            })
    return {
        "summary": rows,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _print_comparison_table(summary: dict) -> None:
    """Print a formatted comparison table."""
    rows = summary["summary"]
    if not rows:
        return

    print(f"\n{'=' * 100}")
    print(f"  COMPARISON TABLE -- All Machines")
    print(f"{'=' * 100}")
    header = f"  {'Machine':<15s} {'Model':<7s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} " \
             f"{'F1':>7s} {'AUC':>7s} {'FPR':>7s} {'FNR':>7s} {'Sep.':>7s}"
    print(header)
    print(f"  {'-' * 94}")

    for r in rows:
        print(f"  {r['machine_id']:<15s} {r['model_type']:<7s} "
              f"{r['accuracy']:>6.1%} {r['precision']:>6.1%} {r['recall']:>6.1%} "
              f"{r['f1_score']:>6.1%} {r['auc_roc']:>7.4f} "
              f"{r['fpr']:>6.1%} {r['fnr']:>6.1%} {r['separation_ratio']:>6.1f}x")

    print(f"{'=' * 100}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Autoencoder Anomaly Detection Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python models/evaluate_model.py --machine_id PUMP-001 --model dense
  python models/evaluate_model.py --machine_id PUMP-001 --model both
  python models/evaluate_model.py --all --model dense
  python models/evaluate_model.py --all
        """
    )
    parser.add_argument("--machine_id", default="PUMP-001",
                        help="Machine ID to evaluate (default: PUMP-001)")
    parser.add_argument("--model", choices=["dense", "lstm", "both"], default="dense",
                        help="Model type to evaluate (default: dense)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available machines")
    args = parser.parse_args()

    if args.all:
        evaluate_all_machines(model_type=args.model)
    else:
        if args.model == "both":
            for mt in ["dense", "lstm"]:
                try:
                    evaluate_autoencoder(args.machine_id, mt)
                except FileNotFoundError as e:
                    log.warning(f"Skipping {mt}: {e}")
        else:
            evaluate_autoencoder(args.machine_id, args.model)
