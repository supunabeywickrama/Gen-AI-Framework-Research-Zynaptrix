"""
validate_dataset.py — Dataset validation and visualization for Phase 1.

Loads the generated CSV and produces:
  1. State distribution bar chart
  2. Sensor time-series overlay (one subplot per sensor)
  3. Boxplot comparing sensor distributions per state

Usage:
    python validate_dataset.py
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves files instead of opening windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import SENSOR_COLUMNS, MOCK_DATA_PATH

STATE_COLORS = {
    "normal":        "#2ecc71",
    "machine_fault": "#e74c3c",
    "sensor_freeze": "#3498db",
    "sensor_drift":  "#f39c12",
    "idle":          "#95a5a6",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")


def plot_state_distribution(df: pd.DataFrame) -> None:
    counts = df["state"].value_counts()
    colors = [STATE_COLORS.get(s, "gray") for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Machine State Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Row count")
    ax.set_xlabel("State")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "state_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"  ✓ Saved: {out}")
    plt.close()


def plot_sensor_timeseries(df: pd.DataFrame, sample_rows: int = 3000) -> None:
    """Plot first `sample_rows` rows for readability."""
    df_sample = df.iloc[:sample_rows].copy()
    df_sample = df_sample.reset_index(drop=True)

    n = len(SENSOR_COLUMNS)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    fig.suptitle(f"Sensor Time-Series (first {sample_rows:,} readings)", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, SENSOR_COLUMNS):
        for state, group in df_sample.groupby("state"):
            ax.scatter(group.index, group[col],
                       c=STATE_COLORS.get(state, "gray"),
                       s=2, alpha=0.6, label=state)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    # Legend only on top subplot
    legend_patches = [
        mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()
    ]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=3)  # type: ignore
    axes[-1].set_xlabel("Reading index")  # type: ignore

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sensor_timeseries.png")
    plt.savefig(out, dpi=150)
    print(f"  ✓ Saved: {out}")
    plt.close()


def plot_boxplots(df: pd.DataFrame) -> None:
    """Compare sensor distributions per state via boxplots."""
    states = sorted(df["state"].unique())
    n = len(SENSOR_COLUMNS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("Sensor Distribution by Machine State", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, SENSOR_COLUMNS):
        data_by_state = [df[df["state"] == s][col].dropna() for s in states]
        bp = ax.boxplot(data_by_state,
                        patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, state in zip(bp["boxes"], states):
            patch.set_facecolor(STATE_COLORS.get(state, "gray"))
        ax.set_title(col, fontsize=9)
        ax.set_xticks(range(1, len(states) + 1))
        ax.set_xticklabels(states, rotation=25, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sensor_boxplots.png")
    plt.savefig(out, dpi=150)
    print(f"  ✓ Saved: {out}")
    plt.close()


def main():
    if not os.path.exists(MOCK_DATA_PATH):
        print(f"Dataset not found at: {MOCK_DATA_PATH}")
        print("Run 'python generate_dataset.py' first.")
        sys.exit(1)

    print(f"Loading dataset: {MOCK_DATA_PATH}")
    df = pd.read_csv(MOCK_DATA_PATH)
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nGenerating charts …")
    plot_state_distribution(df)
    plot_sensor_timeseries(df)
    plot_boxplots(df)

    print(f"\n✓ All charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
