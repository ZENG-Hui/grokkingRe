"""
Visualize training curves and geometry evolution during grokking.

Produces:
  Fig 6: Training & validation curves (loss + accuracy) for all models
  Fig 7: Geometry evolution during training (overlap_ratio, max_overlap, norm_std)
  Fig 8: Combined grokking + geometry (shared x-axis, key insight figure)

Usage:
    python experiments/superposition/plot_training.py

Output:
    experiments/superposition/results/geometry/fig6_training_curves.png
    experiments/superposition/results/geometry/fig7_geometry_evolution.png
    experiments/superposition/results/geometry/fig8_grokking_vs_geometry.png
"""

import sys
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry"

MODEL_ORDER = [
    "dense_baseline",
    "sparse_L0_0.80",
    "sparse_L0_0.50",
    "sparse_L0_0.20",
    "sparse_L0_0.10",
]
MODEL_LABELS = {
    "dense_baseline": "Dense (wd=1)",
    "sparse_L0_0.80": "Sparse L0=0.80",
    "sparse_L0_0.50": "Sparse L0=0.50",
    "sparse_L0_0.20": "Sparse L0=0.20",
    "sparse_L0_0.10": "Sparse L0=0.10",
}
MODEL_COLORS = {
    "dense_baseline": "#2c3e50",
    "sparse_L0_0.80": "#2980b9",
    "sparse_L0_0.50": "#27ae60",
    "sparse_L0_0.20": "#f39c12",
    "sparse_L0_0.10": "#e74c3c",
}
MODEL_LINESTYLES = {
    "dense_baseline": "-",
    "sparse_L0_0.80": "--",
    "sparse_L0_0.50": "-.",
    "sparse_L0_0.20": ":",
    "sparse_L0_0.10": (0, (3, 1, 1, 1)),
}


def load_histories():
    """Load all training history JSON files."""
    histories = {}
    for model in MODEL_ORDER:
        path = CKPT_DIR / f"{model}_history.json"
        if path.exists():
            with open(path) as f:
                histories[model] = json.load(f)
        else:
            print(f"  WARNING: {path} not found, skipping {model}")
    return histories


# ============================================================
# Fig 6: Training & validation curves
# ============================================================
def fig6_training_curves(histories):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    panels = [
        ("train_loss", "Train Loss", axes[0, 0], True),
        ("val_loss", "Val Loss", axes[0, 1], True),
        ("train_acc", "Train Accuracy", axes[1, 0], False),
        ("val_acc", "Val Accuracy", axes[1, 1], False),
    ]

    for key, title, ax, use_log in panels:
        for model in MODEL_ORDER:
            if model not in histories:
                continue
            h = histories[model]
            steps = h["steps"]
            vals = h[key]
            ax.plot(steps, vals,
                    color=MODEL_COLORS[model],
                    linestyle=MODEL_LINESTYLES[model],
                    linewidth=1.5,
                    label=MODEL_LABELS[model])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        if use_log:
            ax.set_yscale('log')
        if 'acc' in key.lower():
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color='grey', linestyle=':', alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle("Training Dynamics: Dense vs Sparse at Different L0 Levels",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = OUT_DIR / "fig6_training_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 7: Geometry evolution during training
# ============================================================
def fig7_geometry_evolution(histories):
    layer = "output_proj"
    metrics = [
        ("overlap_ratio", "E[cos²] / (1/m)", "Overlap Ratio"),
        ("max_abs_overlap", "max |cos sim|", "Max Overlap"),
        ("mean_norm", "mean ||W_i||", "Mean Row Norm"),
        ("std_norm", "std(||W_i||)", "Norm Std Dev"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for idx, (metric_key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for model in MODEL_ORDER:
            if model not in histories:
                continue
            h = histories[model]
            steps = h["steps"]
            geo = h["geometry"].get(layer, {})
            vals = geo.get(metric_key, [])

            if not vals:
                continue

            ax.plot(steps[:len(vals)], vals,
                    color=MODEL_COLORS[model],
                    linestyle=MODEL_LINESTYLES[model],
                    linewidth=1.5,
                    label=MODEL_LABELS[model])

        # Reference lines
        if metric_key == "overlap_ratio":
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1,
                       label="1/m prediction")

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f"Geometry Evolution: {layer} During Training",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = OUT_DIR / "fig7_geometry_evolution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 8: Combined grokking + geometry (the key figure)
# ============================================================
def fig8_grokking_vs_geometry(histories):
    """
    For each model: top panel = val_acc, bottom panel = overlap_ratio.
    Shared x-axis (step). This is the key figure showing whether
    geometry changes correlate with the grokking transition.
    """
    n_models = len([m for m in MODEL_ORDER if m in histories])
    if n_models == 0:
        print("  [fig8 skipped: no data]")
        return

    fig, axes = plt.subplots(3, n_models, figsize=(3.5 * n_models, 8),
                             sharex='col')
    if n_models == 1:
        axes = axes.reshape(3, 1)

    col = 0
    for model in MODEL_ORDER:
        if model not in histories:
            continue
        h = histories[model]
        steps = h["steps"]
        color = MODEL_COLORS[model]

        # Row 0: Val accuracy
        ax0 = axes[0, col]
        ax0.plot(steps, h["val_acc"], color=color, linewidth=1.5)
        ax0.set_ylim(-0.05, 1.05)
        ax0.axhline(1.0, color='grey', linestyle=':', alpha=0.3)
        ax0.set_title(MODEL_LABELS[model], fontsize=10, fontweight='bold')
        if col == 0:
            ax0.set_ylabel("Val Accuracy", fontsize=9)
        ax0.grid(alpha=0.3)

        # Row 1: Overlap ratio (output_proj)
        ax1 = axes[1, col]
        geo_out = h["geometry"].get("output_proj", {})
        ratio = geo_out.get("overlap_ratio", [])
        if ratio:
            ax1.plot(steps[:len(ratio)], ratio, color=color, linewidth=1.5)
            ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        if col == 0:
            ax1.set_ylabel("Overlap Ratio\n(output_proj)", fontsize=9)
        ax1.grid(alpha=0.3)

        # Row 2: Max overlap (output_proj)
        ax2 = axes[2, col]
        max_ov = geo_out.get("max_abs_overlap", [])
        if max_ov:
            ax2.plot(steps[:len(max_ov)], max_ov, color=color, linewidth=1.5)
        if col == 0:
            ax2.set_ylabel("Max |cos sim|\n(output_proj)", fontsize=9)
        ax2.set_xlabel("Step", fontsize=9)
        ax2.grid(alpha=0.3)

        col += 1

    fig.suptitle("Grokking vs Geometry Evolution\n"
                 "(Does geometry change when generalization happens?)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    path = OUT_DIR / "fig8_grokking_vs_geometry.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training histories...")
    histories = load_histories()
    print(f"  Loaded {len(histories)} models")

    if not histories:
        print("ERROR: No history files found. Run train_sweep.py first.")
        return

    # Print summary table
    print()
    print(f"{'Model':<25} {'Steps':>6} {'Final train_acc':>15} {'Final val_acc':>15} {'Final out_ratio':>15}")
    print("-" * 80)
    for model, h in histories.items():
        ta = h["train_acc"][-1] if h["train_acc"] else 0
        va = h["val_acc"][-1] if h["val_acc"] else 0
        geo = h["geometry"].get("output_proj", {})
        ratio = geo.get("overlap_ratio", [0])[-1] if geo.get("overlap_ratio") else 0
        print(f"{model:<25} {h['steps'][-1]:>6} {ta:>15.4f} {va:>15.4f} {ratio:>15.2f}")
    print()

    print("Generating figures...")
    fig6_training_curves(histories)
    fig7_geometry_evolution(histories)
    fig8_grokking_vs_geometry(histories)

    print()
    print("All figures saved to:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
