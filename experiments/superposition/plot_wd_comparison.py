"""
Compare training dynamics and geometry between wd=1.0 and wd=0.3 groups.

Produces:
  Fig 9:  wd=0.3 training curves (same layout as fig6)
  Fig 10: wd=0.3 geometry evolution (same layout as fig7)
  Fig 11: wd comparison — Dense model: wd=1 vs wd=0.3 (grokking timing + geometry)
  Fig 12: wd comparison — All models side by side: val_acc vs overlap_ratio

Usage:
    python experiments/superposition/plot_wd_comparison.py
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

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry"

# ============================================================
# Model groups
# ============================================================

WD1_MODELS = [
    "dense_baseline",
    "sparse_L0_0.80",
    "sparse_L0_0.50",
    "sparse_L0_0.20",
    "sparse_L0_0.10",
]

WD03_MODELS = [
    "dense_baseline_wd0.3",
    "sparse_L0_0.80_wd0.3",
    "sparse_L0_0.50_wd0.3",
    "sparse_L0_0.20_wd0.3",
    "sparse_L0_0.10_wd0.3",
]

LABELS = {
    "dense_baseline": "Dense",
    "sparse_L0_0.80": "L0=0.80",
    "sparse_L0_0.50": "L0=0.50",
    "sparse_L0_0.20": "L0=0.20",
    "sparse_L0_0.10": "L0=0.10",
    "dense_baseline_wd0.3": "Dense",
    "sparse_L0_0.80_wd0.3": "L0=0.80",
    "sparse_L0_0.50_wd0.3": "L0=0.50",
    "sparse_L0_0.20_wd0.3": "L0=0.20",
    "sparse_L0_0.10_wd0.3": "L0=0.10",
}

COLORS = {
    "dense_baseline": "#2c3e50", "dense_baseline_wd0.3": "#2c3e50",
    "sparse_L0_0.80": "#2980b9", "sparse_L0_0.80_wd0.3": "#2980b9",
    "sparse_L0_0.50": "#27ae60", "sparse_L0_0.50_wd0.3": "#27ae60",
    "sparse_L0_0.20": "#f39c12", "sparse_L0_0.20_wd0.3": "#f39c12",
    "sparse_L0_0.10": "#e74c3c", "sparse_L0_0.10_wd0.3": "#e74c3c",
}

LINESTYLES = {
    "dense_baseline": "-", "dense_baseline_wd0.3": "-",
    "sparse_L0_0.80": "--", "sparse_L0_0.80_wd0.3": "--",
    "sparse_L0_0.50": "-.", "sparse_L0_0.50_wd0.3": "-.",
    "sparse_L0_0.20": ":", "sparse_L0_0.20_wd0.3": ":",
    "sparse_L0_0.10": (0, (3, 1, 1, 1)), "sparse_L0_0.10_wd0.3": (0, (3, 1, 1, 1)),
}


def load_histories(model_list):
    histories = {}
    for m in model_list:
        path = CKPT_DIR / f"{m}_history.json"
        if path.exists():
            with open(path) as f:
                histories[m] = json.load(f)
    return histories


def plot_group_training(histories, models, wd_label, fig_num):
    """Training curves for one wd group."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = [
        ("train_loss", "Train Loss", axes[0, 0], True),
        ("val_loss", "Val Loss", axes[0, 1], True),
        ("train_acc", "Train Accuracy", axes[1, 0], False),
        ("val_acc", "Val Accuracy", axes[1, 1], False),
    ]

    for key, title, ax, use_log in panels:
        for m in models:
            if m not in histories:
                continue
            h = histories[m]
            ax.plot(h["steps"], h[key],
                    color=COLORS[m], linestyle=LINESTYLES[m],
                    linewidth=1.5, label=LABELS[m])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        if use_log:
            ax.set_yscale('log')
        if 'acc' in key:
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color='grey', linestyle=':', alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f"Training Dynamics (weight_decay={wd_label})",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / f"fig{fig_num}_training_curves_wd{wd_label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_group_geometry(histories, models, wd_label, fig_num):
    """Geometry evolution for one wd group."""
    layer = "output_proj"
    metrics = [
        ("overlap_ratio", "E[cos²] / (1/m)", "Overlap Ratio"),
        ("max_abs_overlap", "max |cos sim|", "Max Overlap"),
        ("mean_norm", "mean ||W_i||", "Mean Row Norm"),
        ("std_norm", "std(||W_i||)", "Norm Std Dev"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for idx, (mkey, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for m in models:
            if m not in histories:
                continue
            h = histories[m]
            geo = h["geometry"].get(layer, {})
            vals = geo.get(mkey, [])
            if vals:
                ax.plot(h["steps"][:len(vals)], vals,
                        color=COLORS[m], linestyle=LINESTYLES[m],
                        linewidth=1.5, label=LABELS[m])

        if mkey == "overlap_ratio":
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1,
                       label="1/m prediction")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f"Geometry Evolution: output_proj (wd={wd_label})",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / f"fig{fig_num}_geometry_evolution_wd{wd_label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def fig11_wd_dense_comparison(h_wd1, h_wd03):
    """Compare Dense model under wd=1 vs wd=0.3: grokking timing + geometry."""
    m1, m2 = "dense_baseline", "dense_baseline_wd0.3"
    if m1 not in h_wd1 or m2 not in h_wd03:
        print("  [fig11 skipped: missing dense data]")
        return

    d1, d2 = h_wd1[m1], h_wd03[m2]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Row 0: Val accuracy
    ax = axes[0]
    ax.plot(d1["steps"], d1["val_acc"], color='#e74c3c', linewidth=2, label="wd=1.0")
    ax.plot(d2["steps"], d2["val_acc"], color='#2980b9', linewidth=2, label="wd=0.3")
    ax.set_ylabel("Val Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color='grey', linestyle=':', alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title("Grokking: Dense Model Under Different Weight Decay",
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Row 1: Val loss
    ax = axes[1]
    ax.plot(d1["steps"], d1["val_loss"], color='#e74c3c', linewidth=2, label="wd=1.0")
    ax.plot(d2["steps"], d2["val_loss"], color='#2980b9', linewidth=2, label="wd=0.3")
    ax.set_ylabel("Val Loss", fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Row 2: Overlap ratio
    ax = axes[2]
    geo1 = d1["geometry"].get("output_proj", {})
    geo2 = d2["geometry"].get("output_proj", {})
    r1 = geo1.get("overlap_ratio", [])
    r2 = geo2.get("overlap_ratio", [])
    if r1:
        ax.plot(d1["steps"][:len(r1)], r1, color='#e74c3c', linewidth=2, label="wd=1.0")
    if r2:
        ax.plot(d2["steps"][:len(r2)], r2, color='#2980b9', linewidth=2, label="wd=0.3")
    ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5)
    ax.set_ylabel("Overlap Ratio\n(output_proj)", fontsize=11)
    ax.set_xlabel("Step", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig11_wd_dense_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def fig12_wd_all_comparison(h_wd1, h_wd03):
    """Side by side: all models, wd=1 vs wd=0.3. Two rows: val_acc, overlap_ratio."""
    # Pair up models
    pairs = [
        ("dense_baseline", "dense_baseline_wd0.3", "Dense"),
        ("sparse_L0_0.80", "sparse_L0_0.80_wd0.3", "L0=0.80"),
        ("sparse_L0_0.50", "sparse_L0_0.50_wd0.3", "L0=0.50"),
        ("sparse_L0_0.20", "sparse_L0_0.20_wd0.3", "L0=0.20"),
        ("sparse_L0_0.10", "sparse_L0_0.10_wd0.3", "L0=0.10"),
    ]

    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7), sharex='col')

    for col, (m1, m2, label) in enumerate(pairs):
        d1 = h_wd1.get(m1)
        d2 = h_wd03.get(m2)

        # Row 0: val_acc
        ax = axes[0, col]
        if d1:
            ax.plot(d1["steps"], d1["val_acc"], color='#e74c3c', linewidth=1.5, label="wd=1.0")
        if d2:
            ax.plot(d2["steps"], d2["val_acc"], color='#2980b9', linewidth=1.5, label="wd=0.3")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color='grey', linestyle=':', alpha=0.3)
        ax.set_title(label, fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel("Val Accuracy", fontsize=9)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # Row 1: overlap_ratio
        ax = axes[1, col]
        if d1:
            geo1 = d1["geometry"].get("output_proj", {})
            r1 = geo1.get("overlap_ratio", [])
            if r1:
                ax.plot(d1["steps"][:len(r1)], r1, color='#e74c3c', linewidth=1.5)
        if d2:
            geo2 = d2["geometry"].get("output_proj", {})
            r2 = geo2.get("overlap_ratio", [])
            if r2:
                ax.plot(d2["steps"][:len(r2)], r2, color='#2980b9', linewidth=1.5)
        ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5)
        if col == 0:
            ax.set_ylabel("Overlap Ratio\n(output_proj)", fontsize=9)
        ax.set_xlabel("Step", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Weight Decay Comparison: wd=1.0 (red) vs wd=0.3 (blue)\n"
                 "Top: Val Accuracy (grokking)  |  Bottom: Overlap Ratio (geometry)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / "fig12_wd_all_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    h_wd1 = load_histories(WD1_MODELS)
    h_wd03 = load_histories(WD03_MODELS)
    print(f"Loaded: {len(h_wd1)} wd=1 models, {len(h_wd03)} wd=0.3 models")

    # Summary table
    print()
    print(f"{'Model':<30} {'wd':>4} {'Steps':>6} {'val_acc':>10} {'out_ratio':>11}")
    print("-" * 65)
    for m, h in {**h_wd1, **h_wd03}.items():
        va = h["val_acc"][-1]
        wd = h["config"]["weight_decay"]
        geo = h["geometry"].get("output_proj", {})
        r = geo.get("overlap_ratio", [0])[-1]
        print(f"{m:<30} {wd:>4} {h['steps'][-1]:>6} {va:>10.4f} {r:>11.2f}")
    print()

    # Generate wd=0.3 group plots
    print("Generating wd=0.3 group plots...")
    plot_group_training(h_wd03, WD03_MODELS, "0.3", 9)
    plot_group_geometry(h_wd03, WD03_MODELS, "0.3", 10)

    # Generate comparison plots
    print("Generating comparison plots...")
    fig11_wd_dense_comparison(h_wd1, h_wd03)
    fig12_wd_all_comparison(h_wd1, h_wd03)

    print()
    print("All figures saved to:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
