"""
Plot overlap ratio as a function of L0 (sparsity level) for both wd groups.
Shows the U-shaped curve: geometry is best at intermediate sparsity.

Usage:
    python experiments/superposition/plot_L0_sweep.py
"""

import sys
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry"


def load_final_metrics():
    """Load final overlap_ratio and val_acc from all history files."""
    results = []
    for path in sorted(CKPT_DIR.glob("*_history.json")):
        with open(path) as f:
            h = json.load(f)
        config = h["config"]
        wd = config["weight_decay"]
        l0 = config.get("final_L0", 1.0)
        label = h["label"]
        va = h["val_acc"][-1] if h["val_acc"] else 0

        geo = h["geometry"].get("output_proj", {})
        ratio = geo.get("overlap_ratio", [0])[-1] if geo.get("overlap_ratio") else 0
        max_ov = geo.get("max_abs_overlap", [0])[-1] if geo.get("max_abs_overlap") else 0
        std_n = geo.get("std_norm", [0])[-1] if geo.get("std_norm") else 0

        results.append({
            "label": label, "wd": wd, "l0": l0,
            "val_acc": va, "overlap_ratio": ratio,
            "max_abs_overlap": max_ov, "std_norm": std_n,
        })
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_final_metrics()

    # Split by wd group
    wd1 = sorted([r for r in results if abs(r["wd"] - 1.0) < 0.01], key=lambda r: r["l0"])
    wd03 = sorted([r for r in results if abs(r["wd"] - 0.3) < 0.01], key=lambda r: r["l0"])

    # Print table
    print(f"{'Label':<30} {'wd':>4} {'L0':>5} {'val_acc':>9} {'ratio':>8} {'max|cos|':>9} {'norm_std':>9}")
    print("-" * 80)
    for r in wd1 + wd03:
        print(f"{r['label']:<30} {r['wd']:>4} {r['l0']:>5.2f} {r['val_acc']:>9.4f} "
              f"{r['overlap_ratio']:>8.2f} {r['max_abs_overlap']:>9.4f} {r['std_norm']:>9.4f}")
    print()

    # ============================================================
    # Fig 13: Overlap ratio vs L0 (the U-shape figure)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("overlap_ratio", "E[cos²] / (1/m)", "Overlap Ratio vs Sparsity"),
        ("max_abs_overlap", "max |cos sim|", "Max Overlap vs Sparsity"),
        ("val_acc", "Validation Accuracy", "Performance vs Sparsity"),
    ]

    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]

        for data, wd_label, color, marker in [
            (wd1, "wd=1.0", "#e74c3c", "o"),
            (wd03, "wd=0.3", "#2980b9", "s"),
        ]:
            l0s = [r["l0"] for r in data]
            vals = [r[key] for r in data]
            ax.plot(l0s, vals, color=color, marker=marker, markersize=7,
                    linewidth=2, label=wd_label)

            # Annotate min point for overlap metrics
            if key in ("overlap_ratio", "max_abs_overlap") and vals:
                min_idx = np.argmin(vals)
                ax.annotate(f"L0={l0s[min_idx]:.2f}\n{vals[min_idx]:.2f}",
                            xy=(l0s[min_idx], vals[min_idx]),
                            xytext=(10, 15), textcoords='offset points',
                            fontsize=8, fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        if key == "overlap_ratio":
            ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5,
                       label="1/m (uniform)")

        ax.set_xlabel("L0 (fraction of nonzero weights)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.05)
        ax.invert_xaxis()  # Dense (L0=1) on left, sparse on right

    fig.suptitle("Effect of Sparsity Level on Representation Geometry\n"
                 "(x-axis reversed: Dense on left, Sparse on right)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / "fig13_L0_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")

    # ============================================================
    # Fig 14: Overlap ratio vs val_acc scatter (does good geometry = good performance?)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    for data, wd_label, color, marker in [
        (wd1, "wd=1.0", "#e74c3c", "o"),
        (wd03, "wd=0.3", "#2980b9", "s"),
    ]:
        for r in data:
            ax.scatter(r["overlap_ratio"], r["val_acc"],
                       color=color, marker=marker, s=80, alpha=0.8,
                       edgecolors='black', linewidths=0.5)
            ax.annotate(f"L0={r['l0']:.1f}", xy=(r["overlap_ratio"], r["val_acc"]),
                        xytext=(5, -10), textcoords='offset points',
                        fontsize=7, color=color)

    # Legend entries
    ax.scatter([], [], color='#e74c3c', marker='o', s=80, label='wd=1.0', edgecolors='black')
    ax.scatter([], [], color='#2980b9', marker='s', s=80, label='wd=0.3', edgecolors='black')

    ax.axvline(1.0, color='grey', linestyle='--', alpha=0.5, label='1/m (uniform)')
    ax.set_xlabel("Final Overlap Ratio (output_proj)", fontsize=11)
    ax.set_ylabel("Final Validation Accuracy", fontsize=11)
    ax.set_title("Does Geometry Predict Performance?", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    path = OUT_DIR / "fig14_ratio_vs_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
