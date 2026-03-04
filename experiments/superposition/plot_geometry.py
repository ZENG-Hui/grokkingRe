"""
Generate publication-quality visualizations from Phase 1 geometry analysis.

Produces 5 figures:
  Fig 1: Overlap ratio vs L0 (key layers)
  Fig 2: Max absolute overlap vs L0
  Fig 3: Row norm histograms (output_proj across models)
  Fig 4: Ratio heatmap (all layers x all models)
  Fig 5: Variance ratio (ETF-likeness)

Usage:
    python experiments/superposition/plot_geometry.py

Output:
    experiments/superposition/results/geometry/fig_*.png
"""

import sys
import json
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Also load raw models to get actual norm histograms
import torch
from superposition_utils import extract_weight_matrices, compute_row_norms, compute_cosine_gram

# ============================================================
# Config
# ============================================================

JSON_PATH = SCRIPT_DIR / "results" / "geometry" / "geometry_summary.json"
CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry"

# Model display order and labels
MODEL_ORDER = [
    "dense_baseline",
    "sparse_L0_0.80",
    "sparse_L0_0.50",
    "sparse_L0_0.20",
    "sparse_L0_0.10",
]
MODEL_LABELS = {
    "dense_baseline": "Dense",
    "sparse_L0_0.80": "L0=0.80",
    "sparse_L0_0.50": "L0=0.50",
    "sparse_L0_0.20": "L0=0.20",
    "sparse_L0_0.10": "L0=0.10",
}
MODEL_COLORS = {
    "dense_baseline": "#2c3e50",
    "sparse_L0_0.80": "#2980b9",
    "sparse_L0_0.50": "#27ae60",
    "sparse_L0_0.20": "#f39c12",
    "sparse_L0_0.10": "#e74c3c",
}
VAL_ACCS = {
    "dense_baseline": 0.491,
    "sparse_L0_0.80": 0.523,
    "sparse_L0_0.50": 0.962,
    "sparse_L0_0.20": 0.979,
    "sparse_L0_0.10": 0.232,
}

# Key layers to highlight
KEY_LAYERS = ["output_proj", "attn_Q_block1", "ffn_up_block1"]
KEY_LAYER_LABELS = {
    "output_proj": "Output Proj [99×128]",
    "attn_Q_block1": "Attn Q Block1 [128×128]",
    "ffn_up_block1": "FFN Up Block1 [512×128]",
}

# All layers for heatmap (readable order)
ALL_LAYERS = [
    "token_embed",
    "attn_Q_block0", "attn_K_block0", "attn_V_block0", "attn_O_block0",
    "ffn_up_block0", "ffn_down_block0",
    "attn_Q_block1", "attn_K_block1", "attn_V_block1", "attn_O_block1",
    "ffn_up_block1", "ffn_down_block1",
    "output_proj",
]


def load_data():
    with open(JSON_PATH) as f:
        return json.load(f)


def get_val(data, model, layer, category, key):
    """Safely extract a value from the nested JSON."""
    try:
        return data[model][layer][category][key]
    except KeyError:
        return None


# ============================================================
# Fig 1: Overlap ratio vs L0
# ============================================================
def fig1_overlap_ratio(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    offsets = np.arange(len(KEY_LAYERS)) - (len(KEY_LAYERS) - 1) / 2

    layer_colors = ["#3498db", "#e67e22", "#2ecc71"]

    for i, layer in enumerate(KEY_LAYERS):
        ratios = []
        for model in MODEL_ORDER:
            msq = get_val(data, model, layer, "overlap", "mean_sq_overlap")
            one_m = get_val(data, model, layer, "overlap", "one_over_m")
            ratios.append(msq / one_m if one_m else 0)

        bars = ax.bar(x + offsets[i] * width, ratios, width * 0.9,
                      label=KEY_LAYER_LABELS[layer], color=layer_colors[i],
                      edgecolor='white', linewidth=0.5)

        # Annotate bars that are far from 1.0
        for j, v in enumerate(ratios):
            if abs(v - 1.0) > 0.3:
                ax.text(x[j] + offsets[i] * width, v + 0.1, f"{v:.1f}",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.2,
               label='Paper prediction (ratio=1)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_LABELS[m]}\nacc={VAL_ACCS[m]:.2f}" for m in MODEL_ORDER],
                       fontsize=8)
    ax.set_ylabel("E[cos²] / (1/m)", fontsize=11)
    ax.set_title("Overlap Ratio vs Sparsity Level", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, max(6.5, ax.get_ylim()[1]))
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig1_overlap_ratio.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 2: Max absolute overlap vs L0
# ============================================================
def fig2_max_overlap(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    offsets = np.arange(len(KEY_LAYERS)) - (len(KEY_LAYERS) - 1) / 2
    layer_colors = ["#3498db", "#e67e22", "#2ecc71"]

    for i, layer in enumerate(KEY_LAYERS):
        vals = []
        for model in MODEL_ORDER:
            v = get_val(data, model, layer, "overlap", "max_abs_overlap")
            vals.append(v if v else 0)

        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=KEY_LAYER_LABELS[layer], color=layer_colors[i],
                      edgecolor='white', linewidth=0.5)

        for j, v in enumerate(vals):
            if v > 0.5:
                ax.text(x[j] + offsets[i] * width, v + 0.02, f"{v:.2f}",
                        ha='center', va='bottom', fontsize=7, fontweight='bold',
                        color='red')

    # Welch bound for ffn_up_block1 (512 in 128-d)
    ax.axhline(0.0766, color='grey', linestyle=':', alpha=0.5,
               label='Welch bound (512 in 128-d)')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_LABELS[m]}\nacc={VAL_ACCS[m]:.2f}" for m in MODEL_ORDER],
                       fontsize=8)
    ax.set_ylabel("max |cos similarity|", fontsize=11)
    ax.set_title("Maximum Pairwise Overlap vs Sparsity Level", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig2_max_overlap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 3: Row norm histograms for output_proj
# ============================================================
def fig3_norm_histograms():
    """Load actual models and plot row norm histograms for output_proj."""

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)

    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes[idx]
        ckpt_path = CKPT_DIR / f"{model_name}.pt"

        if not ckpt_path.exists():
            ax.set_title(MODEL_LABELS[model_name])
            ax.text(0.5, 0.5, "No checkpoint", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)

        # Find output_proj weight
        W = None
        for key, val in state.items():
            if 'output_proj' in key and 'weight' in key and val.dim() == 2:
                W = val
                break

        if W is None:
            # For dense model, try last linear layer
            for key, val in state.items():
                if val.dim() == 2 and val.shape[0] == 99:
                    W = val

        if W is None:
            ax.set_title(MODEL_LABELS[model_name])
            ax.text(0.5, 0.5, "No output_proj", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        norms = compute_row_norms(W).numpy()

        ax.hist(norms, bins=25, color=MODEL_COLORS[model_name], edgecolor='white',
                linewidth=0.5, alpha=0.85)
        ax.axvline(np.mean(norms), color='black', linestyle='-', linewidth=1.2,
                   label=f'mean={np.mean(norms):.2f}')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(1.0, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_title(f"{MODEL_LABELS[model_name]}\nacc={VAL_ACCS[model_name]:.2f}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("||W_i||", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)

        # Add range annotation
        ax.text(0.95, 0.75, f"std={np.std(norms):.2f}\nrange=[{norms.min():.1f},{norms.max():.1f}]",
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))

    fig.suptitle("Row Norm Distribution: output_proj [99×128]", fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / "fig3_norm_histograms.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 4: Ratio heatmap (all layers x all models)
# ============================================================
def fig4_ratio_heatmap(data):
    # Build matrix
    n_models = len(MODEL_ORDER)
    n_layers = len(ALL_LAYERS)
    matrix = np.full((n_layers, n_models), np.nan)

    for j, model in enumerate(MODEL_ORDER):
        for i, layer in enumerate(ALL_LAYERS):
            msq = get_val(data, model, layer, "overlap", "mean_sq_overlap")
            one_m = get_val(data, model, layer, "overlap", "one_over_m")
            if msq is not None and one_m and one_m > 0:
                matrix[i, j] = msq / one_m

    fig, ax = plt.subplots(figsize=(7, 7))

    # Use diverging colormap centered at 1.0
    vmin = 0.8
    vmax = min(6.0, np.nanmax(matrix))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', norm=norm)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9, rotation=0)
    ax.set_yticks(range(n_layers))

    layer_labels = []
    for l in ALL_LAYERS:
        s = data[MODEL_ORDER[0]][l]["shape"] if l in data[MODEL_ORDER[0]] else []
        layer_labels.append(f"{l} {s}")
    ax.set_yticklabels(layer_labels, fontsize=7)

    # Annotate cells
    for i in range(n_layers):
        for j in range(n_models):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if (v > 3.0 or v < 0.85) else 'black'
                ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                        fontsize=6.5, color=color, fontweight='bold' if abs(v - 1.0) > 0.5 else 'normal')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="E[cos²] / (1/m)")
    cbar.ax.axhline(1.0, color='black', linewidth=1.5)

    ax.set_title("Overlap Ratio Heatmap\n(1.0 = matches paper prediction, red = excess overlap)",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Model", fontsize=10)

    fig.tight_layout()
    path = OUT_DIR / "fig4_ratio_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Fig 5: Variance ratio (ETF-likeness)
# ============================================================
def fig5_variance_ratio(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    offsets = np.arange(len(KEY_LAYERS)) - (len(KEY_LAYERS) - 1) / 2
    layer_colors = ["#3498db", "#e67e22", "#2ecc71"]

    for i, layer in enumerate(KEY_LAYERS):
        vals = []
        for model in MODEL_ORDER:
            var_sq = get_val(data, model, layer, "overlap", "var_sq_overlap")
            rand_var = get_val(data, model, layer, "overlap", "random_var")
            vals.append(var_sq / rand_var if rand_var else 0)

        ax.bar(x + offsets[i] * width, vals, width * 0.9,
               label=KEY_LAYER_LABELS[layer], color=layer_colors[i],
               edgecolor='white', linewidth=0.5)

        for j, v in enumerate(vals):
            if v > 2.5:
                ax.text(x[j] + offsets[i] * width, v + 0.3, f"{v:.1f}",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.2,
               label='Random unit vectors')
    ax.axhline(0.0, color='green', linestyle=':', alpha=0.5, linewidth=1,
               label='ETF (perfect uniformity)')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_LABELS[m]}\nacc={VAL_ACCS[m]:.2f}" for m in MODEL_ORDER],
                       fontsize=8)
    ax.set_ylabel("Var[cos²] / Var_random", fontsize=11)
    ax.set_title("ETF-likeness: Overlap Variance vs Random Vectors",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig5_variance_ratio.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data()

    print("Generating visualizations...")
    print()

    fig1_overlap_ratio(data)
    fig2_max_overlap(data)
    fig3_norm_histograms()
    fig4_ratio_heatmap(data)
    fig5_variance_ratio(data)

    print()
    print("All figures saved to:", OUT_DIR)
    print()
    print("Summary of files:")
    for f in sorted(OUT_DIR.glob("fig_*.png")) or sorted(OUT_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
