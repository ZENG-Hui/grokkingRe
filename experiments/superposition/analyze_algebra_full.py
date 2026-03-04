"""
Analyze algebraic structure encoded across ALL layers of the model.

For each layer, we extract how 97 token representations are transformed,
and check whether the pairwise similarity structure encodes:
  1. Numerical neighborhood (circular topology)
  2. Additive inverses (a, 97-a)
  3. Multiplicative structure (quadratic residues, discrete log)

Layers analyzed:
  - token_embedding: raw input representations [99, 128]
  - Q/K projections: what tokens attend to each other (embed → Q/K space)
  - FFN: internal feature representations (embed → FFN hidden space)
  - output_proj: final answer readout [99, 128]

Usage:
    python experiments/superposition/analyze_algebra_full.py

Output:
    results/geometry/algebraic_structure/  (text + plots)
"""

import sys
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry" / "algebraic_structure"
P = 97


# ============================================================
# Algebraic utilities (minimal, readable)
# ============================================================

def is_quadratic_residue(a, p=P):
    if int(a) % p == 0:
        return True
    return pow(int(a), (p - 1) // 2, p) == 1


def get_additive_inverse_pairs(p=P):
    return [(a, p - a) for a in range(1, (p + 1) // 2)]


def get_qr_groups(p=P):
    qr = [a for a in range(1, p) if is_quadratic_residue(a, p)]
    qnr = [a for a in range(1, p) if not is_quadratic_residue(a, p)]
    return qr, qnr


# ============================================================
# Core analysis: given 97 token vectors, compute structure metrics
# ============================================================

def analyze_token_vectors(V, layer_name):
    """
    V: [97, d] — one vector per token (0..96)
    Returns a dict of structure metrics + the cosine gram matrix.
    """
    V_norm = F.normalize(V, dim=1)
    G = (V_norm @ V_norm.T).numpy()
    np.fill_diagonal(G, 0)

    results = {"layer": layer_name, "shape": list(V.shape)}

    # 1. Numerical neighborhood
    neighborhood = {}
    for dist in [1, 2, 5, 10, 24, 48]:
        pairs = [(a, (a + dist) % P) for a in range(P)]
        sims = [G[a, b] for a, b in pairs if a != b]
        neighborhood[dist] = {"mean": float(np.mean(sims)), "std": float(np.std(sims))}
    results["neighborhood"] = neighborhood

    # 2. Additive inverses
    inv_pairs = get_additive_inverse_pairs()
    inv_sims = np.array([G[a, b] for a, b in inv_pairs])
    mask = np.ones_like(G, dtype=bool)
    np.fill_diagonal(mask, False)
    all_sims = G[mask]
    results["additive_inverse"] = {
        "mean": float(inv_sims.mean()),
        "std": float(inv_sims.std()),
        "all_mean": float(all_sims.mean()),
        "all_std": float(all_sims.std()),
        "signal_stddevs": float((inv_sims.mean() - all_sims.mean()) / (all_sims.std() + 1e-8)),
    }

    # 3. Quadratic residue
    qr, qnr = get_qr_groups()
    qr_sims = [G[i, j] for i in qr for j in qr if i < j]
    qnr_sims = [G[i, j] for i in qnr for j in qnr if i < j]
    cross_sims = [G[i, j] for i in qr for j in qnr]
    results["quadratic_residue"] = {
        "within_qr": float(np.mean(qr_sims)),
        "within_qnr": float(np.mean(qnr_sims)),
        "cross": float(np.mean(cross_sims)),
    }

    return results, G


def extract_token_representations(ckpt_path):
    """
    Load model and extract per-token representations at each layer.
    Returns dict: layer_name -> [97, d] tensor
    """
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    reps = {}

    # 1. Token embedding: direct rows [99, 128] -> take first 97
    for key, val in state.items():
        if 'token_embedding' in key and 'weight' in key and val.dim() == 2:
            reps['token_embedding'] = val[:P].clone()
            embed = val[:P].clone()  # save for projections

    # If no explicit token_embedding found, try alternate names
    if 'token_embedding' not in reps:
        for key, val in state.items():
            if 'embed' in key.lower() and 'weight' in key and val.dim() == 2:
                if val.shape[0] >= P:
                    reps['token_embedding'] = val[:P].clone()
                    embed = val[:P].clone()
                    break

    if 'token_embedding' not in reps:
        print("  WARNING: no embedding found")
        return reps

    # 2. Output projection: direct rows [99, 128] -> take first 97
    for key, val in state.items():
        if 'output_proj' in key and 'weight' in key and val.dim() == 2:
            reps['output_proj'] = val[:P].clone()
        # Dense model fallback
        if val.dim() == 2 and val.shape[0] == P + 2 and 'output_proj' not in reps:
            reps['output_proj'] = val[:P].clone()

    # 3. For attention and FFN: project embeddings through weight matrices
    # Q_i = W_Q @ embed_i  =>  Q = embed @ W_Q^T
    # For packed in_proj_weight: [3*d, d], split into Q, K, V parts
    for key, val in state.items():
        if 'in_proj_weight' in key and val.dim() == 2:
            block_id = _get_block_id(key)
            d = val.shape[0] // 3
            W_Q = val[:d]        # [d, d]
            W_K = val[d:2*d]     # [d, d]
            W_V = val[2*d:]      # [d, d]

            # Project each token's embedding: [97, d] @ [d, d]^T = [97, d]
            reps[f'Q_block{block_id}'] = (embed @ W_Q.T).clone()
            reps[f'K_block{block_id}'] = (embed @ W_K.T).clone()
            reps[f'V_block{block_id}'] = (embed @ W_V.T).clone()

        # Attention output projection
        if 'out_proj.weight' in key and 'self_attn' in key:
            block_id = _get_block_id(key)
            W_O = val  # [d, d]
            reps[f'attn_O_block{block_id}'] = (embed @ W_O.T).clone()

        # FFN up-projection: [4d, d] — match exactly "ffn.0.weight"
        if 'ffn.0.weight' in key and val.dim() == 2:
            block_id = _get_block_id(key)
            W_up = val  # [4d, d]
            reps[f'FFN_up_block{block_id}'] = (embed @ W_up.T).clone()

    return reps


def _get_block_id(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p in ('blocks', 'model'):
            for j in range(i+1, len(parts)):
                try:
                    return int(parts[j])
                except ValueError:
                    continue
    return 0


# ============================================================
# Visualization
# ============================================================

def plot_neighborhood_comparison(all_layer_results, model_label, out_dir):
    """Bar chart: neighborhood similarity across layers."""
    layers = [r["layer"] for r in all_layer_results]
    distances = [1, 2, 5, 10, 48]

    fig, ax = plt.subplots(figsize=(max(10, len(layers)*1.2), 5))
    x = np.arange(len(layers))
    width = 0.15

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    for i, dist in enumerate(distances):
        vals = [r["neighborhood"][dist]["mean"] for r in all_layer_results]
        ax.bar(x + i*width, vals, width*0.9, label=f'd={dist}', color=colors[i], alpha=0.8)

    ax.set_xticks(x + width*2)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(f"Neighborhood Structure Across Layers: {model_label}", fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fname = f"neighborhood_all_layers_{model_label.replace(' ', '_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_inverse_signal(all_layer_results, model_label, out_dir):
    """Bar chart: additive inverse signal strength across layers."""
    layers = [r["layer"] for r in all_layer_results]
    signals = [r["additive_inverse"]["signal_stddevs"] for r in all_layer_results]
    inv_means = [r["additive_inverse"]["mean"] for r in all_layer_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: signal in std devs
    ax = axes[0]
    colors = ['#e74c3c' if abs(s) > 1 else '#95a5a6' for s in signals]
    ax.barh(range(len(layers)), signals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_xlabel("Signal (std devs from mean)")
    ax.set_title("Additive Inverse Signal Strength", fontsize=11, fontweight='bold')
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.axvline(1, color='red', linestyle='--', alpha=0.3, label='1σ')
    ax.axvline(-1, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # Right: raw mean similarity
    ax = axes[1]
    ax.barh(range(len(layers)), inv_means, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_xlabel("Mean cos_sim (inverse pairs)")
    ax.set_title("Additive Inverse Mean Similarity", fontsize=11, fontweight='bold')
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle(model_label, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fname = f"inverse_signal_{model_label.replace(' ', '_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    models = [
        ("dense_baseline", "Dense_wd1"),
        ("sparse_L0_0.40", "Sparse_L0=0.40_wd1"),
    ]

    for ckpt_name, label in models:
        ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"SKIP: {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {label}")
        print(f"{'='*70}")

        reps = extract_token_representations(ckpt_path)
        print(f"  Extracted {len(reps)} layer representations:")
        for name, V in reps.items():
            print(f"    {name}: {list(V.shape)}")

        all_results = []
        for layer_name, V in reps.items():
            results, G = analyze_token_vectors(V, layer_name)
            all_results.append(results)

        # Print summary table
        print(f"\n  {'Layer':<25} {'d1_sim':>7} {'d48_sim':>8} {'inv_sig':>8} {'inv_mean':>9} {'QR_diff':>8}")
        print("  " + "-" * 70)
        for r in all_results:
            d1 = r["neighborhood"][1]["mean"]
            d48 = r["neighborhood"][48]["mean"]
            inv_sig = r["additive_inverse"]["signal_stddevs"]
            inv_mean = r["additive_inverse"]["mean"]
            qr_diff = r["quadratic_residue"]["within_qr"] - r["quadratic_residue"]["cross"]
            print(f"  {r['layer']:<25} {d1:>7.4f} {d48:>8.4f} {inv_sig:>8.2f}σ {inv_mean:>9.4f} {qr_diff:>8.4f}")

        # Generate plots
        plot_neighborhood_comparison(all_results, label, OUT_DIR)
        plot_inverse_signal(all_results, label, OUT_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
