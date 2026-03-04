"""
Analyze the algebraic structure encoded in output_proj representation geometry.

For the task x+y (mod 97), the tokens 0..96 form the cyclic group Z/97Z.
This script examines whether the non-uniform geometry of representation vectors
encodes algebraic relationships such as:
  - Additive inverses: (a, 97-a)
  - Cyclic group structure: tokens that are generators vs non-generators
  - Modular arithmetic neighborhoods

Usage:
    python experiments/superposition/analyze_algebra.py

Output: text report to stdout + plots to results/geometry/
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
from matplotlib.colors import TwoSlopeNorm

from superposition_utils import extract_weight_matrices, compute_cosine_gram, compute_row_norms

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry"
P = 97  # prime modulus


# ============================================================
# Algebraic utilities for Z/97Z
# ============================================================

def additive_inverse(a, p=P):
    """Return (p - a) % p."""
    return (p - a) % p


def is_quadratic_residue(a, p=P):
    """Check if a is a quadratic residue mod p (Euler's criterion)."""
    if a % p == 0:
        return True
    return pow(int(a), (p - 1) // 2, p) == 1


def discrete_log(a, g=5, p=P):
    """Compute discrete log base g of a in Z/pZ. Returns None if a=0."""
    if a == 0:
        return None
    val = 1
    for i in range(p - 1):
        if val == a:
            return i
        val = (val * g) % p
    return None


def get_algebraic_relations(p=P):
    """Build a dict of algebraic relationships between tokens."""
    relations = {}

    # Additive inverse pairs: (a, p-a)
    inv_pairs = []
    for a in range(1, (p + 1) // 2):
        inv_pairs.append((a, p - a))
    relations['additive_inverse_pairs'] = inv_pairs

    # Quadratic residues vs non-residues
    qr = [a for a in range(1, p) if is_quadratic_residue(a, p)]
    qnr = [a for a in range(1, p) if not is_quadratic_residue(a, p)]
    relations['quadratic_residues'] = qr
    relations['quadratic_non_residues'] = qnr

    # Discrete log classes (mod small numbers)
    # Tokens with same discrete_log mod k form a coset
    g = 5  # primitive root of 97
    dlogs = {}
    for a in range(1, p):
        dl = discrete_log(a, g, p)
        if dl is not None:
            dlogs[a] = dl
    relations['discrete_logs'] = dlogs

    return relations


# ============================================================
# Analysis functions
# ============================================================

def load_output_proj(ckpt_path):
    """Load output_proj weight matrix from checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    for key, val in state.items():
        if 'output_proj' in key and 'weight' in key and val.dim() == 2:
            return val
        # Dense model: last linear layer
        if val.dim() == 2 and val.shape[0] == P + 2:  # 99 = 97 + 2 special tokens
            return val

    return None


def analyze_cosine_structure(W, label, relations):
    """Analyze the cosine similarity matrix for algebraic patterns."""
    # Only use token rows 0..96 (skip op and eq tokens at indices 97, 98)
    W_tokens = W[:P]  # [97, 128]
    G = compute_cosine_gram(W_tokens)  # [97, 97]
    G_np = G.numpy()

    # Zero out diagonal for off-diagonal analysis
    np.fill_diagonal(G_np, 0)

    print(f"\n{'='*70}")
    print(f"ALGEBRAIC ANALYSIS: {label}")
    print(f"{'='*70}")

    # 1. Additive inverse analysis
    print(f"\n--- Additive Inverses: cos_sim(a, {P}-a) ---")
    inv_sims = []
    for a, b in relations['additive_inverse_pairs']:
        sim = G_np[a, b]
        inv_sims.append(sim)
    inv_sims = np.array(inv_sims)

    # Random baseline: mean of all off-diagonal entries
    mask = np.ones_like(G_np, dtype=bool)
    np.fill_diagonal(mask, False)
    all_sims = G_np[mask]

    print(f"  Inverse pairs:   mean={inv_sims.mean():.4f}  std={inv_sims.std():.4f}  "
          f"range=[{inv_sims.min():.4f}, {inv_sims.max():.4f}]")
    print(f"  All pairs:       mean={all_sims.mean():.4f}  std={all_sims.std():.4f}  "
          f"range=[{all_sims.min():.4f}, {all_sims.max():.4f}]")
    print(f"  Inverse vs all:  {inv_sims.mean()/all_sims.std():.2f} std devs from mean")

    # 2. Top-K most similar pairs
    print(f"\n--- Top 20 Most Similar Token Pairs ---")
    # Get upper triangle indices
    triu_i, triu_j = np.triu_indices(P, k=1)
    sims = G_np[triu_i, triu_j]
    top_idx = np.argsort(sims)[::-1][:20]

    print(f"  {'Rank':>4} {'Token_i':>7} {'Token_j':>7} {'cos_sim':>8} {'i+j mod 97':>10} {'Relation':>15}")
    for rank, idx in enumerate(top_idx):
        i, j = triu_i[idx], triu_j[idx]
        s = sims[idx]
        sum_mod = (i + j) % P
        # Check if they are additive inverses
        rel = ""
        if sum_mod == 0 and i != 0 and j != 0:
            rel = "ADD_INVERSE"
        elif abs(i - j) == 1 or abs(i - j) == P - 1:
            rel = "NEIGHBOR"
        elif i != 0 and j != 0 and (i * j) % P == 1:
            rel = "MULT_INVERSE"
        # Check if same QR class
        if i > 0 and j > 0:
            qr_i = is_quadratic_residue(i)
            qr_j = is_quadratic_residue(j)
            if qr_i == qr_j and not rel:
                rel = "SAME_QR"
            elif qr_i != qr_j and not rel:
                rel = "DIFF_QR"
        print(f"  {rank+1:>4} {i:>7} {j:>7} {s:>8.4f} {sum_mod:>10} {rel:>15}")

    # 3. Top-K most dissimilar pairs
    print(f"\n--- Top 20 Most Dissimilar Token Pairs ---")
    bot_idx = np.argsort(sims)[:20]
    for rank, idx in enumerate(bot_idx):
        i, j = triu_i[idx], triu_j[idx]
        s = sims[idx]
        sum_mod = (i + j) % P
        rel = ""
        if sum_mod == 0 and i != 0 and j != 0:
            rel = "ADD_INVERSE"
        print(f"  {rank+1:>4} {i:>7} {j:>7} {s:>8.4f} {sum_mod:>10} {rel:>15}")

    # 4. Quadratic residue class analysis
    print(f"\n--- Quadratic Residue Structure ---")
    qr = relations['quadratic_residues']
    qnr = relations['quadratic_non_residues']

    # Mean similarity within QR, within QNR, and between QR-QNR
    qr_sims = [G_np[i, j] for i in qr for j in qr if i < j]
    qnr_sims = [G_np[i, j] for i in qnr for j in qnr if i < j]
    cross_sims = [G_np[i, j] for i in qr for j in qnr]

    print(f"  Within QR  ({len(qr):>2} tokens): mean_sim={np.mean(qr_sims):.4f}  std={np.std(qr_sims):.4f}")
    print(f"  Within QNR ({len(qnr):>2} tokens): mean_sim={np.mean(qnr_sims):.4f}  std={np.std(qnr_sims):.4f}")
    print(f"  Cross QR-QNR:                mean_sim={np.mean(cross_sims):.4f}  std={np.std(cross_sims):.4f}")

    # 5. Discrete log structure: group tokens by dlog mod 4
    print(f"\n--- Discrete Log Structure (mod 4, generator g=5) ---")
    dlogs = relations['discrete_logs']
    for r in range(4):
        group = [a for a in range(1, P) if dlogs.get(a, -1) % 4 == r]
        if len(group) > 1:
            group_sims = [G_np[i, j] for i in group for j in group if i < j]
            print(f"  dlog mod 4 = {r} ({len(group):>2} tokens): mean_sim={np.mean(group_sims):.4f}")

    # Inter-group
    groups_4 = {r: [a for a in range(1, P) if dlogs.get(a, -1) % 4 == r] for r in range(4)}
    for r1 in range(4):
        for r2 in range(r1+1, 4):
            cross = [G_np[i, j] for i in groups_4[r1] for j in groups_4[r2]]
            if cross:
                print(f"  dlog mod 4: {r1} vs {r2}: mean_sim={np.mean(cross):.4f}")

    # 6. Neighborhood analysis: are numerically close tokens more similar?
    print(f"\n--- Numerical Neighborhood ---")
    for dist in [1, 2, 5, 10, 24, 48]:
        pairs = [(a, (a + dist) % P) for a in range(P)]
        pair_sims = [G_np[a, b] for a, b in pairs if a != b]
        print(f"  Distance {dist:>2}: mean_sim={np.mean(pair_sims):.4f}  std={np.std(pair_sims):.4f}")

    # 7. Token 0 analysis (identity element)
    print(f"\n--- Token 0 (Identity Element) ---")
    zero_sims = G_np[0, 1:P]
    print(f"  cos_sim(0, others): mean={zero_sims.mean():.4f}  std={zero_sims.std():.4f}  "
          f"range=[{zero_sims.min():.4f}, {zero_sims.max():.4f}]")

    # Row norm of token 0 vs others
    norms = compute_row_norms(W_tokens).numpy()
    print(f"  ||W_0|| = {norms[0]:.4f},  mean(||W_others||) = {norms[1:].mean():.4f}")

    return G_np, norms


def plot_similarity_matrix(G_np, label, out_dir):
    """Plot the full token-token cosine similarity matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(abs(G_np.min()), abs(G_np.max()))
    im = ax.imshow(G_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f"Token Cosine Similarity: {label}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Token ID")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cos(W_i, W_j)")
    fig.tight_layout()
    fname = f"token_similarity_{label.replace(' ', '_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_similarity_vs_algebraic(G_np, relations, label, out_dir):
    """Plot cosine similarity as a function of (i+j) mod P."""
    triu_i, triu_j = np.triu_indices(P, k=1)
    sims = G_np[triu_i, triu_j]
    sums = (triu_i + triu_j) % P

    # Bin by sum mod P
    bins = {}
    for s, sim in zip(sums, sims):
        bins.setdefault(s, []).append(sim)

    x = sorted(bins.keys())
    means = [np.mean(bins[k]) for k in x]
    stds = [np.std(bins[k]) for k in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean similarity vs (i+j) mod P
    ax = axes[0]
    ax.bar(x, means, width=1.0, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(sims), color='red', linestyle='--', alpha=0.5, label='overall mean')
    # Highlight sum=0 (additive inverses)
    if 0 in bins:
        ax.bar(0, np.mean(bins[0]), width=1.0, color='red', alpha=0.8, label=f'sum≡0 (inverses)')
    ax.set_xlabel("(i + j) mod 97", fontsize=10)
    ax.set_ylabel("Mean cos similarity", fontsize=10)
    ax.set_title(f"Similarity vs Algebraic Sum: {label}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: histogram of all similarities, split by inverse vs non-inverse
    ax = axes[1]
    inv_mask = sums == 0
    ax.hist(sims[~inv_mask], bins=50, alpha=0.6, color='steelblue', label='Non-inverse pairs', density=True)
    ax.hist(sims[inv_mask], bins=20, alpha=0.8, color='red', label='Additive inverse pairs', density=True)
    ax.set_xlabel("cos similarity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Similarity Distribution: {label}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"algebraic_structure_{label.replace(' ', '_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_dlog_reordered(G_np, relations, label, out_dir):
    """Reorder similarity matrix by discrete log to reveal cyclic group structure."""
    dlogs = relations['discrete_logs']
    # Order tokens 1..96 by their discrete log, put 0 at the end
    ordered = sorted(range(1, P), key=lambda a: dlogs.get(a, P))
    ordered = ordered + [0]

    G_reordered = G_np[np.ix_(ordered, ordered)]

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(abs(G_reordered.min()), abs(G_reordered.max()))
    im = ax.imshow(G_reordered, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f"Similarity (reordered by discrete log): {label}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Token (sorted by dlog)")
    ax.set_ylabel("Token (sorted by dlog)")

    # Mark quadrants: first half = even dlog (QR), second half = odd dlog (QNR)
    half = (P - 1) // 2
    ax.axhline(half, color='white', linewidth=0.5, alpha=0.5)
    ax.axvline(half, color='white', linewidth=0.5, alpha=0.5)

    fig.colorbar(im, ax=ax, shrink=0.8, label="cos(W_i, W_j)")
    fig.tight_layout()
    fname = f"dlog_reordered_{label.replace(' ', '_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    relations = get_algebraic_relations()

    # Analyze key models
    models = [
        ("dense_baseline", "Dense wd=1"),
        ("sparse_L0_0.40", "Sparse L0=0.40 wd=1"),
        ("sparse_L0_0.10", "Sparse L0=0.10 wd=1"),
        ("dense_baseline_wd0.3", "Dense wd=0.3"),
        ("sparse_L0_0.60_wd0.3", "Sparse L0=0.60 wd=0.3"),
    ]

    all_results = {}

    for ckpt_name, label in models:
        ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: {ckpt_path} not found")
            continue

        W = load_output_proj(ckpt_path)
        if W is None:
            print(f"  SKIP: no output_proj in {ckpt_name}")
            continue

        G_np, norms = analyze_cosine_structure(W, label, relations)
        all_results[label] = (G_np, norms)

        # Generate plots
        plot_similarity_matrix(G_np, label, OUT_DIR)
        plot_similarity_vs_algebraic(G_np, relations, label, OUT_DIR)
        plot_dlog_reordered(G_np, relations, label, OUT_DIR)

    # Cross-model comparison summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON: Additive Inverse Similarity")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Mean inv_sim':>12} {'Mean all_sim':>12} {'Ratio':>8}")
        print("-" * 65)
        for label, (G_np, _) in all_results.items():
            inv_pairs = relations['additive_inverse_pairs']
            inv_sims = np.array([G_np[a, b] for a, b in inv_pairs])
            mask = np.ones_like(G_np, dtype=bool)
            np.fill_diagonal(mask, False)
            all_sims = G_np[mask]
            ratio = inv_sims.mean() / (all_sims.std() + 1e-8)
            print(f"{label:<30} {inv_sims.mean():>12.4f} {all_sims.mean():>12.4f} {ratio:>8.2f}")


if __name__ == '__main__':
    main()
