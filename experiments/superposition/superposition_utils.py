"""
Geometry analysis utilities for studying superposition in weight matrices.

Implements metrics from "Superposition Yields Robust Neural Scaling":
- Row norm distributions
- Pairwise cosine overlaps (mean, variance, max)
- Welch bound comparison
- ETF-likeness measures
- Uniformity loss (Wang & Isola 2020)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_row_norms(W: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm of each row of W. Returns shape [n_rows]."""
    return torch.norm(W, dim=1)


def compute_cosine_gram(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Gram matrix of cosine similarities between all row pairs.

    Args:
        W: weight matrix [n, m]
    Returns:
        G: cosine similarity matrix [n, n], G[i,j] = cos(W_i, W_j)
    """
    W_norm = F.normalize(W, dim=1, eps=eps)
    return W_norm @ W_norm.T


def compute_overlap_stats(W: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """
    Compute pairwise overlap statistics for rows of W.

    Returns dict with:
        mean_sq_overlap: E[(cos_sim)^2] for i != j
        var_sq_overlap:  Var[(cos_sim)^2] for i != j
        mean_abs_overlap: E[|cos_sim|] for i != j
        max_abs_overlap: max |cos_sim| for i != j
        welch_bound: theoretical lower bound on max |cos_sim|
        random_var: theoretical variance for random unit vectors = 2(m-1)/(m^2(m+2))
        n_rows: number of rows
        n_cols: number of columns (dimension m)
    """
    n, m = W.shape
    G = compute_cosine_gram(W, eps=eps)

    # Extract off-diagonal elements
    mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
    off_diag = G[mask]

    abs_overlaps = off_diag.abs()
    sq_overlaps = off_diag.pow(2)

    # Welch bound: max|cos_sim| >= sqrt((n-m) / (m*(n-1))) when n > m
    if n > m:
        welch = math.sqrt((n - m) / (m * (n - 1)))
    else:
        welch = 0.0  # not in superposition regime

    # Theoretical variance for random unit vectors on S^{m-1}
    random_var = 2 * (m - 1) / (m * m * (m + 2))

    return {
        'mean_sq_overlap': sq_overlaps.mean().item(),
        'var_sq_overlap': sq_overlaps.var().item(),
        'mean_abs_overlap': abs_overlaps.mean().item(),
        'max_abs_overlap': abs_overlaps.max().item(),
        'welch_bound': welch,
        'one_over_m': 1.0 / m,
        'random_var': random_var,
        'n_rows': n,
        'n_cols': m,
    }


def compute_norm_stats(W: torch.Tensor) -> Dict[str, float]:
    """
    Compute row norm statistics.

    Returns dict with:
        mean_norm, std_norm, min_norm, max_norm
        frac_above_half: fraction of rows with ||W_i|| > 0.5 (phi_{1/2})
        frac_above_one:  fraction of rows with ||W_i|| > 1.0 (phi_1, strongly represented)
    """
    norms = compute_row_norms(W)
    n = norms.shape[0]

    return {
        'mean_norm': norms.mean().item(),
        'std_norm': norms.std().item(),
        'min_norm': norms.min().item(),
        'max_norm': norms.max().item(),
        'median_norm': norms.median().item(),
        'frac_above_half': (norms > 0.5).float().mean().item(),
        'frac_above_one': (norms > 1.0).float().mean().item(),
        'n_rows': n,
    }


def compute_full_geometry(W: torch.Tensor, name: str = "") -> Dict:
    """
    Run all geometry analyses on a weight matrix.

    Returns a dict combining norm stats, overlap stats, and metadata.
    """
    result = {'name': name, 'shape': list(W.shape)}
    result['norm'] = compute_norm_stats(W)
    result['overlap'] = compute_overlap_stats(W)
    return result


def uniformity_loss(W: torch.Tensor, t: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Uniformity loss from Wang & Isola (2020).

    L = log E_{i!=j} [exp(-t * ||W_i_hat - W_j_hat||^2)]

    where W_i_hat = W_i / ||W_i||.

    Minimizing this encourages normalized rows to be uniformly distributed
    on the unit hypersphere. The exp(-t*d^2) kernel ensures that ANY close
    pair dominates the loss (soft-min over distances), preventing clustering.

    Args:
        W: weight matrix [n, m]
        t: temperature (higher = more sensitive to close pairs)
    Returns:
        scalar loss
    """
    W_norm = F.normalize(W, dim=1, eps=eps)
    # ||a - b||^2 = 2 - 2*(a . b) for unit vectors
    cosine_sim = W_norm @ W_norm.T
    sq_dist = 2.0 - 2.0 * cosine_sim

    n = W.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
    # logsumexp over off-diagonal pairs, minus log(count) for mean
    return torch.logsumexp(-t * sq_dist[mask], dim=0) - math.log(n * (n - 1))


def format_geometry_report(stats: Dict, indent: str = "") -> str:
    """Format a geometry analysis dict into a readable text report."""
    lines = []
    name = stats.get('name', 'unknown')
    shape = stats.get('shape', [])
    lines.append(f"{indent}=== {name} {shape} ===")

    ns = stats['norm']
    lines.append(f"{indent}  Row norms: mean={ns['mean_norm']:.4f}  std={ns['std_norm']:.4f}  "
                 f"range=[{ns['min_norm']:.4f}, {ns['max_norm']:.4f}]")
    lines.append(f"{indent}  Fraction ||W_i||>0.5: {ns['frac_above_half']:.3f}  "
                 f"||W_i||>1.0: {ns['frac_above_one']:.3f}")

    os = stats['overlap']
    lines.append(f"{indent}  Mean squared overlap:  {os['mean_sq_overlap']:.6f}  "
                 f"(1/m = {os['one_over_m']:.6f})")
    lines.append(f"{indent}  Var of squared overlap: {os['var_sq_overlap']:.6f}  "
                 f"(random = {os['random_var']:.6f})")
    lines.append(f"{indent}  Max |overlap|: {os['max_abs_overlap']:.4f}  "
                 f"(Welch bound = {os['welch_bound']:.4f})")

    # Ratio diagnostics
    if os['one_over_m'] > 0:
        ratio = os['mean_sq_overlap'] / os['one_over_m']
        lines.append(f"{indent}  Overlap / (1/m) ratio: {ratio:.3f}  "
                     f"(=1.0 means matches paper prediction)")
    if os['random_var'] > 0:
        var_ratio = os['var_sq_overlap'] / os['random_var']
        lines.append(f"{indent}  Var / random_var ratio: {var_ratio:.3f}  "
                     f"(<1 means more ETF-like than random)")

    return '\n'.join(lines)


def extract_weight_matrices(model, model_type: str = "sparse") -> Dict[str, torch.Tensor]:
    """
    Extract key weight matrices from a model for geometry analysis.

    Args:
        model: a Transformer or SparseTransformer instance
        model_type: "sparse" or "dense"
    Returns:
        dict mapping descriptive names to weight tensors
    """
    matrices = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases and norm weights

        # Token embedding
        if 'token_embeddings' in name:
            matrices['token_embed'] = param.data.clone()

        # Output projection (most important for superposition analysis)
        if 'output_proj' in name and 'weight' in name:
            matrices['output_proj'] = param.data.clone()

        # For dense model: final linear is inside model.model Sequential
        if model_type == "dense" and name.endswith('.weight'):
            # The last Linear in nn.Sequential
            parts = name.split('.')
            # model.model.{last_idx}.weight
            if 'model' in parts:
                try:
                    idx = int(parts[-2])
                    # Check if this is the final linear layer
                    if hasattr(model, 'model') and idx == len(model.model) - 1:
                        matrices['output_proj'] = param.data.clone()
                except (ValueError, IndexError):
                    pass

        # FFN up-projection (512 x 128, best superposition candidate)
        if 'ffn.0.weight' in name or ('ffn' in name and '0' in name and 'weight' in name):
            block_id = _extract_block_id(name)
            matrices[f'ffn_up_block{block_id}'] = param.data.clone()

        # FFN down-projection
        if 'ffn.2.weight' in name or ('ffn' in name and '2' in name and 'weight' in name):
            block_id = _extract_block_id(name)
            matrices[f'ffn_down_block{block_id}'] = param.data.clone()

        # Attention in_proj (packed Q/K/V)
        if 'in_proj_weight' in name:
            block_id = _extract_block_id(name)
            full = param.data.clone()
            d = full.shape[0] // 3
            matrices[f'attn_Q_block{block_id}'] = full[:d]
            matrices[f'attn_K_block{block_id}'] = full[d:2*d]
            matrices[f'attn_V_block{block_id}'] = full[2*d:]

        # Attention out_proj
        if 'out_proj.weight' in name and 'self_attn' in name:
            block_id = _extract_block_id(name)
            matrices[f'attn_O_block{block_id}'] = param.data.clone()

    return matrices


def _extract_block_id(name: str) -> int:
    """Extract block index from parameter name like 'blocks.0.ffn...'"""
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p in ('blocks', 'model'):
            # next numeric part is the block id
            for j in range(i+1, len(parts)):
                try:
                    return int(parts[j])
                except ValueError:
                    continue
    return 0
