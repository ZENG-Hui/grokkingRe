"""
Phase 1: Analyze the geometric structure of weight matrices in trained models.

Measures row norms, pairwise cosine overlaps, and ETF-likeness for
dense and sparse models at different L0 levels.

Usage:
    # Analyze a single checkpoint
    python experiments/superposition/analyze_geometry.py --checkpoint checkpoints/final_model.pt --model-type sparse

    # Analyze all models from Phase 1 sweep (after train_sweep.py)
    python experiments/superposition/analyze_geometry.py --sweep-dir experiments/superposition/results/checkpoints

Output:
    - Text report to stdout (machine-readable)
    - Plots to experiments/superposition/results/geometry/
    - JSON summary to experiments/superposition/results/geometry/geometry_summary.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np

# Add project root and script dir to path
ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from superposition_utils import (
    compute_full_geometry,
    format_geometry_report,
    extract_weight_matrices,
    compute_row_norms,
    compute_cosine_gram,
)

# Optional: plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_model_from_checkpoint(ckpt_path: str, model_type: str = "sparse"):
    """Load a model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if model_type == "sparse":
        from config_sparse import SparseTrainingConfig
        from model_sparse import create_sparse_model

        if 'config' in ckpt and isinstance(ckpt['config'], SparseTrainingConfig):
            config = ckpt['config']
        else:
            config = SparseTrainingConfig()

        model = create_sparse_model(config)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)

    elif model_type == "dense":
        from model import Transformer

        config = ckpt.get('config', None)
        if config and hasattr(config, 'num_layers'):
            model = Transformer(
                num_layers=config.num_layers,
                dim_model=config.dim_model,
                num_heads=config.num_heads,
                num_tokens=getattr(config, 'num_tokens', 99),
                seq_len=getattr(config, 'seq_len', 5),
                norm_type=getattr(config, 'norm_type', 'layernorm'),
            )
        else:
            model = Transformer(num_layers=2, dim_model=128, num_heads=4,
                                num_tokens=99, seq_len=5)

        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    return model


def analyze_single_model(model, model_type: str, label: str) -> dict:
    """Run full geometry analysis on one model. Returns results dict."""
    print(f"\n{'='*70}")
    print(f"MODEL: {label}")
    print(f"{'='*70}")

    matrices = extract_weight_matrices(model, model_type)

    if not matrices:
        print("  WARNING: No weight matrices extracted!")
        return {}

    print(f"  Extracted {len(matrices)} weight matrices:")
    for name, W in matrices.items():
        print(f"    {name}: {list(W.shape)}")

    results = {}
    for name, W in matrices.items():
        # Skip very small matrices
        if W.shape[0] < 3 or W.shape[1] < 3:
            continue

        stats = compute_full_geometry(W, name=name)
        results[name] = stats
        print()
        print(format_geometry_report(stats, indent="  "))

    # Summary comparison against paper predictions
    print(f"\n  --- Summary for {label} ---")
    for name, stats in results.items():
        n, m = stats['shape']
        ovs = stats['overlap']
        ns = stats['norm']
        ratio = ovs['mean_sq_overlap'] / ovs['one_over_m'] if ovs['one_over_m'] > 0 else float('inf')
        in_superposition = n > m

        print(f"  {name:30s}  n/m={n}/{m}={'SP' if in_superposition else 'no'}  "
              f"<cos^2>={ovs['mean_sq_overlap']:.5f}  "
              f"1/m={ovs['one_over_m']:.5f}  "
              f"ratio={ratio:.2f}  "
              f"max|cos|={ovs['max_abs_overlap']:.3f}  "
              f"welch={ovs['welch_bound']:.3f}")

    return results


def plot_row_norms(all_results: dict, output_dir: str):
    """Plot row norm distributions for a specific layer across models."""
    if not HAS_MPL:
        print("  [plot skipped: matplotlib not available]")
        return

    # Find a layer that exists in all models (prefer ffn_up_block0)
    target_layers = ['ffn_up_block0', 'output_proj', 'token_embed']
    for target in target_layers:
        if all(target in res for res in all_results.values()):
            break
    else:
        print("  [plot skipped: no common layer across models]")
        return

    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 3.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[idx]
        stats = results[target]
        ns = stats['norm']

        # Recreate norm values from stats (we only have summary stats)
        # For proper histogram we need the raw norms — recompute from model if available
        ax.set_title(label, fontsize=9)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5 threshold')
        ax.axvline(1.0, color='blue', linestyle='--', alpha=0.5, label='1.0 threshold')

        # Text annotation with stats
        ax.text(0.95, 0.95,
                f"mean={ns['mean_norm']:.2f}\nstd={ns['std_norm']:.2f}\n"
                f">0.5: {ns['frac_above_half']:.0%}\n>1.0: {ns['frac_above_one']:.0%}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('||W_i||')
        if idx == 0:
            ax.set_ylabel('Count')

    fig.suptitle(f'Row Norm Stats: {target}', fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, f'row_norm_stats_{target}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overlap_scaling(all_results: dict, output_dir: str):
    """Plot mean squared overlap vs 1/m across models and layers."""
    if not HAS_MPL:
        print("  [plot skipped: matplotlib not available]")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (label, results) in enumerate(all_results.items()):
        for lname, stats in results.items():
            n, m = stats['shape']
            ovs = stats['overlap']
            ax.scatter(ovs['one_over_m'], ovs['mean_sq_overlap'],
                       color=colors[idx], marker=markers[idx % len(markers)],
                       s=60, alpha=0.8, edgecolors='k', linewidths=0.5)

    # Reference line: y = x (paper prediction)
    lims = ax.get_xlim()
    x_ref = np.linspace(0, max(lims[1], 0.02), 100)
    ax.plot(x_ref, x_ref, 'k--', alpha=0.5, label='y = 1/m (paper prediction)')

    # Legend for models
    for idx, label in enumerate(all_results.keys()):
        ax.scatter([], [], color=colors[idx], marker=markers[idx % len(markers)],
                   s=60, label=label, edgecolors='k', linewidths=0.5)

    ax.set_xlabel('1/m (inverse dimension)')
    ax.set_ylabel('Mean squared overlap E[cos^2]')
    ax.set_title('Overlap Scaling: Does E[cos^2] ≈ 1/m ?')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(output_dir, 'overlap_scaling.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze weight matrix geometry')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint')
    parser.add_argument('--model-type', type=str, default='sparse',
                        choices=['sparse', 'dense'])
    parser.add_argument('--sweep-dir', type=str, default=None,
                        help='Directory containing multiple checkpoints (label_type.pt)')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/superposition/results/geometry')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    if args.checkpoint:
        # Single model analysis
        label = Path(args.checkpoint).stem
        model = load_model_from_checkpoint(args.checkpoint, args.model_type)
        all_results[label] = analyze_single_model(model, args.model_type, label)

    elif args.sweep_dir:
        # Analyze all checkpoints in directory
        sweep = Path(args.sweep_dir)
        for ckpt_file in sorted(sweep.glob('*.pt')):
            fname = ckpt_file.stem
            # Infer model type from filename
            mtype = 'dense' if 'dense' in fname else 'sparse'
            label = fname

            print(f"\nLoading {ckpt_file}...")
            try:
                model = load_model_from_checkpoint(str(ckpt_file), mtype)
                all_results[label] = analyze_single_model(model, mtype, label)
            except Exception as e:
                print(f"  ERROR loading {ckpt_file}: {e}")
                continue
    else:
        # Default: analyze the existing checkpoint
        default_ckpt = ROOT / 'checkpoints' / 'final_model.pt'
        if default_ckpt.exists():
            model = load_model_from_checkpoint(str(default_ckpt), 'sparse')
            all_results['existing_sparse'] = analyze_single_model(model, 'sparse', 'existing_sparse')
        else:
            print("No checkpoint specified and no default found.")
            print("Usage: python analyze_geometry.py --checkpoint <path> --model-type <sparse|dense>")
            return

    # Save JSON summary
    json_path = os.path.join(args.output_dir, 'geometry_summary.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON summary: {json_path}")

    # Generate plots
    if all_results:
        print("\nGenerating plots...")
        plot_overlap_scaling(all_results, args.output_dir)
        plot_row_norms(all_results, args.output_dir)

    # Final text summary table
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Layer':<25} {'n':>4} {'m':>4} {'E[cos^2]':>10} {'1/m':>10} {'ratio':>7} {'max|cos|':>9} {'welch':>7}")
    print("-" * 110)

    for label, results in all_results.items():
        for lname, stats in results.items():
            n, m = stats['shape']
            ovs = stats['overlap']
            ratio = ovs['mean_sq_overlap'] / ovs['one_over_m'] if ovs['one_over_m'] > 0 else 0
            print(f"{label:<25} {lname:<25} {n:>4} {m:>4} {ovs['mean_sq_overlap']:>10.6f} "
                  f"{ovs['one_over_m']:>10.6f} {ratio:>7.2f} {ovs['max_abs_overlap']:>9.4f} "
                  f"{ovs['welch_bound']:>7.4f}")


if __name__ == '__main__':
    main()
