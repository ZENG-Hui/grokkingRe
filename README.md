# Grokking in Sparse Transformers

Studying grokking phenomena in weight-sparse Transformers on modular arithmetic tasks. Based on the paper *"Automatically Identifying Local and Global Circuits with Linear Computation Graphs"*.

The core idea: force most weights to zero during training via Top-K sparsification, so the model learns to solve the task with a minimal "circuit" — a small subset of connections that reveals interpretable computation structure.

## Quick Start

```bash
# Dense baseline
python cli.py --operation x+y --num_steps 4000

# Sparse training (edit config at top of file, then run)
python training_sparse.py

# Dense vs Sparse comparison
python run_dense_vs_sparse.py --sequential

# L0 parameter sweep
python scan_L0.py
```

## Project Structure

```
Core Code
├── data.py                  Data generation: x op y (mod p)
├── model.py                 Dense Transformer (LayerNorm/RMSNorm)
├── model_sparse.py          Sparse Transformer (RMSNorm + AbsTopK)
├── training.py              Dense training loop
├── training_sparse.py       Sparse training loop (main entry)
├── config_sparse.py         Sparse training configs (tiny/small/medium/large)
├── sparse_utils.py          Top-K sparsification, L0 annealing, param stats
├── lr_scheduler.py          Unified LR scheduler (warmup + cosine)
├── cli.py                   CLI entry for dense training

Experiments
├── run_dense_vs_sparse.py   Automated dense vs sparse comparison
├── scan_L0.py               Sweep final_L0 across values

Visualization
├── visualize_sparsity.py    Weight sparsity heatmaps & statistics
├── visualize_activations.py Activation pattern analysis
├── visualizations/          Generated plots

Other
├── reference_implementations/   Original pre-unification code backup
├── Markdown/                    Archived experiment notes & guides
├── scan/                        L0 sweep results (JSON + plots)
├── checkpoints/                 Saved model weights
└── wandb/                       Experiment tracking logs
```

## How It Works

### Task

Learn `x op y (mod p)` where `op` is `+`, `-`, or `/`, and `p = 97` (prime). Input: 4 tokens `[x, op, y, eq]`, output: the result class. Total ~9.4K samples, 50/50 train/val split.

### Two Model Variants

| | Dense (`model.py`) | Sparse (`model_sparse.py`) |
|---|---|---|
| Norm | LayerNorm or RMSNorm | RMSNorm (preserves zero semantics) |
| Activation | GELU only | GELU + AbsTopK (keep top 25%) |
| Weights | All active | Top-K enforced per step |

Both use the same architecture: `Embedding → N x DecoderBlock → Norm → Linear`. Default: 2 layers, dim=128, 4 heads (~422K params).

### Sparsification Pipeline

Each training step:

```
Forward → Loss → Backward → Grad Clip (RMS) → Optimizer Step → Top-K Sparsify → Log
```

The critical part is **Top-K sparsification after the optimizer step**: for each weight matrix, only the largest `L0 * total` entries (by absolute value) survive; the rest are zeroed out. This is a hard constraint, not a gradient-based regularization.

### L0 Annealing

L0 (fraction of nonzero weights) decreases linearly during training:

```
Step 0          50%         100%
L0:  1.0 -----> final_L0 ----> final_L0
     (dense)    (anneal end)   (sparse fine-tune)
```

The learning rate scales as `lr * 1/sqrt(L0)` to compensate for reduced model capacity at higher sparsity.

### Key Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `final_L0` | 0.01 | Target fraction of nonzero weights |
| `anneal_end_ratio` | 0.5 | L0 reaches target at 50% of training |
| `activation_sparsity_ratio` | 0.25 | AbsTopK keeps top 25% activations |
| `use_L0_lr_scaling` | True | Scale lr by 1/sqrt(L0) |
| `min_connections` | 4 | Minimum nonzero weights per neuron |
| `adam_eps` | 0.1 | Large epsilon for stable sparse updates |
| `grad_clip_rms` | 1.0 | RMS gradient clipping threshold |

### Preset Configs

| Name | Layers | Dim | Target L0 | Steps | Use Case |
|---|---|---|---|---|---|
| `tiny` | 1 | 64 | 5% | 5K | Quick test |
| `small` | 2 | 128 | 1% | 50K | Standard experiment |
| `medium` | 4 | 256 | 0.5% | 100K | Deep study |
| `large` | 8 | 512 | 0.1% | 200K | Paper reproduction |

## Acknowledgements

Thanks to [Antigravity](https://github.com/Antigravity) for the original grokking codebase.
