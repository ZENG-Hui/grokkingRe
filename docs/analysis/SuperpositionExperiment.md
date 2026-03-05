# Superposition Geometry Experiment Plan

## Motivation

The paper *"Superposition Yields Robust Neural Scaling"* shows that in the strong superposition regime, representation vectors' mean squared overlap scales as 1/m, and this geometric property drives the 1/m loss scaling law. We initially asked whether encouraging uniform hyperspherical distribution could improve sparse models. **Our experiments showed this framing was wrong** — see the "Revised Understanding" section below.

The corrected question is: **When we force weight sparsity via Top-K, how does the geometry of representation vectors change, and what task-specific structure does the non-uniform geometry encode?**

## Experiment Overview

Two phases, organized under `experiments/superposition/`:

```
experiments/superposition/
├── analyze_geometry.py        Phase 1: measure geometry of existing models
├── train_with_reg.py          Phase 2: train with uniformity regularizer
├── results/                   all outputs go here
│   ├── geometry/              Phase 1 plots and data
│   └── intervention/          Phase 2 plots and data
└── superposition_utils.py     shared analysis functions
```

---

## Phase 1: Geometry Analysis of Existing Models

### Goal

Measure the geometric structure of weight matrices in dense vs sparse models at different L0 levels, and compare against the paper's predictions.

### Which weight matrices to analyze

| Matrix | Shape | Why |
|---|---|---|
| `output_proj.weight` | [99, 128] | 99 token representations in 128-d space. Direct analog to the paper's W matrix. n=99 > m=128? No, but close. |
| `token_embeddings.weight` | [99, 128] | Input token representations, complementary to output. |
| `self_attn.in_proj_weight` (Q part) | [128, 128] | Attention query vectors per dimension. |
| `ffn.0.weight` (FFN up) | [512, 128] | 512 features represented in 128-d. n=512 >> m=128. Best analog to paper's setup. |

The **FFN up-projection** is the most interesting: 512 rows in 128 dimensions, giving n/m = 4, which is firmly in superposition territory.

### Metrics to compute

For each weight matrix W with rows W_i:

1. **Row norm distribution**: histogram of ||W_i||, check for bimodal structure around 0 and 1
2. **Fraction represented**: phi_{1/2} = fraction of rows with ||W_i|| > median(||W_i||)
3. **Mean squared overlap**: E[(W_i/||W_i|| . W_j/||W_j||)^2] for all i != j, check if ~1/m
4. **Variance of squared overlaps**: compare to random unit vectors (theoretical: 2/m^2) and ETF (theoretical: 0)
5. **Max absolute overlap**: compare to Welch bound kappa = sqrt((n-m)/(m(n-1)))
6. **Overlap vs frequency**: do higher-frequency features (tokens with more training samples) have smaller overlaps? (paper predicts yes)

### Models to analyze

Train a sweep of models and save checkpoints:

| Model | Type | L0 | Description |
|---|---|---|---|
| dense-baseline | Dense | 1.0 | No sparsification, pure L2 |
| sparse-L0-0.80 | Sparse | 0.80 | Light sparsification |
| sparse-L0-0.50 | Sparse | 0.50 | Medium sparsification |
| sparse-L0-0.20 | Sparse | 0.20 | Heavy sparsification |
| sparse-L0-0.10 | Sparse | 0.10 | Very heavy sparsification |

All models use the same architecture (2L x 128D x 4H), task (x+y mod 97), and training steps (10K).

### Expected observations

- Dense model: output_proj rows should show some superposition (99 tokens in 128-d, but n < m so mild)
- Dense model: FFN up-proj should show clear superposition (512 rows in 128-d)
- As L0 decreases, effective capacity drops. Two possibilities:
  - (a) Surviving weights maintain uniform geometry (good: superposition preserved)
  - (b) Geometry becomes distorted, some rows collapse (bad: superposition lost)
- If (b), this motivates Phase 2 intervention

### Output

- `results/geometry/row_norms_{model}.png` — norm distributions per layer
- `results/geometry/overlap_scaling.png` — mean squared overlap vs effective dimension
- `results/geometry/overlap_distribution_{model}.png` — overlap histograms
- `results/geometry/geometry_summary.json` — all numeric metrics

---

## Revised Understanding (from Phase 1 results)

Our Phase 1 experiments revealed that the original Phase 2 plan — adding uniformity regularization to improve sparse models — was based on a misreading of the paper. Key realizations:

### 1/m is a baseline, not a target

The paper observes that mean squared overlap scales as 1/m in the strong superposition regime. This is a **descriptive fact** about high-dimensional geometry, not an optimization objective. After training, overlap ratio >> 1 is expected and healthy — it means the model learned task-specific structure where related tokens have specific angular relationships.

### Uniformity regularization would likely hurt

Pushing representation vectors toward uniform distribution on the hypersphere would destroy the algebraic structure the model needs to learn. In mod-97 arithmetic, tokens are not interchangeable — x and (97-x) are additive inverses, and the model needs to encode such relationships through non-uniform geometry.

### What the U-shape tells us

Overlap ratio follows a U-shape with sparsity level:

```
wd=1.0:  Dense=6.45  0.80=5.32  0.70=4.86  0.60=4.12  0.50=3.83  0.40=3.79  0.30=3.99  0.20=5.53  0.10=9.39
                                                                    ↑ minimum
wd=0.3:  Dense=2.85  0.80=3.18  0.70=3.10  0.60=2.98  0.50=3.29  0.40=4.03  0.30=4.21  0.20=5.15  0.10=4.49
                                             ↑ minimum
```

- Left side (too dense): excess directional freedom allows unnecessary clustering
- Bottom (optimal window): sparsification removes redundant connections, geometry is cleanest while still encoding task structure
- Right side (too sparse): insufficient effective dimensions force excessive interference, approaching representation collapse

### Geometric metrics: diagnostics, not objectives

| Metric | Good diagnostic for | NOT a good target for |
|---|---|---|
| Overlap ratio ≈ 1 | Untrained/random weights | Training regularization |
| Overlap ratio >> 1 | Learned task structure | Nothing — expected |
| Overlap ratio extremely high | Representation collapse | — |
| Max \|cos\| near 1 | Indistinguishable token pairs | — |

---

## Phase 2: Structural Analysis of Representations (Revised)

The original Phase 2 (uniformity regularization) is cancelled. The revised direction focuses on understanding **what structure the non-uniform geometry encodes**.

### Proposed questions

1. **Token clustering**: Do the high-overlap token pairs in output_proj correspond to algebraic relationships in Z/97Z? For example, do additive inverses (x, 97-x) cluster together?

2. **Phase transition**: The U-shaped ratio curve suggests different representation strategies at different sparsity levels. Is there a qualitative change in the overlap structure (not just magnitude) between e.g. L0=0.50 and L0=0.20?

3. **Predictive diagnostics**: Can geometric metrics at early training steps predict which sparse models will eventually grokk? This would be practically valuable for early stopping.

4. **Grokking mechanism**: During the grokking transition, overlap ratio increases monotonically (from ~1 to final value). Does this reflect a transition from memorization (random-like geometry) to algorithmic understanding (structured geometry)?

---

## Execution Summary

### Completed

1. `superposition_utils.py` — geometry analysis functions
2. `analyze_geometry.py` — static checkpoint analysis
3. `train_sweep.py` — training with history tracking (loss/acc/geometry per 50 steps)
4. `plot_geometry.py` — fig1-5: static geometry comparison across models
5. `plot_training.py` — fig6-8: training curves and geometry evolution
6. `plot_wd_comparison.py` — fig9-12: wd=1 vs wd=0.3 comparison
7. `plot_L0_sweep.py` — fig13-14: U-shape curve and ratio-vs-accuracy scatter

### Models trained

- wd=1.0, 4000 steps: dense + L0={0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10}
- wd=0.3, 8000 steps: dense + L0={0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10}

### Outputs

All in `experiments/superposition/results/`:
- `checkpoints/` — model weights + training history JSON
- `geometry/` — 14 figures (fig1-fig14) + geometry_summary.json

## Dependencies

- PyTorch (existing)
- matplotlib (existing)
- No new dependencies needed
