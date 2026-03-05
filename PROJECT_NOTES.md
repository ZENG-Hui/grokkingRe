# PROJECT_NOTES.md — grokkingRe Onboarding Guide

> **Last updated:** 2026-03-05
> **Branch:** deepteneral
> **Status:** Active research — Phase 1 (geometry analysis) complete, Phase 2 revised, algebraic structure analysis complete

---

## 1. Project Goal & Background

### What problem does this solve?

This project studies **grokking** — the phenomenon where neural networks suddenly generalize long after memorizing training data — in the context of **weight-sparse Transformers** on modular arithmetic tasks.

The core research question: **Can we force most weights to zero during training (via Top-K sparsification) so the model learns a minimal "circuit" — a small subset of connections that reveals the interpretable computation structure underlying grokking?**

### Origin

Based on the paper *"Automatically Identifying Local and Global Circuits with Linear Computation Graphs"*. The codebase started from [Antigravity](https://github.com/Antigravity)'s grokking implementation and was extended with sparsity mechanisms, geometry analysis tools, and activation-level interpretability experiments.

### Repository

- **GitHub:** `ZENG-Hui/grokkingRe` (Hui is owner, deepteneral has collaborator access)
- **Default branch:** `deepteneral`

---

## 2. Key Concepts

### Grokking
A training phenomenon: the model first **memorizes** (high train acc, low val acc), then after continued training, suddenly **generalizes** (val acc jumps to ~100%). The transition can be abrupt and occurs well past the point of overfitting. Understanding *why* this happens and *what changes* in the model during the transition is the central question.

### Modular Arithmetic Task
The task is `x ◦ y (mod p)` where:
- `p = 97` (a prime number)
- `◦` can be `+`, `-`, or `/`
- Input: 4 tokens `[x, op, y, eq]`
- Output: the result as a class label (0 to 96)
- Total dataset: ~9,409 samples (97² for addition), split 50/50 train/val

This task is simple enough to analyze mechanistically but complex enough to exhibit grokking.

### Weight Sparsity (Top-K Sparsification)
After each optimizer step, zero out all but the largest `L0 × total` weights (by absolute value) per weight matrix. This is a **hard constraint**, not gradient-based regularization. The idea: force the model to solve the task with minimal connections, revealing the essential computational circuit.

### L0 Norm
The fraction of nonzero weights in the model. L0 = 1.0 means fully dense; L0 = 0.01 means 99% of weights are zero. L0 is annealed linearly from 1.0 to `final_L0` over the first 50% of training, then held constant.

### Superposition
When a model represents more features than it has dimensions, features "share" dimensions. The paper *"Superposition Yields Robust Neural Scaling"* provides theory on how representation geometry scales. This project measures whether sparse models maintain good superposition geometry or collapse.

### Activation Sparsity (AbsTopK)
Separate from weight sparsity: after each sublayer (attention, FFN), keep only the top 25% of activations by absolute value, zeroing the rest. This encourages the model to use sparse, interpretable activation patterns.

---

## 3. Architecture Overview

### Model Variants

| Feature | Dense (`model.py`) | Sparse (`model_sparse.py`) |
|---|---|---|
| Normalization | LayerNorm or RMSNorm | RMSNorm only |
| Activation | GELU | GELU + AbsTopK (top 25%) |
| Weight constraint | None | Top-K enforced per step |
| Class | `Transformer` | `SparseTransformer` |

Both share the same high-level architecture:

```
Input tokens [x, op, y, eq]
    ↓
Token Embedding (num_tokens × dim_model) + Position Embedding (seq_len × dim_model)
    ↓
N × DecoderBlock:
    ├── MultiheadAttention (causal mask) + Residual + Norm
    └── FFN (Linear → GELU → Linear, expansion=4x) + Residual + Norm
    ↓
Final Norm → Linear → Logits [num_tokens]
```

Default config: 2 layers, dim=128, 4 heads, ~422K params.

### Why RMSNorm for sparse models?

RMSNorm doesn't subtract the mean (unlike LayerNorm), so **zero values maintain their privileged meaning** in the residual stream. This is critical when sparsity creates intentional zeros.

### Data Pipeline

`core/data.py` generates all `(x, y)` pairs for the chosen operation mod p:
- For `+` and `-`: all 97² = 9,409 pairs
- For `/`: 97 × 96 = 9,312 pairs (y ≠ 0)

Tokens: numbers 0-96, plus `eq_token = 97` and `op_token = 98`. Input format: `[x, op, y, eq]` (4 tokens), predict last position's output.

---

## 4. Training Pipeline

### Dense Training (`core/training.py`)

Standard training loop with wandb logging. Uses `get_unified_lr_scheduler` from `core/lr_scheduler.py` for warmup + cosine decay. Supports optional L0 sparsification when `regularization_type = "l2-topk"`.

### Sparse Training Pipeline (per step)

```
Forward → Loss → Backward → Grad Clip (RMS ≤ 1.0) → Optimizer Step → Top-K Sparsify → Log
```

Key mechanisms from `core/sparse_utils.py`:

1. **`enforce_weight_sparsity()`**: After each optimizer step, for each weight matrix, compute the L0 × numel threshold and zero out everything below it. Guarantees `min_connections` (default 4) nonzero weights per neuron to prevent dead neurons.

2. **`get_target_L0()`**: Linear annealing from `initial_L0` (1.0) to `final_L0` over the first `anneal_end_ratio` (50%) of training.

3. **`get_sparse_lr()`**: Learning rate = `base_lr × warmup_decay × (1/√L0)`. The `1/√L0` factor compensates for reduced model capacity at higher sparsity.

4. **`clip_grad_rms()`**: Clips gradient RMS (not norm) to `max_rms = 1.0`.

### Key Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `final_L0` | 0.01 | Target fraction of nonzero weights |
| `anneal_end_ratio` | 0.5 | When L0 reaches target (fraction of total steps) |
| `activation_sparsity_ratio` | 0.25 | AbsTopK keeps top 25% activations |
| `use_L0_lr_scaling` | True | Scale lr by 1/√L0 |
| `min_connections` | 4 | Min nonzero weights per neuron |
| `adam_eps` | 0.1 | Large epsilon for stable sparse updates |
| `grad_clip_rms` | 1.0 | RMS gradient clipping threshold |
| `weight_decay` | 0.1 | AdamW weight decay (experiments also tested 1.0 and 0.3) |

### Preset Configs (`core/config_sparse.py`)

| Name | Layers | Dim | Heads | Target L0 | Steps |
|---|---|---|---|---|---|
| `tiny` | 1 | 64 | 2 | 5% | 5K |
| `small` | 2 | 128 | 4 | 1% | 50K |
| `medium` | 4 | 256 | 8 | 0.5% | 100K |
| `large` | 8 | 512 | 8 | 0.1% | 200K |

---

## 5. Module Dependency Map

```
cli.py ──────────────────→ core/training.py ──→ core/model.py (dense)
                               ↓                    ↓
                          core/data.py          core/lr_scheduler.py
                               ↑                    ↑
experiments/superposition/     ↑               core/sparse_utils.py
  train_sweep.py ──────────────┘                    ↑
       ↓                                            │
  superposition_utils.py ←── analyze_geometry.py    │
                         ←── analyze_algebra.py     │
                         ←── analyze_activations.py │
                                                    │
core/model_sparse.py ──→ core/config_sparse.py ─────┘
       (SparseTransformer)    (SparseTrainingConfig)
```

### Core modules:
- **`core/data.py`** — Data generation (pure, no dependencies beyond torch)
- **`core/model.py`** — Dense Transformer (Transformer class, RMSNorm)
- **`core/model_sparse.py`** — Sparse Transformer (SparseTransformer, AbsTopK, SparseDecoderBlock)
- **`core/config_sparse.py`** — Dataclass configs (SparseTrainingConfig + presets)
- **`core/sparse_utils.py`** — Sparsification utilities (enforce_weight_sparsity, L0 annealing, grad clipping, stats)
- **`core/lr_scheduler.py`** — Unified LR scheduler (warmup + cosine + L0 scaling)
- **`core/training.py`** — Dense training loop (also supports L0 sparsification)

### Experiment modules (`experiments/superposition/`):
- **`train_sweep.py`** — Train dense + sparse models at multiple L0 levels, save checkpoints + training history
- **`analyze_geometry.py`** — Phase 1: measure row norms, cosine overlaps, ETF-likeness of weight matrices
- **`analyze_algebra.py`** / **`analyze_algebra_full.py`** — Algebraic structure analysis (ring topology, inverse pairs, multiplicative structure)
- **`analyze_activations.py`** — Dynamic activation analysis (identity, inverse, answer encoding)
- **`superposition_utils.py`** — Shared analysis functions (geometry computation, report formatting)
- **`plot_*.py`** — Visualization scripts

---

## 6. Key Findings So Far

### Finding 1: The Model Learns a Ring Topology

The output projection (and embedding, since they're weight-tied) organizes 97 tokens on a **circular manifold** in 128-d space:
- Numerically adjacent tokens have high cosine similarity (~0.15 at embedding, amplified to ~0.55 at Key projections)
- Diametrically opposite tokens (distance 48) have negative cosine similarity
- This is the natural geometry for addition mod p: shifting x by 1 shifts the answer by 1

### Finding 2: Key Projections Are "Ring Amplifiers"

The Key matrices in attention blocks amplify the ring topology 3-4× compared to the embedding layer. Value matrices also amplify but more smoothly. This makes sense: Keys need precise positional discrimination for attention, Values need smooth interpolation.

### Finding 3: Block 1 Is the Computation Layer

Dynamic activation analysis reveals:
- After Block 0: activations for different inputs are still nearly identical (cos_sim ≈ 0.997)
- After Block 1: activations diverge dramatically — same-answer inputs converge (cos_sim ≈ 0.98), different-answer inputs become orthogonal (cos_sim ≈ 0)
- Block 0 prepares (amplifies ring structure, sets up attention), Block 1 computes (rotates hidden state to "answer direction")

### Finding 4: No Discrete Algebraic Structure Is Explicitly Encoded

- Additive inverses (a, 97-a): no special signal in any layer
- Multiplicative group structure: no signal (expected — task is addition)
- Token 0 (identity element): not geometrically special
- The model uses continuous ring geometry, not discrete algebraic relationships

### Finding 5: Sparsity Selectively Preserves Essential Structure

- Block 0's K and V projections: ring structure preserved well under sparsification (~30% reduction)
- FFN layers: ring structure eliminated by sparsification (from 0.321 to 0.045) — this was redundant
- Block 1's K: heavily reduced — but block 1 computes via different mechanisms

### Finding 6: U-Shaped Overlap Ratio with Sparsity

Representation overlap ratio follows a U-shape across L0 levels:
- Too dense (L0=1.0): excess capacity → unnecessary clustering (ratio ~6.5)
- Optimal window (L0≈0.40-0.50 for wd=1): cleanest geometry, minimal redundancy (ratio ~3.8)
- Too sparse (L0=0.10): insufficient capacity → representation collapse (ratio ~9.4)

### Finding 7: Uniformity Regularization Would Be Harmful

Initially planned to push representations toward uniform hyperspherical distribution. Experiments showed this is wrong — the non-uniform geometry (ring topology) **is** the learned algorithm. Destroying it would destroy performance.

---

## 7. Current State & Open Questions

### What's Done
- ✅ Core codebase unified (dense + sparse in one framework)
- ✅ Phase 1 geometry analysis: overlap metrics, norm distributions, ETF comparison
- ✅ Algebraic structure analysis: ring topology confirmed in output/embedding/Key/Value layers
- ✅ Dynamic activation analysis: computation localized to Block 1, answer encoding characterized
- ✅ Training sweep: dense + L0={0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10} for wd=1 and wd=0.3
- ✅ Phase 2 revised: uniformity regularization cancelled, replaced with structural analysis questions

### Open Questions

1. **Phase transition in geometry**: Is there a qualitative change in overlap structure between L0=0.50 and L0=0.20? The U-shaped curve suggests yes.

2. **Early prediction of grokking**: Can geometric metrics at early training steps predict which sparse models will eventually grokk? This would be practically valuable.

3. **Grokking mechanism via geometry**: During grokking, overlap ratio increases from ~1 to its final value. Does this track the transition from memorization (random-like geometry) to algorithmic understanding (structured ring geometry)?

4. **Multiplicative task**: The analysis predicts that training on `x*y mod 97` would produce multiplicative group structure in Key projections instead of ring topology. This is testable.

5. **Where does Block 1 compute?**: We know *that* Block 1 performs the addition computation, but *how* — through attention, FFN, or both? What's the minimal circuit within Block 1?

6. **Deeper sparse models**: The current analysis is on 2-layer models. How does the computation distribute across layers in the 4-layer and 8-layer preset configs?

---

## 8. How to Run

```bash
# Dense baseline (quick test)
python cli.py --operation x+y --num_steps 4000

# Training sweep (dense + multiple L0 levels)
python experiments/superposition/train_sweep.py

# Geometry analysis on saved checkpoints
python experiments/superposition/analyze_geometry.py --sweep-dir experiments/superposition/results/checkpoints

# Algebraic structure analysis
python experiments/superposition/analyze_algebra.py

# Full multi-layer algebra analysis
python experiments/superposition/analyze_algebra_full.py

# Activation analysis (identity, inverse, answer encoding)
python experiments/superposition/analyze_activations.py
```

All experiment outputs go to `experiments/superposition/results/`.

---

## 9. File Reference

| Path | Purpose |
|---|---|
| `core/data.py` | Generate (x ◦ y mod p) datasets |
| `core/model.py` | Dense Transformer (LayerNorm/RMSNorm) |
| `core/model_sparse.py` | Sparse Transformer (RMSNorm + AbsTopK) |
| `core/config_sparse.py` | Hyperparameter configs (dataclass + presets) |
| `core/sparse_utils.py` | Top-K sparsification, L0 annealing, grad clip |
| `core/lr_scheduler.py` | Warmup + cosine + L0-scaled learning rate |
| `core/training.py` | Dense/sparse training loop |
| `cli.py` | CLI entry for dense training |
| `experiments/superposition/train_sweep.py` | Multi-model training sweep |
| `experiments/superposition/analyze_geometry.py` | Weight matrix geometry analysis |
| `experiments/superposition/analyze_algebra*.py` | Algebraic structure in representations |
| `experiments/superposition/analyze_activations.py` | Dynamic activation analysis |
| `experiments/superposition/superposition_utils.py` | Shared geometry analysis utilities |
| `experiments/superposition/plot_*.py` | Visualization scripts |
| `docs/analysis/AlgebraicStructure.md` | Key findings: ring topology, K amplification, Block 1 computation |
| `docs/analysis/SuperpositionExperiment.md` | Experiment plan, revised understanding, U-shape analysis |

---

## 10. Code Review Notes (2026-03-05, DeepTeneral)

### Geometry Analysis Review

**What the code actually computes (analyze_geometry.py + superposition_utils.py):**
- Row norms, pairwise cosine similarities, mean squared overlap, Welch bound
- These are computed on **weight matrices** (static analysis)
- Metrics are diagnostics, NOT optimization targets (confirmed by paper reading)

**What the code does well:**
- Theoretical grounding: overlap ratio, Welch bound, random baseline comparison
- Cross-model comparison across L0 levels
- output_proj analysis is justified: cross-entropy loss ∝ mean squared cosine similarity of representation vectors (per Superposition Scaling paper)

**What's missing or questionable:**
1. "Ring topology" claim lacks rigorous validation — code shows "numerically close tokens are more similar" but doesn't fit cos_sim(a,b) ≈ f(|a-b| mod 97) to a specific functional form (e.g., cosine on a circle)
2. Geometry snapshot during training only tracks 2 layers × 6 metrics — may miss important structure in other layers
3. No Fourier analysis of the similarity structure — ring topology should produce specific frequency signatures
4. No tracking of candidate order parameters through training dynamics

### Research Direction (agreed with Hui)

**Ultimate goal:** Find order parameters for the grokking phase transition in mod-97 addition

Two sub-goals:
- **A.** Characterize how representation vectors encode Z/97Z algebraic structure
- **B.** Find geometric quantities that show discontinuous change at the grokking transition

**Key insight:** If algebraic structure emerges abruptly at grokking, the degree of algebraic structure IS the order parameter.

### Next Steps (when resuming)
1. Modify `compute_geometry_snapshot` to track more layers and new candidate order parameters
2. Add ring topology fitting: test cos_sim(a,b) ≈ A·cos(2π(a-b)/97) + B
3. Add Fourier analysis of the 97×97 similarity matrix
4. Re-run train_sweep with enhanced tracking (requires PyTorch environment)
5. Plot candidate order parameters vs training step, overlay with val_acc curve
