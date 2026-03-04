# Superposition Yields Robust Neural Scaling

**Paper**: Yizhou Liu, Ziming Liu, Jeff Gore (MIT), NeurIPS 2025
**Code**: https://github.com/liuyz0/SuperpositionScaling

## Core Question

Why does LLM loss decrease as a power law with model size? This paper argues that **representation superposition** — models representing more features than they have dimensions — is a central driver of neural scaling laws.

## Toy Model Setup

Based on Anthropic's superposition model (an autoencoder):

- **Input**: x in R^n, where each x_i = u_i * v_i, u_i ~ Bernoulli(p_i), v_i ~ U(0,2)
- **Architecture**: y = ReLU(W @ W^T @ x + b), where W in R^{n x m}, m << n
- **Loss**: L = E[||y - x||^2] (reconstruction error)
- **Feature frequency**: p_i proportional to 1/i^alpha (power-law, controlled by data exponent alpha)
- **Key knob**: weight decay gamma controls superposition degree
  - Large positive gamma -> weak superposition (only top-m features represented, rest ignored)
  - Small/negative gamma -> strong superposition (all features represented with overlapping vectors)

W_i (row i of W) is the representation vector for feature i in the m-dimensional hidden space.

**Fraction of represented features**: phi_{1/2} = fraction of rows with ||W_i|| > 1/2

## Three Main Results

### Result 1: Weak Superposition — "Power law in, power law out"

When superposition is weak (large weight decay), only the top m features are represented perfectly. The rest are ignored.

**Loss** = sum of frequencies of unrepresented features:

```
L ≈ (4/3) * sum_{i > m} p_i
```

- If p_i ~ 1/i^alpha with alpha > 1, the integral gives L ~ m^{-(alpha-1)}, i.e., model exponent alpha_m = alpha - 1
- If frequencies are NOT power-law (e.g., exponential, linear), the loss is NOT a power law
- **Conclusion**: Scaling law exponent is entirely determined by data distribution. No power-law frequencies -> no power-law scaling.

### Result 2: Strong Superposition — Geometric 1/m scaling

When superposition is strong (small/negative weight decay), almost all features are represented, but their vectors overlap.

**Key geometric facts**:

1. Row norms of W are bimodal around 1. More important features have ||W_i|| > 1 ("strongly represented")
2. The number of strongly represented features scales as ~m^2/2 (not m), following ETF-like geometry
3. Squared overlaps (W_i . W_j)^2 between representation vectors scale as **1/m**, regardless of details
4. This is because isotropic vectors on a unit sphere in R^m have mean squared overlap = 1/m

**Loss arises from interference** between overlapping representations:

```
L ~ mean squared overlap ~ 1/m    (for even or moderately skewed frequencies)
```

- For even frequencies (small alpha): alpha_m ≈ 1, robustly, across different frequency distributions
- For very skewed frequencies (large alpha): alpha_m ≈ 2(alpha - 1), because the ~m^2/2 strongly represented features contribute negligible loss, and unrepresented tail dominates
- **Conclusion**: Strong superposition gives a robust 1/m scaling law that does NOT depend on data being power-law distributed. The geometry of representation vectors alone drives the scaling.

### Result 3: LLMs are in the Strong Superposition Regime

Verified on OPT, GPT-2, Qwen2.5, Pythia (100M to 70B parameters):

1. **Mean squared overlaps** of language model head rows W_i/||W_i|| scale as **~1/m** (confirmed)
2. **Token frequencies** follow Zipf's law with alpha ≈ 1 (flat enough for the robust regime)
3. **Loss scales as**: L = C_m / m^{alpha_m} + L_{\m}, with fitted **alpha_m = 0.91 ± 0.04** (close to 1)
4. **Chinchilla consistency**: From Chinchilla models, N ~ m^{2.52}, giving alpha_m = 2.52 * 0.35 = 0.88 ± 0.06, also close to 1

The cross-entropy loss in LLMs, when expanded to lowest order, is proportional to the mean squared cosine similarity between representation vectors — the same quantity that scales as 1/m.

## Phase Diagram Summary

```
                    Weak superposition          Strong superposition
                    (large weight decay)        (small/negative weight decay)

Even frequencies    Loss NOT power-law          alpha_m ≈ 1 (robust)
(small alpha)       (slow, poor scaling)        ("1/width" scaling from geometry)

Power-law freq.     alpha_m = alpha - 1         alpha_m ≈ 1 (when alpha ~ 1)
(alpha > 1)         ("power law in, out")       alpha_m ≈ 2(alpha-1) (large alpha)

LLMs (alpha ≈ 1)    NOT the operating regime    alpha_m ≈ 0.9 ← LLMs are HERE
```

## Key Insights for Practice

1. **Why scaling laws work**: Superposition forces all features to share the hidden space. The geometric interference (squared overlap ~ 1/m) gives a natural and robust power-law decay.

2. **Can we beat the scaling law?** For natural language (alpha ≈ 1), the answer is **no** — the 1/m scaling is geometric and fundamental. For domain-specific tasks with very skewed feature frequencies (large alpha), alpha_m can exceed 1.

3. **When will scaling stop?** When model dimension m approaches the effective number of independent features (at least vocabulary size), the power-law will break down and loss from width will vanish.

4. **Encouraging superposition helps**: Architectures like nGPT (unit-sphere constraints) and optimizers without weight decay can enhance superposition, letting smaller models match larger ones. But this changes the coefficient, not the exponent.

5. **Width vs Depth**: The paper decomposes model-size loss as f_m(m) + f_l(l), where f_m is representation loss (studied here) and f_l is parsing/processing loss from transformer layers. At optimal width-depth ratio, these should be balanced, so f_l(l) ~ f_m(m) ~ 1/m.

## Limitations

- Analysis is based on the toy model without rigorous analytical solutions in the strong superposition regime
- Does not study scaling with dataset size or training steps (only model width)
- The connection between toy model features and LLM token representations is a simplification
- The role of transformer layers (parsing loss) is acknowledged but not studied
- Encouraging superposition may hurt mechanistic interpretability and AI safety

## Common Misreading: "Uniform Distribution on Hypersphere = Better"

A natural but incorrect takeaway from this paper is: "representation vectors should be as uniformly distributed on the hypersphere as possible, and we should add regularization to encourage this." This misreading confuses a **descriptive observation** with a **prescriptive goal**. Clarifications:

### What the paper actually says

1. **1/m overlap is a statistical baseline, not an optimization target.** In the toy model, when n >> m random features are projected into m dimensions, the mean squared cosine overlap between their representation vectors naturally approaches 1/m. This is a property of high-dimensional geometry (concentration of measure), not a design goal.

2. **ETF-like geometry is an emergent structure, not a loss function.** The paper observes that strongly represented features (||W_i|| > 1) tend to form ETF-like (equiangular tight frame) arrangements. This helps error correction via bias cancellation. But the paper explicitly states (Section 3.2): ETF structure "can help error correction and reduce loss values, but would not change the typical scaling with m." ETF improves the coefficient, not the exponent.

3. **The paper advocates for more superposition, not more uniformity.** The key recommendation (Section 5) is: "encouraging superposition could enable smaller models to match the performance of larger ones." Superposition means representing MORE features with overlap, not distributing existing features more evenly.

4. **Structured deviation from uniformity encodes task knowledge.** A trained model should have representation vectors that reflect the semantic/algebraic relationships between features. Tokens that are algebraically related (e.g., additive inverses in mod-97) should have specific angular relationships — not be equidistant from all others. Overlap ratio >> 1 after training reflects learned structure, not a deficiency.

### What our experiments show

We measured overlap ratio across dense and sparse models at different L0 levels on the grokking task (x+y mod 97):

```
                    wd=1.0 (4K steps)         wd=0.3 (8K steps)
Model               val_acc   ratio           val_acc   ratio
Dense               1.000     6.45            0.997     2.85
Sparse L0=0.80      1.000     5.32            1.000     3.18
Sparse L0=0.50      1.000     3.83            1.000     3.29
Sparse L0=0.40      1.000     3.79            1.000     4.03
Sparse L0=0.20      0.999     5.53            1.000     5.15
Sparse L0=0.10      0.339     9.39            0.429     4.49
```

Key observations:

- **All grokking-successful models have ratio >> 1.** The dense model (ratio=6.45) and sparse L0=0.40 (ratio=3.79) both achieve val_acc=1.0. Pushing ratio toward 1.0 would mean destroying the task-specific representation structure the model learned.

- **Overlap ratio follows a U-shape with sparsity.** There exists an optimal sparsity window (L0 ≈ 0.4-0.6 for wd=1) where ratio is minimized. Too dense → excess directional freedom causes clustering. Too sparse → insufficient effective dimensions forces excessive interference.

- **Extremely high ratio correlates with training failure.** L0=0.10 has ratio=9.39 and val_acc=0.34. Here the model cannot represent enough features — this is representation collapse, not structured learning.

- **Weight decay shifts the U-shape.** Higher wd concentrates energy in fewer weight directions (higher ratio for same L0). Lower wd distributes energy more evenly (lower ratio baseline, but U-shape minimum shifts right).

### Correct interpretation of geometric metrics

| Metric | Good diagnostic for | NOT a good target for |
|---|---|---|
| Overlap ratio ≈ 1 | Detecting untrained/random weights | Training loss (would prevent learning structure) |
| Overlap ratio >> 1 | Confirming model learned task structure | Nothing — this is expected after training |
| Overlap ratio extremely high | Detecting representation collapse | — |
| Max \|cos sim\| near 1 | Detecting token pairs the model cannot distinguish | — |
| Norm std large | Detecting uneven feature importance | — |

### Revised direction

The geometric metrics are valuable as **diagnostic tools**, not as **optimization objectives**. The interesting question is not "how to make representations more uniform" but rather "what algebraic structure does the non-uniform geometry encode?" Specifically:

- Do the high-overlap token pairs correspond to algebraic relationships in Z/97Z (cyclic group structure, cosets, generators)?
- Does the U-shaped ratio curve reflect a phase transition between different representation strategies (memorization vs. algorithmic)?
- Can the geometric structure predict which sparse models will generalize before we see val_acc rise?

## Connection to Our Work

This paper provides theoretical grounding for studying **how sparse models represent features in limited dimensions**. Key connections:

- Our grokking project forces weight sparsity via Top-K, which directly limits the effective representation capacity — analogous to reducing m in the toy model
- The paper's finding that **strong superposition is beneficial** suggests that sparse models may need to develop their own form of superposition to compensate for having fewer active weights
- The relationship between feature frequency distribution and scaling exponent is relevant to understanding which modular arithmetic "features" survive under sparsification
- Weight decay's role as a superposition controller is directly relevant — our sparse training uses weight decay + Top-K, which interact with superposition dynamics
- **Geometric metrics (overlap ratio, max overlap) are diagnostics, not objectives** — they help us understand what the model learned, but regularizing toward uniformity would likely harm performance
