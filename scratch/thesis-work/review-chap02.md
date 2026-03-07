# Review: Chapter 02 vs. Source Paper (Generalized Nested Wilson Loop)

**Thesis chapter:** `chap02.tex` (613 lines)
**Source paper:** `Dirac_NaCuSe_clean_prr.tex` (main, 840 lines) + `SM.tex` (supplementary, 782 lines)

---

## Overall Assessment

The thesis chapter is a **substantially expanded** version of the source PRL-format paper. Most key physics is present, and the thesis adds considerable pedagogical depth (e.g., the discrete Wilson line → continuous limit derivation, gauge invariance discussion, numerical recipe, explicit kz-slice mapping derivation). However, several items from the source paper + SM are missing or insufficiently treated.

---

## Section-by-Section Analysis

### 2.1 Introduction

**Status: Well-covered, minor gaps**

**A) Missing content:**
- The source paper's abstract explicitly mentions "2D nonsymmorphic materials" as a standalone application. The thesis introduction focuses almost exclusively on the 3D DSM → kz-slice → 2D picture. A sentence acknowledging purely 2D higher-order topology (not just as kz slices) would be more faithful.
- The source paper (main text, "Introduction" paragraph 2) mentions β-CuI with C₃ symmetry as a motivating example. The thesis (lines ~100–105) mentions β-CuI but doesn't explain *why* C₃ poses a problem for conventional NBP (the original paper does: the conventional NBP is only defined for systems with M_x and M_y). The thesis alludes to "三重旋转对称性" without spelling out that the conventional nested Wilson loop requires orthogonal reflection symmetries.
- The paper references Zak's Berry phase [PhysRevLett.62.2747] and King-Smith–Vanderbilt [PhysRevB.47.1651] in the introduction alongside Benalcazar. The thesis only cites these later in §2.2. Minor, but contextually relevant.

**B) Too abbreviated:**
- The source paper's third paragraph concisely explains the Dirac semimetal → kz-slice → TQI logic in 3 sentences. The thesis expands this well.

**C) Writing quality:**
- ✅ The introduction reads naturally in Chinese academic prose. No significant AI-like patterns.
- Minor: "拓扑表面态的拓扑表面态" (line ~70) is vivid but slightly informal for a thesis. Consider: "边界态本身具有拓扑结构，进而在次级边界上诱导出保护态".

---

### 2.2 Generalized Nested Wilson Loop Theory (广义嵌套威尔逊环理论)

**Status: Excellent — this is where the thesis adds the most value over the source.**

**A) Missing content:**
- **SM Section I.B "Symmetry Constraints" derivation of the sewing matrix transformation (Eqs. in SM §I.B):** The thesis presents the transformation rule for Wilson loops (Eq. 2.12, `eq:chap2_symmetry_wilson`) and nested Wilson loops (Eq. 2.13, `eq:chap2_symmetry_nested_wilson`), but the intermediate derivation showing how a *single* overlap matrix element $\langle u_{k+Δk} | u_k \rangle$ transforms under symmetry is not shown. The SM (around Eqs. 10–12 in SM §I.B) provides this step-by-step derivation:
  ```
  ⟨u_{k+Δk}|u_k⟩ = B†_{g,k+Δk} ⟨u_{D_g k+Δk}|u_{D_g k}⟩ B_{g,k} · e^{-iD_g Δk·δ}
  ```
  This is the crucial step connecting the nonsymmorphic phase factor to the discrete Wilson line elements. The thesis jumps directly to the Wilson loop transformation without this intermediate step, which is important for a PhD thesis level of rigor.

- **SM Eqs. for specific M_x, M_y, G_x, G_y on Wilson loops (SM Eqs. 13–14, labeled `eq: Reflection Wilson` and `eq: Glide Wilson`):** The SM explicitly writes out:
  ```
  B_{M_x,k} W_{x,k} B†_{M_x,k} = W_{-x, M_x k}
  B_{G_x,k} W_{x,k} B†_{G_x,k} = W_{-x, M_x k} e^{iπ}
  ```
  These concrete instantiations of the general symmetry constraint are not in the thesis. The thesis only gives the general formula and then jumps to the consequences for Wannier bands. For a thesis, these explicit results help the reader verify the general claims.

- **The determinant representation of total polarization** (thesis Eq. `eq:chap2_total_polarization` and the equation below it): The thesis presents `p_x = -i/(2π)^D ∫ log Det[W]` as a standalone equation. The SM (Eq. 4 in SM §I.A) presents this in a more complete hierarchy: individual Wannier center → sum → Berry phase → polarization → determinant. The thesis does cover this but puts the averaging over k_perp into a separate equation; this is fine.

- **2D inversion symmetry constraints (SM §I.B.3 "Inversion Symmetry"):** The SM has an entire subsection showing how 2D inversion acts on Wilson loop eigenvalues:
  ```
  {e^{i2πν_x^j(k_y)}} = {e^{-i2πν_x^j(-k_y)}}
  ```
  and explains that unlike reflection, inversion can map ν_x(k_y) and -ν_x(-k_y) within the same band. It also shows p_y^{ν_x^{s+}} = -p_y^{ν_x^{s-}} mod(1). **This entire inversion symmetry discussion is absent from the thesis.** While the thesis states that inversion can be decomposed as a combination of reflection and glide reflection, the SM's standalone treatment provides additional insight. For a PhD thesis claiming to systematically analyze 80 layer groups, the inversion symmetry case should at least be mentioned.

**B) Too abbreviated:**
- The numerical implementation subsection (the enumerated list at the end of §2.2) is a good addition. However, the **unitarization procedure** (SVD-based F → UV†) is only mentioned in passing in the preamble paragraph. The SM doesn't discuss this either — it's the thesis's own addition — but since it's mentioned, it deserves 1–2 more sentences on when/why it matters (e.g., near band degeneracies, or when the grid is coarse).

**C) Writing quality:**
- ✅ Generally strong. The logical flow from discrete → continuous → gauge invariance → eigenvalue problem is well-structured.
- The two "需要强调" (emphasis) points after Eq. `eq:chap2_wsp_det` are well-written and add genuine insight (branch cut invariance, WSP definability requiring Wannier gaps).

---

### 2.3 Higher-Order Topological Indicators in 2D Layer Groups

**Status: Mostly complete, with one significant omission**

**A) Missing content:**

1. **SM §I.B.2 "Glide Reflection Symmetry" — the three types of glide reflections:**
   The SM carefully distinguishes *three* types of in-plane glide symmetries:
   - G_{x,⊥} = {M_x | 0, 1/2, 0}
   - G_{x,∥} = {M_x | 1/2, 0, 0}
   - G_{x} = {M_x | 1/2, 1/2, 0}
   
   and shows their *different* effects on Wilson loop eigenvalues:
   - G_{x,⊥} acts same as M_x on x-direction Wilson loop (pairs ν_x and -ν_x)
   - G_{x,∥} and G_x pair ν_x and 1/2 - ν_x
   
   **The thesis §2.3.3 only discusses G_x = {M_x | 1/2, 1/2, 0} in detail** (Eq. `eq:chap2_glide_wannier`) and doesn't explicitly distinguish the three types of glide. For a thesis-level treatment, showing all three cases and their different Wannier pairing patterns would be more complete. The distinction matters because the symmetry constraint table (Table `tab:chap2_symmetry_constraint`) lists all these different cases.

2. **SM §I.B.2 — explicit eigenvalue constraints for G_{y,⊥} and G_y vs. M_y and G_{y,∥}:**
   The SM shows:
   ```
   G_{y,⊥}: {e^{i2πν_x(k_y)}} = {e^{i2πν_x(-k_y)} · e^{iπ}}
   G_{y,∥}: {e^{i2πν_x(k_y)}} = {e^{i2πν_x(-k_y)}}
   ```
   These explicit eigenvalue constraints (SM equations around line ~410–425) are not in the thesis, though their *consequences* for Wannier band polarizations appear in Table `tab:chap2_symmetry_constraint`.

3. **SM §I.B.4 "Combination of Reflection and Glide Reflection Symmetries" — the explicit combined constraint equations:**
   The SM shows the quantization result for systems with both M_x, M_y and G_x, G_y:
   ```
   p^{ν_x^{1,j}}_y = 1/2 + p^{ν_x^{2,j}}_y = 1/2 - p^{ν_x^{3,j}}_y = -p^{ν_x^{4,j}}_y = 0 or 1/2 mod(1)
   ```
   The thesis §2.3.4 only gives the p4/nmm example result (Eq. `eq:chap2_p4nmm_quantization`), not the general combined constraint. This is adequate but less general than the SM presentation.

4. **SM §I.D "Generalized Nested Berry Phase in Layer Groups" — detailed discussion per layer group:**
   The SM §I.D discusses:
   - How to choose rectangular cell for non-orthogonal lattices
   - Why orthogonality is needed (for symmetry constraints to quantize WSP)
   - The role of {E | 1/2, 1/2, 0} translational symmetry forming pseudo-glide symmetries
   - The unexplored scenario: coexistence of M_α with G_{α,∥} or G_α with G_{α,⊥} leading to Wannier band degeneracy (e.g., pmaa group)
   
   The thesis §2.3.5 mentions the rectangular cell and pseudo-glide symmetry briefly but **omits the Wannier band degeneracy scenario** (the pmaa case) entirely. This is a subtlety that the SM explicitly calls out as "outside the scope of this article" but which a thesis should at least acknowledge.

5. **SM Table "generalized NBP in layer groups" (the full 80 layer groups table):**
   The thesis notes reference to this in appendix (appendix03.tex, 附录C), which is good. The thesis text at §2.3.5 correctly refers to it. ✅

6. **SM consistency verification (θ^{(2)} along x±y vs. θ^{(4)} along x,y):**
   The SM §I.B.4 and SM §II (Lattice Model) discuss that when θ^{(4)} arises from re-choosing a supercell in an orthogonal lattice, both θ^{(2)} along x±y and θ^{(4)} along x,y capture the same HOTPT. The thesis §2.3.4 mentions this ("我们已经验证..."), but doesn't give any detail on *how* this was verified. The SM presumably has a lattice model section (SM §II, referenced as "SM App: Lattice Model") that demonstrates this with a concrete tight-binding model. **This lattice model is entirely absent from the thesis** — no minimal tight-binding model Hamiltonian for the consistency check is presented.

**B) Too abbreviated:**
- §2.3.2 (反射对称性下的Wannier能带配对与量子化): The derivation is clean but very compressed. The key physical insight — that M_x creates the sectors while M_y quantizes the sector polarization — could use a more explicit 2-sentence explanation of the "division vs. quantization" logic.
- §2.3.4 (层群分类): The 2-paragraph summary is appropriate for a thesis. The actual classification work is in the appendix table.

**C) Writing quality:**
- §2.3.1 reads well but the opening paragraph ("从逻辑上看...") is slightly repetitive of the end of §2.2.
- The explanation of why rectangular cells are needed (§2.3.5) is clear.

---

### 2.3.6 (kz Slice Symmetry Mapping)

**Status: Thorough — thesis adds significant value here**

**A) Missing content:**
- The source paper's treatment of kz-slice symmetry is 1 paragraph. The thesis §2.3.6 expands this to a full subsection with explicit Hamiltonians (Eqs. 2.17–2.22), which is excellent for a thesis.
- However, the thesis could mention that the SM (§I.C) also discusses this mapping, and perhaps note that the key insight "$\mathcal{M}_z\mathcal{T}$ acts the same as $\mathcal{M}_z$ on in-plane polarization" was already argued in the SM using ref. [bernevig2017detailedNBP] for the time-reversal part.

**B) Too abbreviated:** No — this section is more thorough than the source.

---

### 2.3.7 Minimal Model (最小模型示例与高阶费米弧)

**Status: Adequate but missing the Hamiltonian**

**A) Missing content:**

1. **The tight-binding model Hamiltonian itself:** The source paper says "Details of the tight-binding model are presented in SM" and the SM (§II, referenced as `App: Lattice Model`) presumably contains the explicit tight-binding Hamiltonian for the P4/nmm minimal model. **The thesis does not present this Hamiltonian anywhere.** For a thesis chapter, including at least the key terms of the model Hamiltonian (or a reference to where it is given, if in another chapter/appendix) would be important. The reader cannot reproduce the Wannier band plots in Fig. 2.1(c,d) without it.

2. **SM §II — lattice model demonstrating θ^{(2)} vs θ^{(4)} consistency:** As mentioned above, the SM has a section showing that in orthogonal-lattice models, θ^{(2)} along diagonal directions and θ^{(4)} along x,y capture the same HOTPT. This demonstration is absent from the thesis.

3. **Filling anomaly details:** The thesis mentions filling anomaly η = 2 vs. η = 0 and cites Fang2021FillingAnomalyC4, Fang2021ClassificationDSM. The source paper also relegates this to the SM. The thesis could give 1–2 sentences on how η is computed from symmetry indicators (#Γ_{E_{1/2}}, etc.), since it's used as a cross-check.

**B) Too abbreviated:**
- The connection between gNBP jump and HOFA is stated clearly.
- The filling anomaly cross-check is mentioned but not derived (acceptable for thesis).

---

### 2.4 Material Realizations (狄拉克半金属的拓扑保护高阶费米弧)

**Status: Good coverage with some SM details now included**

#### 2.4.1 NaCuSe

**A) Missing content:**
- **Strain data table:** ✅ Now included as Table `tab:chap2_dirac_moving` (the writing plan notes this was added).
- **Wannier band scanning at k_y=0:** ✅ Now included (discussed in the paragraph after the gNBP figure).
- **Surface vs. hinge spectrum interpretation:** The thesis §2.4.1 gives a good explanation of why the surface states are due to Z₂ topology of k_z=0 slice rather than Dirac points. This is clearer than the source paper's brief statement.
- **Lattice relaxation details:** The thesis mentions that only x-y directions were relaxed under strain, matching the source. ✅
- **Missing: Pristine NaCuSe Dirac point position.** The thesis says k_D ≈ 0.032 × 2π/c for unstrained NaCuSe and includes the strain table. The source paper says "we increase the separation between two Dirac points" but doesn't give the unstrained value explicitly — the thesis adds this, which is good.
- **Missing from SM: detailed surface/hinge calculation parameters.** The thesis §2.4.3 (计算方法) mentions 200-layer slab, 150-unit-cell nanorod width, N_KPM = 4096. The SM presumably has similar parameters. The thesis writing plan notes these were added. ✅

#### 2.4.2 KMgBi

**A) Missing content:**
1. **Orbital content of bands:** The thesis (Fig. caption and text) mentions Bi p_{x,y}, Mg s, and Bi s orbitals for the three pairs of bands. The source paper's SM Section III presumably has more detailed orbital-resolved band structures. The thesis gives a brief mention but could benefit from explicitly stating which irreducible representations the bands belong to at the Γ point.

2. **Critically tilted Dirac cone:** The thesis mentions "临界倾斜狄拉克锥" citing [le2017KMgBiCritical], but doesn't explain what "critically tilted" means (type-I to type-II transition). For a thesis, 1 sentence would help.

3. **KMgBi surface/hinge spectrum figures:** The source paper (main text) mentions KMgBi's hinge spectrum exhibits HOFAs [bernevig2020HOFA], but the thesis §2.4.2 only shows Fig. `fig:chap2_kmgbi_structure` with band structure, Wannier bands, and θ^{(4)}(k_z). **No surface or hinge spectrum figure is shown for KMgBi**, unlike NaCuSe which has Fig. `fig:chap2_nacuse_hofa`. Since the source paper itself doesn't show a KMgBi surface/hinge figure (it's in the SM of [bernevig2020HOFA]), this is understandable, but for a thesis it would strengthen the argument to include the KMgBi hinge spectrum.

4. **k·p model limitation discussion:** The thesis §2.4.2 discusses why the k·p model gives wrong θ^{(2)} results. This is an important point well-made. However, the thesis could be more explicit about *what* is different: the k·p model of [bernevig2020HOFA] fitted only 2 pairs of bands near Γ, while the ab initio TB model includes all relevant bands. The source paper makes this point concisely ("our calculated conventional NBP θ^{(2)}(k_z) of kz slices are trivial on either side of Dirac points, which is different from previous results based on a simplified k·p Hamiltonian").

**B) Too abbreviated:**
- The filling anomaly cross-check for KMgBi is mentioned but compressed into 2 sentences. The thesis could expand on which irrep (#Γ_{E_{1/2}}) changes at the band inversion, since this is the mechanism behind η changing.

---

### 2.4.3 Computational Methods (计算方法)

**Status: Good — covers the essential details**

**A) Missing content:**
- **k-point grid and convergence:** Neither the source paper nor the thesis specifies the k-point mesh used for the DFT calculations (e.g., 12×12×12 Γ-centered mesh). For reproducibility in a thesis, this should be stated.
- **Wannier function construction details:** The thesis mentions using Wannier90 and OpenMX but doesn't specify which orbitals were used as initial projections for the Wannier functions, the frozen window, or the disentanglement parameters. For a thesis, this level of detail aids reproducibility.
- **SOC treatment:** Neither document explicitly states whether spin-orbit coupling (SOC) is included. Given PT symmetry and doubly degenerate bands, SOC is clearly included, but it should be stated.
- **pybinding reference:** ✅ The writing plan notes pybinding citation was added.

---

### 2.5 Summary (本章小结)

**Status: Complete and well-organized**

The four-point summary cleanly captures: (1) theory framework, (2) layer group classification, (3) material verification, (4) bulk-hinge correspondence. This matches the source paper's Conclusion section content.

**A) Missing content:**
- The source paper's conclusion mentions "fragile topology" as a potential application of gNBP, citing PhysRevLett.123.186401, PhysRevB.100.115160, PhysRevB.100.205126. The thesis mentions this in the last sentence. ✅
- The source paper's conclusion is 1 short paragraph; the thesis summary is much more detailed. ✅

---

## Global Issues

### Missing SM Content Not Yet Mentioned

1. **SM Section II "Lattice Model" (referenced as App: Lattice Model):**
   This entire section — presumably containing a concrete tight-binding model Hamiltonian and demonstration of θ^{(2)} vs. θ^{(4)} consistency — is **not incorporated into the thesis**. For a PhD thesis, this model could be valuable:
   - It provides a reproducible minimal example
   - It demonstrates the claimed consistency between θ^{(2)} and θ^{(4)}
   - It makes the abstract formalism concrete
   
   **Recommendation:** Either include the lattice model in §2.3.7 or as an appendix, or at minimum state explicitly that such a model exists and direct the reader to the published SM.

2. **SM Section III "Material Details" for KMgBi:**
   The SM for the source paper presumably has a more detailed KMgBi analysis (orbital projections, Wannier band evolution, possibly a hinge spectrum). The thesis §2.4.2 is relatively brief compared to §2.4.1 (NaCuSe). Balancing these two material subsections would improve the chapter.

### Writing Quality — Global Patterns

- **Strengths:** The Chinese academic writing is generally fluent and natural. Technical terms are well-translated with English in parentheses. The logical flow within each section is clear.
- **No significant AI-like patterns detected.** The text reads like a careful human translation/expansion with domain expertise.
- **Minor issues:**
  - Occasional very long paragraphs (e.g., the introduction has paragraphs > 10 sentences). Breaking into shorter paragraphs would improve readability.
  - The writing plan comments at the top of the file (lines 1–58) should be removed before final submission.
  - Line ~70: "拓扑保护稳定边界态" — consider "拓扑保护的束缚态" or "拓扑保护的零能角态" for precision.
  - The phrase "本工作的核心创新在于" (§2.2, ~line 115) is slightly promotional for a thesis; consider "本章形式的关键推广在于".

### Figures

The thesis references 4 figures:
- Fig 2.1 (chap02_Fig1.eps) — model schematic + Wannier bands
- Fig 2.2 (chap02_Fig2.eps) — NaCuSe structure + bands + Wannier + gNBP
- Fig 2.3 (chap02_Fig3.eps) — NaCuSe surface/hinge spectra
- Fig 2.4 (chap02_KMgBi.png) — KMgBi bands + Wannier + gNBP

The source paper has:
- Fig 1 — same as thesis Fig 2.1
- Fig 2 — same as thesis Fig 2.2
- Fig 3 — same as thesis Fig 2.3

**Missing figures from SM:**
- SM presumably has additional figures for the lattice model, the Wannier band k_z scan at k_y=0, and possibly the strain-dependent Dirac point shift plot. The Wannier band scan at k_y=0 is now discussed in text but no separate figure is shown.
- No KMgBi hinge spectrum figure in the thesis.

---

## Priority Recommendations

1. **HIGH: Add the tight-binding model Hamiltonian** for the P4/nmm minimal model (from SM §II). This is referenced but never presented.
2. **HIGH: Add the inversion symmetry discussion** from SM §I.B.3 — even a brief paragraph noting that 2D inversion connects Wannier sectors with opposite polarization, and that it can be decomposed into reflection/glide combinations.
3. **MEDIUM: Distinguish the three types of glide reflections** (G_{x,⊥}, G_{x,∥}, G_x) explicitly in §2.3.3 before jumping to the specific G_x case.
4. **MEDIUM: Include the intermediate derivation** showing how a single overlap matrix element transforms under symmetry (SM Eqs. 10–12), at least in §2.3.1.
5. **MEDIUM: Add KMgBi hinge spectrum** figure for completeness.
6. **LOW: Add DFT computational parameters** (k-mesh, SOC statement, Wannier projection orbitals).
7. **LOW: Remove writing plan comments** from the top of the file.
