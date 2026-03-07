# Chapter 3 Review: Source Papers vs Thesis

## Overview

Chapter 3 integrates two papers:
- **Paper 1**: "Intrinsic Nonlinear Hall Detection of the Néel Vector for 2D Antiferromagnetic Spintronics" (QuantumMetricNonlinearHall — main.tex + SM.tex)
- **Paper 2**: "Spontaneous Inversion Symmetry Breaking and Emergence of Berry Curvature and Orbital Magnetization in Topological ZrTe₅ Films" (BerryCurvatureNonlinearHall — slideBCD_clean.tex + SM_ZrTe5films.tex)

The thesis chapter (chap03.tex, ~786 lines of content) is structured as:
- 3.1 Introduction: quantum metric & Berry curvature driven weak-field NLH
- 3.2 Quantum metric INH for 2D AFM spintronics (Paper 1)
- 3.3 Berry curvature NLH as probe for thin-film surface sliding (Paper 2)
- 3.4 Summary (appears to be incomplete/cut off in file)

---

## Section-by-Section Analysis

### Section 3.1 — Introduction

**Status: Well-developed, with original derivation content**

The introduction goes beyond both source papers by providing:
- A unified framework connecting BCD and INH from the density matrix master equation
- Derivation of BCD from first-order nonequilibrium distribution (Eq. 3.1-BCD)
- Derivation of INH from second-order density matrix (quantum metric contributions)
- Clear taxonomy: BCD (∝τ¹, T-even) vs INH (∝τ⁰, T-odd)

This section references material from a *third* source (HighFieldDynamics/QMO_SM.tex) and the thesis appendix, which is appropriate for a thesis introduction. **No missing content from Papers 1 or 2 here.**

### Section 3.2 — Quantum Metric INH (Paper 1)

#### A) Content MISSING from thesis

1. **MnSe and MnTe results**: Paper 1 main text explicitly mentions "Mn*X* (*X*=S, Se and Te)" as a family and defers MnSe/MnTe to SM. The SM contains detailed calculations for MnSe (Fig. S6) and MnTe (Fig. S7). The thesis mentions these materials briefly in the summary paragraph but does **not** include any numerical results, figures, or discussion of how SOC strength varies across the family (S < Se < Te) and its effect on INH magnitude. This comparison is physically important — it shows the INH effect is robust across the material family.

2. **Magnetocrystalline anisotropy energy (MAE)**: SM Table S1 and Fig. S5 give MAE data (in-plane vs out-of-plane energy differences: 0.429 meV for N∥z, 0.134 meV for N∥y relative to N∥x). The thesis mentions "significant in-plane anisotropy" but does not quote the actual MAE values or show the MAE angular dependence figure.

3. **Symmetry analysis table for all Néel vector directions**: SM Table S2 (and its equivalent) provides a systematic tabulation of magnetic point groups and allowed INH tensor components for N∥x, N∥y, N∥z. The thesis mentions the N∥x case ($2'/m$) and lists allowed components, but does **not** include the full symmetry analysis table covering all directions. This table would strengthen the systematic nature of the thesis.

4. **Out-of-plane Néel vector (N∥z) INHC**: SM discusses the case N∥z with magnetic point group $\bar{3}'m'$ and shows it allows different tensor components (xyz, xzy, etc.). The thesis **only** discusses in-plane Néel vectors. The out-of-plane case is relevant for completeness.

5. **First-principles calculation details (SM Section "First-principles calculations")**: SM provides detailed computational parameters: VASP with GGA+U (U=2.3 eV), 450 eV cutoff, 16×16×1 k-mesh, Wannier90 fitting details, symmetrized Hamiltonian, 501×501×1 energy mesh. The thesis does not include a dedicated computational methods subsection for Paper 1's calculations.

6. **Complete 8-band model derivation**: The SM provides the full derivation of $H_0$, $H_1$, $T_x$, $T_y$ matrices (Eqs. S2–S4), the explicit downfolding procedure (Eq. S5), the treatment of the ε parameter, and the detailed discussion of why ε can be treated as a constant. The thesis includes the final effective Hamiltonian and the 8-band starting point (Eq. for H) but abbreviates the intermediate steps significantly. Specifically:
   - The explicit forms of $H_{i=0,1}$ diagonal blocks (Eq. S2 in SM with $C_{0;i}$ through $C_{5;i}$ terms) are not shown
   - The explicit forms of $T_x$ and $T_y$ off-diagonal blocks (Eq. S3 with $C_6$ through $C_{10}$ terms) are not shown
   - The discussion of why the downfolding energy parameter ε is irrelevant is omitted

7. **Analytic BCP and INH expressions for the tilted Dirac model**: The SM provides the full analytic BCP components $G_{xx}$, $G_{yy}$, $G_{xy}$ (Eq. S9) and the INH integrand $\lambda^{yxx}$ (Eq. S10), plus the Fermi surface integration procedure. **The thesis DOES include these** (Eqs. labeled chap03-BCP-analytic and chap03-chi-fermi-surface). ✓

8. **Angular dependence from C₃ symmetry — general formula**: SM Section S1.2 derives the general Fourier expansion form (Eq. S13) from M_y, T/P, and C₃ constraints systematically. The thesis states the result (Eq. theta_dependence1, theta_dependence2 and Eq. chap03-chi-total-3pairs) but does **not** include the systematic derivation showing how each symmetry constrains the Fourier expansion (Eq. S14 in SM). This derivation is pedagogically valuable for a thesis.

9. **Detailed tilted Dirac cone pair model**: SM Section S1.3 provides:
   - The full downfolding for Dirac cone pairs (Eq. S15, S16, S17)
   - Discussion of how ε works differently at non-Γ points (already-tilted before AFM)
   - The simple 2-band model Hamiltonian for each Dirac cone pair (Eq. S18)
   - Full analytic result for χ_D of a single tilted Dirac cone (Eq. S21)
   - Taylor expansion in γn giving Fourier structure (Eq. S22)
   - Sum over pairs giving final χ_pair (Eq. S23, S24)
   - Sum over 3 pairs giving χ_tot (Eq. S25, S26)
   
   The thesis covers the key results (Eqs. chap03-DC-pair, chap03-chi-tilted-Dirac, chap03-chi-total-3pairs) but omits:
   - The explicit downfolding equations for the Dirac cone pair (Eq. S16, S17)
   - The detailed discussion of ε at non-Γ points
   - The intermediate Taylor/Fourier expansion steps (Eq. S22)
   - The pair summation algebra (Eq. S23, S24)

#### B) Content present but TOO ABBREVIATED

1. **k·p model derivation**: The thesis jumps from the 8-band Hamiltonian directly to the final 4-band effective model. The intermediate steps (explicit H±, T_x, T_y forms, downfolding procedure) are referenced to the supplement but not shown. For a PhD thesis, showing these steps would be standard.

2. **Discussion of quadratic terms**: The thesis mentions "加入二次项后" and references figures but does not explain *which* quadratic terms are added or *why* they qualitatively change the behavior (e.g., making the upper valence band bend over from linear to parabolic, changing the shape of the Fermi surface).

3. **Non-Γ point model**: The pair model derivation is condensed. The thesis gives the final form of $H_{\rm eff,D_1}$ and $H_{\rm eff,D_2}$ but does not show how they arise from downfolding of the original 4-band model at non-Γ points.

4. **Fourier expansion derivation**: Eq. chap03-chi-total-3pairs gives the end result but the symmetry-based derivation is omitted.

#### C) Writing quality issues

1. **Inconsistent notation**: The thesis uses both $\INH$ (custom macro) and $\sigma_{\mathrm{INH}}$ interchangeably. In some places BCP dipole is written as $\Lambda$ vs $\lambda$ without clear distinction of band-resolved vs total.

2. **Abrupt transition**: The transition from the Γ-point model (subsection 3.2.5) to the non-Γ point model (subsection 3.2.6) lacks a bridging paragraph explaining why the non-Γ points matter physically (the thesis just says "对于μ≈-800 meV处的INH" without motivating the reader).

3. **Figure references to SM**: Several figures (FigS1–FigS4) are included as thesis figures but are labeled "改绘自文献[XX]的补充材料". This is fine but the captions could be more self-contained.

---

### Section 3.3 — Berry Curvature NLH / ZrTe₅ (Paper 2)

#### A) Content MISSING from thesis

1. **Detailed structural analysis of bulk noncentrosymmetric phases**: Paper 2 SM Section III ("Calculations for various possible phases of bulk ZrTe₅") provides extensive first-principles comparison of proposed I-breaking modes:
   - Te^z staggered displacement analysis (SM Fig. S2, Tables S1–S2)  
   - Pna2₁ phase energy comparison across 4 different XC functionals (PBE, PBEsol, DFT-D3, optB86b)
   - Force analysis showing Pna2₁ is energetically unfavorable
   
   The thesis mentions "先前提出的若干 I 破缺模式在能量上并不占优" and references the SM, but provides **zero numerical details**. For a thesis, at least a summary table of these competing phases would be valuable.

2. **Thick unit cell (bulk supercell) sliding calculations**: SM Section III.B provides:
   - Energy profiles for 4–16 layer supercells (SM Fig. S3)
   - Table S3 with energy differences and activation energies
   - Discussion of space group changes (Pnnm for 2-layer, Pmn2₁ for thicker cells)
   - Explicit statement that bulk sliding requires external stimuli
   
   The thesis mentions "metastable I-breaking phases separated from the centrosymmetric Cmcm phase by an activation energy of about 6 meV" but does not include the supercell figures or table.

3. **Multilayer sliding results in detail**: SM Section IV provides extensive multilayer analysis:
   - Constraint relaxation method description
   - Different layer sliding configurations in 9-layer films (SM Fig. S5)
   - Energy comparison of different sliding layers (SM Fig. S6)  
   - AFE vs FE sliding configurations (SM Fig. S7)
   - Discussion of which layers contribute most to sliding
   
   The thesis has a brief subsection on multilayer/surface but is very condensed compared to the SM. The planned figures (chap03_Fig10, Fig11, Fig12) are referenced in comments but it's unclear if they are fully integrated.

4. **Band structure comparison P₀ vs P±₁**: SM Fig. S8 shows the band structure of the non-polar P₀ phase for comparison with P±₁. The thesis mentions "similar band structure as the nonpolar P₀ phase" but does not show this comparison figure.

5. **k-resolved BCD density figure**: SM contains a figure showing the k-resolved BCD distribution in the BZ, revealing which k-points contribute most to D_xz. The thesis plan mentions "chap03_Fig9_k_resolved_BCD.pdf" but it's unclear if this is fully integrated into the text with proper discussion.

6. **Band-resolved BC at Γ and S**: Paper 2 main text Fig. 3(b,c) and SM Fig. S12 show band-resolved BC along specific k-paths, explaining why the negative and positive BCD peaks come from Γ and S respectively. The thesis references this but the detailed k-path analysis is abbreviated.

7. **Symmetry constraint table**: Paper 2 SM Table S4 provides the symmetry constraints on physical quantities (P, Ω, m, BCD, KME) for P₀ vs P±₁ phases. Not included in thesis.

8. **Ionic vs electronic contributions to polarization**: Paper 2 SM Section VI.G discusses the decomposition of polarization into ionic and electronic contributions. The thesis mentions "来自离子位移与电子电荷重分布的共同贡献" but provides no quantitative breakdown.

9. **Spin vs orbital contribution to KME**: Paper 2 SM has a dedicated subsection (and Fig. showing band-resolved OM vs spin moment) demonstrating that orbital magnetization dominates over spin. The thesis states this fact but doesn't show the supporting data.

10. **Physical connection between NAHE and KME via AHE**: Paper 2 main text has a paragraph explaining NAHE = KME + AHE conceptually. The thesis includes this paragraph but abbreviates the physical intuition.

11. **Connecting BCD and polarization in parameter space k-λ**: SM Section I.B derives the Berry curvature tensor in the extended (k_x, k_y, λ) parameter space (Eq. S7–S12), showing:
    - $\Omega_{x\lambda}$, $\Omega_{y\lambda}$, $\Omega_z$ components
    - Polarization as integral of $\Omega_{y\lambda}$ (Eq. S13)
    - Why BCD emerges from velocity correction rather than BC change
    
    This elegant derivation connecting sliding, polarization, and BCD through 3D quantum geometry is **entirely absent** from the thesis.

12. **Detailed BCD and KME derivations**: SM Section I.D provides the full analytical calculation:
    - Definition of intermediate parameters (a, b, c, k_μ) — Eq. S19
    - Integral formulas I₁, I₂, I₃ — Eq. S20
    - Complete BCD derivation — Eq. S21 (multi-line)
    - Complete KME derivation — Eq. S22 (multi-line)
    
    The thesis includes the final results but not these intermediate steps. For a thesis, showing the key integration steps would be appropriate.

13. **First-principles methods for Paper 2**: SM Section II provides computational details (VASP, PBE+D3/optB86b, 400 eV cutoff, k-meshes, Wannier fitting, WannierBerri, WannierTools). The thesis does not have a dedicated methods subsection for these calculations.

14. **NEB path details**: The thesis mentions the nudged elastic band calculation but doesn't explain the methodology or show the intermediate images along the switching path.

#### B) Content present but TOO ABBREVIATED

1. **Intralayer distortion mechanism**: The thesis mentions "alternative clockwise and counterclockwise twist of alternating ZrTe₅ pentagons" but the physical explanation of why this stabilizes the structure (offsetting the energy increase from pure sliding) could be expanded.

2. **Comparison with other sliding ferroelectrics**: The thesis includes the comparison with BN and WTe₂ bilayers but doesn't discuss *why* trilayer ZrTe₅ has such dramatically larger polarization (3 orders of magnitude). The source paper attributes this partly to the larger structural distortion.

3. **Multilayer/surface discussion**: This is the weakest part of the thesis section. The source paper has an extended discussion about:
   - How sliding effect diminishes for inner layers
   - Multiple possible sliding configurations on natural cleavage surfaces
   - Connection to experimentally observed temperature-dependent NAHE (disappearing above ~30 K)
   - Ferroelectric Curie temperature estimation
   The thesis covers these points very briefly.

4. **k·p model construction**: The symmetry analysis (using Dirac Γ matrices and theory of invariants) from SM is condensed. The thesis shows the symmetry representations and final Hamiltonian but omits the Clifford algebra argument and the systematic enumeration of allowed terms.

5. **The Γ vs S point argument**: Paper 2 and SM emphasize that Γ and S share the same little group, so both contribute to BCD/KME. The thesis mentions this but doesn't elaborate on why this matters for the total response.

#### C) Writing quality issues

1. **Section 3.3 appears cut off**: The file seems to truncate during the k·p model derivation subsection. The remaining content (BCD analytic derivation, KME derivation, multilayer discussion, device implications) may be incomplete.

2. **Redundant plan comments**: The file contains very extensive planning comments (>100 lines of commented-out material at the top). These should be cleaned up for the final thesis.

3. **Placeholder figure references**: Several figures are referenced by planned filenames but it's unclear if the actual PDF/EPS files exist and are correctly placed.

4. **Language consistency**: Some passages read like direct translations from English ("引发了关于...的激烈争论") while others are naturally written in Chinese. The register should be unified.

---

### Section 3.4 — Summary

**Status: Appears incomplete or truncated in the file**

The file ends during Section 3.3's k·p model derivation. If Section 3.4 exists, it was cut off. Based on the planning comments, it should:
- Summarize both mechanism chains (BCD vs INH)
- Emphasize the role of quantum geometry
- Provide outlook for future directions

---

## Priority Recommendations

### High Priority (essential missing content)
1. **Complete Section 3.3's k·p derivation** — the BCD and KME analytic results need to be shown explicitly
2. **Add the parameter-space Berry curvature argument** (k-λ space) connecting polarization and BCD — this is a unique theoretical insight from Paper 2's SM
3. **Add Section 3.4 summary** — currently missing or truncated
4. **Complete multilayer/surface discussion** with more physical detail

### Medium Priority (strengthen the thesis)
5. Add computational methods subsection(s) for both papers' DFT calculations
6. Include MnSe/MnTe results or at least a comparison table showing SOC trend
7. Show the symmetry analysis tables (allowed tensor components for each N direction; symmetry constraints for P₀ vs P±₁)
8. Expand the 8-band → 4-band downfolding derivation for Paper 1's k·p model
9. Add bulk ZrTe₅ competing phases summary (at least mention energetics)
10. Include the Fourier expansion derivation from C₃ symmetry constraints

### Low Priority (polish)
11. Clean up planning comments
12. Unify notation (σ_INH vs \INH macro; consistent use of $\Lambda$ vs $\lambda$)
13. Add MAE numerical values for MnS
14. Ensure all referenced figures exist as files
