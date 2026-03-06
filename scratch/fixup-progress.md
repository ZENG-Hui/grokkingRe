# Thesis Fixup Progress

## Status: BLOCKED (proxy/server unreachable since ~12:35 UTC)

## Completed Analysis

### A. tab:chap03-INH-symmetry ✅ NO FIX NEEDED
- `\label{tab:chap03-INH-symmetry}` exists at line 207
- `\ref{tab:chap03-INH-symmetry}` used at line 219
- Both present and matching — this ref should compile fine

### B. 3 missing bib entries (chap03 line 427) — CONFIRMED MISSING
All confirmed absent from refs.bib (8411 lines, searched both halves):
1. `PhysRevLett.77.3865` — Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996) — PBE functional
   - Found complete entry in OriginPapers/BerryCurvatureNonlinearHall/slide_ref.bib line 1021
2. `DFT-D3` — Grimme et al., JCP 132, 154104 (2010) — DFT-D3 dispersion
   - NOT found in any OriginPapers bib. Must construct from scratch.
3. `PhysRevB.83.195131` — Klimeš, Bowler, Michaelides, PRB 83, 195131 (2011) — vdW-DF applied to solids
   - Found as key `RN33` in OriginPapers/BerryCurvatureNonlinearHall/slide_ref.bib line 159

### C. 5 missing bib entries (chap05 line 67-69) — CONFIRMED MISSING
All confirmed absent from refs.bib, but ALL found in OriginPapers/HighFieldDynamics/ref.bib:
1. `LEO1992943` — Leo et al., SSC 84, 943 (1992) — Bloch oscillations (line 820)
2. `PhysRevLett.64.3167` — Beltram et al., PRL 64, 3167 (1990) — field-induced localization (line 928)
3. `cai2023signatures` — Cai et al., Nature 622, 63 (2023) — FQAH in twisted MoTe2 (line 1046)
4. `cao2018correlated` — Cao et al., Nature 556, 80 (2018) — correlated insulator MATBG (line 1036)
5. `cao2018unconventional` — Cao et al., Nature 556, 43 (2018) — unconventional SC MATBG (line 1026)

## Remaining Work

### D. Write 8 BibTeX entries to refs.bib
Need to append all 8 entries. BibTeX content prepared (see below).

### E. Compile and verify
Run `cd .../projects/Thesis && bash build.sh`

## Prepared BibTeX entries (ready to append)

```bibtex
@article{PhysRevLett.77.3865,
  title = {Generalized Gradient Approximation Made Simple},
  author = {Perdew, John P. and Burke, Kieron and Ernzerhof, Matthias},
  journal = {Phys. Rev. Lett.},
  volume = {77},
  issue = {18},
  pages = {3865--3868},
  year = {1996},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.77.3865},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.77.3865}
}

@article{DFT-D3,
  title = {A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu},
  author = {Grimme, Stefan and Antony, Jens and Ehrlich, Stephan and Krieg, Helge},
  journal = {J. Chem. Phys.},
  volume = {132},
  number = {15},
  pages = {154104},
  year = {2010},
  doi = {10.1063/1.3382344}
}

@article{PhysRevB.83.195131,
  title = {Van der Waals density functionals applied to solids},
  author = {Klimes, Jiri and Bowler, David R. and Michaelides, Angelos},
  journal = {Phys. Rev. B},
  volume = {83},
  number = {19},
  pages = {195131},
  year = {2011},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.83.195131}
}

@article{LEO1992943,
  title = {Observation of Bloch oscillations in a semiconductor superlattice},
  journal = {Solid State Commun.},
  volume = {84},
  number = {10},
  pages = {943--946},
  year = {1992},
  issn = {0038-1098},
  doi = {10.1016/0038-1098(92)90798-E},
  author = {Leo, Karl and Haring Bolivar, Peter and Bruggemann, Frank and Schwedler, Ralf and Kohler, Klaus}
}

@article{PhysRevLett.64.3167,
  title = {Scattering-controlled transmission resonances and negative differential conductance by field-induced localization in superlattices},
  author = {Beltram, Fabio and Capasso, Federico and Sivco, Deborah L. and Hutchinson, Albert L. and Chu, Sung-Nee G. and Cho, Alfred Y.},
  journal = {Phys. Rev. Lett.},
  volume = {64},
  issue = {26},
  pages = {3167--3170},
  year = {1990},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.64.3167}
}

@article{cai2023signatures,
  title = {Signatures of fractional quantum anomalous Hall states in twisted MoTe2},
  author = {Cai, Jiaqi and Anderson, Eric and Wang, Chong and Zhang, Xiaowei and Liu, Xiaoyu and Holtzmann, William and Zhang, Yinong and Fan, Fengren and Taniguchi, Takashi and Watanabe, Kenji and others},
  journal = {Nature},
  volume = {622},
  number = {7981},
  pages = {63--68},
  year = {2023},
  publisher = {Nature Publishing Group UK London}
}

@article{cao2018correlated,
  title = {Correlated insulator behaviour at half-filling in magic-angle graphene superlattices},
  author = {Cao, Yuan and Fatemi, Valla and Demir, Ahmet and Fang, Shiang and Tomarken, Spencer L and Luo, Jason Y and Sanchez-Yamagishi, Javier D and Watanabe, Kenji and Taniguchi, Takashi and Kaxiras, Efthimios and others},
  journal = {Nature},
  volume = {556},
  number = {7699},
  pages = {80--84},
  year = {2018},
  publisher = {Nature Publishing Group UK London}
}

@article{cao2018unconventional,
  title = {Unconventional superconductivity in magic-angle graphene superlattices},
  author = {Cao, Yuan and Fatemi, Valla and Fang, Shiang and Watanabe, Kenji and Taniguchi, Takashi and Kaxiras, Efthimios and Jarillo-Herrero, Pablo},
  journal = {Nature},
  volume = {556},
  number = {7699},
  pages = {43--50},
  year = {2018},
  publisher = {Nature Publishing Group}
}
```
