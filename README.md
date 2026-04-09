# Self-Organized Criticality in the Drossel--Schwabl Forest Fire Model

## Finite-Size Scaling, Information Theory, and Isomorphic Chemical Engineering Applications

A comprehensive computational investigation of self-organized criticality (SOC) using the Drossel--Schwabl forest fire cellular automaton, enhanced with finite-size scaling analysis, Transfer Entropy computation, and transdisciplinary applications.

**Course:** CLL 788 --- Complexity Science  
**Institution:** IIT Delhi  
**Author:** Ishaan Saini (2022CH11457)  
**Date:** April 2026

---

## Overview

This project applies complexity science to a **forest fire ecosystem**, modelled as a 2D cellular automaton. The system exhibits self-organized criticality: without any external parameter tuning, the forest naturally evolves toward a critical state where fire sizes follow a power-law distribution.

### Key Analyses

| Analysis | Description |
|----------|-------------|
| **SOC Characterization** | Noise (stochastic growth + lightning), avalanches (fire cascades), connectivity (percolation) |
| **Power-Law Fitting** | MLE-based exponent estimation with log-likelihood ratio tests |
| **Finite-Size Scaling** | Data collapse across L=64, 128, 256 confirming scale invariance |
| **Transfer Entropy** | Asymmetric causal information flow between density and avalanche variables |
| **Anisotropic Propagation** | Wind vector and topographical slope modifiers on fire spread |
| **Chemical Engineering Isomorphism** | Mapping to catalyst deactivation in packed-bed reactors |

---

## Project Structure

```
.
├── simulation/
│   ├── forest_fire.py       # Enhanced Drossel-Schwabl model (isotropic + anisotropic)
│   ├── analysis.py          # Full analysis suite (8 plot types + TE computation)
│   ├── visualize.py         # Animated GIF generation
│   ├── run_experiment.py    # Complete experiment runner (6 experiments)
│   └── requirements.txt     # Python dependencies
├── report/
│   ├── main.tex             # LaTeX manuscript (~20 pages, journal-style)
│   ├── main.pdf             # Compiled PDF
│   ├── references.bib       # 16 BibTeX entries
│   └── figures/             # 8 auto-generated publication-quality plots
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd simulation
pip install -r requirements.txt
```

### 2. Run the Full Experiment Suite

```bash
python run_experiment.py --grid-size 256 --steps 8000
```

This runs 6 experiments and generates 8 figures in `report/figures/`:
1. Main simulation (L=256, isotropic)
2. Fire-size distribution + power-law fit
3. Finite-size scaling (L=64, 128, 256) with data collapse
4. Connectivity sweep (p/f = 10 to 1000)
5. Transfer Entropy analysis
6. Anisotropic vs isotropic comparison

### 3. Compile the Report

```bash
cd report
tectonic main.tex
```

---

## Simulation Parameters

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Grid sizes (FSS) | L | 64, 128, 256 |
| Tree growth probability | p | 0.05 |
| Lightning probability | f | 10^-4 |
| p/f ratio | --- | 500 |
| Timesteps | T | 8,000 |
| Thermalization | --- | 1,000 (discarded) |
| Wind vector | **w** | (0.5, 0.3) |

---

## Key Results

- **Power-law distribution** of fire sizes confirmed via MLE
- **Finite-size scaling**: data collapse across grid sizes confirms scale invariance
- **Transfer Entropy**: asymmetric information flow validates causal feedback
- **Anisotropic propagation**: directional fire geometries under wind/slope
- **Chemical engineering isomorphism**: forest fire model maps directly to catalyst deactivation in packed-bed reactors

---

## References

- Bak, P., Tang, C., & Wiesenfeld, K. (1987). *Self-organized criticality*. Phys. Rev. Lett., 59(4), 381.
- Drossel, B. & Schwabl, F. (1992). *Self-organized critical forest-fire model*. Phys. Rev. Lett., 69(11), 1629.
- Malamud, B.D., Morein, G., & Turcotte, D.L. (1998). *Forest fires: An example of SOC*. Science, 281, 1840.
- Clauset, A., Shalizi, C.R., & Newman, M.E.J. (2009). *Power-law distributions in empirical data*. SIAM Rev., 51(4), 661.
- Schreiber, T. (2000). *Measuring information transfer*. Phys. Rev. Lett., 85(2), 461.
- Fogler, H.S. (2020). *Elements of Chemical Reaction Engineering*. Pearson, 6th ed.

---

## Acknowledgements

AI-assisted development using Google Gemini / Antigravity. All simulation results and analysis verified by the author.
