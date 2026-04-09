# Self-Organized Criticality in the Drossel–Schwabl Forest Fire Model

A computational study of self-organized criticality (SOC) using the Drossel–Schwabl forest fire cellular automaton.

**Course:** CLL 788 — Complexity Science  
**Institution:** IIT Delhi  
**Date:** April 2026

---

## Overview

This project applies complexity science to a **forest fire ecosystem**, modelled as a 2D cellular automaton. The system exhibits self-organized criticality: without any external parameter tuning, the forest naturally evolves toward a critical state where fire sizes follow a power-law distribution.

### Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Noise** | Stochastic tree growth (prob. *p*) and random lightning ignition (prob. *f*) |
| **Avalanches** | Fire cascades through connected tree clusters |
| **Connectivity** | Tree density self-tunes toward the site-percolation threshold (~0.593) |
| **Power law** | Fire-size frequency distribution: P(s) ∝ s^(-α) |
| **SOC** | System reaches critical state without fine-tuning |

---

## Project Structure

```
├── simulation/
│   ├── forest_fire.py       # Core Drossel-Schwabl model
│   ├── analysis.py          # Power-law fitting + plotting
│   ├── visualize.py         # Animated GIF generation
│   ├── run_experiment.py    # Full experiment suite
│   └── requirements.txt     # Python dependencies
├── report/
│   ├── main.tex             # LaTeX manuscript
│   ├── references.bib       # Bibliography
│   └── figures/             # Auto-generated plots
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd simulation
pip install -r requirements.txt
```

### 2. Run the Full Experiment

```bash
python run_experiment.py --grid-size 256 --steps 8000
```

This generates all figures in `report/figures/`.

### 3. Generate Animation (Optional)

```bash
python visualize.py
```

### 4. Compile the Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Simulation Parameters

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Grid size | L | 256 |
| Tree growth probability | p | 0.05 |
| Lightning probability | f | 10⁻⁴ |
| p/f ratio | — | 500 |
| Timesteps | T | 8,000 |

---

## Results

The simulation produces the following outputs:

- **Fire-size distribution** (log-log plot with power-law fit)
- **Power-law exponent** estimation via MLE
- **Tree density time series** showing convergence to criticality
- **Connectivity sweep** varying p/f ratio
- **Grid snapshots** showing forest evolution

---

## References

- Bak, P., Tang, C., & Wiesenfeld, K. (1987). *Self-organized criticality*. Phys. Rev. Lett., 59(4), 381.
- Drossel, B. & Schwabl, F. (1992). *Self-organized critical forest-fire model*. Phys. Rev. Lett., 69(11), 1629.
- Malamud, B.D., Morein, G., & Turcotte, D.L. (1998). *Forest fires: An example of SOC*. Science, 281, 1840.
- Clauset, A., Shalizi, C.R., & Newman, M.E.J. (2009). *Power-law distributions in empirical data*. SIAM Rev., 51(4), 661.

---

## Acknowledgements

AI-assisted development using Google Gemini / Antigravity. All simulation results and analysis verified by the author.
