#!/usr/bin/env python3
"""
Enhanced Experiment Suite for the Forest Fire SOC Project
==========================================================
Runs:
  1. Main simulation (L=256, baseline parameters)
  2. Finite-size scaling sweep (L = 64, 128, 256)
  3. Connectivity sweep (varying p/f)
  4. Transfer Entropy analysis
  5. Anisotropic vs isotropic comparison
  6. Generate all figures

Usage:
    python run_experiment.py [--grid-size 256] [--steps 8000]
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forest_fire import ForestFireModel, run_multiscale
from analysis import (
    plot_fire_size_distribution,
    plot_power_law_fit,
    plot_density_timeseries,
    plot_connectivity_vs_avalanche,
    plot_grid_snapshots,
    plot_avalanche_timeseries,
    plot_finite_size_scaling,
    plot_transfer_entropy,
    plot_anisotropic_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Forest Fire SOC Experiment Suite")
    parser.add_argument("--grid-size", "-L", type=int, default=256)
    parser.add_argument("--steps", "-n", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    L = args.grid_size
    steps = args.steps
    seed = args.seed
    outdir = args.outdir or str(Path(__file__).resolve().parent.parent / "report" / "figures")

    print("=" * 65)
    print("  Forest Fire SOC Model --- Enhanced Experiment Suite")
    print("=" * 65)
    print(f"  Grid size : {L} x {L}")
    print(f"  Steps     : {steps}")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {outdir}")
    print("=" * 65)

    # ==================================================================
    # EXPERIMENT 1: Main simulation
    # ==================================================================
    print("\n[1/6] Running main simulation (isotropic)...")
    p_main, f_main = 0.05, 0.0001
    model = ForestFireModel(L=L, p=p_main, f=f_main, seed=seed)

    snapshot_times = [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]
    snapshots = [(0, model.snapshot())]

    for t in range(1, steps + 1):
        model.step()
        if t in snapshot_times:
            snapshots.append((t, model.snapshot()))

    avalanche_sizes = np.array(model.avalanche_sizes)
    print(f"  Total fire events: {len(avalanche_sizes)}")
    print(f"  Max fire size    : {avalanche_sizes.max() if len(avalanche_sizes) > 0 else 0}")
    print(f"  Final density    : {model.tree_density:.4f}")

    iso_snapshot = model.snapshot()

    # ==================================================================
    # EXPERIMENT 2: Generate basic figures
    # ==================================================================
    print("\n[2/6] Generating analysis figures...")
    plot_fire_size_distribution(avalanche_sizes, outdir)
    plot_power_law_fit(avalanche_sizes, outdir)
    plot_density_timeseries(model.density_history, outdir)
    plot_grid_snapshots(snapshots, outdir)
    plot_avalanche_timeseries(avalanche_sizes, outdir)

    # ==================================================================
    # EXPERIMENT 3: Finite-size scaling
    # ==================================================================
    print("\n[3/6] Finite-size scaling analysis...")
    fss_sizes = [64, 128, 256]
    fss_steps = min(steps, 5000)
    thermalization = 1000
    multiscale = run_multiscale(
        grid_sizes=fss_sizes, p=p_main, f=f_main,
        steps=fss_steps, seed=seed, thermalization=thermalization
    )
    plot_finite_size_scaling(multiscale, outdir)

    # ==================================================================
    # EXPERIMENT 4: Connectivity sweep
    # ==================================================================
    print("\n[4/6] Connectivity sweep (varying p/f ratio)...")
    sweep_results = {}
    L_sweep = min(L, 128)
    sweep_steps = min(steps, 4000)

    sweep_params = [
        (0.01, 0.001),    # p/f = 10
        (0.02, 0.001),    # p/f = 20
        (0.05, 0.001),    # p/f = 50
        (0.1,  0.001),    # p/f = 100
        (0.05, 0.0005),   # p/f = 100
        (0.05, 0.0002),   # p/f = 250
        (0.05, 0.0001),   # p/f = 500
        (0.05, 0.00005),  # p/f = 1000
    ]

    for p, f_val in sweep_params:
        print(f"    p={p}, f={f_val}, p/f={p/f_val:.0f} ...", end=" ")
        m = ForestFireModel(L=L_sweep, p=p, f=f_val, seed=seed)
        m.run(sweep_steps, progress=False)
        sizes = np.array(m.avalanche_sizes) if m.avalanche_sizes else np.array([])
        sweep_results[(p, f_val)] = sizes
        print(f"({len(sizes)} events)")

    plot_connectivity_vs_avalanche(sweep_results, outdir)

    # ==================================================================
    # EXPERIMENT 5: Transfer Entropy
    # ==================================================================
    print("\n[5/6] Transfer Entropy analysis...")
    plot_transfer_entropy(model.density_history, avalanche_sizes, outdir)

    # ==================================================================
    # EXPERIMENT 6: Anisotropic comparison
    # ==================================================================
    print("\n[6/6] Anisotropic propagation comparison...")
    # Generate a synthetic slope matrix (hill in one direction)
    slope = np.zeros((L, L))
    for i in range(L):
        slope[i, :] = i / L * 0.5  # gentle slope upward

    model_aniso = ForestFireModel(
        L=L, p=p_main, f=f_main, seed=seed,
        wind_vector=(0.5, 0.3), slope_matrix=slope, anisotropic=True
    )
    model_aniso.run(steps, progress=True)
    aniso_snapshot = model_aniso.snapshot()
    plot_anisotropic_comparison(iso_snapshot, aniso_snapshot, outdir)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 65)
    print("  All figures saved to:", outdir)
    print("=" * 65)
    print("\nDone! Compile the LaTeX report next.")


if __name__ == "__main__":
    main()
