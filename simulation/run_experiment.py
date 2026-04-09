#!/usr/bin/env python3
"""
Run the full experiment suite for the Forest Fire SOC project.
==============================================================
Generates all figures required for the LaTeX report.

Usage:
    python run_experiment.py [--grid-size 256] [--steps 10000]
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Ensure the script can find sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forest_fire import ForestFireModel
from analysis import (
    plot_fire_size_distribution,
    plot_power_law_fit,
    plot_density_timeseries,
    plot_connectivity_vs_avalanche,
    plot_grid_snapshots,
    plot_avalanche_timeseries,
)


def main():
    parser = argparse.ArgumentParser(description="Forest Fire Model Experiment Suite")
    parser.add_argument("--grid-size", "-L", type=int, default=256,
                        help="Side length of the square lattice (default: 256)")
    parser.add_argument("--steps", "-n", type=int, default=8000,
                        help="Number of simulation timesteps (default: 8000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for figures (default: ../report/figures)")
    args = parser.parse_args()

    L = args.grid_size
    steps = args.steps
    seed = args.seed
    outdir = args.outdir or str(Path(__file__).resolve().parent.parent / "report" / "figures")

    print("=" * 60)
    print("  Forest Fire Model — Experiment Suite")
    print("=" * 60)
    print(f"  Grid size : {L}×{L}")
    print(f"  Steps     : {steps}")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {outdir}")
    print("=" * 60)

    # ==================================================================
    # EXPERIMENT 1: Main simulation (baseline parameters)
    # ==================================================================
    print("\n[1/5] Running main simulation...")
    p_main, f_main = 0.05, 0.0001
    model = ForestFireModel(L=L, p=p_main, f=f_main, seed=seed)

    # Collect snapshots at specific timesteps
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

    # ==================================================================
    # EXPERIMENT 2: Generate figures
    # ==================================================================
    print("\n[2/5] Generating fire-size distribution plot...")
    plot_fire_size_distribution(avalanche_sizes, outdir)

    print("\n[3/5] Generating power-law fit analysis...")
    plot_power_law_fit(avalanche_sizes, outdir)

    print("\n[4/5] Generating density time series and snapshots...")
    plot_density_timeseries(model.density_history, outdir)
    plot_grid_snapshots(snapshots, outdir)
    plot_avalanche_timeseries(avalanche_sizes, outdir)

    # ==================================================================
    # EXPERIMENT 3: Connectivity sweep (vary p/f ratio)
    # ==================================================================
    print("\n[5/5] Connectivity sweep (varying p/f ratio)...")
    sweep_results = {}
    L_sweep = min(L, 128)  # Use smaller grid for sweep (speed)
    sweep_steps = min(steps, 5000)

    # Different p/f ratios
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
        n_events = len(sizes)
        print(f"({n_events} events)")

    plot_connectivity_vs_avalanche(sweep_results, outdir)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  All figures saved to:", outdir)
    print("=" * 60)
    print("\nDone! You can now compile the LaTeX report.")


if __name__ == "__main__":
    main()
