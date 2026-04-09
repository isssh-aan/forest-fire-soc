"""
Analysis module for the Forest Fire Model
==========================================
Produces publication-quality figures for the project report:
    1. Fire-size frequency distribution (log-log)
    2. Power-law fit with exponent estimation
    3. Tree density time series
    4. Connectivity (p/f ratio) vs avalanche statistics
    5. Grid snapshot panels
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Optional: nicer plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})


# ---- Colour map for grid snapshots ----
FOREST_CMAP = ListedColormap(["#8B6914", "#228B22", "#FF4500"])  # empty, tree, burning
FOREST_CMAP.set_bad("white")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ======================================================================
# 1. Fire-size frequency distribution
# ======================================================================
def plot_fire_size_distribution(avalanche_sizes: np.ndarray, outdir: str,
                                 fit: bool = True):
    """Log-log histogram of avalanche (fire) sizes with optional power-law fit."""
    sizes = avalanche_sizes[avalanche_sizes > 0]
    if len(sizes) < 10:
        print("  [WARN] Too few avalanche events for distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Logarithmic binning
    log_min = np.log10(sizes.min())
    log_max = np.log10(sizes.max())
    bins = np.logspace(log_min, log_max, num=40)
    counts, edges = np.histogram(sizes, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    nonzero = counts > 0
    # Normalise to probability density
    widths = np.diff(edges)
    density = counts / (widths * sizes.size)

    ax.scatter(centres[nonzero], density[nonzero], s=20, c="#2E86AB",
               edgecolors="k", linewidths=0.3, zorder=3, label="Simulation data")

    # Power-law fit
    if fit:
        try:
            import powerlaw
            fit_result = powerlaw.Fit(sizes, discrete=True, verbose=False)
            alpha = fit_result.power_law.alpha
            xmin = fit_result.power_law.xmin

            # Overlay fit line
            xs = np.logspace(np.log10(xmin), log_max, 200)
            ys = xs ** (-alpha)
            # Scale to match data
            idx = np.argmin(np.abs(centres - xmin))
            if nonzero[idx]:
                scale = density[nonzero][np.argmin(np.abs(centres[nonzero] - xmin))] / (xmin ** (-alpha))
                ax.plot(xs, ys * scale, "r--", lw=1.8,
                        label=rf"Power-law fit ($\alpha$ = {alpha:.2f}, $x_{{\min}}$ = {xmin:.0f})")
        except ImportError:
            print("  [INFO] `powerlaw` package not installed; skipping fit.")
        except Exception as e:
            print(f"  [WARN] Power-law fit failed: {e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fire size $s$ (trees burned)")
    ax.set_ylabel("Probability density $P(s)$")
    ax.set_title("Fire-Size Frequency Distribution")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "fire_size_distribution.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")
    return outpath


# ======================================================================
# 2. Power-law fit details
# ======================================================================
def plot_power_law_fit(avalanche_sizes: np.ndarray, outdir: str):
    """Detailed power-law analysis with comparison to log-normal."""
    sizes = avalanche_sizes[avalanche_sizes > 0]
    if len(sizes) < 30:
        print("  [WARN] Too few events for detailed power-law fit.")
        return

    try:
        import powerlaw
    except ImportError:
        print("  [INFO] `powerlaw` package not installed; skipping.")
        return

    fit_result = powerlaw.Fit(sizes, discrete=True, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) CCDF
    ax = axes[0]
    fit_result.plot_ccdf(ax=ax, color="#2E86AB", linewidth=1.5, label="Empirical CCDF")
    fit_result.power_law.plot_ccdf(ax=ax, color="r", linestyle="--", linewidth=1.5,
                                    label=rf"Power-law ($\alpha$={fit_result.alpha:.2f})")
    ax.set_title("Complementary CDF")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # (b) PDF comparison
    ax = axes[1]
    fit_result.plot_pdf(ax=ax, color="#2E86AB", linewidth=1.5, label="Empirical PDF")
    fit_result.power_law.plot_pdf(ax=ax, color="r", linestyle="--", linewidth=1.5,
                                   label="Power-law fit")
    ax.set_title("Probability Density")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Power-Law Analysis of Fire Sizes", fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = ensure_dir(outdir) / "power_law_fit.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")

    # Print summary
    R, p_val = fit_result.distribution_compare("power_law", "lognormal")
    print(f"    α = {fit_result.alpha:.3f}")
    print(f"    x_min = {fit_result.xmin}")
    print(f"    Power-law vs log-normal: R = {R:.3f}, p = {p_val:.4f}")


# ======================================================================
# 3. Tree density time series
# ======================================================================
def plot_density_timeseries(density_history: list[float], outdir: str):
    """Tree density vs time — shows convergence to critical density."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = np.arange(len(density_history))
    ax.plot(t, density_history, lw=0.6, color="#228B22", alpha=0.8)

    # Running average
    window = max(1, len(density_history) // 100)
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(density_history, kernel, mode="valid")
        ax.plot(t[:len(smooth)], smooth, lw=2, color="#B22222",
                label=f"Running avg (window={window})")

    # Percolation threshold reference
    ax.axhline(0.5927, ls="--", color="gray", lw=1, alpha=0.7,
               label="Site-percolation threshold (0.593)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Tree density $\\rho$")
    ax.set_title("Tree Density Convergence to Critical State")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "tree_density_time_series.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")


# ======================================================================
# 4. Connectivity vs avalanche statistics
# ======================================================================
def plot_connectivity_vs_avalanche(results: dict, outdir: str):
    """
    Plot how the p/f ratio (proxy for connectivity) affects avalanche statistics.

    Parameters
    ----------
    results : dict
        Keys are (p, f) tuples; values are arrays of avalanche sizes.
    """
    if not results:
        print("  [WARN] No multi-parameter results to plot.")
        return

    ratios = []
    mean_sizes = []
    max_sizes = []
    std_sizes = []

    for (p, f_val), sizes in sorted(results.items(), key=lambda x: x[0][0] / x[0][1]):
        if len(sizes) == 0:
            continue
        ratio = p / f_val
        ratios.append(ratio)
        mean_sizes.append(np.mean(sizes))
        max_sizes.append(np.max(sizes))
        std_sizes.append(np.std(sizes))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ratios, mean_sizes, "o-", color="#2E86AB", markersize=6, lw=1.5)
    ax.set_xlabel("$p / f$ ratio (connectivity proxy)")
    ax.set_ylabel("Mean avalanche size")
    ax.set_title("Mean Avalanche Size vs Connectivity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(ratios, max_sizes, "s-", color="#E8333F", markersize=6, lw=1.5)
    ax.set_xlabel("$p / f$ ratio (connectivity proxy)")
    ax.set_ylabel("Max avalanche size")
    ax.set_title("Maximum Avalanche Size vs Connectivity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Effect of Connectivity on Avalanche Behaviour", fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = ensure_dir(outdir) / "connectivity_vs_avalanche.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")


# ======================================================================
# 5. Grid snapshots
# ======================================================================
def plot_grid_snapshots(snapshots: list[tuple[int, np.ndarray]], outdir: str):
    """
    Panel of grid snapshots at different timesteps.

    Parameters
    ----------
    snapshots : list of (timestep, grid_array) tuples
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (t, grid) in zip(axes, snapshots):
        ax.imshow(grid, cmap=FOREST_CMAP, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.set_title(f"$t = {t}$")
        ax.set_xticks([])
        ax.set_yticks([])

    # Custom legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor="#8B6914", edgecolor="k", label="Empty"),
        mpatches.Patch(facecolor="#228B22", edgecolor="k", label="Tree"),
        mpatches.Patch(facecolor="#FF4500", edgecolor="k", label="Burning"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Forest Grid Evolution", fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = ensure_dir(outdir) / "grid_snapshots.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")


# ======================================================================
# 6. Avalanche size over time (time series)
# ======================================================================
def plot_avalanche_timeseries(avalanche_sizes: np.ndarray, outdir: str):
    """Avalanche sizes vs event index — shows bursty dynamics."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(avalanche_sizes)), avalanche_sizes,
           width=1.0, color="#E8333F", edgecolor="none", alpha=0.7)
    ax.set_xlabel("Fire event index")
    ax.set_ylabel("Fire size (trees burned)")
    ax.set_title("Avalanche Sizes Over Time")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "avalanche_timeseries.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath}")


if __name__ == "__main__":
    # Quick test with dummy data
    dummy = np.random.pareto(1.2, 500).astype(int) + 1
    plot_fire_size_distribution(dummy, "/tmp/test_figs")
    plot_density_timeseries(list(np.random.uniform(0.3, 0.7, 300)), "/tmp/test_figs")
    print("Analysis module self-test complete.")
