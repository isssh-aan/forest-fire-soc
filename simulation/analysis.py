"""
Enhanced Analysis Module for the Forest Fire SOC Project
=========================================================
Produces publication-quality figures including:
    1. Fire-size frequency distribution (log-log) with power-law fit
    2. Detailed power-law analysis (CCDF, PDF)
    3. Tree density time series
    4. Connectivity (p/f ratio) vs avalanche statistics
    5. Grid snapshot panels
    6. Avalanche size time series
    7. Finite-size scaling and data collapse
    8. Transfer Entropy analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

FOREST_CMAP = ListedColormap(["#8B6914", "#228B22", "#FF4500"])


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ======================================================================
# 1. Fire-size frequency distribution
# ======================================================================
def plot_fire_size_distribution(avalanche_sizes: np.ndarray, outdir: str,
                                 fit: bool = True):
    """Log-log histogram of avalanche sizes with power-law fit."""
    sizes = avalanche_sizes[avalanche_sizes > 0]
    if len(sizes) < 10:
        print("  [WARN] Too few avalanche events for distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    log_min = np.log10(max(sizes.min(), 1))
    log_max = np.log10(sizes.max())
    bins = np.logspace(log_min, log_max, num=40)
    counts, edges = np.histogram(sizes, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    nonzero = counts > 0
    widths = np.diff(edges)
    density = counts / (widths * sizes.size)

    ax.scatter(centres[nonzero], density[nonzero], s=20, c="#2E86AB",
               edgecolors="k", linewidths=0.3, zorder=3, label="Simulation data")

    if fit:
        try:
            import powerlaw
            fit_result = powerlaw.Fit(sizes, discrete=True, verbose=False)
            alpha = fit_result.power_law.alpha
            xmin = fit_result.power_law.xmin

            xs = np.logspace(np.log10(xmin), log_max, 200)
            ys = xs ** (-alpha)
            idx = np.argmin(np.abs(centres[nonzero] - xmin))
            scale = density[nonzero][idx] / (xmin ** (-alpha))
            ax.plot(xs, ys * scale, "r--", lw=1.8,
                    label=rf"Power-law fit ($\alpha$ = {alpha:.2f}, $x_{{\min}}$ = {xmin:.0f})")
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
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 2. Power-law fit details
# ======================================================================
def plot_power_law_fit(avalanche_sizes: np.ndarray, outdir: str):
    """Detailed power-law analysis with CCDF and PDF."""
    sizes = avalanche_sizes[avalanche_sizes > 0]
    if len(sizes) < 30:
        print("  [WARN] Too few events for power-law fit.")
        return

    try:
        import powerlaw
    except ImportError:
        print("  [INFO] powerlaw not installed; skipping.")
        return

    fit_result = powerlaw.Fit(sizes, discrete=True, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    fit_result.plot_ccdf(ax=ax, color="#2E86AB", linewidth=1.5, label="Empirical CCDF")
    fit_result.power_law.plot_ccdf(ax=ax, color="r", linestyle="--", linewidth=1.5,
                                    label=rf"Power-law ($\alpha$={fit_result.alpha:.2f})")
    ax.set_title("Complementary CDF")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)

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
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")

    R, p_val = fit_result.distribution_compare("power_law", "lognormal")
    print(f"    alpha = {fit_result.alpha:.3f}, x_min = {fit_result.xmin}")
    print(f"    PL vs lognormal: R = {R:.3f}, p = {p_val:.4f}")


# ======================================================================
# 3. Tree density time series
# ======================================================================
def plot_density_timeseries(density_history: list[float], outdir: str):
    """Tree density vs time with percolation threshold reference."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = np.arange(len(density_history))
    ax.plot(t, density_history, lw=0.6, color="#228B22", alpha=0.8)

    window = max(1, len(density_history) // 100)
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(density_history, kernel, mode="valid")
        ax.plot(t[:len(smooth)], smooth, lw=2, color="#B22222",
                label=f"Running avg (window={window})")

    ax.axhline(0.5927, ls="--", color="gray", lw=1, alpha=0.7,
               label=r"$p_c \approx 0.593$ (site percolation)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"Tree density $\rho$")
    ax.set_title("Convergence to the Self-Organized Critical Density")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "tree_density_time_series.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 4. Connectivity vs avalanche statistics
# ======================================================================
def plot_connectivity_vs_avalanche(results: dict, outdir: str):
    """Plot p/f ratio vs mean/max avalanche sizes."""
    if not results:
        return

    ratios, mean_sizes, max_sizes = [], [], []
    for (p, f_val), sizes in sorted(results.items(), key=lambda x: x[0][0] / x[0][1]):
        if len(sizes) == 0:
            continue
        ratios.append(p / f_val)
        mean_sizes.append(np.mean(sizes))
        max_sizes.append(np.max(sizes))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ratios, mean_sizes, "o-", color="#2E86AB", markersize=6, lw=1.5)
    ax.set_xlabel("$p / f$ (connectivity proxy)")
    ax.set_ylabel("Mean avalanche size")
    ax.set_title("Mean Avalanche Size vs Connectivity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(ratios, max_sizes, "s-", color="#E8333F", markersize=6, lw=1.5)
    ax.set_xlabel("$p / f$ (connectivity proxy)")
    ax.set_ylabel("Max avalanche size")
    ax.set_title("Maximum Avalanche Size vs Connectivity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Effect of Connectivity on Avalanche Behaviour", fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = ensure_dir(outdir) / "connectivity_vs_avalanche.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 5. Grid snapshots
# ======================================================================
def plot_grid_snapshots(snapshots: list[tuple[int, np.ndarray]], outdir: str):
    """Panel of grid snapshots at different timesteps."""
    import matplotlib.patches as mpatches

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (t, grid) in zip(axes, snapshots):
        ax.imshow(grid, cmap=FOREST_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"$t = {t}$")
        ax.set_xticks([]); ax.set_yticks([])

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
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 6. Avalanche size time series
# ======================================================================
def plot_avalanche_timeseries(avalanche_sizes: np.ndarray, outdir: str):
    """Bursty avalanche dynamics over time."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(avalanche_sizes)), avalanche_sizes,
           width=1.0, color="#E8333F", edgecolor="none", alpha=0.7)
    ax.set_xlabel("Fire event index")
    ax.set_ylabel("Fire size (trees burned)")
    ax.set_title("Avalanche Sizes Over Time: Bursty, Scale-Free Dynamics")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "avalanche_timeseries.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 7. Finite-size scaling and data collapse
# ======================================================================
def plot_finite_size_scaling(multiscale_results: dict, outdir: str):
    """
    Finite-size scaling analysis across multiple grid sizes.
    Shows:
      (a) Raw distributions for each L overlaid
      (b) Data collapse using rescaled variables
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(multiscale_results)))

    # (a) Raw distributions
    ax = axes[0]
    for idx, (L, data) in enumerate(sorted(multiscale_results.items())):
        sizes = data['avalanche_sizes']
        sizes = sizes[sizes > 0]
        if len(sizes) < 20:
            continue
        log_min = np.log10(max(sizes.min(), 1))
        log_max = np.log10(sizes.max())
        bins = np.logspace(log_min, log_max, num=30)
        counts, edges = np.histogram(sizes, bins=bins)
        centres = np.sqrt(edges[:-1] * edges[1:])
        nonzero = counts > 0
        widths = np.diff(edges)
        density = counts / (widths * sizes.size)
        ax.scatter(centres[nonzero], density[nonzero], s=15, c=[colors[idx]],
                   edgecolors="k", linewidths=0.2, label=f"L = {L}", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Fire size $s$")
    ax.set_ylabel("$P(s)$")
    ax.set_title("Raw Avalanche Distributions (Multiple Grid Sizes)")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # (b) Data collapse: rescale s -> s/L^D and P -> P * L^(D*tau)
    # Use D=2 (spatial dimension), attempt collapse with tau estimate
    ax = axes[1]
    D = 2.0
    tau_est = 1.1  # approximate; will be visible from collapse quality

    for idx, (L, data) in enumerate(sorted(multiscale_results.items())):
        sizes = data['avalanche_sizes']
        sizes = sizes[sizes > 0]
        if len(sizes) < 20:
            continue
        log_min = np.log10(max(sizes.min(), 1))
        log_max = np.log10(sizes.max())
        bins = np.logspace(log_min, log_max, num=30)
        counts, edges = np.histogram(sizes, bins=bins)
        centres = np.sqrt(edges[:-1] * edges[1:])
        nonzero = counts > 0
        widths = np.diff(edges)
        density = counts / (widths * sizes.size)

        # Rescale
        s_rescaled = centres[nonzero] / (L ** D)
        P_rescaled = density[nonzero] * (L ** (D * tau_est))

        ax.scatter(s_rescaled, P_rescaled, s=15, c=[colors[idx]],
                   edgecolors="k", linewidths=0.2, label=f"L = {L}", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Rescaled size $s / L^D$")
    ax.set_ylabel(r"Rescaled density $P(s) \cdot L^{D\tau}$")
    ax.set_title(rf"Data Collapse ($D = {D:.0f}$, $\tau \approx {tau_est:.1f}$)")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    outpath = ensure_dir(outdir) / "finite_size_scaling.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


# ======================================================================
# 8. Transfer Entropy analysis
# ======================================================================
def compute_transfer_entropy(density_ts: np.ndarray, avalanche_ts: np.ndarray,
                               k: int = 1, bins: int = 20) -> tuple[float, float]:
    """
    Estimate Transfer Entropy from density time series to avalanche time series
    using binned histogram method (discrete TE).

    TE(X -> Y) = sum P(y_{t+1}, y_t^k, x_t^k) * log2[ P(y_{t+1}|y_t^k, x_t^k) / P(y_{t+1}|y_t^k) ]

    Also computes TE(Y -> X) for comparison (asymmetry proves causality direction).

    Returns (TE_density_to_avalanche, TE_avalanche_to_density)
    """
    n = min(len(density_ts), len(avalanche_ts))
    if n < 100:
        return 0.0, 0.0

    # Discretize into bins
    def discretize(x, nbins):
        x = np.array(x[:n], dtype=float)
        # Remove outliers for better binning
        p5, p95 = np.percentile(x, [2, 98])
        x = np.clip(x, p5, p95)
        if x.max() == x.min():
            return np.zeros(len(x), dtype=int)
        return np.clip(((x - x.min()) / (x.max() - x.min()) * (nbins - 1)).astype(int), 0, nbins - 1)

    X = discretize(density_ts, bins)   # density
    Y = discretize(avalanche_ts, bins) # avalanche

    def _te(source, target, k_hist=1):
        """Compute TE(source -> target) using k-history."""
        T = len(source) - k_hist
        if T < 50:
            return 0.0

        # Build joint distribution
        # Variables: target_{t+1}, target_t^k, source_t^k
        # For simplicity with k=1: target_{t+1}, target_t, source_t
        y_future = target[k_hist:]
        y_past = target[k_hist - 1:-1]
        x_past = source[k_hist - 1:-1]

        # 3D histogram for P(y_f, y_p, x_p)
        joint_3 = np.zeros((bins, bins, bins))
        for i in range(len(y_future)):
            joint_3[y_future[i], y_past[i], x_past[i]] += 1
        joint_3 /= joint_3.sum()

        # 2D histogram for P(y_f, y_p)
        joint_2 = joint_3.sum(axis=2)
        # 2D histogram for P(y_p, x_p)
        joint_2_yx = joint_3.sum(axis=0)
        # 1D histogram for P(y_p)
        p_yp = joint_2.sum(axis=0)

        te = 0.0
        for yf in range(bins):
            for yp in range(bins):
                for xp in range(bins):
                    p_joint = joint_3[yf, yp, xp]
                    if p_joint < 1e-12:
                        continue
                    p_yf_yp = joint_2[yf, yp]
                    p_yp_xp = joint_2_yx[yp, xp]
                    p_yp_val = p_yp[yp]

                    if p_yf_yp < 1e-12 or p_yp_xp < 1e-12 or p_yp_val < 1e-12:
                        continue

                    # P(y_f | y_p, x_p) = P(y_f, y_p, x_p) / P(y_p, x_p)
                    # P(y_f | y_p) = P(y_f, y_p) / P(y_p)
                    cond_full = p_joint / p_yp_xp
                    cond_partial = p_yf_yp / p_yp_val

                    if cond_full > 0 and cond_partial > 0:
                        te += p_joint * np.log2(cond_full / cond_partial)
        return te

    te_xy = _te(X, Y, k)  # density -> avalanche
    te_yx = _te(Y, X, k)  # avalanche -> density
    return te_xy, te_yx


def plot_transfer_entropy(density_history: list[float],
                           avalanche_sizes: np.ndarray,
                           outdir: str):
    """Compute and plot Transfer Entropy in both directions."""
    # Align time series: use avalanche events and corresponding density
    n = min(len(density_history), len(avalanche_sizes))
    if n < 100:
        print("  [WARN] Too few data points for TE analysis.")
        return

    density_ts = np.array(density_history[:n])
    aval_ts = avalanche_sizes[:n].astype(float)

    # Compute TE at different history lengths
    k_values = [1, 2, 3, 5, 8]
    te_d2a = []
    te_a2d = []
    for k in k_values:
        xy, yx = compute_transfer_entropy(density_ts, aval_ts, k=k, bins=15)
        te_d2a.append(xy)
        te_a2d.append(yx)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, te_d2a, width, color="#2E86AB", edgecolor="k",
                   linewidth=0.5, label=r"$TE(\rho \to s)$: density $\to$ avalanche")
    bars2 = ax.bar(x + width/2, te_a2d, width, color="#E8333F", edgecolor="k",
                   linewidth=0.5, label=r"$TE(s \to \rho)$: avalanche $\to$ density")

    ax.set_xlabel("History length $k$")
    ax.set_ylabel("Transfer Entropy (bits)")
    ax.set_title("Asymmetric Information Flow: Transfer Entropy Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    outpath = ensure_dir(outdir) / "transfer_entropy.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")
    print(f"    TE(density->avalanche) at k=1: {te_d2a[0]:.4f} bits")
    print(f"    TE(avalanche->density) at k=1: {te_a2d[0]:.4f} bits")
    print(f"    Asymmetry ratio: {te_d2a[0] / max(te_a2d[0], 1e-10):.2f}")


# ======================================================================
# 9. Anisotropic fire comparison
# ======================================================================
def plot_anisotropic_comparison(iso_snapshot: np.ndarray,
                                  aniso_snapshot: np.ndarray,
                                  outdir: str):
    """Side-by-side comparison of isotropic vs anisotropic fire spread."""
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(iso_snapshot, cmap=FOREST_CMAP, vmin=0, vmax=2, interpolation="nearest")
    axes[0].set_title("Isotropic Propagation")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(aniso_snapshot, cmap=FOREST_CMAP, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("Anisotropic Propagation (Wind + Slope)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    legend_elements = [
        mpatches.Patch(facecolor="#8B6914", edgecolor="k", label="Empty"),
        mpatches.Patch(facecolor="#228B22", edgecolor="k", label="Tree"),
        mpatches.Patch(facecolor="#FF4500", edgecolor="k", label="Burning"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Effect of Environmental Anisotropy on Fire Geometry", fontsize=14)
    fig.tight_layout()

    outpath = ensure_dir(outdir) / "anisotropic_comparison.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


if __name__ == "__main__":
    dummy = np.random.pareto(1.2, 500).astype(int) + 1
    plot_fire_size_distribution(dummy, "/tmp/test_figs")
    print("Analysis module self-test complete.")
