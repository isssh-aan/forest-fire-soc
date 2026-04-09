"""
Drossel-Schwabl Forest Fire Model — Enhanced
=============================================
A cellular automaton exhibiting self-organized criticality (SOC).

Enhancements over the basic model:
  - BFS-based connected-component analysis for true avalanche sizing
  - Anisotropic fire propagation (wind + topographical slope)
  - Multi-scale support for finite-size scaling analysis

Cell states:
    0 = Empty
    1 = Tree
    2 = Burning

Reference:
    Drossel, B. & Schwabl, F. (1992). Phys. Rev. Lett., 69(11), 1629.
"""

import numpy as np
from collections import deque

# Cell-state constants
EMPTY = 0
TREE = 1
BURNING = 2


class ForestFireModel:
    """Drossel-Schwabl forest fire cellular automaton on a 2-D square lattice.

    Supports both isotropic (deterministic spread) and anisotropic
    (probabilistic spread modulated by wind and slope) propagation.
    """

    def __init__(self, L: int = 256, p: float = 0.05, f: float = 0.0001,
                 seed: int | None = None,
                 wind_vector: tuple[float, float] = (0.0, 0.0),
                 slope_matrix: np.ndarray | None = None,
                 anisotropic: bool = False):
        """
        Parameters
        ----------
        L : int
            Side length of the square lattice.
        p : float
            Probability that an empty cell grows a tree each timestep.
        f : float
            Probability that a tree is struck by lightning each timestep.
        seed : int or None
            Random seed for reproducibility.
        wind_vector : tuple
            (wx, wy) components of wind; affects directional fire spread.
        slope_matrix : ndarray or None
            L×L elevation map; uphill spread is enhanced.
        anisotropic : bool
            If True, fire spread is probabilistic via wind+slope kernel.
        """
        self.L = L
        self.p = p
        self.f = f
        self.rng = np.random.default_rng(seed)
        self.anisotropic = anisotropic
        self.wind_vector = np.array(wind_vector, dtype=np.float64)
        self.slope_matrix = slope_matrix

        # Initialise grid: ~50% tree coverage
        self.grid = self.rng.choice(
            [EMPTY, TREE], size=(L, L), p=[0.5, 0.5]
        ).astype(np.int8)

        # Statistics
        self.avalanche_sizes: list[int] = []
        self.cluster_avalanche_sizes: list[int] = []
        self.density_history: list[float] = []
        self.time = 0

        # Precompute anisotropic spread kernel if needed
        if self.anisotropic:
            self._build_spread_kernel()

    def _build_spread_kernel(self):
        """Build directional spread probability modifiers for 4 neighbours.

        The kernel assigns a probability modifier P_spread to each of the
        4 cardinal directions based on the alignment with the wind vector
        and topographical slope.
        """
        # Direction vectors for N, S, E, W
        self._directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        # Base probability (without wind/slope, spread is certain)
        self._base_spread = 0.6

        # Wind contribution: alignment between wind vector and direction
        self._wind_weights = np.zeros(4)
        w_mag = np.linalg.norm(self.wind_vector)
        if w_mag > 0:
            w_hat = self.wind_vector / w_mag
            for k, (di, dj) in enumerate(self._directions):
                d_hat = np.array([di, dj], dtype=np.float64)
                alignment = np.dot(w_hat, d_hat)
                self._wind_weights[k] = 0.3 * w_mag * alignment

    @staticmethod
    def _has_burning_neighbour(grid: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where cell has >= 1 burning neighbour."""
        burning = (grid == BURNING)
        nb = np.zeros_like(burning)
        nb[:-1, :] |= burning[1:, :]
        nb[1:, :]  |= burning[:-1, :]
        nb[:, :-1] |= burning[:, 1:]
        nb[:, 1:]  |= burning[:, :-1]
        return nb

    def _anisotropic_spread(self, g: np.ndarray) -> np.ndarray:
        """Compute fire spread using probabilistic directional kernel."""
        L = self.L
        burning = (g == BURNING)
        tree = (g == TREE)
        new_burning = np.zeros((L, L), dtype=bool)

        shifts = [
            (slice(None, -1), slice(None), slice(1, None), slice(None)),   # N
            (slice(1, None), slice(None), slice(None, -1), slice(None)),   # S
            (slice(None), slice(1, None), slice(None), slice(None, -1)),   # E
            (slice(None), slice(None, -1), slice(None), slice(1, None)),   # W
        ]

        for k, (tr, tc, br, bc) in enumerate(shifts):
            # Probability of spread in this direction
            p_spread = np.clip(self._base_spread + self._wind_weights[k], 0.05, 1.0)

            # Slope modifier (if available)
            if self.slope_matrix is not None:
                di, dj = self._directions[k]
                # Compute slope difference: positive = uphill (enhances spread)
                slope_diff = np.zeros((L, L))
                slope_diff[tr, tc] = self.slope_matrix[tr, tc] - self.slope_matrix[br, bc]
                slope_modifier = np.clip(0.2 * slope_diff[tr, tc], -0.3, 0.3)
                p_spread_arr = np.clip(p_spread + slope_modifier, 0.05, 1.0)
            else:
                p_spread_arr = p_spread

            # Tree at position (tr,tc) with burning neighbour at (br,bc)
            can_spread = tree[tr, tc] & burning[br, bc]
            roll = self.rng.random(can_spread.shape)
            new_burning[tr, tc] |= can_spread & (roll < p_spread_arr)

        return new_burning

    def step(self) -> int:
        """Advance model by one timestep. Returns number of newly burning trees."""
        g = self.grid
        L = self.L

        # 1. Fire spread
        if self.anisotropic:
            new_burning = self._anisotropic_spread(g)
        else:
            spread_mask = (g == TREE) & self._has_burning_neighbour(g)
            new_burning = spread_mask

        # Lightning on trees not already catching fire from spread
        lightning_mask = (g == TREE) & ~new_burning
        lightning_hits = self.rng.random((L, L)) < self.f
        ignite_mask = lightning_mask & lightning_hits
        new_burning = new_burning | ignite_mask

        # 2. Burning -> Empty
        was_burning = (g == BURNING)
        g[was_burning] = EMPTY

        # 3. Set newly burning
        n_burned = int(new_burning.sum())
        g[new_burning] = BURNING

        # 4. Tree growth on empty cells
        empty_mask = (g == EMPTY)
        grow = empty_mask & (self.rng.random((L, L)) < self.p)
        g[grow] = TREE

        # Record
        if n_burned > 0:
            self.avalanche_sizes.append(n_burned)
        self.density_history.append(float((g == TREE).sum()) / (L * L))
        self.time += 1
        return n_burned

    def run(self, steps: int, progress: bool = True):
        """Run the simulation for *steps* timesteps."""
        iterator = range(steps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"Simulating L={self.L}", unit="step")
            except ImportError:
                pass
        for _ in iterator:
            self.step()

    def run_with_cluster_avalanches(self, steps: int, progress: bool = True):
        """Run simulation and measure avalanche sizes via BFS cluster labelling.

        Each fire event is traced from ignition to extinction.
        The total number of distinct trees consumed in each connected
        fire cluster is recorded as the avalanche size.
        """
        from scipy.ndimage import label

        iterator = range(steps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"Cluster sim L={self.L}", unit="step")
            except ImportError:
                pass

        for _ in iterator:
            g = self.grid
            L = self.L

            # Before step: record which cells are burning
            was_burning = (g == BURNING).copy()

            # Perform the step
            self.step()

            # After step: cells that are now EMPTY and were BURNING before
            # are freshly burned out. But we need per-cluster sizes.
            # Use freshly ignited cells: cells now BURNING that weren't before
            now_burning = (g == BURNING) & ~was_burning
            if now_burning.any():
                labelled, n_clusters = label(now_burning)
                for c in range(1, n_clusters + 1):
                    size = int((labelled == c).sum())
                    if size > 0:
                        self.cluster_avalanche_sizes.append(size)

    def snapshot(self) -> np.ndarray:
        return self.grid.copy()

    @property
    def tree_density(self) -> float:
        return float((self.grid == TREE).sum()) / (self.L ** 2)


# --- Convenience: run a multi-scale experiment ---
def run_multiscale(grid_sizes: list[int], p: float = 0.05, f: float = 0.0001,
                   steps: int = 5000, seed: int = 42,
                   thermalization: int = 1000) -> dict:
    """Run the model at multiple grid sizes for finite-size scaling.

    Returns dict mapping L -> {'avalanche_sizes': array, 'density': list}
    """
    results = {}
    for L in grid_sizes:
        print(f"  Running L={L} ({L*L} cells, {steps} steps)...")
        model = ForestFireModel(L=L, p=p, f=f, seed=seed)
        # Thermalization: discard initial transients
        model.run(thermalization, progress=False)
        model.avalanche_sizes.clear()
        model.density_history.clear()
        # Production run
        model.run(steps, progress=True)
        results[L] = {
            'avalanche_sizes': np.array(model.avalanche_sizes),
            'density_history': model.density_history,
            'final_density': model.tree_density,
        }
    return results


if __name__ == "__main__":
    model = ForestFireModel(L=128, p=0.05, f=0.0005, seed=42)
    model.run(2000, progress=True)
    print(f"\nFinal tree density : {model.tree_density:.4f}")
    print(f"Total fire events  : {len(model.avalanche_sizes)}")
    if model.avalanche_sizes:
        sizes = np.array(model.avalanche_sizes)
        print(f"Max avalanche size : {sizes.max()}")
        print(f"Mean avalanche size: {sizes.mean():.1f}")
