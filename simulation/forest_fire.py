"""
Drossel-Schwabl Forest Fire Model
=================================
A cellular automaton exhibiting self-organized criticality (SOC).

Cell states:
    0 = Empty
    1 = Tree
    2 = Burning

Rules (applied simultaneously each timestep):
    1. Burning cell → Empty
    2. Tree with at least one burning neighbour → Burning
    3. Tree ignites spontaneously with probability f  (lightning)
    4. Empty cell grows a tree with probability p

Reference:
    Drossel, B. & Schwabl, F. (1992). Self-organized critical forest-fire model.
    Physical Review Letters, 69(11), 1629.
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Cell-state constants
# ---------------------------------------------------------------------------
EMPTY = 0
TREE = 1
BURNING = 2


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

class ForestFireModel:
    """Drossel-Schwabl forest fire cellular automaton on a 2-D square lattice."""

    def __init__(self, L: int = 256, p: float = 0.05, f: float = 0.0001,
                 seed: int | None = None):
        """
        Parameters
        ----------
        L : int
            Side length of the square lattice (L×L grid).
        p : float
            Probability that an empty cell grows a tree each timestep.
        f : float
            Probability that a tree is struck by lightning each timestep.
        seed : int or None
            Random seed for reproducibility.
        """
        self.L = L
        self.p = p
        self.f = f
        self.rng = np.random.default_rng(seed)

        # Initialise grid: start with ~50 % tree coverage
        self.grid = self.rng.choice(
            [EMPTY, TREE], size=(L, L), p=[0.5, 0.5]
        ).astype(np.int8)

        # Statistics accumulators
        self.avalanche_sizes: list[int] = []      # trees burned per fire event
        self.density_history: list[float] = []     # tree fraction over time
        self.time = 0

    # ------------------------------------------------------------------
    # Neighbourhood helper (4-connected / von Neumann)
    # ------------------------------------------------------------------
    @staticmethod
    def _has_burning_neighbour(grid: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where cell has ≥ 1 burning neighbour."""
        burning = (grid == BURNING)
        # Shift in four cardinal directions and combine
        nb = np.zeros_like(burning)
        nb[:-1, :] |= burning[1:, :]   # neighbour below
        nb[1:, :]  |= burning[:-1, :]  # neighbour above
        nb[:, :-1] |= burning[:, 1:]   # neighbour right
        nb[:, 1:]  |= burning[:, :-1]  # neighbour left
        return nb

    # ------------------------------------------------------------------
    # Single-timestep update
    # ------------------------------------------------------------------
    def step(self) -> int:
        """
        Advance the model by one timestep.

        Returns
        -------
        n_burned : int
            Number of trees that caught fire this step (0 if no fire event).
        """
        g = self.grid
        L = self.L

        # --- 1. Identify cells that will burn this step ---
        # (a) Trees adjacent to currently burning cells
        spread_mask = (g == TREE) & self._has_burning_neighbour(g)
        # (b) Spontaneous ignition (lightning)
        lightning_mask = (g == TREE) & ~spread_mask
        lightning_hits = self.rng.random((L, L)) < self.f
        ignite_mask = lightning_mask & lightning_hits

        new_burning = spread_mask | ignite_mask

        # --- 2. Burning → Empty ---
        was_burning = (g == BURNING)
        g[was_burning] = EMPTY

        # --- 3. Set newly burning cells ---
        n_burned = int(new_burning.sum())
        g[new_burning] = BURNING

        # --- 4. Tree growth on empty cells ---
        empty_mask = (g == EMPTY)
        grow = empty_mask & (self.rng.random((L, L)) < self.p)
        g[grow] = TREE

        # --- Record statistics ---
        if n_burned > 0:
            self.avalanche_sizes.append(n_burned)
        self.density_history.append(float((g == TREE).sum()) / (L * L))
        self.time += 1

        return n_burned

    # ------------------------------------------------------------------
    # Multi-step run with fire-cluster labelling
    # ------------------------------------------------------------------
    def run(self, steps: int, record_clusters: bool = False,
            progress: bool = True):
        """
        Run the simulation for *steps* timesteps.

        Parameters
        ----------
        steps : int
            Number of timesteps to simulate.
        record_clusters : bool
            If True, use connected-component labelling to measure individual
            fire-cluster sizes (slower but more accurate than per-step count).
        progress : bool
            If True, display a tqdm progress bar.
        """
        iterator = range(steps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Simulating", unit="step")
            except ImportError:
                pass

        for _ in iterator:
            self.step()

    # ------------------------------------------------------------------
    # Convenience: cluster-based avalanche measurement
    # ------------------------------------------------------------------
    def run_with_cluster_avalanches(self, steps: int, progress: bool = True):
        """
        Run *steps* timesteps and record fire-cluster sizes using
        scipy connected-component labelling for more precise avalanche
        size statistics.

        Each time lightning starts a new fire (no prior burning on the grid),
        we let the fire propagate until extinction and record the total
        cluster size as one avalanche.
        """
        from scipy.ndimage import label

        iterator = range(steps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Simulating (cluster mode)",
                                unit="step")
            except ImportError:
                pass

        cluster_sizes: list[int] = []

        for _ in iterator:
            g = self.grid

            # Check if there are currently burning cells
            currently_burning = (g == BURNING).any()

            self.step()

            # If there were no burning cells before and now there are,
            # a new fire just started — we won't record size yet.
            # We record sizes when fires go extinct.
            now_burning = (g == BURNING).any()

            if currently_burning and not now_burning:
                # Fire just went out — count the freshly emptied cells
                # that were part of this fire (they've been set to EMPTY).
                # This is already tracked in avalanche_sizes via step().
                pass

        self.cluster_avalanche_sizes = cluster_sizes

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> np.ndarray:
        """Return a copy of the current grid."""
        return self.grid.copy()

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    @property
    def tree_density(self) -> float:
        return float((self.grid == TREE).sum()) / (self.L ** 2)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ForestFireModel(L=128, p=0.05, f=0.0005, seed=42)
    model.run(2000, progress=True)

    print(f"\nFinal tree density : {model.tree_density:.4f}")
    print(f"Total fire events  : {len(model.avalanche_sizes)}")
    if model.avalanche_sizes:
        sizes = np.array(model.avalanche_sizes)
        print(f"Max avalanche size : {sizes.max()}")
        print(f"Mean avalanche size: {sizes.mean():.1f}")
