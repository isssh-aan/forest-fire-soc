"""
Animated visualisation of the Drossel-Schwabl Forest Fire Model
================================================================
Generates a GIF animation of the forest fire cellular automaton.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from forest_fire import ForestFireModel, EMPTY, TREE, BURNING


FOREST_CMAP = ListedColormap(["#8B6914", "#228B22", "#FF4500"])


def animate_model(L: int = 128, p: float = 0.05, f: float = 0.001,
                  frames: int = 300, interval: int = 80,
                  outpath: str = "forest_fire_animation.gif",
                  seed: int = 42):
    """
    Create an animated GIF of the forest fire model.

    Parameters
    ----------
    L : int
        Grid side length.
    p, f : float
        Tree growth and lightning probabilities.
    frames : int
        Number of animation frames.
    interval : int
        Delay between frames in ms.
    outpath : str
        File path for the output GIF.
    seed : int
        Random seed.
    """
    model = ForestFireModel(L=L, p=p, f=f, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Forest Fire Model — Self-Organized Criticality", fontsize=12)

    im = ax.imshow(model.grid, cmap=FOREST_CMAP, vmin=0, vmax=2,
                   interpolation="nearest")

    # Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor="#8B6914", edgecolor="k", label="Empty"),
        mpatches.Patch(facecolor="#228B22", edgecolor="k", label="Tree"),
        mpatches.Patch(facecolor="#FF4500", edgecolor="k", label="Burning"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.08))

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def update(frame):
        model.step()
        im.set_data(model.grid)
        step_text.set_text(f"t = {model.time}  |  ρ = {model.tree_density:.3f}")
        return [im, step_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(outpath), writer=PillowWriter(fps=1000 // interval))
    plt.close(fig)
    print(f"✓ Animation saved to {outpath}")


if __name__ == "__main__":
    animate_model(L=128, frames=200, outpath="forest_fire_animation.gif")
