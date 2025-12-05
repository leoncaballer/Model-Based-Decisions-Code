"""
Plotting utilities – revised for I0 sweeps only.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _default_plot_path(filename: str) -> str:
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)


####################################
# NEW — plot X* vs I0
####################################

def plot_I0_sweep(sweep_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(sweep_df["I0"], sweep_df["X_mean"], color="C0", lw=2)
    ax.set_xlabel("I0 (initial infrastructure)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title("X* vs I0")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    if out_path is None:
        out_path = _default_plot_path("ev_I0_sweep.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


####################################
# NEW — 2D heatmap phase plot X0 vs I0
####################################

def plot_phase_plot_I0(phase_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    # Expect tidy columns: X0, I0, X_final
    pivot = phase_df.pivot(index="I0", columns="X0", values="X_final").sort_index().sort_index(axis=1)
    I0s = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    plt.figure(figsize=(7,4))
    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], I0s[0], I0s[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("I0 (initial infrastructure)")
    plt.title("Network phase plot: X* over X0 and I0")

    if out_path is None:
        out_path = _default_plot_path("ev_phase_plot_I0.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path