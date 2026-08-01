"""Trajectory and loss-curve plots (Section 11).

Temporal jumps, exploded psi, and stage non-convergence are invisible in single
frames but obvious in these plots. Includes a dedicated tongue-dim monitor to
confirm the `pin` policy keeps the mouth interior bounded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_trajectories(
    expression: np.ndarray,      # (T, E)
    rotations: np.ndarray,       # (T, J, 3)
    translation: np.ndarray,     # (T, 3)
    tongue_dims: np.ndarray,     # (Kt,)
    out_path: str | Path,
    max_dims: int = 12,
) -> None:
    t = expression.shape[0]
    x = np.arange(t)
    # Show the most active non-tongue expression dims.
    tongue_set = set(int(d) for d in tongue_dims)
    active = [d for d in np.argsort(-expression.std(0)) if d not in tongue_set][:max_dims]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    for d in active:
        axes[0].plot(x, expression[:, d], lw=0.8, label=f"ψ{d}")
    axes[0].set_title("Expression ψ (most active dims)")
    axes[0].set_ylabel("value")

    for j in range(rotations.shape[1]):
        axes[1].plot(x, np.linalg.norm(rotations[:, j], axis=-1), lw=0.9, label=f"joint{j}")
    axes[1].set_title("Per-joint rotation angle (rad)")
    axes[1].legend(fontsize=7, ncol=4)

    for k, lbl in enumerate("xyz"):
        axes[2].plot(x, translation[:, k], lw=0.9, label=f"t{lbl}")
    axes[2].set_title("Global translation (m)")
    axes[2].legend(fontsize=7, ncol=3)

    if len(tongue_dims):
        axes[3].plot(x, np.linalg.norm(expression[:, tongue_dims], axis=-1), lw=1.0, color="crimson")
    axes[3].set_title("Tongue-dim magnitude (pin policy should keep this ~0)")
    axes[3].set_xlabel("frame")

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_loss_curves(history: dict, out_path: str | Path) -> None:
    stages = list(history.keys())
    if not stages:
        return
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4), squeeze=False)
    for ax, name in zip(axes[0], stages):
        hist = history[name]
        if not hist:
            continue
        keys = [k for k in hist[0] if k != "total"]
        it = np.arange(len(hist))
        for k in keys:
            vals = np.array([h[k] for h in hist])
            if np.allclose(vals, 0):
                continue
            ax.plot(it, vals + 1e-9, lw=0.9, label=k)
        ax.plot(it, [h["total"] for h in hist], lw=1.6, color="k", label="total")
        ax.set_yscale("log")
        ax.set_title(f"Stage {name}")
        ax.set_xlabel("iter")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
