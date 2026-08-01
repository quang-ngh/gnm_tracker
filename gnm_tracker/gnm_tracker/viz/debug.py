"""Debug-folder orchestrator (Section 11).

Every fit run should dump a debug folder: a few sample-frame overlays, the
side-by-side renders, the full-clip overlay video, the trajectory + loss-curve
plots, and a psi-cleanliness transfer grid. This is required, not a nicety —
id↔expression leakage and jitter are only visible here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gnm_tracker.viz.overlay import draw_landmark_residuals
from gnm_tracker.viz.plots import plot_loss_curves, plot_trajectories
from gnm_tracker.viz.psi_cleanliness_viz import render_transfer_grid
from gnm_tracker.viz.render_compare import overlay_shaded, side_by_side
from gnm_tracker.viz.video import side_by_side_video


def dump_debug(out_dir: str | Path, fitter, frames: np.ndarray, cfg, result=None) -> Path:
    out = Path(out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    t = len(frames)
    n = min(int(cfg.viz.sample_frames), t)
    idxs = np.unique(np.linspace(0, t - 1, n).round().astype(int))

    color = tuple(cfg.get("viz.mesh_color", [0, 210, 0]))
    alpha = float(cfg.get("viz.mesh_alpha", 0.55))
    for i in idxs:
        # Primary: shaded green mesh over the frame (the GNM-teaser look).
        mesh = overlay_shaded(frames[i], fitter, int(i), color=color, alpha=alpha)
        cv2.imwrite(str(out / "overlays" / f"mesh_{i:04d}.png"), mesh[..., ::-1])
        # Secondary: reprojected-vs-detected landmark residuals (clean).
        resid = draw_landmark_residuals(frames[i], fitter, int(i), cfg)
        cv2.imwrite(str(out / "overlays" / f"resid_{i:04d}.png"), resid[..., ::-1])
        # Fit render beside the input.
        sbs = side_by_side(frames[i], fitter, int(i))
        cv2.imwrite(str(out / "overlays" / f"render_{i:04d}.png"), sbs[..., ::-1])

    if bool(cfg.viz.video):
        # Side-by-side (input | fit render) — jitter is only visible in motion.
        side_by_side_video(out / "side_by_side.mp4", fitter, frames, cfg)

    p = fitter.params
    plot_trajectories(
        p.expression.detach().cpu().numpy(),
        p.rotations.detach().cpu().numpy(),
        p.translation.detach().cpu().numpy(),
        fitter.model.tongue_dims.cpu().numpy(),
        out / "trajectories.png",
    )
    plot_loss_curves(fitter.history, out / "loss_curves.png")

    # psi-cleanliness transfer grid: clip's most-expressive psi on a random B.
    try:
        psi = p.expression.detach()
        a_psi = psi[int(torch.linalg.norm(psi, dim=-1).argmax())]
        render_transfer_grid(
            fitter.model, a_psi, p.shape.detach(), torch.randn_like(p.shape),
            out / "psi_cleanliness.png",
        )
    except Exception as exc:  # viz must never crash a fit
        print(f"[viz] psi-cleanliness grid skipped: {exc}")

    return out
