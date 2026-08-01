"""Full-clip overlay video (Section 11).

Temporal jitter is only visible in motion, never in a still frame — the overlay
video (shaded green mesh tracking the face) is how you catch it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnm_tracker.io.video import write_video
from gnm_tracker.viz.overlay import draw_landmark_residuals
from gnm_tracker.viz.render_compare import overlay_shaded, side_by_side


def side_by_side_video(path: str | Path, fitter, frames: np.ndarray, cfg) -> None:
    """Input frame | GNM-colored fit render, per frame (Section 11)."""
    out = [side_by_side(frames[i], fitter, i) for i in range(len(frames))]
    write_video(path, out, fps=float(cfg.viz.video_fps))


def overlay_video(
    path: str | Path, fitter, frames: np.ndarray, cfg, residuals: bool = False
) -> None:
    color = tuple(cfg.get("viz.mesh_color", [0, 210, 0]))
    alpha = float(cfg.get("viz.mesh_alpha", 0.55))
    out = []
    for i in range(len(frames)):
        img = overlay_shaded(frames[i], fitter, i, color=color, alpha=alpha)
        if residuals:
            img = draw_landmark_residuals(img, fitter, i, cfg)
        out.append(img)
    write_video(path, out, fps=float(cfg.viz.video_fps))
