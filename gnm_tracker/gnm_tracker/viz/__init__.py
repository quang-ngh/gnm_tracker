"""Debug visualization (Section 11) — first-class, built alongside the optimizer."""

from gnm_tracker.viz.debug import dump_debug
from gnm_tracker.viz.overlay import draw_landmark_residuals, draw_mesh_overlay
from gnm_tracker.viz.plots import plot_loss_curves, plot_trajectories
from gnm_tracker.viz.psi_cleanliness_viz import render_transfer_grid
from gnm_tracker.viz.render_compare import (
    overlay_shaded,
    render_ortho,
    render_posed,
    side_by_side,
)
from gnm_tracker.viz.video import overlay_video, side_by_side_video

__all__ = [
    "dump_debug",
    "draw_mesh_overlay",
    "draw_landmark_residuals",
    "plot_trajectories",
    "plot_loss_curves",
    "render_transfer_grid",
    "overlay_shaded",
    "render_ortho",
    "render_posed",
    "side_by_side",
    "overlay_video",
    "side_by_side_video",
]
