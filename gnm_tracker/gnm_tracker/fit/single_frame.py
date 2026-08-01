"""Tier 1 — single-frame fitter (the optimizer core).

Analysis-by-synthesis on one image: detect 468 landmarks -> map to GNM surface
via the barycentric correspondence -> optimize pose, then shape, then coarse psi
(all single-frame). Landmark reprojection loss + regularizers; no photometric.

Gate (Section 5): stable, sane convergence — median normalized reprojection
error below the configured threshold, no NaNs, no exploded psi, overlay aligned.
Build `viz/` alongside this (Section 11): every fit script takes `--viz`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gnm_tracker.fit.build import build_fitter


@dataclasses.dataclass
class FitResult:
    params: dict          # snapshot of shape/expression/rotations/translation
    focal: float
    median_reproj_error: float
    converged: bool
    history: dict
    fitter: object        # the Fitter (for viz / further inspection)


def fit_single_frame(model, frame_rgb: np.ndarray, detection, cfg, device="cpu") -> FitResult:
    """Fit GNM to a single frame. ``detection`` is a FrameLandmarks for this frame."""
    fitter = build_fitter(model, cfg, frame_rgb[None], [detection], device=device)
    stages = cfg.fit.stages
    pool = np.arange(1)

    # Stage A: rigid pose only (beta=0, psi=0), stable-landmark subset.
    fitter.run_stage("rigid", stages.rigid, pool, use_rigid_subset=True)
    # Stage B: shape + pose + coarse psi (single frame).
    fitter.run_stage("shape", stages.shape, pool)
    # Stage C: freeze shape, refine per-frame psi + pose.
    fitter.params.freeze_shape()
    fitter.run_stage("expression", stages.expression, pool)

    err = float(fitter.normalized_reproj_error().median())
    threshold = float(cfg.quality.reproj_error_max)
    result = FitResult(
        params=fitter.params.snapshot(),
        focal=float(fitter.camera.focal.detach()),
        median_reproj_error=err,
        converged=bool(err < threshold and _is_finite(fitter.params)),
        history=fitter.history,
        fitter=fitter,
    )
    return result


def _is_finite(params) -> bool:
    for t in (params.shape, params.expression, params.rotations, params.translation):
        if not torch.isfinite(t).all():
            return False
    return True
