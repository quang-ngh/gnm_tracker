"""Assemble a :class:`Fitter` from frames + detections (shared by Tier 1/2/3)."""

from __future__ import annotations

import numpy as np
import torch

from gnm_tracker.detect.mediapipe_detector import stack_detections
from gnm_tracker.fit.losses import ExpressionPrior
from gnm_tracker.fit.stages import Fitter, Targets
from gnm_tracker.model import ClipParams, PerspectiveCamera
from gnm_tracker.mouth import MouthPolicy


def build_fitter(
    model,
    cfg,
    frames: np.ndarray,
    detections: list,
    device: str | torch.device = "cpu",
    with_images: bool = False,
    renderer=None,
) -> Fitter:
    """Create a Fitter with camera, per-clip params, targets and regularizers."""
    stacked = stack_detections(detections)
    num_frames = len(detections)
    image_hw = tuple(frames[0].shape[:2])

    camera = PerspectiveCamera.from_config(cfg, image_hw).to(device)

    params = ClipParams(
        num_frames=num_frames,
        identity_dim=model.identity_dim,
        expression_dim=model.expression_dim,
        num_joints=model.num_joints,
        device=device,
    )
    params.initialize_pose(model, camera, cfg)

    targets = Targets(
        uv=torch.as_tensor(stacked["xy"], dtype=torch.float32, device=device),
        conf=torch.as_tensor(stacked["conf"], dtype=torch.float32, device=device),
        interocular=torch.as_tensor(stacked["interocular"], dtype=torch.float32, device=device),
        images=(
            torch.as_tensor(frames, dtype=torch.float32, device=device) / 255.0
            if with_images else None
        ),
    )

    mouth = MouthPolicy.from_config(cfg, model.tongue_dims)
    expr_prior = ExpressionPrior.from_config(cfg, device)

    rigid_subset = None
    if cfg.model.landmarks == "dense_468":
        rigid_subset = torch.as_tensor(cfg.fit.rigid_landmark_ids, dtype=torch.long, device=device)

    return Fitter(
        model=model, camera=camera, params=params, targets=targets, cfg=cfg,
        mouth=mouth, expr_prior=expr_prior, renderer=renderer, rigid_subset=rigid_subset,
    )
