"""psi-cleanliness visualization (Section 9 / 11).

Render person A's psi applied to person B's shape as images, so a human can
confirm it reads as the *same expression* rather than distorting toward A. A 2x2
grid: rows = identities (A, B), columns = (neutral, +psi). If psi is clean, the
"+psi" column shows the same expression on both faces.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gnm_tracker.viz.render_compare import render_ortho


@torch.no_grad()
def _verts(model, shape, expr):
    j, i = model.num_joints, model.expression_dim
    zero_rot = torch.zeros(1, j, 3, device=shape.device)
    zero_trans = torch.zeros(1, 3, device=shape.device)
    return model(shape, expr[None], zero_rot, zero_trans)[0]


@torch.no_grad()
def render_transfer_grid(
    model,
    psi: torch.Tensor,          # (E,) expression fitted on identity A
    a_shape: torch.Tensor,      # (I,)
    b_shape: torch.Tensor,      # (I,)
    out_path: str | Path,
    hw: tuple[int, int] = (256, 256),
) -> None:
    zero = torch.zeros_like(psi)
    tiles = {
        (0, 0): _verts(model, a_shape, zero),
        (0, 1): _verts(model, a_shape, psi),
        (1, 0): _verts(model, b_shape, zero),
        (1, 1): _verts(model, b_shape, psi),
    }
    imgs = {k: render_ortho(model, v, hw) for k, v in tiles.items()}
    h, w = hw
    grid = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    labels = {(0, 0): "A neutral", (0, 1): "A + psi", (1, 0): "B neutral", (1, 1): "B + psi"}
    for (r, c), img in imgs.items():
        cv2.putText(img, labels[(r, c)], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid[..., ::-1])  # RGB -> BGR for imwrite
