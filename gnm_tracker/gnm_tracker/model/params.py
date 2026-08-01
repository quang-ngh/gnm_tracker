"""Optimizable parameters for one clip.

Design invariant #1 (Section 2): identity **shape** is ONE vector per clip,
static across frames; expression (psi) and pose are per-frame. That asymmetry is
encoded structurally here — ``shape`` is ``(I,)`` while ``expression`` /
``rotations`` / ``translation`` carry a leading frame axis ``T``.

Staged optimization (invariant #2/#3) is realized by freezing/thawing whole
tensors via :meth:`set_trainable`; e.g. Stage C freezes ``shape`` and optimizes
per-frame expression + pose only.
"""

from __future__ import annotations

import torch
from torch import nn

# Stage param name (as written in configs) -> attribute on ClipParams.
# "expression_coarse" and "expression" are the SAME tensor; they differ only in
# the psi-regularizer weight applied per stage (Section 6).
_PARAM_TO_ATTR = {
    "shape": "shape",
    "expression": "expression",
    "expression_coarse": "expression",
    "rotation": "rotations",
    "translation": "translation",
}


class ClipParams(nn.Module):
    def __init__(
        self,
        num_frames: int,
        identity_dim: int,
        expression_dim: int,
        num_joints: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.shape = nn.Parameter(torch.zeros(identity_dim, device=device))
        self.expression = nn.Parameter(
            torch.zeros(num_frames, expression_dim, device=device)
        )
        self.rotations = nn.Parameter(
            torch.zeros(num_frames, num_joints, 3, device=device)
        )
        self.translation = nn.Parameter(torch.zeros(num_frames, 3, device=device))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def initialize_pose(self, model, camera, cfg) -> None:
        """Center the head in front of the camera at zero rotation (good Stage-A init).

        For zero pose, GNM gives ``world = template + translation`` (verified), and
        the camera maps ``x_cam = R_wc @ world``. Solving ``R_wc @ (c + t) = [0,0,Zc]``
        for the template centroid ``c`` gives a translation that lands the head
        centered at distance ``Zc``.
        """
        centroid = model.template_centroid().to(self.translation.device)  # (3,)
        tinit = torch.tensor(
            cfg.camera.translation_init, dtype=torch.float32, device=self.translation.device
        )
        zc = tinit[2]
        target_cam = torch.tensor([0.0, 0.0, zc], device=self.translation.device)
        t = camera.r_wc.T @ target_cam - centroid
        t = t + torch.tensor(
            [tinit[0], tinit[1], 0.0], device=self.translation.device
        )
        self.translation.data[:] = t[None].expand_as(self.translation)
        self.rotations.data.zero_()

    # ------------------------------------------------------------------
    def set_trainable(self, names: list[str], camera=None) -> None:
        """Freeze everything, then unfreeze only the tensors named for this stage."""
        attrs = {_PARAM_TO_ATTR[n] for n in names if n in _PARAM_TO_ATTR}
        for attr in ("shape", "expression", "rotations", "translation"):
            getattr(self, attr).requires_grad_(attr in attrs)
        if camera is not None and hasattr(camera, "log_focal"):
            want = "focal" in names and isinstance(camera.log_focal, nn.Parameter)
            if isinstance(camera.log_focal, nn.Parameter):
                camera.log_focal.requires_grad_(bool(want))

    def freeze_shape(self) -> None:
        """Lock identity before per-frame psi refinement (invariant #3)."""
        self.shape.requires_grad_(False)

    def trainable_parameters(self, camera=None):
        params = [p for p in self.parameters() if p.requires_grad]
        if camera is not None and hasattr(camera, "log_focal"):
            if isinstance(camera.log_focal, nn.Parameter) and camera.log_focal.requires_grad:
                params.append(camera.log_focal)
        return params

    # ------------------------------------------------------------------
    def expression_frames(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.expression if idx is None else self.expression[idx]

    @torch.no_grad()
    def subtract_neutral(self, neutral: torch.Tensor) -> None:
        """Normalize expression so 'psi = 0' means the same thing across clips (Stage D)."""
        self.expression.data = self.expression.data - neutral[None]

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "shape": self.shape.detach().clone(),
            "expression": self.expression.detach().clone(),
            "rotations": self.rotations.detach().clone(),
            "translation": self.translation.detach().clone(),
        }
