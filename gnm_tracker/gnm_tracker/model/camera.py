"""Pinhole camera for landmark reprojection.

The GNM model already outputs *world-space* posed vertices (neck rotation +
translation are applied inside GNM). So the camera carries only:

  * intrinsics: focal length (pixels) + principal point, and
  * a fixed world->camera rotation ``R_wc`` that maps GNM's canonical axes
    (+X subject-left, +Y up, +Z forward/face) onto a camera looking down +Z
    with +X right, +Y down.

For GNM's frame, ``R_wc`` is a 180 deg rotation about X (``diag(1, -1, -1)``):
that sends face-forward toward the camera and "up" to image-up. See
``configs/default.yaml: camera.base_rotation_axis_angle`` and
``src/dump_gnm_groups.py`` for the empirical orientation check.

In-the-wild intrinsics are unknown, so focal is an assumed default and may be
weakly optimized (Stage A) with a strong log-space prior.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis-angle ``(..., 3)`` -> rotation matrix ``(..., 3, 3)``."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (..., 1)
    small = theta < 1e-8
    axis = aa / torch.clamp(theta, min=1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    k = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(aa.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(k)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    rot = eye + s * k + (1 - c) * (k @ k)
    return torch.where(small[..., None], eye, rot)


class PerspectiveCamera(nn.Module):
    """Projects world points ``(..., 3)`` to pixels ``(..., 2)``."""

    def __init__(
        self,
        focal_px: float,
        principal_point: tuple[float, float],
        r_wc: np.ndarray,
        optimize_focal: bool = False,
        z_near: float = 0.05,
    ):
        super().__init__()
        self.z_near = float(z_near)
        self._focal_init = float(focal_px)
        # Optimize focal in log-space to keep it positive and scale-stable.
        log_focal = torch.tensor(float(np.log(focal_px)), dtype=torch.float32)
        if optimize_focal:
            self.log_focal = nn.Parameter(log_focal)
        else:
            self.register_buffer("log_focal", log_focal)
        self.register_buffer(
            "principal_point",
            torch.tensor(principal_point, dtype=torch.float32),
        )
        self.register_buffer(
            "r_wc", torch.as_tensor(np.asarray(r_wc), dtype=torch.float32)
        )

    @property
    def focal(self) -> torch.Tensor:
        return torch.exp(self.log_focal)

    def focal_log_reg(self) -> torch.Tensor:
        """Penalty pulling focal back toward its assumed init (Section 6, Stage A)."""
        return (self.log_focal - np.log(self._focal_init)) ** 2

    def to_camera(self, points_world: torch.Tensor) -> torch.Tensor:
        """Rotate world points into camera space (no camera translation)."""
        return points_world @ self.r_wc.T

    def project(self, points_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(uv (..., 2), z_cam (...))``. z_cam <= z_near is behind camera."""
        cam = self.to_camera(points_world)
        z = cam[..., 2]
        z_safe = torch.clamp(z, min=self.z_near)
        f = self.focal
        cx, cy = self.principal_point[0], self.principal_point[1]
        u = f * cam[..., 0] / z_safe + cx
        v = f * cam[..., 1] / z_safe + cy
        return torch.stack([u, v], dim=-1), z

    @classmethod
    def from_config(cls, cfg, image_hw: tuple[int, int]) -> "PerspectiveCamera":
        h, w = image_hw
        focal = float(cfg.camera.focal_ratio) * max(h, w)
        pp = cfg.get("camera.principal_point")
        if pp is None:
            pp = (w / 2.0, h / 2.0)
        aa = torch.tensor(cfg.camera.base_rotation_axis_angle, dtype=torch.float32)
        r_wc = axis_angle_to_matrix(aa).numpy()
        return cls(
            focal_px=focal,
            principal_point=(float(pp[0]), float(pp[1])),
            r_wc=r_wc,
            optimize_focal=bool(cfg.camera.optimize_focal),
            z_near=float(cfg.camera.z_near),
        )
