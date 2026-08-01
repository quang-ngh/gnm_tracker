"""Differentiable GNM wrapper.

Thin layer over ``gnm_pytorch.GNM`` (the PyTorch backend, so gradients flow to
identity / expression / pose). Adds the pieces the tracker needs that GNM does
not surface directly:

  * a barycentric landmark layer (dense-468 MediaPipe or sparse-68),
  * the tongue expression-dim group for the mouth policy (Section 8),
  * vertex masks for the photometric loss (exclude mouth/eye interior),
  * a convenience forward returning both vertices and landmarks.

GNM facts baked in (v3.0 HEAD, verified in src/dump_gnm_groups.py):
V=17821 vertices, J=4 joints [neck, head, left_eye, right_eye], I=253, E=383,
tongue dims 350..381, units = meters.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from gnm_tracker.config import Config, resolve_path
from gnm_tracker.model.landmark_layer import BarycentricLandmarks


class GnmModel(nn.Module):
    def __init__(self, cfg: Config, device: str | torch.device = "cpu"):
        super().__init__()
        from gnm.shape import gnm_pytorch  # local import: torch-only dependency

        self.cfg = cfg
        version = gnm_pytorch.GNMMajorVersion(str(cfg.model.version))
        variant = gnm_pytorch.GNMVariant(str(cfg.model.variant))
        gnm = gnm_pytorch.GNM.from_local(version=version, variant=variant)
        # GNM is a dataclass-backed nn.Module with eq=True -> unhashable, which
        # breaks nn.Module submodule bookkeeping (.to()/.children() hash modules).
        # Keep it OUT of self._modules and move its buffers to device explicitly.
        object.__setattr__(self, "gnm", gnm)

        self.num_vertices = int(self.gnm.num_vertices)
        self.num_joints = int(self.gnm.num_joints)
        self.identity_dim = int(self.gnm.identity_dim)
        self.expression_dim = int(self.gnm.expression_dim)

        # -- landmark correspondence --------------------------------------
        if cfg.model.landmarks == "dense_468":
            self.landmarks = BarycentricLandmarks.from_txt(
                resolve_path(cfg.model.dense_468_file)
            )
        else:
            self.landmarks = BarycentricLandmarks.gnm_sparse_68()
        # sparse-68 is always available for the psi-cleanliness reference metric.
        self.sparse68 = BarycentricLandmarks.gnm_sparse_68()

        # -- named groups (mouth policy + photometric masks) --------------
        groups = json.loads(resolve_path(cfg.model.expression_groups_file).read_text())
        tongue = torch.as_tensor(groups["mouth_interior_expr_dims"], dtype=torch.long)
        self.register_buffer("tongue_dims", tongue)
        self.register_buffer(
            "face_vertex_mask",
            self._build_face_mask(groups),
        )
        # Triangles for viz / rasterization.
        self.register_buffer(
            "triangles", torch.as_tensor(np.asarray(self.gnm.triangles), dtype=torch.long)
        )
        self.register_buffer(
            "template_vertices",
            torch.as_tensor(np.asarray(self.gnm.template_vertex_positions), dtype=torch.float32),
        )
        self.gnm.to(device)   # move GNM's buffers (it is not tracked as a submodule)
        self.to(device)

    # ------------------------------------------------------------------
    def _build_face_mask(self, groups: dict) -> torch.Tensor:
        """Visible exterior skin, excluding mouth/eye interior (M_face, Section 6)."""
        skin = set(np.asarray(self.gnm.vertex_group_indices("skin_exterior")).tolist())
        interior: set[int] = set()
        for name in groups.get("mouth_interior_vertex_groups", []) + groups.get(
            "eye_interior_vertex_groups", []
        ):
            if name in self.gnm.vertex_group_names:
                interior |= set(np.asarray(self.gnm.vertex_group_indices(name)).tolist())
        keep = np.array(sorted(skin - interior), dtype=np.int64)
        mask = torch.zeros(self.num_vertices, dtype=torch.bool)
        mask[keep] = True
        return mask

    # ------------------------------------------------------------------
    @staticmethod
    def _expand_shape(shape: torch.Tensor, n: int) -> torch.Tensor:
        """Broadcast a single (I,) identity to (N, I) — shape is per-clip static."""
        if shape.ndim == 1:
            return shape[None].expand(n, -1)
        return shape

    def forward(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        rotations: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """(shape, expr (N,E), rot (N,J,3), trans (N,3)) -> vertices (N, V, 3)."""
        n = expression.shape[0]
        identity = self._expand_shape(shape, n)
        return self.gnm(
            identity=identity,
            expression=expression,
            rotations=rotations,
            translation=translation,
        )

    def forward_landmarks(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        rotations: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        verts = self.forward(shape, expression, rotations, translation)
        return verts, self.landmarks(verts)

    def template_centroid(self) -> torch.Tensor:
        return self.template_vertices.mean(dim=0)
