"""Barycentric landmark layer.

A GNM landmark is a fixed barycentric combination of three mesh vertices, stored
as `i0 w0 i1 w1 i2 w2` per row (the format shared by GNM's `head_sparse_68.txt`
and the vendored `head_dense_468.txt`). Because the anchors are barycentric on
the topology they are identity/expression/pose invariant — the SAME layer maps
any posed GNM mesh to landmark positions, which is exactly what the reprojection
loss needs (Section 6, `P_i(beta, psi, theta)`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BarycentricLandmarks(nn.Module):
    """Maps GNM vertices ``(..., V, 3)`` to landmarks ``(..., L, 3)``."""

    def __init__(self, indices: np.ndarray, weights: np.ndarray):
        super().__init__()
        indices = np.asarray(indices)
        weights = np.asarray(weights, dtype=np.float32)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(f"indices must be (L, 3), got {indices.shape}")
        if weights.shape != indices.shape:
            raise ValueError("indices and weights must have the same shape")
        # Barycentric weights should sum to ~1 per landmark.
        wsum = weights.sum(axis=1)
        if not np.allclose(wsum, 1.0, atol=1e-3):
            bad = int(np.argmax(np.abs(wsum - 1.0)))
            raise ValueError(
                f"barycentric weights must sum to 1; row {bad} sums to {wsum[bad]:.4f}"
            )
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    @property
    def num_landmarks(self) -> int:
        return int(self.indices.shape[0])

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """vertices ``(..., V, 3)`` -> landmarks ``(..., L, 3)``."""
        # Advanced-index the vertex axis with the (L, 3) anchor indices.
        face = vertices[..., self.indices, :]  # (..., L, 3, 3)
        weights = self.weights.to(face.dtype)
        return (face * weights[..., None]).sum(dim=-2)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_txt(cls, path: str | Path) -> "BarycentricLandmarks":
        """Load an `i0 w0 i1 w1 i2 w2` correspondence file."""
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"expected (L, 6) correspondence, got {data.shape}")
        indices = data[:, ::2].astype(np.int64)
        weights = data[:, 1::2].astype(np.float32)
        return cls(indices, weights)

    @classmethod
    def gnm_sparse_68(cls) -> "BarycentricLandmarks":
        """GNM's shipped 68-point correspondence, via its own loader."""
        from gnm.shape import gnm_landmarks

        cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
        return cls(cfg.indices.astype(np.int64), cfg.weights.astype(np.float32))
