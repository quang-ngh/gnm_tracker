"""Dataset record schema (Section 7).

One record per clip. `expr`/`pose` are per-frame; `shape` is the single per-clip
identity. `audio_feat` is DEFERRED (not exported in v1) but its slot is reserved
so the format can grow without breaking.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class ClipRecord:
    clip_id: str
    shape: np.ndarray          # (n_id,)  single identity vector for this clip
    expr: np.ndarray           # (T, n_expr)  per-frame psi, neutral-normalized
    pose: np.ndarray           # (T, n_pose)  per-frame joint rotations + translation
    neutral_expr: np.ndarray   # (n_expr,)  per-clip neutral used for normalization
    valid: np.ndarray          # (T,) bool  quality mask (False = do not train on)
    meta: dict[str, Any]
    audio_feat: np.ndarray | None = None   # DEFERRED in v1 (reserved slot)

    # ------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return int(self.expr.shape[0])

    @property
    def num_valid(self) -> int:
        return int(self.valid.sum())

    def to_npz_dict(self) -> dict[str, Any]:
        d = {
            "clip_id": self.clip_id,
            "shape": self.shape.astype(np.float32),
            "expr": self.expr.astype(np.float32),
            "pose": self.pose.astype(np.float32),
            "neutral_expr": self.neutral_expr.astype(np.float32),
            "valid": self.valid.astype(bool),
            "meta_json": np.array(_dump_meta(self.meta)),
        }
        if self.audio_feat is not None:
            d["audio_feat"] = self.audio_feat.astype(np.float32)
        return d

    # ------------------------------------------------------------------
    @classmethod
    def from_sequence_result(
        cls, clip_id: str, result, cfg, extra_meta: dict | None = None
    ) -> "ClipRecord":
        p = result.params
        rotations = _np(p["rotations"])            # (T, J, 3)
        translation = _np(p["translation"])        # (T, 3)
        pose = np.concatenate([rotations.reshape(rotations.shape[0], -1), translation], axis=1)

        meta = {
            "clip_id": clip_id,
            "gnm_version": str(cfg.model.version),
            "gnm_variant": str(cfg.model.variant),
            "landmarks": str(cfg.model.landmarks),
            "reproj_median": float(result.reproj_median),
            "psi_cleanliness_mean_drift": float(result.psi_cleanliness["mean_drift"]),
            "psi_clean_pass": bool(result.psi_clean_pass),
            "identity_cluster_id": None,  # Guardrail: no cross-clip clustering in v1
            "audio_feat": None,           # DEFERRED
        }
        if extra_meta:
            meta.update(extra_meta)

        return cls(
            clip_id=clip_id,
            shape=_np(p["shape"]),
            expr=_np(p["expression"]),
            pose=pose,
            neutral_expr=np.asarray(result.neutral, dtype=np.float32),
            valid=np.asarray(result.valid, dtype=bool),
            meta=meta,
        )


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _dump_meta(meta: dict) -> str:
    import json

    return json.dumps(meta)
