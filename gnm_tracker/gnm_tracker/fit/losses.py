"""Losses and regularizers (Section 6).

Primary supervision is landmark reprojection; photometric is a refinement-only
term (Stage C). Regularizers pin identity, keep expression in-prior, smooth
motion in time, and bound the mouth interior. All terms are combined with
per-stage weights (`weighted_total`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Landmark reprojection (primary signal)
# ---------------------------------------------------------------------------
def _huber(dist: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber applied to a (nonnegative) residual magnitude."""
    quad = 0.5 * dist**2
    lin = delta * (dist - 0.5 * delta)
    return torch.where(dist <= delta, quad, lin)


def landmark_reprojection_loss(
    pred_uv: torch.Tensor,      # (T, L, 2)
    target_uv: torch.Tensor,    # (T, L, 2)
    conf: torch.Tensor,         # (T, L)
    delta: float,
    subset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Confidence-weighted Huber on 2D landmark residuals (L_lmk)."""
    if subset is not None:
        pred_uv = pred_uv[:, subset]
        target_uv = target_uv[:, subset]
        conf = conf[:, subset]
    dist = torch.linalg.norm(pred_uv - target_uv, dim=-1)  # (T, L)
    h = _huber(dist, delta)
    return (conf * h).sum() / (conf.sum() + _EPS)


# ---------------------------------------------------------------------------
# Expression prior (L_psi) — Mahalanobis if available, else strong L2 (Sec 6)
# ---------------------------------------------------------------------------
class ExpressionPrior:
    def __init__(self, mode: str = "l2", precision: torch.Tensor | None = None):
        self.mode = mode if (mode != "mahalanobis" or precision is not None) else "l2"
        self.precision = precision

    def loss(self, psi: torch.Tensor) -> torch.Tensor:  # psi (T, E)
        if self.mode == "mahalanobis":
            p = self.precision.to(psi.dtype).to(psi.device)
            return torch.einsum("te,ef,tf->t", psi, p, psi).mean()
        return (psi**2).sum(dim=-1).mean()

    @classmethod
    def from_config(cls, cfg, device: str | torch.device = "cpu") -> "ExpressionPrior":
        mode = cfg.get("fit.regularizers.psi_prior", "l2")
        precision = None
        if mode == "mahalanobis":
            from gnm_tracker.config import resolve_path

            p = resolve_path("data/groups/psi_precision.npz")
            if Path(p).exists():
                data = np.load(p)
                precision = torch.as_tensor(data["precision"], dtype=torch.float32, device=device)
            else:
                # Graceful fallback (Section 6: "else plain L2").
                print(
                    "[ExpressionPrior] no psi_precision.npz found; falling back to L2. "
                    "Run src/estimate_psi_prior.py to enable Mahalanobis."
                )
                mode = "l2"
        return cls(mode=mode, precision=precision)


def identity_prior_loss(beta: torch.Tensor) -> torch.Tensor:
    """L_beta = ||beta||^2."""
    return (beta**2).sum()


# ---------------------------------------------------------------------------
# Temporal smoothness (L_temp) — second difference, WITHIN a clip only (Sec 6)
# ---------------------------------------------------------------------------
def temporal_second_diff(x: torch.Tensor) -> torch.Tensor:
    """Second-difference penalty over the leading (time) axis of ``x`` (T, ...)."""
    if x.shape[0] < 3:
        return x.new_zeros(())
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return (d2.reshape(d2.shape[0], -1) ** 2).sum(dim=-1).mean()


def pose_vector(rotations: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Flatten per-frame pose to (T, J*3 + 3) for temporal smoothing."""
    return torch.cat([rotations.reshape(rotations.shape[0], -1), translation], dim=-1)


def temporal_loss(psi: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return temporal_second_diff(psi) + temporal_second_diff(pose_vector(rotations, translation))


# ---------------------------------------------------------------------------
# Collision (L_col) — optional/advanced (Section 6). Placeholder no-op in v1.
# ---------------------------------------------------------------------------
def collision_loss(verts: torch.Tensor) -> torch.Tensor:
    """Teeth/tongue-vs-lip interpenetration penalty. Not implemented in v1."""
    return verts.new_zeros(())


# ---------------------------------------------------------------------------
# Total
# ---------------------------------------------------------------------------
def weighted_total(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Sum ``weights[k] * terms[k]`` over present terms with nonzero weight."""
    total = None
    for key, value in terms.items():
        w = float(weights.get(key, 0.0))
        if w == 0.0:
            continue
        contrib = w * value
        total = contrib if total is None else total + contrib
    if total is None:
        # No active weighted term this stage; return a differentiable zero.
        any_term = next(iter(terms.values()))
        return any_term.new_zeros(())
    return total
