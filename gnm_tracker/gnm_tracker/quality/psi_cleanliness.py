"""psi-cleanliness acceptance test (Section 9) — the REAL Tier 2/3 gate.

The property we need is "same expression => same psi, regardless of identity."
We test it by **cross-identity expression transfer**: take a psi fitted on person
A, apply it to person B's shape, and check it produces the *same expression* on B
rather than a distortion that echoes A's face.

Geometry formulation (no rendering needed): the expression displacement is the
per-landmark motion from neutral, in each identity's own scale. If psi is
disentangled, that displacement is (nearly) identity-independent, so the drift
between identity A and identity B is small. A large drift means psi is
identity-contaminated — revisit Stage B (shape fit) and increase L_psi.
"""

from __future__ import annotations

import numpy as np

_IBUG_EYE_R_OUTER = 36
_IBUG_EYE_L_OUTER = 45


def _expression_displacement(model, shape, psi):
    """Per-landmark motion from neutral for a batch of expressions, identity-scaled.

    shape: ``(I,)``; psi: ``(M, E)`` -> displacement ``(M, L, 3)`` in units of the
    identity's inter-ocular distance.
    """
    import torch

    m = psi.shape[0]
    layer = model.sparse68
    zeros_rot = torch.zeros(m, model.num_joints, 3, device=psi.device)
    zeros_trans = torch.zeros(m, 3, device=psi.device)
    zero_expr = torch.zeros_like(psi)

    verts0 = model(shape, zero_expr, zeros_rot, zeros_trans)
    verts1 = model(shape, psi, zeros_rot, zeros_trans)
    lm0, lm1 = layer(verts0), layer(verts1)                # (M, L, 3)
    scale = torch.linalg.norm(
        lm0[:, _IBUG_EYE_R_OUTER] - lm0[:, _IBUG_EYE_L_OUTER], dim=-1
    ).clamp_min(1e-6)                                       # (M,)
    return (lm1 - lm0) / scale[:, None, None]


def psi_cleanliness_score(
    model,
    psi_samples,                # (M, E) expressions to test
    a_shape,                    # (I,) the clip's fitted identity
    b_shapes=None,              # (K, I) other identities; sampled if None
    num_pairs: int = 8,
    identity_scale: float = 1.0,
) -> dict:
    """Mean normalized cross-identity displacement drift. Lower is cleaner."""
    import torch

    device = psi_samples.device
    with torch.no_grad():
        if b_shapes is None:
            # Sample plausible alternate identities as small id-space perturbations.
            b_shapes = identity_scale * torch.randn(num_pairs, model.identity_dim, device=device)

        da = _expression_displacement(model, a_shape, psi_samples)  # (M, L, 3)
        drifts = []
        for b in b_shapes:
            db = _expression_displacement(model, b, psi_samples)    # (M, L, 3)
            drift = torch.linalg.norm(da - db, dim=-1).mean()      # scalar
            drifts.append(float(drift))
    drifts = np.asarray(drifts)
    return {
        "mean_drift": float(drifts.mean()),
        "max_drift": float(drifts.max()),
        "per_identity": drifts,
    }


def passes(score: dict, threshold: float) -> bool:
    return score["mean_drift"] < threshold
