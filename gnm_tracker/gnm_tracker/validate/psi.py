"""Standalone psi-cleanliness CI check (Section 9).

Independent of any fitted clip: sample random expressions and identities and
assert the cross-identity displacement drift stays below threshold. Wired into
`src/validate_psi.py` for CI. Requires torch.
"""

from __future__ import annotations


def run_psi_cleanliness_check(cfg, device: str = "cpu", num_expr: int = 16, num_pairs: int = 8) -> dict:
    import torch

    from gnm_tracker.model import GnmModel
    from gnm_tracker.quality.psi_cleanliness import passes, psi_cleanliness_score

    model = GnmModel(cfg, device)
    psi = 0.3 * torch.randn(num_expr, model.expression_dim, device=device)
    a_shape = 0.5 * torch.randn(model.identity_dim, device=device)
    score = psi_cleanliness_score(model, psi, a_shape, num_pairs=num_pairs)
    threshold = float(cfg.psi_cleanliness.consistency_threshold)
    return {"score": score, "pass": passes(score, threshold), "threshold": threshold}
