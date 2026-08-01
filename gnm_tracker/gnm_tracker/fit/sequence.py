"""Tier 2 — sequence fitter with disentanglement (quality lives or dies here).

Schedule (Section 5/6), one clip == one Fitter (temporal smoothness never
crosses clip boundaries):

  A. rigid pose on all frames (stable-landmark subset).
  B. one SHARED shape across K sampled frames + per-frame pose + coarse psi.
  C. LOCK shape, then per-frame psi + pose on all frames (+photometric if a
     renderer is supplied, +temporal, +mouth).
  D. estimate per-clip neutral psi and normalize; compute the quality mask.

Gate: passes the psi-cleanliness test (Section 9) at the configured threshold
and the full-clip overlay is jitter-free.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gnm_tracker.fit.build import build_fitter
from gnm_tracker.quality import compute_metrics, compute_validity
from gnm_tracker.quality.psi_cleanliness import passes, psi_cleanliness_score


@dataclasses.dataclass
class SequenceResult:
    params: dict
    neutral: np.ndarray
    valid: np.ndarray
    metrics: dict
    reproj_median: float
    psi_cleanliness: dict
    psi_clean_pass: bool
    history: dict
    fitter: object


def sample_k_frames(num_frames: int, detected: np.ndarray, k: int) -> np.ndarray:
    """K frames spread across the clip (detected only), to span its pose/expr range."""
    idx = np.where(detected)[0] if detected.any() else np.arange(num_frames)
    if len(idx) <= k:
        return idx
    picks = np.linspace(0, len(idx) - 1, k).round().astype(int)
    return idx[np.unique(picks)]


@torch.no_grad()
def estimate_neutral(psi: torch.Tensor, valid: np.ndarray, cfg) -> torch.Tensor:
    """Per-clip neutral psi from the calmest (lowest-velocity) frames (Stage D)."""
    t = psi.shape[0]
    if t == 1:
        return psi[0].clone()
    vel = torch.zeros(t, device=psi.device)
    vel[1:] = torch.linalg.norm(psi[1:] - psi[:-1], dim=-1)
    pool = np.where(valid)[0]
    if len(pool) == 0:
        pool = np.arange(t)
    vel_pool = vel[pool]
    pct = float(cfg.neutral.min_motion_percentile) / 100.0
    thresh = torch.quantile(vel_pool, pct)
    calm = pool[(vel_pool <= thresh).cpu().numpy()]
    if len(calm) < int(cfg.neutral.min_frames):
        order = pool[torch.argsort(vel_pool).cpu().numpy()]
        calm = order[: int(cfg.neutral.min_frames)]
    return psi[calm].mean(dim=0)


def _select_psi_samples(psi: torch.Tensor, valid: np.ndarray, k: int = 16) -> torch.Tensor:
    """Pick the most expressive valid frames to probe psi-cleanliness."""
    pool = np.where(valid)[0]
    if len(pool) == 0:
        pool = np.arange(psi.shape[0])
    mag = torch.linalg.norm(psi[pool], dim=-1)
    order = pool[torch.argsort(mag, descending=True).cpu().numpy()]
    return psi[order[: min(k, len(order))]]


def fit_sequence(model, frames, detections, cfg, device="cpu", renderer=None) -> SequenceResult:
    with_images = renderer is not None
    fitter = build_fitter(model, cfg, frames, detections, device=device,
                          with_images=with_images, renderer=renderer)
    stages = cfg.fit.stages
    t = len(detections)
    all_pool = np.arange(t)
    bs = int(cfg.fit.batch_size)
    detected = (fitter.targets.conf.sum(dim=-1) > 0).cpu().numpy()

    # A: rigid pose (all frames, minibatched).
    fitter.run_stage("rigid", stages.rigid, all_pool, batch_size=bs, use_rigid_subset=True)
    # B: shared shape on K frames spanning the clip.
    kf = sample_k_frames(t, detected, int(cfg.fit.k_shape_frames))
    fitter.run_stage("shape", stages.shape, kf)
    # C: LOCK shape, then per-frame psi + pose on all frames.
    fitter.params.freeze_shape()
    fitter.run_stage("expression", stages.expression, all_pool, batch_size=bs)

    # D: quality mask on the image-matched fit, then estimate the per-clip neutral.
    # IMPORTANT: keep the fitter's params ABSOLUTE (image-matched). Neutralization is
    # a *storage* normalization applied only at export (expr = ψ - neutral; absolute =
    # expr + neutral_expr). Subtracting it in-place would shift the mesh off the image
    # and make viz/metrics render ψ - neutral — e.g. an under-open mouth, because a
    # held smile makes `neutral` itself non-neutral (it can carry a large mouth term).
    metrics = compute_metrics(fitter)
    valid, _ = compute_validity(metrics, cfg)
    neutral = estimate_neutral(fitter.params.expression.detach(), valid, cfg)

    # psi-cleanliness acceptance gate (Section 9).
    psi_samples = _select_psi_samples(fitter.params.expression.detach(), valid)
    score = psi_cleanliness_score(
        model, psi_samples, fitter.params.shape.detach(),
        num_pairs=int(cfg.psi_cleanliness.num_identity_pairs),
    )
    clean_pass = passes(score, float(cfg.psi_cleanliness.consistency_threshold))

    return SequenceResult(
        params=fitter.params.snapshot(),
        neutral=neutral.cpu().numpy(),
        valid=valid,
        metrics=metrics,
        reproj_median=float(np.median(metrics["reproj_med"][valid])) if valid.any() else float("nan"),
        psi_cleanliness=score,
        psi_clean_pass=clean_pass,
        history=fitter.history,
        fitter=fitter,
    )
