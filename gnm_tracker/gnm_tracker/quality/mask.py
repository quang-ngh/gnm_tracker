"""Per-frame quality mask (Section 9).

A frame is dropped (``valid=False``, never exported — Guardrail) if any of:
  * normalized landmark reprojection error above threshold,
  * MediaPipe presence/visibility below threshold (occlusion / motion blur),
  * temporal jerk (third difference of psi) above threshold,
  * extreme head pose beyond configured yaw/pitch limits,
  * no face detected.

Isolated short valid runs are also dropped (`min_valid_run`).
"""

from __future__ import annotations

import numpy as np

# iBUG-68 outer eye corners (used as a length scale on the sparse-68 set).
_IBUG_EYE_R_OUTER = 36
_IBUG_EYE_L_OUTER = 45


def head_pose_angles_deg(rotations):
    """Yaw/pitch (degrees) of the head from neck∘head joint rotations.

    Zero rotation is frontal (face toward camera, by our convention). Yaw is
    rotation about GNM's up axis (+Y), pitch about +X, read off the posed
    face-forward vector (+Z).
    """
    import torch

    from gnm_tracker.model.camera import axis_angle_to_matrix

    r_neck = axis_angle_to_matrix(rotations[:, 0])   # (T, 3, 3)
    r_head = axis_angle_to_matrix(rotations[:, 1]) if rotations.shape[1] > 1 else r_neck.new_zeros(0)
    r = r_neck @ r_head if rotations.shape[1] > 1 else r_neck
    fwd = r @ torch.tensor([0.0, 0.0, 1.0], dtype=rotations.dtype, device=rotations.device)
    yaw = torch.rad2deg(torch.atan2(fwd[:, 0], fwd[:, 2]))
    pitch = torch.rad2deg(torch.atan2(fwd[:, 1], torch.linalg.norm(fwd[:, [0, 2]], dim=-1)))
    return yaw, pitch


def _third_diff(psi):
    """Per-frame third difference magnitude of psi (temporal jerk), (T,)."""
    import torch

    t = psi.shape[0]
    jerk = psi.new_zeros(t)
    if t >= 4:
        d3 = psi[3:] - 3 * psi[2:-1] + 3 * psi[1:-2] - psi[:-3]
        jerk[3:] = torch.linalg.norm(d3, dim=-1)
    return jerk


def compute_metrics(fitter) -> dict[str, np.ndarray]:
    """Compute the per-frame metrics used by the quality gate."""
    import torch

    with torch.no_grad():
        p = fitter.params
        reproj = fitter.normalized_reproj_error()             # (T, L)
        reproj_med = reproj.median(dim=-1).values             # (T,)
        conf_mean = fitter.targets.conf.mean(dim=-1)          # (T,)
        # Jerk = third difference of psi, made scale-free by the clip's robust
        # expression amplitude (median distance from the mean expression). This
        # flags genuine jumps without penalizing normal fast motion; dividing by
        # a global std (dominated by the ~380 inactive dims) over-scales it.
        psi = p.expression
        amplitude = torch.linalg.norm(psi - psi.mean(dim=0), dim=-1).median().clamp_min(1e-3)
        jerk = _third_diff(psi) / amplitude
        yaw, pitch = head_pose_angles_deg(p.rotations)
        detected = (fitter.targets.conf.sum(dim=-1) > 0)
    return {
        "reproj_med": reproj_med.cpu().numpy(),
        "conf_mean": conf_mean.cpu().numpy(),
        "jerk": jerk.cpu().numpy(),
        "yaw_deg": yaw.cpu().numpy(),
        "pitch_deg": pitch.cpu().numpy(),
        "detected": detected.cpu().numpy(),
    }


def _enforce_min_run(valid: np.ndarray, min_run: int) -> np.ndarray:
    """Drop valid runs shorter than ``min_run`` frames."""
    if min_run <= 1:
        return valid
    out = valid.copy()
    n = len(valid)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if (j - i) < min_run:
            out[i:j] = False
        i = j
    return out


def compute_validity(metrics: dict[str, np.ndarray], cfg) -> tuple[np.ndarray, dict]:
    """Return ``(valid (T,) bool, reasons)`` from per-frame metrics."""
    q = cfg.quality
    # Jerk: absolute floor OR a per-clip robust threshold (median + k*MAD). Real
    # per-frame fits carry a baseline frame-to-frame noise (jerk ~ O(1) relative to
    # amplitude); an absolute-only threshold would drop entire clips, so we adapt to
    # the clip's own noise floor and reject only genuine jumps above it.
    jerk = metrics["jerk"]
    jmed = float(np.median(jerk))
    jmad = float(np.median(np.abs(jerk - jmed)))
    jerk_thr = max(float(q.jerk_max), jmed + float(q.get("jerk_rel_k", 5.0)) * (jmad + 1e-6))
    reasons = {
        "reproj": metrics["reproj_med"] <= float(q.reproj_error_max),
        "confidence": metrics["conf_mean"] >= min(float(q.presence_min), float(q.visibility_min)),
        "jerk": jerk <= jerk_thr,
        "yaw": np.abs(metrics["yaw_deg"]) <= float(q.pose_yaw_max_deg),
        "pitch": np.abs(metrics["pitch_deg"]) <= float(q.pose_pitch_max_deg),
        "detected": metrics["detected"].astype(bool),
    }
    valid = np.ones_like(reasons["detected"], dtype=bool)
    for m in reasons.values():
        valid &= m
    valid = _enforce_min_run(valid, int(q.min_valid_run))
    reasons["final"] = valid
    return valid, reasons
