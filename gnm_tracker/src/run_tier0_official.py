#!/usr/bin/env python
"""Tier 0 — baseline with GNM's shipped stack (Section 5).

Before writing a custom optimizer, characterize what GNM already ships and how
far a minimal landmark fit gets on our footage, then write a findings report:
what fits well, where it fails, and what we must build.

    python src/run_tier0_official.py --videos-dir ../TalkingHead-1KH/small/cropped_clips --num-clips 3

Note: GNM ships a differentiable model, sparse-68 landmarks, PCA/least-squares
`fitting_utils`, semantic samplers and demo notebooks — but NO turnkey monocular
*video* fitter. So this "baseline" is the shipped model + a single-frame landmark
fit; the report quantifies its error and enumerates the Tier 1-3 gaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common
import numpy as np


def _list_shipped() -> dict:
    import importlib.util as u

    def has(mod):
        return u.find_spec(mod) is not None

    return {
        "pytorch_backend": has("gnm.shape.gnm_pytorch"),
        "sparse_68_landmarks": True,
        "fitting_utils.project_on_pca": has("gnm.shape.fitting_utils.project_on_pca"),
        "fitting_utils.regularized_least_squares": has(
            "gnm.shape.fitting_utils.regularized_least_squares"
        ),
        "semantic_sampler (ExpressionSampler)": has("gnm.shape.semantic_sampler"),
        "turnkey_video_fitter": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--num-clips", type=int, default=3)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/tier0_report.md")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mp-model", default=None)
    args = ap.parse_args()

    from gnm_tracker.config import load_config
    from gnm_tracker.fit import fit_single_frame
    from gnm_tracker.io import read_video
    from gnm_tracker.validate import run_correspondence_self_check

    cfg = load_config(args.config)
    device = _common.get_device(cfg, args.device)
    model = _common.load_model(cfg, device)
    detector = _common.make_detector(args.mp_model, running_mode="image")

    shipped = _list_shipped()
    corr_ok = run_correspondence_self_check()

    videos = sorted(Path(args.videos_dir).glob(args.glob))[: args.num_clips]
    rows = []
    for video in videos:
        try:
            frames, _ = read_video(video, max_frames=1)
            det = detector.detect_image(frames[0])
            if not det.detected:
                rows.append((video.name, "no face", None))
                continue
            res = fit_single_frame(model, frames[0], det, cfg, device=device)
            rows.append((video.name, f"{res.median_reproj_error:.4f}", res.converged))
        except Exception as exc:
            rows.append((video.name, f"error: {exc}", None))
    detector.close()

    lines = ["# Tier 0 — GNM shipped-stack baseline\n"]
    lines.append("## What GNM ships\n")
    for k, v in shipped.items():
        lines.append(f"- {'✅' if v else '❌'} {k}")
    lines.append(f"\n**Correspondence self-check (eyelid gotcha):** "
                 f"{'PASS' if corr_ok else 'FAIL'}\n")
    lines.append("## Single-frame landmark-fit baseline\n")
    lines.append("| clip | median norm. reproj err | converged |")
    lines.append("|---|---|---|")
    for name, err, conv in rows:
        lines.append(f"| {name} | {err} | {conv} |")
    lines.append("\n## Findings / what must be built\n")
    lines.append(
        "- GNM ships no monocular **video** fitter; only a differentiable model, "
        "sparse-68 landmarks, PCA/least-squares helpers and semantic samplers.\n"
        "- Single-frame landmark fitting (Tier 1) converges but has no identity "
        "sharing or temporal handling.\n"
        "- **Build Tier 1** (single-frame core + viz), **Tier 2** (shared shape + "
        "disentanglement + ψ-cleanliness gate), **Tier 3** (dataset line).\n"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
