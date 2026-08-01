#!/usr/bin/env python
"""Tier 1 — fit GNM to a single frame and (optionally) dump debug viz.

    python src/fit_single_frame.py --image face.png --out debug/ --viz
    python src/fit_single_frame.py --video clip.mp4 --frame 30 --out debug/ --viz
"""

from __future__ import annotations

import argparse

import _common
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image file")
    src.add_argument("--video", help="video file (use --frame to pick a frame)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="debug/single_frame")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mp-model", default=None, help="path to face_landmarker.task")
    args = ap.parse_args()

    from gnm_tracker.config import load_config
    from gnm_tracker.fit import fit_single_frame

    cfg = load_config(args.config)
    device = _common.get_device(cfg, args.device)
    model = _common.load_model(cfg, device)

    if args.image:
        import imageio.v2 as imageio

        frame = np.asarray(imageio.imread(args.image))[..., :3]
    else:
        from gnm_tracker.io import read_video

        frames, _ = read_video(args.video, max_frames=args.frame + 1)
        frame = frames[args.frame]

    detector = _common.make_detector(args.mp_model, running_mode="image")
    detection = detector.detect_image(frame)
    detector.close()
    if not detection.detected:
        raise SystemExit("no face detected in the frame")

    result = fit_single_frame(model, frame, detection, cfg, device=device)
    print(f"median normalized reproj error: {result.median_reproj_error:.4f}")
    print(f"converged: {result.converged}  (threshold {cfg.quality.reproj_error_max})")

    if args.viz:
        from gnm_tracker.viz import dump_debug

        out = dump_debug(args.out, result.fitter, frame[None], cfg)
        print(f"debug viz -> {out}")


if __name__ == "__main__":
    main()
