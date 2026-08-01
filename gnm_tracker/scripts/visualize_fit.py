#!/usr/bin/env python
"""Re-visualize a saved dataset record over its source video (Section 11).

    python scripts/visualize_fit.py --record outputs/clips/<id>.npz --video clip.mp4 --out debug/

Rebuilds a Fitter for the clip, loads the exported params, and dumps the debug
folder (overlays, video, plots) so a saved fit can be eyeballed after the fact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="clip .npz")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="debug/visualize")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mp-model", default=None)
    args = ap.parse_args()

    import torch

    from gnm_tracker.config import load_config
    from gnm_tracker.fit import build_fitter
    from gnm_tracker.io import read_video
    from gnm_tracker.viz import dump_debug

    cfg = load_config(args.config)
    device = _common.get_device(cfg, args.device)
    model = _common.load_model(cfg, device)

    data = np.load(args.record, allow_pickle=True)
    frames, fps = read_video(args.video, max_frames=args.max_frames)

    detector = _common.make_detector(args.mp_model, running_mode="video")
    detections = detector.detect_video(frames, fps)
    detector.close()

    fitter = build_fitter(model, cfg, frames, detections, device=device)

    # Load exported params back into the fitter (add neutral back so the mesh
    # matches the image; the record stores neutral-normalized expression).
    j = model.num_joints
    pose = data["pose"]
    rotations = pose[:, : j * 3].reshape(-1, j, 3)
    translation = pose[:, j * 3 :]
    expr = data["expr"] + data["neutral_expr"][None]
    with torch.no_grad():
        fitter.params.shape.copy_(torch.as_tensor(data["shape"], device=device))
        fitter.params.expression.copy_(torch.as_tensor(expr, device=device))
        fitter.params.rotations.copy_(torch.as_tensor(rotations, device=device))
        fitter.params.translation.copy_(torch.as_tensor(translation, device=device))

    out = dump_debug(args.out, fitter, frames, cfg)
    print(f"debug viz -> {out}")


if __name__ == "__main__":
    main()
