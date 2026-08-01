#!/usr/bin/env python
"""Tier 2 — fit a clip (sequence) and export a dataset record + debug viz.

    python scripts/fit_sequence.py --video clip.mp4 --out outputs/ --viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mp-model", default=None)
    args = ap.parse_args()

    from gnm_tracker.config import load_config
    from gnm_tracker.export import ClipRecord, write_record
    from gnm_tracker.fit import fit_sequence
    from gnm_tracker.io import read_video

    cfg = load_config(args.config)
    device = _common.get_device(cfg, args.device)
    model = _common.load_model(cfg, device)

    frames, fps = read_video(args.video, max_frames=args.max_frames)
    print(f"decoded {len(frames)} frames @ {fps:.2f} fps")

    detector = _common.make_detector(args.mp_model, running_mode="video")
    detections = detector.detect_video(frames, fps)
    detector.close()
    ndet = sum(d.detected for d in detections)
    print(f"detected a face in {ndet}/{len(frames)} frames")

    result = fit_sequence(model, frames, detections, cfg, device=device)
    print(f"reproj median (valid): {result.reproj_median:.4f}")
    print(f"psi-cleanliness mean drift: {result.psi_cleanliness['mean_drift']:.4f} "
          f"-> pass={result.psi_clean_pass}")
    print(f"valid frames: {int(result.valid.sum())}/{len(frames)}")

    clip_id = Path(args.video).stem
    record = ClipRecord.from_sequence_result(clip_id, result, cfg, extra_meta={"fps": fps})
    npz = write_record(record, cfg)
    print(f"record -> {npz}")

    if args.viz:
        from gnm_tracker.viz import dump_debug

        out = dump_debug(Path(args.out) / "debug" / clip_id, result.fitter, frames, cfg, result)
        print(f"debug viz -> {out}")


if __name__ == "__main__":
    main()
