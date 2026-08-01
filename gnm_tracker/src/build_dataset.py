#!/usr/bin/env python
"""Tier 3 — dataset production line.

Runs the sequence fitter over a folder of clips, applies quality filtering, emits
the export schema (Section 7), a manifest with per-clip validity, and dataset
statistics.

    python src/build_dataset.py --videos-dir ../TalkingHead-1KH/small/cropped_clips --out outputs/
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import _common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap number of clips")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mp-model", default=None)
    args = ap.parse_args()

    from gnm_tracker.config import load_config
    from gnm_tracker.export import ClipRecord, write_record, write_stats
    from gnm_tracker.fit import fit_sequence
    from gnm_tracker.io import read_video

    cfg = load_config(args.config)
    device = _common.get_device(cfg, args.device)
    model = _common.load_model(cfg, device)
    detector = _common.make_detector(args.mp_model, running_mode="video")

    videos = sorted(Path(args.videos_dir).glob(args.glob))
    if args.limit:
        videos = videos[: args.limit]
    print(f"found {len(videos)} clips under {args.videos_dir}")

    records = []
    for i, video in enumerate(videos):
        clip_id = video.stem
        print(f"[{i + 1}/{len(videos)}] {clip_id}")
        try:
            frames, fps = read_video(video, max_frames=args.max_frames)
            detections = detector.detect_video(frames, fps)
            result = fit_sequence(model, frames, detections, cfg, device=device)
            record = ClipRecord.from_sequence_result(clip_id, result, cfg, extra_meta={"fps": fps})
            write_record(record, cfg)
            records.append(record)
            print(f"    valid {record.num_valid}/{record.num_frames}, "
                  f"psi_clean_pass={result.psi_clean_pass}")
            if args.viz:
                from gnm_tracker.viz import dump_debug

                dump_debug(Path(args.out) / "debug" / clip_id, result.fitter, frames, cfg, result)
        except Exception:  # keep the pipeline going on a bad clip
            print(f"    FAILED {clip_id}:\n{traceback.format_exc()}")

    detector.close()
    if records:
        stats = write_stats(records, cfg)
        print(f"dataset stats -> {stats}")
    print(f"done: {len(records)}/{len(videos)} clips exported")


if __name__ == "__main__":
    main()
