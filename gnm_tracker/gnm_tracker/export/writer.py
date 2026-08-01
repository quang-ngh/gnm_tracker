"""Write clip records, the global manifest, and dataset statistics (Section 7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnm_tracker.config import resolve_path
from gnm_tracker.export.schema import ClipRecord


def write_record(record: ClipRecord, cfg) -> Path:
    """Write one clip's `.npz` and append its manifest row. Returns the npz path."""
    out_dir = resolve_path(cfg.export.output_dir) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{record.clip_id}.npz"

    save = np.savez_compressed if bool(cfg.export.compress_npz) else np.savez
    save(npz_path, **record.to_npz_dict())

    append_manifest(
        resolve_path(cfg.export.manifest),
        {
            "clip_id": record.clip_id,
            "path": str(npz_path),
            "num_frames": record.num_frames,
            "num_valid": record.num_valid,
            "num_dropped": record.num_frames - record.num_valid,
            "reproj_median": record.meta.get("reproj_median"),
            "psi_clean_pass": record.meta.get("psi_clean_pass"),
        },
    )
    return npz_path


def append_manifest(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_stats(records: list[ClipRecord], cfg) -> Path:
    """Aggregate dataset statistics (Section 7: confirm the mouth policy behaved)."""
    stats_path = resolve_path(cfg.export.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = sum(r.num_frames for r in records)
    total_valid = sum(r.num_valid for r in records)

    # psi distribution over VALID frames only.
    valid_expr = [r.expr[r.valid] for r in records if r.valid.any()]
    if valid_expr:
        expr = np.concatenate(valid_expr, axis=0)
        psi_mean = expr.mean(axis=0)
        psi_var = expr.var(axis=0)
    else:
        n_expr = records[0].expr.shape[1] if records else 0
        psi_mean = np.zeros(n_expr)
        psi_var = np.zeros(n_expr)

    # Tongue-dim distribution — confirm the pin policy kept them ~0.
    import json as _json

    groups = _json.loads(resolve_path(cfg.model.expression_groups_file).read_text())
    tongue = np.array(groups["mouth_interior_expr_dims"], dtype=int)
    tongue_abs_mean = float(np.abs(psi_mean[tongue]).mean()) if len(tongue) else 0.0

    stats = {
        "num_clips": len(records),
        "num_frames": int(total_frames),
        "num_valid_frames": int(total_valid),
        "num_dropped_frames": int(total_frames - total_valid),
        "kept_fraction": float(total_valid / total_frames) if total_frames else 0.0,
        "num_psi_clean_pass": int(sum(bool(r.meta.get("psi_clean_pass")) for r in records)),
        "psi_mean": psi_mean.tolist(),
        "psi_var": psi_var.tolist(),
        "tongue_dims": tongue.tolist(),
        "tongue_abs_mean_over_dataset": tongue_abs_mean,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats_path
