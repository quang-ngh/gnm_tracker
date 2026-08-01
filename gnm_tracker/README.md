# GNM Monocular Tracker

Offline, analysis-by-synthesis fitting of Google's **GNM Head** model to
monocular talking-head video. For each clip it produces a dataset record of
per-frame **expression (ψ)** and **pose**, a single per-clip **identity/shape**
vector, and a per-frame **quality mask** — the training parameters for a
downstream talking-head pipeline. **No audio in v1.**

See `../claude/tracker_implementation.md` for the full build spec. This
implementation follows its design invariants (Section 2), staged schedule
(Section 6), mouth policy (Section 8), ψ-cleanliness gate (Section 9) and
first-class debug visualization (Section 11).

## Layout

```
gnm_tracker/
  io/         video decode / frame extraction
  detect/     MediaPipe Face Landmarker (468 landmarks + confidence)
  model/      GNM PyTorch wrapper, barycentric landmark layer, camera, clip params
  fit/        losses + staged optimizer (A rigid, B shape, C expression, D neutralize)
  mouth/      tongue/teeth dimension policy (v1: pin)
  quality/    per-frame validity mask + ψ-cleanliness cross-identity test
  viz/        debug visualization (overlays, trajectories, loss curves, videos)
  export/     dataset writer (npz + manifest.jsonl + stats)
  validate/   correspondence self-check + ψ-cleanliness gate
data/
  correspondence/  vendored MediaPipe↔GNM 468 correspondence (+ self-check)
  groups/          expression-dim groups (tongue 350..381) + orientation info
configs/     default.yaml (every threshold / loss weight / schedule knob)
scripts/     tier0, fit_single_frame, fit_sequence, build_dataset, validate_psi, visualize_fit
tests/
```

## Setup

Requires the sibling GNM checkout at `../gnm`. Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../gnm/shape[pytorch]      # GNM Head, PyTorch backend
pip install -e .[dev]                      # this package + pytest
# regenerate the derived group/orientation file (optional; already vendored):
python scripts/dump_gnm_groups.py
```

`nvdiffrast` (photometric stage) needs CUDA; on Apple Silicon use `pytorch3d`
(`.[photometric-mac]`) or stay on the OpenCV viz fallback — landmark-only stages
(Tier 0/1 and most of Tier 2) run fine on CPU/MPS.

## Milestones (Section 5)

| Tier | Script | Gate |
|---|---|---|
| 0 baseline | `scripts/run_tier0_official.py` | is GNM's shipped stack enough? |
| 1 single-frame | `scripts/fit_single_frame.py --viz` | stable convergence, overlay aligned |
| 2 sequence | `scripts/fit_sequence.py --viz` | passes ψ-cleanliness, jitter-free |
| 3 dataset | `scripts/build_dataset.py` | spot-check + automated gates pass |

## Quick start

```bash
# validate the vendored correspondence on our exact GNM build (eyelid gotcha):
python scripts/validate_psi.py --correspondence-only
# fit one clip and dump debug viz:
python scripts/fit_sequence.py --video path/to/clip.mp4 --out outputs/ --viz
```
