# GNM Monocular Tracker — Implementation Notes

What was built, how it maps to the build spec (`../../claude/tracker_implementation.md`),
and how to run it. Companion: [RENDERING.md](RENDERING.md) for the visualization/render paths.

## 1. Status

An offline, analysis-by-synthesis tracker that fits Google's **GNM Head** model to
monocular talking-head video and exports, per clip: one **identity/shape** vector,
per-frame **expression (ψ)** and **pose**, a per-frame **validity mask**, and a
per-clip **neutral ψ**. v1 output is GNM parameters only — **no audio**.

- Runs end-to-end **on CPU**. `37/37` unit tests pass.
- Verified on a real TalkingHead-1KH clip: MediaPipe detects the face, the fit
  converges (median normalized reprojection error ≈ 0.031, gate < 0.06), the
  ψ-cleanliness gate passes, quality filtering keeps the good frames, an `.npz`
  record + manifest + full debug viz are written.

## 2. Repository layout

```
gnm_tracker/
  gnm_tracker/            # the package
    io/        video decode / frame extraction / debug-video writer
    detect/    MediaPipe Face Landmarker (468 landmarks + presence/visibility)
    model/     GNM PyTorch wrapper, barycentric landmark layer, camera, clip params
    fit/       losses + staged optimizer (build/stages/single_frame/sequence)
    mouth/     tongue/teeth dimension policy (v1: pin)
    quality/   per-frame validity mask + ψ-cleanliness gate
    viz/       debug visualization (overlays, render, trajectories, loss curves, videos)
    export/    dataset writer (npz + manifest.jsonl + stats)
    validate/  correspondence self-check + ψ-cleanliness CI check
  data/
    correspondence/  vendored MediaPipe↔GNM 468 correspondence (+ self-check + build script)
    groups/          expression_groups.json (tongue dims 350–381) + orientation info
    mediapipe/       face_landmarker.task (downloaded)
  configs/default.yaml   every threshold / loss weight / schedule knob
  src/         Python entry points (one per task)
  scripts/     bash wrappers (only .sh here)
  tests/       pytest unit tests
  docs/        this file + RENDERING.md
```

**Convention:** `src/*.py` are the Python entry points; `scripts/*.sh` are thin bash
wrappers around them. (The Python folder used to be called `scripts/`; it was renamed
to `src/`, and `scripts/` now holds bash only.)

### Entry points (`src/`)
| File | Tier / purpose |
|---|---|
| `dump_gnm_groups.py` | regenerate `data/groups/expression_groups.json` + orientation check |
| `run_tier0_official.py` | Tier 0 — characterize GNM's shipped stack, write a findings report |
| `fit_single_frame.py` | Tier 1 — single-frame fit (+`--viz`) |
| `fit_sequence.py` | Tier 2 — sequence fit → record + debug viz |
| `build_dataset.py` | Tier 3 — run over a folder, export dataset + stats |
| `validate_psi.py` | CI gates: correspondence self-check (+ ψ-cleanliness) |
| `visualize_fit.py` | re-render debug viz from a saved `.npz` |
| `estimate_psi_prior.py` | (optional) build the Mahalanobis ψ-prior precision |
| `render_pyrender.py` | GNM's own pyrender render (Linux/GPU) — see RENDERING.md |

## 3. Mapping to the build spec

| Plan section | Where implemented |
|---|---|
| §2 Design invariants | `model/params.py` (shape is one vector; freeze/thaw), `fit/stages.py` |
| §3 Tech stack | GNM PyTorch backend, MediaPipe Tasks API, OpenCV/matplotlib viz |
| §4 Repo structure | the tree above |
| §5 Milestones (Tier 0–3) | `src/run_tier0_official.py`, `fit_single_frame.py`, `fit_sequence.py`, `build_dataset.py` |
| §6 Losses + schedule | `fit/losses.py`, `fit/stages.py` (Stage A/B/C/D), weights in `configs/default.yaml` |
| §7 Export schema | `export/schema.py`, `export/writer.py` |
| §8 Mouth policy | `mouth/policy.py` (pin; viseme/audio raise, deferred) |
| §9 Quality + ψ-cleanliness | `quality/mask.py`, `quality/psi_cleanliness.py` |
| §10 Correspondence self-check | `validate/correspondence_check.py` (vendored gist self-check) |
| §11 Debug visualization | `viz/` (built alongside the optimizer) |
| §12 Guardrails | enforced structurally (see §6 below) |

## 4. Fitting pipeline & schedule

Detection → staged optimization → neutralize → quality mask → export.
One `Fitter` (`fit/stages.py`) == one clip (so temporal smoothness never crosses
clip boundaries). The camera is an OpenCV pinhole; GNM outputs *world-space* posed
vertices, and the camera carries only intrinsics + a fixed world→camera rotation.

- **Stage A — rigid pose.** `β=0, ψ=0`; optimize neck/head/eye rotations + translation
  using a stable landmark subset (nose bridge, eye corners, chin). Landmark loss only.
- **Stage B — shared shape.** On K sampled frames: optimize one shared `β` + per-frame
  pose + coarse ψ. Pins identity as the time-invariant component.
- **Stage C — expression refine.** **Freeze `β`**, then optimize per-frame ψ + pose on
  all frames (+ temporal smoothness, + mouth pin; photometric is wired but off by
  default — see RENDERING.md). Minibatched over frames; temporal terms on the whole clip.
- **Stage D — neutralize + mask.** Estimate the per-clip neutral ψ from the calmest
  frames, subtract it, then compute the per-frame validity mask.

**Losses** (`fit/losses.py`): confidence-weighted Huber landmark reprojection (primary),
optional photometric L1, ψ prior (Mahalanobis if a precision file exists else strong
L2), identity L2, second-difference temporal smoothness, mouth pin, collision (no-op v1).
Weights are per-stage in `configs/default.yaml: fit.stages.*.weights`.

## 5. Verified GNM facts (v3.0 HEAD)

`V=17821` vertices, `J=4` joints `[neck, head, left_eye, right_eye]`, `I=253` identity
dims, `E=383` expression dims, units = **meters**. Expression groups: `left_eye_region`
0–99, `right_eye_region` 100–199, `lower_face_region` 200–349, `tongue_mean` 350,
`tongue` 351–381, `pupils` 382 → **mouth-interior (pin) dims = 350–381**. Orientation:
`+X`=subject-left, `+Y`=up, `+Z`=forward, so the camera world→cam rotation is a 180°
turn about X (`diag(1,-1,-1)`). Zero-pose ⇒ `world = template + translation` (exact),
and the barycentric landmark layer matches GNM's own `vertices_and_landmarks` exactly.
Regenerate the derived facts with `python src/dump_gnm_groups.py`.

## 6. Design-invariant enforcement (§2 / §12)

- **Shape is one vector per clip** — `ClipParams.shape` is `(I,)`; ψ/pose carry a frame
  axis. Structurally impossible to make identity per-frame.
- **Staged, not joint-from-random** — `Fitter.run_stage` thaws only the named tensors.
- **Lock shape before ψ** — `ClipParams.freeze_shape()` is called before Stage C.
- **Tongue/teeth never free** — the mouth pin is always on (`mouth/policy.py`).
- **No invalid frames exported** — `export` writes the `valid` mask; `build_dataset`
  reports kept/dropped.
- **No temporal smoothing across clips** — one Fitter per clip.
- **No cross-clip identity clustering in v1** — `identity_cluster_id` left null.

## 7. Environment & running

Needs the sibling GNM checkout (`../gnm`) and Python 3.11+. A CPU-torch venv works
(built here with `uv`; `torch`, `gnm -e`, `mediapipe`, `cv2`, `matplotlib` all install
on Python 3.13 arm64). **torch is required for the fit** (the optimization needs autograd
via GNM's PyTorch backend); MediaPipe handles detection; numpy alone can't fit.

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv
pip install -e ../gnm/shape                            # GNM Head (PyTorch backend)
pip install torch mediapipe opencv-python imageio[ffmpeg] matplotlib pyyaml tqdm pytest
# MediaPipe model:
curl -Lo data/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

pytest -q
python src/validate_psi.py --correspondence-only
python src/fit_sequence.py --video <clip>.mp4 --out outputs --viz --device cpu \
       --mp-model data/mediapipe/face_landmarker.task
python src/build_dataset.py --videos-dir ../TalkingHead-1KH/small/cropped_clips --out outputs
```

## 8. Export schema (§7)

Per clip, an `.npz`: `shape (I,)`, `expr (T,E)` (neutral-normalized), `pose (T, J*3+3)`,
`neutral_expr (E,)`, `valid (T,) bool`, `meta_json` (fps, gnm version/variant, landmarks,
reproj median, ψ-cleanliness, `identity_cluster_id=null`, `audio_feat=null` — reserved).
Plus a global `manifest.jsonl` and `dataset_stats.json` (kept/dropped counts, ψ
mean/var, tongue-dim magnitude to confirm the pin behaved).

## 9. Bugs found & fixed while running on CPU

- **GNM is an unhashable `dataclass`-`nn.Module`** → assigning it as a submodule broke
  `.to()`/`.children()`. Fixed by keeping it out of `_modules` (`object.__setattr__`) and
  moving its buffers to device explicitly (`model/gnm_wrapper.py`).
- **Jerk metric over-scaled** — dividing the third difference of ψ by a global `std`
  (dominated by ~380 zero dims) dropped every frame. Now normalized by the clip's robust
  expression amplitude (`quality/mask.py`).

## 10. Not done / next steps

- **Photometric loss** needs a differentiable rasterizer (nvdiffrast=CUDA-only; deferred
  on Mac). The landmark-only path is complete and sufficient. See RENDERING.md.
- **Mahalanobis ψ prior**: run `src/estimate_psi_prior.py` (needs TF) to enable it;
  otherwise a strong L2 prior is used.
- **Full dataset**: run `src/build_dataset.py` over the whole clip folder.
- **Audio (Phase 1) / renderer (Phase 2)**: out of scope for this tracker.
