# GNM Monocular Tracker — Implementation Plan

> **Purpose of this document.** This is a build spec for a coding agent. It describes a
> monocular, offline **tracker** that fits Google's GNM Head model to in-the-wild talking
> videos. The tracker exists to **produce a dataset** for a downstream talking-head pipeline.
> Read the *Design invariants* and *Guardrails* sections carefully — they encode decisions
> that are easy to get wrong and expensive to fix later.
>
> **v1 scope (read first):** the output is **GNM parameters** (shape / ψ / pose) — ready to feed
> straight into a model or renderer. **No audio needed** at this stage. And **debug
> visualization must exist from the start** (Section 11) — this is required, not optional.

---

## 1. Goal & scope

**Goal.** Given a folder of monocular videos (one face per video, arbitrary view/pose,
uncontrolled lighting), output for each video a per-frame sequence of **GNM expression (ψ)
and pose** plus a **single per-video identity/shape vector** and a per-frame quality mask.
These are **GNM parameters** — ready to feed straight into a model/renderer. Frame-aligned
audio features are **deferred**; not needed at this stage.

**In scope**
- Analysis-by-synthesis fitting of GNM (identity + expression + pose) to video.
- Offline, batch, multi-pass optimization (NOT real-time).
- MediaPipe-based 2D landmark supervision + optional photometric refinement.
- **Debug visualization** (Section 11) — first-class, built alongside the optimizer.
- Quality filtering and a ψ-cleanliness acceptance gate.
- Dataset export of GNM parameters in a well-defined schema.

**Explicitly NOT in scope (do not build these here)**
- Any audio handling (feature extraction/alignment). v1 output is GNM parameters only.
- The audio→ψ model itself (Phase 1). We only produce its training parameters.
- The renderer / Gaussian appearance layer (Phase 2).
- Real-time or streaming inference. This tracker is offline by design.
- Multi-person / multi-face video. Assume exactly one face per clip.

---

## 2. Design invariants (do not violate)

1. **Identity shape is ONE vector per video**, static across all frames. Expression ψ and
   pose are **per-frame**. Never let shape vary per frame.
2. **Coarse-to-fine, staged optimization.** Never release all parameters to free joint
   optimization from random init — that is the main cause of identity↔expression leakage.
   Order: rigid pose → shared shape → per-frame expression.
3. **Lock shape before refining per-frame ψ.** Once shape is solved on a batch of frames,
   freeze it; only then optimize ψ per-frame. This is what stops ψ from absorbing shape error.
4. **The dataset's blind spots become the model's blind spots.** The tongue/teeth ψ
   dimensions are NOT observable from pixels. Decide their policy explicitly (Section 8) —
   leaving them free produces garbage; zeroing them teaches the downstream model that the
   tongue never moves.
5. **Acceptance is ψ-cleanliness, not per-frame PSNR.** A fit that matches the image but
   whose ψ is identity-contaminated is a *failed* fit for our purpose. See Section 9.
6. **Never emit fitted frames that failed quality checks into the dataset.** Filtering is a
   deliverable, not optional.

---

## 3. Tech stack

- **Language / DL framework:** Python 3.11+, PyTorch. Use GNM's **PyTorch backend** so the
  model is a differentiable layer and gradients flow to identity/expression/pose.
- **GNM:** `google/GNM`, v3.0 **HEAD** variant (17,821 vertices), Apache 2.0. Use its shipped
  landmark files and, if present, its official perception/fitting stack.
- **Landmark detector:** MediaPipe Face Landmarker (Tasks API) → 468 landmarks, per-landmark
  presence/visibility, and the face transform matrix. Optionally its 52 blendshape scores
  (only as a weak initializer, never as ground truth).
- **MediaPipe↔GNM correspondence:** the community gist `head_dense_468.txt`
  (468 rows, `i0 w0 i1 w1 i2 w2` barycentric on GNM topology, identity/expression/pose
  invariant). Vendor the file into `data/`. **Run its self-check on our exact GNM version
  before trusting it** (see Section 10). It is unofficial and built for v3.0 HEAD — if our
  topology differs, regenerate with its build script.
- **Differentiable rasterizer (for photometric stage only):** `nvdiffrast` (preferred, fast)
  or PyTorch3D (easier). Landmark-only stages need only a camera projection, no rasterizer.
- **Visualization:** OpenCV (overlays), matplotlib (plots), imageio/ffmpeg (debug videos).
  Simple and sufficient; no fancy interactive viewer in v1 (rerun.io optional).
- **Audio features:** **DEFERRED** — not needed at this stage. The tracker does not consume
  audio and the output is GNM parameters only. Add wav2vec 2.0 / HuBERT at a later export step
  when needed (to train Phase 1, or to enable the audio-driven mouth mode in Section 8).

---

## 4. Repository structure

```
gnm_tracker/
  data/                     # head_dense_468.txt, camera defaults, dim indices
  gnm_tracker/
    io/                     # video decode, frame extraction
    detect/                 # MediaPipe landmarks + confidence
    model/                  # GNM PyTorch wrapper; 468 barycentric landmark layer; camera
    fit/                    # optimizer stages, losses, schedule
    mouth/                  # tongue/teeth dimension policy
    quality/                # reprojection / jerk filtering, validity mask
    viz/                    # debug visualization tooling (see Section 11)
    export/                 # dataset writer (GNM parameters)
    validate/               # gist self-check, ψ-cleanliness cross-identity test
  scripts/
    run_tier0_official.py   # try GNM's official fitting stack first
    fit_single_frame.py
    fit_sequence.py
    build_dataset.py
    validate_psi.py
    visualize_fit.py        # overlay the fit on video for eyeballing
  configs/                  # yaml: loss weights, schedule, paths
  tests/
```

---

## 5. Milestones (build in this order; each has a gate)

### Tier 0 — Baseline with GNM's official fitting stack
Before writing a custom optimizer, run GNM's shipped perception/fitting stack on 3–5 clips.
**Deliverable:** a short report of what it fits well, where it fails, and what we must build.
**Gate:** if the official stack is already good enough on our footage, stop and use it. Only
proceed to Tier 1 for the parts it cannot handle.

### Tier 1 — Single-frame fitter (the optimizer core) + visualization
Analysis-by-synthesis on one image: detect 468 landmarks → map to target GNM surface points
via the barycentric correspondence → optimize pose, then shape, then coarse ψ (still single
frame). Landmark reprojection loss + regularizers; no photometric yet.
**Required:** build the debug visualization tooling (Section 11) starting at Tier 1 — you cannot
catch leakage or misalignment without eyeballing the fit. Build `viz/` **alongside** the
optimizer, not after.
**Gate:** stable, sane convergence on a variety of single frames (median normalized reprojection
error below the threshold in `configs`), no NaNs, no exploded ψ, and overlay looks aligned.

### Tier 2 — Sequence fitter with disentanglement (quality lives or dies here)
- Sample K frames spanning the clip's expression/pose range.
- Solve **one shared shape** across those K frames jointly with per-frame pose + coarse ψ.
- **Lock shape.** Then optimize per-frame ψ + pose on *all* frames, add photometric loss,
  add temporal smoothness, apply the mouth policy.
- Estimate per-clip neutral ψ and normalize.
**Gate:** passes the ψ-cleanliness test (Section 9) at the configured threshold; the full-clip
overlay video is jitter-free.

### Tier 3 — Dataset production line
Run at scale over the video folder; apply quality filtering; emit the export schema
(Section 7); write a manifest with per-clip and per-frame validity. Produce dataset statistics
(num clips, frames kept/dropped, ψ distribution).
**Gate:** a held-out sample of exported clips passes manual spot-check + automated gates.

---

## 6. Losses and optimization schedule

Notation: identity `β`, expression `ψ`, per-frame pose `θ` (global rotation+translation,
neck, eyes), camera projection `π`. `P_i(β,ψ,θ)` = the i-th landmark's 3D position, computed
as the barycentric combination of GNM vertices (identity/expression-invariant anchors), then
posed. `x_i` = detected 2D landmark, `c_i` = its confidence.

**Landmark reprojection (primary signal):**
`L_lmk = Σ_i c_i · huber( π(P_i(β,ψ,θ)) − x_i )`

**Photometric (refinement stage only):**
`L_pho = || M_face ⊙ (render(β,ψ,θ) − I) ||_1`, where `M_face` excludes hair/background and
the interior mouth.

**Regularizers:**
- `L_ψ = ψᵀ Σ⁻¹ ψ` — Mahalanobis under GNM's expression prior if available (GNM ships an
  ExpressionSampler; use its statistics), else plain L2. Keep this **strong**.
- `L_β = ||β||²` — identity prior.
- `L_temp = Σ_t ||ψ_t − 2ψ_{t−1} + ψ_{t−2}||² + (same on pose)` — second-difference temporal
  smoothness, **within a clip only** (never across clip boundaries).
- `L_mouth` — see Section 8.
- `L_col` (optional/advanced) — penalize interpenetration between teeth/tongue and lips.

**Total:** `L = λ_lmk L_lmk + λ_pho L_pho + λ_ψ L_ψ + λ_β L_β + λ_temp L_temp + λ_mouth L_mouth (+ λ_col L_col)`
All weights in `configs`; different stages use different weight sets.

**Schedule (per clip):**
- **Stage A — rigid pose.** Optimize global rotation + translation (and a weakly-optimized or
  assumed focal length; in-the-wild intrinsics are unknown). `β=0, ψ=0`. Use only a subset of
  stable landmarks (nose bridge, eye corners, jaw). Landmark loss only.
- **Stage B — shared shape.** On K sampled frames: optimize `β` (shared) + per-frame `θ` +
  coarse `ψ`. Landmark loss + `L_β + L_ψ`. This is where identity is pinned as the
  time-invariant component.
- **Stage C — expression refine.** **Freeze `β`.** Optimize per-frame `ψ + θ` on all frames.
  Add `L_pho`, `L_temp`, apply mouth policy. Batch frames for speed.
- **Stage D — neutralize + mask.** Estimate per-clip neutral `ψ` (e.g. robust min-motion
  frames), store it so "expression = 0" means the same thing across clips; compute quality
  mask (Section 9).

Optimizers: Adam for early stages; L-BFGS optional for final per-frame polish. Consider an
image pyramid for the photometric stage.

---

## 7. Output / dataset schema

One record per clip (e.g. an `.npz` plus a row in a global `manifest.jsonl`):

- `shape` — `float32[n_id]`, the single identity vector for this clip.
- `expr` — `float32[T, n_expr]`, per-frame ψ (already neutral-normalized).
- `pose` — `float32[T, n_pose]`, per-frame global/neck/eye pose + translation.
- `neutral_expr` — `float32[n_expr]`, the per-clip neutral used for normalization.
- `valid` — `bool[T]`, per-frame quality mask (False = do not train on this frame).
- `meta` — fps, assumed/estimated intrinsics, GNM version+variant, detector version, clip id,
  and an optional `identity_cluster_id` (left null in v1; see Guardrails).
- `audio_feat` — **DEFERRED, not exported in v1.** Current output is GNM parameters only
  (shape/expr/pose). Reserve the slot so it can be added later without breaking the format.

Also emit: `dataset_stats.json` (clip count, kept/dropped frames, ψ mean/variance per dim,
distribution of tongue/teeth dims to confirm the mouth policy behaved).

---

## 8. Mouth-interior (tongue/teeth) policy — decide, don't drift

The tongue/teeth ψ dimensions are **not observable from a single camera**. First, identify
those dimension indices in GNM's expression space and expose them as a named group in
`data/`. Then implement a configurable policy with three modes:

- `pin` (default v1): strongly regularize (or hard-zero) tongue/teeth dims via `L_mouth`, so
  they never take image-driven garbage values. Simple, safe, but the downstream model will
  learn a static tongue. **Since v1 uses no audio, this is the only runnable mode right now.**
- `viseme` (needs audio, deferred): drive tongue/teeth dims from a phoneme→viseme prior (needs a
  forced aligner on the audio track). Better mouth realism without full audio-visual fitting.
- `audio` (research extension, needs audio): condition those dims on audio/phoneme features
  **inside** the fit, so the dataset contains plausible, audio-correlated tongue motion. This is
  the item that both removes the dataset blind spot and constitutes the project's novel
  contribution; implement behind a flag once audio is introduced.

`L_mouth` must always keep these dims bounded whenever they are not being actively driven.

---

## 9. Quality filtering & the ψ-cleanliness gate

**Per-frame quality mask** (`valid`): drop a frame if any of —
- normalized landmark reprojection error above threshold,
- MediaPipe presence/visibility below threshold (occlusion, motion blur),
- temporal jerk (third difference of ψ/pose) above threshold,
- extreme head pose beyond a configured yaw/pitch limit.

**ψ-cleanliness acceptance test** (the real gate for Tiers 2–3). The property we need is
"same expression ⇒ same ψ, regardless of identity." Test it by **cross-identity expression
transfer**: take ψ fitted on person A, apply it to person B's shape, render, and check it
produces the *same expression* on B (not a distortion that echoes A's face). Quantify with a
landmark/geometry consistency metric against a reference and assert it stays below threshold.
If it fails, ψ is identity-contaminated — revisit Stage B (shape fit) and increase `L_ψ`
before proceeding. Wire this into CI as `scripts/validate_psi.py`.

---

## 10. Correspondence validation (the eyelid gotcha)

Before using `head_dense_468.txt`, run the gist's self-check on **our** GNM build. It asserts
that structures like the palpebral aperture, mouth width, and inner-lip gap remain resolvable
— a landmark embedding can be numerically valid yet silently collapse upper/lower eyelid
landmarks onto the same triangle (aperture reads ~0.5 mm instead of ~6 mm), breaking any
eye-closure/blink signal with no error raised. If the self-check fails, regenerate the
correspondence with its build script against our topology. Also confirm MediaPipe left/right
are anatomical (subject) sides, per the gist's verified indices.

---

## 11. Debug visualization (build with the optimizer, not after)

Fitting a 3DMM without seeing the result is nearly undebuggable — id↔expression leakage, pose
drift, jumping landmarks, exploded ψ are all **invisible in the numbers** but obvious the moment
you overlay the fit on the image. Treat this as a first-class tool, not a nicety. Every fit
script must take a `--viz` flag, and each fit run should dump a debug folder (a few sample-frame
overlays + the clip video + the plots).

Minimum set:

- **Mesh overlay on the frame.** Draw the fitted GNM mesh (wireframe or semi-transparent render)
  over the input image. The #1 debug tool — bad pose/shape shows instantly.
- **Landmarks: reprojected vs detected.** Plot the reprojected GNM landmarks (via the
  correspondence) and the detected MediaPipe landmarks on the same frame, joined by residual
  segments, colored by error magnitude. Shows immediately which landmark is pulling wrong.
- **Full-clip overlay video.** Export an mp4 overlaying the fit on the whole clip. **Temporal
  jitter is only visible in motion**, never in a still frame — this is how you catch it.
- **ψ and pose trajectories over time.** Line-plot ψ/pose dims across frames — spot jumps,
  discontinuities, or ψ hitting its bounds. Include a separate monitor for the tongue/teeth dims
  to confirm the `pin` policy keeps them bounded.
- **Per-stage loss curves.** Plot landmark/photometric/each regularizer over iterations, split
  by Stage A/B/C — to see which stage fails to converge.
- **Fit render beside the input.** For the photometric stage: rendered mesh next to the input.
- **ψ-cleanliness visualization.** Render person A's ψ applied to person B's shape (Section 9) as
  images, so a human can confirm it reads as the *same expression* rather than distorting toward A.

Implementation: keep it simple — OpenCV overlays, matplotlib plots, imageio/ffmpeg for debug
videos. No fancy interactive viewer needed in v1 (rerun.io optional).

---

## 12. Guardrails — things the agent must NOT do

- Do **not** let identity `β` vary per frame, or optimize `β, ψ, θ` jointly from random init.
- Do **not** apply temporal smoothness across clip boundaries.
- Do **not** leave tongue/teeth dims free; always apply the mouth policy (v1: `pin`).
- Do **not** export frames where `valid == False`.
- Do **not** add any audio handling in v1 (output is GNM parameters only).
- Do **not** defer the visualization; build `viz/` alongside the optimizer from Tier 1.
- Do **not** treat MediaPipe's 52 blendshapes as ground-truth expression; at most a weak init.
- Do **not** trust the vendored correspondence without running its self-check on our GNM build.
- Do **not** add cross-clip identity clustering (ArcFace) in v1. We are not assuming any
  person appears in multiple clips; per-clip shape + per-clip neutral-ψ normalization is the
  v1 approach. Leave a clean extension point (`identity_cluster_id`) but keep it null. Only
  build clustering if the ψ-cleanliness gate later shows cross-clip inconsistency AND we
  confirm repeated identities exist.

---

## 13. Suggested first actions for the agent

1. Scaffold the repo (Section 4) and a `config` with all thresholds/weights as named knobs.
2. Implement `model/` — GNM PyTorch wrapper + the 468 barycentric landmark layer + a camera
   projection module — and get `scripts/validate_psi.py` and the gist self-check running on
   the neutral mesh first (cheap, catches setup errors early).
3. Run `run_tier0_official.py` (Tier 0) on a few clips and write the findings report.
4. Only then implement Tier 1 — **and build `viz/` alongside the optimizer from the very first
   frame** (Section 11); gate on its acceptance criterion, and proceed upward.
