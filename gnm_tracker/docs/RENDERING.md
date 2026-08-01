# Rendering & debug visualization

Two render paths produce the "GNM-teaser" look (blue head, visible eyes). Both
**reuse GNM's own rendering ingredients**; they differ only in the rasterizer.

| Path | File | Where it runs | Rasterizer |
|---|---|---|---|
| Software (portable) | `gnm_tracker/viz/render_compare.py` | anywhere incl. **macOS/CPU** | our z-buffered Gouraud raster |
| GNM pyrender (native) | `src/render_pyrender.py` + `scripts/render_pyrender.sh` | **Linux/GPU** (needs OSMesa/EGL) | `gnm.shape.visualization.gnm_pyrender.render` |

## GNM ingredients reused by both

- **Per-vertex colors:** `gnm.shape.visualization.vertex_colors.get_vertex_colors`
  (skin = the demo blue `50,156,237`, bright scleras, dark irises, teeth). GNM leaves
  non-highlighted groups (eye sockets/interior, ears, gums) at black; we **fill those
  with the skin color** so the eyes read as eyeballs, not black holes.
- **Per-vertex normals:** GNM's `compute_vertex_normals`.
- **Topology:** GNM triangles (per mesh component for pyrender: skin, eyes, teeth, tongue).
- **Lighting:** a camera head-light, matching `gnm_pyrender`'s DirectionalLight-on-camera.

## Why pyrender can't run on macOS

`gnm_pyrender` uses OpenGL offscreen (`PYOPENGL_PLATFORM=osmesa`). macOS (Apple Silicon)
has **no system EGL and no OSMesa** — Apple deprecated OpenGL for Metal, and neither
`brew` nor `conda` mesa is present here. Building Mesa/OSMesa on arm64 mac is a yak-shave
and pyrender's mac support is poor. So on this machine we use the software rasterizer,
which yields the same look (same colors, normals, headlight). On Linux, `render_pyrender`
gives the true GL render.

## Camera bridge (software ⇄ pyrender)

The tracker camera is an **OpenCV pinhole**: `cam = R_wc · world`, `u = f·x/z + cx`, with
the head positioned by GNM's translation in world space (so the extrinsic is
rotation-only, `R_wc = diag(1,-1,-1)`). `render_pyrender.py` converts intrinsics +
extrinsics to OpenGL using **GNM's own** `camera_conversions`:

- `opencv_intrinsics_matrix_to_opengl_view_matrix(K, W, H, near, far)` → `camera_to_image`
- `opencv_extrinsics_to_opengl(world_to_camera_cv)` → `world_to_camera`

## Debug outputs

`viz.dump_debug` (called by `fit_sequence --viz` and `visualize_fit`) writes to the debug
folder:

- `overlays/render_XXXX.png` — **input | GNM-colored fit render** (side-by-side).
- `overlays/mesh_XXXX.png` — semi-transparent shaded mesh over the frame.
- `overlays/resid_XXXX.png` — reprojected (red) vs detected (white) landmarks, residuals
  colored by error.
- `side_by_side.mp4` — the full clip, `input | render` per frame (jitter is only visible
  in motion).
- `trajectories.png` — ψ / pose over time + a tongue-dim monitor (pin should keep ~0).
- `loss_curves.png` — per-stage loss curves.
- `psi_cleanliness.png` — A's ψ on B's shape (same expression ⇒ clean).

Mesh color / opacity of the overlay: `configs/default.yaml: viz.mesh_color`,
`viz.mesh_alpha`. Render skin color / ambient: constants in `viz/render_compare.py`.

## Running each

Software (macOS/CPU or anywhere):
```bash
python src/fit_sequence.py --video <clip>.mp4 --out outputs --viz --device cpu \
       --mp-model data/mediapipe/face_landmarker.task
# or re-render from a saved record:
python src/visualize_fit.py --record outputs/clips/<id>.npz --video <clip>.mp4 --out debug/
```

GNM pyrender (Linux):
```bash
# apt-get install libosmesa6 libosmesa6-dev ; pip install pyrender pyopengl
scripts/render_pyrender.sh outputs/clips/<id>.npz <clip>.mp4 outputs/pyrender_<id>.mp4 osmesa
# GPU/EGL:  ... egl     (or PYTHON=.venv/bin/python  MAX_FRAMES=60  scripts/render_pyrender.sh ...)
```

## Photometric note

The photometric *loss* (Stage C) needs a **differentiable** rasterizer (nvdiffrast, CUDA
only; or PyTorch3D). It is wired in `fit/stages.py` behind an optional `renderer` and is
off by default, so the landmark-only fit runs everywhere. The renderers here (software +
pyrender) are for **eyeballing only**, not gradients.
