# MediaPipe 468-landmark correspondence for GNM Head

An unofficial mapping from MediaPipe FaceMesh's 468 canonical landmarks onto
Google's [GNM](https://github.com/google/GNM) head model, in the same file format
as GNM's shipped `head_sparse_68.txt`.

Built because MediaPipe detection is generally more robust than 68-point FAN/iBUG
detectors, and GNM currently ships only the 68-point correspondence.

**Not affiliated with Google. Community-built, provided as-is.**

| File | What it is |
|---|---|
| `head_dense_468.txt` | The correspondence: 468 rows, `i0 w0 i1 w1 i2 w2` (barycentric) |
| `gnm_mediapipe_landmarks.py` | Loader + self-check (run it to verify) |
| `build_mediapipe_correspondence.py` | Regenerate it from scratch |

Built against **GNM v3.0, `HEAD` variant** (17,821 vertices). If GNM's topology
changes, regenerate rather than reusing the file.

## Use

GNM's `gnm_landmarks.load_landmarks()` dispatches off the `GNMLandmarksType` enum
and only knows `HEAD_SPARSE_68`, so read this file directly:

```python
import numpy as np
from gnm.shape import gnm_numpy

emb = np.loadtxt("head_dense_468.txt")
idx, w = emb[:, ::2].astype(int), emb[:, 1::2]        # (468,3) each

gnm = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,
                               variant=gnm_numpy.GNMVariant.HEAD)
verts = np.asarray(gnm(None, None, None, None))       # any identity/expression/pose
landmarks = (verts[idx] * w[..., None]).sum(axis=1)   # -> (468, 3)
```

The anchors are barycentric on the GNM topology, so the same file works for every
identity, expression and pose without re-registration.

## How it was built

1. Compute GNM's shipped 68 sparse landmarks in 3D.
2. Map the canonical 468 mesh to those 68 via the standard iBUG ordering and solve
   a Umeyama similarity transform. The L/R-mirrored ordering is also tried and the
   lower-residual fit wins — worth doing, since landmark-set conventions differ
   between sources (here: 18.84 mm normal vs 47.88 mm mirrored, so normal).
3. Project each canonical vertex onto the GNM surface, **constrained to triangles
   touching an anatomically appropriate vertex group**, and store the barycentric
   anchor.

## Accuracy

Distance from each embedded landmark to its similarity-aligned canonical position:

| | mm |
|---|---|
| mean | 2.59 |
| median | 2.49 |
| p95 | 5.09 |
| max | 6.65 |

Per-landmark, for commonly used indices: mouth corners (61, 291) **0.1–0.3 mm**;
inner lips (13, 14) **0.7–2.5 mm**; eye outer corners (33, 263) **4.7–4.9 mm**.

Largest residuals sit at the face-oval boundary, where MediaPipe's canonical mesh
stops at the hairline and GNM continues over the skull. Those landmarks slide
along the surface; if your application depends on the face outline, check them.

This is agreement with the aligned canonical mesh, **not** ground truth — the two
models simply have different face shapes, and some of this residual is that
difference rather than registration error.

## The eyelid gotcha

If you build your own correspondence, this will probably bite you.

The MediaPipe eyelid-margin landmarks (159/145, 386/374 and neighbours) must be
projected onto GNM's **`eye_sockets`** group, **split into upper and lower bands
per eye**. Project them onto the surrounding orbital skin — or onto one undivided
socket region — and the upper and lower lid landmarks snap to the *same triangle*.

The palpebral aperture then reads **~0.5 mm instead of ~6.3 mm**.

Nothing raises an error. Projection residuals stay small. Every obvious sanity
check still passes. But blink detection, eye-closure/lagophthalmos measurement,
ptosis and any oral-ocular metric silently stop working, because their geometric
input has been flattened. It took a while to find, which is why
`gnm_mediapipe_landmarks.py` ships a self-check that asserts these structures are
still resolvable:

```
[ok] inter-ocular distance    85.22 mm
[ok] mouth width              45.64 mm
[ok] palpebral aperture R      7.19 mm
[ok] palpebral aperture L      7.63 mm
[ok] inner-lip gap             4.22 mm
[ok] eye width R              26.30 mm
```

Worth running whenever the model version or the projection regions change.

## Side naming

MediaPipe's own docs are inconsistent about whether "left" means the subject's
left or the viewer's left. Verified against GNM's own `left`/`right` vertex
groups, these indices are **anatomical (subject) sides**:

- `61` = subject's **right** mouth corner, `291` = subject's **left**
- `33` = subject's **right** eye outer corner, `263` = subject's **left**

If you report per-side results to anyone clinical, pin this down explicitly.

## Regenerating

```bash
pip install numpy scipy
curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
python build_mediapipe_correspondence.py --canonical canonical_face_model.obj --out head_dense_468.txt
```

Reproduces `head_dense_468.txt` byte-for-byte from GNM v3.0.

## Licence

Derived from two Apache-2.0 works — Google's GNM head model and MediaPipe's
canonical face model — and released on the same terms. Please keep the
attribution to both.
