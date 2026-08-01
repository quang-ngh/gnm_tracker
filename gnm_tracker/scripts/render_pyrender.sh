#!/bin/bash
# Render a fitted clip with GNM's own pyrender pipeline (Linux/GPU debug).
#
# Bash wrapper around src/render_pyrender.py. Needs an offscreen GL backend
# (OSMesa or EGL) — available on Linux, NOT on macOS.
#
#   # OSMesa (CPU):  apt-get install libosmesa6 libosmesa6-dev
#   # EGL   (GPU):   use a GPU box with EGL drivers
#   pip install pyrender pyopengl
#
# Usage:
#   scripts/render_pyrender.sh <record.npz> <video.mp4> [out.mp4] [osmesa|egl]
#
# Optional env:
#   PYTHON=.venv/bin/python  scripts/render_pyrender.sh ...   # pick interpreter
#   MAX_FRAMES=60            scripts/render_pyrender.sh ...    # cap frames

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RECORD="${1:?usage: render_pyrender.sh <record.npz> <video.mp4> [out.mp4] [osmesa|egl]}"
VIDEO="${2:?usage: render_pyrender.sh <record.npz> <video.mp4> [out.mp4] [osmesa|egl]}"
OUT="${3:-outputs/pyrender_side_by_side.mp4}"
PLATFORM="${4:-osmesa}"
PY="${PYTHON:-python}"

EXTRA=()
if [ -n "${MAX_FRAMES:-}" ]; then EXTRA+=(--max-frames "${MAX_FRAMES}"); fi
if [ -n "${DEVICE:-}" ]; then EXTRA+=(--device "${DEVICE}"); fi

cd "$ROOT"
exec "$PY" src/render_pyrender.py \
  --record "$RECORD" \
  --video "$VIDEO" \
  --out "$OUT" \
  --platform "$PLATFORM" \
  "${EXTRA[@]}"
