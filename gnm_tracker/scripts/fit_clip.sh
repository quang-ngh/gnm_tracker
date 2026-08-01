#!/bin/bash
# Fit GNM to a SINGLE clip -> dataset record (.npz) + debug viz.
# Bash wrapper around src/fit_sequence.py.
#
# Usage:
#   scripts/fit_clip.sh <clip.mp4> [out_dir]
#
# Optional env:
#   PYTHON=.venv/bin/python   pick interpreter (else repo .venv, else `python`)
#   DEVICE=cuda|cpu|mps       force device (default: auto = CUDA if available else CPU)
#   MAX_FRAMES=60             cap frames per clip
#   CONFIG=configs/fast.yaml  use a different config
#   MP_MODEL=path/to.task     MediaPipe face_landmarker model
#   NO_VIZ=1                  skip debug visualization
#
# Example:
#   scripts/fit_clip.sh ../TalkingHead-1KH/small/cropped_clips/<clip>.mp4 outputs

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

VIDEO="${1:?usage: fit_clip.sh <clip.mp4> [out_dir]}"
OUT="${2:-outputs}"
MP_MODEL="${MP_MODEL:-data/mediapipe/face_landmarker.task}"

# Pick interpreter: $PYTHON, else the repo's .venv, else `python`.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY="python"; fi

ARGS=(--video "$VIDEO" --out "$OUT" --mp-model "$MP_MODEL")
[ -z "${NO_VIZ:-}" ] && ARGS+=(--viz)
[ -n "${DEVICE:-}" ] && ARGS+=(--device "$DEVICE")
[ -n "${MAX_FRAMES:-}" ] && ARGS+=(--max-frames "$MAX_FRAMES")
[ -n "${CONFIG:-}" ] && ARGS+=(--config "$CONFIG")

cd "$ROOT"
echo "[fit_clip] $PY src/fit_sequence.py ${ARGS[*]}"
exec "$PY" src/fit_sequence.py "${ARGS[@]}"
