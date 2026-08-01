#!/bin/bash
# Fit GNM to a WHOLE FOLDER of clips -> dataset (per-clip .npz + manifest.jsonl + stats).
# Bash wrapper around src/build_dataset.py.
#
# Usage:
#   scripts/fit_folder.sh <videos_dir> [out_dir]
#
# Optional env:
#   PYTHON=.venv/bin/python   pick interpreter (else repo .venv, else `python`)
#   DEVICE=cuda|cpu|mps       force device (default: auto = CUDA if available else CPU)
#   GLOB='*.mp4'              which files to fit (default: *.mp4)
#   LIMIT=10                  cap number of clips
#   MAX_FRAMES=300            cap frames per clip
#   CONFIG=configs/fast.yaml  use a different config
#   MP_MODEL=path/to.task     MediaPipe face_landmarker model
#   VIZ=1                     also dump debug viz per clip (off by default; many files)
#
# Example:
#   scripts/fit_folder.sh ../TalkingHead-1KH/small/cropped_clips outputs

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DIR="${1:?usage: fit_folder.sh <videos_dir> [out_dir]}"
OUT="${2:-outputs}"
MP_MODEL="${MP_MODEL:-data/mediapipe/face_landmarker.task}"
GLOB="${GLOB:-*.mp4}"

# Pick interpreter: $PYTHON, else the repo's .venv, else `python`.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY="python"; fi

ARGS=(--videos-dir "$DIR" --out "$OUT" --glob "$GLOB" --mp-model "$MP_MODEL")
[ -n "${VIZ:-}" ] && ARGS+=(--viz)
[ -n "${DEVICE:-}" ] && ARGS+=(--device "$DEVICE")
[ -n "${MAX_FRAMES:-}" ] && ARGS+=(--max-frames "$MAX_FRAMES")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
[ -n "${CONFIG:-}" ] && ARGS+=(--config "$CONFIG")

cd "$ROOT"
echo "[fit_folder] $PY src/build_dataset.py ${ARGS[*]}"
exec "$PY" src/build_dataset.py "${ARGS[@]}"
