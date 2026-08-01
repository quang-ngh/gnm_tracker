"""Shared helpers for the CLI scripts (device / model / detector setup)."""

from __future__ import annotations

import pathlib
import sys

# Allow `python scripts/foo.py` without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def get_device(cfg, override: str | None = None) -> str:
    d = override or cfg.device
    if d == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return d


def load_model(cfg, device: str):
    from gnm_tracker.model import GnmModel

    return GnmModel(cfg, device)


def make_detector(model_path=None, running_mode: str = "video"):
    from gnm_tracker.detect import MediaPipeDetector

    return MediaPipeDetector(model_path=model_path, running_mode=running_mode)
