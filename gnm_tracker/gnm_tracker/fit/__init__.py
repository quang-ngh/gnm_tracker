"""Losses and the staged analysis-by-synthesis optimizer."""

from gnm_tracker.fit.build import build_fitter
from gnm_tracker.fit.losses import ExpressionPrior
from gnm_tracker.fit.sequence import SequenceResult, fit_sequence
from gnm_tracker.fit.single_frame import FitResult, fit_single_frame
from gnm_tracker.fit.stages import Fitter, Targets

__all__ = [
    "build_fitter",
    "ExpressionPrior",
    "Fitter",
    "Targets",
    "FitResult",
    "fit_single_frame",
    "SequenceResult",
    "fit_sequence",
]
