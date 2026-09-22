"""Utilities for image alignment and multiple-choice visual question answering."""

from .data import ANSWER_TO_ID, ID_TO_ANSWER, build_mcq_prompt
from .projector import GatedResidualProjector

__all__ = ["ANSWER_TO_ID", "ID_TO_ANSWER", "build_mcq_prompt", "GatedResidualProjector"]
