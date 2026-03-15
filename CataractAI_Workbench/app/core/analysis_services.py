"""Pure-logic analysis services with no Qt dependency.

This module re-exports all service classes for backward compatibility.
Each service lives in its own submodule under ``core.services``.
"""

from __future__ import annotations

from .services.base import AnalysisService
from .services.checkpoint_service import CheckpointAnalysisService
from .services.dataset_service import DatasetAnalysisService
from .services.parameter_service import ParameterRecommendationEngine

__all__ = [
    "AnalysisService",
    "CheckpointAnalysisService",
    "DatasetAnalysisService",
    "ParameterRecommendationEngine",
]
