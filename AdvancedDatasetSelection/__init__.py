"""
Advanced Dataset Selection Pipeline for DETR Fine-tuning

This package provides tools for intelligent dataset selection using:
- DINO: Semantic feature extraction and clustering
- SAM v3: Segmentation masks and scene complexity metrics
- Fourier: Frequency domain analysis for redundancy elimination
- EL2N + k-Center: Sample selection methods from literature
"""

from typing import TYPE_CHECKING

__version__ = "1.0.0"
__author__ = "PhD Project"

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from .feature_extractors import DINOExtractor, SAMExtractor, FourierAnalyzer
    from .selection_methods import EL2NScorer, KCenterGreedy, CombinedSelector

__all__ = [
    "CombinedSelector",
    "DINOExtractor",
    "EL2NScorer",
    "FourierAnalyzer",
    "KCenterGreedy",
    "SAMExtractor",
]


def __getattr__(name):
    if name == "DINOExtractor":
        from .feature_extractors import DINOExtractor

        return DINOExtractor
    if name == "SAMExtractor":
        from .feature_extractors import SAMExtractor

        return SAMExtractor
    if name == "FourierAnalyzer":
        from .feature_extractors import FourierAnalyzer

        return FourierAnalyzer
    if name == "EL2NScorer":
        from .selection_methods import EL2NScorer

        return EL2NScorer
    if name == "KCenterGreedy":
        from .selection_methods import KCenterGreedy

        return KCenterGreedy
    if name == "CombinedSelector":
        from .selection_methods import CombinedSelector

        return CombinedSelector
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
