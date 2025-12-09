"""
Advanced Dataset Selection Pipeline for DETR Fine-tuning

This package provides tools for intelligent dataset selection using:
- DINO: Semantic feature extraction and clustering
- SAM v3: Segmentation masks and scene complexity metrics
- Fourier: Frequency domain analysis for redundancy elimination
- EL2N + k-Center: Sample selection methods from literature
"""

__version__ = "1.0.0"
__author__ = "PhD Project"

from .feature_extractors import DINOExtractor, SAMExtractor, FourierAnalyzer
from .selection_methods import EL2NScorer, KCenterGreedy, CombinedSelector
