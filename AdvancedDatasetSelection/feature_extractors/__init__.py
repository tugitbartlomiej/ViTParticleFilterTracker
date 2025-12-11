"""Feature extractors for dataset selection."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .dino_extractor import DINOExtractor
    from .fourier_analyzer import FourierAnalyzer
    from .sam_extractor import SAMExtractor

__all__ = ["DINOExtractor", "SAMExtractor", "FourierAnalyzer"]


def __getattr__(name):
    if name == "DINOExtractor":
        from .dino_extractor import DINOExtractor

        return DINOExtractor
    if name == "SAMExtractor":
        from .sam_extractor import SAMExtractor

        return SAMExtractor
    if name == "FourierAnalyzer":
        from .fourier_analyzer import FourierAnalyzer

        return FourierAnalyzer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
