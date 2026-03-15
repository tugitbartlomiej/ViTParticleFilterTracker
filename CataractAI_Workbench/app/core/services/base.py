"""Abstract base class for all analysis services."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AnalysisService(ABC):
    """Interface every analysis service must implement."""

    @abstractmethod
    def analyze(self, data: dict) -> dict:
        """Accept structured input, return structured output."""
