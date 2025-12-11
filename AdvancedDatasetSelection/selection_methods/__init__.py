"""Selection methods for dataset curation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from .combined_selector import CombinedSelector
    from .el2n_scorer import EL2NScorer
    from .k_center_greedy import KCenterGreedy

__all__ = ["EL2NScorer", "KCenterGreedy", "CombinedSelector"]


def __getattr__(name):
    if name == "EL2NScorer":
        from .el2n_scorer import EL2NScorer

        return EL2NScorer
    if name == "KCenterGreedy":
        from .k_center_greedy import KCenterGreedy

        return KCenterGreedy
    if name == "CombinedSelector":
        from .combined_selector import CombinedSelector

        return CombinedSelector
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
