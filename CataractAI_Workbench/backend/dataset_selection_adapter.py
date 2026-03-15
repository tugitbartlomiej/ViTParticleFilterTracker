"""Adapter wrapping AdvancedDatasetSelectionPipeline for GUI consumption."""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that
# ``AdvancedDatasetSelection`` can be imported as a package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DatasetSelectionAdapter:
    """High-level adapter around :class:`AdvancedDatasetSelectionPipeline`.

    This class is meant to be used from the GUI layer.  It defers the heavy
    import of the pipeline (which pulls in torch, transformers, etc.) until
    :meth:`run_pipeline` is called so the UI stays responsive during startup.

    Parameters
    ----------
    config_path : str
        Path to the YAML config consumed by the pipeline.
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._pipeline = None  # lazy
        self._config: Dict = self._load_config(config_path)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_config(self) -> Dict:
        """Return a copy of the current configuration dict."""
        return dict(self._config)

    def update_config(self, updates: Dict):
        """Deep-merge *updates* into the current config.

        The merged result is also written back to the YAML file so that the
        pipeline picks it up on the next run.
        """
        self._deep_merge(self._config, updates)
        # Persist
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        target_size: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict:
        """Run the full selection pipeline.

        Parameters
        ----------
        target_size : int
            Number of images to select.
        progress_callback : callable(int, str) | None
            ``(percent, message)`` callback for progress updates.

        Returns
        -------
        dict
            Pipeline results (selected_indices, selected_paths, statistics, ...).
        """
        if progress_callback:
            progress_callback(0, "Importing pipeline modules...")

        # Lazy import - heavy dependencies
        from AdvancedDatasetSelection.main_selection_pipeline import (
            AdvancedDatasetSelectionPipeline,
        )

        if progress_callback:
            progress_callback(5, "Initializing pipeline...")

        self._pipeline = AdvancedDatasetSelectionPipeline(self._config_path)

        if progress_callback:
            progress_callback(10, "Loading datasets...")

        image_paths, annotations = self._pipeline.load_datasets()

        if progress_callback:
            progress_callback(15, f"Loaded {len(image_paths)} images. Running selection...")

        results = self._pipeline.run(
            image_paths=image_paths,
            target_size=target_size,
        )

        if progress_callback:
            progress_callback(100, "Pipeline completed.")

        # Refresh cached config (pipeline may have mutated it)
        self._config = dict(self._pipeline.config)

        return results

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def get_visualizations(self, output_dir: Optional[str] = None) -> List[str]:
        """Return paths of generated plot images found in *output_dir*.

        If *output_dir* is None, the output directory is read from config.
        """
        if output_dir is None:
            output_dir = self._config.get("datasets", {}).get(
                "output", "./output/selected_dataset"
            )
        viz_dir = Path(output_dir) / "visualizations"
        if not viz_dir.exists():
            return []

        extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        return sorted(
            str(p)
            for p in viz_dir.iterdir()
            if p.suffix.lower() in extensions
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict, override: Dict):
        """Recursively merge *override* into *base* in-place."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                DatasetSelectionAdapter._deep_merge(base[key], value)
            else:
                base[key] = value
