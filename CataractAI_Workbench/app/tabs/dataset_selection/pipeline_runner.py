"""QThread worker that runs the dataset selection pipeline."""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from ...core.worker_base import PipelineWorker

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _LogInterceptor(logging.Handler):
    """Logging handler that forwards records to the worker's signal."""

    def __init__(self, worker: "DatasetSelectionWorker"):
        super().__init__(logging.DEBUG)
        self._worker = worker

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._worker.emit_log(msg)
        except RuntimeError:
            pass  # worker already deleted


class DatasetSelectionWorker(PipelineWorker):
    """Run :class:`DatasetSelectionAdapter` inside a QThread.

    Parameters
    ----------
    config_path : str
        Path to the pipeline config YAML.
    target_size : int
        Number of images to select.
    config_overrides : dict | None
        Extra config values to merge before running.
    """

    def __init__(
        self,
        config_path: str,
        target_size: int,
        config_overrides: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._config_path = config_path
        self._target_size = target_size
        self._config_overrides = config_overrides

    def run_task(self) -> dict:
        """Execute the pipeline (called on the worker thread)."""
        # Install log interceptor
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = _LogInterceptor(self)
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            self.stage_changed.emit("Importing pipeline modules")
            self.emit_log("INFO - Importing AdvancedDatasetSelection modules...")

            from CataractAI_Workbench.backend.dataset_selection_adapter import (
                DatasetSelectionAdapter,
            )

            if self._cancelled:
                return {}

            self.stage_changed.emit("Initializing adapter")

            adapter = DatasetSelectionAdapter(self._config_path)

            # Apply config overrides if provided
            if self._config_overrides:
                adapter.update_config(self._config_overrides)

            if self._cancelled:
                return {}

            self.stage_changed.emit("Running pipeline")

            def _progress(percent: int, message: str):
                if not self._cancelled:
                    self.emit_progress(percent, message)
                    self.emit_log(f"INFO - [{percent}%] {message}")

            results = adapter.run_pipeline(
                target_size=self._target_size,
                progress_callback=_progress,
            )

            if self._cancelled:
                return {}

            # Collect visualization paths
            output_dir = adapter.get_config().get("datasets", {}).get("output")
            viz_paths = adapter.get_visualizations(output_dir)
            results["visualization_paths"] = viz_paths

            self.stage_changed.emit("Done")
            self.emit_log(f"INFO - Pipeline finished. Selected {len(results.get('selected_paths', []))} images.")
            return results

        except Exception as exc:
            tb = traceback.format_exc()
            self.emit_log(f"ERROR - Pipeline failed: {exc}\n{tb}")
            raise
        finally:
            root_logger.removeHandler(handler)
