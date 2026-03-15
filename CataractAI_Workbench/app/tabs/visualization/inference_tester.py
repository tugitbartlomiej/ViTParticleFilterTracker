"""Thin orchestrator for the Inference Tester tab.

Composes sub-widgets (controls, image viewer, stats) and delegates all
heavy logic to the backend ``ModelInference`` and ``metrics`` modules.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMessageBox, QFileDialog, QApplication,
)
from PyQt6.QtCore import Qt

from .inference_controls import ModelConfigPanel, DatasetPanel, BatchTestPanel, NavigationBar
from .inference_image_viewer import InferenceImageViewer
from .inference_stats import StatsPanel, DetectionDetailsTable, ConfusionMatrixPanel
from .inference_worker import InferenceWorker


class InferenceTester(QWidget):
    """Interactive widget for testing DETR/YOLO model inference."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._adapter = None
        self._image_paths: List[str] = []
        self._current_idx: int = -1
        self._annotations: Dict[str, List[dict]] = {}
        self._categories: Dict[int, str] = {}
        self._current_predictions: List[dict] = []
        self._total_tp = 0
        self._total_fp = 0
        self._total_fn = 0
        self._tested_count = 0
        self._all_scores: List[float] = []
        self._batch_results: Optional[dict] = None
        self._worker: Optional[InferenceWorker] = None
        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self._model_panel = ModelConfigPanel()
        self._dataset_panel = DatasetPanel()
        self._stats_panel = StatsPanel()
        left_lay.addWidget(self._model_panel)
        left_lay.addWidget(self._dataset_panel)
        left_lay.addWidget(self._stats_panel)
        left_lay.addStretch()
        splitter.addWidget(left)

        # Right panel
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)
        self._nav_bar = NavigationBar()
        right_lay.addWidget(self._nav_bar)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        self._image_viewer = InferenceImageViewer()
        v_splitter.addWidget(self._image_viewer)
        self._det_table = DetectionDetailsTable()
        v_splitter.addWidget(self._det_table)

        bottom = QWidget()
        bottom_lay = QHBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(6)
        self._batch_panel = BatchTestPanel()
        self._cm_panel = ConfusionMatrixPanel()
        bottom_lay.addWidget(self._batch_panel)
        bottom_lay.addWidget(self._cm_panel)
        v_splitter.addWidget(bottom)

        v_splitter.setStretchFactor(0, 4)
        v_splitter.setStretchFactor(1, 1)
        v_splitter.setStretchFactor(2, 1)
        right_lay.addWidget(v_splitter)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._model_panel.load_requested.connect(self._on_load_model)
        self._dataset_panel.load_requested.connect(self._on_load_dataset)
        self._stats_panel.reset_requested.connect(self._on_reset_stats)
        self._nav_bar.prev_clicked.connect(self._on_prev)
        self._nav_bar.next_clicked.connect(self._on_next)
        self._nav_bar.random_clicked.connect(self._on_random)
        self._nav_bar.run_inference_clicked.connect(self._on_run_single)
        self._batch_panel.run_requested.connect(self._on_run_batch)
        self._batch_panel.cancel_requested.connect(self._on_cancel_batch)
        self._batch_panel.export_csv_requested.connect(self._on_export_csv)
        self._batch_panel.export_json_requested.connect(self._on_export_json)

    # -- Model loading --

    def _on_load_model(self) -> None:
        ckpt = self._model_panel.checkpoint_path
        if not ckpt or not Path(ckpt).is_file():
            QMessageBox.warning(self, "Error", "Please select a valid checkpoint file.")
            return
        self._model_panel.set_load_enabled(False)
        self._model_panel.set_status("Loading model...", "#E6C07B")
        QApplication.processEvents()
        try:
            from CataractAI_Workbench.backend.inference_adapter import InferenceFactory
            model_type = self._model_panel.model_type_str
            device = self._model_panel.device
            self._adapter = InferenceFactory.create(model_type)
            if model_type == "detr":
                self._adapter.load(ckpt, device, num_labels=self._model_panel.num_labels)
                self._model_panel.set_status(
                    f"DETR loaded ({self._model_panel.num_labels} labels) on {device}\n"
                    f"{Path(ckpt).name}", "#6BCB77",
                )
            else:
                self._adapter.load(ckpt, device)
                self._model_panel.set_status(f"YOLO loaded\n{Path(ckpt).name}", "#6BCB77")
        except Exception as e:
            self._adapter = None
            self._model_panel.set_status(f"Load failed: {e}", "#FF6B6B")
            QMessageBox.critical(self, "Model Load Error", str(e))
        finally:
            self._model_panel.set_load_enabled(True)

    # -- Dataset loading --

    def _on_load_dataset(self) -> None:
        img_dir = self._dataset_panel.image_dir
        if not img_dir or not Path(img_dir).is_dir():
            QMessageBox.warning(self, "Error", "Please select a valid images directory.")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self._image_paths = sorted(
            str(p) for p in Path(img_dir).iterdir() if p.suffix.lower() in exts
        )
        if not self._image_paths:
            QMessageBox.warning(self, "Error", "No image files found in the directory.")
            return
        self._annotations.clear()
        self._categories.clear()
        ann_path = self._dataset_panel.annotations_path
        if ann_path and Path(ann_path).is_file():
            try:
                from CataractAI_Workbench.backend.metrics import load_coco_annotations
                coco = load_coco_annotations(ann_path)
                self._annotations = coco.get("annotations", {})
                self._categories = coco.get("categories", {})
            except Exception as e:
                QMessageBox.warning(
                    self, "Annotation Warning",
                    f"Could not parse annotations:\n{e}\n\nProceeding without ground truth.",
                )
        self._image_viewer.set_categories(self._categories)
        self._current_idx = 0
        self._update_nav_label()
        gt_info = f", {sum(len(v) for v in self._annotations.values())} GT annotations" if self._annotations else ""
        cat_info = f", {len(self._categories)} categories" if self._categories else ""
        self._dataset_panel.set_status(
            f"{len(self._image_paths)} images loaded{gt_info}{cat_info}", "#6BCB77",
        )
        self._show_current_image(run_inference=False)

    # -- Navigation --

    def _on_prev(self) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_nav_label()
            self._show_current_image(run_inference=False)

    def _on_next(self) -> None:
        if self._current_idx < len(self._image_paths) - 1:
            self._current_idx += 1
            self._update_nav_label()
            self._show_current_image(run_inference=False)

    def _on_random(self) -> None:
        if self._image_paths:
            self._current_idx = random.randint(0, len(self._image_paths) - 1)
            self._update_nav_label()
            self._show_current_image(run_inference=False)

    def _update_nav_label(self) -> None:
        if self._image_paths:
            fname = Path(self._image_paths[self._current_idx]).name
            self._nav_bar.set_position(
                f"Image {self._current_idx + 1}/{len(self._image_paths)}  |  {fname}"
            )
        else:
            self._nav_bar.set_position("No images loaded")

    # -- Single-image inference --

    def _on_run_single(self) -> None:
        if self._adapter is None:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return
        if not self._image_paths:
            QMessageBox.warning(self, "Error", "Load a dataset first.")
            return
        self._show_current_image(run_inference=True)

    def _show_current_image(self, run_inference: bool = False) -> None:
        if self._current_idx < 0 or self._current_idx >= len(self._image_paths):
            return
        img_path = self._image_paths[self._current_idx]
        fname = Path(img_path).name
        gt_boxes = self._annotations.get(fname, [])
        predictions: List[dict] = []

        if run_inference and self._adapter is not None:
            self._nav_bar.set_run_enabled(False)
            QApplication.processEvents()
            try:
                from CataractAI_Workbench.backend.metrics import compute_metrics
                predictions = self._adapter.predict(img_path, self._model_panel.confidence)
                self._current_predictions = predictions
                if gt_boxes:
                    m = compute_metrics(predictions, gt_boxes, self._batch_panel.iou_threshold)
                    self._total_tp += m["tp"]
                    self._total_fp += m["fp"]
                    self._total_fn += m["fn"]
                    self._tested_count += 1
                    self._all_scores.extend(p.get("score", 0) for p in predictions)
                    self._stats_panel.update_stats(
                        self._tested_count, self._total_tp, self._total_fp,
                        self._total_fn, self._all_scores,
                    )
            except Exception as e:
                QMessageBox.critical(self, "Inference Error", str(e))
            finally:
                self._nav_bar.set_run_enabled(True)
        else:
            self._current_predictions = []

        self._image_viewer.display(img_path, predictions, gt_boxes)
        self._det_table.fill(predictions, gt_boxes, self._categories)

    # -- Statistics --

    def _on_reset_stats(self) -> None:
        self._total_tp = self._total_fp = self._total_fn = self._tested_count = 0
        self._all_scores.clear()
        self._stats_panel.reset()

    # -- Batch testing --

    def _on_run_batch(self) -> None:
        if self._adapter is None:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return
        if not self._image_paths:
            QMessageBox.warning(self, "Error", "Load a dataset first.")
            return
        if not self._annotations:
            QMessageBox.warning(
                self, "Warning",
                "No annotations loaded. Batch test requires ground truth "
                "annotations to compute metrics. Load a COCO annotations file first.",
            )
            return
        self._batch_panel.set_running(True)
        self._worker = InferenceWorker(
            adapter=self._adapter, image_paths=self._image_paths,
            annotations=self._annotations, confidence=self._model_panel.confidence,
            iou_threshold=self._batch_panel.iou_threshold, parent=self,
        )
        self._worker.progress.connect(self._batch_panel.set_progress)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _on_cancel_batch(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._batch_panel.set_cancelling()

    def _on_batch_finished(self, results: dict) -> None:
        self._batch_results = results
        self._batch_panel.set_running(False)
        self._batch_panel.set_finished(results["images_tested"])
        self._cm_panel.update_from_results(results)
        self._total_tp = results["total_tp"]
        self._total_fp = results["total_fp"]
        self._total_fn = results["total_fn"]
        self._tested_count = results["images_tested"]
        self._all_scores = [
            p.get("score", 0) for img_r in results["per_image"] for p in img_r["predictions"]
        ]
        self._stats_panel.update_stats(
            self._tested_count, self._total_tp, self._total_fp,
            self._total_fn, self._all_scores,
        )
        self._worker = None

    def _on_batch_error(self, message: str) -> None:
        self._batch_panel.set_running(False)
        self._batch_panel.set_error(message)
        QMessageBox.critical(self, "Batch Test Error", message)
        self._worker = None

    # -- Export --

    def _on_export_csv(self) -> None:
        if not self._batch_results:
            QMessageBox.information(self, "Info", "Run a batch test first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "inference_results.csv", "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            r = self._batch_results
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "gt_count", "pred_count", "tp", "fp", "fn"])
                for img_r in r["per_image"]:
                    writer.writerow([
                        img_r["filename"], img_r["gt_count"], img_r["pred_count"],
                        img_r["tp"], img_r["fp"], img_r["fn"],
                    ])
                writer.writerow([])
                writer.writerow(["SUMMARY"])
                for key in ["images_tested", "total_tp", "total_fp", "total_fn",
                             "precision", "recall", "f1", "avg_confidence"]:
                    writer.writerow([key, r[key]])
            QMessageBox.information(self, "Export", f"Results exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_json(self) -> None:
        if not self._batch_results:
            QMessageBox.information(self, "Info", "Run a batch test first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "inference_results.json", "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._batch_results, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Export", f"Results exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
