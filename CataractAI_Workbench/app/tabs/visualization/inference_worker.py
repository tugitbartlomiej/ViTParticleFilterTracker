"""Background worker thread for batch inference evaluation."""

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    """Runs batch inference in a background thread.

    Emits ``progress(current, total, filename)`` after each image,
    ``finished(results_dict)`` on completion, or ``error(message)``
    on failure.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, adapter, image_paths, annotations, confidence,
                 iou_threshold, parent=None):
        super().__init__(parent)
        self._adapter = adapter
        self._image_paths = image_paths
        self._annotations = annotations
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from CataractAI_Workbench.backend.metrics import compute_metrics

            total = len(self._image_paths)
            total_tp = 0
            total_fp = 0
            total_fn = 0
            all_scores: List[float] = []
            per_image: List[dict] = []

            for idx, img_path in enumerate(self._image_paths):
                if self._cancelled:
                    break

                fname = Path(img_path).name
                preds = self._adapter.predict(img_path, self._confidence)
                gt = self._annotations.get(fname, [])
                metrics = compute_metrics(preds, gt, self._iou_threshold)

                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]

                for p in preds:
                    all_scores.append(p.get("score", 0))

                per_image.append({
                    "filename": fname,
                    "predictions": preds,
                    "gt_count": len(gt),
                    "pred_count": len(preds),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                })

                self.progress.emit(idx + 1, total, fname)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            avg_conf = sum(all_scores) / len(all_scores) if all_scores else 0.0

            self.finished.emit({
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "avg_confidence": round(avg_conf, 4),
                "images_tested": len(per_image),
                "per_image": per_image,
            })
        except Exception as e:
            self.error.emit(str(e))
