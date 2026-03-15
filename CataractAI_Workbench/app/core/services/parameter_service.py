"""Parameter recommendation engine -- rule-based training config analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import AnalysisService


class ParameterRecommendationEngine(AnalysisService):
    """Generate rule-based training parameter recommendations."""

    VRAM_PER_BATCH: Dict[str, float] = {
        "detr": 2.5,
        "resnet": 0.8,
        "vit": 1.5,
        "yolo": 1.0,
    }

    def analyze(self, data: dict) -> dict:
        """Expect ``{config, dataset_dir?, history?}``.

        Returns ``{summary, recommendations, history_analysis?}``.
        """
        config = data["config"]
        dataset_dir = data.get("dataset_dir")
        history = data.get("history")

        summary = self._build_summary(config, dataset_dir)
        recommendations = self._generate_recommendations(config, dataset_dir)

        result: dict = {"summary": summary, "recommendations": recommendations}
        if history is not None:
            result["history_analysis"] = self._analyze_history(history)

        return result

    # -- summary -----------------------------------------------------

    def _build_summary(self, config: dict, dataset_dir: Optional[str]) -> str:
        tp = config.get("training_parameters", {})
        opt = config.get("optimization", {})
        es = config.get("early_stopping", {})
        dp = config.get("dataset_paths", {})

        img_count = 0
        ann_count = 0
        cat_names: List[str] = []
        if dataset_dir and Path(dataset_dir).is_dir():
            img_count, ann_count, cat_names = self._count_dataset(dataset_dir, dp)

        lines = [
            f"Epochs:         {tp.get('epochs', '?')}",
            f"Batch size:     {tp.get('batch_size', '?')}",
            f"LR:             {tp.get('learning_rate', '?')}",
            f"Weight decay:   {tp.get('weight_decay', '?')}",
            f"Warmup:         {tp.get('warmup_epochs', '?')}",
            f"Scheduler:      {opt.get('scheduler', {}).get('type', '?')}",
            f"AMP:            {'enabled' if opt.get('use_amp') else 'disabled'}",
            f"Early stop:     patience {es.get('patience', '?')}",
            "",
            "Dataset:",
        ]

        if img_count > 0:
            lines.append(f"  Images:       {img_count}")
            if ann_count > 0:
                lines.append(f"  Annotations:  {ann_count}")
            if cat_names:
                lines.append(f"  Categories:   {', '.join(cat_names)}")
        else:
            lines.append("  (Set dataset dir for details)")

        return "\n".join(lines)

    # -- recommendations ---------------------------------------------

    def _generate_recommendations(self, config: dict, dataset_dir: Optional[str]) -> str:
        tp = config.get("training_parameters", {})
        opt = config.get("optimization", {})
        es = config.get("early_stopping", {})
        dp = config.get("dataset_paths", {})
        ms = config.get("model_settings", {})

        lr = tp.get("learning_rate", 2e-6)
        bs = tp.get("batch_size", 2)
        epochs = tp.get("epochs", 5)
        warmup = tp.get("warmup_epochs", 1)
        patience = es.get("patience", 6)
        base_model = ms.get("base_model", "").lower()

        model_type = "detr"
        for key in self.VRAM_PER_BATCH:
            if key in base_model:
                model_type = key
                break

        img_count = 0
        if dataset_dir and Path(dataset_dir).is_dir():
            img_count, _, _ = self._count_dataset(dataset_dir, dp)

        recs: List[str] = []
        recs.extend(self._lr_recs(lr, img_count))
        recs.extend(self._vram_recs(bs, model_type))
        recs.extend(self._epoch_recs(epochs, bs, img_count))
        recs.extend(self._warmup_recs(warmup, lr, epochs))
        recs.extend(self._early_stopping_recs(es, patience, epochs))
        recs.extend(self._class_balance_recs(dataset_dir, dp))
        recs.extend(self._scheduler_recs(opt, epochs))
        recs.extend(self._amp_recs(opt))
        recs.extend(self._dataset_size_recs(img_count))

        return "\n\n".join(recs)

    # -- individual recommendation rules -----------------------------

    @staticmethod
    def _lr_recs(lr: float, img_count: int) -> List[str]:
        if lr < 1e-6:
            suffix = f" for {img_count} images." if img_count > 0 else "."
            return [f"[!] Learning rate {lr:.1e} is very conservative{suffix}"
                    f" Consider 5e-6 to 1e-5 for faster convergence."]
        if lr > 1e-4:
            return [f"[!] Learning rate {lr:.1e} is quite high for fine-tuning."
                    f" Consider 1e-5 to 5e-5 to avoid catastrophic forgetting."]
        return [f"[OK] Learning rate {lr:.1e} is in a reasonable range."]

    def _vram_recs(self, bs: int, model_type: str) -> List[str]:
        vram_per = self.VRAM_PER_BATCH.get(model_type, 2.5)
        est_vram = bs * vram_per
        if est_vram <= 10:
            return [f"[OK] Batch size {bs} is appropriate (~{est_vram:.1f}GB VRAM estimated"
                    f" for {model_type.upper()})."]
        return [f"[!] Batch size {bs} may require ~{est_vram:.1f}GB VRAM for"
                f" {model_type.upper()}. Consider reducing if you have <= 10GB."]

    @staticmethod
    def _epoch_recs(epochs: int, bs: int, img_count: int) -> List[str]:
        if img_count <= 0:
            return []
        batches_per_epoch = max(img_count // bs, 1)
        if batches_per_epoch < 50 and epochs < 10:
            return [f"[!] With {img_count} images and batch_size={bs}, you get"
                    f" {batches_per_epoch} batches/epoch. Consider more epochs"
                    f" (10-20) for better convergence."]
        if epochs > 50 and img_count > 5000:
            return [f"[!] {epochs} epochs with {img_count} images may be excessive."
                    f" Consider 10-30 epochs with early stopping."]
        return [f"[OK] {epochs} epochs with {batches_per_epoch} batches/epoch"
                f" is reasonable."]

    @staticmethod
    def _warmup_recs(warmup: int, lr: float, epochs: int) -> List[str]:
        if warmup == 0 and lr > 5e-6:
            return [f"[!] No warmup with LR={lr:.1e} may cause instability."
                    f" Consider 1-2 warmup epochs."]
        if warmup > 0 and warmup >= epochs // 2:
            return [f"[!] Warmup ({warmup} epochs) is large relative to total"
                    f" epochs ({epochs}). Consider reducing warmup."]
        return [f"[OK] Warmup={warmup} epoch(s) is appropriate."]

    @staticmethod
    def _early_stopping_recs(es: dict, patience: int, epochs: int) -> List[str]:
        if not es.get("enabled", True):
            return ["[!] Early stopping is disabled. Enable it to prevent overfitting."]
        if patience > epochs:
            return [f"[!] Early stopping patience ({patience}) exceeds total"
                    f" epochs ({epochs}). It will never trigger."]
        return [f"[OK] Early stopping patience={patience} is good for"
                f" this configuration."]

    def _class_balance_recs(self, dataset_dir: Optional[str], dp: dict) -> List[str]:
        if not dataset_dir or not Path(dataset_dir).is_dir():
            return []
        ann_file = dp.get("annotations_file", "annotations/mixed_annotations.json")
        ann_path = Path(dataset_dir) / ann_file
        if not ann_path.is_file():
            return []
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
            annotations = coco.get("annotations", [])
            images = coco.get("images", [])
            if not images or not annotations:
                return []
            annotated_ids = set(a["image_id"] for a in annotations)
            n_total = len(images)
            n_background = n_total - len(annotated_ids)
            if n_total == 0:
                return []
            bg_pct = n_background / n_total * 100
            if bg_pct > 80:
                return [f"[!] Class imbalance: {bg_pct:.0f}% background vs"
                        f" {100 - bg_pct:.0f}% annotated. Consider weighted"
                        f" loss or oversampling."]
            if bg_pct > 60:
                return [f"[~] Moderate imbalance: {bg_pct:.0f}% background."
                        f" Monitor for bias toward background predictions."]
            return [f"[OK] Class balance is reasonable"
                    f" ({100 - bg_pct:.0f}% annotated)."]
        except Exception:
            return []

    @staticmethod
    def _scheduler_recs(opt: dict, epochs: int) -> List[str]:
        sched_type = opt.get("scheduler", {}).get("type", "StepLR")
        if sched_type == "StepLR" and epochs > 20:
            return [f"[~] Consider CosineAnnealingLR for longer training"
                    f" ({epochs} epochs). It often gives smoother convergence."]
        return [f"[OK] Scheduler '{sched_type}' is suitable."]

    @staticmethod
    def _amp_recs(opt: dict) -> List[str]:
        if opt.get("use_amp", True):
            return ["[OK] Mixed precision (AMP) is enabled -- good for speed and memory."]
        return ["[~] AMP is disabled. Enable it to reduce memory usage and speed up training."]

    @staticmethod
    def _dataset_size_recs(img_count: int) -> List[str]:
        if 0 < img_count < 100:
            return [f"[!] Very small dataset ({img_count} images). Consider data"
                    f" augmentation or collecting more data."]
        if 0 < img_count < 500:
            return [f"[~] Small dataset ({img_count} images). Augmentation is"
                    f" recommended for better generalization."]
        return []

    # -- training history analysis -----------------------------------

    @staticmethod
    def _analyze_history(history: Any) -> dict:
        """Parse and analyze a training history structure.

        Returns ``{text, train_losses, val_losses, lrs}``.
        """
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        lrs: List[Optional[float]] = []

        if isinstance(history, list):
            for entry in history:
                if isinstance(entry, dict):
                    train_losses.append(entry.get("loss", entry.get("train_loss", 0)))
                    val_losses.append(entry.get("val_loss"))
                    lrs.append(entry.get("lr", entry.get("learning_rate")))
                elif isinstance(entry, (int, float)):
                    train_losses.append(float(entry))
        elif isinstance(history, dict):
            train_losses = history.get("losses", history.get("train_losses", []))
            val_losses = history.get("val_losses", [])
            lrs = history.get("lrs", history.get("learning_rates", []))

        if not train_losses:
            return {"text": "No loss data found in history file.",
                    "train_losses": [], "val_losses": [], "lrs": []}

        n = len(train_losses)
        lines: List[str] = []

        # Monotonic decrease check
        decreasing_count = sum(
            1 for i in range(1, n) if train_losses[i] < train_losses[i - 1]
        )
        if decreasing_count == n - 1:
            lines.append("Loss curve shape: [OK] Monotonically decreasing")
        elif decreasing_count > n * 0.7:
            lines.append("Loss curve shape: [OK] Generally decreasing")
        else:
            lines.append("Loss curve shape: [!] Non-monotonic -- possible instability")

        # Plateau detection
        if n >= 3:
            last_3_delta = abs(train_losses[-1] - train_losses[-3])
            if last_3_delta < 0.001 * abs(train_losses[0]):
                plateau_start = max(1, n - 3)
                lines.append(f"Convergence: Reached plateau around epoch {plateau_start}")
            else:
                lines.append("Convergence: Still improving -- more epochs may help")
        else:
            lines.append("Convergence: Too few epochs to determine")

        # Overfitting detection
        valid_val = [v for v in val_losses if v is not None]
        if valid_val and len(valid_val) >= 3:
            val_increasing = sum(
                1 for i in range(1, len(valid_val))
                if valid_val[i] > valid_val[i - 1]
            )
            if val_increasing > len(valid_val) * 0.5:
                lines.append("Overfitting risk: [!] High (validation loss increasing)")
            else:
                lines.append("Overfitting risk: [OK] Low (no val loss increase)")
        else:
            lines.append("Overfitting risk: N/A (no validation loss data)")

        # Recommendation
        if n < 5:
            lines.append(f"Recommendation: Train more epochs (currently {n})")
        elif n >= 3 and abs(train_losses[-1] - train_losses[-3]) < 0.001 * abs(train_losses[0]):
            lines.append("Recommendation: Training has converged. Stop or reduce LR.")
        else:
            lines.append("Recommendation: Train 3-5 more epochs")

        return {
            "text": "\n".join(lines),
            "train_losses": train_losses,
            "val_losses": valid_val,
            "lrs": lrs,
        }

    # -- dataset counting helper -------------------------------------

    @staticmethod
    def _count_dataset(dataset_dir: str, dp: dict) -> Tuple[int, int, List[str]]:
        ds_path = Path(dataset_dir)
        img_count = 0
        ann_count = 0
        cat_names: List[str] = []

        images_subdir = dp.get("images_dir", "all_images")
        img_dir = ds_path / images_subdir
        if img_dir.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            img_count = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in exts)

        ann_file = dp.get("annotations_file", "annotations/mixed_annotations.json")
        ann_path = ds_path / ann_file
        if ann_path.is_file():
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    coco = json.load(f)
                ann_count = len(coco.get("annotations", []))
                cat_names = [
                    c.get("name", f"id={c.get('id')}") for c in coco.get("categories", [])
                ]
            except Exception:
                pass

        return img_count, ann_count, cat_names
