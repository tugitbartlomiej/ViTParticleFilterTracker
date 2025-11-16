#!/usr/bin/env python3
"""
Benchmark Visualization Script
================================

Generates visual comparisons of model predictions vs ground truth.

For each model in the benchmark:
- Blue bounding boxes = Ground Truth (from COCO annotations)
- Green bounding boxes = Model predictions
- Samples 100 images uniformly from ~4253 validation images

Output structure:
    benchmark_visualizations/
    ├── yolo_best/
    │   ├── image_0000.jpg
    │   ├── image_0042.jpg
    │   └── ...
    ├── detr_base/
    ├── detr_epoch140/
    └── summary_report.json

Author: Benchmark Visualization System
Date: 2025-11-13
"""

import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging

# HuggingFace Transformers for DETR
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig

# Ultralytics for YOLO
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """
    Main benchmark visualization system.

    Compares multiple models (YOLO, DETR variants) against ground truth
    with visual outputs showing GT (blue) vs predictions (green).
    """

    def __init__(
        self,
        coco_annotations_path: str,
        images_dir: str,
        output_dir: str = "benchmark_visualizations",
        num_samples: int = 100,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize benchmark visualizer.

        Args:
            coco_annotations_path: Path to COCO format annotations JSON
            images_dir: Directory containing validation images
            output_dir: Output directory for visualizations
            num_samples: Number of images to sample (default: 100)
            confidence_threshold: Confidence threshold for predictions
        """
        self.coco_path = Path(coco_annotations_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")

        # Load COCO annotations
        self.coco_data = self._load_coco_annotations()
        self.image_id_to_path = self._build_image_mapping()
        self.image_id_to_annotations = self._build_annotation_mapping()

        # Models dictionary (will be populated)
        self.models = {}

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"✅ Benchmark Visualizer initialized")
        logger.info(f"📊 Total images: {len(self.coco_data['images'])}")
        logger.info(f"🎯 Will sample: {self.num_samples} images")

    def _load_coco_annotations(self) -> Dict:
        """Load and validate COCO annotations."""
        logger.info(f"📂 Loading COCO annotations from: {self.coco_path}")

        with open(self.coco_path, 'r') as f:
            coco_data = json.load(f)

        logger.info(f"   Images: {len(coco_data['images'])}")
        logger.info(f"   Annotations: {len(coco_data['annotations'])}")
        logger.info(f"   Categories: {coco_data['categories']}")

        return coco_data

    def _build_image_mapping(self) -> Dict[int, Path]:
        """Build mapping from image_id to file path."""
        mapping = {}
        for img_info in self.coco_data['images']:
            img_id = img_info['id']
            img_path = self.images_dir / img_info['file_name']
            mapping[img_id] = img_path
        return mapping

    def _build_annotation_mapping(self) -> Dict[int, List[Dict]]:
        """Build mapping from image_id to list of annotations."""
        mapping = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(ann)
        return mapping

    def sample_images(self) -> List[int]:
        """
        Sample images uniformly from the dataset.

        Returns:
            List of sampled image IDs
        """
        total_images = len(self.coco_data['images'])

        # Calculate sampling interval
        interval = total_images // self.num_samples

        # Sample image IDs uniformly
        sampled_ids = []
        for i in range(self.num_samples):
            idx = i * interval
            if idx < total_images:
                img_id = self.coco_data['images'][idx]['id']
                sampled_ids.append(img_id)

        logger.info(f"✅ Sampled {len(sampled_ids)} images (every {interval}th image)")
        return sampled_ids

    def add_model(self, model_name: str, model_loader_fn):
        """
        Add a model to the benchmark.

        Args:
            model_name: Name of the model (used for folder naming)
            model_loader_fn: Function that returns a model wrapper
        """
        logger.info(f"➕ Adding model: {model_name}")
        self.models[model_name] = model_loader_fn()

        # Create output directory for this model
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"   Output dir: {model_output_dir}")

    def visualize_sample(
        self,
        image_id: int,
        model_name: str,
        predictions: List[Dict],
        save_path: Path
    ):
        """
        Create visualization with GT (blue) and predictions (green).

        Args:
            image_id: COCO image ID
            model_name: Name of the model
            predictions: List of prediction dicts with 'bbox' and 'score'
            save_path: Where to save the visualization
        """
        # Load image
        img_path = self.image_id_to_path[image_id]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get ground truth annotations
        gt_annotations = self.image_id_to_annotations.get(image_id, [])

        # Create matplotlib figure
        fig, ax = plt.subplots(1, figsize=(16, 9), dpi=100)
        ax.imshow(image)

        # Draw ground truth boxes (BLUE)
        for ann in gt_annotations:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = bbox

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3,
                edgecolor='blue',
                facecolor='none',
                label='Ground Truth'
            )
            ax.add_patch(rect)

            # Add GT label
            ax.text(
                x, y - 10,
                'GT',
                color='blue',
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3)
            )

        # Draw prediction boxes (GREEN)
        for pred in predictions:
            bbox = pred['bbox']  # Should be [x, y, width, height]
            score = pred['score']
            x, y, w, h = bbox

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3,
                edgecolor='green',
                facecolor='none',
                label='Prediction'
            )
            ax.add_patch(rect)

            # Add prediction label with confidence
            ax.text(
                x, y + h + 20,
                f'Pred: {score:.2f}',
                color='green',
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3)
            )

        # Add title and legend
        ax.set_title(
            f"{model_name} - Image ID: {image_id}\n"
            f"GT boxes: {len(gt_annotations)} | Predictions: {len(predictions)}",
            fontsize=14,
            weight='bold'
        )

        # Create legend (avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

        ax.axis('off')
        plt.tight_layout()

        # Save
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    def run_benchmark(self):
        """
        Run complete benchmark for all models.

        For each model:
        1. Sample images
        2. Run inference
        3. Generate visualizations
        4. Save results
        """
        # Sample images once (same samples for all models)
        sampled_image_ids = self.sample_images()

        # Store results for summary
        summary_results = {
            'num_samples': len(sampled_image_ids),
            'confidence_threshold': self.confidence_threshold,
            'models': {}
        }

        # Process each model
        for model_name, model_wrapper in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🔧 Processing model: {model_name}")
            logger.info(f"{'='*60}")

            model_output_dir = self.output_dir / model_name
            model_stats = {
                'total_predictions': 0,
                'total_gt': 0,
                'images_processed': 0
            }

            # Process each sampled image
            for idx, image_id in enumerate(tqdm(sampled_image_ids, desc=f"{model_name}")):
                try:
                    # Get image path
                    img_path = self.image_id_to_path[image_id]

                    # Run inference
                    predictions = model_wrapper.predict(
                        img_path,
                        confidence_threshold=self.confidence_threshold
                    )

                    # Get ground truth count
                    gt_count = len(self.image_id_to_annotations.get(image_id, []))

                    # Update statistics
                    model_stats['total_predictions'] += len(predictions)
                    model_stats['total_gt'] += gt_count
                    model_stats['images_processed'] += 1

                    # Create visualization
                    save_path = model_output_dir / f"image_{idx:04d}_id{image_id}.jpg"
                    self.visualize_sample(
                        image_id=image_id,
                        model_name=model_name,
                        predictions=predictions,
                        save_path=save_path
                    )

                except Exception as e:
                    logger.error(f"❌ Error processing image {image_id}: {e}")
                    continue

            # Save model statistics
            summary_results['models'][model_name] = model_stats
            logger.info(f"✅ {model_name} complete:")
            logger.info(f"   Images processed: {model_stats['images_processed']}")
            logger.info(f"   Total GT boxes: {model_stats['total_gt']}")
            logger.info(f"   Total predictions: {model_stats['total_predictions']}")

        # Save summary report
        summary_path = self.output_dir / "summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ BENCHMARK COMPLETE!")
        logger.info(f"📊 Summary saved to: {summary_path}")
        logger.info(f"{'='*60}")


# ============================================================================
# MODEL WRAPPERS
# ============================================================================

class YOLOWrapper:
    """Wrapper for YOLO model inference."""

    def __init__(self, model_path: str):
        """Initialize YOLO model."""
        self.model = YOLO(model_path)
        logger.info(f"✅ YOLO model loaded from: {model_path}")

    def predict(self, image_path: Path, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Run YOLO inference.

        Returns:
            List of predictions with 'bbox' [x, y, w, h] and 'score'
        """
        results = self.model(str(image_path), verbose=False)[0]

        predictions = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= confidence_threshold:
                # YOLO returns xyxy format, convert to xywh
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                predictions.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': conf
                })

        return predictions


class DETRWrapper:
    """Wrapper for DETR model inference."""

    def __init__(self, model_path: str, num_labels: int = 1):
        """Initialize DETR model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle nested structure
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            if isinstance(model_state_dict, dict) and 'model_state_dict' in model_state_dict:
                model_state_dict = model_state_dict['model_state_dict']
        else:
            model_state_dict = checkpoint

        # Create model with config
        config = DetrConfig.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            num_queries=100,
            id2label={0: "tool"},
            label2id={"tool": 0}
        )

        self.model = DetrForObjectDetection(config)
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Load processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        logger.info(f"✅ DETR model loaded from: {model_path}")

    def predict(self, image_path: Path, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Run DETR inference.

        Returns:
            List of predictions with 'bbox' [x, y, w, h] and 'score'
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get image size
        img_width, img_height = image.size
        target_sizes = torch.tensor([[img_height, img_width]]).to(self.device)

        # Post-process
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]

        # Convert to standard format
        predictions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # DETR returns xyxy format, convert to xywh
            x1, y1, x2, y2 = box.cpu().numpy()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            predictions.append({
                'bbox': [float(x), float(y), float(w), float(h)],
                'score': float(score)
            })

        return predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main benchmark execution."""

    # Configuration
    COCO_ANNOTATIONS = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\coco_annotations_from_yolo_dataset_20250218.json"
    IMAGES_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Yolo\images\val"
    OUTPUT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\benchmark_visualizations"

    # Model paths
    YOLO_MODEL = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\epoch70.pt"
    DETR_BASE = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\checkpoint_epoch_100.pth"
    DETR_EPOCH140 = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\checkpoint_epoch_140.pth"

    # Initialize visualizer
    visualizer = BenchmarkVisualizer(
        coco_annotations_path=COCO_ANNOTATIONS,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=100,
        confidence_threshold=0.5
    )

    # Add models
    logger.info("\n" + "="*60)
    logger.info("🚀 ADDING MODELS TO BENCHMARK")
    logger.info("="*60)

    visualizer.add_model(
        "yolo_best",
        lambda: YOLOWrapper(YOLO_MODEL)
    )

    visualizer.add_model(
        "detr_base",
        lambda: DETRWrapper(DETR_BASE, num_labels=1)
    )

    visualizer.add_model(
        "detr_epoch140",
        lambda: DETRWrapper(DETR_EPOCH140, num_labels=1)
    )

    # Run benchmark
    logger.info("\n" + "="*60)
    logger.info("🚀 STARTING BENCHMARK")
    logger.info("="*60)

    visualizer.run_benchmark()

    logger.info("\n✅ ALL DONE!")


if __name__ == "__main__":
    main()
