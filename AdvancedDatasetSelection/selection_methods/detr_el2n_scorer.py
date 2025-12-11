"""
DETR-based EL2N Scorer for Dataset Selection.

Uses a pre-trained DETR model with QUERY 81 ONLY (best for tooltip detection).
For object detection, difficulty is measured by:
- Low confidence from Q81 = hard sample
- No detection from Q81 = very hard sample
- High confidence from Q81 = easy sample

Includes visualization of detections for reporting.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query 81 is the best for tooltip detection (from benchmark analysis)
DETR_QUERY_ID = 81


class DETR_EL2N_Scorer:
    """
    Computes EL2N-like difficulty scores using DETR with Query 81 ONLY.

    Query 81 has been identified as the best query for tooltip detection
    based on benchmark analysis.
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = "cuda",
                 confidence_threshold: float = 0.3,
                 num_labels: int = 1,
                 save_visualizations: bool = True,
                 vis_output_dir: str = None):
        """
        Initialize DETR EL2N Scorer.

        Args:
            checkpoint_path: Path to DETR checkpoint (.pth file)
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Threshold for valid detections
            num_labels: Number of classes (1 = tooltip only)
            save_visualizations: Whether to save detection visualizations
            vis_output_dir: Directory for visualizations
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.num_labels = num_labels
        self.save_visualizations = save_visualizations
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None

        self.model = None
        self.processor = None
        self._initialized = False
        self.scores = {}
        self.detection_info = {}  # Store detection info for reporting

    def unload(self):
        """Unload model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._initialized = False

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("DETR EL2N model unloaded from memory")

    def _initialize_model(self):
        """Load DETR model from checkpoint."""
        if self._initialized:
            return

        logger.info(f"Loading DETR from checkpoint: {self.checkpoint_path}")
        logger.info(f"Using QUERY {DETR_QUERY_ID} ONLY for detections")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Detect num_labels from checkpoint
        classifier_weight = checkpoint['model_state_dict'].get('class_labels_classifier.weight')
        if classifier_weight is not None:
            detected_num_classes = classifier_weight.shape[0]
            logger.info(f"Detected {detected_num_classes} output classes from checkpoint")
            self.num_labels = detected_num_classes - 1 if detected_num_classes > 1 else 1

        # Create model config
        id2label = {0: "tooltip"}
        label2id = {"tooltip": 0}

        config = DetrConfig.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id
        )

        # Create model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            config=config,
            ignore_mismatched_sizes=True
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Create processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        self._initialized = True
        logger.info(f"DETR model loaded (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")

    def compute_single_image_difficulty(self, image_path: str, save_vis: bool = False) -> Dict:
        """
        Compute difficulty metrics using ONLY Query 81.

        Args:
            image_path: Path to image
            save_vis: Whether to save visualization

        Returns:
            Dictionary with difficulty metrics and detection info
        """
        self._initialize_model()

        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits and boxes
        logits = outputs.logits[0]  # [100, num_classes+1]
        boxes = outputs.pred_boxes[0]  # [100, 4] normalized cxcywh

        # Apply softmax
        probs = F.softmax(logits, dim=-1)

        # Get ONLY Query 81's prediction (class 0 = tooltip)
        q81_score = float(probs[DETR_QUERY_ID, 0])
        q81_no_object = float(probs[DETR_QUERY_ID, -1])  # "no object" probability

        # Get Q81 box
        q81_box_normalized = boxes[DETR_QUERY_ID].cpu().numpy()  # [cx, cy, w, h] normalized
        cx, cy, w, h = q81_box_normalized

        # Convert to pixel coordinates (xyxy)
        x1 = int((cx - w/2) * img_width)
        y1 = int((cy - h/2) * img_height)
        x2 = int((cx + w/2) * img_width)
        y2 = int((cy + h/2) * img_height)
        q81_box_pixel = [x1, y1, x2, y2]

        # Determine if Q81 detected anything
        has_detection = q81_score >= self.confidence_threshold

        if has_detection:
            # Detection = easier sample (high confidence = low difficulty)
            difficulty = 1.0 - q81_score
        else:
            # No detection from Q81 = hard sample
            difficulty = 1.0

        # Also compute entropy for Q81's prediction
        q81_probs = probs[DETR_QUERY_ID]
        entropy = -torch.sum(q81_probs * torch.log(q81_probs + 1e-10)).item()
        max_entropy = np.log(q81_probs.shape[0])
        normalized_entropy = entropy / max_entropy

        # Combined EL2N score
        el2n_score = 0.7 * difficulty + 0.3 * normalized_entropy

        result = {
            'difficulty': float(difficulty),
            'q81_score': float(q81_score),
            'q81_no_object': float(q81_no_object),
            'has_detection': has_detection,
            'q81_box': q81_box_pixel,
            'entropy': float(normalized_entropy),
            'el2n_score': float(el2n_score),
            'image_size': (img_width, img_height)
        }

        # Save visualization if requested
        if save_vis and self.vis_output_dir:
            self._save_detection_visualization(image, result, image_path)

        return result

    def _save_detection_visualization(self, image: Image.Image, result: Dict, image_path: str):
        """Save visualization of Q81 detection."""
        self.vis_output_dir.mkdir(parents=True, exist_ok=True)

        draw = ImageDraw.Draw(image)

        # Draw Q81 box
        box = result['q81_box']
        score = result['q81_score']
        has_det = result['has_detection']

        # Color based on detection status
        if has_det:
            color = (0, 255, 0)  # Green for detection
            status = "DETECTED"
        else:
            color = (255, 0, 0)  # Red for no detection
            status = "NO DETECTION"

        # Draw box
        draw.rectangle(box, outline=color, width=3)

        # Draw label
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        label = f"Q81: {score:.2f} ({status})"
        draw.text((box[0], box[1] - 20), label, fill=color, font=font)

        # Draw difficulty score
        el2n = result['el2n_score']
        diff_label = f"EL2N: {el2n:.3f}"
        draw.text((10, 10), diff_label, fill=(255, 255, 0), font=font)

        # Save
        img_name = Path(image_path).stem
        save_path = self.vis_output_dir / f"{img_name}_q81_detection.jpg"
        image.save(save_path)

    def compute_el2n_scores(self,
                            image_paths: List[str],
                            batch_size: int = 8,
                            show_progress: bool = True,
                            save_vis_for_selected: bool = False) -> Dict[str, float]:
        """
        Compute EL2N scores for all images using Query 81 only.

        Args:
            image_paths: List of image paths
            batch_size: Batch size (not used, single image processing)
            show_progress: Whether to show progress bar
            save_vis_for_selected: Whether to save visualizations

        Returns:
            Dictionary mapping image paths to EL2N scores
        """
        self._initialize_model()

        logger.info(f"Computing DETR Q81 EL2N scores for {len(image_paths)} images...")

        self.scores = {}
        self.detection_info = {}

        iterator = tqdm(image_paths, desc="Computing DETR Q81 EL2N") if show_progress else image_paths

        detected_count = 0
        no_detection_count = 0

        for path in iterator:
            try:
                result = self.compute_single_image_difficulty(path, save_vis=save_vis_for_selected)
                self.scores[path] = result['el2n_score']
                self.detection_info[path] = result

                if result['has_detection']:
                    detected_count += 1
                else:
                    no_detection_count += 1

            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")
                self.scores[path] = 0.5
                self.detection_info[path] = {'error': str(e)}

        logger.info(f"Q81 Detection stats: {detected_count} detected, {no_detection_count} no detection")

        return self.scores

    def get_detection_report(self) -> Dict:
        """
        Get detailed report of all detections for explaining selections.

        Returns:
            Dictionary with detection statistics and per-image info
        """
        if not self.detection_info:
            return {}

        # Compute statistics
        q81_scores = [info.get('q81_score', 0) for info in self.detection_info.values() if 'q81_score' in info]
        el2n_scores = [info.get('el2n_score', 0) for info in self.detection_info.values() if 'el2n_score' in info]
        detections = [info.get('has_detection', False) for info in self.detection_info.values() if 'has_detection' in info]

        report = {
            'query_used': DETR_QUERY_ID,
            'confidence_threshold': self.confidence_threshold,
            'total_images': len(self.detection_info),
            'images_with_detection': sum(detections),
            'images_without_detection': len(detections) - sum(detections),
            'detection_rate': sum(detections) / len(detections) if detections else 0,
            'q81_score_stats': {
                'mean': float(np.mean(q81_scores)) if q81_scores else 0,
                'std': float(np.std(q81_scores)) if q81_scores else 0,
                'min': float(np.min(q81_scores)) if q81_scores else 0,
                'max': float(np.max(q81_scores)) if q81_scores else 0,
            },
            'el2n_score_stats': {
                'mean': float(np.mean(el2n_scores)) if el2n_scores else 0,
                'std': float(np.std(el2n_scores)) if el2n_scores else 0,
                'min': float(np.min(el2n_scores)) if el2n_scores else 0,
                'max': float(np.max(el2n_scores)) if el2n_scores else 0,
            },
            'per_image_info': self.detection_info
        }

        return report

    def save_selected_visualizations(self, selected_paths: List[str], output_dir: str):
        """
        Save visualizations for selected images with detection boxes.

        Args:
            selected_paths: List of selected image paths
            output_dir: Output directory for visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving Q81 detection visualizations for {len(selected_paths)} selected images...")

        for path in tqdm(selected_paths, desc="Saving visualizations"):
            try:
                image = Image.open(path).convert('RGB')
                info = self.detection_info.get(path, {})

                if 'q81_box' in info:
                    draw = ImageDraw.Draw(image)

                    box = info['q81_box']
                    score = info['q81_score']
                    has_det = info['has_detection']
                    el2n = info['el2n_score']

                    # Color based on detection
                    color = (0, 255, 0) if has_det else (255, 0, 0)

                    # Draw box
                    draw.rectangle(box, outline=color, width=3)

                    # Try to load font
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                        small_font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                        small_font = font

                    # Draw labels
                    status = "DETECTED" if has_det else "NO DETECTION"
                    draw.text((box[0], box[1] - 25), f"Q81: {score:.3f}", fill=color, font=font)

                    # Draw EL2N score and reason
                    draw.rectangle([5, 5, 250, 80], fill=(0, 0, 0, 180))
                    draw.text((10, 10), f"EL2N: {el2n:.3f}", fill=(255, 255, 0), font=font)
                    draw.text((10, 35), f"Q81 Score: {score:.3f}", fill=(255, 255, 255), font=small_font)
                    draw.text((10, 55), f"Status: {status}", fill=color, font=small_font)

                # Save with original filename (preserving name for COCO lookup)
                original_filename = Path(path).name
                save_path = output_dir / original_filename
                image.save(save_path, quality=95)

            except Exception as e:
                logger.warning(f"Error saving visualization for {path}: {e}")

        logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    print("Testing DETR Q81 EL2N Scorer...")
    print(f"Using Query {DETR_QUERY_ID} for tooltip detection")

    # Test with checkpoint
    checkpoint = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/checkpoint_epoch_170.pth"

    if Path(checkpoint).exists():
        scorer = DETR_EL2N_Scorer(checkpoint)

        # Test single image
        test_image = "E:/cataract_surgery_Instruments_detection.v1i.coco/valid/images/frame_1025_jpg.rf.b0f62b7b16f5baabc1063020c3ea2db9.jpg"
        if Path(test_image).exists():
            result = scorer.compute_single_image_difficulty(test_image)
            print(f"\nTest result for {Path(test_image).name}:")
            print(f"  Q81 Score: {result['q81_score']:.4f}")
            print(f"  Has Detection: {result['has_detection']}")
            print(f"  EL2N Score: {result['el2n_score']:.4f}")
            print(f"  Q81 Box: {result['q81_box']}")

        scorer.unload()
    else:
        print(f"Checkpoint not found: {checkpoint}")

    print("\nDETR Q81 EL2N Scorer test completed!")
