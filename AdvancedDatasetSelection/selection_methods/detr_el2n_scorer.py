"""
DETR-based EL2N Scorer for Dataset Selection.

Uses a pre-trained DETR model with QUERY 81 ONLY (best for tooltip detection).
For object detection, difficulty is measured by:
- Low confidence from Q81 = hard sample
- No detection from Q81 = very hard sample
- High confidence from Q81 = easy sample

Includes visualization of detections for reporting.

Optimized with:
- Batched inference for GPU efficiency
- Checkpoint saving for resumable processing
- DataLoader with parallel workers
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
import json
import time
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query 81 is the best for tooltip detection (from benchmark analysis)
DETR_QUERY_ID = 81

# Checkpoint settings
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N images


class ImageDataset(Dataset):
    """Dataset for efficient batch loading of images."""

    def __init__(self, image_paths: List[str], processor: DetrImageProcessor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            img_width, img_height = image.size
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'pixel_mask': inputs['pixel_mask'].squeeze(0),
                'path': path,
                'img_size': (img_width, img_height),
                'valid': True
            }
        except Exception as e:
            # Return placeholder for failed images
            return {
                'pixel_values': torch.zeros(3, 800, 800),
                'pixel_mask': torch.zeros(800, 800),
                'path': path,
                'img_size': (0, 0),
                'valid': False,
                'error': str(e)
            }


def collate_fn(batch):
    """Custom collate function to handle variable size images."""
    valid_items = [item for item in batch if item['valid']]
    invalid_items = [item for item in batch if not item['valid']]

    if not valid_items:
        return None

    # Stack tensors
    pixel_values = torch.stack([item['pixel_values'] for item in valid_items])
    pixel_mask = torch.stack([item['pixel_mask'] for item in valid_items])

    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'paths': [item['path'] for item in valid_items],
        'img_sizes': [item['img_size'] for item in valid_items],
        'invalid_items': invalid_items
    }


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
                 vis_output_dir: str = None,
                 progress_checkpoint_dir: str = None):
        """
        Initialize DETR EL2N Scorer.

        Args:
            checkpoint_path: Path to DETR checkpoint (.pth file)
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Threshold for valid detections
            num_labels: Number of classes (1 = tooltip only)
            save_visualizations: Whether to save detection visualizations
            vis_output_dir: Directory for visualizations
            progress_checkpoint_dir: Directory for saving progress checkpoints
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.num_labels = num_labels
        self.save_visualizations = save_visualizations
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None
        self.progress_checkpoint_dir = Path(progress_checkpoint_dir) if progress_checkpoint_dir else Path("./detr_el2n_checkpoints")

        self.model = None
        self.processor = None
        self._initialized = False
        self.scores = {}
        self.detection_info = {}  # Store detection info for reporting
        self._processed_count = 0
        self._start_time = None

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

    def _get_checkpoint_path(self) -> Path:
        """Get the path to the progress checkpoint file."""
        return self.progress_checkpoint_dir / "detr_el2n_progress.json"

    def _save_checkpoint(self, all_paths: List[str]):
        """Save current progress to checkpoint file."""
        self.progress_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'scores': self.scores,
            'detection_info': self.detection_info,
            'processed_count': self._processed_count,
            'total_paths': len(all_paths),
            'all_paths': all_paths,  # Store original order
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': time.time() - self._start_time if self._start_time else 0
        }

        checkpoint_path = self._get_checkpoint_path()
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved: {self._processed_count}/{len(all_paths)} images ({100*self._processed_count/len(all_paths):.1f}%)")

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load progress from checkpoint file if exists."""
        checkpoint_path = self._get_checkpoint_path()

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            logger.info(f"Found checkpoint: {checkpoint_data['processed_count']}/{checkpoint_data['total_paths']} images")
            logger.info(f"Checkpoint from: {checkpoint_data['timestamp']}")

            return checkpoint_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _get_remaining_paths(self, all_paths: List[str], checkpoint: Dict) -> List[str]:
        """Get list of paths that still need processing."""
        processed_paths = set(checkpoint['scores'].keys())
        remaining = [p for p in all_paths if p not in processed_paths]
        logger.info(f"Resuming: {len(remaining)} images remaining")
        return remaining

    def clear_checkpoint(self):
        """Clear the checkpoint file to start fresh."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint cleared: {checkpoint_path}")
        self.scores = {}
        self.detection_info = {}
        self._processed_count = 0

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

    def _process_batch_outputs(self, outputs, paths: List[str], img_sizes: List[Tuple[int, int]]) -> List[Dict]:
        """Process a batch of model outputs and return results for each image."""
        batch_results = []
        batch_size = len(paths)

        # Get logits and boxes for the batch
        logits = outputs.logits  # [batch, 100, num_classes+1]
        boxes = outputs.pred_boxes  # [batch, 100, 4]

        # Apply softmax
        probs = F.softmax(logits, dim=-1)

        for i in range(batch_size):
            img_width, img_height = img_sizes[i]

            # Get ONLY Query 81's prediction (class 0 = tooltip)
            q81_score = float(probs[i, DETR_QUERY_ID, 0])
            q81_no_object = float(probs[i, DETR_QUERY_ID, -1])

            # Get Q81 box
            q81_box_normalized = boxes[i, DETR_QUERY_ID].cpu().numpy()
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
                difficulty = 1.0 - q81_score
            else:
                difficulty = 1.0

            # Compute entropy for Q81's prediction
            q81_probs = probs[i, DETR_QUERY_ID]
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
            batch_results.append(result)

        return batch_results

    def compute_el2n_scores(self,
                            image_paths: List[str],
                            batch_size: int = 16,
                            num_workers: int = 4,
                            show_progress: bool = True,
                            save_vis_for_selected: bool = False,
                            resume: bool = True) -> Dict[str, float]:
        """
        Compute EL2N scores for all images using Query 81 only.

        OPTIMIZED VERSION with:
        - Batched GPU inference
        - DataLoader with parallel workers
        - Checkpoint saving every 1000 images
        - Resume from checkpoint support

        Args:
            image_paths: List of image paths
            batch_size: Batch size for GPU inference (default: 16)
            num_workers: Number of DataLoader workers (default: 4)
            show_progress: Whether to show progress bar
            save_vis_for_selected: Whether to save visualizations
            resume: Whether to resume from checkpoint if available

        Returns:
            Dictionary mapping image paths to EL2N scores
        """
        self._initialize_model()
        self._start_time = time.time()

        # Check for existing checkpoint
        paths_to_process = image_paths
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                # Restore previous progress
                self.scores = checkpoint['scores']
                self.detection_info = checkpoint['detection_info']
                self._processed_count = checkpoint['processed_count']

                # Get remaining paths
                paths_to_process = self._get_remaining_paths(image_paths, checkpoint)

                if not paths_to_process:
                    logger.info("All images already processed! Returning cached results.")
                    return self.scores
        else:
            self.scores = {}
            self.detection_info = {}
            self._processed_count = 0

        total_images = len(image_paths)
        remaining_images = len(paths_to_process)

        logger.info(f"Computing DETR Q81 EL2N scores: {remaining_images} images to process (batch_size={batch_size})")
        if self._processed_count > 0:
            logger.info(f"Resuming from checkpoint: {self._processed_count}/{total_images} already done")

        # Create dataset and dataloader
        # Windows workaround: multiprocessing can cause issues
        if sys.platform == 'win32' and num_workers > 0:
            logger.info("Windows detected: using num_workers=0 (multiprocessing issues)")
            num_workers = 0

        dataset = ImageDataset(paths_to_process, self.processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        detected_count = sum(1 for info in self.detection_info.values() if info.get('has_detection', False))
        no_detection_count = sum(1 for info in self.detection_info.values() if not info.get('has_detection', True) and 'error' not in info)

        # Process batches
        pbar = tqdm(dataloader, desc="Computing DETR Q81 EL2N (batched)", disable=not show_progress)

        batch_count = 0
        for batch in pbar:
            if batch is None:
                continue

            # Handle invalid items
            for invalid_item in batch['invalid_items']:
                path = invalid_item['path']
                self.scores[path] = 0.5
                self.detection_info[path] = {'error': invalid_item.get('error', 'Unknown error')}
                self._processed_count += 1

            # Process valid items
            if batch['pixel_values'].size(0) > 0:
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                pixel_mask = batch['pixel_mask'].to(self.device)

                # Inference
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                # Process outputs
                results = self._process_batch_outputs(outputs, batch['paths'], batch['img_sizes'])

                # Store results
                for path, result in zip(batch['paths'], results):
                    self.scores[path] = result['el2n_score']
                    self.detection_info[path] = result
                    self._processed_count += 1

                    if result['has_detection']:
                        detected_count += 1
                    else:
                        no_detection_count += 1

            batch_count += 1

            # Update progress bar
            pbar.set_postfix({
                'processed': f"{self._processed_count}/{total_images}",
                'det': detected_count,
                'no_det': no_detection_count
            })

            # Save checkpoint periodically
            if batch_count % (CHECKPOINT_INTERVAL // batch_size) == 0:
                self._save_checkpoint(image_paths)

        # Final checkpoint save
        self._save_checkpoint(image_paths)

        elapsed = time.time() - self._start_time
        speed = self._processed_count / elapsed if elapsed > 0 else 0
        logger.info(f"Completed in {elapsed:.1f}s ({speed:.1f} img/s)")
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
