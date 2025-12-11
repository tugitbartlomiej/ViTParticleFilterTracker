"""
SAM3 (Segment Anything Model 3) Feature Extractor for Dataset Selection.

Uses HuggingFace transformers SAM3 for automatic mask generation.
Extracts segmentation masks and computes scene complexity metrics:
- Number of segments
- Segment size distribution
- Coverage ratio
- Edge density
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAMExtractor:
    """Extracts segmentation-based features using SAM3 from HuggingFace."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 points_per_batch: int = 64,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 100,
                 # Legacy parameters (ignored, kept for compatibility)
                 checkpoint_path: Optional[str] = None,
                 model_type: str = "vit_h",
                 points_per_side: int = 32):
        """
        Initialize SAM3 Extractor.

        Args:
            model_path: Path to local SAM3 HuggingFace model or "facebook/sam3"
            device: Device to run on ('cuda' or 'cpu')
            points_per_batch: Points per batch for automatic mask generation
            pred_iou_thresh: Predicted IoU threshold
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask region area
        """
        self.model_path = model_path or "facebook/sam3"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        self.model = None
        self.processor = None
        self.pipeline = None
        self._initialized = False

    def unload(self):
        """Unload model from GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
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
        logger.info("SAM3 model unloaded from memory")

    def _initialize_model(self):
        """Lazy initialization of SAM3 model."""
        if self._initialized:
            return True

        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading SAM3 model from {self.model_path}")

            # Use mask-generation pipeline for automatic segmentation
            self.pipeline = hf_pipeline(
                "mask-generation",
                model=self.model_path,
                device=0 if self.device == "cuda" else -1,
                points_per_batch=self.points_per_batch
            )

            self._initialized = True
            logger.info("SAM3 model initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"transformers not installed or SAM3 not available: {e}")
            logger.error("Install with: pip install transformers>=4.50")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            return False

    def extract_masks(self, image: np.ndarray) -> List[Dict]:
        """
        Extract automatic masks from image using SAM3.

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            List of mask dictionaries with keys:
            - 'segmentation': Binary mask (numpy array)
            - 'area': Mask area
            - 'bbox': Bounding box [x, y, w, h]
            - 'predicted_iou': Predicted IoU score (if available)
            - 'stability_score': Stability score (if available)
        """
        if not self._initialize_model():
            return []

        try:
            # Convert BGR to RGB if needed (check if image looks BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR from cv2, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Convert to PIL Image for pipeline
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)

            # Run mask generation pipeline
            outputs = self.pipeline(pil_image, points_per_batch=self.points_per_batch)

            # Convert outputs to standard format
            masks = []
            for mask_data in outputs.get('masks', []):
                # mask_data is already a numpy array from pipeline
                if isinstance(mask_data, np.ndarray):
                    mask_np = mask_data
                else:
                    mask_np = np.array(mask_data)

                # Ensure binary mask
                if mask_np.dtype != bool:
                    mask_np = mask_np > 0.5

                area = int(np.sum(mask_np))

                # Skip small masks
                if area < self.min_mask_region_area:
                    continue

                # Compute bounding box
                rows = np.any(mask_np, axis=1)
                cols = np.any(mask_np, axis=0)
                if not np.any(rows) or not np.any(cols):
                    continue

                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]

                masks.append({
                    'segmentation': mask_np,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': 1.0,  # Not available from pipeline
                    'stability_score': 1.0  # Not available from pipeline
                })

            return masks

        except Exception as e:
            logger.error(f"Error generating masks with SAM3: {e}")
            return []

    def compute_scene_complexity(self, masks: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Compute scene complexity metrics from masks.

        Args:
            masks: List of mask dictionaries from SAM3
            image_shape: (height, width) of original image

        Returns:
            Dictionary with complexity metrics
        """
        if not masks:
            return {
                'num_segments': 0,
                'avg_segment_size': 0,
                'segment_size_std': 0,
                'coverage_ratio': 0,
                'edge_density': 0,
                'avg_stability': 0,
                'avg_iou': 0,
                'complexity_score': 0
            }

        height, width = image_shape[:2]
        total_pixels = height * width

        # Extract metrics from masks
        areas = [m['area'] for m in masks]
        stability_scores = [m.get('stability_score', 1.0) for m in masks]
        iou_scores = [m.get('predicted_iou', 1.0) for m in masks]

        # Compute edge density from masks
        edge_pixels = 0
        for mask_dict in masks:
            mask = mask_dict['segmentation'].astype(np.uint8)
            edges = cv2.Canny(mask * 255, 100, 200)
            edge_pixels += np.sum(edges > 0)

        # Normalize metrics
        num_segments = len(masks)
        avg_segment_size = np.mean(areas) / total_pixels if areas else 0
        segment_size_std = np.std(areas) / total_pixels if len(areas) > 1 else 0
        coverage_ratio = min(sum(areas), total_pixels) / total_pixels
        edge_density = edge_pixels / total_pixels
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        avg_iou = np.mean(iou_scores) if iou_scores else 0

        # Composite complexity score (0-1)
        # Higher = more complex scene
        complexity_score = (
            0.3 * min(num_segments / 50, 1.0) +  # Number of segments (normalized)
            0.2 * segment_size_std * 10 +         # Size diversity
            0.2 * edge_density * 10 +             # Edge density
            0.2 * coverage_ratio +                # Coverage
            0.1 * (1 - avg_stability)             # Instability (harder scenes)
        )
        complexity_score = min(max(complexity_score, 0), 1)

        return {
            'num_segments': num_segments,
            'avg_segment_size': float(avg_segment_size),
            'segment_size_std': float(segment_size_std),
            'coverage_ratio': float(coverage_ratio),
            'edge_density': float(edge_density),
            'avg_stability': float(avg_stability),
            'avg_iou': float(avg_iou),
            'complexity_score': float(complexity_score)
        }

    def compute_features(self, image: np.ndarray) -> Dict:
        """
        Compute all SAM3-based features for an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Dictionary with all features
        """
        masks = self.extract_masks(image)
        complexity = self.compute_scene_complexity(masks, image.shape)

        return {
            'masks': masks,
            'complexity': complexity,
            'complexity_score': complexity['complexity_score']
        }

    def compute_features_from_path(self, image_path: str) -> Optional[Dict]:
        """Compute features from image path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
            return self.compute_features(image)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def compute_complexity_batch(self,
                                  image_paths: List[str],
                                  show_progress: bool = True) -> Tuple[List[float], List[str]]:
        """
        Compute complexity scores for a batch of images.

        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (complexity scores, valid paths)
        """
        scores = []
        valid_paths = []

        iterator = tqdm(image_paths, desc="Computing SAM3 complexity") if show_progress else image_paths

        for path in iterator:
            result = self.compute_features_from_path(path)
            if result is not None:
                scores.append(result['complexity_score'])
                valid_paths.append(path)

        logger.info(f"Computed complexity for {len(valid_paths)}/{len(image_paths)} images")
        return scores, valid_paths

    def compute_complexity_without_sam(self, image: np.ndarray) -> Dict:
        """
        Compute approximate complexity without SAM model.
        Uses edge detection and blob detection as proxy.

        FALLBACK METHOD - used when SAM3 is not available.

        Args:
            image: Input image (BGR format)

        Returns:
            Dictionary with approximate complexity metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        # Blob detection (simple connected components)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Filter small blobs
        min_area = 100
        valid_blobs = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
        num_segments = len(valid_blobs)

        # Size statistics
        if valid_blobs:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_blobs]
            avg_segment_size = np.mean(areas) / (height * width)
            segment_size_std = np.std(areas) / (height * width)
        else:
            avg_segment_size = 0
            segment_size_std = 0

        # Texture complexity (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var() / 10000  # Normalize

        # Composite score
        complexity_score = (
            0.3 * min(num_segments / 30, 1.0) +
            0.3 * edge_density * 5 +
            0.2 * min(texture_variance, 1.0) +
            0.2 * segment_size_std * 10
        )
        complexity_score = min(max(complexity_score, 0), 1)

        return {
            'num_segments': num_segments,
            'avg_segment_size': float(avg_segment_size),
            'segment_size_std': float(segment_size_std),
            'edge_density': float(edge_density),
            'texture_variance': float(texture_variance),
            'complexity_score': float(complexity_score)
        }


if __name__ == "__main__":
    print("Testing SAM3Extractor...")

    # Test with local model path
    extractor = SAMExtractor(
        model_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/External/Models/sam3"
    )

    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test proxy method (always works)
    print("\nTesting proxy method...")
    complexity = extractor.compute_complexity_without_sam(test_image)
    print(f"Complexity metrics (proxy): {complexity}")

    # Test full SAM3 (requires model)
    print("\nTesting SAM3 model...")
    try:
        result = extractor.compute_features(test_image)
        print(f"SAM3 complexity: {result['complexity_score']:.3f}")
        print(f"Number of masks: {len(result['masks'])}")
    except Exception as e:
        print(f"SAM3 test failed (expected if model not available): {e}")

    print("\nSAM3Extractor test completed!")
