"""
SAM (Segment Anything Model) Feature Extractor for Dataset Selection.

Extracts segmentation masks and computes scene complexity metrics:
- Number of segments
- Segment size distribution
- Coverage ratio
- Edge density
- Tool region features
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
    """Extracts segmentation-based features using SAM."""

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 model_type: str = "vit_h",
                 device: str = "cuda",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 100):
        """
        Initialize SAM Extractor.

        Args:
            checkpoint_path: Path to SAM checkpoint file
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to run on ('cuda' or 'cpu')
            points_per_side: Points per side for automatic mask generation
            pred_iou_thresh: Predicted IoU threshold
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask region area
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        self.sam = None
        self.mask_generator = None
        self._initialized = False

    def _initialize_model(self):
        """Lazy initialization of SAM model."""
        if self._initialized:
            return

        if self.checkpoint_path is None:
            logger.warning("No checkpoint path provided, SAM features will not be available")
            return

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            logger.info(f"Loading SAM model ({self.model_type}) from {self.checkpoint_path}")
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)

            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
            self._initialized = True
            logger.info("SAM model initialized successfully")

        except ImportError:
            logger.error("segment_anything not installed. Install with: "
                        "pip install git+https://github.com/facebookresearch/segment-anything.git")
        except Exception as e:
            logger.error(f"Failed to initialize SAM: {e}")

    def extract_masks(self, image: np.ndarray) -> List[Dict]:
        """
        Extract automatic masks from image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of mask dictionaries with keys:
            - 'segmentation': Binary mask
            - 'area': Mask area
            - 'bbox': Bounding box [x, y, w, h]
            - 'predicted_iou': Predicted IoU score
            - 'stability_score': Stability score
        """
        self._initialize_model()

        if self.mask_generator is None:
            return []

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            masks = self.mask_generator.generate(image_rgb)
            return masks
        except Exception as e:
            logger.error(f"Error generating masks: {e}")
            return []

    def compute_scene_complexity(self, masks: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Compute scene complexity metrics from masks.

        Args:
            masks: List of mask dictionaries from SAM
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
        stability_scores = [m.get('stability_score', 0) for m in masks]
        iou_scores = [m.get('predicted_iou', 0) for m in masks]

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

    def extract_tool_region_features(self, masks: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract features for potential tool regions (largest masks).

        Args:
            masks: List of mask dictionaries
            image_shape: (height, width) of original image

        Returns:
            Feature vector for tool regions
        """
        if not masks:
            return np.zeros(6, dtype=np.float32)

        height, width = image_shape[:2]

        # Sort by area, take top 3 largest
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:3]

        features = []
        for mask_dict in sorted_masks:
            mask = mask_dict['segmentation']
            bbox = mask_dict['bbox']  # [x, y, w, h]

            # Centroid position (normalized)
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                centroid_x = np.mean(x_coords) / width
                centroid_y = np.mean(y_coords) / height
            else:
                centroid_x, centroid_y = 0.5, 0.5

            # Aspect ratio
            aspect_ratio = bbox[2] / max(bbox[3], 1)

            # Compactness (4*pi*area/perimeter^2)
            area = mask_dict['area']
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            compactness = 4 * np.pi * area / (perimeter**2 + 1e-10)

            features.extend([centroid_x, centroid_y, aspect_ratio, compactness])

        # Pad if less than 3 masks
        while len(features) < 12:
            features.extend([0, 0, 1, 0])

        # Return first 6 features (2 largest masks)
        return np.array(features[:6], dtype=np.float32)

    def compute_features(self, image: np.ndarray) -> Dict:
        """
        Compute all SAM-based features for an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Dictionary with all features
        """
        masks = self.extract_masks(image)
        complexity = self.compute_scene_complexity(masks, image.shape)
        tool_features = self.extract_tool_region_features(masks, image.shape)

        return {
            'masks': masks,
            'complexity': complexity,
            'tool_features': tool_features,
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

        iterator = tqdm(image_paths, desc="Computing SAM complexity") if show_progress else image_paths

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
    # Test the SAM extractor (without actual SAM model)
    print("Testing SAMExtractor...")

    extractor = SAMExtractor(checkpoint_path=None)

    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test without SAM (using proxy methods)
    complexity = extractor.compute_complexity_without_sam(test_image)
    print(f"Complexity metrics (proxy): {complexity}")

    print("\nSAMExtractor test completed!")
    print("Note: Full SAM functionality requires checkpoint file.")
