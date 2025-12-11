"""
FastSAM Feature Extractor for Dataset Selection.

Uses Ultralytics FastSAM for fast automatic mask generation.
Much faster than SAM3 (~10x) with acceptable quality.

Extracts:
- Number of segments
- Segment size distribution
- Coverage ratio
- Edge density
- Complexity score
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

# Default FastSAM model path
DEFAULT_FASTSAM_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/External/Models/FastSAM/FastSAM-x.pt"


class FastSAMExtractor:
    """Extracts segmentation-based features using FastSAM from Ultralytics."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 conf: float = 0.4,
                 iou: float = 0.9,
                 imgsz: int = 1024,
                 retina_masks: bool = True,
                 min_mask_area: int = 100):
        """
        Initialize FastSAM Extractor.

        Args:
            model_path: Path to FastSAM model (.pt file)
            device: Device to run on ('cuda' or 'cpu')
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            retina_masks: Use high-resolution masks
            min_mask_area: Minimum mask area to keep
        """
        self.model_path = model_path or DEFAULT_FASTSAM_PATH
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.retina_masks = retina_masks
        self.min_mask_area = min_mask_area

        self.model = None
        self._initialized = False

    def _initialize_model(self):
        """Lazy initialization of FastSAM model."""
        if self._initialized:
            return True

        try:
            from ultralytics import FastSAM

            logger.info(f"Loading FastSAM model from {self.model_path}")
            self.model = FastSAM(self.model_path)
            self._initialized = True
            logger.info(f"FastSAM model initialized on {self.device}")
            return True

        except ImportError as e:
            logger.error(f"ultralytics not installed: {e}")
            logger.error("Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize FastSAM: {e}")
            return False

    def unload(self):
        """Unload model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("FastSAM model unloaded from memory")

    def extract_masks(self, image_path: str) -> List[Dict]:
        """
        Extract automatic masks from image using FastSAM.

        Args:
            image_path: Path to input image

        Returns:
            List of mask dictionaries with keys:
            - 'segmentation': Binary mask (numpy array)
            - 'area': Mask area
            - 'bbox': Bounding box [x, y, w, h]
            - 'confidence': Confidence score
        """
        if not self._initialize_model():
            return []

        try:
            # Run FastSAM inference
            results = self.model(
                image_path,
                device=self.device,
                retina_masks=self.retina_masks,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            if not results or results[0].masks is None:
                return []

            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            masks = []
            for i, mask in enumerate(masks_data):
                mask_binary = mask > 0.5
                area = int(np.sum(mask_binary))

                # Skip small masks
                if area < self.min_mask_area:
                    continue

                # Get bounding box
                if boxes is not None and i < len(boxes):
                    box = boxes[i].xyxy[0].cpu().numpy()
                    bbox = [int(box[0]), int(box[1]),
                            int(box[2] - box[0]), int(box[3] - box[1])]
                    conf = float(boxes[i].conf[0]) if boxes[i].conf is not None else 1.0
                else:
                    # Compute bbox from mask
                    rows = np.any(mask_binary, axis=1)
                    cols = np.any(mask_binary, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                    conf = 1.0

                masks.append({
                    'segmentation': mask_binary,
                    'area': area,
                    'bbox': bbox,
                    'confidence': conf
                })

            return masks

        except Exception as e:
            logger.error(f"Error generating masks with FastSAM: {e}")
            return []

    def extract_masks_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Extract masks from numpy array image.

        Args:
            image: Input image (BGR format from cv2)

        Returns:
            List of mask dictionaries
        """
        if not self._initialize_model():
            return []

        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = self.model(
                image_rgb,
                device=self.device,
                retina_masks=self.retina_masks,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            if not results or results[0].masks is None:
                return []

            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            masks = []
            for i, mask in enumerate(masks_data):
                mask_binary = mask > 0.5
                area = int(np.sum(mask_binary))

                if area < self.min_mask_area:
                    continue

                if boxes is not None and i < len(boxes):
                    box = boxes[i].xyxy[0].cpu().numpy()
                    bbox = [int(box[0]), int(box[1]),
                            int(box[2] - box[0]), int(box[3] - box[1])]
                    conf = float(boxes[i].conf[0]) if boxes[i].conf is not None else 1.0
                else:
                    rows = np.any(mask_binary, axis=1)
                    cols = np.any(mask_binary, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                    conf = 1.0

                masks.append({
                    'segmentation': mask_binary,
                    'area': area,
                    'bbox': bbox,
                    'confidence': conf
                })

            return masks

        except Exception as e:
            logger.error(f"Error generating masks: {e}")
            return []

    def compute_scene_complexity(self, masks: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Compute scene complexity metrics from masks.

        Args:
            masks: List of mask dictionaries from FastSAM
            image_shape: (height, width) of original image

        Returns:
            Dictionary with complexity metrics
        """
        if not masks:
            return {
                'num_segments': 0,
                'avg_segment_size': 0.0,
                'segment_size_std': 0.0,
                'coverage_ratio': 0.0,
                'edge_density': 0.0,
                'avg_confidence': 0.0,
                'complexity_score': 0.0
            }

        height, width = image_shape[:2]
        total_pixels = height * width

        areas = [m['area'] for m in masks]
        confidences = [m.get('confidence', 1.0) for m in masks]

        # Compute edge density from masks
        edge_pixels = 0
        for mask_dict in masks:
            mask = mask_dict['segmentation'].astype(np.uint8)
            edges = cv2.Canny(mask * 255, 100, 200)
            edge_pixels += np.sum(edges > 0)

        # Metrics
        num_segments = len(masks)
        avg_segment_size = np.mean(areas) / total_pixels if areas else 0
        segment_size_std = np.std(areas) / total_pixels if len(areas) > 1 else 0
        coverage_ratio = min(sum(areas), total_pixels) / total_pixels
        edge_density = edge_pixels / total_pixels
        avg_confidence = np.mean(confidences) if confidences else 0

        # Composite complexity score (0-1)
        complexity_score = (
            0.25 * min(num_segments / 50, 1.0) +
            0.20 * segment_size_std * 10 +
            0.20 * edge_density * 10 +
            0.20 * coverage_ratio +
            0.15 * (1 - avg_confidence)  # Lower confidence = harder
        )
        complexity_score = min(max(complexity_score, 0), 1)

        return {
            'num_segments': num_segments,
            'avg_segment_size': float(avg_segment_size),
            'segment_size_std': float(segment_size_std),
            'coverage_ratio': float(coverage_ratio),
            'edge_density': float(edge_density),
            'avg_confidence': float(avg_confidence),
            'complexity_score': float(complexity_score)
        }

    def compute_features_from_path(self, image_path: str) -> Optional[Dict]:
        """
        Compute all FastSAM-based features for an image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with all features or None on error
        """
        try:
            # Read image for shape
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None

            masks = self.extract_masks(image_path)
            complexity = self.compute_scene_complexity(masks, image.shape)

            return {
                'num_masks': len(masks),
                'complexity': complexity,
                'complexity_score': complexity['complexity_score']
            }

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

        iterator = tqdm(image_paths, desc="Computing FastSAM complexity") if show_progress else image_paths

        for path in iterator:
            result = self.compute_features_from_path(path)
            if result is not None:
                scores.append(result['complexity_score'])
                valid_paths.append(path)

        logger.info(f"Computed complexity for {len(valid_paths)}/{len(image_paths)} images")
        return scores, valid_paths

    def compute_complexity_without_fastsam(self, image: np.ndarray) -> Dict:
        """
        Fallback: Compute approximate complexity without FastSAM model.
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

        # Blob detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        min_area = 100
        valid_blobs = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
        num_segments = len(valid_blobs)

        if valid_blobs:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_blobs]
            avg_segment_size = np.mean(areas) / (height * width)
            segment_size_std = np.std(areas) / (height * width)
        else:
            avg_segment_size = 0
            segment_size_std = 0

        # Texture complexity
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var() / 10000

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
    print("Testing FastSAMExtractor...")

    extractor = FastSAMExtractor()

    # Test image
    test_image_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Datasets/datasets_20250606.tar/datasets_20250606/datasets/DETR_augmented_dataset_20250218/images/test01_frame_0000868.jpg"

    if Path(test_image_path).exists():
        print(f"\nTesting with: {test_image_path}")
        result = extractor.compute_features_from_path(test_image_path)
        if result:
            print(f"Number of masks: {result['num_masks']}")
            print(f"Complexity score: {result['complexity_score']:.3f}")
            print(f"Full metrics: {result['complexity']}")
    else:
        print(f"Test image not found: {test_image_path}")

    # Cleanup
    extractor.unload()
    print("\nFastSAMExtractor test completed!")
