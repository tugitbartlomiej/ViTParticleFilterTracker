"""
DINO/DINOv2/DINOv3 Feature Extractor for Dataset Selection.

Supports:
- DINOv1 (torch.hub facebookresearch/dino)
- DINOv2 (torch.hub facebookresearch/dinov2)
- DINOv3 (torch.hub facebookresearch/dinov3 OR local HuggingFace model)

Extracts semantic features:
- CLS token features (384/768/1024-dim depending on model)
- Semantic similarity computation
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging
import os
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DINOExtractor:
    """Extracts semantic features using DINO/DINOv2/DINOv3 models."""

    # Default cache directory for models
    DEFAULT_CACHE_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/External/Models"

    # Available models and their feature dimensions (for torch.hub)
    MODELS = {
        # DINOv1 (patch 16 and 8)
        'dino_vits16': {'repo': 'facebookresearch/dino:main', 'dim': 384},
        'dino_vits8': {'repo': 'facebookresearch/dino:main', 'dim': 384},
        'dino_vitb16': {'repo': 'facebookresearch/dino:main', 'dim': 768},
        'dino_vitb8': {'repo': 'facebookresearch/dino:main', 'dim': 768},
        # DINOv2 (official, patch 14)
        'dinov2_vits14': {'repo': 'facebookresearch/dinov2:main', 'dim': 384},
        'dinov2_vitb14': {'repo': 'facebookresearch/dinov2:main', 'dim': 768},
        'dinov2_vitl14': {'repo': 'facebookresearch/dinov2:main', 'dim': 1024},
        'dinov2_vitg14': {'repo': 'facebookresearch/dinov2:main', 'dim': 1536},
        # DINOv3 (official, patch 16)
        'dinov3_vits16': {'repo': 'facebookresearch/dinov3:main', 'dim': 384},
        'dinov3_vitb16': {'repo': 'facebookresearch/dinov3:main', 'dim': 768},
        'dinov3_vitl16': {'repo': 'facebookresearch/dinov3:main', 'dim': 1024},
        'dinov3_vith16': {'repo': 'facebookresearch/dinov3:main', 'dim': 1280},
    }

    def __init__(self,
                 model_name: str = "dinov2_vitl14",
                 device: str = "cuda",
                 image_size: int = 224,
                 cache_dir: str = None,
                 model_path: str = None):
        """
        Initialize DINO/DINOv2/DINOv3 Extractor.

        Args:
            model_name: Model name for torch.hub (e.g., 'dinov3_vitl16', 'dinov2_vitl14')
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size
            cache_dir: Directory for caching downloaded models (torch.hub)
            model_path: Path to local HuggingFace model directory (overrides torch.hub)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.model_path = model_path

        # If local model path provided, detect feature dim from config
        if model_path and os.path.isdir(model_path):
            self.use_local_model = True
            self.feature_dim = self._detect_feature_dim_from_config(model_path)
            logger.info(f"Using local HuggingFace model from {model_path}")
            logger.info(f"Detected feature_dim={self.feature_dim}")
        else:
            self.use_local_model = False
            # Get model info from MODELS dict
            if model_name not in self.MODELS:
                logger.warning(f"Unknown model {model_name}, defaulting to dinov2_vitl14")
                model_name = 'dinov2_vitl14'
                self.model_name = model_name
            self.model_info = self.MODELS[model_name]
            self.feature_dim = self.model_info['dim']

        self.model = None
        self.processor = None
        self._initialized = False

        # Image preprocessing (same for all DINO models)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _detect_feature_dim_from_config(self, model_path: str) -> int:
        """Detect feature dimension from HuggingFace config.json."""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            hidden_size = config.get('hidden_size', 384)
            return hidden_size
        return 384  # Default to ViT-S

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
        logger.info("DINO model unloaded from memory")

    def _initialize_model(self):
        """Lazy initialization of model."""
        if self._initialized:
            return

        if self.use_local_model:
            self._initialize_from_huggingface()
        else:
            self._initialize_from_torch_hub()

    def _initialize_from_huggingface(self):
        """Load model from local HuggingFace directory."""
        try:
            from transformers import AutoModel, AutoImageProcessor

            logger.info(f"Loading model from local path: {self.model_path}")

            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()

            # Try to load processor, fall back to manual transform
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_path)
                logger.info("Loaded HuggingFace image processor")
            except Exception:
                self.processor = None
                logger.info("Using manual image preprocessing")

            self._initialized = True
            logger.info(f"Model loaded successfully (feature_dim={self.feature_dim})")

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def _initialize_from_torch_hub(self):
        """Load model from torch.hub."""
        # Set cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.hub.set_dir(self.cache_dir)

        repo = self.model_info['repo']
        logger.info(f"Loading {self.model_name} from {repo}")
        logger.info(f"Model cache directory: {self.cache_dir}")

        try:
            self.model = torch.hub.load(repo, self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"Model loaded successfully (feature_dim={self.feature_dim})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_image(self, image: Union[np.ndarray, str, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Use HuggingFace processor if available
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values']

        return self.transform(image).unsqueeze(0)

    def extract_cls_features(self, image: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """
        Extract CLS token features from image.

        Args:
            image: Input image

        Returns:
            Feature vector (feature_dim,)
        """
        self._initialize_model()

        tensor = self.preprocess_image(image).to(self.device)

        with torch.no_grad():
            if self.use_local_model:
                # HuggingFace model returns BaseModelOutput
                outputs = self.model(tensor)
                # Get CLS token (first token of last_hidden_state)
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    features = outputs[0][:, 0, :]
            else:
                # torch.hub model returns tensor directly
                features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def compute_features_batch(self,
                               image_paths: List[str],
                               batch_size: int = 8,
                               show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for a batch of images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (features array, valid paths)
        """
        self._initialize_model()

        features_list = []
        valid_paths = []

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting {self.model_name} features",
                           total=len(image_paths) // batch_size + 1)

        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Could not process {path}: {e}")

            if batch_tensors:
                batch = torch.cat(batch_tensors, dim=0).to(self.device)

                with torch.no_grad():
                    if self.use_local_model:
                        outputs = self.model(batch)
                        if hasattr(outputs, 'last_hidden_state'):
                            batch_features = outputs.last_hidden_state[:, 0, :]
                        elif hasattr(outputs, 'pooler_output'):
                            batch_features = outputs.pooler_output
                        else:
                            batch_features = outputs[0][:, 0, :]
                    else:
                        batch_features = self.model(batch)

                features_list.append(batch_features.cpu().numpy())

        if not features_list:
            return np.array([]), []

        features_array = np.vstack(features_list)
        logger.info(f"Extracted features for {len(valid_paths)}/{len(image_paths)} images")
        return features_array, valid_paths

    def compute_semantic_similarity(self,
                                    features1: np.ndarray,
                                    features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors."""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(features1, features2) / (norm1 * norm2))

    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        features_norm = features / norms
        similarity = np.dot(features_norm, features_norm.T)
        return similarity


if __name__ == "__main__":
    print("Testing DINOExtractor...")
    print(f"Available torch.hub models: {list(DINOExtractor.MODELS.keys())}")

    # Test with local HuggingFace model
    local_model_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/External/Models/dinov3-vitl16-pretrain-lvd1689m"

    if os.path.isdir(local_model_path):
        print(f"\nTesting with local model: {local_model_path}")
        extractor = DINOExtractor(
            model_path=local_model_path,
            device='cuda'
        )
        print(f"Feature dim: {extractor.feature_dim}")

        # Test with a few images
        from glob import glob
        test_dir = "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
        test_images = glob(f"{test_dir}/*.jpg")[:3]

        if test_images:
            print(f"\nTesting on {len(test_images)} images...")
            features, valid = extractor.compute_features_batch(test_images, show_progress=False)
            print(f"Features shape: {features.shape}")
        else:
            print("No test images found")
    else:
        print(f"Local model not found at {local_model_path}")

    print("\nDINOExtractor test completed!")
