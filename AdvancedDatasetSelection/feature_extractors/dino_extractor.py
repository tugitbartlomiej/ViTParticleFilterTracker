"""
DINO (Self-Distillation with No Labels) Feature Extractor for Dataset Selection.

Extracts semantic features using DINO ViT:
- CLS token features (768-dim)
- Attention-based features
- Semantic similarity computation
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DINOExtractor:
    """Extracts semantic features using DINO ViT model."""

    def __init__(self,
                 model_name: str = "dino_vitb16",
                 device: str = "cuda",
                 image_size: int = 224):
        """
        Initialize DINO Extractor.

        Args:
            model_name: DINO model name ('dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8')
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size for DINO
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size

        self.model = None
        self._initialized = False

        # Feature dimension based on model
        self.feature_dims = {
            'dino_vits16': 384,
            'dino_vits8': 384,
            'dino_vitb16': 768,
            'dino_vitb8': 768
        }
        self.feature_dim = self.feature_dims.get(model_name, 768)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _initialize_model(self):
        """Lazy initialization of DINO model."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading DINO model: {self.model_name}")
            self.model = torch.hub.load('facebookresearch/dino:main', self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"DINO model loaded successfully (feature_dim={self.feature_dim})")
        except Exception as e:
            logger.error(f"Failed to load DINO model: {e}")
            raise

    def preprocess_image(self, image: Union[np.ndarray, str, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for DINO.

        Args:
            image: Input image (numpy array BGR, file path, or PIL Image)

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Assume BGR format from cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

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
            features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def extract_attention_maps(self, image: Union[np.ndarray, str, Image.Image]) -> Dict:
        """
        Extract attention maps from DINO.

        Args:
            image: Input image

        Returns:
            Dictionary with attention information
        """
        self._initialize_model()

        tensor = self.preprocess_image(image).to(self.device)

        # Get attention from last layer
        with torch.no_grad():
            # Forward pass to get attention
            attentions = self.model.get_last_selfattention(tensor)

        # attentions shape: (1, num_heads, num_patches+1, num_patches+1)
        nh = attentions.shape[1]  # Number of heads

        # Get CLS token attention to patches
        cls_attention = attentions[0, :, 0, 1:]  # (num_heads, num_patches)

        # Reshape to spatial
        patch_size = 16 if '16' in self.model_name else 8
        num_patches_side = self.image_size // patch_size
        cls_attention = cls_attention.reshape(nh, num_patches_side, num_patches_side)

        # Compute attention metrics
        attention_np = cls_attention.cpu().numpy()

        # Entropy per head
        attention_flat = attention_np.reshape(nh, -1)
        attention_probs = attention_flat / (attention_flat.sum(axis=1, keepdims=True) + 1e-10)
        attention_probs = np.clip(attention_probs, 1e-10, 1.0)
        entropy_per_head = -np.sum(attention_probs * np.log2(attention_probs), axis=1)

        # Average entropy
        max_entropy = np.log2(attention_flat.shape[1])
        avg_entropy = np.mean(entropy_per_head) / max_entropy

        # Attention diversity (std across heads)
        attention_diversity = np.std(attention_np.mean(axis=(1, 2)))

        # Spatial coverage (fraction of patches with significant attention)
        threshold = attention_np.mean() + attention_np.std()
        coverage = np.mean(attention_np > threshold)

        return {
            'attention_maps': attention_np,
            'attention_entropy': float(avg_entropy),
            'attention_diversity': float(attention_diversity),
            'spatial_coverage': float(coverage),
            'num_heads': nh
        }

    def extract_attention_features(self, image: Union[np.ndarray, str, Image.Image]) -> Dict:
        """
        Extract attention-based features (lightweight version).

        Args:
            image: Input image

        Returns:
            Dictionary with attention metrics
        """
        try:
            return self.extract_attention_maps(image)
        except Exception as e:
            logger.warning(f"Could not extract attention maps: {e}")
            return {
                'attention_entropy': 0.5,
                'attention_diversity': 0.1,
                'spatial_coverage': 0.5,
                'num_heads': 0
            }

    def compute_features(self, image: Union[np.ndarray, str, Image.Image]) -> Dict:
        """
        Compute all DINO features for an image.

        Args:
            image: Input image

        Returns:
            Dictionary with CLS features and attention metrics
        """
        cls_features = self.extract_cls_features(image)
        attention_info = self.extract_attention_features(image)

        return {
            'cls_features': cls_features,
            'attention_entropy': attention_info['attention_entropy'],
            'attention_diversity': attention_info['attention_diversity'],
            'spatial_coverage': attention_info['spatial_coverage']
        }

    def compute_features_batch(self,
                               image_paths: List[str],
                               batch_size: int = 16,
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

        # Process in batches
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting DINO features",
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
        """
        Compute cosine similarity between feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Cosine similarity (0-1)
        """
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(features1, features2) / (norm1 * norm2))

    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            features: Feature matrix (N x D)

        Returns:
            Similarity matrix (N x N)
        """
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        features_norm = features / norms

        # Cosine similarity
        similarity = np.dot(features_norm, features_norm.T)
        return similarity

    def cluster_features(self,
                        features: np.ndarray,
                        n_clusters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster features using K-means.

        Args:
            features: Feature matrix (N x D)
            n_clusters: Number of clusters

        Returns:
            Tuple of (cluster labels, cluster centers)
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        return labels, centers


if __name__ == "__main__":
    # Test the DINO extractor
    print("Testing DINOExtractor...")

    # Test with dummy data (no actual model loading)
    print("Creating DINOExtractor instance...")
    extractor = DINOExtractor(model_name="dino_vitb16")
    print(f"Feature dimension: {extractor.feature_dim}")

    # Test preprocessing
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = extractor.preprocess_image(test_image)
    print(f"Preprocessed tensor shape: {tensor.shape}")

    # Test similarity computation
    features1 = np.random.randn(768)
    features2 = np.random.randn(768)
    sim = extractor.compute_semantic_similarity(features1, features2)
    print(f"Test similarity: {sim:.3f}")

    # Test similarity matrix
    test_features = np.random.randn(10, 768)
    sim_matrix = extractor.compute_similarity_matrix(test_features)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    print("\nDINOExtractor test completed!")
    print("Note: Full feature extraction requires model download from torch.hub")
