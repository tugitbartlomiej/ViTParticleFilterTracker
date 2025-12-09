"""
Fourier Frequency Analysis for Dataset Selection.

Analyzes images in frequency domain to:
1. Compute frequency band features (low/mid/high energy)
2. Calculate spectral entropy and frequency centroid
3. Identify and filter redundant/similar frames
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FourierAnalyzer:
    """Analyzes images in frequency domain for diversity and redundancy detection."""

    def __init__(self,
                 frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 similarity_threshold: float = 0.85):
        """
        Initialize Fourier Analyzer.

        Args:
            frequency_bands: Dict mapping band names to (start, end) fractions.
                            Default: {'low': (0, 0.1), 'mid': (0.1, 0.5), 'high': (0.5, 1.0)}
            similarity_threshold: Threshold for considering images similar (0-1)
        """
        self.frequency_bands = frequency_bands or {
            'low': (0.0, 0.1),
            'mid': (0.1, 0.5),
            'high': (0.5, 1.0)
        }
        self.similarity_threshold = similarity_threshold
        self.feature_dim = 5  # low, mid, high energy + entropy + centroid

    def compute_frequency_features(self, image: np.ndarray) -> np.ndarray:
        """
        Compute frequency domain features for an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Feature vector: [low_energy, mid_energy, high_energy, spectral_entropy, frequency_centroid]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize to float
        gray = gray.astype(np.float32) / 255.0

        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Magnitude spectrum
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)  # Log scale for better distribution

        # Create radial frequency mask
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        max_radius = np.sqrt(center_row**2 + center_col**2)

        y, x = np.ogrid[:rows, :cols]
        radius = np.sqrt((y - center_row)**2 + (x - center_col)**2)
        normalized_radius = radius / max_radius

        # Compute band energies
        band_energies = []
        for band_name, (start, end) in self.frequency_bands.items():
            mask = (normalized_radius >= start) & (normalized_radius < end)
            band_energy = np.sum(magnitude[mask])
            band_energies.append(band_energy)

        # Normalize energies
        total_energy = sum(band_energies) + 1e-10
        band_energies = [e / total_energy for e in band_energies]

        # Spectral entropy
        magnitude_norm = magnitude / (np.sum(magnitude) + 1e-10)
        magnitude_norm = np.clip(magnitude_norm, 1e-10, 1.0)
        spectral_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm))
        # Normalize entropy
        max_entropy = np.log2(rows * cols)
        spectral_entropy = spectral_entropy / max_entropy

        # Frequency centroid (weighted average of frequencies)
        frequency_centroid = np.sum(normalized_radius * magnitude) / (np.sum(magnitude) + 1e-10)

        features = np.array(band_energies + [spectral_entropy, frequency_centroid], dtype=np.float32)
        return features

    def compute_features_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """Compute features from image path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
            return self.compute_frequency_features(image)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def compute_features_batch(self,
                               image_paths: List[str],
                               num_workers: int = 4,
                               show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Compute features for a batch of images.

        Args:
            image_paths: List of image paths
            num_workers: Number of parallel workers
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (features array, valid paths)
        """
        features_list = []
        valid_paths = []

        iterator = tqdm(image_paths, desc="Extracting Fourier features") if show_progress else image_paths

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {executor.submit(self.compute_features_from_path, p): p
                             for p in image_paths}

            for future in tqdm(as_completed(future_to_path), total=len(image_paths),
                              desc="Extracting Fourier features", disable=not show_progress):
                path = future_to_path[future]
                try:
                    features = future.result()
                    if features is not None:
                        features_list.append(features)
                        valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")

        if not features_list:
            return np.array([]), []

        features_array = np.vstack(features_list)
        logger.info(f"Extracted features for {len(valid_paths)}/{len(image_paths)} images")
        return features_array, valid_paths

    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between feature vectors.

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

    def filter_redundant(self,
                         features: np.ndarray,
                         threshold: Optional[float] = None) -> List[int]:
        """
        Greedy filtering to remove redundant images based on Fourier similarity.

        Args:
            features: Feature matrix (N x D)
            threshold: Similarity threshold (default: self.similarity_threshold)

        Returns:
            List of indices to keep
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if len(features) == 0:
            return []

        n_samples = len(features)
        logger.info(f"Filtering {n_samples} samples with threshold {threshold}")

        # Compute similarity matrix
        similarity = self.compute_similarity_matrix(features)

        # Greedy selection
        selected = []
        remaining = set(range(n_samples))

        while remaining:
            # Pick first remaining sample
            current = min(remaining)
            selected.append(current)
            remaining.remove(current)

            # Remove all similar samples
            to_remove = set()
            for idx in remaining:
                if similarity[current, idx] > threshold:
                    to_remove.add(idx)

            remaining -= to_remove

        logger.info(f"Kept {len(selected)}/{n_samples} samples ({len(selected)/n_samples*100:.1f}%)")
        return selected

    def filter_redundant_streaming(self,
                                   features: np.ndarray,
                                   threshold: Optional[float] = None,
                                   batch_size: int = 1000) -> List[int]:
        """
        Memory-efficient redundancy filtering for large datasets.

        Args:
            features: Feature matrix (N x D)
            threshold: Similarity threshold
            batch_size: Process similarity in batches

        Returns:
            List of indices to keep
        """
        if threshold is None:
            threshold = self.similarity_threshold

        n_samples = len(features)
        if n_samples == 0:
            return []

        # Normalize features once
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        features_norm = features / norms

        selected = []
        is_removed = np.zeros(n_samples, dtype=bool)

        for i in tqdm(range(n_samples), desc="Filtering redundant samples"):
            if is_removed[i]:
                continue

            selected.append(i)

            # Compute similarity to remaining samples
            remaining_indices = np.where(~is_removed)[0]
            remaining_indices = remaining_indices[remaining_indices > i]

            if len(remaining_indices) == 0:
                continue

            # Batch similarity computation
            current_feat = features_norm[i:i+1]
            similarities = np.dot(features_norm[remaining_indices], current_feat.T).flatten()

            # Mark similar samples for removal
            similar_mask = similarities > threshold
            is_removed[remaining_indices[similar_mask]] = True

        logger.info(f"Streaming filter: kept {len(selected)}/{n_samples} samples")
        return selected

    def get_feature_names(self) -> List[str]:
        """Get names of computed features."""
        band_names = [f"{band}_band_energy" for band in self.frequency_bands.keys()]
        return band_names + ['spectral_entropy', 'frequency_centroid']

    def analyze_diversity(self, features: np.ndarray) -> Dict:
        """
        Analyze diversity of the dataset based on Fourier features.

        Args:
            features: Feature matrix (N x D)

        Returns:
            Dictionary with diversity metrics
        """
        if len(features) == 0:
            return {}

        similarity = self.compute_similarity_matrix(features)

        # Remove diagonal
        np.fill_diagonal(similarity, 0)

        return {
            'mean_similarity': float(np.mean(similarity)),
            'max_similarity': float(np.max(similarity)),
            'min_similarity': float(np.min(similarity[similarity > 0])) if np.any(similarity > 0) else 0,
            'std_similarity': float(np.std(similarity)),
            'highly_similar_pairs': int(np.sum(similarity > self.similarity_threshold) / 2),
            'feature_means': features.mean(axis=0).tolist(),
            'feature_stds': features.std(axis=0).tolist(),
            'feature_names': self.get_feature_names()
        }


if __name__ == "__main__":
    # Test the Fourier analyzer
    import sys

    print("Testing FourierAnalyzer...")

    analyzer = FourierAnalyzer()

    # Create a test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test feature extraction
    features = analyzer.compute_frequency_features(test_image)
    print(f"Feature shape: {features.shape}")
    print(f"Feature names: {analyzer.get_feature_names()}")
    print(f"Features: {features}")

    # Test with multiple random images
    n_test = 20
    test_features = np.array([
        analyzer.compute_frequency_features(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ) for _ in range(n_test)
    ])

    print(f"\nBatch features shape: {test_features.shape}")

    # Test similarity
    sim_matrix = analyzer.compute_similarity_matrix(test_features)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Mean similarity: {np.mean(sim_matrix):.3f}")

    # Test filtering
    selected = analyzer.filter_redundant(test_features, threshold=0.95)
    print(f"Selected {len(selected)}/{n_test} samples")

    # Test diversity analysis
    diversity = analyzer.analyze_diversity(test_features)
    print(f"\nDiversity analysis:")
    for k, v in diversity.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    print("\nFourierAnalyzer test completed successfully!")
