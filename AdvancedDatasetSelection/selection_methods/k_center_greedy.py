"""
k-Center Greedy Selection for Dataset Diversity.

Selects a diverse subset of samples by maximizing minimum distance
between selected and unselected points.

Reference: "Active Learning for Convolutional Neural Networks: A Core-Set Approach" (Sener & Savarese, 2018)
"""

import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KCenterGreedy:
    """k-Center Greedy selection for diverse subset selection."""

    def __init__(self, feature_dim: int = 768, distance_metric: str = "euclidean"):
        """
        Initialize k-Center Greedy selector.

        Args:
            feature_dim: Expected feature dimension
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        self.feature_dim = feature_dim
        self.distance_metric = distance_metric

    def compute_distances(self,
                          features: np.ndarray,
                          center_indices: List[int]) -> np.ndarray:
        """
        Compute minimum distances from each point to the nearest center.

        Args:
            features: Feature matrix (N x D)
            center_indices: Indices of current centers

        Returns:
            Array of minimum distances (N,)
        """
        if not center_indices:
            return np.full(len(features), np.inf)

        centers = features[center_indices]

        if self.distance_metric == "cosine":
            # Normalize features and centers
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
            centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
            # Cosine distance = 1 - cosine_similarity
            similarities = np.dot(features_norm, centers_norm.T)
            distances = 1 - similarities
        else:
            # Euclidean distance
            # (N, D), (K, D) -> (N, K)
            sq_distances = (
                np.sum(features**2, axis=1, keepdims=True) +
                np.sum(centers**2, axis=1) -
                2 * np.dot(features, centers.T)
            )
            # Clamp to avoid numerical issues with sqrt
            sq_distances = np.maximum(sq_distances, 0)
            distances = np.sqrt(sq_distances)

        # Minimum distance to any center
        min_distances = np.min(distances, axis=1)
        return min_distances

    def select(self,
               features: np.ndarray,
               k: int,
               seed: Optional[int] = None,
               show_progress: bool = True) -> List[int]:
        """
        Select k diverse samples using k-Center Greedy algorithm.

        Algorithm:
        1. Start with a random point as first center
        2. Repeat k-1 times:
           a. Compute min distance from each point to nearest center
           b. Select point with maximum min distance as new center
        3. Return indices of k selected centers

        Args:
            features: Feature matrix (N x D)
            k: Number of samples to select
            seed: Random seed for initial selection
            show_progress: Whether to show progress bar

        Returns:
            List of k selected indices
        """
        n_samples = len(features)

        if k >= n_samples:
            logger.warning(f"k ({k}) >= n_samples ({n_samples}), returning all indices")
            return list(range(n_samples))

        if seed is not None:
            np.random.seed(seed)

        # Start with random center
        selected = [np.random.randint(n_samples)]
        remaining = set(range(n_samples)) - set(selected)

        # Initialize distances
        min_distances = self.compute_distances(features, selected)

        iterator = range(k - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="k-Center selection")

        for _ in iterator:
            # Find point with maximum minimum distance
            remaining_list = list(remaining)
            remaining_distances = min_distances[remaining_list]
            max_idx = remaining_list[np.argmax(remaining_distances)]

            selected.append(max_idx)
            remaining.remove(max_idx)

            if not remaining:
                break

            # Update distances (only need to check distance to new center)
            new_center = features[max_idx:max_idx+1]

            if self.distance_metric == "cosine":
                features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
                center_norm = new_center / (np.linalg.norm(new_center) + 1e-10)
                new_distances = 1 - np.dot(features_norm, center_norm.T).flatten()
            else:
                new_distances = np.sqrt(
                    np.sum((features - new_center)**2, axis=1)
                )

            min_distances = np.minimum(min_distances, new_distances)

        logger.info(f"Selected {len(selected)} samples with k-Center Greedy")
        return selected

    def select_with_diversity_guarantee(self,
                                         features: np.ndarray,
                                         k: int,
                                         min_distance: float = 0.1,
                                         seed: Optional[int] = None,
                                         show_progress: bool = True) -> List[int]:
        """
        Select samples with guaranteed minimum distance between them.

        Args:
            features: Feature matrix (N x D)
            k: Maximum number of samples to select
            min_distance: Minimum distance between selected samples
            seed: Random seed
            show_progress: Whether to show progress

        Returns:
            List of selected indices (may be < k if min_distance constraint is tight)
        """
        n_samples = len(features)

        if seed is not None:
            np.random.seed(seed)

        # Start with random center
        selected = [np.random.randint(n_samples)]
        remaining = set(range(n_samples)) - set(selected)

        min_distances = self.compute_distances(features, selected)

        iterator = range(k - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="k-Center with diversity guarantee")

        for _ in iterator:
            # Find points that satisfy minimum distance
            remaining_list = list(remaining)
            remaining_distances = min_distances[remaining_list]

            # Filter by minimum distance
            valid_mask = remaining_distances >= min_distance
            if not np.any(valid_mask):
                logger.info(f"No more samples satisfy min_distance={min_distance}")
                break

            # Select point with maximum distance among valid points
            valid_indices = np.array(remaining_list)[valid_mask]
            valid_distances = remaining_distances[valid_mask]
            max_idx = valid_indices[np.argmax(valid_distances)]

            selected.append(max_idx)
            remaining.remove(max_idx)

            if not remaining:
                break

            # Update distances
            new_center = features[max_idx:max_idx+1]

            if self.distance_metric == "cosine":
                features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
                center_norm = new_center / (np.linalg.norm(new_center) + 1e-10)
                new_distances = 1 - np.dot(features_norm, center_norm.T).flatten()
            else:
                new_distances = np.sqrt(np.sum((features - new_center)**2, axis=1))

            min_distances = np.minimum(min_distances, new_distances)

        logger.info(f"Selected {len(selected)} samples with diversity guarantee")
        return selected

    def compute_coverage_radius(self,
                                features: np.ndarray,
                                selected_indices: List[int]) -> float:
        """
        Compute the coverage radius (max min-distance to centers).

        Args:
            features: Feature matrix
            selected_indices: Indices of selected samples

        Returns:
            Coverage radius
        """
        if not selected_indices:
            return float('inf')

        min_distances = self.compute_distances(features, selected_indices)
        return float(np.max(min_distances))

    def analyze_diversity(self,
                          features: np.ndarray,
                          selected_indices: List[int]) -> dict:
        """
        Analyze diversity of selected subset.

        Args:
            features: Feature matrix
            selected_indices: Indices of selected samples

        Returns:
            Dictionary with diversity metrics
        """
        if not selected_indices:
            return {}

        selected_features = features[selected_indices]

        # Pairwise distances among selected
        n_selected = len(selected_indices)
        pairwise_distances = []

        for i in range(n_selected):
            for j in range(i + 1, n_selected):
                if self.distance_metric == "cosine":
                    fi = selected_features[i] / (np.linalg.norm(selected_features[i]) + 1e-10)
                    fj = selected_features[j] / (np.linalg.norm(selected_features[j]) + 1e-10)
                    dist = 1 - np.dot(fi, fj)
                else:
                    dist = np.linalg.norm(selected_features[i] - selected_features[j])
                pairwise_distances.append(dist)

        pairwise_distances = np.array(pairwise_distances)

        # Coverage radius
        coverage_radius = self.compute_coverage_radius(features, selected_indices)

        return {
            'n_selected': n_selected,
            'coverage_radius': float(coverage_radius),
            'mean_pairwise_distance': float(np.mean(pairwise_distances)),
            'min_pairwise_distance': float(np.min(pairwise_distances)),
            'max_pairwise_distance': float(np.max(pairwise_distances)),
            'std_pairwise_distance': float(np.std(pairwise_distances))
        }


def select_diverse_subset(features: np.ndarray,
                          k: int,
                          method: str = "k_center",
                          **kwargs) -> List[int]:
    """
    Convenience function for diverse subset selection.

    Args:
        features: Feature matrix (N x D)
        k: Number of samples to select
        method: Selection method ('k_center', 'random', 'stratified')
        **kwargs: Additional arguments for the selector

    Returns:
        List of selected indices
    """
    if method == "k_center":
        selector = KCenterGreedy(**kwargs)
        return selector.select(features, k)
    elif method == "random":
        return list(np.random.choice(len(features), k, replace=False))
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test the k-Center Greedy selector
    print("Testing KCenterGreedy...")

    np.random.seed(42)

    # Create synthetic features
    n_samples = 500
    n_features = 768
    features = np.random.randn(n_samples, n_features)

    # Select subset
    selector = KCenterGreedy(feature_dim=n_features, distance_metric="euclidean")
    k = 50

    selected = selector.select(features, k, seed=42, show_progress=True)
    print(f"\nSelected {len(selected)} samples")

    # Analyze diversity
    diversity = selector.analyze_diversity(features, selected)
    print(f"Diversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.4f}")

    # Test with cosine distance
    selector_cosine = KCenterGreedy(distance_metric="cosine")
    selected_cosine = selector_cosine.select(features, k, seed=42, show_progress=False)
    print(f"\nCosine distance selected: {len(selected_cosine)} samples")

    # Test diversity guarantee
    selected_div = selector.select_with_diversity_guarantee(
        features, k=100, min_distance=5.0, seed=42, show_progress=False
    )
    print(f"With diversity guarantee: {len(selected_div)} samples")

    print("\nKCenterGreedy test completed!")
