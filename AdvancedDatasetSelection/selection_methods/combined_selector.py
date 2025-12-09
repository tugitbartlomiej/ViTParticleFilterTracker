"""
Combined Selector - Hybrid selection using EL2N + k-Center + Fourier + SAM.

Implements 4-step selection strategy:
1. Fourier pre-filtering (redundancy removal)
2. Feature extraction (DINO + SAM)
3. k-Center Greedy (diversity)
4. EL2N ranking (difficulty)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

from ..feature_extractors.fourier_analyzer import FourierAnalyzer
from ..feature_extractors.dino_extractor import DINOExtractor
from ..feature_extractors.sam_extractor import SAMExtractor
from .el2n_scorer import EL2NScorer, compute_proxy_el2n_from_features
from .k_center_greedy import KCenterGreedy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedSelector:
    """Hybrid selection combining multiple methods."""

    def __init__(self,
                 fourier_analyzer: Optional[FourierAnalyzer] = None,
                 dino_extractor: Optional[DINOExtractor] = None,
                 sam_extractor: Optional[SAMExtractor] = None,
                 el2n_scorer: Optional[EL2NScorer] = None,
                 k_center: Optional[KCenterGreedy] = None,
                 weights: Optional[Dict[str, float]] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize Combined Selector.

        Args:
            fourier_analyzer: Fourier analyzer instance
            dino_extractor: DINO feature extractor
            sam_extractor: SAM feature extractor
            el2n_scorer: EL2N scorer
            k_center: k-Center greedy selector
            weights: Weights for combining scores
            cache_dir: Directory for caching features
        """
        self.fourier = fourier_analyzer or FourierAnalyzer()
        self.dino = dino_extractor or DINOExtractor()
        self.sam = sam_extractor or SAMExtractor()
        self.el2n = el2n_scorer or EL2NScorer()
        self.k_center = k_center or KCenterGreedy()

        self.weights = weights or {
            'fourier_uniqueness': 0.15,
            'dino_diversity': 0.35,
            'sam_complexity': 0.20,
            'el2n_difficulty': 0.30
        }

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature cache
        self.fourier_features = None
        self.dino_features = None
        self.sam_scores = None
        self.el2n_scores = None
        self.valid_paths = None

    def _load_cache(self, name: str) -> Optional[np.ndarray]:
        """Load cached features."""
        if self.cache_dir is None:
            return None

        cache_path = self.cache_dir / f"{name}.pkl"
        if cache_path.exists():
            logger.info(f"Loading cached {name}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, name: str, data):
        """Save features to cache."""
        if self.cache_dir is None:
            return

        cache_path = self.cache_dir / f"{name}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved {name} to cache")

    def extract_all_features(self,
                             image_paths: List[str],
                             labels: Optional[List[int]] = None,
                             use_cache: bool = True,
                             show_progress: bool = True) -> Dict:
        """
        Extract all features for images.

        Args:
            image_paths: List of image paths
            labels: Optional labels for EL2N (default: all same label)
            use_cache: Whether to use cached features
            show_progress: Show progress bars

        Returns:
            Dictionary with all extracted features
        """
        logger.info(f"Extracting features for {len(image_paths)} images")

        # Fourier features
        if use_cache:
            cached = self._load_cache('fourier_features')
            if cached is not None:
                self.fourier_features, self.valid_paths = cached
            else:
                self.fourier_features, self.valid_paths = self.fourier.compute_features_batch(
                    image_paths, show_progress=show_progress
                )
                self._save_cache('fourier_features', (self.fourier_features, self.valid_paths))
        else:
            self.fourier_features, self.valid_paths = self.fourier.compute_features_batch(
                image_paths, show_progress=show_progress
            )

        # Use only valid paths from here
        valid_image_paths = self.valid_paths

        # SAM complexity scores (using proxy without actual SAM model)
        if use_cache:
            cached = self._load_cache('sam_scores')
            if cached is not None:
                self.sam_scores = cached
            else:
                self.sam_scores = self._compute_sam_proxy_scores(valid_image_paths, show_progress)
                self._save_cache('sam_scores', self.sam_scores)
        else:
            self.sam_scores = self._compute_sam_proxy_scores(valid_image_paths, show_progress)

        # DINO features (if model is available, otherwise use placeholder)
        try:
            if use_cache:
                cached = self._load_cache('dino_features')
                if cached is not None:
                    self.dino_features = cached
                else:
                    self.dino_features, _ = self.dino.compute_features_batch(
                        valid_image_paths, show_progress=show_progress
                    )
                    self._save_cache('dino_features', self.dino_features)
            else:
                self.dino_features, _ = self.dino.compute_features_batch(
                    valid_image_paths, show_progress=show_progress
                )
        except Exception as e:
            logger.warning(f"DINO extraction failed: {e}. Using Fourier features as proxy.")
            self.dino_features = self.fourier_features

        # EL2N scores (using features directly)
        if labels is None:
            labels = [0] * len(valid_image_paths)  # All same label for unsupervised

        if len(labels) != len(valid_image_paths):
            labels = labels[:len(valid_image_paths)]

        self.el2n_scores = compute_proxy_el2n_from_features(
            self.dino_features, np.array(labels)
        )

        return {
            'fourier_features': self.fourier_features,
            'dino_features': self.dino_features,
            'sam_scores': self.sam_scores,
            'el2n_scores': self.el2n_scores,
            'valid_paths': self.valid_paths
        }

    def _compute_sam_proxy_scores(self,
                                   image_paths: List[str],
                                   show_progress: bool = True) -> np.ndarray:
        """Compute SAM-like complexity scores without actual SAM model."""
        import cv2

        scores = []
        iterator = tqdm(image_paths, desc="Computing SAM proxy scores") if show_progress else image_paths

        for path in iterator:
            try:
                image = cv2.imread(path)
                if image is None:
                    scores.append(0.5)
                    continue

                complexity = self.sam.compute_complexity_without_sam(image)
                scores.append(complexity['complexity_score'])
            except Exception as e:
                scores.append(0.5)

        return np.array(scores)

    def step1_fourier_prefilter(self,
                                 threshold: float = 0.85) -> Tuple[List[int], List[str]]:
        """
        Step 1: Fourier pre-filtering to remove redundant samples.

        Args:
            threshold: Similarity threshold for redundancy

        Returns:
            Tuple of (selected indices, selected paths)
        """
        if self.fourier_features is None:
            raise ValueError("Features not extracted. Call extract_all_features first.")

        logger.info(f"Step 1: Fourier pre-filtering (threshold={threshold})")

        selected_indices = self.fourier.filter_redundant(
            self.fourier_features, threshold=threshold
        )

        selected_paths = [self.valid_paths[i] for i in selected_indices]

        logger.info(f"Fourier pre-filter: {len(self.valid_paths)} -> {len(selected_indices)} samples")
        return selected_indices, selected_paths

    def step2_combine_features(self, indices: List[int]) -> np.ndarray:
        """
        Step 2: Combine DINO and SAM features.

        Args:
            indices: Indices of samples to include

        Returns:
            Combined feature matrix
        """
        logger.info("Step 2: Combining DINO and SAM features")

        # Get subset of features
        dino_subset = self.dino_features[indices]
        sam_subset = self.sam_scores[indices].reshape(-1, 1)

        # Normalize DINO features
        dino_norm = dino_subset / (np.linalg.norm(dino_subset, axis=1, keepdims=True) + 1e-10)

        # Normalize SAM scores to [0, 1]
        sam_norm = (sam_subset - sam_subset.min()) / (sam_subset.max() - sam_subset.min() + 1e-10)

        # Weight and combine
        dino_weight = self.weights['dino_diversity']
        sam_weight = self.weights['sam_complexity']
        total = dino_weight + sam_weight

        # Combined features: scaled DINO + SAM as extra dimension
        combined = np.hstack([
            dino_norm * (dino_weight / total),
            sam_norm * (sam_weight / total) * np.sqrt(dino_norm.shape[1])  # Scale SAM to similar magnitude
        ])

        return combined

    def step3_k_center_select(self,
                               combined_features: np.ndarray,
                               original_indices: List[int],
                               target_size: int,
                               oversampling: float = 2.0) -> List[int]:
        """
        Step 3: k-Center Greedy selection for diversity.

        Args:
            combined_features: Combined feature matrix
            original_indices: Original indices in full dataset
            target_size: Target number of samples
            oversampling: Oversampling factor

        Returns:
            Selected indices (in original dataset indexing)
        """
        k = int(target_size * oversampling)
        k = min(k, len(combined_features))

        logger.info(f"Step 3: k-Center Greedy (selecting {k} from {len(combined_features)})")

        local_selected = self.k_center.select(
            combined_features, k=k, show_progress=True
        )

        # Map back to original indices
        selected_indices = [original_indices[i] for i in local_selected]

        return selected_indices

    def step4_el2n_rank(self,
                        indices: List[int],
                        target_size: int,
                        keep_hard: bool = True) -> List[int]:
        """
        Step 4: EL2N ranking to select final samples.

        Args:
            indices: Indices from step 3
            target_size: Final target size
            keep_hard: Keep hardest samples (True) or easiest (False)

        Returns:
            Final selected indices
        """
        logger.info(f"Step 4: EL2N ranking (selecting {target_size} from {len(indices)})")

        # Get EL2N scores for selected indices
        scores = self.el2n_scores[indices]

        # Sort by difficulty
        sorted_order = np.argsort(scores)
        if keep_hard:
            sorted_order = sorted_order[::-1]  # Highest first

        # Select top target_size
        final_local = sorted_order[:target_size]
        final_indices = [indices[i] for i in final_local]

        return final_indices

    def select_optimal_subset(self,
                               image_paths: List[str],
                               target_size: int,
                               labels: Optional[List[int]] = None,
                               fourier_threshold: float = 0.85,
                               oversampling: float = 2.0,
                               keep_hard: bool = True,
                               use_cache: bool = True) -> Tuple[List[int], List[str], Dict]:
        """
        Run complete selection pipeline.

        Args:
            image_paths: List of image paths
            target_size: Final number of samples to select
            labels: Optional labels for EL2N
            fourier_threshold: Threshold for Fourier pre-filtering
            oversampling: Oversampling factor for k-Center
            keep_hard: Keep hard samples in EL2N ranking
            use_cache: Use feature cache

        Returns:
            Tuple of (selected indices, selected paths, statistics)
        """
        logger.info(f"Starting combined selection: {len(image_paths)} -> {target_size}")

        # Extract all features
        self.extract_all_features(image_paths, labels, use_cache=use_cache)

        # Step 1: Fourier pre-filtering
        step1_indices, step1_paths = self.step1_fourier_prefilter(threshold=fourier_threshold)

        # Step 2: Combine features
        combined_features = self.step2_combine_features(step1_indices)

        # Step 3: k-Center selection
        step3_indices = self.step3_k_center_select(
            combined_features, step1_indices, target_size, oversampling
        )

        # Step 4: EL2N ranking
        final_indices = self.step4_el2n_rank(step3_indices, target_size, keep_hard)

        # Get final paths
        final_paths = [self.valid_paths[i] for i in final_indices]

        # Compute statistics
        stats = {
            'original_count': len(image_paths),
            'valid_count': len(self.valid_paths),
            'after_fourier': len(step1_indices),
            'after_k_center': len(step3_indices),
            'final_count': len(final_indices),
            'reduction_ratio': len(final_indices) / len(image_paths),
            'el2n_mean': float(np.mean(self.el2n_scores[final_indices])),
            'el2n_std': float(np.std(self.el2n_scores[final_indices])),
            'sam_mean': float(np.mean(self.sam_scores[final_indices])),
        }

        logger.info(f"Selection complete: {stats['original_count']} -> {stats['final_count']} samples")

        return final_indices, final_paths, stats

    def compute_combined_score(self, index: int) -> Dict:
        """
        Compute combined score for a single sample.

        Args:
            index: Sample index

        Returns:
            Dictionary with individual and combined scores
        """
        if self.fourier_features is None:
            raise ValueError("Features not extracted")

        # Fourier uniqueness (inverse of max similarity to others)
        fourier_sim = self.fourier.compute_similarity_matrix(self.fourier_features)
        max_sim = np.max(np.delete(fourier_sim[index], index))
        fourier_uniqueness = 1 - max_sim

        # DINO diversity (distance to mean)
        dino_mean = np.mean(self.dino_features, axis=0)
        dino_dist = np.linalg.norm(self.dino_features[index] - dino_mean)
        dino_diversity = dino_dist / (np.max(np.linalg.norm(self.dino_features - dino_mean, axis=1)) + 1e-10)

        # SAM complexity
        sam_complexity = self.sam_scores[index]

        # EL2N difficulty
        el2n_difficulty = self.el2n_scores[index]
        el2n_difficulty = (el2n_difficulty - self.el2n_scores.min()) / (
            self.el2n_scores.max() - self.el2n_scores.min() + 1e-10
        )

        # Combined score
        final_score = (
            self.weights['fourier_uniqueness'] * fourier_uniqueness +
            self.weights['dino_diversity'] * dino_diversity +
            self.weights['sam_complexity'] * sam_complexity +
            self.weights['el2n_difficulty'] * el2n_difficulty
        )

        return {
            'fourier_uniqueness': float(fourier_uniqueness),
            'dino_diversity': float(dino_diversity),
            'sam_complexity': float(sam_complexity),
            'el2n_difficulty': float(el2n_difficulty),
            'final_score': float(final_score)
        }


if __name__ == "__main__":
    # Test combined selector
    print("Testing CombinedSelector...")

    # Create with default components
    selector = CombinedSelector()

    # Test with synthetic data
    n_samples = 100
    features = np.random.randn(n_samples, 768)
    labels = np.random.randint(0, 2, n_samples)

    # Manually set features for testing
    selector.fourier_features = np.random.randn(n_samples, 5)
    selector.dino_features = features
    selector.sam_scores = np.random.rand(n_samples)
    selector.el2n_scores = np.random.rand(n_samples)
    selector.valid_paths = [f"img_{i}.jpg" for i in range(n_samples)]

    # Test step by step
    print("\nStep 1: Fourier pre-filter")
    step1_indices, _ = selector.step1_fourier_prefilter(threshold=0.95)
    print(f"  Selected: {len(step1_indices)}")

    print("\nStep 2: Combine features")
    combined = selector.step2_combine_features(step1_indices)
    print(f"  Combined shape: {combined.shape}")

    print("\nStep 3: k-Center selection")
    step3_indices = selector.step3_k_center_select(combined, step1_indices, target_size=20)
    print(f"  Selected: {len(step3_indices)}")

    print("\nStep 4: EL2N ranking")
    final_indices = selector.step4_el2n_rank(step3_indices, target_size=10)
    print(f"  Final: {len(final_indices)}")

    print("\nCombinedSelector test completed!")
