"""
Cluster-Based Coreset Selector - Literature-based approach.

Based on:
- ELFS (ICLR 2025): DINO features + clustering + difficulty scores
- CCS (ICLR 2023): Coverage-centric selection with stratified sampling
- Class-Proportional Coreset Selection: Per-cluster representative selection

Pipeline:
1. Extract ALL features: DINO + Fourier + SAM + EL2N (as feature, not just ranking)
2. Normalize and combine into unified feature space
3. K-Means clustering on combined features
4. Select representative from each cluster (centroid or max EL2N)
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import pickle

from ..feature_extractors.fourier_analyzer import FourierAnalyzer
from ..feature_extractors.dino_extractor import DINOExtractor
from ..feature_extractors.sam_extractor import SAMExtractor
from .detr_el2n_scorer import DETR_EL2N_Scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterBasedSelector:
    """
    Cluster-based coreset selection following literature best practices.

    Key differences from k-Center Greedy:
    - Uses k-means clustering to find natural groups
    - EL2N is a FEATURE (not just final ranking)
    - Guarantees coverage: each cluster has a representative
    - Supports multiple selection strategies per cluster
    """

    def __init__(self,
                 fourier_analyzer: Optional[FourierAnalyzer] = None,
                 dino_extractor: Optional[DINOExtractor] = None,
                 sam_extractor: Optional[SAMExtractor] = None,
                 detr_checkpoint_path: Optional[str] = None,
                 detr_device: str = "cuda",
                 weights: Optional[Dict[str, float]] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize Cluster-Based Selector.

        Args:
            fourier_analyzer: Fourier analyzer instance
            dino_extractor: DINO feature extractor
            sam_extractor: SAM feature extractor
            detr_checkpoint_path: Path to DETR checkpoint for EL2N
            detr_device: Device for DETR model
            weights: Weights for feature combination (for normalization scaling)
            cache_dir: Directory for caching features
        """
        self.fourier = fourier_analyzer or FourierAnalyzer()
        self.dino = dino_extractor or DINOExtractor()
        self.sam = sam_extractor or SAMExtractor()

        # DETR-based EL2N scorer
        self.detr_checkpoint_path = detr_checkpoint_path
        if detr_checkpoint_path:
            logger.info(f"Using DETR-based EL2N scorer: {detr_checkpoint_path}")
            self.detr_el2n = DETR_EL2N_Scorer(
                checkpoint_path=detr_checkpoint_path,
                device=detr_device
            )
        else:
            logger.warning("No DETR checkpoint provided - EL2N will use proxy scores")
            self.detr_el2n = None

        # Feature weights (used for scaling importance, not hard weighting)
        self.weights = weights or {
            'dino': 0.35,
            'fourier': 0.15,
            'sam': 0.20,
            'el2n': 0.30
        }

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature storage
        self.dino_features = None
        self.fourier_features = None
        self.sam_scores = None
        self.el2n_scores = None
        self.combined_features = None
        self.valid_paths = None

        # Clustering results
        self.cluster_labels = None
        self.cluster_centers = None
        self.n_clusters = None

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

    def _compute_sam_scores(self, image_paths: List[str], show_progress: bool = True) -> np.ndarray:
        """Compute SAM3 complexity scores using full model."""
        import cv2
        scores = []
        iterator = tqdm(image_paths, desc="Computing SAM3 complexity") if show_progress else image_paths
        for path in iterator:
            try:
                image = cv2.imread(path)
                if image is None:
                    scores.append(0.5)
                    continue
                # Try full SAM3 first
                result = self.sam.compute_features(image)
                if result and 'complexity_score' in result:
                    scores.append(result['complexity_score'])
                else:
                    # Fallback to proxy if SAM3 fails
                    complexity = self.sam.compute_complexity_without_sam(image)
                    scores.append(complexity['complexity_score'])
            except Exception as e:
                logger.warning(f"SAM3 failed for {path}: {e}, using proxy")
                try:
                    image = cv2.imread(path)
                    complexity = self.sam.compute_complexity_without_sam(image)
                    scores.append(complexity['complexity_score'])
                except:
                    scores.append(0.5)
        return np.array(scores)

    def extract_all_features(self,
                             image_paths: List[str],
                             use_cache: bool = True,
                             show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract ALL features for images including EL2N as a feature.

        Models are loaded and unloaded SEQUENTIALLY to save GPU memory:
        Fourier (CPU) -> DINO (GPU) -> unload -> SAM3 (GPU) -> unload -> DETR (GPU) -> unload

        Args:
            image_paths: List of image paths
            use_cache: Whether to use cached features
            show_progress: Show progress bars

        Returns:
            Dictionary with all extracted features
        """
        logger.info(f"Extracting features for {len(image_paths)} images")
        logger.info("Sequential loading: models will be loaded/unloaded one at a time to save GPU memory")

        # 1. Fourier features (9-dim) - CPU only, no GPU memory needed
        logger.info("\n[1/4] Fourier features (CPU)...")
        cached = self._load_cache('fourier_features') if use_cache else None
        if cached is not None:
            self.fourier_features, self.valid_paths = cached
            logger.info("Loaded Fourier features from cache")
        else:
            self.fourier_features, self.valid_paths = self.fourier.compute_features_batch(
                image_paths, show_progress=show_progress
            )
            self._save_cache('fourier_features', (self.fourier_features, self.valid_paths))

        valid_image_paths = self.valid_paths
        n_valid = len(valid_image_paths)
        logger.info(f"Valid images: {n_valid}/{len(image_paths)}")

        # 2. DINO features (1024-dim for ViT-L) - LOAD -> EXTRACT -> UNLOAD
        logger.info("\n[2/4] DINO features (GPU - will unload after)...")
        try:
            cached = self._load_cache('dino_features') if use_cache else None
            if cached is not None:
                self.dino_features = cached
                logger.info("Loaded DINO features from cache (model not loaded)")
            else:
                self.dino_features, _ = self.dino.compute_features_batch(
                    valid_image_paths, show_progress=show_progress
                )
                self._save_cache('dino_features', self.dino_features)
                # UNLOAD DINO to free GPU memory
                self.dino.unload()
        except Exception as e:
            logger.warning(f"DINO extraction failed: {e}. Using Fourier features as fallback.")
            self.dino_features = self.fourier_features
            try:
                self.dino.unload()
            except:
                pass

        # 3. SAM3 complexity scores (1-dim) - LOAD -> EXTRACT -> UNLOAD
        logger.info("\n[3/4] SAM3 complexity (GPU - will unload after)...")
        cached = self._load_cache('sam_scores') if use_cache else None
        if cached is not None:
            self.sam_scores = cached
            logger.info("Loaded SAM scores from cache (model not loaded)")
        else:
            self.sam_scores = self._compute_sam_scores(valid_image_paths, show_progress)
            self._save_cache('sam_scores', self.sam_scores)
            # UNLOAD SAM3 to free GPU memory
            self.sam.unload()

        # 4. EL2N scores (1-dim) - LOAD -> EXTRACT -> UNLOAD
        logger.info("\n[4/4] DETR EL2N scores (GPU - will unload after)...")
        if self.detr_el2n:
            cached = self._load_cache('el2n_scores') if use_cache else None
            if cached is not None:
                self.el2n_scores = cached
                logger.info("Loaded EL2N scores from cache (model not loaded)")
            else:
                el2n_dict = self.detr_el2n.compute_el2n_scores(valid_image_paths, show_progress=show_progress)
                self.el2n_scores = np.array([el2n_dict.get(p, 0.5) for p in valid_image_paths])
                self._save_cache('el2n_scores', self.el2n_scores)
                # UNLOAD DETR to free GPU memory
                self.detr_el2n.unload()
            logger.info(f"EL2N stats: mean={self.el2n_scores.mean():.3f}, std={self.el2n_scores.std():.3f}")
        else:
            # Proxy: use gradient magnitude from Fourier as difficulty proxy
            logger.info("Using Fourier-based difficulty proxy (no DETR checkpoint)")
            self.el2n_scores = self.fourier_features[:, 2]  # High-frequency energy as proxy

        logger.info("\nAll features extracted! GPU memory should be free now.")

        return {
            'dino_features': self.dino_features,      # (N, 1024)
            'fourier_features': self.fourier_features,  # (N, 9)
            'sam_scores': self.sam_scores,              # (N,)
            'el2n_scores': self.el2n_scores,            # (N,)
            'valid_paths': self.valid_paths
        }

    def combine_features(self, normalize: bool = True) -> np.ndarray:
        """
        Combine all features into unified feature space.

        Creates a single feature vector per image:
        [DINO (1024) | Fourier (9) | SAM (1) | EL2N (1)] = 1035 dimensions

        Args:
            normalize: Whether to standardize features (recommended)

        Returns:
            Combined feature matrix (N, 1035)
        """
        logger.info("Combining all features into unified space")

        if self.dino_features is None:
            raise ValueError("Features not extracted. Call extract_all_features first.")

        n_samples = len(self.dino_features)

        # Reshape 1-dim features
        sam_reshaped = self.sam_scores.reshape(-1, 1)
        el2n_reshaped = self.el2n_scores.reshape(-1, 1)

        # Concatenate all features
        combined = np.hstack([
            self.dino_features,      # (N, 1024)
            self.fourier_features,   # (N, 9)
            sam_reshaped,            # (N, 1)
            el2n_reshaped            # (N, 1)
        ])

        logger.info(f"Combined features shape: {combined.shape}")

        if normalize:
            # StandardScaler: z-score normalization per feature
            scaler = StandardScaler()
            combined = scaler.fit_transform(combined)
            logger.info("Applied z-score normalization")

        self.combined_features = combined
        return combined

    def _cluster_with_faiss_gpu(self, features: np.ndarray, n_clusters: int,
                                 niter: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated K-means using Facebook's faiss library.
        10-50x faster than scikit-learn for large datasets.

        Uses subprocess to run in conda environment (avoids numpy version conflicts).

        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            niter: Number of iterations
            seed: Random seed

        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        import subprocess
        import tempfile

        n_samples, d = features.shape
        logger.info(f"FAISS K-Means: {n_samples:,} samples x {d} features -> {n_clusters:,} clusters")

        # Paths
        conda_python = r"F:\Instalki\Conda\python.exe"
        worker_script = Path(__file__).parent / "faiss_clustering_worker.py"

        if not os.path.exists(conda_python):
            logger.warning(f"Conda Python not found at {conda_python}, falling back to faiss-cpu")
            # Try direct faiss import (will use CPU if available)
            return self._cluster_with_faiss_direct(features, n_clusters, niter, seed)

        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        # Use temp files to pass data
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.pkl")
            output_path = os.path.join(tmpdir, "result.pkl")

            # Save features
            logger.info(f"Saving features to temp file...")
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)

            # Run faiss worker in conda environment
            cmd = [
                conda_python,
                str(worker_script),
                features_path,
                str(n_clusters),
                output_path,
                str(niter),
                str(seed)
            ]

            logger.info(f"Running FAISS worker in conda environment...")
            print(f"\n{'='*60}")
            print(f"FAISS GPU K-Means (conda subprocess)")
            print(f"{'='*60}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Show output in real-time
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

                if result.returncode != 0:
                    raise RuntimeError(f"FAISS worker failed with return code {result.returncode}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("FAISS worker timed out after 1 hour")

            # Load results
            logger.info(f"Loading clustering results...")
            with open(output_path, 'rb') as f:
                clustering_result = pickle.load(f)

            labels = clustering_result['labels']
            centers = clustering_result['centers']
            n_gpus = clustering_result['n_gpus']
            elapsed = clustering_result['elapsed_seconds']

            print(f"{'='*60}")
            mode = "GPU" if n_gpus > 0 else "CPU"
            print(f"FAISS {mode} clustering complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"{'='*60}\n")

            logger.info(f"Clustering complete! ({n_gpus} GPU(s), {elapsed:.1f}s)")

        return labels, centers

    def _cluster_with_faiss_direct(self, features: np.ndarray, n_clusters: int,
                                    niter: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct faiss clustering (fallback when conda not available).
        Uses faiss-cpu if installed.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. Install with:\n"
                "  pip install faiss-cpu  (for CPU)\n"
                "  conda install -c conda-forge faiss-gpu  (for GPU)"
            )

        n_samples, d = features.shape
        n_gpus = faiss.get_num_gpus()
        use_gpu = n_gpus > 0

        logger.info(f"{'GPU' if use_gpu else 'CPU'} K-means: {n_samples:,} x {d} -> {n_clusters:,} clusters")

        features_f32 = np.ascontiguousarray(features.astype(np.float32))

        kmeans = faiss.Kmeans(
            d, n_clusters,
            niter=niter,
            gpu=use_gpu,
            seed=seed,
            verbose=True,
            spherical=False,
            min_points_per_centroid=1
        )

        kmeans.train(features_f32)

        _, labels = kmeans.index.search(features_f32, 1)
        labels = labels.flatten()
        centers = kmeans.centroids

        return labels, centers

    def cluster_features(self,
                         n_clusters: int,
                         method: Literal['kmeans', 'minibatch', 'faiss-gpu', 'auto'] = 'auto',
                         random_state: int = 42) -> np.ndarray:
        """
        Cluster combined features using k-means.

        Args:
            n_clusters: Number of clusters (= target coreset size)
            method:
                - 'kmeans': scikit-learn exact K-means (slow)
                - 'minibatch': scikit-learn MiniBatchKMeans (faster)
                - 'faiss-gpu': Facebook faiss GPU K-means (fastest, 10-50x speedup)
                - 'auto': Smart selection based on problem size
            random_state: Random seed for reproducibility

        Returns:
            Cluster labels for each sample
        """
        if self.combined_features is None:
            raise ValueError("Features not combined. Call combine_features first.")

        n_samples = len(self.combined_features)
        import time

        # AUTO MODE: Intelligently select method and parameters based on problem size
        if method == 'auto':
            # Try faiss first for large problems (best performance)
            if n_samples > 10000 or n_clusters > 5000:
                try:
                    import faiss
                    n_gpus = faiss.get_num_gpus()
                    if n_gpus > 0:
                        method = 'faiss-gpu'
                        logger.info(f"Auto-selected faiss-gpu ({n_gpus} GPU(s)) (n_samples={n_samples}, n_clusters={n_clusters})")
                    else:
                        # faiss-cpu is still 2-5x faster than sklearn due to AVX/SIMD
                        method = 'faiss-gpu'  # Will fallback to CPU in _cluster_with_faiss_gpu
                        logger.info(f"Auto-selected faiss-cpu (faster than sklearn) (n_samples={n_samples}, n_clusters={n_clusters})")
                except ImportError:
                    method = 'minibatch'
                    logger.info(f"Auto-selected MiniBatchKMeans - faiss not installed (n_samples={n_samples}, n_clusters={n_clusters})")
            else:
                method = 'kmeans'
                logger.info(f"Auto-selected standard KMeans (n_samples={n_samples}, n_clusters={n_clusters})")

        # FAISS-GPU: Fastest option (10-50x speedup)
        if method == 'faiss-gpu':
            logger.info(f"Clustering {n_samples} samples into {n_clusters} clusters (method=faiss-gpu)")
            start_time = time.time()

            self.cluster_labels, self.cluster_centers = self._cluster_with_faiss_gpu(
                self.combined_features, n_clusters, niter=300, seed=random_state
            )

            elapsed = time.time() - start_time
            self.n_clusters = n_clusters

            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            logger.info(f"Clustering completed in {elapsed/60:.1f} minutes (faiss-gpu)")
            logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

            return self.cluster_labels

        # SKLEARN METHODS: MiniBatch or standard KMeans
        # Adaptive n_init: fewer restarts for large k (diminishing returns)
        if n_clusters > 10000:
            n_init = 3  # Very large k: 3 restarts enough
        elif n_clusters > 5000:
            n_init = 5  # Large k: 5 restarts
        else:
            n_init = 10  # Standard: 10 restarts

        logger.info(f"Clustering {n_samples} samples into {n_clusters} clusters (method={method}, n_init={n_init})")

        if method == 'minibatch':
            # MiniBatchKMeans: O(n * k * batch_size * n_init) - much faster for large datasets
            batch_size = min(4096, n_samples)  # Larger batch = better quality, still fast
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size,
                n_init=n_init,
                max_iter=300,
                verbose=1 if n_clusters > 5000 else 0  # Progress for large jobs
            )
        else:
            # Standard KMeans: O(n * k * d * max_iter * n_init) - exact but slow
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=300,
                verbose=1 if n_clusters > 5000 else 0
            )

        start_time = time.time()
        self.cluster_labels = clusterer.fit_predict(self.combined_features)
        elapsed = time.time() - start_time

        self.cluster_centers = clusterer.cluster_centers_
        self.n_clusters = n_clusters

        # Log cluster sizes
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        logger.info(f"Clustering completed in {elapsed/60:.1f} minutes")
        logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

        return self.cluster_labels

    def select_representatives(self,
                               strategy: Literal['centroid', 'max_el2n', 'medoid'] = 'centroid',
                               samples_per_cluster: int = 1) -> Tuple[List[int], Dict]:
        """
        Select representative sample(s) from each cluster.

        Args:
            strategy: Selection strategy:
                - 'centroid': Sample closest to cluster center (most typical)
                - 'max_el2n': Sample with highest EL2N in cluster (hardest)
                - 'medoid': Sample minimizing distance to all cluster members
            samples_per_cluster: Number of samples to select per cluster

        Returns:
            Tuple of (selected indices, statistics)
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed. Call cluster_features first.")

        logger.info(f"Selecting representatives using '{strategy}' strategy")

        selected_indices = []
        cluster_stats = []

        for cluster_id in range(self.n_clusters):
            # Get indices of samples in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                logger.warning(f"Cluster {cluster_id} is empty!")
                continue

            cluster_features = self.combined_features[cluster_indices]
            cluster_el2n = self.el2n_scores[cluster_indices]

            if strategy == 'centroid':
                # Find sample closest to cluster center
                center = self.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                local_selected = np.argsort(distances)[:samples_per_cluster]

            elif strategy == 'max_el2n':
                # Find sample(s) with highest EL2N (hardest examples)
                local_selected = np.argsort(cluster_el2n)[-samples_per_cluster:][::-1]

            elif strategy == 'medoid':
                # Find medoid (sample minimizing sum of distances to all cluster members)
                if len(cluster_indices) <= samples_per_cluster:
                    local_selected = np.arange(len(cluster_indices))
                else:
                    dist_matrix = pairwise_distances(cluster_features)
                    sum_distances = dist_matrix.sum(axis=1)
                    local_selected = np.argsort(sum_distances)[:samples_per_cluster]

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Map back to global indices
            global_selected = cluster_indices[local_selected]
            selected_indices.extend(global_selected.tolist())

            # Collect stats
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_indices),
                'selected': len(local_selected),
                'el2n_mean': float(cluster_el2n.mean()),
                'el2n_std': float(cluster_el2n.std()),
                'selected_el2n': float(self.el2n_scores[global_selected].mean())
            })

        logger.info(f"Selected {len(selected_indices)} samples from {self.n_clusters} clusters")

        stats = {
            'n_clusters': self.n_clusters,
            'n_selected': len(selected_indices),
            'strategy': strategy,
            'samples_per_cluster': samples_per_cluster,
            'cluster_stats': cluster_stats,
            'coverage': len(set(self.cluster_labels[selected_indices])) / self.n_clusters
        }

        return selected_indices, stats

    def select_optimal_subset(self,
                              image_paths: List[str],
                              target_size: int,
                              strategy: Literal['centroid', 'max_el2n', 'medoid'] = 'centroid',
                              use_cache: bool = True,
                              normalize: bool = True) -> Tuple[List[int], List[str], Dict]:
        """
        Run complete cluster-based selection pipeline.

        Args:
            image_paths: List of image paths
            target_size: Final number of samples to select (= number of clusters)
            strategy: Representative selection strategy
            use_cache: Use feature cache
            normalize: Normalize features before clustering

        Returns:
            Tuple of (selected indices, selected paths, statistics)
        """
        logger.info(f"=" * 60)
        logger.info(f"Starting Cluster-Based Selection Pipeline")
        logger.info(f"Input: {len(image_paths)} images -> Target: {target_size} images")
        logger.info(f"=" * 60)

        # Stage 1: Extract all features
        logger.info("\n[Stage 1] Extracting all features...")
        self.extract_all_features(image_paths, use_cache=use_cache)

        # Stage 2: Combine features
        logger.info("\n[Stage 2] Combining features into unified space...")
        self.combine_features(normalize=normalize)

        # Stage 3: Cluster
        logger.info(f"\n[Stage 3] Clustering into {target_size} clusters...")
        self.cluster_features(n_clusters=target_size)

        # Stage 4: Select representatives
        logger.info(f"\n[Stage 4] Selecting representatives ({strategy} strategy)...")
        selected_indices, cluster_stats = self.select_representatives(strategy=strategy)

        # Get selected paths
        selected_paths = [self.valid_paths[i] for i in selected_indices]

        # Compile final statistics
        stats = {
            'original_count': len(image_paths),
            'valid_count': len(self.valid_paths),
            'final_count': len(selected_indices),
            'reduction_ratio': len(selected_indices) / len(image_paths),
            'n_clusters': target_size,
            'selection_strategy': strategy,
            'feature_dims': {
                'dino': self.dino_features.shape[1],
                'fourier': self.fourier_features.shape[1],
                'sam': 1,
                'el2n': 1,
                'combined': self.combined_features.shape[1]
            },
            'el2n_selected_mean': float(np.mean(self.el2n_scores[selected_indices])),
            'el2n_selected_std': float(np.std(self.el2n_scores[selected_indices])),
            'el2n_all_mean': float(np.mean(self.el2n_scores)),
            'sam_selected_mean': float(np.mean(self.sam_scores[selected_indices])),
            'cluster_coverage': cluster_stats['coverage'],
            'cluster_details': cluster_stats['cluster_stats']
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Selection Complete!")
        logger.info(f"  Input: {stats['original_count']} images")
        logger.info(f"  Output: {stats['final_count']} images")
        logger.info(f"  Clusters: {stats['n_clusters']}")
        logger.info(f"  Coverage: {stats['cluster_coverage'] * 100:.1f}%")
        logger.info(f"  EL2N (selected): {stats['el2n_selected_mean']:.3f} +/- {stats['el2n_selected_std']:.3f}")
        logger.info(f"  EL2N (all): {stats['el2n_all_mean']:.3f}")
        logger.info(f"{'=' * 60}")

        return selected_indices, selected_paths, stats

    def get_cluster_info(self) -> Dict:
        """Get detailed cluster information for visualization."""
        if self.cluster_labels is None:
            return {}

        info = {
            'labels': self.cluster_labels,
            'centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'combined_features': self.combined_features,
            'el2n_scores': self.el2n_scores,
            'sam_scores': self.sam_scores,
            'dino_features': self.dino_features,
            'fourier_features': self.fourier_features
        }
        return info

    def generate_selection_report(self,
                                  selected_indices: List[int],
                                  strategy: str) -> Dict:
        """
        Generate detailed report explaining WHY each image was selected.

        Args:
            selected_indices: Indices of selected images
            strategy: Selection strategy used

        Returns:
            Dictionary with detailed selection reasons per image
        """
        logger.info("Generating detailed selection report...")

        report = {
            'summary': {
                'total_images': len(self.valid_paths),
                'selected_images': len(selected_indices),
                'n_clusters': self.n_clusters,
                'strategy': strategy,
                'el2n_mean_all': float(np.mean(self.el2n_scores)),
                'el2n_mean_selected': float(np.mean(self.el2n_scores[selected_indices])),
                'sam_mean_all': float(np.mean(self.sam_scores)),
                'sam_mean_selected': float(np.mean(self.sam_scores[selected_indices])),
            },
            'images': []
        }

        # Get DETR detection info if available
        detr_info = {}
        if self.detr_el2n and hasattr(self.detr_el2n, 'detection_info'):
            detr_info = self.detr_el2n.detection_info

        for rank, idx in enumerate(selected_indices, 1):
            img_path = self.valid_paths[idx]
            cluster_id = int(self.cluster_labels[idx])

            # Get cluster members for context
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = int(np.sum(cluster_mask))
            cluster_el2n_mean = float(np.mean(self.el2n_scores[cluster_mask]))
            cluster_el2n_std = float(np.std(self.el2n_scores[cluster_mask]))

            # Distance to cluster center
            center = self.cluster_centers[cluster_id]
            dist_to_center = float(np.linalg.norm(self.combined_features[idx] - center))

            # EL2N rank within cluster
            cluster_indices = np.where(cluster_mask)[0]
            cluster_el2n = self.el2n_scores[cluster_indices]
            el2n_rank_in_cluster = int(np.sum(cluster_el2n > self.el2n_scores[idx])) + 1

            # Build reason string
            reasons = []
            if strategy == 'centroid':
                reasons.append(f"Closest to cluster {cluster_id} center (dist={dist_to_center:.3f})")
            elif strategy == 'max_el2n':
                reasons.append(f"Highest EL2N in cluster {cluster_id} (rank {el2n_rank_in_cluster}/{cluster_size})")
            elif strategy == 'medoid':
                reasons.append(f"Medoid of cluster {cluster_id}")

            reasons.append(f"EL2N={self.el2n_scores[idx]:.3f} (cluster mean={cluster_el2n_mean:.3f})")
            reasons.append(f"SAM complexity={self.sam_scores[idx]:.3f}")

            # Get DETR Q81 detection info
            detr_detection = detr_info.get(img_path, {})
            q81_score = detr_detection.get('q81_score', None)
            has_detection = detr_detection.get('has_detection', None)
            q81_box = detr_detection.get('q81_box', None)

            if q81_score is not None:
                det_status = "DETECTED" if has_detection else "NO DETECTION"
                reasons.append(f"DETR Q81: {q81_score:.3f} ({det_status})")

            image_report = {
                'rank': rank,
                'path': img_path,
                'filename': Path(img_path).name,
                'cluster_id': cluster_id,
                'cluster_size': cluster_size,
                'el2n_score': float(self.el2n_scores[idx]),
                'sam_score': float(self.sam_scores[idx]),
                'distance_to_center': dist_to_center,
                'el2n_rank_in_cluster': el2n_rank_in_cluster,
                'cluster_el2n_mean': cluster_el2n_mean,
                'cluster_el2n_std': cluster_el2n_std,
                'reasons': reasons,
                'detr_q81': {
                    'score': q81_score,
                    'has_detection': has_detection,
                    'box': q81_box
                } if q81_score is not None else None
            }

            report['images'].append(image_report)

        logger.info(f"Generated report for {len(report['images'])} selected images")
        return report

    def save_selection_report(self,
                              selected_indices: List[int],
                              strategy: str,
                              output_path: str):
        """
        Save detailed selection report to JSON file.

        Args:
            selected_indices: Indices of selected images
            strategy: Selection strategy used
            output_path: Path to save report
        """
        import json
        from datetime import datetime

        report = self.generate_selection_report(selected_indices, strategy)
        report['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_version': '2.0 (Cluster-Based)',
            'detr_query': 81
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Selection report saved to {output_path}")

        # Also save human-readable summary
        summary_path = output_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET SELECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Images: {report['summary']['total_images']}\n")
            f.write(f"Selected Images: {report['summary']['selected_images']}\n")
            f.write(f"Clusters: {report['summary']['n_clusters']}\n")
            f.write(f"Strategy: {report['summary']['strategy']}\n")
            f.write(f"EL2N Mean (all): {report['summary']['el2n_mean_all']:.3f}\n")
            f.write(f"EL2N Mean (selected): {report['summary']['el2n_mean_selected']:.3f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("WHY EACH IMAGE WAS SELECTED:\n")
            f.write("-" * 80 + "\n\n")

            for img in report['images']:
                f.write(f"[{img['rank']:3d}] {img['filename']}\n")
                f.write(f"      Cluster: {img['cluster_id']} (size={img['cluster_size']})\n")
                for reason in img['reasons']:
                    f.write(f"      - {reason}\n")
                f.write("\n")

        logger.info(f"Human-readable report saved to {summary_path}")


if __name__ == "__main__":
    print("Testing ClusterBasedSelector...")

    # Create with default components (no DETR checkpoint for testing)
    selector = ClusterBasedSelector()

    # Test with synthetic data
    n_samples = 100
    selector.dino_features = np.random.randn(n_samples, 1024)
    selector.fourier_features = np.random.randn(n_samples, 9)
    selector.sam_scores = np.random.rand(n_samples)
    selector.el2n_scores = np.random.rand(n_samples)
    selector.valid_paths = [f"img_{i}.jpg" for i in range(n_samples)]

    # Test pipeline
    print("\n[Test] Combining features...")
    combined = selector.combine_features(normalize=True)
    print(f"  Combined shape: {combined.shape}")

    print("\n[Test] Clustering...")
    labels = selector.cluster_features(n_clusters=10)
    print(f"  Cluster labels shape: {labels.shape}")
    print(f"  Unique clusters: {len(np.unique(labels))}")

    print("\n[Test] Selecting representatives (centroid)...")
    indices, stats = selector.select_representatives(strategy='centroid')
    print(f"  Selected: {len(indices)}")

    print("\n[Test] Selecting representatives (max_el2n)...")
    indices, stats = selector.select_representatives(strategy='max_el2n')
    print(f"  Selected: {len(indices)}")

    print("\nClusterBasedSelector test completed!")
