#!/usr/bin/env python3
"""
Feature Clustering Visualization for Scientific Paper

Generates publication-quality visualizations showing:
1. t-SNE/UMAP dimensionality reduction of combined features
2. K-Means clustering visualization
3. Cluster centroid analysis
4. Feature space exploration with sample images

Usage Examples:
    # Select first 500 images from directory (default)
    py -3.11 visualize_clustering.py --num_images 500

    # Random selection of 200 images from dataset
    py -3.11 visualize_clustering.py --num_images 200 --random

    # Specific images by filename
    py -3.11 visualize_clustering.py --images image1.jpg image2.jpg image3.jpg

    # Specific images with full paths
    py -3.11 visualize_clustering.py --images "F:/path/to/image1.jpg" "F:/path/to/image2.jpg"

    # Different number of clusters
    py -3.11 visualize_clustering.py --num_images 500 --n_clusters 15

    # Large dataset (e.g., 20k images): use PCA/auto embedding and limit sample grid
    py -3.11 visualize_clustering.py --num_images 20000 --n_clusters 50 --embedding pca --sample_clusters 20

    # Control performance (DINO batching + auto embedding threshold)
    py -3.11 visualize_clustering.py --num_images 20000 --n_clusters 50 --dino_batch_size 16 --max_tsne_samples 2000

    # Show help
    py -3.11 visualize_clustering.py --help
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


SIMPLE_FEATURE_DIM = 101  # 3*32 HSV hist + laplacian(2) + edge(1) + brightness/contrast(2)


class ClusteringVisualizer:
    """Professional clustering visualization for paper figures."""

    def __init__(self, device='cuda'):
        self.device = device

        # Initialize feature extractors
        self.dino_extractor = None
        self.fourier_analyzer = None

        # Set matplotlib style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def _initialize_dino(self):
        """Lazy initialization of DINO extractor."""
        if self.dino_extractor is None:
            try:
                from feature_extractors.dino_extractor import DINOExtractor
                self.dino_extractor = DINOExtractor(model_name='dino_vits16', device=self.device)
                print("DINO extractor initialized")
            except Exception as e:
                print(f"Could not initialize DINO: {e}")
                print("Using simple feature extraction as fallback")

    def _initialize_fourier(self):
        """Lazy initialization of Fourier analyzer."""
        if self.fourier_analyzer is None:
            try:
                from feature_extractors.fourier_analyzer import FourierAnalyzer
                self.fourier_analyzer = FourierAnalyzer()
                print("Fourier analyzer initialized")
            except Exception as e:
                print(f"Could not initialize Fourier: {e}")

    def extract_simple_features(self, image_path):
        """Extract simple color and texture features (fallback)."""
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Resize for consistency
        image = cv2.resize(image, (256, 256))

        features = []

        # Color histogram features (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist)

        # Texture features (Gabor-like)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.std(laplacian))

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)

        # Brightness and contrast
        features.append(np.mean(gray))
        features.append(np.std(gray))

        return np.array(features, dtype=np.float32)

    def extract_features_batch(self, image_paths, use_dino=True, show_progress=True, dino_batch_size=8):
        """Extract features for all images.

        Uses batched DINO extraction for speed on large datasets.
        """
        if use_dino:
            self._initialize_dino()

        # Fast path: batched DINO + simple features
        if use_dino and self.dino_extractor is not None:
            dino_features, valid_paths = self.dino_extractor.compute_features_batch(
                list(image_paths),
                batch_size=dino_batch_size,
                show_progress=show_progress
            )
            if len(valid_paths) == 0:
                return np.array([]), []

            # Compute simple features for the same set of valid paths (keeps feature dim consistent)
            simple_features = []
            iterator = tqdm(valid_paths, desc="Extracting simple features") if show_progress else valid_paths
            for img_path in iterator:
                feat = self.extract_simple_features(img_path)
                if feat is None:
                    feat = np.zeros(SIMPLE_FEATURE_DIM, dtype=np.float32)
                simple_features.append(feat)

            simple_features = np.vstack(simple_features).astype(np.float32)
            dino_features = np.asarray(dino_features, dtype=np.float32)
            combined = np.hstack([dino_features, simple_features])
            return combined, valid_paths

        # Fallback: simple features only (no DINO)
        features_list = []
        valid_paths = []
        iterator = tqdm(image_paths, desc="Extracting simple features") if show_progress else image_paths

        for img_path in iterator:
            try:
                combined = self.extract_simple_features(img_path)
                if combined is None:
                    continue
                features_list.append(combined)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if not features_list:
            return np.array([]), []

        return np.vstack(features_list).astype(np.float32), valid_paths

    def perform_clustering(self, features, n_clusters=10):
        """Perform K-Means clustering on features."""
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        # Ensure n_clusters doesn't exceed n_samples
        n_samples = len(features)
        if n_clusters > n_samples:
            print(f"Warning: n_clusters ({n_clusters}) > n_samples ({n_samples}). Reducing to {n_samples}.")
            n_clusters = n_samples

        # Use MiniBatchKMeans for large datasets
        if len(features) > 1000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        cluster_labels = kmeans.fit_predict(features_norm)
        cluster_centers = kmeans.cluster_centers_

        return {
            'labels': cluster_labels,
            'centers': cluster_centers,
            'features_norm': features_norm,
            'scaler': scaler,
            'kmeans': kmeans,
            'inertia': kmeans.inertia_
        }

    def _select_cluster_ids(self, labels, n_clusters, n_select, strategy="quantiles"):
        """Select cluster IDs to display in a scalable way."""
        if n_select <= 0:
            return []

        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        non_empty = np.where(cluster_sizes > 0)[0]
        if len(non_empty) == 0:
            return []
        if len(non_empty) <= n_select:
            return non_empty.tolist()

        if strategy == "largest":
            sorted_ids = non_empty[np.argsort(cluster_sizes[non_empty])[::-1]]
            return sorted_ids[:n_select].tolist()

        if strategy == "random":
            return np.random.choice(non_empty, size=n_select, replace=False).tolist()

        if strategy == "quantiles":
            sorted_ids = non_empty[np.argsort(cluster_sizes[non_empty])]
            idxs = np.linspace(0, len(sorted_ids) - 1, n_select).round().astype(int)
            return sorted_ids[idxs].tolist()

        raise ValueError(f"Unknown selection strategy: {strategy}")

    def compute_tsne(self, features, perplexity=30, n_iter=1000):
        """Compute t-SNE embedding."""
        print("Computing t-SNE embedding...")

        n_samples = len(features)

        # t-SNE requires n_samples > perplexity * 3 and perplexity > 0
        if n_samples < 4:
            print(f"Warning: Too few samples ({n_samples}) for t-SNE. Need at least 4.")
            # Return simple 2D projection for very small datasets
            if features.shape[1] >= 2:
                return features[:, :2]
            else:
                return np.column_stack([features[:, 0], np.zeros(n_samples)])

        # Calculate safe perplexity: must be < n_samples and ideally n_samples > 3*perplexity
        max_perplexity = max(1, (n_samples - 1) // 3)
        safe_perplexity = min(perplexity, max_perplexity, n_samples - 1)
        safe_perplexity = max(1, safe_perplexity)  # Ensure at least 1

        print(f"Using perplexity={safe_perplexity} for {n_samples} samples")

        # Use PCA first if high-dimensional
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, n_samples - 1))
            features_pca = pca.fit_transform(features)
        else:
            features_pca = features

        # scikit-learn compatibility: TSNE used to accept `n_iter`, newer versions use `max_iter`.
        tsne_kwargs = dict(n_components=2, perplexity=safe_perplexity, random_state=42, init='pca')
        try:
            tsne = TSNE(**tsne_kwargs, n_iter=n_iter)
        except TypeError as e:
            if "n_iter" not in str(e):
                raise
            tsne = TSNE(**tsne_kwargs, max_iter=n_iter)
        embedding = tsne.fit_transform(features_pca)

        return embedding

    def compute_embedding(self,
                          features,
                          method="auto",
                          tsne_perplexity=30,
                          tsne_iter=1000,
                          max_tsne_samples=5000,
                          umap_n_neighbors=15,
                          umap_min_dist=0.1):
        """Compute 2D embedding with automatic scaling for large datasets."""
        n_samples = len(features)
        chosen = method.lower()

        if chosen == "auto":
            if n_samples <= max_tsne_samples:
                chosen = "tsne"
            else:
                # Prefer UMAP for large datasets if available
                try:
                    import umap  # type: ignore
                except Exception:
                    chosen = "pca"
                else:
                    chosen = "umap"

        if chosen == "tsne":
            embedding = self.compute_tsne(features, perplexity=tsne_perplexity, n_iter=tsne_iter)
            info = {"method": "tsne", "x_label": "t-SNE 1", "y_label": "t-SNE 2"}
            return embedding, info

        if chosen == "pca":
            print("Computing PCA projection...")
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(features)
            info = {
                "method": "pca",
                "x_label": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "y_label": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            }
            return embedding, info

        if chosen == "umap":
            try:
                import umap  # type: ignore
            except Exception:
                print("UMAP not available (install umap-learn). Falling back to PCA.")
                return self.compute_embedding(
                    features,
                    method="pca",
                    tsne_perplexity=tsne_perplexity,
                    tsne_iter=tsne_iter,
                    max_tsne_samples=max_tsne_samples,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                )

            print("Computing UMAP embedding...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                random_state=42,
            )
            embedding = reducer.fit_transform(features)
            info = {"method": "umap", "x_label": "UMAP 1", "y_label": "UMAP 2"}
            return embedding, info

        raise ValueError(f"Unknown embedding method: {method}")

    def visualize_clustering(self,
                             image_paths,
                             output_path,
                             n_clusters=10,
                             max_images=500,
                             show=False,
                             embedding_method="auto",
                             tsne_perplexity=30,
                             tsne_iter=1000,
                             max_tsne_samples=5000,
                             dino_batch_size=8):
        """Create comprehensive clustering visualization."""
        # Limit number of images
        image_paths = image_paths[:max_images]
        print(f"Processing {len(image_paths)} images...")

        # Extract features
        features, valid_paths = self.extract_features_batch(
            image_paths,
            use_dino=True,
            show_progress=True,
            dino_batch_size=dino_batch_size
        )
        if len(features) == 0:
            print("No features extracted!")
            return None, [], None

        print(f"Extracted features for {len(valid_paths)} images, shape: {features.shape}")

        # Perform clustering
        cluster_result = self.perform_clustering(features, n_clusters=n_clusters)

        labels = cluster_result['labels']
        centers = cluster_result['centers']

        # Compute embedding (auto-scales for large N)
        embedding, emb_info = self.compute_embedding(
            cluster_result['features_norm'],
            method=embedding_method,
            tsne_perplexity=tsne_perplexity,
            tsne_iter=tsne_iter,
            max_tsne_samples=max_tsne_samples
        )

        x_label = emb_info["x_label"]
        y_label = emb_info["y_label"]

        # Pre-compute scalable cluster statistics
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        # Per-sample distance to its centroid (vectorized)
        dist_to_centroid = np.linalg.norm(cluster_result['features_norm'] - centers[labels], axis=1)
        sum_dist = np.bincount(labels, weights=dist_to_centroid, minlength=n_clusters)
        avg_dist_per_cluster = sum_dist / np.maximum(cluster_sizes, 1)

        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main embedding plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        max_categorical_clusters = 50
        if n_clusters <= max_categorical_clusters:
            cmap = plt.cm.get_cmap('gist_ncar', n_clusters) if n_clusters > 20 else plt.cm.get_cmap('tab20', n_clusters)
            scatter = ax1.scatter(
                embedding[:, 0], embedding[:, 1],
                c=labels, cmap=cmap,
                alpha=0.7, s=10, edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cluster ID')
            ax1.set_title(f"(a) {emb_info['method'].upper()} Space with K-Means Clustering", fontweight='bold')
        else:
            sizes_per_point = cluster_sizes[labels]
            vmax = int(sizes_per_point.max()) if len(sizes_per_point) else 1
            norm = LogNorm(vmin=1, vmax=max(1, vmax)) if vmax > 1 else None
            scatter = ax1.scatter(
                embedding[:, 0], embedding[:, 1],
                c=sizes_per_point, cmap='viridis', norm=norm,
                alpha=0.6, s=6, edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cluster size')
            ax1.set_title(f"(a) {emb_info['method'].upper()} Space (colored by cluster size)", fontweight='bold')

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)

        # Cluster size distribution
        ax2 = fig.add_subplot(gs[0, 2])
        if n_clusters <= 50:
            unique = np.arange(n_clusters)
            counts = cluster_sizes
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 2)))
            ax2.bar(unique, counts, color=colors[:n_clusters], edgecolor='black')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('(b) Cluster Size Distribution', fontweight='bold')
        else:
            counts = cluster_sizes[cluster_sizes > 0]
            ax2.hist(counts, bins=50, color='steelblue', edgecolor='black', alpha=0.8, log=True)
            ax2.axvline(np.mean(counts), color='red', linestyle='--', label=f"Mean: {np.mean(counts):.1f}")
            ax2.axvline(np.median(counts), color='black', linestyle=':', label=f"Median: {np.median(counts):.1f}")
            ax2.set_xlabel('Cluster size')
            ax2.set_ylabel('Count (log)')
            ax2.set_title('(b) Cluster Size Histogram', fontweight='bold')
            ax2.legend(fontsize=9)

        # Cluster proportions / rank-size plot
        ax3 = fig.add_subplot(gs[0, 3])
        if n_clusters <= 10:
            unique = np.arange(n_clusters)
            counts = cluster_sizes
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            ax3.pie(counts, labels=[f'C{i}' for i in unique], autopct='%1.1f%%',
                    colors=colors, startangle=90)
            ax3.set_title('(c) Cluster Proportions', fontweight='bold')
        else:
            sizes_sorted = np.sort(cluster_sizes[cluster_sizes > 0])[::-1]
            ax3.plot(np.arange(1, len(sizes_sorted) + 1), sizes_sorted, color='black', linewidth=2)
            ax3.set_yscale('log')
            ax3.set_xlabel('Cluster rank')
            ax3.set_ylabel('Cluster size (log)')
            ax3.set_title('(c) Cluster Size Rank Plot', fontweight='bold')

        # Elbow plot (small k) or centroid-distance distribution (large k)
        ax4 = fig.add_subplot(gs[1, 2])
        if n_clusters <= 15 and len(features) <= 5000:
            k_range = range(2, min(15, len(features) // 2))
            inertias = []
            for k in k_range:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
                km.fit(cluster_result['features_norm'])
                inertias.append(km.inertia_)
            ax4.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
            if n_clusters in k_range:
                ax4.axvline(x=n_clusters, color='r', linestyle='--', label=f'Selected k={n_clusters}')
            ax4.set_xlabel('Number of Clusters (k)')
            ax4.set_ylabel('Inertia')
            ax4.set_title('(d) Elbow Method (MiniBatchKMeans)', fontweight='bold')
            handles, legend_labels = ax4.get_legend_handles_labels()
            if legend_labels:
                ax4.legend(fontsize=9)
        else:
            ax4.hist(dist_to_centroid, bins=50, color='purple', edgecolor='black', alpha=0.8)
            ax4.set_xlabel('Distance to centroid')
            ax4.set_ylabel('Count')
            ax4.set_title('(d) Distances to Centroid (per sample)', fontweight='bold')

        # Cluster compactness (avg distance) - scalable rendering
        ax5 = fig.add_subplot(gs[1, 3])
        if n_clusters <= 30:
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 2)))
            ax5.barh(np.arange(n_clusters), avg_dist_per_cluster, color=colors[:n_clusters], edgecolor='black')
            ax5.set_ylabel('Cluster ID')
            ax5.set_xlabel('Avg Distance to Centroid')
            ax5.set_title('(e) Cluster Compactness', fontweight='bold')
        else:
            non_empty = cluster_sizes > 0
            ax5.scatter(
                cluster_sizes[non_empty],
                avg_dist_per_cluster[non_empty],
                s=12, alpha=0.5, color='teal', edgecolors='none'
            )
            ax5.set_xscale('log')
            ax5.set_xlabel('Cluster size (log)')
            ax5.set_ylabel('Avg distance to centroid')
            ax5.set_title('(e) Compactness vs Cluster Size', fontweight='bold')

        # Sample images from each cluster (Row 3)
        n_samples_per_cluster = 4
        clusters_to_show = self._select_cluster_ids(labels, n_clusters, n_select=min(4, n_clusters), strategy="quantiles")
        for col in range(4):
            ax = fig.add_subplot(gs[2, col])
            if col >= len(clusters_to_show):
                ax.axis('off')
                continue

            cluster_id = clusters_to_show[col]

            # Get images from this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Select random samples
                sample_indices = np.random.choice(cluster_indices,
                                                  size=min(n_samples_per_cluster, len(cluster_indices)),
                                                  replace=False)

                # Create grid of sample images
                grid_size = 2
                grid_img = np.ones((grid_size * 64, grid_size * 64, 3), dtype=np.uint8) * 255

                for i, idx in enumerate(sample_indices[:grid_size * grid_size]):
                    img = cv2.imread(valid_paths[idx])
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        row = i // grid_size
                        col = i % grid_size
                        grid_img[row*64:(row+1)*64, col*64:(col+1)*64] = img

                ax.imshow(grid_img)
                ax.set_title(f'(f{col+1}) Cluster {cluster_id}\n({np.sum(cluster_mask)} samples)',
                             fontweight='bold')
            else:
                ax.set_title(f'Cluster {cluster_id}\n(empty)', fontweight='bold')

            ax.axis('off')

        plt.suptitle(f'Feature Clustering Analysis ({len(valid_paths)} images, {n_clusters} clusters)',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

        return cluster_result, valid_paths, embedding

    def visualize_cluster_samples(self,
                                  image_paths,
                                  cluster_result,
                                  valid_paths,
                                  output_path,
                                  n_clusters=10,
                                  samples_per_cluster=6,
                                  max_clusters=20,
                                  cluster_selection="largest",
                                  show=False):
        """Visualize sample images from clusters (auto-limited for large k)."""
        labels = cluster_result['labels']
        n_clusters_total = int(np.max(labels)) + 1 if len(labels) else 0

        clusters_to_show = min(max_clusters, n_clusters_total)
        if n_clusters_total > clusters_to_show:
            print(f"Too many clusters to show ({n_clusters_total}); displaying {clusters_to_show} ({cluster_selection})")

        cluster_ids = self._select_cluster_ids(
            labels,
            n_clusters_total,
            n_select=clusters_to_show,
            strategy=cluster_selection
        )

        if not cluster_ids:
            print("No clusters to display in sample grid.")
            return

        n_cols = samples_per_cluster
        n_rows = len(cluster_ids)

        fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3 * (n_cols + 1), 3 * n_rows))
        axes = np.atleast_2d(axes)

        for row, cluster_id in enumerate(cluster_ids):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Cluster info cell
            axes[row, 0].text(0.5, 0.5, f'Cluster {cluster_id}\n{int(cluster_mask.sum())} images',
                              ha='center', va='center', fontsize=12, fontweight='bold')
            axes[row, 0].axis('off')

            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(
                    cluster_indices,
                    size=min(samples_per_cluster, len(cluster_indices)),
                    replace=False
                )

                for i, idx in enumerate(sample_indices):
                    img = cv2.imread(valid_paths[idx])
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[row, i + 1].imshow(img)
                    axes[row, i + 1].axis('off')

                # Fill empty cells
                for i in range(len(sample_indices), samples_per_cluster):
                    axes[row, i + 1].axis('off')
            else:
                for i in range(1, n_cols + 1):
                    axes[row, i].axis('off')

        plt.suptitle('Sample Images from Selected Clusters', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Feature Clustering Visualization for Scientific Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First 500 images (sequential)
  py -3.11 visualize_clustering.py --num_images 500

  # Random 200 images from dataset
  py -3.11 visualize_clustering.py --num_images 200 --random

  # Specific images by filename (searches in --images_dir)
  py -3.11 visualize_clustering.py --images frame_001.jpg frame_002.jpg

  # Specific images with full paths
  py -3.11 visualize_clustering.py --images "F:/path/image1.jpg" "F:/path/image2.jpg"

  # Different number of clusters
  py -3.11 visualize_clustering.py --num_images 500 --n_clusters 15

  # Large dataset (e.g., 20k images)
  py -3.11 visualize_clustering.py --num_images 20000 --n_clusters 50 --embedding pca --sample_clusters 20
        """
    )

    # Image selection options
    selection_group = parser.add_argument_group('Image Selection')
    selection_group.add_argument('--images_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images',
                        help='Directory with images (default: selected_dataset/images)')
    selection_group.add_argument('--images', nargs='+', type=str, default=None,
                        help='Specific image files (filenames or full paths). Overrides --num_images')
    selection_group.add_argument('--num_images', type=int, default=500,
                        help='Number of images to process (default: 500)')
    selection_group.add_argument('--random', action='store_true',
                        help='Randomly select images instead of first N sequential')
    selection_group.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible selection (optional)')

    # Clustering options
    clustering_group = parser.add_argument_group('Clustering Options')
    clustering_group.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters (default: 10)')
    clustering_group.add_argument('--embedding', type=str, default='auto',
                        choices=['auto', 'pca', 'tsne', 'umap'],
                        help='2D embedding method (default: auto)')
    clustering_group.add_argument('--max_tsne_samples', type=int, default=5000,
                        help='Auto: use t-SNE only up to this many samples (default: 5000)')
    clustering_group.add_argument('--tsne_perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    clustering_group.add_argument('--tsne_iter', type=int, default=1000,
                        help='t-SNE iterations/max_iter (default: 1000)')
    clustering_group.add_argument('--dino_batch_size', type=int, default=8,
                        help='Batch size for DINO feature extraction (default: 8)')
    clustering_group.add_argument('--sample_clusters', type=int, default=20,
                        help='Max clusters to display in cluster_samples.png (default: 20)')
    clustering_group.add_argument('--samples_per_cluster', type=int, default=6,
                        help='Images per cluster row in cluster_samples.png (default: 6)')
    clustering_group.add_argument('--cluster_selection', type=str, default='largest',
                        choices=['largest', 'random', 'quantiles'],
                        help='How to choose clusters for cluster_samples.png (default: largest)')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\clustering',
                        help='Output directory for visualizations')
    output_group.add_argument('--show', action='store_true',
                        help='Show plots interactively (default: save only)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image files based on selection mode
    images_dir = Path(args.images_dir)

    if args.images:
        # Mode 1: Specific images provided
        image_files = []
        for img_name in args.images:
            img_path = Path(img_name)
            if img_path.is_absolute() and img_path.exists():
                # Full path provided
                image_files.append(img_path)
            else:
                # Filename only - search in images_dir
                full_path = images_dir / img_name
                if full_path.exists():
                    image_files.append(full_path)
                else:
                    # Try to find with glob pattern
                    matches = list(images_dir.glob(f"*{img_name}*"))
                    if matches:
                        image_files.append(matches[0])
                        print(f"Found match: {img_name} -> {matches[0].name}")
                    else:
                        print(f"Warning: Image not found: {img_name}")
        print(f"Selected {len(image_files)} specific images")
    else:
        # Mode 2: Select from directory
        all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if not all_images:
            print(f"No images found in {images_dir}")
            return

        if args.random:
            # Random selection
            image_files = random.sample(all_images, min(args.num_images, len(all_images)))
            print(f"Randomly selected {len(image_files)} images from {len(all_images)} total")
        else:
            # Sequential selection (first N)
            image_files = all_images[:args.num_images]
            print(f"Selected first {len(image_files)} images from {len(all_images)} total")

    if not image_files:
        print("No images to process!")
        return

    print(f"\nImages to process: {len(image_files)}")

    # Initialize visualizer
    visualizer = ClusteringVisualizer()

    # Generate main clustering visualization
    print("\n=== Generating clustering visualization ===")
    main_output = output_dir / "clustering_analysis.png"
    cluster_result, valid_paths, embedding = visualizer.visualize_clustering(
        [str(p) for p in image_files],
        str(main_output),
        n_clusters=args.n_clusters,
        max_images=args.num_images,
        show=args.show,
        embedding_method=args.embedding,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iter=args.tsne_iter,
        max_tsne_samples=args.max_tsne_samples,
        dino_batch_size=args.dino_batch_size
    )

    # Generate cluster samples visualization
    if cluster_result is not None:
        print("\n=== Generating cluster samples visualization ===")
        samples_output = output_dir / "cluster_samples.png"
        visualizer.visualize_cluster_samples(
            [str(p) for p in image_files],
            cluster_result,
            valid_paths,
            str(samples_output),
            n_clusters=args.n_clusters,
            samples_per_cluster=args.samples_per_cluster,
            max_clusters=args.sample_clusters,
            cluster_selection=args.cluster_selection,
            show=args.show
        )

    print(f"\n=== Done! Visualizations saved to: {output_dir} ===")


if __name__ == '__main__':
    main()
