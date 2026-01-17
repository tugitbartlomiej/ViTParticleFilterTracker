#!/usr/bin/env python3
"""
3D Feature Clustering Visualization for Scientific Paper

Generates publication-quality 3D visualizations showing:
1. PCA reduction from high-dimensional features (DINO+Fourier+EL2N+SAM) to 3 components
2. K-Means clustering visualization in 3D space
3. Multiple viewing angles for publication
4. Explained variance for each principal component

Features used (from cache):
- DINO features: 1024 dims (self-supervised visual features)
- Fourier features: 9 dims (spectral analysis)
- EL2N scores: 1 dim (training difficulty)
- SAM scores: 1 dim (segmentation complexity)
- TOTAL: 1035 dimensions -> PCA -> 3 dimensions

Usage Examples:
    # Basic 3D visualization with 50 clusters (uses cached features)
    py -3.11 visualize_clustering_3d.py --n_clusters 50

    # Limit to 20k images
    py -3.11 visualize_clustering_3d.py --num_images 20000 --n_clusters 50

    # Generate multiple viewing angles
    py -3.11 visualize_clustering_3d.py --n_clusters 50 --multi_view

    # Custom viewing angles (elev,azim pairs)
    py -3.11 visualize_clustering_3d.py --angles 30,45 60,90 0,0

    # Interactive mode (rotate with mouse)
    py -3.11 visualize_clustering_3d.py --n_clusters 20 --interactive

    # Choose which features to include
    py -3.11 visualize_clustering_3d.py --features dino fourier el2n sam

    # Only DINO features
    py -3.11 visualize_clustering_3d.py --features dino
"""

import os
import sys
import argparse
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Default paths
DEFAULT_CACHE_DIR = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\feature_cache')
DEFAULT_OUTPUT_DIR = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\clustering_3d')


class Clustering3DVisualizer:
    """Professional 3D clustering visualization for paper figures using cached features."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        # Cached data
        self.dino_features = None
        self.fourier_features = None
        self.el2n_scores = None
        self.sam_scores = None
        self.image_paths = None

        # Set matplotlib style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def load_cached_features(self):
        """Load all cached features from pickle files."""
        print(f"\n{'='*60}")
        print("LOADING CACHED FEATURES")
        print(f"{'='*60}")
        print(f"Cache directory: {self.cache_dir}")

        # Load DINO features (1024 dims)
        dino_path = self.cache_dir / 'dino_features.pkl'
        if dino_path.exists():
            self.dino_features = pickle.load(open(dino_path, 'rb'))
            print(f"  DINO features: {self.dino_features.shape} ({self.dino_features.shape[1]} dims)")
        else:
            print(f"  WARNING: DINO features not found at {dino_path}")

        # Load Fourier features (9 dims)
        fourier_path = self.cache_dir / 'fourier_features.pkl'
        if fourier_path.exists():
            fourier_data = pickle.load(open(fourier_path, 'rb'))
            if isinstance(fourier_data, tuple):
                self.fourier_features = fourier_data[0]  # First element is the feature array
                self.image_paths = fourier_data[1]  # Second element is list of image paths
            else:
                self.fourier_features = fourier_data
            print(f"  Fourier features: {self.fourier_features.shape} ({self.fourier_features.shape[1]} dims)")
        else:
            print(f"  WARNING: Fourier features not found at {fourier_path}")

        # Load EL2N scores (1 dim)
        el2n_path = self.cache_dir / 'el2n_scores.pkl'
        if el2n_path.exists():
            self.el2n_scores = pickle.load(open(el2n_path, 'rb'))
            if self.el2n_scores.ndim == 1:
                self.el2n_scores = self.el2n_scores.reshape(-1, 1)
            print(f"  EL2N scores: {self.el2n_scores.shape} ({self.el2n_scores.shape[1]} dims)")
        else:
            print(f"  WARNING: EL2N scores not found at {el2n_path}")

        # Load SAM scores (1 dim)
        sam_path = self.cache_dir / 'sam_scores.pkl'
        if sam_path.exists():
            self.sam_scores = pickle.load(open(sam_path, 'rb'))
            if self.sam_scores.ndim == 1:
                self.sam_scores = self.sam_scores.reshape(-1, 1)
            print(f"  SAM scores: {self.sam_scores.shape} ({self.sam_scores.shape[1]} dims)")
        else:
            print(f"  WARNING: SAM scores not found at {sam_path}")

        # Verify all features have same number of samples
        n_samples = None
        for name, feat in [('DINO', self.dino_features),
                           ('Fourier', self.fourier_features),
                           ('EL2N', self.el2n_scores),
                           ('SAM', self.sam_scores)]:
            if feat is not None:
                if n_samples is None:
                    n_samples = len(feat)
                elif len(feat) != n_samples:
                    print(f"  WARNING: {name} has {len(feat)} samples, expected {n_samples}")

        print(f"\nTotal samples available: {n_samples}")
        return n_samples

    def combine_features(self, feature_types=None, max_images=None, random_sample=False, seed=None):
        """Combine selected feature types into a single feature matrix."""
        if feature_types is None:
            feature_types = ['dino', 'fourier', 'el2n', 'sam']

        print(f"\n{'='*60}")
        print("COMBINING FEATURES")
        print(f"{'='*60}")
        print(f"Selected feature types: {feature_types}")

        # Map feature names to arrays
        feature_map = {
            'dino': self.dino_features,
            'fourier': self.fourier_features,
            'el2n': self.el2n_scores,
            'sam': self.sam_scores
        }

        # Collect valid features
        features_to_combine = []
        feature_dims = []

        for feat_name in feature_types:
            feat = feature_map.get(feat_name)
            if feat is not None:
                features_to_combine.append(feat)
                feature_dims.append((feat_name, feat.shape[1]))
                print(f"  Including {feat_name}: {feat.shape[1]} dims")
            else:
                print(f"  Skipping {feat_name}: not loaded")

        if not features_to_combine:
            raise ValueError("No features available to combine!")

        # Concatenate features
        combined = np.hstack(features_to_combine)
        total_dims = combined.shape[1]

        print(f"\nCombined feature matrix: {combined.shape}")
        print(f"Total dimensions: {total_dims}")

        # Sample if needed
        n_samples = len(combined)
        indices = np.arange(n_samples)

        if max_images is not None and max_images < n_samples:
            if random_sample:
                if seed is not None:
                    np.random.seed(seed)
                indices = np.random.choice(n_samples, max_images, replace=False)
                indices = np.sort(indices)
                print(f"\nRandomly sampled {max_images} images (seed={seed})")
            else:
                indices = indices[:max_images]
                print(f"\nSelected first {max_images} images")

            combined = combined[indices]

        # Handle NaN/Inf values
        nan_mask = np.isnan(combined).any(axis=1) | np.isinf(combined).any(axis=1)
        if nan_mask.any():
            print(f"  Removing {nan_mask.sum()} samples with NaN/Inf values")
            combined = combined[~nan_mask]
            indices = indices[~nan_mask]

        print(f"\nFinal feature matrix: {combined.shape}")

        return combined, indices, feature_dims

    def perform_clustering(self, features, n_clusters=50):
        """Perform K-Means clustering on features."""
        print(f"\n{'='*60}")
        print("K-MEANS CLUSTERING")
        print(f"{'='*60}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of samples: {len(features)}")

        # Normalize features
        print("Normalizing features...")
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        n_samples = len(features)
        if n_clusters > n_samples:
            print(f"WARNING: n_clusters ({n_clusters}) > n_samples ({n_samples}). Reducing.")
            n_clusters = n_samples

        # Use MiniBatchKMeans for large datasets
        print("Running K-Means...")
        if len(features) > 1000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=3)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        cluster_labels = kmeans.fit_predict(features_norm)
        cluster_centers = kmeans.cluster_centers_

        # Statistics
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        print(f"\nCluster statistics:")
        print(f"  Min size: {cluster_sizes.min()}")
        print(f"  Max size: {cluster_sizes.max()}")
        print(f"  Mean size: {cluster_sizes.mean():.1f}")
        print(f"  Std size: {cluster_sizes.std():.1f}")

        return {
            'labels': cluster_labels,
            'centers': cluster_centers,
            'features_norm': features_norm,
            'scaler': scaler,
            'kmeans': kmeans,
            'inertia': kmeans.inertia_,
            'cluster_sizes': cluster_sizes
        }

    def compute_pca_3d(self, features):
        """Compute PCA reduction to 3 components."""
        print(f"\n{'='*60}")
        print("PCA DIMENSIONALITY REDUCTION")
        print(f"{'='*60}")
        print(f"Input dimensions: {features.shape[1]}")
        print(f"Output dimensions: 3")

        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)

        explained_var = pca.explained_variance_ratio_
        total_var = sum(explained_var) * 100

        print(f"\nExplained Variance:")
        print(f"  PC1: {explained_var[0]*100:.2f}%")
        print(f"  PC2: {explained_var[1]*100:.2f}%")
        print(f"  PC3: {explained_var[2]*100:.2f}%")
        print(f"  ─────────────────")
        print(f"  Total: {total_var:.2f}%")

        return {
            'features_3d': features_3d,
            'pca': pca,
            'explained_variance': explained_var,
            'total_variance': total_var
        }

    def create_3d_figure(self, features_3d, labels, pca_result, n_clusters,
                         elev=30, azim=45, title_suffix=""):
        """Create a single 3D scatter plot figure."""
        explained_var = pca_result['explained_variance']

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Color mapping
        if n_clusters <= 50:
            cmap = plt.cm.get_cmap('gist_ncar', n_clusters) if n_clusters > 20 else plt.cm.get_cmap('tab20', n_clusters)
            scatter = ax.scatter(
                features_3d[:, 0],
                features_3d[:, 1],
                features_3d[:, 2],
                c=labels,
                cmap=cmap,
                s=8,
                alpha=0.7,
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Cluster ID', fontsize=12)
        else:
            # For many clusters, color by cluster size
            cluster_sizes = np.bincount(labels, minlength=n_clusters)
            sizes_per_point = cluster_sizes[labels]
            scatter = ax.scatter(
                features_3d[:, 0],
                features_3d[:, 1],
                features_3d[:, 2],
                c=sizes_per_point,
                cmap='viridis',
                s=6,
                alpha=0.6,
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Cluster Size', fontsize=12)

        # Labels (without variance percentages for cleaner look)
        ax.set_xlabel('PC1', fontsize=14, labelpad=10)
        ax.set_ylabel('PC2', fontsize=14, labelpad=10)
        ax.set_zlabel('PC3', fontsize=14, labelpad=10)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Title (clean, without variance)
        title = f'3D PCA Feature Space with K-Means Clustering\n'
        title += f'{len(features_3d):,} images, {n_clusters} clusters'
        if title_suffix:
            title += f'\n{title_suffix}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Grid styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

        return fig, ax

    def _create_multi_panel_figure(self, features_3d, labels, pca_result, n_clusters, output_path):
        """Create a multi-panel figure with different viewing angles."""
        print("\nGenerating multi-panel figure...")

        explained_var = pca_result['explained_variance']

        fig = plt.figure(figsize=(20, 15))

        # 4 different viewing angles
        views = [
            (30, 45, "Front-Right"),
            (30, 135, "Front-Left"),
            (60, 45, "Top-Right"),
            (0, 0, "Front")
        ]

        if n_clusters <= 50:
            cmap = plt.cm.get_cmap('gist_ncar', n_clusters) if n_clusters > 20 else plt.cm.get_cmap('tab20', n_clusters)
        else:
            cmap = 'viridis'

        for i, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')

            if n_clusters <= 50:
                scatter = ax.scatter(
                    features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                    c=labels, cmap=cmap, s=5, alpha=0.6, edgecolors='none'
                )
            else:
                cluster_sizes = np.bincount(labels, minlength=n_clusters)
                sizes_per_point = cluster_sizes[labels]
                scatter = ax.scatter(
                    features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                    c=sizes_per_point, cmap=cmap, s=4, alpha=0.5, edgecolors='none'
                )

            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.set_zlabel('PC3', fontsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'({chr(97+i)}) {view_name} View', fontsize=14, fontweight='bold')

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        fig.suptitle(
            f'3D PCA Feature Space - Multiple Views\n'
            f'{len(features_3d):,} images, {n_clusters} clusters',
            fontsize=18, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()

    def _create_statistics_figure(self, features_3d, labels, pca_result, cluster_result,
                                   n_clusters, feature_dims, output_path):
        """Create statistics figure with cluster analysis."""
        print("\nGenerating statistics figure...")

        explained_var = pca_result['explained_variance']
        cluster_sizes = np.bincount(labels, minlength=n_clusters)

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        # (a) 3D scatter - main view
        ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
        if n_clusters <= 50:
            cmap = plt.cm.get_cmap('gist_ncar', n_clusters) if n_clusters > 20 else plt.cm.get_cmap('tab20', n_clusters)
            scatter = ax1.scatter(
                features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                c=labels, cmap=cmap, s=6, alpha=0.6, edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
            cbar.set_label('Cluster ID')
        else:
            sizes_per_point = cluster_sizes[labels]
            scatter = ax1.scatter(
                features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                c=sizes_per_point, cmap='viridis', s=5, alpha=0.5, edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
            cbar.set_label('Cluster Size')

        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.view_init(elev=30, azim=45)
        ax1.set_title('(a) 3D PCA Space with K-Means Clustering', fontweight='bold')

        # (b) Cluster size distribution (moved here for better layout)
        ax2 = fig.add_subplot(gs[0, 2])
        if n_clusters <= 50:
            colors_bar = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max(n_clusters, 2)))
            ax2.bar(range(n_clusters), cluster_sizes, color=colors_bar[:n_clusters], edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Cluster ID')
        else:
            ax2.hist(cluster_sizes[cluster_sizes > 0], bins=50, color='steelblue', edgecolor='black', alpha=0.8)
            ax2.axvline(float(np.mean(cluster_sizes)), color='red', linestyle='--', label=f'Mean: {np.mean(cluster_sizes):.0f}')
            ax2.axvline(float(np.median(cluster_sizes)), color='orange', linestyle=':', label=f'Median: {np.median(cluster_sizes):.0f}')
            ax2.set_xlabel('Cluster Size')
            ax2.legend()
        ax2.set_ylabel('Count')
        ax2.set_title('(b) Cluster Size Distribution', fontweight='bold')

        # (c) Cluster size rank plot
        ax3 = fig.add_subplot(gs[0, 3])
        sizes_sorted = np.sort(cluster_sizes[cluster_sizes > 0])[::-1]
        ax3.plot(range(1, len(sizes_sorted) + 1), sizes_sorted, 'k-', linewidth=2)
        ax3.fill_between(range(1, len(sizes_sorted) + 1), sizes_sorted, alpha=0.3)
        ax3.set_yscale('log')
        ax3.set_xlabel('Cluster Rank')
        ax3.set_ylabel('Cluster Size (log)')
        ax3.set_title('(c) Cluster Size Rank Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # (d) PC1 vs PC2 projection
        ax4 = fig.add_subplot(gs[1, 0])
        if n_clusters <= 50:
            ax4.scatter(features_3d[:, 0], features_3d[:, 1], c=labels, cmap=cmap, s=3, alpha=0.5)
        else:
            ax4.scatter(features_3d[:, 0], features_3d[:, 1], c=cluster_sizes[labels],
                       cmap='viridis', s=2, alpha=0.4)
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC2')
        ax4.set_title('(d) PC1 vs PC2 Projection', fontweight='bold')

        # (e) PC1 vs PC3 projection
        ax5 = fig.add_subplot(gs[1, 1])
        if n_clusters <= 50:
            ax5.scatter(features_3d[:, 0], features_3d[:, 2], c=labels, cmap=cmap, s=3, alpha=0.5)
        else:
            ax5.scatter(features_3d[:, 0], features_3d[:, 2], c=cluster_sizes[labels],
                       cmap='viridis', s=2, alpha=0.4)
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC3')
        ax5.set_title('(e) PC1 vs PC3 Projection', fontweight='bold')

        # (f) PC2 vs PC3 projection
        ax6 = fig.add_subplot(gs[1, 2])
        if n_clusters <= 50:
            ax6.scatter(features_3d[:, 1], features_3d[:, 2], c=labels, cmap=cmap, s=3, alpha=0.5)
        else:
            ax6.scatter(features_3d[:, 1], features_3d[:, 2], c=cluster_sizes[labels],
                       cmap='viridis', s=2, alpha=0.4)
        ax6.set_xlabel('PC2')
        ax6.set_ylabel('PC3')
        ax6.set_title('(f) PC2 vs PC3 Projection', fontweight='bold')

        # (g) Distance to centroid histogram
        ax7 = fig.add_subplot(gs[1, 3])
        # Calculate distances to cluster centroids
        cluster_centers = pca_result['pca'].transform(
            np.vstack([cluster_result['centers'][i] for i in range(n_clusters)])
        )
        distances = np.linalg.norm(features_3d - cluster_centers[labels], axis=1)
        ax7.hist(distances, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax7.axvline(float(np.mean(distances)), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}')
        ax7.set_xlabel('Distance to Centroid')
        ax7.set_ylabel('Count')
        ax7.set_title('(g) Intra-cluster Distances', fontweight='bold')
        ax7.legend(fontsize=9)

        total_dims = sum(f[1] for f in feature_dims)
        fig.suptitle(
            f'3D PCA Feature Clustering Analysis ({total_dims} dims → 3 dims)\n'
            f'{len(features_3d):,} images, {n_clusters} clusters',
            fontsize=18, fontweight='bold', y=0.99
        )

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()

    def visualize_3d_clustering(self, output_dir, n_clusters=50, max_images=None,
                                 feature_types=None, random_sample=False, seed=None,
                                 angles=None, interactive=False, multi_view=False):
        """Main visualization function."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("3D CLUSTERING VISUALIZATION")
        print(f"{'='*60}")

        # Load cached features
        self.load_cached_features()

        # Combine features
        combined_features, indices, feature_dims = self.combine_features(
            feature_types=feature_types,
            max_images=max_images,
            random_sample=random_sample,
            seed=seed
        )

        # Perform clustering
        cluster_result = self.perform_clustering(combined_features, n_clusters=n_clusters)
        labels = cluster_result['labels']

        # Compute 3D PCA
        pca_result = self.compute_pca_3d(cluster_result['features_norm'])
        features_3d = pca_result['features_3d']

        # Default angles if not specified
        if angles is None:
            if multi_view:
                angles = [(30, 45), (30, 135), (30, 225), (30, 315), (60, 45), (0, 0)]
            else:
                angles = [(30, 45)]

        # Generate visualizations
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")

        saved_files = []

        for i, (elev, azim) in enumerate(angles):
            print(f"\nGenerating view {i+1}/{len(angles)}: elev={elev}, azim={azim}")

            fig, ax = self.create_3d_figure(
                features_3d, labels, pca_result, n_clusters,
                elev=elev, azim=azim,
                title_suffix=f"View: elevation={elev}, azimuth={azim}" if len(angles) > 1 else ""
            )

            if len(angles) == 1:
                output_path = output_dir / "clustering_3d.png"
            else:
                output_path = output_dir / f"clustering_3d_view{i+1}_e{elev}_a{azim}.png"

            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_path}")
            saved_files.append(output_path)

            if not interactive:
                plt.close()

        # Generate combined multi-panel figure
        if multi_view and len(angles) >= 4:
            self._create_multi_panel_figure(
                features_3d, labels, pca_result, n_clusters,
                output_dir / "clustering_3d_multi_panel.png"
            )
            saved_files.append(output_dir / "clustering_3d_multi_panel.png")

        # Generate statistics figure
        self._create_statistics_figure(
            features_3d, labels, pca_result, cluster_result, n_clusters, feature_dims,
            output_dir / "clustering_3d_statistics.png"
        )
        saved_files.append(output_dir / "clustering_3d_statistics.png")

        # Interactive mode
        if interactive:
            print("\n[Interactive mode] Rotate the plot with your mouse. Close window to exit.")
            plt.show()

        print(f"\n{'='*60}")
        print(f"DONE! Saved {len(saved_files)} files to: {output_dir}")
        print(f"{'='*60}")

        return {
            'features_3d': features_3d,
            'labels': labels,
            'pca_result': pca_result,
            'cluster_result': cluster_result,
            'indices': indices,
            'feature_dims': feature_dims,
            'saved_files': saved_files
        }


def main():
    parser = argparse.ArgumentParser(
        description='3D Feature Clustering Visualization for Scientific Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 3D visualization with 50 clusters (uses all cached features)
  py -3.11 visualize_clustering_3d.py --n_clusters 50

  # Limit to 20k images
  py -3.11 visualize_clustering_3d.py --num_images 20000 --n_clusters 50

  # Generate multiple viewing angles
  py -3.11 visualize_clustering_3d.py --n_clusters 50 --multi_view

  # Custom viewing angles (elev,azim pairs)
  py -3.11 visualize_clustering_3d.py --angles 30,45 60,90 0,0

  # Interactive mode (rotate with mouse)
  py -3.11 visualize_clustering_3d.py --n_clusters 20 --interactive

  # Only DINO features
  py -3.11 visualize_clustering_3d.py --features dino

  # DINO + Fourier features only
  py -3.11 visualize_clustering_3d.py --features dino fourier

  # Random sample with seed
  py -3.11 visualize_clustering_3d.py --num_images 5000 --random --seed 42
        """
    )

    # Feature selection
    feature_group = parser.add_argument_group('Feature Selection')
    feature_group.add_argument('--features', nargs='+', type=str,
        default=['dino', 'fourier', 'el2n', 'sam'],
        choices=['dino', 'fourier', 'el2n', 'sam'],
        help='Feature types to use (default: all)')
    feature_group.add_argument('--cache_dir', type=str, default=None,
        help=f'Feature cache directory (default: {DEFAULT_CACHE_DIR})')

    # Sample selection
    sample_group = parser.add_argument_group('Sample Selection')
    sample_group.add_argument('--num_images', type=int, default=None,
        help='Number of images to use (default: all)')
    sample_group.add_argument('--random', action='store_true',
        help='Randomly sample images')
    sample_group.add_argument('--seed', type=int, default=None,
        help='Random seed for reproducibility')

    # Clustering options
    clustering_group = parser.add_argument_group('Clustering Options')
    clustering_group.add_argument('--n_clusters', type=int, default=50,
        help='Number of K-Means clusters (default: 50)')

    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument('--angles', nargs='+', type=str, default=None,
        help='Viewing angles as "elev,azim" pairs (e.g., 30,45 60,90)')
    viz_group.add_argument('--interactive', action='store_true',
        help='Interactive mode - rotate plot with mouse')
    viz_group.add_argument('--multi_view', action='store_true',
        help='Generate multiple standard viewing angles')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    output_group.add_argument('--show', action='store_true',
        help='Show plots interactively after saving')

    args = parser.parse_args()

    # Parse viewing angles
    angles = None
    if args.angles:
        angles = []
        for angle_str in args.angles:
            elev, azim = map(int, angle_str.split(','))
            angles.append((elev, azim))
        print(f"Custom viewing angles: {angles}")

    # Initialize visualizer and run
    visualizer = Clustering3DVisualizer(cache_dir=args.cache_dir)

    result = visualizer.visualize_3d_clustering(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        max_images=args.num_images,
        feature_types=args.features,
        random_sample=args.random,
        seed=args.seed,
        angles=angles,
        interactive=args.interactive,
        multi_view=args.multi_view
    )

    if args.show and result:
        print("\n[Show mode] Displaying final plot...")
        plt.show()


if __name__ == '__main__':
    main()
