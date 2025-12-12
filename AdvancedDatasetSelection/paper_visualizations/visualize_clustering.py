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

    def extract_features_batch(self, image_paths, use_dino=True, show_progress=True):
        """Extract features for all images."""
        features_list = []
        valid_paths = []

        iterator = tqdm(image_paths, desc="Extracting features") if show_progress else image_paths

        if use_dino:
            self._initialize_dino()

        for img_path in iterator:
            try:
                if use_dino and self.dino_extractor is not None:
                    # Use DINO features
                    dino_feat = self.dino_extractor.extract_cls_features(img_path)

                    # Add simple features for diversity
                    simple_feat = self.extract_simple_features(img_path)
                    if simple_feat is not None:
                        combined = np.concatenate([dino_feat, simple_feat])
                    else:
                        combined = dino_feat
                else:
                    # Fallback to simple features only
                    combined = self.extract_simple_features(img_path)
                    if combined is None:
                        continue

                features_list.append(combined)
                valid_paths.append(img_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if not features_list:
            return np.array([]), []

        return np.vstack(features_list), valid_paths

    def perform_clustering(self, features, n_clusters=10):
        """Perform K-Means clustering on features."""
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

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

    def compute_tsne(self, features, perplexity=30, n_iter=1000):
        """Compute t-SNE embedding."""
        print("Computing t-SNE embedding...")

        # Use PCA first if high-dimensional
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features)
        else:
            features_pca = features

        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features) - 1),
                   n_iter=n_iter, random_state=42, init='pca')
        embedding = tsne.fit_transform(features_pca)

        return embedding

    def visualize_clustering(self, image_paths, output_path, n_clusters=10, max_images=500, show=False):
        """Create comprehensive clustering visualization."""
        # Limit number of images
        image_paths = image_paths[:max_images]
        print(f"Processing {len(image_paths)} images...")

        # Extract features
        features, valid_paths = self.extract_features_batch(image_paths, use_dino=True)
        if len(features) == 0:
            print("No features extracted!")
            return

        print(f"Extracted features for {len(valid_paths)} images, shape: {features.shape}")

        # Perform clustering
        cluster_result = self.perform_clustering(features, n_clusters=n_clusters)

        # Compute t-SNE embedding
        embedding = self.compute_tsne(cluster_result['features_norm'])

        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main t-SNE plot with cluster colors
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        scatter = ax1.scatter(embedding[:, 0], embedding[:, 1],
                             c=cluster_result['labels'], cmap='tab10',
                             alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.set_title('(a) t-SNE Feature Space with K-Means Clustering', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster ID')

        # Cluster size distribution
        ax2 = fig.add_subplot(gs[0, 2])
        unique, counts = np.unique(cluster_result['labels'], return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        ax2.bar(unique, counts, color=colors, edgecolor='black')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('(b) Cluster Size Distribution', fontweight='bold')

        # Cluster pie chart
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.pie(counts, labels=[f'C{i}' for i in unique], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax3.set_title('(c) Cluster Proportions', fontweight='bold')

        # Elbow plot simulation (inertia for different k)
        ax4 = fig.add_subplot(gs[1, 2])
        k_range = range(2, min(15, len(features) // 2))
        inertias = []
        for k in k_range:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
            km.fit(cluster_result['features_norm'])
            inertias.append(km.inertia_)
        ax4.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        ax4.axvline(x=n_clusters, color='r', linestyle='--', label=f'Selected k={n_clusters}')
        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('Inertia')
        ax4.set_title('(d) Elbow Method', fontweight='bold')
        ax4.legend()

        # Silhouette-like visualization (cluster compactness)
        ax5 = fig.add_subplot(gs[1, 3])
        # Compute average distance to centroid per cluster
        avg_distances = []
        for i in range(n_clusters):
            mask = cluster_result['labels'] == i
            if np.sum(mask) > 0:
                cluster_points = cluster_result['features_norm'][mask]
                center = cluster_result['centers'][i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_distances.append(np.mean(distances))
            else:
                avg_distances.append(0)
        ax5.barh(range(n_clusters), avg_distances, color=colors, edgecolor='black')
        ax5.set_ylabel('Cluster ID')
        ax5.set_xlabel('Avg Distance to Centroid')
        ax5.set_title('(e) Cluster Compactness', fontweight='bold')

        # Sample images from each cluster (Row 3)
        n_samples_per_cluster = 4
        for cluster_id in range(min(4, n_clusters)):
            ax = fig.add_subplot(gs[2, cluster_id])

            # Get images from this cluster
            cluster_mask = cluster_result['labels'] == cluster_id
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
                ax.set_title(f'(f{cluster_id+1}) Cluster {cluster_id}\n({np.sum(cluster_mask)} samples)',
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

    def visualize_cluster_samples(self, image_paths, cluster_result, valid_paths, output_path,
                                   n_clusters=10, samples_per_cluster=6, show=False):
        """Visualize sample images from each cluster."""
        n_cols = samples_per_cluster
        n_rows = n_clusters

        fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3 * (n_cols + 1), 3 * n_rows))

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_result['labels'] == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Cluster info cell
            axes[cluster_id, 0].text(0.5, 0.5, f'Cluster {cluster_id}\n{np.sum(cluster_mask)} images',
                                    ha='center', va='center', fontsize=12, fontweight='bold')
            axes[cluster_id, 0].axis('off')

            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(cluster_indices,
                                                  size=min(samples_per_cluster, len(cluster_indices)),
                                                  replace=False)

                for i, idx in enumerate(sample_indices):
                    img = cv2.imread(valid_paths[idx])
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[cluster_id, i + 1].imshow(img)
                    axes[cluster_id, i + 1].axis('off')

                # Fill empty cells
                for i in range(len(sample_indices), samples_per_cluster):
                    axes[cluster_id, i + 1].axis('off')
            else:
                for i in range(1, n_cols + 1):
                    axes[cluster_id, i].axis('off')

        plt.suptitle('Sample Images from Each Cluster', fontsize=16, fontweight='bold')
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
        show=args.show
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
            show=args.show
        )

    print(f"\n=== Done! Visualizations saved to: {output_dir} ===")


if __name__ == '__main__':
    main()
