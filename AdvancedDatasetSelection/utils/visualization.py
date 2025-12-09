"""
Visualization utilities for dataset selection pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization tools for dataset selection analysis."""

    def __init__(self, output_dir: str = "./output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pca_coverage(self,
                          features: np.ndarray,
                          selected_indices: List[int],
                          title: str = "Feature Space Coverage (PCA)",
                          save_name: str = "pca_coverage.png"):
        """Plot 2D PCA visualization of feature coverage."""
        from sklearn.decomposition import PCA

        # Reduce to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot all points
        ax.scatter(features_2d[:, 0], features_2d[:, 1],
                   c='lightgray', alpha=0.3, s=10, label='All samples')

        # Highlight selected points
        selected_mask = np.zeros(len(features), dtype=bool)
        selected_mask[selected_indices] = True
        ax.scatter(features_2d[selected_mask, 0], features_2d[selected_mask, 1],
                   c='red', alpha=0.8, s=30, label='Selected samples')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved PCA coverage plot to {save_path}")

    def plot_difficulty_distribution(self,
                                     scores: Dict[str, float],
                                     selected_keys: Optional[List[str]] = None,
                                     title: str = "EL2N Difficulty Score Distribution",
                                     save_name: str = "difficulty_distribution.png"):
        """Plot histogram of difficulty scores."""
        all_scores = list(scores.values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full distribution
        axes[0].hist(all_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('EL2N Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title('All Samples')
        axes[0].axvline(np.mean(all_scores), color='red', linestyle='--',
                        label=f'Mean: {np.mean(all_scores):.3f}')
        axes[0].legend()

        # Selected distribution
        if selected_keys:
            selected_scores = [scores[k] for k in selected_keys if k in scores]
            axes[1].hist(selected_scores, bins=30, color='coral', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('EL2N Score')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Selected Samples')
            axes[1].axvline(np.mean(selected_scores), color='red', linestyle='--',
                            label=f'Mean: {np.mean(selected_scores):.3f}')
            axes[1].legend()

        fig.suptitle(title)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved difficulty distribution plot to {save_path}")

    def plot_fourier_diversity(self,
                               fourier_features: np.ndarray,
                               labels: Optional[List[str]] = None,
                               save_name: str = "fourier_diversity.png"):
        """Plot Fourier feature diversity analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Feature dimensions assumed: [low_energy, mid_energy, high_energy, entropy, centroid]
        feature_names = ['Low Band Energy', 'Mid Band Energy', 'High Band Energy',
                        'Spectral Entropy', 'Frequency Centroid']

        # Energy distribution
        if fourier_features.shape[1] >= 3:
            energy_data = fourier_features[:, :3]
            axes[0, 0].boxplot(energy_data, labels=feature_names[:3])
            axes[0, 0].set_title('Frequency Band Energy Distribution')
            axes[0, 0].set_ylabel('Energy')

        # Entropy histogram
        if fourier_features.shape[1] >= 4:
            axes[0, 1].hist(fourier_features[:, 3], bins=40, color='purple', alpha=0.7)
            axes[0, 1].set_xlabel('Spectral Entropy')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Spectral Entropy Distribution')

        # Scatter: Low vs High energy
        if fourier_features.shape[1] >= 3:
            scatter = axes[1, 0].scatter(fourier_features[:, 0], fourier_features[:, 2],
                                          c=fourier_features[:, 1], cmap='viridis',
                                          alpha=0.5, s=10)
            axes[1, 0].set_xlabel('Low Band Energy')
            axes[1, 0].set_ylabel('High Band Energy')
            axes[1, 0].set_title('Low vs High Frequency (colored by Mid)')
            plt.colorbar(scatter, ax=axes[1, 0], label='Mid Band Energy')

        # Similarity matrix sample (if small enough)
        if len(fourier_features) <= 500:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(fourier_features)
            im = axes[1, 1].imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, 1].set_title('Fourier Similarity Matrix')
            plt.colorbar(im, ax=axes[1, 1], label='Cosine Similarity')
        else:
            # Sample for visualization
            sample_idx = np.random.choice(len(fourier_features), 200, replace=False)
            sample_features = fourier_features[sample_idx]
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(sample_features)
            im = axes[1, 1].imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, 1].set_title('Fourier Similarity Matrix (200 samples)')
            plt.colorbar(im, ax=axes[1, 1], label='Cosine Similarity')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved Fourier diversity plot to {save_path}")

    def plot_sam_complexity_distribution(self,
                                         complexity_scores: List[float],
                                         selected_indices: Optional[List[int]] = None,
                                         save_name: str = "sam_complexity.png"):
        """Plot SAM scene complexity distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # All scores
        axes[0].hist(complexity_scores, bins=40, color='teal', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Scene Complexity Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title('All Samples - Scene Complexity')
        axes[0].axvline(np.mean(complexity_scores), color='red', linestyle='--',
                        label=f'Mean: {np.mean(complexity_scores):.3f}')
        axes[0].legend()

        # Selected scores
        if selected_indices:
            selected_scores = [complexity_scores[i] for i in selected_indices
                              if i < len(complexity_scores)]
            axes[1].hist(selected_scores, bins=30, color='orange', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Scene Complexity Score')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Selected Samples - Scene Complexity')
            if selected_scores:
                axes[1].axvline(np.mean(selected_scores), color='red', linestyle='--',
                                label=f'Mean: {np.mean(selected_scores):.3f}')
                axes[1].legend()

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved SAM complexity plot to {save_path}")

    def plot_selection_summary(self,
                               stats: Dict,
                               save_name: str = "selection_summary.png"):
        """Plot summary of selection process."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Reduction funnel
        stages = stats.get('stages', ['Original', 'After Fourier', 'After k-Center', 'Final'])
        counts = stats.get('counts', [100, 70, 50, 30])
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(stages)))

        axes[0, 0].barh(stages, counts, color=colors)
        axes[0, 0].set_xlabel('Number of Samples')
        axes[0, 0].set_title('Selection Pipeline Stages')
        for i, (stage, count) in enumerate(zip(stages, counts)):
            axes[0, 0].text(count + max(counts)*0.02, i, f'{count}', va='center')

        # Category distribution before/after
        if 'category_before' in stats and 'category_after' in stats:
            categories = list(stats['category_before'].keys())
            before = list(stats['category_before'].values())
            after = list(stats['category_after'].values())

            x = np.arange(len(categories))
            width = 0.35

            axes[0, 1].bar(x - width/2, before, width, label='Before', color='lightblue')
            axes[0, 1].bar(x + width/2, after, width, label='After', color='coral')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Category Distribution')
            axes[0, 1].legend()

        # Score distribution comparison
        if 'scores_before' in stats and 'scores_after' in stats:
            axes[1, 0].hist(stats['scores_before'], bins=30, alpha=0.5, label='Before', color='blue')
            axes[1, 0].hist(stats['scores_after'], bins=30, alpha=0.5, label='After', color='red')
            axes[1, 0].set_xlabel('Combined Score')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Score Distribution Before/After Selection')
            axes[1, 0].legend()

        # Processing time breakdown
        if 'time_breakdown' in stats:
            labels = list(stats['time_breakdown'].keys())
            times = list(stats['time_breakdown'].values())
            axes[1, 1].pie(times, labels=labels, autopct='%1.1f%%',
                          colors=plt.cm.Set3(np.linspace(0, 1, len(labels))))
            axes[1, 1].set_title('Processing Time Breakdown')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved selection summary plot to {save_path}")

    def create_sample_grid(self,
                           image_paths: List[str],
                           title: str = "Selected Samples",
                           grid_size: Tuple[int, int] = (4, 4),
                           save_name: str = "sample_grid.png"):
        """Create grid visualization of sample images."""
        import cv2

        n_samples = min(len(image_paths), grid_size[0] * grid_size[1])
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 16))
        axes = axes.flatten()

        for i in range(n_samples):
            img = cv2.imread(image_paths[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(Path(image_paths[i]).name, fontsize=8)

        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved sample grid to {save_path}")


if __name__ == "__main__":
    # Test visualizer
    viz = Visualizer()
    print(f"Visualizer initialized, output dir: {viz.output_dir}")
