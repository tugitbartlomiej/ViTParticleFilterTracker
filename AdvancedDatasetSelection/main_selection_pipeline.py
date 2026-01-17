"""
Advanced Dataset Selection Pipeline for DETR Fine-tuning.

Main orchestrator that combines all selection methods:
- DINO semantic features (1024-dim from DINOv3)
- SAM segmentation complexity (proxy score)
- Fourier frequency analysis (9-dim with directional features)
- EL2N difficulty scoring (from trained DETR model)

Selection Methods:
- 'cluster': Cluster-based selection (recommended, based on ELFS/CCS literature)
- 'kcenter': Legacy k-Center Greedy + EL2N ranking

Usage:
    python main_selection_pipeline.py --config config.yaml --target 50
    python main_selection_pipeline.py --config config.yaml --target 50 --method cluster
"""

import os
import sys
import warnings

# Suppress PyTorch weight loading warnings
warnings.filterwarnings('ignore', message='.*copying from a non-meta parameter.*')
warnings.filterwarnings('ignore', message='.*were not used when initializing.*')
warnings.filterwarnings('ignore', message='.*were not initialized from the model checkpoint.*')
warnings.filterwarnings('ignore', message='.*Using `TRANSFORMERS_CACHE`.*')

import yaml
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup model cache BEFORE importing torch/transformers
from External.model_cache_config import setup_model_cache
setup_model_cache()

from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
from AdvancedDatasetSelection.feature_extractors.dino_extractor import DINOExtractor
from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor
from AdvancedDatasetSelection.selection_methods.el2n_scorer import EL2NScorer
from AdvancedDatasetSelection.selection_methods.k_center_greedy import KCenterGreedy
from AdvancedDatasetSelection.selection_methods.combined_selector import CombinedSelector
from AdvancedDatasetSelection.selection_methods.cluster_selector import ClusterBasedSelector
from AdvancedDatasetSelection.utils.coco_handler import COCOHandler
from AdvancedDatasetSelection.utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedDatasetSelectionPipeline:
    """Main pipeline for advanced dataset selection."""

    def __init__(self, config_path: str):
        """
        Initialize pipeline from config file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.config_path = Path(config_path)

        # Initialize components
        self._init_extractors()
        self._init_selectors()
        self._init_utils()

        # Results storage
        self.results = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config

    def _init_extractors(self):
        """Initialize feature extractors."""
        # Fourier analyzer
        fourier_config = self.config.get('fourier', {})
        self.fourier = FourierAnalyzer(
            similarity_threshold=fourier_config.get('similarity_threshold', 0.85)
        )

        # DINO extractor (supports local HuggingFace model or torch.hub)
        dino_config = self.config.get('models', {}).get('dino', {})

        # Resolve relative paths
        dino_model_path = dino_config.get('model_path')
        if dino_model_path and not os.path.isabs(dino_model_path):
            dino_model_path = os.path.join(os.path.dirname(self.config_path), dino_model_path)

        dino_cache_dir = dino_config.get('cache_dir')
        if dino_cache_dir and not os.path.isabs(dino_cache_dir):
            dino_cache_dir = os.path.join(os.path.dirname(self.config_path), dino_cache_dir)

        self.dino = DINOExtractor(
            model_name=dino_config.get('model_name', 'dinov2_vitl14'),
            device=dino_config.get('device', 'cuda'),
            cache_dir=dino_cache_dir,
            model_path=dino_model_path
        )

        # SAM3 extractor
        sam_config = self.config.get('models', {}).get('sam', {})
        sam_model_path = sam_config.get('model_path')
        if sam_model_path and not os.path.isabs(sam_model_path):
            sam_model_path = os.path.join(os.path.dirname(self.config_path), sam_model_path)

        self.sam = SAMExtractor(
            model_path=sam_model_path,
            device=sam_config.get('device', 'cuda')
        )

        logger.info("Initialized feature extractors")

    def _init_selectors(self):
        """Initialize selection methods."""
        selection_config = self.config.get('selection', {})

        # Get DETR checkpoint path from config
        detr_config = self.config.get('models', {}).get('detr', {})
        detr_checkpoint = detr_config.get('checkpoint')

        # Resolve relative path
        if detr_checkpoint and not os.path.isabs(detr_checkpoint):
            detr_checkpoint = os.path.join(os.path.dirname(self.config_path), detr_checkpoint)

        # Get FastSAM path from config (preferred over SAM3)
        fastsam_config = self.config.get('models', {}).get('fastsam', {})
        fastsam_path = fastsam_config.get('model_path')
        if fastsam_path and not os.path.isabs(fastsam_path):
            fastsam_path = os.path.join(os.path.dirname(self.config_path), fastsam_path)
        # Check if FastSAM model exists
        if fastsam_path and os.path.exists(fastsam_path):
            logger.info(f"FastSAM model found: {fastsam_path}")
        else:
            logger.warning(f"FastSAM model not found at {fastsam_path}, will use proxy method")
            fastsam_path = None

        # Cache directory
        cache_dir = self.config.get('processing', {}).get('cache_dir')
        if cache_dir and not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(self.config_path), cache_dir)

        # Selection method from config (default: cluster for new approach)
        self.selection_method = selection_config.get('method', 'cluster')

        # Initialize ClusterBasedSelector (new, recommended)
        self.cluster_selector = ClusterBasedSelector(
            fourier_analyzer=self.fourier,
            dino_extractor=self.dino,
            sam_extractor=self.sam,
            fastsam_path=fastsam_path,
            detr_checkpoint_path=detr_checkpoint,
            detr_device=detr_config.get('device', 'cuda'),
            weights=self.config.get('weights'),
            cache_dir=cache_dir
        )

        # Initialize legacy CombinedSelector (k-Center + EL2N ranking)
        el2n_config = selection_config.get('el2n', {})
        self.el2n = EL2NScorer(
            proxy_epochs=el2n_config.get('proxy_epochs', 20),
            batch_size=el2n_config.get('batch_size', 16)
        )

        k_center_config = selection_config.get('k_center', {})
        self.k_center = KCenterGreedy(
            feature_dim=self.config.get('dino', {}).get('feature_dim', 768)
        )

        self.combined_selector = CombinedSelector(
            fourier_analyzer=self.fourier,
            dino_extractor=self.dino,
            sam_extractor=self.sam,
            el2n_scorer=self.el2n,
            k_center=self.k_center,
            weights=self.config.get('weights'),
            cache_dir=cache_dir,
            detr_checkpoint_path=detr_checkpoint,
            detr_device=detr_config.get('device', 'cuda')
        )

        # Default selector based on config
        self.selector = self.cluster_selector if self.selection_method == 'cluster' else self.combined_selector

        logger.info(f"Initialized selectors (method: {self.selection_method})")

    def _init_utils(self):
        """Initialize utility classes."""
        self.coco_handler = COCOHandler()

        output_config = self.config.get('output', {})
        output_dir = self.config.get('datasets', {}).get('output', './output')
        self.visualizer = Visualizer(output_dir=os.path.join(output_dir, 'visualizations'))

        logger.info("Initialized utilities")

    def load_datasets(self) -> Tuple[List[str], Optional[Dict]]:
        """
        Load and merge datasets.

        Returns:
            Tuple of (image_paths, merged_annotations)
        """
        datasets_config = self.config.get('datasets', {})
        all_image_paths = []
        all_annotations = []

        # Load existing dataset
        existing_path = datasets_config.get('existing')
        if existing_path and os.path.exists(existing_path):
            logger.info(f"Loading existing dataset from {existing_path}")
            image_paths = self._get_images_from_directory(existing_path)
            all_image_paths.extend(image_paths)
            logger.info(f"Found {len(image_paths)} images in existing dataset")

            # Try to load COCO annotations
            ann_path = os.path.join(existing_path, 'annotations.json')
            if os.path.exists(ann_path):
                self.coco_handler.load_annotations(ann_path)
                all_annotations.append(self.coco_handler.annotations)

        # Load new dataset
        new_path = datasets_config.get('new_source')
        if new_path and os.path.exists(new_path):
            logger.info(f"Loading new dataset from {new_path}")
            image_paths = self._get_images_from_directory(new_path)
            all_image_paths.extend(image_paths)
            logger.info(f"Found {len(image_paths)} images in new dataset")

            # Try to load COCO annotations
            ann_candidates = [
                os.path.join(new_path, 'annotations.json'),
                os.path.join(new_path, '_annotations.coco.json'),
                os.path.join(os.path.dirname(new_path), 'annotations', 'instances_train.json')
            ]
            for ann_path in ann_candidates:
                if os.path.exists(ann_path):
                    handler = COCOHandler()
                    handler.load_annotations(ann_path)
                    all_annotations.append(handler.annotations)
                    break

        # Merge annotations if multiple
        merged_annotations = None
        if len(all_annotations) > 1:
            merged_annotations = self.coco_handler.merge_datasets(all_annotations)
        elif len(all_annotations) == 1:
            merged_annotations = all_annotations[0]

        logger.info(f"Total images loaded: {len(all_image_paths)}")
        return all_image_paths, merged_annotations

    def _get_images_from_directory(self, directory: str) -> List[str]:
        """Get all image paths from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def run(self,
            image_paths: Optional[List[str]] = None,
            target_size: Optional[int] = None) -> Dict:
        """
        Run the complete selection pipeline.

        Args:
            image_paths: List of image paths (loads from config if None)
            target_size: Target size (uses config if None)

        Returns:
            Dictionary with selection results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Advanced Dataset Selection Pipeline")
        logger.info("=" * 60)

        # Load datasets if not provided
        annotations = None
        if image_paths is None:
            image_paths, annotations = self.load_datasets()

        if not image_paths:
            raise ValueError("No images found to process")

        # Get target size
        if target_size is None:
            target_size = self.config.get('output', {}).get('target_size', 5000)

        logger.info(f"Input: {len(image_paths)} images")
        logger.info(f"Target: {target_size} images")

        # Get selection parameters
        fourier_config = self.config.get('fourier', {})
        selection_config = self.config.get('selection', {})
        use_cache = self.config.get('processing', {}).get('use_cache', True)

        # Run selection based on method
        if self.selection_method == 'cluster':
            # New cluster-based approach (ELFS/CCS literature)
            strategy = selection_config.get('strategy', 'centroid')  # 'centroid', 'max_el2n', 'medoid'

            # NEW: PCA and weighting parameters for balanced feature contributions
            dino_pca_dim = selection_config.get('dino_pca_dim', 32)  # Reduce DINO from 1024 to 32 dims
            apply_weights = selection_config.get('apply_weights', True)  # Balance feature group contributions

            logger.info(f"Feature balancing: PCA={dino_pca_dim}, weights={apply_weights}")

            selected_indices, selected_paths, stats = self.cluster_selector.select_optimal_subset(
                image_paths=image_paths,
                target_size=target_size,
                strategy=strategy,
                use_cache=use_cache,
                normalize=True,
                dino_pca_dim=dino_pca_dim,
                apply_weights=apply_weights
            )
        else:
            # Legacy k-Center Greedy + EL2N ranking
            selected_indices, selected_paths, stats = self.combined_selector.select_optimal_subset(
                image_paths=image_paths,
                target_size=target_size,
                fourier_threshold=fourier_config.get('similarity_threshold', 0.85),
                oversampling=selection_config.get('k_center', {}).get('oversampling_factor', 2.0),
                keep_hard=True,
                use_cache=use_cache
            )

        # Store results
        self.results = {
            'selected_indices': selected_indices,
            'selected_paths': selected_paths,
            'statistics': stats,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'config': self.config
        }

        # Generate output
        output_dir = self.config.get('datasets', {}).get('output', './output/selected_dataset')
        self._generate_output(selected_paths, annotations, output_dir)

        # Generate visualizations
        if self.config.get('output', {}).get('generate_visualizations', True):
            self._generate_visualizations(selected_indices, output_dir)

        # Generate detailed selection report (why each image was selected)
        if self.selection_method == 'cluster':
            strategy = self.config.get('selection', {}).get('strategy', 'centroid')
            report_path = os.path.join(output_dir, 'selection_reasons.json')
            self.cluster_selector.save_selection_report(selected_indices, strategy, report_path)

        # Generate report
        self._generate_report(output_dir)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"Pipeline completed in {elapsed:.1f} seconds")
        logger.info(f"Selected {len(selected_paths)} images")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)

        return self.results

    def _generate_output(self,
                         selected_paths: List[str],
                         annotations: Optional[Dict],
                         output_dir: str):
        """Generate output dataset."""
        logger.info("Generating output dataset...")

        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy selected images (preserve original filenames for COCO annotation lookup!)
        path_mapping = {}
        for src_path in tqdm(selected_paths, desc="Copying images"):
            original_name = Path(src_path).name  # Keep original filename!
            dst_path = images_dir / original_name
            shutil.copy2(src_path, dst_path)
            path_mapping[src_path] = original_name

        # Generate annotations
        if annotations is not None:
            # Find image IDs for selected paths
            selected_image_ids = []
            for path in selected_paths:
                filename = Path(path).name
                for img_id, img_info in self.coco_handler.images.items():
                    if img_info['file_name'] == filename:
                        selected_image_ids.append(img_id)
                        break

            if selected_image_ids:
                filtered_annotations = self.coco_handler.filter_by_image_ids(selected_image_ids)
                self.coco_handler.save_annotations(
                    filtered_annotations,
                    str(output_path / 'annotations.json')
                )

        logger.info(f"Copied {len(selected_paths)} images to {images_dir}")

    def _generate_visualizations(self, selected_indices: List[int], output_dir: str):
        """Generate visualizations."""
        logger.info("Generating visualizations...")

        try:
            # PCA coverage
            if self.selector.dino_features is not None:
                self.visualizer.plot_pca_coverage(
                    self.selector.dino_features,
                    selected_indices,
                    title="DINO Feature Space Coverage"
                )

            # Difficulty distribution
            if self.selector.el2n_scores is not None:
                scores_dict = {f"img_{i}": float(self.selector.el2n_scores[i])
                              for i in range(len(self.selector.el2n_scores))}
                selected_keys = [f"img_{i}" for i in selected_indices]
                self.visualizer.plot_difficulty_distribution(
                    scores_dict, selected_keys
                )

            # SAM complexity
            if self.selector.sam_scores is not None:
                self.visualizer.plot_sam_complexity_distribution(
                    list(self.selector.sam_scores),
                    selected_indices
                )

            # Fourier diversity
            if self.selector.fourier_features is not None:
                self.visualizer.plot_fourier_diversity(
                    self.selector.fourier_features
                )

            # Cluster visualization (for cluster-based method)
            if self.selection_method == 'cluster' and hasattr(self.cluster_selector, 'cluster_labels'):
                if self.cluster_selector.cluster_labels is not None:
                    self.visualizer.plot_cluster_visualization(
                        self.cluster_selector.get_cluster_info(),
                        selected_indices
                    )

            # Selection summary
            if 'statistics' in self.results:
                stats = self.results['statistics']
                if self.selection_method == 'cluster':
                    # Cluster-based stages
                    summary_stats = {
                        'stages': ['Original', 'Valid', 'Clusters', 'Final'],
                        'counts': [
                            stats.get('original_count', 0),
                            stats.get('valid_count', 0),
                            stats.get('n_clusters', 0),
                            stats.get('final_count', 0)
                        ]
                    }
                else:
                    # Legacy k-center stages
                    summary_stats = {
                        'stages': ['Original', 'Valid', 'After Fourier', 'After k-Center', 'Final'],
                        'counts': [
                            stats.get('original_count', 0),
                            stats.get('valid_count', 0),
                            stats.get('after_fourier', 0),
                            stats.get('after_k_center', 0),
                            stats.get('final_count', 0)
                        ]
                    }
                self.visualizer.plot_selection_summary(summary_stats)

            logger.info("Visualizations saved")

        except Exception as e:
            logger.warning(f"Could not generate all visualizations: {e}")

    def _generate_report(self, output_dir: str):
        """Generate selection report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'statistics': self.results.get('statistics', {}),
            'selection_parameters': {
                'fourier_threshold': self.config.get('fourier', {}).get('similarity_threshold'),
                'k_center_oversampling': self.config.get('selection', {}).get('k_center', {}).get('oversampling_factor'),
                'weights': self.config.get('weights', {})
            },
            'feature_statistics': {}
        }

        # Add feature statistics
        if self.selector.fourier_features is not None:
            report['feature_statistics']['fourier'] = self.fourier.analyze_diversity(
                self.selector.fourier_features
            )

        if self.selector.el2n_scores is not None:
            selected_el2n = self.selector.el2n_scores[self.results.get('selected_indices', [])]
            report['feature_statistics']['el2n'] = {
                'mean': float(np.mean(selected_el2n)),
                'std': float(np.std(selected_el2n)),
                'min': float(np.min(selected_el2n)),
                'max': float(np.max(selected_el2n))
            }

        # Save report
        report_path = os.path.join(output_dir, 'selection_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Advanced Dataset Selection Pipeline for DETR Fine-tuning'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--target-size', '-t',
        type=int,
        default=None,
        help='Override target size from config'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['cluster', 'kcenter'],
        default=None,
        help='Selection method: cluster (new, recommended) or kcenter (legacy)'
    )
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['centroid', 'max_el2n', 'medoid'],
        default=None,
        help='Cluster representative strategy (only for cluster method)'
    )

    args = parser.parse_args()

    # Check config exists
    if not os.path.exists(args.config):
        # Try default location
        default_config = Path(__file__).parent / 'config.yaml'
        if default_config.exists():
            args.config = str(default_config)
        else:
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

    # Run pipeline
    pipeline = AdvancedDatasetSelectionPipeline(args.config)

    # Override output if specified
    if args.output:
        pipeline.config['datasets']['output'] = args.output

    # Override method if specified
    if args.method:
        pipeline.selection_method = args.method
        pipeline.selector = pipeline.cluster_selector if args.method == 'cluster' else pipeline.combined_selector
        pipeline.config.setdefault('selection', {})['method'] = args.method
        logger.info(f"Selection method overridden to: {args.method}")

    # Override strategy if specified (only for cluster method)
    if args.strategy:
        pipeline.config.setdefault('selection', {})['strategy'] = args.strategy
        logger.info(f"Selection strategy overridden to: {args.strategy}")

    results = pipeline.run(target_size=args.target_size)

    print(f"\nSelection complete!")
    print(f"  Input: {results['statistics']['original_count']} images")
    print(f"  Output: {results['statistics']['final_count']} images")
    print(f"  Reduction: {results['statistics']['reduction_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
