"""
Advanced Dataset Selection Pipeline for DETR Fine-tuning.

Main orchestrator that combines all selection methods:
- DINO semantic features
- SAM segmentation complexity
- Fourier frequency analysis
- EL2N + k-Center selection

Usage:
    python main_selection_pipeline.py --config config.yaml
"""

import os
import sys
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

from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
from AdvancedDatasetSelection.feature_extractors.dino_extractor import DINOExtractor
from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor
from AdvancedDatasetSelection.selection_methods.el2n_scorer import EL2NScorer
from AdvancedDatasetSelection.selection_methods.k_center_greedy import KCenterGreedy
from AdvancedDatasetSelection.selection_methods.combined_selector import CombinedSelector
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

        # DINO extractor
        dino_config = self.config.get('models', {}).get('dino', {})
        self.dino = DINOExtractor(
            model_name=dino_config.get('model_name', 'dino_vitb16'),
            device=dino_config.get('device', 'cuda')
        )

        # SAM extractor
        sam_config = self.config.get('models', {}).get('sam', {})
        self.sam = SAMExtractor(
            checkpoint_path=sam_config.get('checkpoint'),
            model_type=sam_config.get('model_type', 'vit_h'),
            device=sam_config.get('device', 'cuda')
        )

        logger.info("Initialized feature extractors")

    def _init_selectors(self):
        """Initialize selection methods."""
        selection_config = self.config.get('selection', {})

        # EL2N scorer
        el2n_config = selection_config.get('el2n', {})
        self.el2n = EL2NScorer(
            proxy_epochs=el2n_config.get('proxy_epochs', 20),
            batch_size=el2n_config.get('batch_size', 16)
        )

        # k-Center selector
        k_center_config = selection_config.get('k_center', {})
        self.k_center = KCenterGreedy(
            feature_dim=self.config.get('dino', {}).get('feature_dim', 768)
        )

        # Combined selector
        self.selector = CombinedSelector(
            fourier_analyzer=self.fourier,
            dino_extractor=self.dino,
            sam_extractor=self.sam,
            el2n_scorer=self.el2n,
            k_center=self.k_center,
            weights=self.config.get('weights'),
            cache_dir=self.config.get('processing', {}).get('cache_dir')
        )

        logger.info("Initialized selectors")

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

        # Run combined selection
        selected_indices, selected_paths, stats = self.selector.select_optimal_subset(
            image_paths=image_paths,
            target_size=target_size,
            fourier_threshold=fourier_config.get('similarity_threshold', 0.85),
            oversampling=selection_config.get('k_center', {}).get('oversampling_factor', 2.0),
            keep_hard=True,
            use_cache=self.config.get('processing', {}).get('use_cache', True)
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

        # Copy selected images
        path_mapping = {}
        for i, src_path in enumerate(tqdm(selected_paths, desc="Copying images")):
            ext = Path(src_path).suffix
            new_name = f"img_{i:05d}{ext}"
            dst_path = images_dir / new_name
            shutil.copy2(src_path, dst_path)
            path_mapping[src_path] = new_name

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

            # Selection summary
            if 'statistics' in self.results:
                stats = self.results['statistics']
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

    results = pipeline.run(target_size=args.target_size)

    print(f"\nSelection complete!")
    print(f"  Input: {results['statistics']['original_count']} images")
    print(f"  Output: {results['statistics']['final_count']} images")
    print(f"  Reduction: {results['statistics']['reduction_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
