"""
Dataset Mixer - COCO Format Dataset Mixing
==========================================

Mixes tooltip and background datasets in COCO format with specified proportions
for DETR mixed gentle training (70% tooltip + 30% background).

Features:
- COCO format annotation mixing
- Configurable mixing ratios
- Train/validation split
- Progress reporting integration
- Duplicate detection and handling
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from progress_reporter import ProgressReporter

def load_coco_annotations(annotations_path: Path) -> Dict[str, Any]:
    """Load COCO format annotations"""
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Validate COCO format
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in coco_data:
                raise ValueError(f"Missing required COCO key: {key}")
        
        return coco_data
    
    except Exception as e:
        raise ValueError(f"Failed to load COCO annotations from {annotations_path}: {e}")

def create_image_id_mapping(tooltip_data: Dict, background_data: Dict) -> Tuple[Dict, Dict]:
    """Create mappings to avoid image ID conflicts"""
    
    # Get max image ID from tooltip data
    max_tooltip_id = max([img['id'] for img in tooltip_data['images']], default=0)
    
    # Create mapping for background images (shift IDs)
    bg_image_mapping = {}
    for img in background_data['images']:
        old_id = img['id']
        new_id = max_tooltip_id + old_id + 1
        bg_image_mapping[old_id] = new_id
    
    # Create mapping for background annotations
    bg_ann_mapping = {}
    max_tooltip_ann_id = max([ann['id'] for ann in tooltip_data['annotations']], default=0)
    
    for i, ann in enumerate(background_data['annotations']):
        old_id = ann['id']
        new_id = max_tooltip_ann_id + i + 1
        bg_ann_mapping[old_id] = new_id
    
    return bg_image_mapping, bg_ann_mapping

def create_category_mapping(tooltip_data: Dict, background_data: Dict) -> Dict[int, int]:
    """Create category ID mapping to avoid conflicts"""
    
    # Get tooltip categories
    tooltip_categories = {cat['name']: cat['id'] for cat in tooltip_data['categories']}
    
    # Create mapping for background categories
    category_mapping = {}
    next_cat_id = max(tooltip_categories.values()) + 1
    
    for bg_cat in background_data['categories']:
        bg_name = bg_cat['name']
        
        if bg_name in tooltip_categories:
            # Category exists in tooltip dataset
            category_mapping[bg_cat['id']] = tooltip_categories[bg_name]
        else:
            # New category - assign new ID
            category_mapping[bg_cat['id']] = next_cat_id
            next_cat_id += 1
    
    return category_mapping

def mix_datasets(tooltip_data: Dict, background_data: Dict, 
                tooltip_ratio: float = 0.7, 
                background_ratio: float = 0.3,
                reporter: Optional[ProgressReporter] = None) -> Dict[str, Any]:
    """Mix tooltip and background datasets with specified ratios"""
    
    if reporter:
        reporter.start_stage("Mixing datasets")
    
    # Create ID mappings to avoid conflicts
    bg_image_mapping, bg_ann_mapping = create_image_id_mapping(tooltip_data, background_data)
    category_mapping = create_category_mapping(tooltip_data, background_data)
    
    # Calculate sample counts
    total_tooltip_images = len(tooltip_data['images'])
    total_background_images = len(background_data['images'])
    
    # Calculate how many images to take from each dataset
    total_ratio = tooltip_ratio + background_ratio
    normalized_tooltip_ratio = tooltip_ratio / total_ratio
    normalized_background_ratio = background_ratio / total_ratio
    
    # Determine final counts
    desired_total = min(
        int(total_tooltip_images / normalized_tooltip_ratio),
        int(total_background_images / normalized_background_ratio)
    )
    
    tooltip_count = int(desired_total * normalized_tooltip_ratio)
    background_count = int(desired_total * normalized_background_ratio)
    
    if reporter:
        reporter.log_info(f"Mixing {tooltip_count} tooltip images + {background_count} background images")
    
    # Sample images from tooltip dataset
    tooltip_images = random.sample(tooltip_data['images'], 
                                 min(tooltip_count, len(tooltip_data['images'])))
    tooltip_image_ids = {img['id'] for img in tooltip_images}
    
    # Sample images from background dataset  
    background_images = random.sample(background_data['images'],
                                    min(background_count, len(background_data['images'])))
    background_image_ids = {img['id'] for img in background_images}
    
    # Update background image IDs
    for img in background_images:
        img['id'] = bg_image_mapping[img['id']]
    
    # Collect tooltip annotations
    tooltip_annotations = [ann for ann in tooltip_data['annotations'] 
                          if ann['image_id'] in tooltip_image_ids]
    
    # Collect and update background annotations
    background_annotations = [ann for ann in background_data['annotations']
                             if ann['image_id'] in background_image_ids]
    
    for ann in background_annotations:
        # Update image ID
        ann['image_id'] = bg_image_mapping[ann['image_id']]
        # Update annotation ID
        ann['id'] = bg_ann_mapping[ann['id']]
        # Update category ID
        ann['category_id'] = category_mapping[ann['category_id']]
    
    # Combine categories
    mixed_categories = tooltip_data['categories'].copy()
    
    # Add new background categories
    for bg_cat in background_data['categories']:
        new_cat_id = category_mapping[bg_cat['id']]
        if new_cat_id not in [cat['id'] for cat in mixed_categories]:
            new_category = bg_cat.copy()
            new_category['id'] = new_cat_id
            mixed_categories.append(new_category)
    
    # Create mixed dataset
    mixed_data = {
        'info': {
            'description': 'Mixed DETR dataset (tooltip + background)',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'DETR Pipeline Orchestrator',
            'date_created': datetime.now().isoformat()
        },
        'licenses': tooltip_data.get('licenses', []),
        'images': tooltip_images + background_images,
        'annotations': tooltip_annotations + background_annotations,
        'categories': mixed_categories
    }
    
    if reporter:
        reporter.complete_stage(f"Mixed dataset created: {len(mixed_data['images'])} images, {len(mixed_data['annotations'])} annotations")
    
    return mixed_data

def split_dataset(mixed_data: Dict, validation_split: float = 0.2) -> Tuple[Dict, Dict]:
    """Split mixed dataset into train and validation sets"""
    
    # Get all image IDs
    image_ids = [img['id'] for img in mixed_data['images']]
    random.shuffle(image_ids)
    
    # Calculate split point
    split_point = int(len(image_ids) * (1 - validation_split))
    train_image_ids = set(image_ids[:split_point])
    val_image_ids = set(image_ids[split_point:])
    
    # Split images
    train_images = [img for img in mixed_data['images'] if img['id'] in train_image_ids]
    val_images = [img for img in mixed_data['images'] if img['id'] in val_image_ids]
    
    # Split annotations
    train_annotations = [ann for ann in mixed_data['annotations'] 
                        if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in mixed_data['annotations']
                      if ann['image_id'] in val_image_ids]
    
    # Create train dataset
    train_data = mixed_data.copy()
    train_data['images'] = train_images
    train_data['annotations'] = train_annotations
    
    # Create validation dataset
    val_data = mixed_data.copy()
    val_data['images'] = val_images
    val_data['annotations'] = val_annotations
    
    return train_data, val_data

def copy_images_to_output(image_list: List[Dict], 
                         source_tooltip_dir: Path, 
                         source_background_dir: Path,
                         output_dir: Path,
                         reporter: Optional[ProgressReporter] = None) -> List[str]:
    """Copy images from source directories to output directory"""
    
    if reporter:
        reporter.start_stage("Copying images")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    
    for i, img_info in enumerate(image_list):
        filename = img_info['file_name']
        
        # Try tooltip directory first
        source_path = source_tooltip_dir / filename
        if not source_path.exists():
            # Try background directory
            source_path = source_background_dir / filename
        
        if source_path.exists():
            output_path = output_dir / filename
            
            # Copy file if it doesn't exist or is different
            if not output_path.exists() or output_path.stat().st_size != source_path.stat().st_size:
                shutil.copy2(source_path, output_path)
                copied_files.append(str(output_path))
        else:
            if reporter:
                reporter.log_warning(f"Image not found: {filename}")
        
        if reporter and i % 50 == 0:
            reporter.update_progress(i + 1, len(image_list), "Copying images")
    
    if reporter:
        reporter.complete_stage(f"Copied {len(copied_files)} images")
    
    return copied_files

def generate_mixing_report(tooltip_data: Dict, background_data: Dict, 
                          mixed_data: Dict, train_data: Dict, val_data: Dict,
                          args) -> Dict[str, Any]:
    """Generate a detailed mixing report"""
    
    # Count annotations by category in each dataset
    def count_by_category(data):
        counts = defaultdict(int)
        for ann in data['annotations']:
            cat_id = ann['category_id']
            # Find category name
            cat_name = next((cat['name'] for cat in data['categories'] if cat['id'] == cat_id), f"category_{cat_id}")
            counts[cat_name] += 1
        return dict(counts)
    
    report = {
        'mixing_parameters': {
            'tooltip_ratio': args.tooltip_ratio,
            'background_ratio': args.background_ratio,
            'validation_split': args.validation_split
        },
        'source_datasets': {
            'tooltip': {
                'images': len(tooltip_data['images']),
                'annotations': len(tooltip_data['annotations']),
                'categories': count_by_category(tooltip_data)
            },
            'background': {
                'images': len(background_data['images']),
                'annotations': len(background_data['annotations']),
                'categories': count_by_category(background_data)
            }
        },
        'mixed_dataset': {
            'total_images': len(mixed_data['images']),
            'total_annotations': len(mixed_data['annotations']),
            'categories': count_by_category(mixed_data)
        },
        'splits': {
            'train': {
                'images': len(train_data['images']),
                'annotations': len(train_data['annotations']),
                'categories': count_by_category(train_data)
            },
            'validation': {
                'images': len(val_data['images']),
                'annotations': len(val_data['annotations']),
                'categories': count_by_category(val_data)
            }
        },
        'created_at': datetime.now().isoformat()
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Mix tooltip and background datasets for DETR training')
    
    # Input paths
    parser.add_argument('--tooltip_images', type=str, required=True,
                       help='Directory containing tooltip images')
    parser.add_argument('--tooltip_annotations', type=str, required=True,
                       help='Path to tooltip COCO annotations file')
    parser.add_argument('--background_images', type=str, required=True,
                       help='Directory containing background images')
    parser.add_argument('--background_annotations', type=str, required=True,
                       help='Path to background COCO annotations file')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for mixed dataset')
    
    # Mixing parameters
    parser.add_argument('--tooltip_ratio', type=float, default=0.7,
                       help='Ratio of tooltip images (default: 0.7)')
    parser.add_argument('--background_ratio', type=float, default=0.3,
                       help='Ratio of background images (default: 0.3)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--copy_images', action='store_true',
                       help='Copy images to output directory')
    
    args = parser.parse_args()
    
    # Initialize progress reporter
    reporter = ProgressReporter()
    
    try:
        # Set random seed
        random.seed(args.seed)
        
        # Setup paths
        tooltip_images_dir = Path(args.tooltip_images)
        background_images_dir = Path(args.background_images)
        tooltip_annotations_path = Path(args.tooltip_annotations)
        background_annotations_path = Path(args.background_annotations)
        output_dir = Path(args.output_dir)
        
        # Validate input paths
        if not tooltip_images_dir.exists():
            raise FileNotFoundError(f"Tooltip images directory not found: {tooltip_images_dir}")
        if not background_images_dir.exists():
            raise FileNotFoundError(f"Background images directory not found: {background_images_dir}")
        if not tooltip_annotations_path.exists():
            raise FileNotFoundError(f"Tooltip annotations not found: {tooltip_annotations_path}")
        if not background_annotations_path.exists():
            raise FileNotFoundError(f"Background annotations not found: {background_annotations_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        reporter.start_stage("Loading datasets")
        tooltip_data = load_coco_annotations(tooltip_annotations_path)
        background_data = load_coco_annotations(background_annotations_path)
        reporter.complete_stage("Datasets loaded successfully")
        
        # Mix datasets
        mixed_data = mix_datasets(tooltip_data, background_data, 
                                args.tooltip_ratio, args.background_ratio, reporter)
        
        # Split into train/validation
        reporter.start_stage("Splitting dataset")
        train_data, val_data = split_dataset(mixed_data, args.validation_split)
        reporter.complete_stage("Dataset split completed")
        
        # Create output directories
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        annotations_dir = output_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotations
        reporter.start_stage("Saving annotations")
        
        with open(annotations_dir / 'train_annotations.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(annotations_dir / 'val_annotations.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        
        reporter.complete_stage("Annotations saved")
        
        # Copy images if requested
        if args.copy_images:
            # Copy training images
            train_dir.mkdir(parents=True, exist_ok=True)
            copy_images_to_output(train_data['images'], 
                                tooltip_images_dir, background_images_dir,
                                train_dir, reporter)
            
            # Copy validation images  
            val_dir.mkdir(parents=True, exist_ok=True)
            copy_images_to_output(val_data['images'],
                                tooltip_images_dir, background_images_dir, 
                                val_dir, reporter)
        
        # Generate and save report
        reporter.start_stage("Generating report")
        mixing_report = generate_mixing_report(tooltip_data, background_data,
                                             mixed_data, train_data, val_data, args)
        
        with open(output_dir / 'mixing_report.json', 'w') as f:
            json.dump(mixing_report, f, indent=2)
        
        reporter.complete_stage("Report generated")
        
        # Print summary
        reporter.log_info(f"Mixed dataset created successfully:")
        reporter.log_info(f"  - Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
        reporter.log_info(f"  - Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
        reporter.log_info(f"  - Output: {output_dir}")
        
        reporter.report_completion("Dataset mixing completed successfully")
        
    except Exception as e:
        reporter.report_error(f"Dataset mixing failed: {str(e)}", recoverable=False)
        raise

if __name__ == '__main__':
    main()