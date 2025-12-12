#!/usr/bin/env python3
"""
FastSAM Segmentation Visualization for Scientific Paper

Generates publication-quality visualizations showing:
1. Automatic mask generation (all segments)
2. Individual segment analysis
3. Complexity metrics visualization
4. Segment size distribution

Usage Examples:
    # First 5 images (sequential)
    py -3.11 visualize_fastsam_segmentation.py --num_images 5

    # Random 5 images from dataset
    py -3.11 visualize_fastsam_segmentation.py --num_images 5 --random

    # Specific images by filename
    py -3.11 visualize_fastsam_segmentation.py --images image1.jpg image2.jpg

    # Show help
    py -3.11 visualize_fastsam_segmentation.py --help
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
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm import tqdm
import colorsys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_colors(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.7 + np.random.random() * 0.3
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


class FastSAMVisualizer:
    """Professional FastSAM segmentation visualization for paper figures."""

    def __init__(self, model_path=None, device='cuda'):
        self.model_path = model_path or "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/External/Models/FastSAM/FastSAM-x.pt"
        self.device = device

        # Check if model exists
        if not Path(self.model_path).exists():
            print(f"WARNING: FastSAM model not found at {self.model_path}")
            print("Will use fallback edge-based segmentation")
            self.model = None
        else:
            try:
                from ultralytics import FastSAM
                print(f"Loading FastSAM model from: {self.model_path}")
                self.model = FastSAM(self.model_path)
                print("FastSAM model loaded successfully")
            except Exception as e:
                print(f"Failed to load FastSAM: {e}")
                self.model = None

        # Set matplotlib style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def extract_masks_fastsam(self, image_path):
        """Extract masks using FastSAM."""
        if self.model is None:
            return self.extract_masks_fallback(image_path)

        try:
            results = self.model(
                image_path,
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
                verbose=False
            )

            if not results or results[0].masks is None:
                return []

            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            masks = []
            for i, mask in enumerate(masks_data):
                mask_binary = (mask > 0.5).astype(np.uint8)
                area = int(np.sum(mask_binary))

                if area < 100:  # Skip tiny masks
                    continue

                # Get bounding box
                if boxes is not None and i < len(boxes):
                    box = boxes[i].xyxy[0].cpu().numpy()
                    bbox = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
                    conf = float(boxes[i].conf[0]) if boxes[i].conf is not None else 1.0
                else:
                    rows = np.any(mask_binary, axis=1)
                    cols = np.any(mask_binary, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                    conf = 1.0

                masks.append({
                    'segmentation': mask_binary,
                    'area': area,
                    'bbox': bbox,
                    'confidence': conf
                })

            return masks

        except Exception as e:
            print(f"FastSAM error: {e}")
            return self.extract_masks_fallback(image_path)

    def extract_masks_fallback(self, image_path):
        """Fallback segmentation using edge detection and contours."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            # Create mask from contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)

            x, y, w, h = cv2.boundingRect(contour)

            masks.append({
                'segmentation': mask,
                'area': int(area),
                'bbox': [x, y, w, h],
                'confidence': 0.8
            })

        return masks

    def compute_complexity_metrics(self, masks, image_shape):
        """Compute scene complexity metrics from masks."""
        if not masks:
            return {
                'num_segments': 0,
                'avg_segment_size': 0.0,
                'segment_size_std': 0.0,
                'coverage_ratio': 0.0,
                'edge_density': 0.0,
                'complexity_score': 0.0
            }

        height, width = image_shape[:2]
        total_pixels = height * width

        areas = [m['area'] for m in masks]
        confidences = [m.get('confidence', 1.0) for m in masks]

        # Edge density
        edge_pixels = 0
        for mask in masks:
            mask_uint8 = (mask['segmentation'] * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, 100, 200)
            edge_pixels += np.sum(edges > 0)

        num_segments = len(masks)
        avg_segment_size = np.mean(areas) / total_pixels
        segment_size_std = np.std(areas) / total_pixels if len(areas) > 1 else 0
        coverage_ratio = min(sum(areas), total_pixels) / total_pixels
        edge_density = edge_pixels / total_pixels
        avg_confidence = np.mean(confidences)

        # Composite complexity score
        complexity_score = (
            0.25 * min(num_segments / 50, 1.0) +
            0.20 * segment_size_std * 10 +
            0.20 * edge_density * 10 +
            0.20 * coverage_ratio +
            0.15 * (1 - avg_confidence)
        )
        complexity_score = min(max(complexity_score, 0), 1)

        return {
            'num_segments': num_segments,
            'avg_segment_size': float(avg_segment_size),
            'segment_size_std': float(segment_size_std),
            'coverage_ratio': float(coverage_ratio),
            'edge_density': float(edge_density),
            'avg_confidence': float(avg_confidence),
            'complexity_score': float(complexity_score)
        }

    def visualize_single_image(self, image_path, output_path, show=False):
        """Create comprehensive segmentation visualization for a single image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.extract_masks_fastsam(image_path)
        metrics = self.compute_complexity_metrics(masks, image.shape)

        # Generate colors for masks
        colors = generate_colors(len(masks)) if masks else []

        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Original, All masks colored, Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('(a) Original Image', fontweight='bold')
        ax1.axis('off')

        # All masks with different colors
        ax2 = fig.add_subplot(gs[0, 1])
        mask_overlay = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
        for i, mask in enumerate(masks):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            mask_overlay[mask['segmentation'] > 0] = color
        ax2.imshow(mask_overlay)
        ax2.set_title(f'(b) Segmentation Masks ({len(masks)} segments)', fontweight='bold')
        ax2.axis('off')

        # Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = image_rgb.copy()
        for i, mask in enumerate(masks):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            mask_bool = mask['segmentation'] > 0
            overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * np.array(color)).astype(np.uint8)
        ax3.imshow(overlay)
        ax3.set_title('(c) Overlay', fontweight='bold')
        ax3.axis('off')

        # Bounding boxes
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image_rgb)
        for i, mask in enumerate(masks):
            bbox = mask['bbox']
            color = np.array(colors[i]) / 255.0 if i < len(colors) else (1, 1, 1)
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=2, edgecolor=color, facecolor='none')
            ax4.add_patch(rect)
        ax4.set_title('(d) Bounding Boxes', fontweight='bold')
        ax4.axis('off')

        # Row 2: Individual masks (top 4 by area)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:4]
        for i, mask in enumerate(sorted_masks):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(image_rgb)
            mask_vis = np.zeros((*image.shape[:2], 4))
            mask_vis[..., 0] = 1.0  # Red channel
            mask_vis[..., 3] = mask['segmentation'] * 0.5  # Alpha
            ax.imshow(mask_vis)
            area_pct = mask['area'] / (image.shape[0] * image.shape[1]) * 100
            ax.set_title(f'(e{i+1}) Segment {i+1}\nArea: {area_pct:.1f}%', fontweight='bold')
            ax.axis('off')

        # Row 3: Metrics
        # Segment size histogram
        ax9 = fig.add_subplot(gs[2, 0:2])
        if masks:
            areas = [m['area'] for m in masks]
            total_pixels = image.shape[0] * image.shape[1]
            areas_pct = [a / total_pixels * 100 for a in areas]
            ax9.hist(areas_pct, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax9.axvline(np.mean(areas_pct), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(areas_pct):.2f}%')
            ax9.set_xlabel('Segment Size (% of image)')
            ax9.set_ylabel('Frequency')
            ax9.legend()
        ax9.set_title('(f) Segment Size Distribution', fontweight='bold')

        # Complexity metrics bar chart
        ax10 = fig.add_subplot(gs[2, 2])
        metric_names = ['Coverage', 'Edge Density', 'Size Std', 'Complexity']
        metric_values = [
            metrics['coverage_ratio'],
            min(metrics['edge_density'] * 5, 1.0),  # Scale for visibility
            min(metrics['segment_size_std'] * 10, 1.0),
            metrics['complexity_score']
        ]
        colors_metrics = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        bars = ax10.bar(metric_names, metric_values, color=colors_metrics, edgecolor='black')
        ax10.set_ylim(0, 1)
        ax10.set_ylabel('Normalized Value')
        ax10.set_title('(g) Complexity Metrics', fontweight='bold')
        for bar, val in zip(bars, metric_values):
            ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        # Summary text
        ax11 = fig.add_subplot(gs[2, 3])
        ax11.axis('off')
        summary_text = f"""
        Segmentation Summary
        ─────────────────────
        Total Segments: {metrics['num_segments']}
        Average Size: {metrics['avg_segment_size']*100:.2f}%
        Coverage: {metrics['coverage_ratio']*100:.1f}%
        Edge Density: {metrics['edge_density']:.4f}

        Complexity Score: {metrics['complexity_score']:.3f}
        (0 = simple, 1 = complex)

        Model: {'FastSAM' if self.model else 'Fallback (edge-based)'}
        """
        ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax11.set_title('(h) Summary', fontweight='bold')

        plt.suptitle(f'FastSAM Segmentation Analysis: {Path(image_path).name}',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

        return masks, metrics

    def visualize_comparison(self, image_paths, output_path, show=False):
        """Compare segmentation across multiple images."""
        n_images = len(image_paths)

        fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)

        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.extract_masks_fastsam(img_path)
            metrics = self.compute_complexity_metrics(masks, image.shape)
            colors = generate_colors(len(masks))

            # Original
            axes[idx, 0].imshow(image_rgb)
            axes[idx, 0].set_title(Path(img_path).stem[:20], fontsize=10)
            axes[idx, 0].axis('off')

            # Masks overlay
            overlay = image_rgb.copy()
            for i, mask in enumerate(masks):
                color = colors[i] if i < len(colors) else (255, 255, 255)
                mask_bool = mask['segmentation'] > 0
                overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * np.array(color)).astype(np.uint8)
            axes[idx, 1].imshow(overlay)
            axes[idx, 1].set_title(f'{len(masks)} segments', fontsize=10)
            axes[idx, 1].axis('off')

            # Complexity bar
            axes[idx, 2].barh(['Complexity'], [metrics['complexity_score']], color='steelblue')
            axes[idx, 2].set_xlim(0, 1)
            axes[idx, 2].text(metrics['complexity_score'] + 0.02, 0,
                            f'{metrics["complexity_score"]:.2f}', va='center')
            if idx == 0:
                axes[idx, 2].set_title('Complexity', fontweight='bold')

            # Coverage bar
            axes[idx, 3].barh(['Coverage'], [metrics['coverage_ratio']], color='coral')
            axes[idx, 3].set_xlim(0, 1)
            axes[idx, 3].text(metrics['coverage_ratio'] + 0.02, 0,
                            f'{metrics["coverage_ratio"]*100:.1f}%', va='center')
            if idx == 0:
                axes[idx, 3].set_title('Coverage', fontweight='bold')

        plt.suptitle('FastSAM Segmentation Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='FastSAM Segmentation Visualization for Scientific Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First 5 images (sequential)
  py -3.11 visualize_fastsam_segmentation.py --num_images 5

  # Random 5 images from dataset
  py -3.11 visualize_fastsam_segmentation.py --num_images 5 --random

  # Specific images by filename (searches in --images_dir)
  py -3.11 visualize_fastsam_segmentation.py --images frame_001.jpg frame_002.jpg

  # Specific images with full paths
  py -3.11 visualize_fastsam_segmentation.py --images "F:/path/image1.jpg" "F:/path/image2.jpg"

  # Custom FastSAM model
  py -3.11 visualize_fastsam_segmentation.py --num_images 3 --model_path "path/to/FastSAM-x.pt"

  # Show plots interactively
  py -3.11 visualize_fastsam_segmentation.py --num_images 2 --show
        """
    )

    # Image selection options
    selection_group = parser.add_argument_group('Image Selection')
    selection_group.add_argument('--images_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images',
                        help='Directory with images (default: selected_dataset/images)')
    selection_group.add_argument('--images', nargs='+', type=str, default=None,
                        help='Specific image files (filenames or full paths). Overrides --num_images')
    selection_group.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize (default: 5)')
    selection_group.add_argument('--random', action='store_true',
                        help='Randomly select images instead of first N sequential')
    selection_group.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible selection (optional)')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\fastsam',
                        help='Output directory for visualizations')
    output_group.add_argument('--show', action='store_true',
                        help='Show plots interactively (default: save only)')

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model_path', type=str, default=None,
                        help='Path to FastSAM model (default: External/Models/FastSAM/FastSAM-x.pt)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
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
                image_files.append(img_path)
            else:
                full_path = images_dir / img_name
                if full_path.exists():
                    image_files.append(full_path)
                else:
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
            image_files = random.sample(all_images, min(args.num_images, len(all_images)))
            print(f"Randomly selected {len(image_files)} images from {len(all_images)} total")
        else:
            image_files = all_images[:args.num_images]
            print(f"Selected first {len(image_files)} images from {len(all_images)} total")

    if not image_files:
        print("No images to process!")
        return

    print(f"\nImages to process:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")

    # Initialize visualizer
    visualizer = FastSAMVisualizer(model_path=args.model_path)

    # Generate individual visualizations
    print("\n=== Generating individual FastSAM visualizations ===")
    for img_path in tqdm(image_files, desc="Processing"):
        output_path = output_dir / f"{img_path.stem}_fastsam_analysis.png"
        visualizer.visualize_single_image(str(img_path), str(output_path), show=args.show)

    # Generate comparison
    print("\n=== Generating comparison grid ===")
    comparison_path = output_dir / "fastsam_comparison.png"
    visualizer.visualize_comparison(
        [str(p) for p in image_files[:min(5, len(image_files))]],
        str(comparison_path),
        show=args.show
    )

    print(f"\n=== Done! Visualizations saved to: {output_dir} ===")


if __name__ == '__main__':
    main()
