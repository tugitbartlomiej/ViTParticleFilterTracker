#!/usr/bin/env python3
"""
DINO Feature Visualization for Scientific Paper

Generates publication-quality visualizations showing:
1. DINO attention maps (what the model focuses on)
2. CLS token feature extraction process
3. Patch-level attention heatmaps
4. Multi-head attention comparison

Usage Examples:
    # Select first 5 images from directory
    py -3.11 visualize_dino_features.py --num_images 5

    # Random selection of 5 images from dataset
    py -3.11 visualize_dino_features.py --num_images 5 --random

    # Specific images by filename
    py -3.11 visualize_dino_features.py --images image1.jpg image2.jpg image3.jpg

    # Specific images with full paths
    py -3.11 visualize_dino_features.py --images "F:/path/to/image1.jpg" "F:/path/to/image2.jpg"

    # Show help
    py -3.11 visualize_dino_features.py --help
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path
from tqdm import tqdm
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DINOVisualizer:
    """Professional DINO visualization for paper figures."""

    def __init__(self, model_name='dino_vits16', device='cuda'):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.patch_size = 16 if '16' in model_name else 8

        print(f"Loading DINO model: {model_name}")
        print(f"Device: {self.device}")

        # Load model
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Set matplotlib style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def get_attention_maps(self, image_path):
        """Extract multi-head attention maps from DINO."""
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get attention from last transformer block
            attentions = self.model.get_last_selfattention(img_tensor)

            # attentions shape: [1, num_heads, num_patches+1, num_patches+1]
            nh = attentions.shape[1]  # number of heads

            # Get CLS token attention to all patches (excluding CLS itself)
            # Shape: [num_heads, num_patches]
            cls_attention = attentions[0, :, 0, 1:]

            # Reshape to spatial grid
            w_featmap = 224 // self.patch_size
            h_featmap = 224 // self.patch_size

            attention_maps = cls_attention.reshape(nh, h_featmap, w_featmap)

            # Also get CLS token features
            features = self.model(img_tensor)

        return {
            'attention_maps': attention_maps.cpu().numpy(),  # [nh, h, w]
            'mean_attention': attention_maps.mean(0).cpu().numpy(),  # [h, w]
            'features': features.cpu().numpy().flatten(),
            'original_size': image.size,
            'num_heads': nh
        }

    def visualize_single_image(self, image_path, output_path, show=False):
        """Create comprehensive visualization for a single image."""
        result = self.get_attention_maps(image_path)
        image = Image.open(image_path).convert('RGB')

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)

        # Row 1: Original image and mean attention
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(image)
        ax1.set_title('(a) Original Image', fontweight='bold')
        ax1.axis('off')

        # Mean attention heatmap
        mean_att = result['mean_attention']
        mean_att_resized = cv2.resize(mean_att, image.size, interpolation=cv2.INTER_CUBIC)

        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(image)
        im = ax2.imshow(mean_att_resized, cmap='jet', alpha=0.6)
        ax2.set_title('(b) DINO Attention Overlay', fontweight='bold')
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score', fontsize=10)

        # Row 2: Individual attention heads (first 4)
        num_heads = min(4, result['num_heads'])
        for i in range(num_heads):
            ax = fig.add_subplot(gs[1, i])
            head_att = result['attention_maps'][i]
            head_att_resized = cv2.resize(head_att, image.size, interpolation=cv2.INTER_CUBIC)
            ax.imshow(image)
            ax.imshow(head_att_resized, cmap='hot', alpha=0.5)
            ax.set_title(f'(c{i+1}) Head {i+1}', fontweight='bold')
            ax.axis('off')

        # Row 3: Attention statistics and feature visualization
        ax5 = fig.add_subplot(gs[2, 0:2])
        # Attention distribution histogram
        att_flat = mean_att.flatten()
        ax5.hist(att_flat, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(att_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(att_flat):.3f}')
        ax5.axvline(np.percentile(att_flat, 75), color='orange', linestyle='--', linewidth=2, label=f'75th %ile: {np.percentile(att_flat, 75):.3f}')
        ax5.set_xlabel('Attention Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('(d) Attention Distribution', fontweight='bold')
        ax5.legend(fontsize=9)

        # Feature vector visualization (first 100 dimensions)
        ax6 = fig.add_subplot(gs[2, 2:4])
        features = result['features'][:100]  # First 100 dims
        ax6.bar(range(len(features)), features, color='darkgreen', alpha=0.7)
        ax6.set_xlabel('Feature Dimension')
        ax6.set_ylabel('Activation')
        ax6.set_title(f'(e) CLS Token Features (first 100/{len(result["features"])} dims)', fontweight='bold')
        ax6.set_xlim(-1, 100)

        plt.suptitle(f'DINO Feature Analysis: {Path(image_path).name}', fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

    def visualize_attention_comparison(self, image_paths, output_path, show=False):
        """Create grid comparison of multiple images."""
        n_images = len(image_paths)
        fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))

        if n_images == 1:
            axes = axes.reshape(1, -1)

        for idx, img_path in enumerate(image_paths):
            result = self.get_attention_maps(img_path)
            image = Image.open(img_path).convert('RGB')
            mean_att = result['mean_attention']
            mean_att_resized = cv2.resize(mean_att, image.size, interpolation=cv2.INTER_CUBIC)

            # Original
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'{Path(img_path).stem}' if idx == 0 else '', fontsize=10)
            axes[idx, 0].axis('off')
            if idx == 0:
                axes[idx, 0].set_ylabel('Original', fontsize=12, fontweight='bold')

            # Attention heatmap only
            im = axes[idx, 1].imshow(mean_att_resized, cmap='jet')
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title('Attention Map', fontsize=12, fontweight='bold')

            # Overlay
            axes[idx, 2].imshow(image)
            axes[idx, 2].imshow(mean_att_resized, cmap='jet', alpha=0.5)
            axes[idx, 2].axis('off')
            if idx == 0:
                axes[idx, 2].set_title('Overlay', fontsize=12, fontweight='bold')

        plt.suptitle('DINO Attention Analysis Across Frames', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

    def visualize_all_heads(self, image_path, output_path, show=False):
        """Visualize all attention heads in a grid."""
        result = self.get_attention_maps(image_path)
        image = Image.open(image_path).convert('RGB')

        nh = result['num_heads']
        n_cols = 4
        n_rows = (nh + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i in range(nh):
            head_att = result['attention_maps'][i]
            head_att_resized = cv2.resize(head_att, image.size, interpolation=cv2.INTER_CUBIC)

            axes[i].imshow(image)
            axes[i].imshow(head_att_resized, cmap='hot', alpha=0.5)
            axes[i].set_title(f'Head {i+1}', fontweight='bold')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(nh, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'All {nh} DINO Attention Heads', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='DINO Feature Visualization for Scientific Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First 5 images (sequential)
  py -3.11 visualize_dino_features.py --num_images 5

  # Random 5 images from dataset
  py -3.11 visualize_dino_features.py --num_images 5 --random

  # Specific images by filename (searches in --images_dir)
  py -3.11 visualize_dino_features.py --images frame_001.jpg frame_002.jpg

  # Specific images with full paths
  py -3.11 visualize_dino_features.py --images "F:/path/image1.jpg" "F:/path/image2.jpg"

  # Different DINO model
  py -3.11 visualize_dino_features.py --num_images 3 --model dino_vitb16

  # Show plots interactively
  py -3.11 visualize_dino_features.py --num_images 2 --show
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
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\dino',
                        help='Output directory for visualizations')
    output_group.add_argument('--show', action='store_true',
                        help='Show plots interactively (default: save only)')

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO model variant (default: dino_vits16)')

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

    print(f"\nImages to process:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")

    # Initialize visualizer
    visualizer = DINOVisualizer(model_name=args.model)

    # Generate visualizations
    print("\n=== Generating individual visualizations ===")
    for img_path in tqdm(image_files, desc="Processing"):
        output_path = output_dir / f"{img_path.stem}_dino_analysis.png"
        visualizer.visualize_single_image(str(img_path), str(output_path), show=args.show)

    # Generate comparison grid
    print("\n=== Generating comparison grid ===")
    comparison_path = output_dir / "dino_comparison_grid.png"
    visualizer.visualize_attention_comparison(
        [str(p) for p in image_files[:min(5, len(image_files))]],
        str(comparison_path),
        show=args.show
    )

    # Generate all heads visualization for first image
    print("\n=== Generating all heads visualization ===")
    heads_path = output_dir / f"{image_files[0].stem}_all_heads.png"
    visualizer.visualize_all_heads(str(image_files[0]), str(heads_path), show=args.show)

    print(f"\n=== Done! Visualizations saved to: {output_dir} ===")


if __name__ == '__main__':
    main()
