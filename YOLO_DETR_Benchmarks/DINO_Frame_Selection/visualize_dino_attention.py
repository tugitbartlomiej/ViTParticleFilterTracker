#!/usr/bin/env python3
"""
DINO Attention Visualization for Frame Selection

This script visualizes what DINO "sees" by showing attention maps as heatmaps.
It's designed to work with the existing frame selection pipeline and shows
the attention patterns that DINO uses for semantic understanding.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import seaborn as sns

class DINOAttentionVisualizer:
    """
    Visualizes DINO attention maps for understanding what the model focuses on
    """
    
    def __init__(self, model_name='dino_vits16', patch_size=16, device='auto'):
        """
        Initialize DINO attention visualizer
        
        Args:
            model_name: DINO model variant
            patch_size: Patch size for vision transformer
            device: Device to run model on
        """
        self.model_name = model_name
        self.patch_size = patch_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load DINO model
        self.model = self._load_dino_model()
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Set up visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def _load_dino_model(self):
        """Load pre-trained DINO model with attention"""
        try:
            # Load DINO model from torch hub
            model = torch.hub.load('facebookresearch/dino:main', self.model_name)
            model.to(self.device)
            
            # Monkey patch to get attention weights
            def get_attention(module, input, output):
                module.attention = output
            
            # Register hooks for attention
            for name, module in model.named_modules():
                if hasattr(module, 'attention'):
                    module.register_forward_hook(get_attention)
            
            return model
        except Exception as e:
            print(f"Error loading DINO model: {e}")
            raise
    
    def extract_attention_maps(self, image_path):
        """
        Extract attention maps from DINO model
        
        Args:
            image_path: Path to image
            
        Returns:
            Dict with attention maps and features
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get model output with attention
        with torch.no_grad():
            # Forward pass
            features = self.model(img_tensor)
            
            # Get attention from last layer
            attentions = self.model.get_last_selfattention(img_tensor)
            
            # Average attention across heads
            nh = attentions.shape[1]  # number of heads
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            
            # Get dimensions
            if self.patch_size == 16:
                w_featmap = img_tensor.shape[-1] // 16
                h_featmap = img_tensor.shape[-2] // 16
            else:
                w_featmap = img_tensor.shape[-1] // 8
                h_featmap = img_tensor.shape[-2] // 8
            
            # Reshape attention to spatial dimensions
            attentions = attentions.reshape(nh, h_featmap, w_featmap)
            
            # Average over heads
            attentions = attentions.mean(0)
            
        return {
            'attention_map': attentions.cpu().numpy(),
            'features': features.cpu().numpy(),
            'image_size': image.size,
            'patch_grid': (h_featmap, w_featmap)
        }
    
    def visualize_attention(self, image_path, save_path=None, show_patches=True):
        """
        Visualize DINO attention as a heatmap overlay
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
            show_patches: Whether to show patch grid
        """
        # Extract attention
        result = self.extract_attention_maps(image_path)
        attention_map = result['attention_map']
        h_featmap, w_featmap = result['patch_grid']
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Resize attention map to image size
        attention_resized = cv2.resize(attention_map, image.size, 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # Original image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Attention heatmap
        ax2 = fig.add_subplot(gs[1])
        im = ax2.imshow(attention_resized, cmap='hot', alpha=0.9)
        ax2.set_title('DINO Attention Map', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score', rotation=270, labelpad=15)
        
        # Overlay
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(image)
        ax3.imshow(attention_resized, cmap='hot', alpha=0.5)
        ax3.set_title('Attention Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Show patch grid if requested
        if show_patches:
            # Calculate patch positions
            patch_w = image.size[0] / w_featmap
            patch_h = image.size[1] / h_featmap
            
            for i in range(h_featmap):
                for j in range(w_featmap):
                    # Get attention value for this patch
                    att_val = attention_map[i, j]
                    
                    # Draw rectangle with color based on attention
                    rect = patches.Rectangle(
                        (j * patch_w, i * patch_h), 
                        patch_w, patch_h,
                        linewidth=1,
                        edgecolor='white',
                        facecolor='none',
                        alpha=0.3
                    )
                    ax3.add_patch(rect)
                    
                    # Add text with attention value for high-attention patches
                    if att_val > np.percentile(attention_map, 75):
                        ax3.text(j * patch_w + patch_w/2, 
                               i * patch_h + patch_h/2,
                               f'{att_val:.2f}',
                               ha='center', va='center',
                               fontsize=8, color='white',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        
        return attention_resized
    
    def create_attention_grid(self, image_paths, output_path, max_images=9):
        """
        Create a grid visualization of multiple images with their attention maps
        
        Args:
            image_paths: List of image paths
            output_path: Path to save grid visualization
            max_images: Maximum number of images in grid
        """
        n_images = min(len(image_paths), max_images)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5 * n_cols, 10 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(2, -1)
        
        for idx, img_path in enumerate(image_paths[:n_images]):
            row = (idx // n_cols) * 2
            col = idx % n_cols
            
            # Extract attention
            result = self.extract_attention_maps(img_path)
            attention_map = result['attention_map']
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Resize attention
            attention_resized = cv2.resize(attention_map, image.size, 
                                         interpolation=cv2.INTER_CUBIC)
            
            # Show original image
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'{Path(img_path).stem}', fontsize=10)
            axes[row, col].axis('off')
            
            # Show attention overlay
            axes[row + 1, col].imshow(image)
            im = axes[row + 1, col].imshow(attention_resized, cmap='hot', alpha=0.6)
            axes[row + 1, col].axis('off')
        
        # Remove empty subplots
        for idx in range(n_images, n_rows * n_cols):
            row = (idx // n_cols) * 2
            col = idx % n_cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
        
        plt.suptitle('DINO Attention Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to: {output_path}")
        plt.close()
    
    def analyze_frame_attention(self, image_path):
        """
        Analyze attention distribution for a single frame
        
        Returns statistics about attention distribution
        """
        result = self.extract_attention_maps(image_path)
        attention_map = result['attention_map']
        
        # Calculate statistics
        stats = {
            'mean_attention': float(np.mean(attention_map)),
            'std_attention': float(np.std(attention_map)),
            'max_attention': float(np.max(attention_map)),
            'min_attention': float(np.min(attention_map)),
            'high_attention_ratio': float(np.sum(attention_map > np.percentile(attention_map, 75)) / attention_map.size),
            'attention_entropy': float(-np.sum(attention_map * np.log(attention_map + 1e-10))),
            'top_5_patches': self._get_top_patches(attention_map, n=5)
        }
        
        return stats
    
    def _get_top_patches(self, attention_map, n=5):
        """Get coordinates of top n attention patches"""
        flat_indices = np.argpartition(attention_map.ravel(), -n)[-n:]
        top_coords = np.unravel_index(flat_indices, attention_map.shape)
        top_values = attention_map[top_coords]
        
        patches = []
        for i in range(n):
            patches.append({
                'row': int(top_coords[0][i]),
                'col': int(top_coords[1][i]),
                'value': float(top_values[i])
            })
        
        return sorted(patches, key=lambda x: x['value'], reverse=True)
    
    def process_frame_directory(self, input_dir, output_dir, visualize_all=False):
        """
        Process all frames in a directory and generate attention visualizations
        
        Args:
            input_dir: Directory with input frames
            output_dir: Directory to save visualizations
            visualize_all: Whether to create individual visualizations for all frames
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images to process")
        
        # Create subdirectories
        (output_path / 'individual').mkdir(exist_ok=True)
        (output_path / 'analysis').mkdir(exist_ok=True)
        
        # Process each frame
        all_stats = []
        
        for img_path in tqdm(image_files, desc="Processing frames"):
            # Analyze attention
            stats = self.analyze_frame_attention(img_path)
            stats['filename'] = img_path.name
            all_stats.append(stats)
            
            # Create individual visualization if requested
            if visualize_all:
                save_path = output_path / 'individual' / f'{img_path.stem}_attention.png'
                self.visualize_attention(img_path, save_path=save_path, show_patches=True)
        
        # Save analysis results
        analysis_path = output_path / 'analysis' / 'attention_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Create grid visualization
        grid_path = output_path / 'attention_grid.png'
        self.create_attention_grid(image_files[:9], grid_path)
        
        # Create summary statistics
        self._create_summary_plot(all_stats, output_path / 'attention_summary.png')
        
        print(f"Processing complete. Results saved to: {output_path}")
    
    def _create_summary_plot(self, stats_list, save_path):
        """Create summary plot of attention statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean attention distribution
        mean_attentions = [s['mean_attention'] for s in stats_list]
        axes[0, 0].hist(mean_attentions, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Mean Attention', fontweight='bold')
        axes[0, 0].set_xlabel('Mean Attention')
        axes[0, 0].set_ylabel('Frequency')
        
        # High attention ratio
        high_ratios = [s['high_attention_ratio'] for s in stats_list]
        axes[0, 1].hist(high_ratios, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Distribution of High Attention Ratio', fontweight='bold')
        axes[0, 1].set_xlabel('High Attention Ratio')
        axes[0, 1].set_ylabel('Frequency')
        
        # Attention entropy
        entropies = [s['attention_entropy'] for s in stats_list]
        axes[1, 0].hist(entropies, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Distribution of Attention Entropy', fontweight='bold')
        axes[1, 0].set_xlabel('Entropy')
        axes[1, 0].set_ylabel('Frequency')
        
        # Top frames by mean attention
        sorted_stats = sorted(stats_list, key=lambda x: x['mean_attention'], reverse=True)[:10]
        names = [s['filename'][:15] + '...' if len(s['filename']) > 15 else s['filename'] 
                for s in sorted_stats]
        values = [s['mean_attention'] for s in sorted_stats]
        
        axes[1, 1].barh(names, values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Top 10 Frames by Mean Attention', fontweight='bold')
        axes[1, 1].set_xlabel('Mean Attention')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize DINO attention maps')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing frames to analyze')
    parser.add_argument('--output_dir', type=str, default='dino_attention_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--model_name', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO model variant')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Create individual visualizations for all frames')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run model on')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DINOAttentionVisualizer(
        model_name=args.model_name,
        device=args.device
    )
    
    # Process frames
    visualizer.process_frame_directory(
        args.input_dir,
        args.output_dir,
        visualize_all=args.visualize_all
    )

if __name__ == "__main__":
    main()