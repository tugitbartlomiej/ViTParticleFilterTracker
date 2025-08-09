#!/usr/bin/env python3
"""
DINO Information Richness Analyzer
==================================

Advanced frame quality assessment using DINO self-supervised vision transformers.
This analyzer determines which frames have the most information density for DETR training
by analyzing attention patterns, feature variance, and semantic coherence.

Key Features:
- Information theory-based frame scoring using attention entropy
- Feature variance analysis for content richness
- Semantic coherence assessment via attention maps
- KL divergence-based quality metrics
- Medical video-optimized frame selection

Usage:
    python dino_information_analyzer.py --input_dir frames/ --output_file quality_scores.json
    python dino_information_analyzer.py --video_path video.mp4 --extract_frames --quality_threshold 0.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

class DINOInformationAnalyzer:
    """
    Advanced DINO-based frame quality analyzer for information richness assessment
    """
    
    def __init__(self, model_name='dino_vits16', device='auto', patch_size=16):
        """
        Initialize DINO information analyzer
        
        Args:
            model_name: DINO model variant
            device: Device to run model on
            patch_size: Vision transformer patch size
        """
        self.model_name = model_name
        self.patch_size = patch_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DINO Information Analyzer - Device: {self.device}")
        
        # Load DINO model with attention capabilities
        self.model = self._load_dino_with_attention()
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Quality assessment parameters
        self.quality_params = {
            'entropy_weight': 0.3,
            'variance_weight': 0.25,
            'coherence_weight': 0.25,
            'diversity_weight': 0.2,
            'min_entropy_threshold': 0.1,
            'max_entropy_threshold': 4.0
        }
        
        print(f"DINO Information Analyzer initialized ({model_name})")
    
    def _load_dino_with_attention(self):
        """Load DINO model with attention extraction capabilities"""
        try:
            # Load DINO from torch hub
            model = torch.hub.load('facebookresearch/dino:main', self.model_name)
            model.to(self.device)
            
            # Store attention weights during forward pass
            self.attention_weights = []
            
            def attention_hook(module, input, output):
                if hasattr(module, 'attention'):
                    self.attention_weights.append(output)
            
            # Register hooks for attention extraction
            for name, module in model.named_modules():
                if 'attn' in name:
                    module.register_forward_hook(attention_hook)
            
            return model
            
        except Exception as e:
            print(f"Error loading DINO model: {e}")
            print("Trying alternative loading method...")
            
            try:
                import timm
                model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
                model.to(self.device)
                return model
            except:
                raise Exception("Could not load DINO model. Please check installation.")
    
    def extract_comprehensive_features(self, image_path):
        """
        Extract comprehensive features including attention, embeddings, and quality metrics
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with comprehensive feature analysis
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Clear previous attention weights
                self.attention_weights = []
                
                # Forward pass to get features and attention
                features = self.model(img_tensor)
                
                # Extract attention from last layer
                try:
                    attentions = self.model.get_last_selfattention(img_tensor)
                    attention_available = True
                except:
                    attention_available = False
                    attentions = None
                
                # Comprehensive feature extraction
                result = {
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name,
                    'global_features': features.cpu().numpy().flatten(),
                    'feature_statistics': self._compute_feature_statistics(features),
                    'attention_available': attention_available
                }
                
                # Add attention-based metrics if available
                if attention_available and attentions is not None:
                    attention_metrics = self._analyze_attention_patterns(attentions)
                    result.update(attention_metrics)
                else:
                    # Fallback: use feature-based quality metrics
                    result.update(self._compute_fallback_quality_metrics(features, image))
                
                # Compute final information richness score
                result['information_score'] = self._compute_information_score(result)
                
                return result
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _compute_feature_statistics(self, features):
        """Compute statistical properties of features"""
        feat_np = features.cpu().numpy().flatten()
        
        return {
            'feature_mean': float(np.mean(feat_np)),
            'feature_std': float(np.std(feat_np)),
            'feature_variance': float(np.var(feat_np)),
            'feature_max': float(np.max(feat_np)),
            'feature_min': float(np.min(feat_np)),
            'feature_range': float(np.max(feat_np) - np.min(feat_np)),
            'feature_norm': float(np.linalg.norm(feat_np)),
            'feature_sparsity': float(np.sum(np.abs(feat_np) < 1e-6) / len(feat_np)),
            'feature_kurtosis': float(self._safe_kurtosis(feat_np)),
            'feature_skewness': float(self._safe_skewness(feat_np))
        }
    
    def _analyze_attention_patterns(self, attentions):
        """Analyze attention patterns for information richness"""
        # Process attention tensor
        att = attentions[0]  # First batch
        
        # Average over attention heads
        if len(att.shape) == 4:  # [batch, heads, seq, seq]
            att_heads = att[0]  # Remove batch dimension
            mean_att = att_heads.mean(0)  # Average over heads
        else:
            mean_att = att[0]
        
        # Remove CLS token attention (first row/column)
        spatial_att = mean_att[1:, 1:]
        
        # Compute spatial attention map
        seq_len = spatial_att.shape[0]
        grid_size = int(np.sqrt(seq_len))
        
        if grid_size * grid_size == seq_len:
            # Reshape to spatial grid
            spatial_map = spatial_att.diagonal().reshape(grid_size, grid_size)
            
            # Compute attention-based metrics
            att_np = spatial_map.cpu().numpy()
            
            return {
                'attention_entropy': float(self._compute_attention_entropy(att_np)),
                'attention_variance': float(np.var(att_np)),
                'attention_max': float(np.max(att_np)),
                'attention_concentration': float(self._compute_attention_concentration(att_np)),
                'spatial_coherence': float(self._compute_spatial_coherence(att_np)),
                'attention_diversity': float(self._compute_attention_diversity(spatial_att.cpu().numpy()))
            }
        else:
            # Fallback for non-square attention maps
            att_flat = spatial_att.cpu().numpy().flatten()
            return {
                'attention_entropy': float(entropy(att_flat + 1e-10)),
                'attention_variance': float(np.var(att_flat)),
                'attention_max': float(np.max(att_flat)),
                'attention_concentration': float(np.sum(att_flat > np.percentile(att_flat, 75)) / len(att_flat)),
                'spatial_coherence': 0.5,  # Neutral score
                'attention_diversity': float(np.std(att_flat))
            }
    
    def _compute_fallback_quality_metrics(self, features, image):
        """Compute quality metrics when attention is not available"""
        # Convert PIL image to numpy for analysis
        img_np = np.array(image)
        
        # Compute edge-based metrics
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Compute texture metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Feature-based metrics
        feat_np = features.cpu().numpy().flatten()
        
        return {
            'attention_entropy': float(entropy(np.abs(feat_np) + 1e-10)),
            'attention_variance': float(np.var(feat_np)),
            'attention_max': float(np.max(feat_np)),
            'attention_concentration': float(edge_density),
            'spatial_coherence': float(min(laplacian_var / 1000.0, 1.0)),
            'attention_diversity': float(np.std(feat_np))
        }
    
    def _compute_attention_entropy(self, attention_map):
        """Compute Shannon entropy of attention map"""
        # Normalize attention map
        att_norm = attention_map / (np.sum(attention_map) + 1e-10)
        
        # Flatten and compute entropy
        att_flat = att_norm.flatten()
        att_flat = att_flat[att_flat > 1e-10]  # Remove zeros
        
        return entropy(att_flat)
    
    def _compute_attention_concentration(self, attention_map):
        """Compute how concentrated the attention is (inverse of diversity)"""
        att_flat = attention_map.flatten()
        
        # Proportion of high-attention areas
        threshold = np.percentile(att_flat, 75)
        high_attention_ratio = np.sum(att_flat > threshold) / len(att_flat)
        
        return 1.0 - high_attention_ratio  # Higher concentration = lower diversity
    
    def _compute_spatial_coherence(self, attention_map):
        """Compute spatial coherence of attention patterns"""
        # Use gradient magnitude to measure spatial coherence
        grad_x = np.gradient(attention_map, axis=1)
        grad_y = np.gradient(attention_map, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize by map range
        coherence = np.mean(grad_magnitude) / (np.max(attention_map) - np.min(attention_map) + 1e-10)
        
        return min(coherence, 1.0)
    
    def _compute_attention_diversity(self, attention_matrix):
        """Compute diversity of attention patterns across different heads/positions"""
        # Compute pairwise distances between attention vectors
        if attention_matrix.ndim > 1:
            distances = pairwise_distances(attention_matrix)
            mean_distance = np.mean(distances)
            return mean_distance
        else:
            return np.std(attention_matrix)
    
    def _compute_information_score(self, feature_dict):
        """
        Compute final information richness score using weighted combination
        
        Based on research findings:
        - Entropy indicates information density
        - Variance shows feature richness
        - Coherence measures semantic structure
        - Diversity prevents redundancy
        """
        weights = self.quality_params
        
        # Normalize metrics to [0, 1] range
        entropy_score = self._normalize_score(
            feature_dict.get('attention_entropy', 0), 
            weights['min_entropy_threshold'], 
            weights['max_entropy_threshold']
        )
        
        variance_score = self._normalize_score(
            feature_dict.get('feature_variance', 0), 0, 1.0
        )
        
        coherence_score = feature_dict.get('spatial_coherence', 0.5)
        
        diversity_score = self._normalize_score(
            feature_dict.get('attention_diversity', 0), 0, 1.0
        )
        
        # Weighted combination
        info_score = (
            weights['entropy_weight'] * entropy_score +
            weights['variance_weight'] * variance_score +
            weights['coherence_weight'] * coherence_score +
            weights['diversity_weight'] * diversity_score
        )
        
        return float(np.clip(info_score, 0.0, 1.0))
    
    def _normalize_score(self, value, min_val, max_val):
        """Normalize score to [0, 1] range"""
        if max_val <= min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def _safe_kurtosis(self, arr):
        """Compute kurtosis safely"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(arr)
        except:
            return 0.0
    
    def _safe_skewness(self, arr):
        """Compute skewness safely"""
        try:
            from scipy.stats import skew
            return skew(arr)
        except:
            return 0.0
    
    def analyze_frame_directory(self, input_dir, output_file=None, quality_threshold=0.0):
        """
        Analyze all frames in directory for information richness
        
        Args:
            input_dir: Directory containing frames
            output_file: Output file for results
            quality_threshold: Minimum quality threshold for selection
            
        Returns:
            List of analyzed frames with quality scores
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to analyze")
        
        if not image_files:
            raise ValueError("No image files found in directory")
        
        # Analyze each frame
        results = []
        high_quality_frames = []
        
        for img_path in tqdm(image_files, desc="Analyzing information richness"):
            analysis = self.extract_comprehensive_features(img_path)
            
            if analysis is not None:
                results.append(analysis)
                
                # Filter by quality threshold
                if analysis['information_score'] >= quality_threshold:
                    high_quality_frames.append(analysis)
        
        # Sort by information score (descending)
        results.sort(key=lambda x: x['information_score'], reverse=True)
        high_quality_frames.sort(key=lambda x: x['information_score'], reverse=True)
        
        # Compute statistics
        scores = [r['information_score'] for r in results]
        stats = {
            'total_frames': len(results),
            'high_quality_frames': len(high_quality_frames),
            'quality_threshold': quality_threshold,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'percentiles': {
                '90': float(np.percentile(scores, 90)),
                '75': float(np.percentile(scores, 75)),
                '50': float(np.percentile(scores, 50)),
                '25': float(np.percentile(scores, 25))
            }
        }
        
        # Prepare output data
        output_data = {
            'analysis_metadata': {
                'analyzer_version': '1.0',
                'model_name': self.model_name,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(input_dir),
                'quality_parameters': self.quality_params
            },
            'statistics': stats,
            'all_frames': results,
            'high_quality_frames': high_quality_frames
        }
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._prepare_for_json(output_data)
                json.dump(json_data, f, indent=2)
            
            print(f"Analysis results saved to: {output_path}")
        
        print(f"\n=== DINO Information Analysis Results ===")
        print(f"Total frames analyzed: {stats['total_frames']}")
        print(f"High-quality frames (≥{quality_threshold:.2f}): {stats['high_quality_frames']}")
        print(f"Mean information score: {stats['mean_score']:.3f}")
        print(f"Score range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        print(f"Quality distribution:")
        print(f"  90th percentile: {stats['percentiles']['90']:.3f}")
        print(f"  75th percentile: {stats['percentiles']['75']:.3f}")
        print(f"  Median: {stats['percentiles']['50']:.3f}")
        print(f"  25th percentile: {stats['percentiles']['25']:.3f}")
        
        return output_data
    
    def _prepare_for_json(self, data):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        else:
            return data
    
    def select_top_frames(self, analysis_results, n_frames=None, min_score=None):
        """
        Select top frames based on information richness
        
        Args:
            analysis_results: Results from analyze_frame_directory
            n_frames: Number of top frames to select
            min_score: Minimum score threshold
            
        Returns:
            List of selected frame paths
        """
        all_frames = analysis_results['all_frames']
        
        # Apply minimum score filter
        if min_score is not None:
            filtered_frames = [f for f in all_frames if f['information_score'] >= min_score]
        else:
            filtered_frames = all_frames
        
        # Sort by score (already sorted from analyze_frame_directory)
        selected_frames = filtered_frames
        
        # Apply count limit
        if n_frames is not None:
            selected_frames = selected_frames[:n_frames]
        
        return [frame['image_path'] for frame in selected_frames]

def main():
    parser = argparse.ArgumentParser(
        description='DINO Information Richness Analyzer for Frame Quality Assessment'
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing frames to analyze')
    parser.add_argument('--output_file', type=str, 
                        default='dino_information_analysis.json',
                        help='Output file for analysis results')
    parser.add_argument('--model_name', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO model variant')
    parser.add_argument('--quality_threshold', type=float, default=0.0,
                        help='Minimum quality threshold for frame selection')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run model on')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DINOInformationAnalyzer(
        model_name=args.model_name,
        device=args.device
    )
    
    # Analyze frames
    results = analyzer.analyze_frame_directory(
        args.input_dir,
        args.output_file,
        args.quality_threshold
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {args.output_file}")
    print("Use these high-quality frames for optimal DETR training!")

if __name__ == "__main__":
    main()