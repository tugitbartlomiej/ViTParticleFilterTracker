#!/usr/bin/env python3
"""
Test Script for DINO Information Analyzer
=========================================

Test the DINO information richness analyzer on existing frames to validate
the quality assessment approach before full integration.

Usage:
    python test_dino_analyzer.py --test_frames_dir path/to/frames
    python test_dino_analyzer.py --quick_test
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add DINO analyzer to path
sys.path.insert(0, str(Path(__file__).parent))
from dino_information_analyzer import DINOInformationAnalyzer

def test_single_frame(analyzer, frame_path):
    """Test analyzer on a single frame"""
    print(f"\n=== Testing Frame: {Path(frame_path).name} ===")
    
    result = analyzer.extract_comprehensive_features(frame_path)
    
    if result is None:
        print("Analysis failed")
        return None
    
    # Display key metrics
    print(f"Information Score: {result['information_score']:.4f}")
    print(f"Attention Available: {result['attention_available']}")
    
    if result['attention_available']:
        print(f"Attention Entropy: {result.get('attention_entropy', 'N/A'):.4f}")
        print(f"Attention Variance: {result.get('attention_variance', 'N/A'):.4f}")
        print(f"Spatial Coherence: {result.get('spatial_coherence', 'N/A'):.4f}")
        print(f"Attention Diversity: {result.get('attention_diversity', 'N/A'):.4f}")
    
    # Feature statistics
    feat_stats = result['feature_statistics']
    print(f"Feature Mean: {feat_stats['feature_mean']:.4f}")
    print(f"Feature Std: {feat_stats['feature_std']:.4f}")
    print(f"Feature Variance: {feat_stats['feature_variance']:.4f}")
    print(f"Feature Sparsity: {feat_stats['feature_sparsity']:.4f}")
    
    return result

def test_frame_directory(analyzer, frames_dir, output_file=None):
    """Test analyzer on directory of frames"""
    print(f"\n=== Testing Frame Directory: {frames_dir} ===")
    
    results = analyzer.analyze_frame_directory(
        frames_dir, 
        output_file=output_file,
        quality_threshold=0.0  # Include all frames for analysis
    )
    
    return results

def create_quality_visualization(results, output_dir):
    """Create visualization of quality scores"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_frames = results['all_frames']
    scores = [frame['information_score'] for frame in all_frames]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DINO Information Richness Analysis Results', fontsize=16, fontweight='bold')
    
    # Score distribution histogram
    axes[0, 0].hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Information Scores')
    axes[0, 0].set_xlabel('Information Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    axes[0, 0].legend()
    
    # Top frames by score
    sorted_frames = sorted(all_frames, key=lambda x: x['information_score'], reverse=True)
    top_frames = sorted_frames[:10]
    
    frame_names = [Path(f['image_name']).stem[:15] + '...' if len(Path(f['image_name']).stem) > 15 
                   else Path(f['image_name']).stem for f in top_frames]
    top_scores = [f['information_score'] for f in top_frames]
    
    axes[0, 1].barh(range(len(frame_names)), top_scores, color='green', alpha=0.7)
    axes[0, 1].set_yticks(range(len(frame_names)))
    axes[0, 1].set_yticklabels(frame_names)
    axes[0, 1].set_title('Top 10 Frames by Information Score')
    axes[0, 1].set_xlabel('Information Score')
    
    # Attention entropy vs Information score (if available)
    attention_frames = [f for f in all_frames if f.get('attention_entropy') is not None]
    if attention_frames:
        att_entropies = [f['attention_entropy'] for f in attention_frames]
        att_scores = [f['information_score'] for f in attention_frames]
        
        axes[1, 0].scatter(att_entropies, att_scores, alpha=0.6, color='purple')
        axes[1, 0].set_title('Attention Entropy vs Information Score')
        axes[1, 0].set_xlabel('Attention Entropy')
        axes[1, 0].set_ylabel('Information Score')
        
        # Add correlation coefficient
        corr = np.corrcoef(att_entropies, att_scores)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'No attention data available', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('Attention Analysis (Not Available)')
    
    # Feature variance vs Information score
    variances = [f['feature_statistics']['feature_variance'] for f in all_frames]
    axes[1, 1].scatter(variances, scores, alpha=0.6, color='orange')
    axes[1, 1].set_title('Feature Variance vs Information Score')
    axes[1, 1].set_xlabel('Feature Variance')
    axes[1, 1].set_ylabel('Information Score')
    
    # Add correlation coefficient
    if len(variances) > 1:
        corr = np.corrcoef(variances, scores)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'quality_analysis_visualization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Quality visualization saved: {plot_path}")

def run_quick_test():
    """Run a quick test on pipeline output frames if available"""
    # Look for existing frames in pipeline output
    possible_dirs = [
        "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/background_frames/train",
        "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/background_frames/images",
        "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output/background_frames/images"
    ]
    
    test_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            # Check if it contains images
            image_files = list(Path(dir_path).glob('*.jpg')) + list(Path(dir_path).glob('*.png'))
            if image_files:
                test_dir = dir_path
                print(f"Found {len(image_files)} images in {dir_path}")
                break
    
    if test_dir is None:
        print("No existing frames found for quick test")
        print("Available directories to check:")
        for dir_path in possible_dirs:
            exists = "EXISTS" if Path(dir_path).exists() else "NOT_FOUND"
            print(f"  {exists}: {dir_path}")
        return False
    
    print(f"Found test frames in: {test_dir}")
    
    # Initialize analyzer
    analyzer = DINOInformationAnalyzer(device='auto')
    
    # Run analysis
    output_file = Path(__file__).parent / 'test_results' / 'quick_test_analysis.json'
    results = test_frame_directory(analyzer, test_dir, str(output_file))
    
    if results:
        # Create visualization
        create_quality_visualization(results, Path(__file__).parent / 'test_results')
        
        # Print summary
        stats = results['statistics']
        print(f"\n=== Quick Test Results ===")
        print(f"Frames analyzed: {stats['total_frames']}")
        print(f"Mean quality score: {stats['mean_score']:.4f}")
        print(f"Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
        print(f"High-quality frames (≥0.5): {len([f for f in results['all_frames'] if f['information_score'] >= 0.5])}")
        
        return True
    else:
        print("❌ Analysis failed")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test DINO Information Analyzer')
    
    parser.add_argument('--test_frames_dir', type=str,
                        help='Directory containing frames to test')
    parser.add_argument('--single_frame', type=str,
                        help='Test single frame file')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test on pipeline output frames')
    parser.add_argument('--output_dir', type=str,
                        default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device for DINO model')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.quick_test:
        print("=== DINO ANALYZER QUICK TEST ===")
        success = run_quick_test()
        return 0 if success else 1
    
    # Initialize analyzer
    try:
        print("Initializing DINO Information Analyzer...")
        analyzer = DINOInformationAnalyzer(device=args.device)
        print("✓ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return 1
    
    # Run tests
    try:
        if args.single_frame:
            # Test single frame
            if not os.path.exists(args.single_frame):
                print(f"❌ Frame not found: {args.single_frame}")
                return 1
            
            result = test_single_frame(analyzer, args.single_frame)
            
            # Save result
            if result:
                result_file = output_path / f'single_frame_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"✓ Result saved: {result_file}")
        
        elif args.test_frames_dir:
            # Test frame directory
            if not os.path.exists(args.test_frames_dir):
                print(f"❌ Directory not found: {args.test_frames_dir}")
                return 1
            
            output_file = output_path / f'directory_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            results = test_frame_directory(analyzer, args.test_frames_dir, str(output_file))
            
            if results:
                # Create visualization
                create_quality_visualization(results, output_path)
                
                # Print summary
                stats = results['statistics']
                print(f"\n=== Test Results Summary ===")
                print(f"Total frames: {stats['total_frames']}")
                print(f"Mean quality: {stats['mean_score']:.4f}")
                print(f"Std quality: {stats['std_score']:.4f}")
                print(f"Range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
                print(f"90th percentile: {stats['percentiles']['90']:.4f}")
                print(f"Results saved: {output_file}")
        
        else:
            print("❌ Must specify either --single_frame, --test_frames_dir, or --quick_test")
            return 1
        
        print("✓ Testing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())