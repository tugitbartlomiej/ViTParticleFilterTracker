#!/usr/bin/env python3
"""
Test script for DINO attention visualization
Tests the visualization on pre-selected frames from the background selector
"""

import os
import sys
from pathlib import Path

# Add DINO scripts to path
sys.path.insert(0, str(Path(__file__).parent / "DINO_Frame_Selection"))

from visualize_dino_attention import DINOAttentionVisualizer

def test_dino_visualization():
    """Test DINO visualization on selected background frames"""
    
    # Paths
    background_frames_dir = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/background_frames")
    output_dir = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/dino_visualization_results")
    
    # Check if background frames exist
    if not background_frames_dir.exists():
        print(f"Error: Background frames directory not found: {background_frames_dir}")
        return
    
    # Get first few frames for testing
    frame_files = list(background_frames_dir.glob("*.jpg"))[:5]
    
    if not frame_files:
        print("No frames found in background directory")
        return
    
    print(f"Found {len(frame_files)} frames for testing")
    print("Testing DINO attention visualization...")
    
    # Create visualizer
    visualizer = DINOAttentionVisualizer(device='auto')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test individual visualization
    print("\n1. Testing individual frame visualization...")
    test_frame = frame_files[0]
    print(f"   Processing: {test_frame.name}")
    
    save_path = output_dir / f"{test_frame.stem}_attention_test.png"
    visualizer.visualize_attention(test_frame, save_path=save_path, show_patches=True)
    
    # Test attention analysis
    print("\n2. Testing attention analysis...")
    stats = visualizer.analyze_frame_attention(test_frame)
    print(f"   Mean attention: {stats['mean_attention']:.4f}")
    print(f"   High attention ratio: {stats['high_attention_ratio']:.4f}")
    print(f"   Top attention patches: {stats['top_5_patches'][:3]}")
    
    # Test grid visualization
    print("\n3. Creating attention grid...")
    grid_path = output_dir / "test_attention_grid.png"
    visualizer.create_attention_grid(frame_files, grid_path, max_images=4)
    
    # Test batch processing
    print("\n4. Processing multiple frames...")
    test_output_dir = output_dir / "batch_test"
    visualizer.process_frame_directory(
        background_frames_dir, 
        test_output_dir,
        visualize_all=False  # Only create grid and summary
    )
    
    print(f"\nVisualization test completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    test_dino_visualization()