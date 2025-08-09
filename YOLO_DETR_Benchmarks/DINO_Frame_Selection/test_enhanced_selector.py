#!/usr/bin/env python3
"""
Test Enhanced Background Selector
=================================

Test script for the enhanced background selector with DINO integration.
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=== ENHANCED BACKGROUND SELECTOR TEST ===")
    
    # Test video path - use one from the single test
    test_video = "E:/Cataract/videos/micro/test01.mp4"
    
    # Check if video exists
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Looking for alternative videos...")
        
        # Try to find any video in the micro directory
        micro_dir = Path("E:/Cataract/videos/micro")
        if micro_dir.exists():
            video_files = list(micro_dir.glob('*.mp4'))
            if video_files:
                test_video = str(video_files[0])
                print(f"Using alternative video: {test_video}")
            else:
                print("No MP4 videos found in micro directory")
                return 1
        else:
            print("Micro video directory not found")
            return 1
    
    # Output directory
    output_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DINO_Frame_Selection/test_output"
    
    print(f"Test video: {test_video}")
    print(f"Output directory: {output_dir}")
    
    # Import enhanced selector
    try:
        from enhanced_background_selector import EnhancedBackgroundSelector
        print("Successfully imported EnhancedBackgroundSelector")
    except Exception as e:
        print(f"Failed to import EnhancedBackgroundSelector: {e}")
        return 1
    
    # Initialize selector
    try:
        print("Initializing enhanced background selector...")
        selector = EnhancedBackgroundSelector(
            output_dir=output_dir,
            device='auto',
            use_dino_quality=True
        )
        
        # Configure for quick test
        selector.config.update({
            'yolo_threshold': 0.3,
            'detr_threshold': 0.8,
            'frame_interval': 60,  # Every 60 frames for quick test
            'max_frames_per_video': 50,  # Limit for testing
            'info_quality_threshold': 0.3,  # Lower threshold
            'max_background_frames': 20,
            'min_background_frames': 5
        })
        
        print("Enhanced selector initialized")
        
    except Exception as e:
        print(f"Failed to initialize selector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load models
    try:
        print("Loading models...")
        if not selector.load_models():
            print("Failed to load models")
            return 1
        print("Models loaded successfully")
    except Exception as e:
        print(f"Model loading error: {e}")
        return 1
    
    # Process single video
    try:
        print(f"Processing video: {Path(test_video).name}")
        selected_frames = selector.process_single_video(test_video)
        
        if selected_frames:
            print(f"Selected {len(selected_frames)} high-quality background frames")
            
            # Show quality scores
            scores = [f.get('information_score', 0.0) for f in selected_frames]
            if scores:
                print(f"Quality score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"Average quality: {sum(scores)/len(scores):.4f}")
            
            # Create dataset
            selector.save_background_dataset(selected_frames)
            
            print("Test completed successfully!")
            print(f"Results saved to: {output_dir}")
            return 0
            
        else:
            print("No background frames selected")
            return 1
            
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())