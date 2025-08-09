#!/usr/bin/env python3
"""
Simple test for DINO Information Analyzer
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=== DINO ANALYZER SIMPLE TEST ===")
    
    # Test directory with frames
    test_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/background_frames/train"
    
    # Check if directory exists and has images
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Test directory not found: {test_dir}")
        return 1
    
    image_files = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
    if not image_files:
        print(f"No images found in: {test_dir}")
        return 1
    
    print(f"Found {len(image_files)} images in test directory")
    
    # Try to import and initialize DINO analyzer
    try:
        from dino_information_analyzer import DINOInformationAnalyzer
        print("Successfully imported DINOInformationAnalyzer")
    except Exception as e:
        print(f"Failed to import DINOInformationAnalyzer: {e}")
        return 1
    
    # Initialize analyzer
    try:
        print("Initializing DINO analyzer...")
        analyzer = DINOInformationAnalyzer(device='auto')
        print("DINO analyzer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test on first few frames
    test_frames = image_files[:3]  # Test first 3 frames
    results = []
    
    for frame_path in test_frames:
        print(f"\nTesting frame: {frame_path.name}")
        try:
            result = analyzer.extract_comprehensive_features(str(frame_path))
            if result:
                info_score = result['information_score']
                attention_available = result['attention_available']
                print(f"  Information score: {info_score:.4f}")
                print(f"  Attention available: {attention_available}")
                results.append(result)
            else:
                print("  Analysis failed")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if results:
        scores = [r['information_score'] for r in results]
        print(f"\n=== Test Results ===")
        print(f"Frames analyzed: {len(results)}")
        print(f"Average score: {sum(scores)/len(scores):.4f}")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print("Test completed successfully!")
        return 0
    else:
        print("No successful analyses")
        return 1

if __name__ == "__main__":
    sys.exit(main())