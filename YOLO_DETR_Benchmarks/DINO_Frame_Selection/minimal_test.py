#!/usr/bin/env python3
"""
Minimal test for enhanced selector components
"""

import sys
from pathlib import Path

def test_imports():
    """Test importing required components"""
    print("Testing imports...")
    
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "DINO_Frame_Selection" / "scripts"))
    
    try:
        # Test DINO analyzer import
        from dino_information_analyzer import DINOInformationAnalyzer
        print("  DINO Information Analyzer: OK")
    except Exception as e:
        print(f"  DINO Information Analyzer: FAILED - {e}")
        return False
    
    try:
        # Test basic DINO components
        from dino_feature_extractor import DINOFeatureExtractor
        print("  DINO Feature Extractor: OK")
    except Exception as e:
        print(f"  DINO Feature Extractor: FAILED - {e}")
        # This might fail if the file doesn't exist, which is OK
    
    try:
        # Test model libraries
        from ultralytics import YOLO
        print("  YOLO (ultralytics): OK")
    except Exception as e:
        print(f"  YOLO (ultralytics): FAILED - {e}")
        return False
    
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        print("  DETR (transformers): OK")
    except Exception as e:
        print(f"  DETR (transformers): FAILED - {e}")
        return False
    
    return True

def test_dino_only():
    """Test just DINO functionality"""
    print("\nTesting DINO analyzer standalone...")
    
    # Test frames directory
    test_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/background_frames/train"
    
    if not Path(test_dir).exists():
        print(f"Test directory not found: {test_dir}")
        return False
    
    try:
        from dino_information_analyzer import DINOInformationAnalyzer
        
        # Initialize analyzer
        analyzer = DINOInformationAnalyzer(device='cpu')  # Force CPU for reliability
        
        # Test on one frame
        image_files = list(Path(test_dir).glob('*.jpg'))
        if not image_files:
            print("No test images found")
            return False
        
        test_frame = image_files[0]
        print(f"Testing frame: {test_frame.name}")
        
        result = analyzer.extract_comprehensive_features(str(test_frame))
        
        if result:
            print(f"  Information score: {result['information_score']:.4f}")
            print(f"  Attention available: {result['attention_available']}")
            print("DINO test successful!")
            return True
        else:
            print("DINO analysis failed")
            return False
            
    except Exception as e:
        print(f"DINO test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== MINIMAL COMPONENT TEST ===")
    
    # Test imports first
    if not test_imports():
        print("Import test failed - missing dependencies")
        return 1
    
    # Test DINO analyzer
    if not test_dino_only():
        print("DINO analyzer test failed")
        return 1
    
    print("\n=== All tests passed! ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())