#!/usr/bin/env python3
"""
Test Model Loading Script
========================

This script tests the loading of YOLO and DETR models to ensure they work
correctly before running the full background frame selection pipeline.
"""

import os
import sys
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "DINO_Frame_Selection" / "scripts"))

def test_yolo_model():
    """Test YOLO model loading and inference"""
    print("Testing YOLO model...")
    
    yolo_model_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt"
    
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model not found at: {yolo_model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(yolo_model_path)
        print(f"✅ YOLO model loaded successfully from: {yolo_model_path}")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_dummy.jpg", dummy_image)
        
        # Run inference
        results = model("test_dummy.jpg", conf=0.3)
        print(f"✅ YOLO inference successful")
        
        # Check results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"   Detected {len(boxes)} objects")
            else:
                print(f"   No objects detected")
        
        # Clean up
        if os.path.exists("test_dummy.jpg"):
            os.remove("test_dummy.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing YOLO model: {e}")
        return False

def test_detr_model():
    """Test DETR model loading and inference"""
    print("\nTesting DETR model...")
    
    detr_model_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final/model.safetensors"
    
    if not os.path.exists(detr_model_path):
        print(f"❌ DETR model not found at: {detr_model_path}")
        return False
    
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        
        # Load model and processor
        model_dir = Path(detr_model_path).parent
        model = DetrForObjectDetection.from_pretrained(str(model_dir))
        processor = DetrImageProcessor.from_pretrained(str(model_dir))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ DETR model loaded successfully from: {detr_model_path}")
        print(f"   Using device: {device}")
        
        # Test with dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        # Process image
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ DETR inference successful")
        
        # Post-process results
        target_sizes = torch.tensor([dummy_image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )
        
        for result in results:
            scores = result['scores']
            print(f"   Found {len(scores)} detections above threshold")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DETR model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dino_extractor():
    """Test DINO feature extractor"""
    print("\nTesting DINO feature extractor...")
    
    try:
        from dino_feature_extractor import DINOFeatureExtractor
        
        # Initialize extractor
        extractor = DINOFeatureExtractor(
            model_name='dino_vits16',
            device='auto'
        )
        
        print(f"✅ DINO feature extractor initialized successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite("test_dummy_dino.jpg", dummy_image)
        
        # Extract features
        features = extractor.extract_features("test_dummy_dino.jpg")
        
        if features is not None:
            print(f"✅ DINO feature extraction successful")
            print(f"   Feature dimension: {features['feature_dim']}")
            print(f"   Feature norm: {features['feature_norm']:.3f}")
        else:
            print(f"❌ DINO feature extraction failed")
            return False
        
        # Clean up
        if os.path.exists("test_dummy_dino.jpg"):
            os.remove("test_dummy_dino.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DINO extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_access():
    """Test access to video directory"""
    print("\nTesting video directory access...")
    
    videos_dir = "E:/Cataract/videos/micro"
    
    if not os.path.exists(videos_dir):
        print(f"❌ Video directory not found: {videos_dir}")
        return False
    
    # Count video files
    video_files = list(Path(videos_dir).glob("*.mp4"))
    print(f"✅ Video directory accessible: {videos_dir}")
    print(f"   Found {len(video_files)} MP4 files")
    
    # Test opening first video
    if video_files:
        test_video = video_files[0]
        try:
            cap = cv2.VideoCapture(str(test_video))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Test video opened successfully: {test_video.name}")
                print(f"   Frames: {frame_count}, FPS: {fps:.1f}")
                cap.release()
                return True
            else:
                print(f"❌ Could not open test video: {test_video}")
                return False
        except Exception as e:
            print(f"❌ Error opening test video: {e}")
            return False
    
    return True

def main():
    """Main test function"""
    print("INTELLIGENT BACKGROUND FRAME SELECTOR - MODEL LOADING TEST")
    print("=" * 70)
    
    # Test all components
    tests = [
        test_yolo_model,
        test_detr_model,
        test_dino_extractor,
        test_video_access
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "YOLO Model Loading",
        "DETR Model Loading", 
        "DINO Feature Extractor",
        "Video Directory Access"
    ]
    
    all_passed = True
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run background frame selection.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the main script.")
        return 1

if __name__ == "__main__":
    sys.exit(main())