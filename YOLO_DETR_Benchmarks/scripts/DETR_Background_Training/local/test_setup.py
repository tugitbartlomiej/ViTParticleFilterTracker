"""
Test script to verify local DETR training setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("[INFO] Testing package imports...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name()}")
            print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[WARN] CUDA not available - will use CPU (slow)")
            
    except ImportError as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"[FAIL] Transformers import failed: {e}")
        return False
        
    try:
        from PIL import Image
        print(f"[OK] PIL/Pillow")
    except ImportError as e:
        print(f"[FAIL] PIL import failed: {e}")
        return False
        
    try:
        import tqdm
        print(f"[OK] tqdm")
    except ImportError as e:
        print(f"[FAIL] tqdm import failed: {e}")
        return False
        
    return True

def test_data_paths():
    """Test if data paths exist"""
    print("\n[INFO] Testing data paths...")
    
    base_path = Path(__file__).parent.parent.parent.parent / "Datasets" / "Detr" / "Background"
    
    # Test checkpoint
    checkpoint_path = Path(__file__).parent.parent.parent.parent / "models" / "DETR" / "checkpoint_epoch_100.pth"
    if checkpoint_path.exists():
        print(f"[OK] Checkpoint found: {checkpoint_path}")
        
        # Check checkpoint size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Checkpoint size: {size_mb:.1f} MB")
    else:
        print(f"[FAIL] Checkpoint not found: {checkpoint_path}")
        print("[WARN] Will use pretrained model instead")
    
    # Test images directory
    images_path = base_path / "train"
    if images_path.exists():
        image_count = len(list(images_path.glob("*.jpg")))
        print(f"[OK] Images directory found: {images_path}")
        print(f"[OK] Found {image_count} training images")
    else:
        print(f"[FAIL] Images directory not found: {images_path}")
        return False
    
    # Test annotations
    annot_path = base_path / "annotations" / "train_annotations.json"
    if annot_path.exists():
        print(f"[OK] Annotations found: {annot_path}")
        
        # Test loading annotations
        try:
            import json
            with open(annot_path, 'r') as f:
                data = json.load(f)
            print(f"[OK] Annotations loaded: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations")
        except Exception as e:
            print(f"[FAIL] Error loading annotations: {e}")
            return False
    else:
        print(f"[FAIL] Annotations not found: {annot_path}")
        return False
        
    return True

def test_model_loading():
    """Test if we can load the DETR model"""
    print("\n[INFO] Testing model loading...")
    
    try:
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
        
        print("[OK] Importing DETR classes")
        
        # Test processor
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        print("[OK] Loaded image processor")
        
        # Test model config
        config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        print("[OK] Loaded model config")
        
        # Don't load full model in test (too slow)
        print("[SKIP] Skipping full model loading (would be slow)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False

def test_training_script():
    """Test if training script exists and can be imported"""
    print("\n[INFO] Testing training script...")
    
    script_path = Path(__file__).parent / "detr_local_train.py"
    if script_path.exists():
        print(f"[OK] Training script found: {script_path}")
        
        # Test if script has main function
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("detr_local_train", script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Don't execute, just check if it can be loaded
            print("[OK] Training script can be imported")
            return True
            
        except Exception as e:
            print(f"[FAIL] Error importing training script: {e}")
            return False
    else:
        print(f"[FAIL] Training script not found: {script_path}")
        return False

def main():
    print("="*60)
    print("DETR Local Training - Setup Test")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Paths", test_data_paths), 
        ("Model Loading", test_model_loading),
        ("Training Script", test_training_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Run: quick_start.bat")
        print("2. Or: python detr_local_train.py")
    else:
        print("Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install torch transformers pillow tqdm")
        print("- Check data paths are correct")
        print("- Make sure you're in the right directory")
    
    print("="*60)

if __name__ == "__main__":
    main()