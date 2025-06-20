#!/usr/bin/env python3
"""
Szybki test YOLO - sprawdza czy wszystko działa
"""

import os
import sys

print("="*60)
print("YOLO QUICK TEST")
print("="*60)

# Test 1: Importy
print("\n1. Testing imports...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print(f"✓ YOLO imported successfully")
except Exception as e:
    print(f"✗ YOLO import error: {e}")
    sys.exit(1)

# Test 2: Dataset
print("\n2. Checking dataset...")
dataset_path = "/mnt/evafs/faculty/home/bpiotrowski/datasets/YOLO_03062025/dataset.yaml"
if os.path.exists(dataset_path):
    print(f"✓ Dataset found: {dataset_path}")
    # Spróbuj wczytać yaml
    try:
        import yaml
        with open(dataset_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"  Classes: {data.get('nc', 'unknown')}")
        print(f"  Names: {data.get('names', 'unknown')}")
    except Exception as e:
        print(f"  Warning: Could not parse YAML: {e}")
else:
    print(f"✗ Dataset NOT found!")
    sys.exit(1)

# Test 3: Mini trening (1 epoka, mały batch)
print("\n3. Testing mini training...")
try:
    # Inicjalizacja modelu
    model = YOLO('yolov8n.pt')  # Najmniejszy model
    print("✓ Model initialized")
    
    # Uruchom mini trening
    print("\nStarting 1-epoch test training...")
    results = model.train(
        data=dataset_path,
        epochs=1,        # Tylko 1 epoka
        batch=4,         # Mały batch
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True,
        project='test_run',
        name='quick_test',
        exist_ok=True
    )
    
    print("\n✓ Training test PASSED!")
    print("Your setup is working correctly!")
    
except Exception as e:
    print(f"\n✗ Training test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! Ready for full training.")
print("="*60)