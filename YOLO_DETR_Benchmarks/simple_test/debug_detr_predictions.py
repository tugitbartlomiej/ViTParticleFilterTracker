import torch
from PIL import Image
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configuration
FRAMES_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\test_frames"
MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\DETR_Background_Training\local\checkpoints\best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_detr_model():
    """Load DETR model with proper configuration"""
    print(f"Loading DETR model from: {MODEL_PATH}")
    
    # Initialize model
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=2,  # 3 classes total in checkpoint
        ignore_mismatched_sizes=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    
    # Initialize processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    return model, processor

def debug_single_frame():
    """Debug predictions on a single frame"""
    
    # Load model
    model, processor = load_detr_model()
    
    # Load frame info
    frame_info_path = os.path.join(FRAMES_DIR, 'frame_info.json')
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    # Take first frame from first video
    first_video = list(frame_info.keys())[0]
    first_frame = frame_info[first_video][0]
    frame_path = first_frame['path']
    frame_filename = first_frame['filename']
    
    print(f"\nDebugging frame: {frame_filename}")
    print(f"Path: {frame_path}")
    
    # Load image
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    print(f"Image size: {image.size}")
    print(f"Input tensor shape: {inputs.pixel_values.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nRaw model outputs:")
    print(f"logits shape: {outputs.logits.shape}")  # Should be [1, 100, 3]
    print(f"pred_boxes shape: {outputs.pred_boxes.shape}")  # Should be [1, 100, 4]
    
    # Check class predictions (before softmax)
    logits = outputs.logits[0]  # Shape: [100, 3]
    probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
    
    print(f"\nClass probabilities analysis:")
    print(f"Logits range: {logits.min():.3f} to {logits.max():.3f}")
    print(f"Probs range: {probs.min():.3f} to {probs.max():.3f}")
    
    # Show top predictions for each class
    for class_id in range(3):
        top_probs, top_indices = torch.topk(probs[:, class_id], k=5)
        print(f"Class {class_id} - Top 5 probabilities: {top_probs.tolist()}")
    
    # Check what post-processing gives us with very low threshold
    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    
    print(f"\nTesting different thresholds:")
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=threshold
        )[0]
        
        print(f"Threshold {threshold}: {len(results['scores'])} detections")
        if len(results['scores']) > 0:
            print(f"  Labels: {results['labels'][:5].tolist()}")
            print(f"  Scores: {[f'{s:.3f}' for s in results['scores'][:5].tolist()]}")

if __name__ == "__main__":
    print("Debugging DETR predictions...")
    print(f"Device: {DEVICE}")
    debug_single_frame()
    print("\nDone!")