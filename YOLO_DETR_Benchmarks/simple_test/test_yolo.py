import torch
from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
FRAMES_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\test_frames"
MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\YOLO\yolo_inference_model_final\yolo_inference_model.pt"
OUTPUT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\yolo_results"

# YOLO Model Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.25

def load_yolo_model():
    """Load YOLO model from checkpoint"""
    print(f"Loading YOLO model from: {MODEL_PATH}")
    
    try:
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Try alternative model path
        alt_model_path = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\YOLO\best.pt"
        try:
            print(f"Trying alternative model: {alt_model_path}")
            model = YOLO(alt_model_path)
            print("Alternative YOLO model loaded successfully!")
            return model
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            return None

def visualize_yolo_predictions(image, results, save_path, frame_filename):
    """Visualize YOLO predictions on image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Add predictions
    if results and len(results) > 0:
        result = results[0]  # Get first result
        
        # Check if there are any detections
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Get confidences
            classes = result.boxes.cls.cpu().numpy()  # Get class indices
            
            # Draw each detection
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                if conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f'Class {int(cls)}: {conf:.2f}'
                    ax.text(x1, y1-10, label, fontsize=10, color='blue', weight='bold')
            
            print(f"Found {len(boxes)} detections")
        else:
            print("No detections found")
    
    ax.axis('off')
    plt.title(f'YOLO Detection Results - {frame_filename}')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def test_yolo_on_frames():
    """Test YOLO model on extracted frames"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = load_yolo_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load frame information
    frame_info_path = os.path.join(FRAMES_DIR, 'frame_info.json')
    if not os.path.exists(frame_info_path):
        print(f"Frame info file not found: {frame_info_path}")
        print("Please run extract_frames.py first")
        return
    
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    print(f"Testing YOLO on frames from {len(frame_info)} videos")
    
    # Process each video's frames
    for video_name, frames in frame_info.items():
        print(f"\nProcessing frames from video: {video_name}")
        
        for frame_data in frames:
            frame_filename = frame_data['filename']
            frame_path = frame_data['path']
            
            if not os.path.exists(frame_path):
                print(f"Frame file not found: {frame_path}")
                continue
            
            print(f"Processing frame: {frame_filename}")
            
            # Load image
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Failed to load image: {frame_path}")
                continue
            
            # Run inference
            try:
                results = model(image, conf=CONFIDENCE_THRESHOLD)
                print(f"YOLO inference completed")
                
                # Print detection info
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        print(f"Number of detections: {len(result.boxes)}")
                        for i, box in enumerate(result.boxes):
                            conf = box.conf.item()
                            cls = int(box.cls.item())
                            print(f"  Detection {i+1}: Class {cls}, Confidence {conf:.3f}")
                    else:
                        print("No detections found")
                
            except Exception as e:
                print(f"Error during inference: {e}")
                results = None
            
            # Visualize and save results
            output_filename = frame_filename.replace('.jpg', '_yolo_result.png')
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            visualize_yolo_predictions(image, results, output_path, frame_filename)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    print("Testing YOLO model on extracted frames...")
    print(f"Device: {DEVICE}")
    test_yolo_on_frames()
    print("Done!")