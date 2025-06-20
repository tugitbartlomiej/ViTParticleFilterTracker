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
MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth"
OUTPUT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\detr_results_fixed"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CRITICAL FIX: Use high threshold to filter out false positives
CONFIDENCE_THRESHOLD = 0.5  # Initial threshold for post-processing
MIN_CONFIDENCE_FOR_SINGLE_BOX = 0.9  # Minimum confidence for the single best box
NUM_LABELS = 1  # DETR adds +1, so 1 -> 2 classes in checkpoint

def load_detr_model():
    """Load DETR model with proper configuration"""
    print(f"Loading DETR model from: {MODEL_PATH}")
    
    # Initialize model with correct number of classes - MUST match checkpoint!
    # Checkpoint has 2 classes, not 3
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_LABELS,  # 2 classes
        ignore_mismatched_sizes=True
    )
    
    # Load checkpoint - it has nested structure!
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Navigate nested structure
    if 'model_state_dict' in checkpoint:
        inner_dict = checkpoint['model_state_dict']
        if 'model_state_dict' in inner_dict:
            model_state_dict = inner_dict['model_state_dict']
        else:
            model_state_dict = inner_dict
    else:
        model_state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(model_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    # Initialize processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    return model, processor

def apply_nms(boxes, scores, threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Sort by score
    order = scores_np.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU
        xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
        yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
        xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
        yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
        area = (boxes_np[order[1:], 2] - boxes_np[order[1:], 0]) * (boxes_np[order[1:], 3] - boxes_np[order[1:], 1])
        
        iou = inter / (area_i + area - inter)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep

def test_detr_fixed():
    """Test DETR with improved filtering"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, processor = load_detr_model()
    
    # Load frame info
    frame_info_path = os.path.join(FRAMES_DIR, 'frame_info.json')
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    print(f"Testing DETR with fixed threshold on {len(frame_info)} videos")
    
    # Process each video's frames
    for video_name, frames in frame_info.items():
        print(f"\nProcessing frames from video: {video_name}")
        
        for frame_data in frames:
            frame_filename = frame_data['filename']
            frame_path = frame_data['path']
            
            if not os.path.exists(frame_path):
                continue
            
            print(f"Processing frame: {frame_filename}")
            
            # Load and process image
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process with higher threshold
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=CONFIDENCE_THRESHOLD
            )[0]
            
            # Additional filtering:
            # 1. Remove detections with class 0 (background) if that's how it was trained
            # 2. Apply NMS to remove duplicates
            
            valid_indices = []
            for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
                # With num_labels=1, DETR uses: 0=tool, 1=no_object/background
                # Only keep tool detections (class 0)
                if label.item() == 0:  # Keep tool class (0)
                    valid_indices.append(i)
            
            if valid_indices:
                filtered_scores = results["scores"][valid_indices]
                filtered_labels = results["labels"][valid_indices]
                filtered_boxes = results["boxes"][valid_indices]
                
                # CRITICAL: Take only the SINGLE BEST detection (highest confidence)
                if len(filtered_scores) > 0:
                    best_idx = torch.argmax(filtered_scores)
                    best_score = filtered_scores[best_idx]
                    
                    # Only keep if confidence is high enough
                    if best_score >= MIN_CONFIDENCE_FOR_SINGLE_BOX:
                        final_scores = filtered_scores[best_idx:best_idx+1]
                        final_labels = filtered_labels[best_idx:best_idx+1]
                        final_boxes = filtered_boxes[best_idx:best_idx+1]
                        print(f"  Best detection confidence: {best_score:.3f}")
                    else:
                        print(f"  Best detection confidence {best_score:.3f} < {MIN_CONFIDENCE_FOR_SINGLE_BOX}, discarding")
                        final_scores = torch.tensor([])
                        final_labels = torch.tensor([])
                        final_boxes = torch.tensor([])
                else:
                    final_scores = torch.tensor([])
                    final_labels = torch.tensor([])
                    final_boxes = torch.tensor([])
            else:
                final_scores = torch.tensor([])
                final_labels = torch.tensor([])
                final_boxes = torch.tensor([])
            
            print(f"Detections after filtering: {len(final_scores)}")
            
            # Visualize
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            for score, label, box in zip(final_scores, final_labels, final_boxes):
                box = [round(i, 2) for i in box.tolist()]
                x_min, y_min, x_max, y_max = box
                
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x_min, y_min-10, f'Tool: {score:.2f}',
                    fontsize=10, color='red', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
            
            ax.axis('off')
            plt.title(f'DETR Fixed - {frame_filename}')
            
            # Save
            output_filename = frame_filename.replace('.jpg', '_detr_fixed.png')
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

if __name__ == "__main__":
    print("Testing DETR with improved filtering...")
    print(f"Device: {DEVICE}")
    test_detr_fixed()
    print("Done!")