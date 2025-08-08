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
TOOLTIP_MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth"
FINETUNED_MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\DETR_Background_Training\local\checkpoints_local\final_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tooltip_model():
    """Load original tooltip model"""
    print(f"Loading TOOLTIP model from: {TOOLTIP_MODEL_PATH}")
    
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,  # 2 classes total in checkpoint  
        ignore_mismatched_sizes=True
    )
    
    checkpoint = torch.load(TOOLTIP_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        if 'model_state_dict' in checkpoint['model_state_dict']:
            model_state_dict = checkpoint['model_state_dict']['model_state_dict'] 
        else:
            model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    print("TOOLTIP model: 2 classes (tool=0, no_object=1)")
    return model, processor

def load_finetuned_model():
    """Load finetuned model"""
    print(f"Loading FINETUNED model from: {FINETUNED_MODEL_PATH}")
    
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=2,  # 3 classes total in checkpoint
        ignore_mismatched_sizes=True
    )
    
    checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        if 'model_state_dict' in checkpoint['model_state_dict']:
            model_state_dict = checkpoint['model_state_dict']['model_state_dict'] 
        else:
            model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    print("FINETUNED model: 3 classes (surgical_tool=0, background=1, no_object=2)")
    return model, processor

def analyze_predictions(image_path, model_tooltip, model_finetuned, processor):
    """Compare predictions of both models on the same image"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    print(f"Image size: {image.size}")
    
    # ============== TOOLTIP MODEL ==============
    with torch.no_grad():
        outputs_tooltip = model_tooltip(**inputs)
    
    logits_tooltip = outputs_tooltip.logits[0]  # [100, 2]
    probs_tooltip = torch.softmax(logits_tooltip, dim=-1)
    
    print(f"\n--- TOOLTIP MODEL ANALYSIS ---")
    print(f"Logits shape: {logits_tooltip.shape}")
    print(f"Probs range: {probs_tooltip.min():.3f} to {probs_tooltip.max():.3f}")
    
    # Top predictions for each class
    for class_id in range(2):
        class_name = "tool" if class_id == 0 else "no_object"
        top_probs, top_indices = torch.topk(probs_tooltip[:, class_id], k=3)
        print(f"Class {class_id} ({class_name}) - Top 3: {[f'{p:.4f}' for p in top_probs.tolist()]}")
    
    # Post-processing with different thresholds
    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    
    print(f"Post-processing thresholds:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        results = processor.post_process_object_detection(
            outputs_tooltip, target_sizes=target_sizes, threshold=threshold
        )[0]
        valid_detections = sum(1 for label in results['labels'] if label.item() == 0)  # tool class
        if len(results['scores']) > 0:
            print(f"  Threshold {threshold}: {len(results['scores'])} total, {valid_detections} tools, best score: {results['scores'][0]:.3f}")
        else:
            print(f"  Threshold {threshold}: 0 detections")
    
    # ============== FINETUNED MODEL ==============
    with torch.no_grad():
        outputs_finetuned = model_finetuned(**inputs)
    
    logits_finetuned = outputs_finetuned.logits[0]  # [100, 3]
    probs_finetuned = torch.softmax(logits_finetuned, dim=-1)
    
    print(f"\n--- FINETUNED MODEL ANALYSIS ---")
    print(f"Logits shape: {logits_finetuned.shape}")
    print(f"Probs range: {probs_finetuned.min():.3f} to {probs_finetuned.max():.3f}")
    
    # Top predictions for each class
    for class_id in range(3):
        class_name = ["surgical_tool", "background", "no_object"][class_id]
        top_probs, top_indices = torch.topk(probs_finetuned[:, class_id], k=3)
        print(f"Class {class_id} ({class_name}) - Top 3: {[f'{p:.4f}' for p in top_probs.tolist()]}")
    
    # Post-processing with different thresholds
    print(f"Post-processing thresholds:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        results = processor.post_process_object_detection(
            outputs_finetuned, target_sizes=target_sizes, threshold=threshold
        )[0]
        tool_detections = sum(1 for label in results['labels'] if label.item() == 0)  # surgical_tool class
        background_detections = sum(1 for label in results['labels'] if label.item() == 1)  # background class
        if len(results['scores']) > 0:
            print(f"  Threshold {threshold}: {len(results['scores'])} total, {tool_detections} surgical_tools, {background_detections} backgrounds, best: {results['scores'][0]:.3f}")
            # Show class distribution of top detections
            if len(results['labels']) > 0:
                top_5_labels = results['labels'][:5].tolist()
                top_5_scores = [f"{s:.3f}" for s in results['scores'][:5].tolist()]
                print(f"    Top 5: labels={top_5_labels}, scores={top_5_scores}")
        else:
            print(f"  Threshold {threshold}: 0 detections")
    
    # ============== DIRECT COMPARISON ==============
    print(f"\n--- DIRECT COMPARISON ---")
    
    # Find best tool detection in tooltip model
    tooltip_tool_probs = probs_tooltip[:, 0]  # tool class
    best_tooltip_idx = torch.argmax(tooltip_tool_probs)
    best_tooltip_score = tooltip_tool_probs[best_tooltip_idx].item()
    
    # Find best surgical_tool detection in finetuned model
    finetuned_tool_probs = probs_finetuned[:, 0]  # surgical_tool class
    best_finetuned_idx = torch.argmax(finetuned_tool_probs)
    best_finetuned_score = finetuned_tool_probs[best_finetuned_idx].item()
    
    print(f"Best tool detection:")
    print(f"  TOOLTIP model: query {best_tooltip_idx}, score {best_tooltip_score:.4f}")
    print(f"  FINETUNED model: query {best_finetuned_idx}, score {best_finetuned_score:.4f}")
    print(f"  DIFFERENCE: {best_finetuned_score - best_tooltip_score:+.4f}")
    
    # Check if same query gives different results
    if best_tooltip_idx == best_finetuned_idx:
        print(f"  Same query index! Comparing class distributions:")
        tooltip_dist = probs_tooltip[best_tooltip_idx]
        finetuned_dist = probs_finetuned[best_finetuned_idx]
        print(f"    TOOLTIP:   tool={tooltip_dist[0]:.4f}, no_object={tooltip_dist[1]:.4f}")
        print(f"    FINETUNED: surgical_tool={finetuned_dist[0]:.4f}, background={finetuned_dist[1]:.4f}, no_object={finetuned_dist[2]:.4f}")

def main():
    """Run deep analysis"""
    print("=== DEEP ANALYSIS: TOOLTIP vs FINETUNED MODEL ===")
    
    # Load models
    model_tooltip, processor = load_tooltip_model()
    model_finetuned, processor = load_finetuned_model()
    
    # Load frame info
    frame_info_path = os.path.join(FRAMES_DIR, 'frame_info.json')
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    # Analyze first few frames
    frame_count = 0
    for video_name, frames in frame_info.items():
        for frame_data in frames:
            if frame_count >= 3:  # Analyze first 3 frames only
                break
                
            frame_path = frame_data['path']
            if os.path.exists(frame_path):
                analyze_predictions(frame_path, model_tooltip, model_finetuned, processor)
                frame_count += 1
        
        if frame_count >= 3:
            break
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()