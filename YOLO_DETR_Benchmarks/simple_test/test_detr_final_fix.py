import torch
from PIL import Image
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configuration
FRAMES_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\test_frames"
MODEL_PATH = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth"
OUTPUT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\detr_results_final"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CRITICAL: Match training parameters
NUM_QUERIES = 100  # MUST match training!
NUM_LABELS = 1  # Only 'tool' class (no background class in your dataset)
CONFIDENCE_THRESHOLD = 0.5  # Start with moderate threshold

def load_detr_model():
    """Load DETR model with exact training configuration"""
    print(f"Loading DETR model from: {MODEL_PATH}")
    
    # Load checkpoint to check configuration
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract model state
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    # Initialize model with correct configuration
    from transformers import DetrConfig
    
    config = DetrConfig.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_LABELS,
        num_queries=NUM_QUERIES,  # Critical!
        id2label={0: "tool"},  # Your dataset only has 'tool' class
        label2id={"tool": 0}
    )
    
    model = DetrForObjectDetection(config)
    
    # Load weights
    model.load_state_dict(model_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    # Initialize processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    print(f"Model loaded with num_queries={model.config.num_queries}, num_labels={model.config.num_labels}")
    
    return model, processor

def test_detr_final():
    """Test DETR with corrected configuration"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, processor = load_detr_model()
    
    # Load frame info
    frame_info_path = os.path.join(FRAMES_DIR, 'frame_info.json')
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    print(f"\nTesting DETR with corrected configuration")
    print(f"num_queries: {NUM_QUERIES}")
    print(f"num_labels: {NUM_LABELS}")
    print(f"confidence_threshold: {CONFIDENCE_THRESHOLD}")
    
    # Process frames
    for video_name, frames in frame_info.items():
        print(f"\nProcessing {video_name}")
        
        for frame_data in frames:
            frame_filename = frame_data['filename']
            frame_path = frame_data['path']
            
            if not os.path.exists(frame_path):
                continue
            
            # Load image
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process - this is critical!
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=CONFIDENCE_THRESHOLD
            )[0]
            
            # DETR returns 100 predictions, but most should be filtered by threshold
            print(f"{frame_filename}: {len(results['scores'])} detections above threshold")
            
            # Visualize
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Only draw high-confidence detections
            high_conf_count = 0
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > CONFIDENCE_THRESHOLD:
                    high_conf_count += 1
                    box = [round(i, 2) for i in box.tolist()]
                    x_min, y_min, x_max, y_max = box
                    
                    # Draw box
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(
                        x_min, y_min-10, f'Tool: {score:.2f}',
                        fontsize=10, color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                    )
            
            print(f"  High confidence detections: {high_conf_count}")
            
            ax.axis('off')
            plt.title(f'DETR Final - {frame_filename}')
            
            # Save
            output_filename = frame_filename.replace('.jpg', '_detr_final.png')
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

if __name__ == "__main__":
    print("Testing DETR with final configuration...")
    print(f"Device: {DEVICE}")
    test_detr_final()
    print("\nDone!")