import torch
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor

# --- Configuration ---
# Use forward slashes for paths to ensure cross-platform compatibility
FRAMES_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/simple_test/test_frames"
# Load the converted model from the correct directory
MODEL_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final"
OUTPUT_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/simple_test/detr_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.25  # Match YOLO's threshold for a fair comparison

# --- Main Functions ---

def test_detr_on_frames():
    """
    Loads a fine-tuned DETR model, runs inference on all frames in a directory,
    and saves the visualized results.
    """
    print(f"Using device: {DEVICE}")

    # 1. Load Model and Processor
    # Load the fine-tuned model and its processor from the directory created by the conversion script.
    # This ensures the architecture and weights are perfectly matched.
    try:
        print(f"Loading model and processor from: {MODEL_PATH}")
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(DEVICE)
        processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
        model.eval() # Set the model to evaluation mode
        print("Model and processor loaded successfully.")
    except OSError as e:
        print(f"[ERROR] Could not load model from path: {MODEL_PATH}")
        print(f"Please ensure you have run the 'convert_detr_checkpoint.py' script first.")
        print(f"Original error: {e}")
        return

    # 2. Prepare Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # 3. Process Each Frame
    frame_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(frame_files)} frames to process.")

    for frame_filename in frame_files:
        frame_path = os.path.join(FRAMES_DIR, frame_filename)
        print(f"\n--- Processing: {frame_filename} ---")

        # Load and prepare the image
        try:
            image = Image.open(frame_path).convert("RGB")
        except IOError:
            print(f"Could not read image: {frame_path}")
            continue

        # 4. Run Inference
        # The processor handles resizing, normalization, and tensor conversion.
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # 5. Post-process and Visualize Results
        # The processor converts model outputs to bounding boxes and scores.
        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD)[0]

        print(f"Found {len(results['scores'])} objects with confidence > {CONFIDENCE_THRESHOLD}")

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(f"  - Detected object (class: {label.item()}) with score {score.item():.3f} at {box}")
            
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, 
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min, y_min - 15, f'DETR: {score:.2f}', 
                fontsize=14, color='lime', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
            )

        ax.axis('off')
        plt.title(f'DETR Detections on {frame_filename}')
        
        # Save the output image
        output_filename = frame_filename.replace('.jpg', '_detr_result.png')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to free memory
        print(f"Result saved to: {output_path}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    test_detr_on_frames()
