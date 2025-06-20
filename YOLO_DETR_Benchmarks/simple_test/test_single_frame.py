import torch
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configuration
FRAME_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/simple_test/test_frames/test01_frame_003867.jpg"
# The model should be loaded from the directory created by the conversion script
MODEL_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final"
OUTPUT_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/simple_test/test_single_detr.png"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned model and processor from the specified directory
print(f"Loading fine-tuned model and processor from: {MODEL_PATH}")
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(DEVICE)
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model.eval()
print("Model and processor loaded successfully.")


print("Loading and processing image...")
image = cv2.imread(FRAME_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image)

# Process the image
inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)

print("Running inference...")
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the output
target_sizes = torch.tensor([pil_image.size[::-1]]).to(DEVICE)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

print(f"Found {len(results['scores'])} objects with score > 0.5")

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(pil_image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected object with score {round(score.item(), 3)} at location {box}")

    x_min, y_min, x_max, y_max = box
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x_min, y_min - 15, f'DETR: {score:.2f}', fontsize=14, color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.axis('off')
plt.title('DETR Detection')
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=150)
plt.close()

print(f"Saved corrected visualization to: {OUTPUT_PATH}")