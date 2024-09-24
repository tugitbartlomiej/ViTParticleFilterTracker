import torch
from transformers import TimesformerModel, TimesformerForVideoClassification, AutoImageProcessor
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Timesformer model
model_name = 'facebook/timesformer-base-finetuned-k400'

# Load the model with the classification head
model = TimesformerForVideoClassification.from_pretrained(model_name)

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Function to extract frames from the video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
            frames.append(frame_pil)
    cap.release()
    return frames

# Path to your video
video_path = 'E:/BB formation during DALK in eyes with advanced KC – Video 1 of DALK surgery [ID 277738].mp4'  # Change to the actual path to your video file

# Extract frames from the video
frames = extract_frames(video_path, num_frames=180)  # You can change the number of frames

# Process the frames using the image processor
inputs = image_processor(frames, return_tensors="pt")
pixel_values = inputs['pixel_values']  # [1, num_frames, 3, height, width]

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
pixel_values = pixel_values.to(device)

# Run the model and get predictions
model.eval()
with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits  # [batch_size, num_labels]

    # Compute probabilities and get the predicted class
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = logits.argmax(-1).item()
    confidence = probabilities[0][predicted_class].item()

# Get the class mapping from the model's configuration
kinetics_id2label = model.config.id2label

# Retrieve the predicted label
predicted_label = kinetics_id2label.get(predicted_class, "Nieznana klasa")

# Log the confidence score
print(f"Przewidywana akcja: {predicted_label} (pewność: {confidence:.2f})")

# Handle unknown actions
if predicted_label == "Nieznana klasa":
    print("Akcja nie została rozpoznana w bazie Kinetics-400. Możliwe, że wideo nie pasuje do żadnej klasy akcji.")
