import os

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrConfig
from transformers import DetrForObjectDetection, DetrImageProcessor


def box_cxcywh_to_xyxy(box, img_w, img_h):
    """Convert from center coordinates to corner coordinates.
    Note: No mirroring is applied as this was causing issues in original code."""
    cx, cy, w, h = box
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return [x_min, y_min, x_max, y_max]


def draw_bounding_boxes(image, boxes, scores=None, threshold=0.7):
    """Draw bounding boxes on image with higher threshold and NMS-like processing."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Only keep highest scoring box if multiple boxes are above threshold
    if scores is not None and len(scores) > 0:
        # Sort boxes by score (highest first)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        boxes = [boxes[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        # Only draw highest scoring box above threshold
        if scores[0] >= threshold:
            xmin, ymin, xmax, ymax = boxes[0]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            text = f"Score: {scores[0]:.2f}"
            bbox = draw.textbbox((xmin, ymin), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([xmin, ymin, xmin + text_width, ymin + text_height], fill="red")
            draw.text((xmin, ymin), text, fill="white", font=font)

    return image


def inference_and_sort(model, processor, video_path, output_dir, thresholds, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for threshold in thresholds:
        folder_name = f"{int(threshold * 100)}_{int((threshold + 0.25) * 100)}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(os.path.join(folder_path, "Annotated_Frames"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "Raw_Frames"), exist_ok=True)

    # Read video using OpenCV
    video_capture = cv2.VideoCapture(video_path)
    frame_index = 0

    print(f"Processing video: {video_path}")
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Convert frame to PIL format (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process frame with DETR model
        encoding = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)

        # Calculate probabilities and filter detections
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5  # Lower initial threshold to keep potential detections

        boxes = outputs.pred_boxes[0, keep]
        scores = probas[keep].max(-1).values

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        max_score = scores_np.max() if len(scores_np) > 0 else 0

        if frame_index % 10 == 0:
            print(f"Processing frame {frame_index}/{frame_count}, max confidence: {max_score:.4f}")

        # Sort frames into appropriate folders based on max_score
        for i in range(len(thresholds)):
            if thresholds[i] <= max_score < thresholds[i] + 0.25:
                folder_name = f"{int(thresholds[i] * 100)}_{int((thresholds[i] + 0.25) * 100)}"
                folder_path = os.path.join(output_dir, folder_name)

                # Save original frame (without annotations)
                raw_frame_path = os.path.join(folder_path, "Raw_Frames", f"frame_{frame_index:04d}.jpg")
                image.save(raw_frame_path)

                # Draw annotations on frame and save
                if len(boxes_np) > 0:
                    img_w, img_h = image.size
                    boxes_xyxy = [box_cxcywh_to_xyxy(box, img_w, img_h) for box in boxes_np]

                    # Use higher threshold for visualization - only show confident detections
                    annotated_image = draw_bounding_boxes(image.copy(), boxes_xyxy, scores_np, threshold=0.7)
                    annotated_frame_path = os.path.join(folder_path, "Annotated_Frames", f"frame_{frame_index:04d}.jpg")
                    annotated_image.save(annotated_frame_path)

                break

        frame_index += 1

    video_capture.release()
    print(f"Processing complete. Processed {frame_index} frames.")


def main():
    model_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/detr_training_output/final_model"
    video_path = 'E:/Cataract/videos/micro/train02.mp4'
    output_dir = "output/sorted_frames_mycomp"
    thresholds = [0, 0.25, 0.5, 0.75]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, will be used")

    # Create configuration with num_queries=2 to match training
    config = DetrConfig.from_pretrained(model_dir)
    config.num_queries = 2  # Set to match your training configuration

    # Load model with the modified config and ignore mismatched sizes
    model = DetrForObjectDetection.from_pretrained(
        model_dir,
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    processor = DetrImageProcessor.from_pretrained(model_dir)
    model.eval()

    # Add debugging to confirm configuration
    print(f"Model loaded with num_queries = {model.config.num_queries}")

    inference_and_sort(model, processor, video_path, output_dir, thresholds, device)


if __name__ == "__main__":
    main()