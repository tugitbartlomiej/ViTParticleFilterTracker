import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def box_cxcywh_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return [x_min, y_min, x_max, y_max]


def draw_bounding_boxes(image, boxes, scores=None, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        if scores is not None:
            text = f"Score: {scores[i]:.2f}"
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

    # Odczyt video za pomocą OpenCV
    video_capture = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Koniec filmu

        # Konwersja klatki do formatu PIL (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Przetwarzanie klatki za pomocą modelu DETR
        encoding = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**encoding)

        # Obliczanie prawdopodobieństw i filtrowanie detekcji
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9  # Zmodyfikuj próg w razie potrzeby

        boxes = outputs.pred_boxes[0, keep]
        scores = probas[keep].max(-1).values

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        max_score = scores_np.max() if len(scores_np) > 0 else 0
        print(f"Processing frame {frame_index}, max confidence: {max_score:.4f}")

        # Sortowanie klatek do odpowiednich folderów na podstawie max_score
        for i in range(len(thresholds)):
            if thresholds[i] <= max_score < thresholds[i] + 0.25:
                folder_name = f"{int(thresholds[i] * 100)}_{int((thresholds[i] + 0.25) * 100)}"
                folder_path = os.path.join(output_dir, folder_name)

                # Zapis oryginalnej klatki (bez anotacji)
                raw_frame_path = os.path.join(folder_path, "Raw_Frames", f"frame_{frame_index:04d}.jpg")
                image.save(raw_frame_path)

                # Rysowanie anotacji na klatce i zapis
                if len(boxes_np) > 0:
                    img_w, img_h = image.size
                    boxes_xyxy = [box_cxcywh_to_xyxy(box, img_w, img_h) for box in boxes_np]
                    annotated_image = draw_bounding_boxes(image.copy(), boxes_xyxy, scores_np)
                    annotated_frame_path = os.path.join(folder_path, "Annotated_Frames", f"frame_{frame_index:04d}.jpg")
                    annotated_image.save(annotated_frame_path)

                break

        frame_index += 1

    video_capture.release()


def main():
    model_dir = "./detr_tool_tracking_model_best"
    video_path = 'E:/Cataract/videos/micro/train01.mp4'
    output_dir = "output/sorted_frames"
    thresholds = [0, 0.25, 0.5, 0.75]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, will be use")

    model = DetrForObjectDetection.from_pretrained(model_dir).to(device)
    processor = DetrImageProcessor.from_pretrained(model_dir)

    model.eval()

    inference_and_sort(model, processor, video_path, output_dir, thresholds, device)


if __name__ == "__main__":
    main()
