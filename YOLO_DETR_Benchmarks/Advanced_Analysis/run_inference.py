import yaml
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_files(images_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Sortowanie numeryczne, jeśli pliki mają nazwy typu frame_1.jpg, frame_10.jpg
    try:
        image_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    except ValueError:
        pass # Standardowe sortowanie alfabetyczne
    return image_files

def run_yolo_inference(config):
    print("Running YOLOv8 inference...")
    model_path = config['models']['yolo']['path']
    images_dir = config['dataset']['images_dir']
    output_dir = config['output']['directory']
    conf_threshold = config['inference_params']['confidence_threshold']

    model = YOLO(model_path)
    image_files = get_image_files(images_dir)
    
    coco_results = []
    image_id_map = {name: i for i, name in enumerate(image_files)}

    for image_name in tqdm(image_files, desc="YOLO Inference"):
        image_path = os.path.join(images_dir, image_name)
        results = model(image_path, conf=conf_threshold, verbose=False)
        
        image_id = image_id_map[image_name]

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                score = box.conf[0].item()
                category_id = int(box.cls[0].item()) + 1 # Zakładając, że klasy YOLO zaczynają się od 0

                coco_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "score": score,
                })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'yolo_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)
    print(f"YOLO predictions saved to {output_path}")

def run_detr_inference(config):
    print("Running DETR inference...")
    model_path = config['models']['detr']['path']
    images_dir = config['dataset']['images_dir']
    output_dir = config['output']['directory']
    conf_threshold = config['inference_params']['confidence_threshold']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    
    image_files = get_image_files(images_dir)
    coco_results = []
    image_id_map = {name: i for i, name in enumerate(image_files)}

    for image_name in tqdm(image_files, desc="DETR Inference"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]
        
        image_id = image_id_map[image_name]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            coco_results.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [x1, y1, width, height],
                "score": round(score.item(), 3),
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detr_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)
    print(f"DETR predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for YOLO or DETR models.")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolo', 'detr'], help='Type of model to run inference for.')
    args = parser.parse_args()

    config = load_config()
    
    if args.model_type == 'yolo':
        run_yolo_inference(config)
    elif args.model_type == 'detr':
        run_detr_inference(config)
