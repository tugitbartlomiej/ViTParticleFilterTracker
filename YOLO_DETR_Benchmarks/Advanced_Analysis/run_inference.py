import yaml
import os
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO

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

def build_coco_image_id_map(annotations_path):
    """
    Build a mapping from image file basename -> COCO image id using ground truth JSON.
    If annotations are unavailable or unreadable, return None.
    """
    try:
        if not annotations_path or not os.path.exists(annotations_path):
            return None
        coco = COCO(annotations_path)
        mapping = {}
        for img in coco.dataset.get('images', []):
            fname = os.path.basename(img.get('file_name', ''))
            if fname:
                mapping[fname] = img['id']
        # Fallback: if mapping is empty, return None
        return mapping or None
    except Exception:
        return None

def get_category_id_mapper(config, model_type):
    """
    Optional label map from model label index -> COCO category id.
    If not provided, defaults to:
      - YOLO: idx + 1 (common case when categories start at 1)
      - DETR: identity (assumes labels already match COCO ids)
    """
    label_map = (config.get('label_map', {}) or {}).get(model_type)
    if isinstance(label_map, dict):
        # Ensure keys are ints (they may come as strings from YAML)
        return {int(k): int(v) for k, v in label_map.items()}
    return None

def run_yolo_inference(config):
    print("Running YOLOv8 inference...")
    model_path = config['models']['yolo']['path']
    images_dir = config['dataset']['images_dir']
    output_dir = config['output']['directory']
    conf_threshold = config['inference_params']['confidence_threshold']

    model = YOLO(model_path)
    image_files = get_image_files(images_dir)

    coco_results = []
    # Prefer COCO image id mapping if GT is available; fallback to enumeration
    gt_map = build_coco_image_id_map(config['dataset'].get('annotations_path'))
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}
    # Optional category mapper
    cat_map = get_category_id_mapper(config, 'yolo')

    # Performance measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for image_name in tqdm(image_files, desc="YOLO Inference"):
        image_path = os.path.join(images_dir, image_name)
        results = model(image_path, conf=conf_threshold, verbose=False)
        
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            # Fallback to enumeration index if not found
            image_id = image_files.index(image_name)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                score = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                # Map class index to COCO category id
                if cat_map is not None:
                    category_id = int(cat_map.get(cls_idx, cls_idx))
                else:
                    category_id = cls_idx + 1  # common default when COCO ids start at 1

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

    # Save performance metrics
    total_time = max(time.time() - start_time, 1e-9)
    fps = len(image_files) / total_time
    vram_mb = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    perf = {"frames": len(image_files), "total_seconds": total_time, "fps": fps, "vram_mb": vram_mb}
    perf_path = os.path.join(output_dir, 'yolo_performance.json')
    with open(perf_path, 'w') as f:
        json.dump(perf, f, indent=4)
    print(f"YOLO performance saved to {perf_path}")

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
    # Prefer COCO image id mapping if GT is available; fallback to enumeration
    gt_map = build_coco_image_id_map(config['dataset'].get('annotations_path'))
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}
    # Optional category mapper
    cat_map = get_category_id_mapper(config, 'detr')

    # Performance measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for image_name in tqdm(image_files, desc="DETR Inference"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]
        
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            image_id = image_files.index(image_name)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            cls_idx = int(label.item())
            if cat_map is not None:
                category_id = int(cat_map.get(cls_idx, cls_idx))
            else:
                category_id = cls_idx  # assume DETR already aligns to COCO ids

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": round(score.item(), 3),
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detr_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)
    print(f"DETR predictions saved to {output_path}")

    # Save performance metrics
    total_time = max(time.time() - start_time, 1e-9)
    fps = len(image_files) / total_time
    vram_mb = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    perf = {"frames": len(image_files), "total_seconds": total_time, "fps": fps, "vram_mb": vram_mb}
    perf_path = os.path.join(output_dir, 'detr_performance.json')
    with open(perf_path, 'w') as f:
        json.dump(perf, f, indent=4)
    print(f"DETR performance saved to {perf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for YOLO or DETR models.")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolo', 'detr'], help='Type of model to run inference for.')
    args = parser.parse_args()

    config = load_config()
    
    if args.model_type == 'yolo':
        run_yolo_inference(config)
    elif args.model_type == 'detr':
        run_detr_inference(config)
