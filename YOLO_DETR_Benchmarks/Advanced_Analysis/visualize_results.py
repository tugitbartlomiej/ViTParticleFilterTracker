import yaml
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_files(images_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    try:
        image_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    except ValueError:
        pass
    return image_files

def draw_boxes(image, boxes, color, label_prefix):
    for box in boxes:
        x, y, w, h = [int(c) for c in box['bbox']]
        score = box.get('score', 1.0)
        label = f"{label_prefix} ({score:.2f})"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def create_side_by_side_video(config):
    print("Creating side-by-side comparison video...")
    output_dir = config['output']['directory']
    images_dir = config['dataset']['images_dir']
    
    with open(os.path.join(output_dir, 'yolo_predictions.json'), 'r') as f:
        yolo_preds = json.load(f)
    with open(os.path.join(output_dir, 'detr_predictions.json'), 'r') as f:
        detr_preds = json.load(f)
    with open(config['dataset']['annotations_path'], 'r') as f:
        gt_data = json.load(f)

    preds_by_image = {'yolo': {}, 'detr': {}}
    for p in yolo_preds:
        preds_by_image['yolo'].setdefault(p['image_id'], []).append(p)
    for p in detr_preds:
        preds_by_image['detr'].setdefault(p['image_id'], []).append(p)
        
    gt_by_image = {}
    for ann in gt_data['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)

    image_files = get_image_files(images_dir)
    # Map filename -> COCO image_id for alignment with predictions
    file_to_id = {img['file_name']: img['id'] for img in gt_data.get('images', [])}

    # Video writer setup
    first_image_path = os.path.join(images_dir, image_files[0])
    h, w, _ = cv2.imread(first_image_path).shape
    video_path = os.path.join(output_dir, 'comparison_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (w * 2, h))

    for img_name in tqdm(image_files, desc="Generating video"):
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        img_id = file_to_id.get(img_name)
        # Fallback: try basename (if GT stores only basename)
        if img_id is None:
            img_id = file_to_id.get(os.path.basename(img_name))
        # If still None, fallback to implicit index (won't match pred ids, but avoids crash)
        if img_id is None:
            img_id = image_files.index(img_name)
        
        yolo_frame = image.copy()
        detr_frame = image.copy()

        # Draw GT boxes
        gt_boxes = gt_by_image.get(img_id, [])
        yolo_frame = draw_boxes(yolo_frame, gt_boxes, (0, 255, 0), "GT")
        detr_frame = draw_boxes(detr_frame, gt_boxes, (0, 255, 0), "GT")

        # Draw prediction boxes
        yolo_frame = draw_boxes(yolo_frame, preds_by_image['yolo'].get(img_id, []), (255, 0, 0), "YOLO")
        detr_frame = draw_boxes(detr_frame, preds_by_image['detr'].get(img_id, []), (0, 0, 255), "DETR")
        
        # Add labels
        cv2.putText(yolo_frame, "YOLOv8", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(detr_frame, "DETR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        combined_frame = np.hstack((yolo_frame, detr_frame))
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Comparison video saved to {video_path}")

def plot_summary_charts(config):
    print("Generating summary charts...")
    output_dir = config['output']['directory']
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(report_path, 'r') as f:
        report = json.load(f)

    labels = list(report.keys())
    map50 = [r['standard_metrics']['mAP_0.5'] for r in report.values()]
    temporal_iou = [r['temporal_metrics']['avg_temporal_iou'] for r in report.values()]
    fps = [r['performance_metrics']['fps'] for r in report.values()]

    # mAP@.50 Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=map50)
    plt.title('mAP@.50 Comparison')
    plt.ylabel('mAP@.50')
    plt.savefig(os.path.join(output_dir, 'map_comparison.png'))
    plt.close()

    # Temporal IoU Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=temporal_iou)
    plt.title('Temporal IoU Stability Comparison')
    plt.ylabel('Average IoU between consecutive frames')
    plt.savefig(os.path.join(output_dir, 'temporal_iou_comparison.png'))
    plt.close()

    # FPS Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=fps)
    plt.title('Performance (FPS) Comparison')
    plt.ylabel('Frames Per Second (Estimated)')
    plt.savefig(os.path.join(output_dir, 'fps_comparison.png'))
    plt.close()
    
    print(f"Summary charts saved to {output_dir}")

if __name__ == "__main__":
    config = load_config()
    create_side_by_side_video(config)
    plot_summary_charts(config)
