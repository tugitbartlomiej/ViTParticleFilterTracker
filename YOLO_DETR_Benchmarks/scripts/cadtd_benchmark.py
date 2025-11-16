"""
CaDTD Complete Benchmark: YOLO vs DETR
Comprehensive benchmark on CaDTD dataset with VOC annotations
Tests: YOLO epoch 100, DETR epoch 100, DETR epoch 160
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
import subprocess

# PyTorch 2.6+ compatibility: Create dummy DFLoss class if not exists
try:
    from ultralytics.utils.loss import DFLoss
except (ImportError, AttributeError):
    # Create dummy DFLoss for old checkpoints
    import sys
    from types import ModuleType

    class DFLoss:
        """Dummy DFLoss for compatibility with old YOLO checkpoints"""
        pass

    # Add to ultralytics.utils.loss module
    if 'ultralytics.utils.loss' in sys.modules:
        sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss

# Force weights_only=False for all torch.load calls
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that always uses weights_only=False"""
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "Advanced_Analysis"))
from run_inference import load_detr_from_checkpoint
from transformers import DetrImageProcessor

# For YOLO
from ultralytics import YOLO


class CaDTDBenchmark:
    def __init__(self,
                 voc_labels_dir,
                 images_dir,
                 yolo_checkpoint,
                 detr_checkpoints,
                 output_dir):
        """
        Args:
            voc_labels_dir: E:\CaDTD-main\Setup 2\VOC Labels
            images_dir: E:\CaDTD-main\Setup 2\Labels
            yolo_checkpoint: Path to YOLO .pt file
            detr_checkpoints: Dict of {epoch: checkpoint_path}
            output_dir: Output benchmark folder
        """
        self.voc_labels_dir = Path(voc_labels_dir)
        self.images_dir = Path(images_dir)
        self.yolo_checkpoint = Path(yolo_checkpoint)
        self.detr_checkpoints = detr_checkpoints
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        # COCO annotations path (will be created)
        self.coco_json_path = self.output_dir / "cadtd_coco_annotations.json"
        self.coco_gt = None

        # Results
        self.all_results = {}

    def step1_convert_voc_to_coco(self):
        """Convert VOC XML to COCO JSON"""
        print(f"\n{'='*60}")
        print("STEP 1: VOC → COCO Conversion")
        print(f"{'='*60}")

        # Use the converter script
        converter_script = Path(__file__).parent / "voc_to_coco_converter.py"

        cmd = [
            "py", "-3.11", str(converter_script),
            "--voc-labels", str(self.voc_labels_dir),
            "--images", str(self.images_dir),
            "--output", str(self.coco_json_path)
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            raise RuntimeError("VOC to COCO conversion failed")

        print(result.stdout)

        # Load COCO GT
        self.coco_gt = COCO(str(self.coco_json_path))
        print(f"COCO GT loaded: {len(self.coco_gt.getImgIds())} images")

    def step2_run_yolo_inference(self):
        """Run YOLO inference on CaDTD"""
        print(f"\n{'='*60}")
        print("STEP 2: YOLO Inference")
        print(f"{'='*60}")

        # Load YOLO model
        print(f"Loading YOLO from: {self.yolo_checkpoint}")
        model = YOLO(str(self.yolo_checkpoint))

        # Get all images
        image_files = sorted(list(self.images_dir.glob("*.png")))
        print(f"\nFound {len(image_files)} images to process")
        print(f"Processing with YOLO epoch 100...")

        # Image ID mapping
        image_id_map = {}
        for img_info in self.coco_gt.dataset['images']:
            image_id_map[img_info['file_name']] = img_info['id']

        # Predictions
        predictions = []
        detection_count = 0

        print(f"\nStarting YOLO inference on {len(image_files)} images...")
        for idx, img_path in enumerate(tqdm(image_files, desc="YOLO inference", unit="img")):
            # Run YOLO
            results = model.predict(
                source=str(img_path),
                conf=0.5,
                verbose=False
            )

            # Get image_id
            img_name = img_path.name
            image_id = image_id_map.get(img_name, -1)

            if image_id == -1:
                continue

            # Parse results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    predictions.append({
                        "image_id": int(image_id),
                        "category_id": 1,  # surgical_tool
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": conf
                    })
                    detection_count += 1

            # Log progress every 500 images
            if (idx + 1) % 500 == 0:
                print(f"\n  Processed {idx + 1}/{len(image_files)} images, {detection_count} detections so far...")

        # Save predictions
        yolo_output_dir = self.output_dir / "yolo_baseline"
        yolo_output_dir.mkdir(exist_ok=True)

        pred_path = yolo_output_dir / "yolo_predictions.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions, f)

        print(f"\nYOLO inference complete!")
        print(f"  Total detections: {len(predictions)}")
        print(f"  Average detections per image: {len(predictions)/len(image_files):.2f}")
        print(f"  Predictions saved to: {pred_path}")

        # Evaluate
        metrics = self.evaluate_predictions(predictions, "YOLO Epoch 100")
        self.all_results["yolo_baseline"] = {
            "description": "YOLO Epoch 100",
            "predictions_path": str(pred_path),
            **metrics
        }

        # Save performance
        with open(yolo_output_dir / "yolo_performance.json", 'w') as f:
            json.dump(self.all_results["yolo_baseline"], f, indent=2)

        return metrics

    def step3_run_detr_inference(self, checkpoint_path, epoch_num):
        """Run DETR inference on CaDTD"""
        print(f"\n{'='*60}")
        print(f"STEP 3: DETR Epoch {epoch_num} Inference")
        print(f"{'='*60}")

        # Load DETR model
        print(f"Loading DETR from: {checkpoint_path}")
        model = load_detr_from_checkpoint(str(checkpoint_path), self.device)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Get all images
        image_files = sorted(list(self.images_dir.glob("*.png")))
        print(f"\nFound {len(image_files)} images to process")
        print(f"Processing with DETR epoch {epoch_num}...")

        # Image ID mapping
        image_id_map = {}
        for img_info in self.coco_gt.dataset['images']:
            image_id_map[img_info['file_name']] = img_info['id']

        # Predictions
        predictions = []
        detection_count = 0

        print(f"\nStarting DETR epoch {epoch_num} inference on {len(image_files)} images...")
        for idx, img_path in enumerate(tqdm(image_files, desc=f"DETR epoch {epoch_num}", unit="img")):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]

            # Get image_id
            img_name = img_path.name
            image_id = image_id_map.get(img_name, -1)

            if image_id == -1:
                continue

            # Convert to COCO format
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]

                predictions.append({
                    "image_id": int(image_id),
                    "category_id": 1,  # surgical_tool
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item())
                })
                detection_count += 1

            # Log progress every 500 images
            if (idx + 1) % 500 == 0:
                print(f"\n  Processed {idx + 1}/{len(image_files)} images, {detection_count} detections so far...")

        # Save predictions
        detr_output_dir = self.output_dir / f"detr_epoch_{epoch_num}"
        detr_output_dir.mkdir(exist_ok=True)

        pred_path = detr_output_dir / "detr_predictions.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions, f)

        print(f"\nDETR epoch {epoch_num} inference complete!")
        print(f"  Total detections: {len(predictions)}")
        print(f"  Average detections per image: {len(predictions)/len(image_files):.2f}")
        print(f"  Predictions saved to: {pred_path}")

        # Evaluate
        metrics = self.evaluate_predictions(predictions, f"DETR Epoch {epoch_num}")
        self.all_results[f"detr_epoch_{epoch_num}"] = {
            "description": f"DETR Epoch {epoch_num}",
            "checkpoint": str(checkpoint_path),
            "predictions_path": str(pred_path),
            **metrics
        }

        # Save performance
        with open(detr_output_dir / "detr_performance.json", 'w') as f:
            json.dump(self.all_results[f"detr_epoch_{epoch_num}"], f, indent=2)

        return metrics

    def evaluate_predictions(self, predictions, model_name):
        """Evaluate predictions using COCO API"""
        if not predictions:
            print(f"WARNING: No predictions for {model_name}")
            return None

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*60}")
        print(f"Total predictions: {len(predictions)}")

        # Save predictions temporarily
        temp_path = self.output_dir / f"temp_predictions_{model_name.replace(' ', '_')}.json"
        with open(temp_path, 'w') as f:
            json.dump(predictions, f)

        # Load and evaluate
        try:
            pred_coco = self.coco_gt.loadRes(str(temp_path))
            coco_eval = COCOeval(self.coco_gt, pred_coco, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            results = {
                "mAP_0.5": float(coco_eval.stats[1]),
                "mAP_0.5:0.95": float(coco_eval.stats[0]),
                "mAP_0.75": float(coco_eval.stats[2]),
                "AR_max_100": float(coco_eval.stats[8]),
                "AR_small": float(coco_eval.stats[9]),
                "AR_medium": float(coco_eval.stats[10]),
                "AR_large": float(coco_eval.stats[11]),
            }

            # Cleanup
            temp_path.unlink()

            return results

        except Exception as e:
            print(f"ERROR: Evaluation error: {e}")
            return None

    def step4_generate_summary(self):
        """Generate summary JSON"""
        print(f"\n{'='*60}")
        print("STEP 4: Generating Summary")
        print(f"{'='*60}")

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"Summary saved: {summary_path}")

        # Print comparison
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS COMPARISON")
        print(f"{'='*60}")

        print(f"\n{'Model':<25} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'AR@100':<10}")
        print("-" * 65)

        for key, result in self.all_results.items():
            if result is None:
                continue
            desc = result.get('description', key)
            map50 = result.get('mAP_0.5', 0) * 100
            map5095 = result.get('mAP_0.5:0.95', 0) * 100
            ar100 = result.get('AR_max_100', 0) * 100

            print(f"{desc:<25} {map50:<12.2f} {map5095:<15.2f} {ar100:<10.2f}")

        # Determine winner
        yolo_map = self.all_results['yolo_baseline']['mAP_0.5']
        best_detr = max(
            [v for k, v in self.all_results.items() if 'detr' in k],
            key=lambda x: x['mAP_0.5']
        )

        print(f"\n{'='*60}")
        if yolo_map > best_detr['mAP_0.5']:
            gap = (yolo_map - best_detr['mAP_0.5']) * 100
            print(f"WINNER: YOLO (beats DETR by {gap:.2f}%)")
        else:
            gap = (best_detr['mAP_0.5'] - yolo_map) * 100
            print(f"WINNER: {best_detr['description']} (beats YOLO by {gap:.2f}%)")
        print(f"{'='*60}")

    def run_full_benchmark(self):
        """Run complete benchmark pipeline"""
        print(f"\n{'='*80}")
        print("CaDTD COMPLETE BENCHMARK: YOLO vs DETR")
        print(f"{'='*80}")
        print(f"Dataset: CaDTD Setup 2 (Tool Heads)")
        print(f"Models: YOLO epoch 100, DETR epochs {list(self.detr_checkpoints.keys())}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}")

        # Step 1: Convert VOC to COCO
        self.step1_convert_voc_to_coco()

        # Step 2: YOLO inference
        self.step2_run_yolo_inference()

        # Step 3: DETR inference (multiple epochs)
        for epoch, checkpoint_path in self.detr_checkpoints.items():
            if not Path(checkpoint_path).exists():
                print(f"WARNING: DETR checkpoint for epoch {epoch} not found, skipping...")
                continue
            self.step3_run_detr_inference(checkpoint_path, epoch)

        # Step 4: Generate summary
        self.step4_generate_summary()

        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE!")
        print(f"{'='*80}")
        print(f"Next step: Generate visualizations and report")
        print(f"Run:")
        print(f"  cd YOLO_DETR_Benchmarks")
        print(f"  py -3.11 Scripts/generate_benchmark_report.py \\")
        print(f"    --benchmark-dir \"{self.output_dir.relative_to(Path.cwd())}\" \\")
        print(f"    --images-dir \"{self.images_dir}\" \\")
        print(f"    --annotations \"{self.coco_json_path}\"")
        print(f"{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CaDTD Complete Benchmark')
    parser.add_argument('--voc-labels', default=r'E:\CaDTD-main\Setup 2\VOC Labels',
                       help='Path to VOC Labels directory')
    parser.add_argument('--images', default=r'E:\CaDTD-main\Setup 2\Labels',
                       help='Path to Images directory')
    parser.add_argument('--yolo-checkpoint',
                       default='YOLO_DETR_Benchmarks/models/YOLO/epoch100.pt',
                       help='Path to YOLO checkpoint')
    parser.add_argument('--detr-100',
                       default='Eden/Checkpoints/DETR/checkpoint_epoch_100.pth',
                       help='Path to DETR epoch 100 checkpoint')
    parser.add_argument('--detr-160',
                       default='Eden/Checkpoints/DETR/checkpoint_epoch_160.pth',
                       help='Path to DETR epoch 160 checkpoint')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"YOLO_DETR_Benchmarks/Benchmarks/{timestamp}_CaDTD_YOLO_vs_DETR")

    # DETR checkpoints
    detr_checkpoints = {
        100: args.detr_100,
        160: args.detr_160
    }

    # Run benchmark
    benchmark = CaDTDBenchmark(
        voc_labels_dir=args.voc_labels,
        images_dir=args.images,
        yolo_checkpoint=args.yolo_checkpoint,
        detr_checkpoints=detr_checkpoints,
        output_dir=output_dir
    )

    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
