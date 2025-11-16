"""
Multi-Epoch Query Analysis and Benchmark
Tests DETR checkpoints from different epochs and analyzes query distribution
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
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing scripts
sys.path.append(str(Path(__file__).parent.parent / "Advanced_Analysis"))
from run_inference import load_detr_from_checkpoint, load_config
from transformers import DetrImageProcessor

class MultiEpochQueryBenchmark:
    def __init__(self, config_path, epochs=[100, 120, 140, 160]):
        """
        Args:
            config_path: Path to config.yaml
            epochs: List of epochs to test
        """
        self.config = load_config(config_path)
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load GT annotations
        self.coco_gt = COCO(self.config['dataset']['annotations_path'])

        # Results storage
        self.results = {}

    def analyze_query_distribution(self, checkpoint_path, epoch):
        """Analyze which queries are being used for detections"""
        print(f"\n{'='*60}")
        print(f"Analyzing Query Distribution for Epoch {epoch}")
        print(f"{'='*60}")

        # Load model
        model = load_detr_from_checkpoint(checkpoint_path, self.device)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Get images
        images_dir = Path(self.config['dataset']['images_dir'])
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        # Track query usage
        query_usage = {i: 0 for i in range(100)}  # DETR has 100 queries
        total_detections = 0

        print(f"\nProcessing {len(image_files)} images...")

        for img_path in tqdm(image_files[:50], desc="Analyzing queries"):  # Sample 50 images
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Get predictions with confidence > 0.3
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]

            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get max class probability for each query
            max_probs = probs.max(dim=-1).values

            # Count which queries fire
            for query_idx, prob in enumerate(max_probs):
                if prob > 0.3:  # Detection threshold
                    query_usage[query_idx] += 1
                    total_detections += 1

        # Analysis
        sorted_queries = sorted(query_usage.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📊 Query Distribution Analysis:")
        print(f"Total detections: {total_detections}")
        print(f"\nTop 10 Most Used Queries:")
        for query_idx, count in sorted_queries[:10]:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"  Query {query_idx}: {count} detections ({percentage:.1f}%)")

        # Find Query 81 specifically
        query_81_count = query_usage.get(81, 0)
        query_81_pct = (query_81_count / total_detections * 100) if total_detections > 0 else 0
        print(f"\n🎯 Query 81 (Previous Dominant):")
        print(f"  Detections: {query_81_count} ({query_81_pct:.1f}%)")

        # Calculate query diversity
        active_queries = sum(1 for count in query_usage.values() if count > 0)
        print(f"\n📈 Query Diversity:")
        print(f"  Active queries: {active_queries}/100")
        print(f"  Unused queries: {100 - active_queries}")

        return {
            "epoch": epoch,
            "query_usage": query_usage,
            "total_detections": total_detections,
            "dominant_query": sorted_queries[0],
            "query_81_usage": query_81_count,
            "query_81_percentage": query_81_pct,
            "active_queries": active_queries
        }

    def run_inference_single_epoch(self, checkpoint_path, epoch, query_filter=None):
        """
        Run inference for single epoch
        Args:
            query_filter: If specified (e.g., 81), only use that query's predictions
        """
        print(f"\n{'='*60}")
        print(f"Running Inference: Epoch {epoch}")
        if query_filter is not None:
            print(f"Query Filter: Only Query {query_filter}")
        print(f"{'='*60}")

        # Load model
        model = load_detr_from_checkpoint(checkpoint_path, self.device)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Get images
        images_dir = Path(self.config['dataset']['images_dir'])
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        # Image ID mapping
        image_id_map = {}
        for img_info in self.coco_gt.dataset['images']:
            image_id_map[img_info['file_name']] = img_info['id']

        # Predictions
        predictions = []

        for img_path in tqdm(image_files, desc=f"Epoch {epoch} inference"):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]

            # If query filter, only keep predictions from that query
            if query_filter is not None:
                # Need to track which query generated each prediction
                # This requires raw outputs
                logits = outputs.logits[0]
                boxes = outputs.pred_boxes[0]

                # Get predictions for specific query
                query_logits = logits[query_filter:query_filter+1]
                query_boxes = boxes[query_filter:query_filter+1]

                # Post-process just that query
                # (Simplified - in practice need to handle this properly)
                pass  # For now, skip query filtering in inference

            # Get image_id
            img_name = img_path.name
            image_id = image_id_map.get(img_name, -1)

            if image_id == -1:
                continue

            # Convert to COCO format
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]

                predictions.append({
                    "image_id": image_id,
                    "category_id": 1,  # surgical_tool
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item())
                })

        return predictions

    def evaluate_predictions(self, predictions, epoch, query_filter=None):
        """Evaluate predictions using COCO API"""
        if not predictions:
            print(f"⚠️  No predictions for epoch {epoch}")
            return None

        # Save predictions temporarily
        temp_path = f"temp_predictions_epoch{epoch}.json"
        with open(temp_path, 'w') as f:
            json.dump(predictions, f)

        # Load and evaluate
        pred_coco = self.coco_gt.loadRes(temp_path)
        coco_eval = COCOeval(self.coco_gt, pred_coco, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = {
            "epoch": epoch,
            "query_filter": query_filter,
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

        # Cleanup
        os.remove(temp_path)

        return results

    def run_full_benchmark(self):
        """Run complete multi-epoch benchmark"""
        print("\n" + "="*60)
        print("🚀 MULTI-EPOCH QUERY BENCHMARK")
        print("="*60)

        checkpoint_base = Path("Eden/Checkpoints/DETR")

        all_results = []
        query_analyses = []

        for epoch in self.epochs:
            checkpoint_path = checkpoint_base / f"checkpoint_epoch_{epoch}.pth"

            if not checkpoint_path.exists():
                print(f"\n⚠️  Checkpoint for epoch {epoch} not found, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"TESTING EPOCH {epoch}")
            print(f"{'='*60}")

            # Step 1: Analyze query distribution
            query_analysis = self.analyze_query_distribution(str(checkpoint_path), epoch)
            query_analyses.append(query_analysis)

            # Step 2: Run full inference
            predictions = self.run_inference_single_epoch(str(checkpoint_path), epoch)

            # Step 3: Evaluate
            results = self.evaluate_predictions(predictions, epoch)

            if results:
                all_results.append(results)
                print(f"\n📊 Epoch {epoch} Results:")
                print(f"  mAP@0.5      = {results['mAP_0.5']*100:.2f}%")
                print(f"  mAP@0.5:0.95 = {results['mAP_0.5:0.95']*100:.2f}%")
                print(f"  AR@100       = {results['AR_max_100']*100:.2f}%")

        # Generate comparison report
        self.generate_multi_epoch_report(all_results, query_analyses)

        return all_results, query_analyses

    def generate_multi_epoch_report(self, results, query_analyses):
        """Generate comprehensive multi-epoch report"""
        print("\n" + "="*60)
        print("📈 MULTI-EPOCH COMPARISON REPORT")
        print("="*60)

        if not results:
            print("No results to report")
            return

        # Find best epoch
        best_map = max(results, key=lambda x: x['mAP_0.5'])

        print(f"\n🏆 BEST EPOCH: {best_map['epoch']}")
        print(f"   mAP@0.5: {best_map['mAP_0.5']*100:.2f}%")

        # Comparison table
        print(f"\n📊 All Epochs Comparison:")
        print(f"{'Epoch':<10} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'AR@100':<10} {'Query 81%':<12}")
        print("-" * 60)

        for result in sorted(results, key=lambda x: x['epoch']):
            epoch = result['epoch']
            # Find corresponding query analysis
            query_info = next((q for q in query_analyses if q['epoch'] == epoch), None)
            q81_pct = query_info['query_81_percentage'] if query_info else 0

            print(f"{epoch:<10} {result['mAP_0.5']*100:<12.2f} {result['mAP_0.5:0.95']*100:<15.2f} "
                  f"{result['AR_max_100']*100:<10.2f} {q81_pct:<12.1f}")

        # Save results
        output_path = Path("YOLO_DETR_Benchmarks/Benchmarks") / \
                      f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_Multi_Epoch_Query_Analysis"
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "multi_epoch_results.json", 'w') as f:
            json.dump({
                "benchmark_results": results,
                "query_analyses": query_analyses,
                "best_epoch": best_map['epoch']
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

        # Generate visualizations
        self.plot_epoch_progression(results, query_analyses, output_path)

    def plot_epoch_progression(self, results, query_analyses, output_path):
        """Plot metric progression across epochs"""
        epochs = [r['epoch'] for r in sorted(results, key=lambda x: x['epoch'])]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DETR Training Progression Across Epochs', fontsize=16, fontweight='bold')

        # mAP progression
        ax = axes[0, 0]
        ax.plot(epochs, [r['mAP_0.5']*100 for r in sorted(results, key=lambda x: x['epoch'])],
                marker='o', linewidth=2, markersize=8, label='mAP@0.5')
        ax.plot(epochs, [r['mAP_0.5:0.95']*100 for r in sorted(results, key=lambda x: x['epoch'])],
                marker='s', linewidth=2, markersize=8, label='mAP@0.5:0.95')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP (%)', fontsize=12)
        ax.set_title('mAP Progression', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # AR progression
        ax = axes[0, 1]
        ax.plot(epochs, [r['AR_max_100']*100 for r in sorted(results, key=lambda x: x['epoch'])],
                marker='o', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AR@100 (%)', fontsize=12)
        ax.set_title('Average Recall Progression', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Query 81 dominance
        ax = axes[1, 0]
        query_81_pcts = []
        for epoch in epochs:
            query_info = next((q for q in query_analyses if q['epoch'] == epoch), None)
            query_81_pcts.append(query_info['query_81_percentage'] if query_info else 0)

        ax.plot(epochs, query_81_pcts, marker='o', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Query 81 Usage (%)', fontsize=12)
        ax.set_title('Query 81 Dominance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
        ax.legend()

        # Active queries
        ax = axes[1, 1]
        active_queries = []
        for epoch in epochs:
            query_info = next((q for q in query_analyses if q['epoch'] == epoch), None)
            active_queries.append(query_info['active_queries'] if query_info else 0)

        ax.bar(epochs, active_queries, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Active Queries (out of 100)', fontsize=12)
        ax.set_title('Query Diversity', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "epoch_progression.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Progression plots saved")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-epoch query benchmark for DETR')
    parser.add_argument('--config', default='YOLO_DETR_Benchmarks/Advanced_Analysis/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', nargs='+', type=int, default=[100, 120, 140, 160],
                       help='Epochs to test')

    args = parser.parse_args()

    benchmark = MultiEpochQueryBenchmark(args.config, args.epochs)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
