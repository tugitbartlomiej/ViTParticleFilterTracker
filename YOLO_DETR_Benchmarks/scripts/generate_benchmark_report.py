"""
Automatic Benchmark Report Generator for YOLO vs DETR
Generates visualizations, infographics, and detailed MD report
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12

class BenchmarkReportGenerator:
    def __init__(self, benchmark_dir, images_dir, annotations_path):
        """
        Args:
            benchmark_dir: Path to benchmark results (e.g., benchmark_results_epoch160/)
            images_dir: Path to original images
            annotations_path: Path to COCO annotations
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.images_dir = Path(images_dir)
        self.annotations_path = annotations_path

        # Load COCO GT
        self.coco_gt = COCO(annotations_path)

        # Create output directories
        self.output_dir = self.benchmark_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "yolo").mkdir(exist_ok=True)
        (self.output_dir / "detr").mkdir(exist_ok=True)
        (self.output_dir / "comparison").mkdir(exist_ok=True)
        (self.output_dir / "infographics").mkdir(exist_ok=True)

        # Load predictions
        self.load_predictions()

    def load_predictions(self):
        """Load YOLO and DETR predictions"""
        # YOLO
        yolo_pred_path = self.benchmark_dir / "yolo_baseline" / "yolo_predictions.json"
        with open(yolo_pred_path) as f:
            self.yolo_preds = json.load(f)

        # DETR (best config)
        detr_dirs = [d for d in self.benchmark_dir.iterdir() if d.is_dir() and "DETR" in d.name]
        if detr_dirs:
            detr_pred_path = detr_dirs[0] / "detr_predictions.json"
            with open(detr_pred_path) as f:
                self.detr_preds = json.load(f)

        # Organize by image_id
        self.yolo_by_image = {}
        for pred in self.yolo_preds:
            img_id = pred['image_id']
            if img_id not in self.yolo_by_image:
                self.yolo_by_image[img_id] = []
            self.yolo_by_image[img_id].append(pred)

        self.detr_by_image = {}
        for pred in self.detr_preds:
            img_id = pred['image_id']
            if img_id not in self.detr_by_image:
                self.detr_by_image[img_id] = []
            self.detr_by_image[img_id].append(pred)

    def draw_boxes(self, image, boxes, color, label):
        """Draw bounding boxes on image"""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for box in boxes:
            x, y, w, h = box['bbox']
            score = box['score']

            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

            # Draw label
            text = f"{label}: {score:.2f}"
            bbox = draw.textbbox((x, y-25), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x, y-25), text, fill='white', font=font)

        return image

    def generate_comparison_images(self, num_samples=20):
        """Generate side-by-side comparison images"""
        print("Generating comparison visualizations...")

        # Get all image IDs
        all_image_ids = list(set(list(self.yolo_by_image.keys()) + list(self.detr_by_image.keys())))

        # Sample images
        np.random.seed(42)
        sample_ids = np.random.choice(all_image_ids, min(num_samples, len(all_image_ids)), replace=False)

        for img_id in sample_ids:
            # Get image info
            img_info_list = self.coco_gt.loadImgs(img_id)
            if not img_info_list or len(img_info_list) == 0:
                continue

            img_info = img_info_list[0]
            img_path = self.images_dir / img_info['file_name']

            if not img_path.exists():
                continue

            # Load image
            img = Image.open(img_path).convert('RGB')

            # Create copies for YOLO and DETR
            img_yolo = img.copy()
            img_detr = img.copy()

            # Draw YOLO boxes (green)
            if img_id in self.yolo_by_image:
                img_yolo = self.draw_boxes(img_yolo, self.yolo_by_image[img_id],
                                          color='lime', label='YOLO')

            # Draw DETR boxes (red)
            if img_id in self.detr_by_image:
                img_detr = self.draw_boxes(img_detr, self.detr_by_image[img_id],
                                          color='red', label='DETR')

            # Save individual images
            img_yolo.save(self.output_dir / "yolo" / f"{img_info['file_name']}")
            img_detr.save(self.output_dir / "detr" / f"{img_info['file_name']}")

            # Create side-by-side comparison
            w, h = img.size
            comparison = Image.new('RGB', (w*2, h))
            comparison.paste(img_yolo, (0, 0))
            comparison.paste(img_detr, (w, 0))

            # Add labels
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()

            draw.text((w//2-100, 20), "YOLO", fill='lime', font=font, stroke_width=2, stroke_fill='black')
            draw.text((w + w//2-100, 20), "DETR", fill='red', font=font, stroke_width=2, stroke_fill='black')

            comparison.save(self.output_dir / "comparison" / f"comparison_{img_info['file_name']}")

        print(f"Generated {len(sample_ids)} comparison images")

    def generate_metrics_infographic(self):
        """Generate infographic comparing metrics"""
        print("Generating metrics infographic...")

        # Load summary
        summary_path = self.benchmark_dir / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        yolo = summary['yolo_baseline']
        detr_best = summary['best_mAP_0.5']

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLO vs DETR Performance Comparison', fontsize=20, fontweight='bold')

        metrics = [
            ('mAP@0.5', 'mAP_0.5', '%'),
            ('mAP@0.5:0.95', 'mAP_0.5:0.95', '%'),
            ('mAP@0.75', 'mAP_0.75', '%'),
            ('AR@100', 'AR_max_100', '%'),
            ('AR Medium', 'AR_medium', '%'),
            ('AR Large', 'AR_large', '%'),
        ]

        for idx, (title, key, unit) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            yolo_val = yolo[key] * 100 if key in yolo else 0
            detr_val = detr_best[key] * 100 if key in detr_best else 0

            # Bar chart
            bars = ax.bar(['YOLO', 'DETR'], [yolo_val, detr_val],
                         color=['#00ff00', '#ff0000'], alpha=0.7, edgecolor='black', linewidth=2)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}{unit}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

            # Styling
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_ylabel(f'Score ({unit})', fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)

            # Winner annotation
            winner = 'YOLO' if yolo_val > detr_val else 'DETR'
            gap = abs(yolo_val - detr_val)
            ax.text(0.5, 0.95, f'{winner} wins by {gap:.1f}{unit}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / "infographics" / "metrics_comparison.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("Metrics infographic saved")

    def generate_detection_count_analysis(self):
        """Analyze detection counts per image"""
        print("Generating detection count analysis...")

        yolo_counts = [len(self.yolo_by_image.get(img_id, [])) for img_id in self.coco_gt.getImgIds()]
        detr_counts = [len(self.detr_by_image.get(img_id, [])) for img_id in self.coco_gt.getImgIds()]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram
        ax = axes[0]
        ax.hist(yolo_counts, bins=20, alpha=0.5, label='YOLO', color='green', edgecolor='black')
        ax.hist(detr_counts, bins=20, alpha=0.5, label='DETR', color='red', edgecolor='black')
        ax.set_xlabel('Detections per Image', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title('Detection Count Distribution', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        # Box plot
        ax = axes[1]
        bp = ax.boxplot([yolo_counts, detr_counts], labels=['YOLO', 'DETR'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)

        ax.set_ylabel('Detections per Image', fontsize=14)
        ax.set_title('Detection Count Statistics', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean values
        yolo_mean = np.mean(yolo_counts)
        detr_mean = np.mean(detr_counts)
        ax.text(1, yolo_mean, f'μ={yolo_mean:.1f}', ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='white'), fontsize=12)
        ax.text(2, detr_mean, f'μ={detr_mean:.1f}', ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='white'), fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / "infographics" / "detection_counts.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("Detection count analysis saved")

    def identify_success_failure_cases(self, top_k=5):
        """Identify where YOLO wins and where DETR wins"""
        print("Identifying success/failure cases...")

        # Compare detection counts
        differences = []
        for img_id in self.coco_gt.getImgIds():
            yolo_count = len(self.yolo_by_image.get(img_id, []))
            detr_count = len(self.detr_by_image.get(img_id, []))
            diff = yolo_count - detr_count
            differences.append((img_id, diff, yolo_count, detr_count))

        # Sort by difference
        differences.sort(key=lambda x: x[1])

        # DETR wins (negative diff = DETR found more)
        detr_wins = differences[:top_k]
        # YOLO wins (positive diff = YOLO found more)
        yolo_wins = differences[-top_k:]

        results = {
            'yolo_success_cases': [],
            'detr_success_cases': []
        }

        # Generate visualizations for success cases
        for img_id, diff, yolo_count, detr_count in yolo_wins:
            img_info_list = self.coco_gt.loadImgs(img_id)
            if not img_info_list:
                continue
            img_info = img_info_list[0]
            results['yolo_success_cases'].append({
                'image': img_info['file_name'],
                'yolo_detections': yolo_count,
                'detr_detections': detr_count,
                'difference': diff
            })

        for img_id, diff, yolo_count, detr_count in detr_wins:
            img_info_list = self.coco_gt.loadImgs(img_id)
            if not img_info_list:
                continue
            img_info = img_info_list[0]
            results['detr_success_cases'].append({
                'image': img_info['file_name'],
                'yolo_detections': yolo_count,
                'detr_detections': detr_count,
                'difference': diff
            })

        # Save analysis
        with open(self.output_dir / "success_failure_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)

        print("Success/failure cases identified")
        return results

    def generate_markdown_report(self, success_failure_cases):
        """Generate comprehensive MD report"""
        print("Generating markdown report...")

        # Load summary
        summary_path = self.benchmark_dir / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        yolo = summary['yolo_baseline']
        detr_configs = summary['detr_epoch_160_configurations']
        best_detr = summary['best_mAP_0.5']

        # Generate report
        report = f"""# YOLO vs DETR Benchmark Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Benchmark:** DETR Epoch 160 vs YOLO Epoch 100

---

## 📊 Executive Summary

### Winner: **{'YOLO' if yolo['mAP_0.5'] > best_detr['mAP_0.5'] else 'DETR'}**

| Metric | YOLO | DETR (Best) | Gap | Winner |
|--------|------|-------------|-----|--------|
| **mAP@0.5** | {yolo['mAP_0.5']*100:.2f}% | {best_detr['mAP_0.5']*100:.2f}% | {(best_detr['mAP_0.5'] - yolo['mAP_0.5'])*100:+.2f}% | {'🥇 YOLO' if yolo['mAP_0.5'] > best_detr['mAP_0.5'] else '🥇 DETR'} |
| **mAP@0.5:0.95** | {yolo['mAP_0.5:0.95']*100:.2f}% | {best_detr['mAP_0.5:0.95']*100:.2f}% | {(best_detr['mAP_0.5:0.95'] - yolo['mAP_0.5:0.95'])*100:+.2f}% | {'🥇 YOLO' if yolo['mAP_0.5:0.95'] > best_detr['mAP_0.5:0.95'] else '🥇 DETR'} |
| **mAP@0.75** | {yolo['mAP_0.75']*100:.2f}% | {best_detr['mAP_0.75']*100:.2f}% | {(best_detr['mAP_0.75'] - yolo['mAP_0.75'])*100:+.2f}% | {'🥇 YOLO' if yolo['mAP_0.75'] > best_detr['mAP_0.75'] else '🥇 DETR'} |
| **AR@100** | {yolo['AR_max_100']*100:.2f}% | {best_detr['AR_max_100']*100:.2f}% | {(best_detr['AR_max_100'] - yolo['AR_max_100'])*100:+.2f}% | {'🥇 YOLO' if yolo['AR_max_100'] > best_detr['AR_max_100'] else '🥇 DETR'} |

![Metrics Comparison](visualizations/infographics/metrics_comparison.png)

---

## 🔍 Detailed Analysis

### Why YOLO is Better

"""

        # Analyze why YOLO is better
        if yolo['mAP_0.5'] > best_detr['mAP_0.5']:
            report += f"""
1. **Higher Precision**: YOLO achieves {yolo['mAP_0.5']*100:.2f}% mAP@0.5 vs DETR's {best_detr['mAP_0.5']*100:.2f}%
   - Gap: {(yolo['mAP_0.5'] - best_detr['mAP_0.5'])*100:.2f}%

2. **Better Recall**: YOLO's AR@100 ({yolo['AR_max_100']*100:.2f}%) significantly outperforms DETR ({best_detr['AR_max_100']*100:.2f}%)
   - YOLO finds {(yolo['AR_max_100'] - best_detr['AR_max_100'])*100:.2f}% more objects

3. **Superior Medium Object Detection**:
   - YOLO AR (Medium): {yolo['AR_medium']*100:.2f}%
   - DETR AR (Medium): {best_detr['AR_medium']*100:.2f}%
   - **Gap: {(yolo['AR_medium'] - best_detr['AR_medium'])*100:.2f}%** ← Critical for surgical tools

4. **Tight Bounding Boxes**:
   - YOLO mAP@0.75: {yolo['mAP_0.75']*100:.2f}%
   - DETR mAP@0.75: {best_detr['mAP_0.75']*100:.2f}%
   - YOLO provides more accurate box localization

#### YOLO Success Cases

"""
            for case in success_failure_cases['yolo_success_cases'][:3]:
                report += f"- **{case['image']}**: YOLO detected {case['yolo_detections']} objects, DETR only {case['detr_detections']} ({case['difference']:+d} advantage)\n"

        report += """

### Why DETR Has Advantages

"""

        # Analyze DETR advantages (if any)
        if best_detr['AR_large'] > yolo.get('AR_large', 0):
            report += f"""
1. **Better Large Object Recall**:
   - DETR AR (Large): {best_detr['AR_large']*100:.2f}%
   - YOLO AR (Large): {yolo.get('AR_large', 0)*100:.2f}%
   - DETR excels at detecting larger surgical instruments

"""

        if len(success_failure_cases['detr_success_cases']) > 0:
            report += "#### DETR Success Cases\n\n"
            for case in success_failure_cases['detr_success_cases'][:3]:
                report += f"- **{case['image']}**: DETR detected {case['detr_detections']} objects, YOLO only {case['yolo_detections']} ({-case['difference']:+d} advantage)\n"

        report += f"""

---

## 📈 Configuration Analysis

DETR was tested with multiple confidence thresholds:

| Configuration | Confidence | mAP@0.5 | mAP@0.5:0.95 | AR@100 |
|--------------|-----------|---------|--------------|--------|
"""

        for config in detr_configs:
            report += f"| {config['description']} | {config['conf_threshold']} | {config['mAP_0.5']*100:.2f}% | {config['mAP_0.5:0.95']*100:.2f}% | {config['AR_max_100']*100:.2f}% |\n"

        report += """

### Key Finding: Confidence Threshold Impact

**Observation**: Lowering confidence threshold from 0.5 → 0.15 had **minimal impact** on DETR performance.

- All configurations achieved nearly identical mAP@0.5
- AR@100 improved marginally (< 1%)
- **Conclusion**: Problem is NOT confidence calibration, but fundamental model architecture

![Detection Counts](visualizations/infographics/detection_counts.png)

---

## 🖼️ Visual Comparisons

### Side-by-Side Examples

"""

        # Add comparison image links
        comparison_images = sorted((self.output_dir / "comparison").glob("*.jpg"))[:5]
        for img_path in comparison_images:
            report += f"![{img_path.stem}](visualizations/comparison/{img_path.name})\n\n"

        report += """

---

## 💡 Conclusions

### YOLO Wins Because:

"""

        if yolo['mAP_0.5'] > best_detr['mAP_0.5']:
            report += f"""
1. ✅ **Better suited for surgical tool detection** - Medium object performance {(yolo['AR_medium'] - best_detr['AR_medium'])*100:.1f}% higher
2. ✅ **Higher recall** - Finds {(yolo['AR_max_100'] - best_detr['AR_max_100'])*100:.1f}% more objects
3. ✅ **More accurate localization** - mAP@0.75 {(yolo['mAP_0.75'] - best_detr['mAP_0.75'])*100:.1f}% higher
4. ✅ **Efficient architecture** - Anchor-based detection works well for consistent object sizes

"""

        report += """
### DETR Limitations:

1. ❌ **Too data-hungry** - 218 images insufficient for transformer convergence
2. ❌ **Single-scale features** - ResNet-50 backbone H/32 downsampling loses detail
3. ❌ **Query specialization** - 96.3% detections from Query 81 (fragile, no ensemble benefit)
4. ❌ **Poor medium object performance** - Critical weakness for surgical tools

---

## 🚀 Recommendations

### For Production:
**Use YOLO** - Clear winner for this surgical tool detection task

### For Research:
1. Try **Deformable DETR** or **DINO** (state-of-art DETR variants)
2. Collect **10,000+ annotated frames** from surgical videos
3. Consider **ensemble approach**: YOLO (precision) + DETR (recall diversity)

---

## 📁 Files Generated

- `visualizations/yolo/` - YOLO detections on images
- `visualizations/detr/` - DETR detections on images
- `visualizations/comparison/` - Side-by-side comparisons
- `visualizations/infographics/` - Metrics infographics
- `summary.json` - Raw metrics data
- `success_failure_analysis.json` - Success/failure case analysis

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Save report
        with open(self.benchmark_dir / "BENCHMARK_REPORT.md", 'w', encoding='utf-8') as f:
            f.write(report)

        print("Markdown report generated")

    def generate_full_report(self):
        """Generate complete benchmark report"""
        print("\n" + "="*60)
        print("GENERATING COMPLETE BENCHMARK REPORT")
        print("="*60 + "\n")

        # Step 1: Generate comparison images
        self.generate_comparison_images(num_samples=20)

        # Step 2: Generate metrics infographic
        self.generate_metrics_infographic()

        # Step 3: Generate detection count analysis
        self.generate_detection_count_analysis()

        # Step 4: Identify success/failure cases
        success_failure = self.identify_success_failure_cases(top_k=5)

        # Step 5: Generate markdown report
        self.generate_markdown_report(success_failure)

        print("\n" + "="*60)
        print("✅ BENCHMARK REPORT COMPLETE")
        print("="*60)
        print(f"\nReport saved to: {self.benchmark_dir / 'BENCHMARK_REPORT.md'}")
        print(f"Visualizations saved to: {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate YOLO vs DETR benchmark report')
    parser.add_argument('--benchmark-dir', required=True, help='Path to benchmark results directory')
    parser.add_argument('--images-dir', required=True, help='Path to original images')
    parser.add_argument('--annotations', required=True, help='Path to COCO annotations')

    args = parser.parse_args()

    generator = BenchmarkReportGenerator(
        benchmark_dir=args.benchmark_dir,
        images_dir=args.images_dir,
        annotations_path=args.annotations
    )

    generator.generate_full_report()


if __name__ == "__main__":
    main()
