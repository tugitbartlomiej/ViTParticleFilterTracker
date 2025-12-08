#!/usr/bin/env python3
"""
=============================================================================
DETR QUERY VISUALIZATION - MULTI-EPOCH ANALYSIS
=============================================================================
Visualizes which DETR queries (0-99) are responsible for detections.
Each query gets a unique, consistent color across all epochs and images.

Purpose:
- Identify which queries most frequently detect tooltips
- Track query behavior across training epochs
- Visualize query specialization patterns

Output:
- Color-coded bounding boxes where color = query ID
- Statistics about query usage per epoch
- Side-by-side epoch comparisons

Author: PhD Research - Cataract Surgery Instrument Detection
Date: 2025-12-08
=============================================================================
"""

import sys
import json
import time
import colorsys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker")

# DETR Checkpoints to analyze
DETR_CHECKPOINTS = {
    100: BASE_PATH / "Eden/Checkpoints/DETR/checkpoint_epoch_100.pth",
    120: BASE_PATH / "Eden/Checkpoints/DETR/checkpoint_epoch_120.pth",
    140: BASE_PATH / "Eden/Checkpoints/DETR/checkpoint_epoch_140.pth",
    160: BASE_PATH / "Eden/Checkpoints/DETR/checkpoint_epoch_160.pth",
    170: BASE_PATH / "Eden/Checkpoints/DETR/checkpoint_epoch_170.pth",
}

# Dataset - all splits combined
DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
ALL_SPLITS = ["train", "valid", "test"]  # Combine all splits as one test set

# Output
OUTPUT_DIR = BASE_PATH / "YOLO_DETR_Benchmarks/Benchmarks"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
VIS_OUTPUT_DIR = OUTPUT_DIR / f"DETR_QUERY_VISUALIZATION_{TIMESTAMP}"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection settings
CONF_THRESHOLD = 0.3  # Minimum confidence to show detection
NUM_QUERIES = 100     # DETR has 100 object queries

# Visualization settings
MAX_IMAGES_PER_SPLIT = 100  # Maximum images per split (total ~300 from all splits)
FONT_SIZE = 12

print(f"""
{'='*80}
DETR QUERY VISUALIZATION - MULTI-EPOCH ANALYSIS
{'='*80}
Device: {DEVICE}
Dataset: {DATASET_ROOT}
Splits: {ALL_SPLITS} (combined as one test set)
Output: {VIS_OUTPUT_DIR}

DETR Checkpoints: {list(DETR_CHECKPOINTS.keys())}
Confidence Threshold: {CONF_THRESHOLD}
Max Images per Split: {MAX_IMAGES_PER_SPLIT}
{'='*80}
""")

# =============================================================================
# COLOR PALETTE FOR 100 QUERIES
# =============================================================================

def generate_query_colors(num_queries=100):
    """
    Generate unique, visually distinct colors for each query index.
    Uses HSV color space for better distribution.

    Returns:
        dict: {query_idx: (R, G, B)} where RGB values are 0-255
    """
    colors = {}
    for i in range(num_queries):
        # Use golden ratio to distribute hues evenly
        hue = (i * 0.618033988749895) % 1.0
        # Vary saturation and value slightly for more distinction
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.15

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))

    return colors

# Pre-generate colors for all queries
QUERY_COLORS = generate_query_colors(NUM_QUERIES)


def get_query_color(query_idx):
    """Get RGB color for a specific query index."""
    return QUERY_COLORS.get(query_idx, (128, 128, 128))


# =============================================================================
# MODEL LOADING
# =============================================================================

DETR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


def load_detr_model(checkpoint_path):
    """Load DETR model from checkpoint."""
    print(f"  Loading DETR from {checkpoint_path.name}...")
    start_load = time.time()

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    load_time = time.time() - start_load
    print(f"    -> Loaded in {load_time:.2f}s")
    return model


# =============================================================================
# QUERY-AWARE INFERENCE
# =============================================================================

def run_inference_with_query_tracking(model, image_path):
    """
    Run DETR inference and track which query produces each detection.

    Returns:
        list: List of detections with query_idx, bbox, score
    """
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    inputs = DETR_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get raw outputs: logits [1, 100, num_classes+1] and boxes [1, 100, 4]
    logits = outputs.logits[0]  # [100, num_classes+1]
    boxes = outputs.pred_boxes[0]  # [100, 4] in normalized cxcywh format

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # For single-class detection, class 0 is the object, last class is "no object"
    # Get probability of being an object (not background)
    object_probs = probs[:, 0]  # Probability of class 0 (tool/instrument)

    detections = []

    for query_idx in range(NUM_QUERIES):
        score = float(object_probs[query_idx])

        if score >= CONF_THRESHOLD:
            # Convert normalized cxcywh to pixel xyxy
            cx, cy, w, h = boxes[query_idx].cpu().numpy()

            # Denormalize
            cx *= img_width
            cy *= img_height
            w *= img_width
            h *= img_height

            # Convert to x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            detections.append({
                'query_idx': query_idx,
                'bbox': [x1, y1, x2, y2],
                'score': score
            })

    return detections, image


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_query_detections(image, detections, epoch, show_legend=True):
    """
    Draw detections with query-specific colors.

    Args:
        image: PIL Image
        detections: List of dicts with query_idx, bbox, score
        epoch: Epoch number for title
        show_legend: Whether to show color legend

    Returns:
        PIL Image with annotations
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        font_small = ImageFont.truetype("arial.ttf", FONT_SIZE - 2)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw title
    title = f"DETR_epoch{epoch}"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_h = title_bbox[3] - title_bbox[1]
    draw.rectangle([0, 0, 200, title_h + 10], fill=(50, 50, 50))
    draw.text((5, 5), title, fill="white", font=font)

    # Draw detection count
    det_text = f"Detections: {len(detections)}"
    draw.text((5, title_h + 15), det_text, fill="yellow", font=font_small)

    # Draw each detection with query-specific color
    for det in detections:
        query_idx = det['query_idx']
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        color = get_query_color(query_idx)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        label = f"Q{query_idx}:{score:.2f}"
        label_bbox = draw.textbbox((x1, y1), label, font=font_small)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]

        # Position label above box, or below if near top
        if y1 > label_h + 5:
            label_y = y1 - label_h - 4
        else:
            label_y = y2 + 2

        draw.rectangle([x1, label_y, x1 + label_w + 6, label_y + label_h + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font_small)

    return img


def create_multi_epoch_comparison(image_path, all_epoch_detections, output_path):
    """
    Create a grid showing the same image with detections from different epochs.

    Args:
        image_path: Path to original image
        all_epoch_detections: Dict of {epoch: detections_list}
        output_path: Where to save the comparison
    """
    original_img = Image.open(image_path).convert("RGB")
    img_w, img_h = original_img.size

    epochs = sorted(all_epoch_detections.keys())
    num_epochs = len(epochs)

    # Grid layout: 2 rows for up to 6 epochs
    cols = min(4, num_epochs)
    rows = (num_epochs + cols - 1) // cols

    # Add extra row for ground truth if needed
    padding = 5
    grid_w = cols * img_w + (cols + 1) * padding
    grid_h = rows * img_h + (rows + 1) * padding

    grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    for idx, epoch in enumerate(epochs):
        detections = all_epoch_detections[epoch]
        annotated = draw_query_detections(original_img, detections, epoch)

        row = idx // cols
        col = idx % cols
        x = padding + col * (img_w + padding)
        y = padding + row * (img_h + padding)

        grid_img.paste(annotated, (x, y))

    grid_img.save(output_path, quality=95)


def create_query_legend(output_path, top_queries=20):
    """
    Create a legend showing query colors for the most active queries.

    Args:
        output_path: Where to save the legend
        top_queries: Number of top queries to show
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top_queries + 1)
    ax.set_title("DETR Query Color Legend (Top Active Queries)", fontsize=14, fontweight='bold')
    ax.axis('off')

    for i in range(top_queries):
        color = get_query_color(i)
        # Normalize to 0-1 for matplotlib
        color_norm = (color[0]/255, color[1]/255, color[2]/255)

        y = top_queries - i
        rect = patches.Rectangle((0.5, y - 0.4), 1, 0.8,
                                   linewidth=1, edgecolor='black',
                                   facecolor=color_norm)
        ax.add_patch(rect)
        ax.text(2, y, f"Query {i}", fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# STATISTICS
# =============================================================================

def compute_query_statistics(all_results):
    """
    Compute statistics about query usage across epochs.

    Args:
        all_results: Dict of {epoch: {image_name: detections}}

    Returns:
        dict: Statistics including query counts, dominant queries, etc.
    """
    stats = {
        'per_epoch': {},
        'overall': defaultdict(int),
        'query_by_epoch': defaultdict(lambda: defaultdict(int))
    }

    for epoch, image_results in all_results.items():
        epoch_query_counts = defaultdict(int)
        total_detections = 0

        for img_name, detections in image_results.items():
            for det in detections:
                query_idx = det['query_idx']
                epoch_query_counts[query_idx] += 1
                stats['overall'][query_idx] += 1
                stats['query_by_epoch'][epoch][query_idx] += 1
                total_detections += 1

        # Sort by count
        sorted_queries = sorted(epoch_query_counts.items(), key=lambda x: x[1], reverse=True)

        stats['per_epoch'][epoch] = {
            'total_detections': total_detections,
            'active_queries': len(epoch_query_counts),
            'top_queries': sorted_queries[:10],
            'dominant_query': sorted_queries[0] if sorted_queries else (None, 0)
        }

    # Overall top queries
    stats['overall_top'] = sorted(stats['overall'].items(), key=lambda x: x[1], reverse=True)[:20]

    return stats


def print_statistics(stats):
    """Print query statistics in a readable format."""
    print(f"\n{'='*70}")
    print("QUERY USAGE STATISTICS")
    print(f"{'='*70}")

    for epoch in sorted(stats['per_epoch'].keys()):
        epoch_stats = stats['per_epoch'][epoch]
        print(f"\n--- Epoch {epoch} ---")
        print(f"Total detections: {epoch_stats['total_detections']}")
        print(f"Active queries: {epoch_stats['active_queries']}/100")

        dom_query, dom_count = epoch_stats['dominant_query']
        if dom_query is not None:
            pct = (dom_count / epoch_stats['total_detections'] * 100) if epoch_stats['total_detections'] > 0 else 0
            print(f"Dominant query: Q{dom_query} ({dom_count} detections, {pct:.1f}%)")

        print("Top 5 queries:")
        for query_idx, count in epoch_stats['top_queries'][:5]:
            pct = (count / epoch_stats['total_detections'] * 100) if epoch_stats['total_detections'] > 0 else 0
            print(f"  Q{query_idx}: {count} ({pct:.1f}%)")

    print(f"\n{'='*70}")
    print("OVERALL TOP 10 QUERIES (across all epochs)")
    print(f"{'='*70}")
    for query_idx, count in stats['overall_top'][:10]:
        print(f"  Q{query_idx}: {count} total detections")


def save_statistics_report(stats, output_path):
    """Save statistics to a JSON file and markdown report."""
    # JSON
    json_path = output_path / "query_statistics.json"
    with open(json_path, 'w') as f:
        # Convert defaultdicts to regular dicts for JSON
        json_stats = {
            'per_epoch': stats['per_epoch'],
            'overall_top': stats['overall_top'],
            'query_by_epoch': {k: dict(v) for k, v in stats['query_by_epoch'].items()}
        }
        json.dump(json_stats, f, indent=2)

    # Markdown report
    md_path = output_path / "QUERY_ANALYSIS_REPORT.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# DETR Query Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Confidence Threshold:** {CONF_THRESHOLD}\n\n")

        f.write("## Query Usage by Epoch\n\n")
        f.write("| Epoch | Total Detections | Active Queries | Dominant Query | Dominant % |\n")
        f.write("|-------|------------------|----------------|----------------|------------|\n")

        for epoch in sorted(stats['per_epoch'].keys()):
            es = stats['per_epoch'][epoch]
            dom_q, dom_c = es['dominant_query']
            pct = (dom_c / es['total_detections'] * 100) if es['total_detections'] > 0 else 0
            f.write(f"| {epoch} | {es['total_detections']} | {es['active_queries']} | Q{dom_q} | {pct:.1f}% |\n")

        f.write("\n## Top 10 Queries Overall\n\n")
        f.write("| Query | Total Detections |\n")
        f.write("|-------|------------------|\n")
        for query_idx, count in stats['overall_top'][:10]:
            f.write(f"| Q{query_idx} | {count} |\n")

        f.write("\n## Query Color Reference\n\n")
        f.write("Each query has a unique color assigned. Colors are consistent across all visualizations.\n")
        f.write("See `query_color_legend.png` for visual reference.\n")

    print(f"\nStatistics saved to: {output_path}")


def plot_query_distribution(stats, output_path):
    """Create visualization of query distribution across epochs."""
    epochs = sorted(stats['per_epoch'].keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DETR Query Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Dominant query percentage per epoch
    ax = axes[0, 0]
    dominant_pcts = []
    for epoch in epochs:
        es = stats['per_epoch'][epoch]
        _, dom_c = es['dominant_query']
        pct = (dom_c / es['total_detections'] * 100) if es['total_detections'] > 0 else 0
        dominant_pcts.append(pct)

    ax.bar(epochs, dominant_pcts, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dominant Query Usage (%)', fontsize=12)
    ax.set_title('Dominant Query Concentration', fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Active queries per epoch
    ax = axes[0, 1]
    active_counts = [stats['per_epoch'][e]['active_queries'] for e in epochs]
    ax.bar(epochs, active_counts, color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Active Queries (out of 100)', fontsize=12)
    ax.set_title('Query Diversity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # 3. Total detections per epoch
    ax = axes[1, 0]
    total_dets = [stats['per_epoch'][e]['total_detections'] for e in epochs]
    ax.plot(epochs, total_dets, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Detections', fontsize=12)
    ax.set_title('Detection Count Progression', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Top 5 queries heatmap-style
    ax = axes[1, 1]
    top_5_queries = [q for q, _ in stats['overall_top'][:5]]

    data = []
    for epoch in epochs:
        row = []
        for q in top_5_queries:
            count = stats['query_by_epoch'][epoch].get(q, 0)
            row.append(count)
        data.append(row)

    data = np.array(data)
    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_yticks(range(len(top_5_queries)))
    ax.set_yticklabels([f'Q{q}' for q in top_5_queries])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)
    ax.set_title('Top 5 Queries Activity Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Detection Count')

    plt.tight_layout()
    plt.savefig(output_path / "query_distribution_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    start_time = time.time()

    # Create output directory
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify checkpoints
    print("Verifying checkpoints...")
    available_checkpoints = {}
    for epoch, path in DETR_CHECKPOINTS.items():
        if path.exists():
            available_checkpoints[epoch] = path
            print(f"  Epoch {epoch}: OK")
        else:
            print(f"  Epoch {epoch}: NOT FOUND")

    if not available_checkpoints:
        print("ERROR: No checkpoints found!")
        sys.exit(1)

    # Load images from all splits (train, valid, test) combined
    print(f"\nLoading images from all splits: {ALL_SPLITS}...")
    image_files = []

    for split in ALL_SPLITS:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            split_images = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
            # Limit per split
            split_images = split_images[:MAX_IMAGES_PER_SPLIT]
            image_files.extend(split_images)
            print(f"  {split}: {len(split_images)} images loaded")
        else:
            print(f"  {split}: NOT FOUND")

    if not image_files:
        print("ERROR: No images found!")
        sys.exit(1)

    print(f"  TOTAL: {len(image_files)} images to process")

    # Store all results
    all_results = {}  # {epoch: {image_name: detections}}

    # Process each checkpoint
    for epoch, checkpoint_path in sorted(available_checkpoints.items()):
        print(f"\n{'='*60}")
        print(f"PROCESSING EPOCH {epoch}")
        print(f"{'='*60}")

        model = load_detr_model(checkpoint_path)
        all_results[epoch] = {}

        # Create epoch output directory
        epoch_dir = VIS_OUTPUT_DIR / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)

        for img_path in tqdm(image_files, desc=f"Epoch {epoch} inference"):
            detections, image = run_inference_with_query_tracking(model, img_path)
            all_results[epoch][img_path.name] = detections

            # Save individual visualization
            annotated = draw_query_detections(image, detections, epoch)
            output_path = epoch_dir / f"query_vis_{img_path.stem}.jpg"
            annotated.save(output_path, quality=95)

        # Free memory
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Create comparison visualizations
    print(f"\n{'='*60}")
    print("CREATING MULTI-EPOCH COMPARISONS")
    print(f"{'='*60}")

    comparison_dir = VIS_OUTPUT_DIR / "epoch_comparisons"
    comparison_dir.mkdir(exist_ok=True)

    for img_path in tqdm(image_files[:20], desc="Creating comparisons"):  # First 20 images
        epoch_detections = {epoch: all_results[epoch][img_path.name]
                           for epoch in all_results.keys()}

        output_path = comparison_dir / f"comparison_{img_path.stem}.jpg"
        create_multi_epoch_comparison(img_path, epoch_detections, output_path)

    # Compute and save statistics
    print(f"\n{'='*60}")
    print("COMPUTING STATISTICS")
    print(f"{'='*60}")

    stats = compute_query_statistics(all_results)
    print_statistics(stats)
    save_statistics_report(stats, VIS_OUTPUT_DIR)
    plot_query_distribution(stats, VIS_OUTPUT_DIR)

    # Create color legend
    create_query_legend(VIS_OUTPUT_DIR / "query_color_legend.png")

    elapsed_time = time.time() - start_time

    print(f"""
{'='*80}
VISUALIZATION COMPLETE
{'='*80}
Total time: {elapsed_time/60:.2f} minutes
Output directory: {VIS_OUTPUT_DIR}

Contents:
- epoch_XXX/          : Individual image visualizations per epoch
- epoch_comparisons/  : Side-by-side multi-epoch comparisons
- query_statistics.json
- QUERY_ANALYSIS_REPORT.md
- query_distribution_analysis.png
- query_color_legend.png
{'='*80}
""")


if __name__ == "__main__":
    main()
