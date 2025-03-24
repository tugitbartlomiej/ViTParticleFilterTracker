import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# HARDCODED PATHS - REPLACE THESE WITH YOUR ACTUAL PATHS
METRICS_FILE = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/metrics_400_1000.json"
INFERENCE_FILE = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/inference_results_400_1000.json"
BASE_OUTPUT_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/visualizations"


def load_json(file_path):
    """Load JSON file"""
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: First box in format [x, y, width, height] (COCO format)
        box2: Second box in format [x, y, width, height] (COCO format)

    Returns:
        IoU value
    """
    # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection area
    x1_inter = max(box1_x1, box2_x1)
    y1_inter = max(box1_y1, box2_y1)
    x2_inter = min(box1_x2, box2_x2)
    y2_inter = min(box1_y2, box2_y2)

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def calculate_all_ious(inference_results):
    """Calculate IoU for all predictions against ground truth"""
    all_ious = []

    for result in inference_results:
        gt_boxes = [ann['bbox'] for ann in result['ground_truth']]
        if not gt_boxes:
            continue

        pred_boxes = []
        for pred in result['predictions']:
            # Handle different box formats
            if 'box' in pred:
                pred_box = pred['box']
                # Convert xyxy to xywh if needed
                if len(pred_box) == 4 and pred_box[2] > pred_box[0] and pred_box[3] > pred_box[1]:
                    pred_boxes.append([
                        pred_box[0],
                        pred_box[1],
                        pred_box[2] - pred_box[0],
                        pred_box[3] - pred_box[1]
                    ])
                else:
                    pred_boxes.append(pred_box)
            elif 'bbox' in pred:
                pred_boxes.append(pred['bbox'])

        # Calculate IoUs between all boxes
        for pred_box in pred_boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            if max_iou > 0:  # Only add valid IoUs
                all_ious.append(max_iou)

    return all_ious


def evaluate_detections(inference_results, iou_threshold=0.5):
    """
    Evaluate object detections with advanced metrics.

    Args:
        inference_results: List of inference results
        iou_threshold: IoU threshold for determining true positives

    Returns:
        Dictionary with advanced metrics
    """
    # Lists to store results
    all_predictions = []  # [confidence, is_tp]
    total_gt = 0  # Total number of ground truth objects

    # IoU statistics
    all_ious = []

    # Process each image
    for result in inference_results:
        gt_boxes = [ann['bbox'] for ann in result['ground_truth']]
        pred_boxes = []
        pred_scores = []

        for pred in result['predictions']:
            # Handle different box formats
            if 'box' in pred:
                box = pred['box']
                # Convert xyxy to xywh if needed
                if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                    pred_boxes.append([
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1]
                    ])
                else:
                    pred_boxes.append(box)
            elif 'bbox' in pred:
                pred_boxes.append(pred['bbox'])

            pred_scores.append(pred['score'])

        # Track already matched ground truth boxes to avoid double-counting
        matched_gt = [False] * len(gt_boxes)

        # Count total ground truth boxes
        total_gt += len(gt_boxes)

        # For each prediction, find best matching ground truth
        for pred_idx, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth box
            for gt_idx, gt_box in enumerate(gt_boxes):
                if matched_gt[gt_idx]:
                    continue  # Skip already matched ground truth boxes

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Record IoU
            if best_iou > 0:
                all_ious.append(best_iou)

            # Check if it's a true positive
            is_tp = best_iou >= iou_threshold

            # If it's a true positive, mark the ground truth as matched
            if is_tp and best_gt_idx >= 0:
                matched_gt[best_gt_idx] = True

            # Add to predictions list
            all_predictions.append((pred_score, is_tp))

    # Calculate precision-recall curve
    # Sort predictions by confidence score (descending)
    all_predictions.sort(key=lambda x: x[0], reverse=True)

    # Initialize lists for precision-recall curve
    precisions = []
    recalls = []

    true_positives = 0
    false_positives = 0

    # Calculate precision and recall at each prediction
    for _, is_tp in all_predictions:
        if is_tp:
            true_positives += 1
        else:
            false_positives += 1

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_gt if total_gt > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Calculate average IoU
    avg_iou = np.mean(all_ious) if all_ious else 0

    # Calculate final precision and recall
    final_precision = true_positives / (true_positives + false_positives) if (
                                                                                         true_positives + false_positives) > 0 else 0
    final_recall = true_positives / total_gt if total_gt > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (
                                                                                                      final_precision + final_recall) > 0 else 0

    # Calculate AP using area under PR curve
    if recalls and precisions and len(set(recalls)) > 1:
        # Simple approximation of area under PR curve
        ap = 0
        for i in range(len(recalls) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    else:
        ap = 0.0

    # For single class model, mAP equals AP
    map_score = ap

    return {
        "iou_threshold": iou_threshold,
        "average_iou": avg_iou,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": f1_score,
        "AP": ap,
        "mAP": map_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "ground_truth": total_gt,
        "precisions": precisions,
        "recalls": recalls,
        "all_ious": all_ious
    }


# ==== ORYGINALNE FUNKCJE WIZUALIZACJI ====

def plot_summary_metrics(metrics, output_dir):
    """Create a bar chart for the main metrics"""
    # Extract relevant metrics
    plot_metrics = {
        'Detection Rate': metrics['detection_rate'] * 100,
        'Correct Detection Rate': metrics['correct_detection_rate'] * 100,
        'Avg Confidence': metrics['avg_confidence'] * 100
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(plot_metrics.keys(), plot_metrics.values(), color=['#0088FE', '#00C49F', '#FFBB28'])

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylim(0, 100)
    ax.set_title('DETR Model Performance Metrics', fontsize=14)
    ax.set_ylabel('Percentage (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=300)
    print(f"Saved summary metrics plot to: {os.path.join(output_dir, 'summary_metrics.png')}")
    plt.close()


def plot_count_comparison(metrics, output_dir):
    """Create a bar chart comparing ground truth and predicted box counts"""
    counts = {
        'Ground Truth Boxes': metrics['total_gt_boxes'],
        'Predicted Boxes': metrics['total_pred_boxes']
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(counts.keys(), counts.values(), color=['#8884d8', '#82ca9d'])

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title('Ground Truth vs. Predicted Box Counts', fontsize=14)
    ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'count_comparison.png'), dpi=300)
    print(f"Saved count comparison plot to: {os.path.join(output_dir, 'count_comparison.png')}")
    plt.close()


def plot_detection_results(metrics, output_dir):
    """Create a pie chart showing detection results"""
    correct_detections = metrics['correct_detection_count']
    missing_detections = metrics['total_images'] - metrics['images_with_detections']
    false_positives = metrics['total_pred_boxes'] - (
                metrics['total_gt_boxes'] - (metrics['total_images'] - metrics['images_with_detections']))

    # Ensure we don't have negative values due to approximation
    false_positives = max(0, false_positives)

    labels = ['Correct Detections', 'Missing Detections', 'False Positives']
    sizes = [correct_detections, missing_detections, false_positives]
    colors = ['#0088FE', '#00C49F', '#FFBB28']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=False, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    ax.set_title('Detection Result Breakdown', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_results_pie.png'), dpi=300)
    print(f"Saved detection results pie chart to: {os.path.join(output_dir, 'detection_results_pie.png')}")
    plt.close()


def plot_confidence_distribution(inference_results, output_dir):
    """Create a histogram of confidence scores"""
    # Extract all confidence scores
    confidence_scores = []
    for result in inference_results:
        for pred in result['predictions']:
            confidence_scores.append(pred['score'])

    if not confidence_scores:
        print("No confidence scores found in inference results")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with 10 bins from 0 to 1
    n, bins, patches = ax.hist(confidence_scores, bins=10, range=(0, 1),
                               edgecolor='black', alpha=0.7, color='#8884d8')

    # Add count labels on top of each bar
    for i, patch in enumerate(patches):
        height = patch.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title('Distribution of Confidence Scores', fontsize=14)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    print(f"Saved confidence distribution histogram to: {os.path.join(output_dir, 'confidence_distribution.png')}")
    plt.close()


def plot_scatter_gt_vs_pred(inference_results, output_dir):
    """Create a scatter plot comparing GT boxes vs predicted boxes per image"""
    gt_counts = []
    pred_counts = []

    for result in inference_results:
        gt_count = len(result['ground_truth'])
        pred_count = len(result['predictions'])
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)

    # Create a dictionary to count occurrences of each (gt, pred) pair
    point_counts = defaultdict(int)
    for gt, pred in zip(gt_counts, pred_counts):
        point_counts[(gt, pred)] += 1

    # Extract points and their sizes (based on count)
    points_x = []
    points_y = []
    sizes = []
    annotations = []

    for (gt, pred), count in point_counts.items():
        points_x.append(gt)
        points_y.append(pred)
        sizes.append(count * 20)  # Scale the size
        annotations.append(str(count))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot diagonal line (perfect prediction)
    max_val = max(max(gt_counts), max(pred_counts)) if gt_counts and pred_counts else 1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    # Create scatter plot
    scatter = ax.scatter(points_x, points_y, s=sizes, alpha=0.6,
                         c=range(len(points_x)), cmap='viridis')

    # Add count annotations
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (points_x[i], points_y[i]),
                    xytext=(5, 5), textcoords="offset points")

    ax.set_title('GT Boxes vs. Predicted Boxes per Image', fontsize=14)
    ax.set_xlabel('Number of Ground Truth Boxes')
    ax.set_ylabel('Number of Predicted Boxes')

    # Set equal scale and start from 0
    ax.set_xlim(0, max_val + 0.5)
    ax.set_ylim(0, max_val + 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gt_vs_pred_scatter.png'), dpi=300)
    print(f"Saved GT vs Pred scatter plot to: {os.path.join(output_dir, 'gt_vs_pred_scatter.png')}")
    plt.close()


def plot_per_image_confidence(inference_results, output_dir):
    """Create a plot showing the average confidence score per image"""
    image_indices = []
    avg_confidences = []

    for i, result in enumerate(inference_results):
        if result['predictions']:
            image_indices.append(i)
            avg_conf = sum(pred['score'] for pred in result['predictions']) / len(result['predictions'])
            avg_confidences.append(avg_conf)

    if not avg_confidences:
        print("No confidence scores to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(image_indices, avg_confidences, marker='o', linestyle='-', alpha=0.6)

    ax.set_title('Average Confidence Score per Image', fontsize=14)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Average Confidence Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_image_confidence.png'), dpi=300)
    print(f"Saved per-image confidence plot to: {os.path.join(output_dir, 'per_image_confidence.png')}")
    plt.close()


# ==== NOWE FUNKCJE WIZUALIZACJI ZAAWANSOWANYCH METRYK ====

def plot_precision_recall_curve(advanced_metrics, output_dir):
    """Plot precision-recall curve"""
    precisions = advanced_metrics["precisions"]
    recalls = advanced_metrics["recalls"]
    ap = advanced_metrics["AP"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, marker='.', linestyle='-', label=f'Precision-Recall (AP={ap:.3f})')

    # Set labels, title, and legend
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    print(f"Saved precision-recall curve to: {os.path.join(output_dir, 'precision_recall_curve.png')}")
    plt.close()


def plot_iou_histogram(ious, output_dir):
    """Plot histogram of IoU values"""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    n, bins, patches = ax.hist(ious, bins=bins, edgecolor='black', alpha=0.7, color='#8884d8')

    # Add count labels
    for i, patch in enumerate(patches):
        height = patch.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(patch.get_x() + patch.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_title('Distribution of IoU Values', fontsize=14)
    ax.set_xlabel('IoU Value')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_distribution.png'), dpi=300)
    print(f"Saved IoU distribution histogram to: {os.path.join(output_dir, 'iou_distribution.png')}")
    plt.close()


def plot_advanced_metrics_radar(advanced_metrics, output_dir):
    """Plot radar chart of advanced metrics"""
    # Prepare the metrics for radar chart
    metrics = {
        'Precision': advanced_metrics['precision'],
        'Recall': advanced_metrics['recall'],
        'F1 Score': advanced_metrics['f1_score'],
        'AP': advanced_metrics['AP'],
        'Avg IoU': advanced_metrics['average_iou']
    }

    # Number of variables
    categories = list(metrics.keys())
    N = len(categories)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create values list
    values = list(metrics.values())
    values += values[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw polygon for metrics
    ax.plot(angles, values, linewidth=1, linestyle='solid', label='Metrics')
    ax.fill(angles, values, alpha=0.1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set radial limits
    ax.set_ylim(0, 1)

    # Add chart title
    plt.title('Detection Performance Metrics', size=15, y=1.1)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_metrics_radar.png'), dpi=300)
    print(f"Saved advanced metrics radar chart to: {os.path.join(output_dir, 'advanced_metrics_radar.png')}")
    plt.close()


def plot_advanced_metrics_bar(advanced_metrics, output_dir):
    """Plot bar chart of advanced metrics"""
    # Prepare metrics
    metrics = {
        'Precision': advanced_metrics['precision'],
        'Recall': advanced_metrics['recall'],
        'F1 Score': advanced_metrics['f1_score'],
        'AP': advanced_metrics['AP'],
        'mAP': advanced_metrics['mAP'],
        'Avg IoU': advanced_metrics['average_iou']
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(),
                  color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFCC99', '#C2C2F0'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylim(0, 1.05)
    ax.set_title('Advanced Detection Metrics (IoU threshold = 0.5)', fontsize=14)
    ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_metrics_bar.png'), dpi=300)
    print(f"Saved advanced metrics bar chart to: {os.path.join(output_dir, 'advanced_metrics_bar.png')}")
    plt.close()


def plot_tp_fp_counts(advanced_metrics, output_dir):
    """Plot true positives and false positives as a bar chart"""
    metrics = {
        'True Positives': advanced_metrics['true_positives'],
        'False Positives': advanced_metrics['false_positives'],
        'Ground Truth': advanced_metrics['ground_truth']
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#4CAF50', '#F44336', '#2196F3'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title('Detection Counts (IoU threshold = 0.5)', fontsize=14)
    ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tp_fp_counts.png'), dpi=300)
    print(f"Saved TP/FP counts chart to: {os.path.join(output_dir, 'tp_fp_counts.png')}")
    plt.close()


def main():
    """Create all visualizations for the DETR metrics"""
    # Create timestamp for the output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"visualization_{timestamp}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    try:
        metrics = load_json(METRICS_FILE)
        inference_results = load_json(INFERENCE_FILE)

        # Save a copy of the metrics file in the visualization folder
        with open(os.path.join(output_dir, 'metrics_copy.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Create a summary of inference results
        summary = {
            "total_images": len(inference_results),
            "total_predictions": sum(len(r["predictions"]) for r in inference_results),
            "total_ground_truth": sum(len(r["ground_truth"]) for r in inference_results),
            "visualization_timestamp": timestamp
        }
        with open(os.path.join(output_dir, 'inference_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Loaded metrics for {metrics['total_images']} images")
        print(f"Loaded inference results for {len(inference_results)} images")

        # Calculate advanced metrics
        print("Calculating advanced metrics (IoU, Precision, Recall, AP, mAP)...")
        advanced_metrics = evaluate_detections(inference_results, iou_threshold=0.5)
        print("Advanced metrics calculated:")
        for key, value in advanced_metrics.items():
            if key not in ["precisions", "recalls", "all_ious"]:  # Don't print large arrays
                print(f"  - {key}: {value}")

        # Save advanced metrics to JSON
        with open(os.path.join(output_dir, 'advanced_metrics.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in advanced_metrics.items()}
            json.dump(serializable_metrics, f, indent=4)

        # Create visualizations - BOTH ORIGINAL AND NEW

        # Original visualizations
        print("\nGenerating original visualizations...")
        plot_summary_metrics(metrics, output_dir)
        plot_count_comparison(metrics, output_dir)
        plot_detection_results(metrics, output_dir)
        plot_confidence_distribution(inference_results, output_dir)
        plot_scatter_gt_vs_pred(inference_results, output_dir)
        plot_per_image_confidence(inference_results, output_dir)

        # New advanced metrics visualizations
        print("\nGenerating advanced metrics visualizations...")
        plot_precision_recall_curve(advanced_metrics, output_dir)
        plot_iou_histogram(advanced_metrics["all_ious"], output_dir)
        plot_advanced_metrics_radar(advanced_metrics, output_dir)
        plot_advanced_metrics_bar(advanced_metrics, output_dir)
        plot_tp_fp_counts(advanced_metrics, output_dir)

        print("\nAll visualizations created successfully!")

        # Create a README file with basic and advanced metrics
        with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
            f.write(f"DETR Model Visualization Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Metrics file: {METRICS_FILE}\n")
            f.write(f"Inference file: {INFERENCE_FILE}\n\n")

            f.write(f"Basic Summary:\n")
            f.write(f"- Total images: {metrics['total_images']}\n")
            f.write(f"- Images with detections: {metrics['images_with_detections']}\n")
            f.write(f"- Detection rate: {metrics['detection_rate'] * 100:.1f}%\n")
            f.write(f"- Correct detection rate: {metrics['correct_detection_rate'] * 100:.1f}%\n")
            f.write(f"- Average confidence: {metrics['avg_confidence'] * 100:.1f}%\n\n")

            f.write(f"Advanced Metrics (IoU threshold = 0.5):\n")
            f.write(f"- Precision: {advanced_metrics['precision']:.4f}\n")
            f.write(f"- Recall: {advanced_metrics['recall']:.4f}\n")
            f.write(f"- F1 Score: {advanced_metrics['f1_score']:.4f}\n")
            f.write(f"- Average Precision (AP): {advanced_metrics['AP']:.4f}\n")
            f.write(f"- Mean Average Precision (mAP): {advanced_metrics['mAP']:.4f}\n")
            f.write(f"- Average IoU: {advanced_metrics['average_iou']:.4f}\n")
            f.write(f"- True Positives: {advanced_metrics['true_positives']}\n")
            f.write(f"- False Positives: {advanced_metrics['false_positives']}\n")
            f.write(f"- Ground Truth: {advanced_metrics['ground_truth']}\n")

        print(f"\nVisualization report saved to: {output_dir}")

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting DETR metrics visualization with both original and advanced metrics...")
    main()
    print("Visualization complete!")