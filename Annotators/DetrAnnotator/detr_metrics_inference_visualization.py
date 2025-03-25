import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# HARDCODED PATHS - REPLACE THESE WITH YOUR ACTUAL PATHS
METRICS_FILE = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/metrics_500_600.json"
INFERENCE_FILE = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/inference_results_500_600.json"
OUTPUT_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/inference_ranged/visualizations"


def load_json(file_path):
    """Load JSON file"""
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


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
                    xytext=(5, 5), textcoords='offset points')

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


def main():
    """Create all visualizations for the DETR metrics"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data
    try:
        metrics = load_json(METRICS_FILE)
        inference_results = load_json(INFERENCE_FILE)

        print(f"Loaded metrics for {metrics['total_images']} images")
        print(f"Loaded inference results for {len(inference_results)} images")

        # Create visualizations
        plot_summary_metrics(metrics, OUTPUT_DIR)
        plot_count_comparison(metrics, OUTPUT_DIR)
        plot_detection_results(metrics, OUTPUT_DIR)
        plot_confidence_distribution(inference_results, OUTPUT_DIR)
        plot_scatter_gt_vs_pred(inference_results, OUTPUT_DIR)
        plot_per_image_confidence(inference_results, OUTPUT_DIR)

        print(f"All visualizations created successfully in {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting DETR metrics visualization...")
    main()
    print("Visualization complete!")