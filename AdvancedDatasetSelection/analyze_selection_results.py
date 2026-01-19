"""
Analyze Dataset Selection Results.

Generates comprehensive statistics and visualizations from selection_reasons.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import argparse


def load_selection_data(json_path: str) -> dict:
    """Load selection reasons from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_selection(data: dict) -> dict:
    """Compute comprehensive statistics from selection data."""

    stats = {
        'overview': {},
        'el2n': {},
        'clusters': {},
        'sam_complexity': {},
        'sources': {}
    }

    # Get summary from correct location
    summary = data.get('summary', data)  # fallback to root if no summary key

    # Overview
    stats['overview'] = {
        'total_input': summary.get('total_images', 0),
        'total_selected': summary.get('selected_images', 0),
        'n_clusters': summary.get('n_clusters', 0),
        'strategy': summary.get('strategy', 'unknown'),
        'compression_ratio': summary.get('total_images', 1) / max(summary.get('selected_images', 1), 1),
        'el2n_mean_all': summary.get('el2n_mean_all', 0),
        'el2n_mean_selected': summary.get('el2n_mean_selected', 0),
        'sam_mean_all': summary.get('sam_mean_all', 0),
        'sam_mean_selected': summary.get('sam_mean_selected', 0)
    }

    # Extract per-image data - check for 'images' list format
    images_list = data.get('images', [])
    reasons = data.get('reasons', {})

    el2n_scores = []
    cluster_sizes = []
    sam_complexities = []
    distances_to_center = []
    source_videos = Counter()

    # Handle both formats: list of images or dict of reasons
    if images_list:
        # New format: list of image dicts
        for img_info in images_list:
            if 'el2n_score' in img_info:
                el2n_scores.append(img_info['el2n_score'])
            if 'cluster_size' in img_info:
                cluster_sizes.append(img_info['cluster_size'])
            if 'sam_score' in img_info:
                sam_complexities.append(img_info['sam_score'])
            if 'distance_to_center' in img_info:
                distances_to_center.append(img_info['distance_to_center'])

            # Extract source video from filename
            filename = img_info.get('filename', '')
            parts = filename.split('_frame_')
            if len(parts) >= 1:
                source = parts[0]
                source_videos[source] += 1
    else:
        # Old format: dict with image names as keys
        for img_name, info in reasons.items():
            if 'el2n' in info:
                el2n_scores.append(info['el2n'])
            if 'cluster_size' in info:
                cluster_sizes.append(info['cluster_size'])
            if 'sam_complexity' in info:
                sam_complexities.append(info['sam_complexity'])
            if 'distance_to_center' in info:
                distances_to_center.append(info['distance_to_center'])

            # Extract source video from filename
            parts = img_name.split('_frame_')
            if len(parts) >= 1:
                source = parts[0]
                source_videos[source] += 1

    # EL2N statistics
    if el2n_scores:
        el2n_arr = np.array(el2n_scores)
        stats['el2n'] = {
            'mean': float(np.mean(el2n_arr)),
            'std': float(np.std(el2n_arr)),
            'min': float(np.min(el2n_arr)),
            'max': float(np.max(el2n_arr)),
            'median': float(np.median(el2n_arr)),
            'q25': float(np.percentile(el2n_arr, 25)),
            'q75': float(np.percentile(el2n_arr, 75)),
            'easy_samples': int(np.sum(el2n_arr < 0.1)),
            'medium_samples': int(np.sum((el2n_arr >= 0.1) & (el2n_arr < 0.5))),
            'hard_samples': int(np.sum(el2n_arr >= 0.5))
        }

    # Cluster statistics
    if cluster_sizes:
        sizes_arr = np.array(cluster_sizes)
        stats['clusters'] = {
            'mean_size': float(np.mean(sizes_arr)),
            'std_size': float(np.std(sizes_arr)),
            'min_size': int(np.min(sizes_arr)),
            'max_size': int(np.max(sizes_arr)),
            'singleton_clusters': int(np.sum(sizes_arr == 1)),
            'large_clusters_10plus': int(np.sum(sizes_arr >= 10))
        }

    # SAM complexity statistics
    if sam_complexities:
        sam_arr = np.array(sam_complexities)
        stats['sam_complexity'] = {
            'mean': float(np.mean(sam_arr)),
            'std': float(np.std(sam_arr)),
            'min': float(np.min(sam_arr)),
            'max': float(np.max(sam_arr)),
            'low_complexity': int(np.sum(sam_arr < 0.3)),
            'medium_complexity': int(np.sum((sam_arr >= 0.3) & (sam_arr < 0.6))),
            'high_complexity': int(np.sum(sam_arr >= 0.6))
        }

    # Source video distribution
    stats['sources'] = {
        'n_unique_videos': len(source_videos),
        'top_10_videos': dict(source_videos.most_common(10)),
        'samples_per_video_mean': np.mean(list(source_videos.values())) if source_videos else 0,
        'samples_per_video_std': np.std(list(source_videos.values())) if source_videos else 0
    }

    # Store raw data for plotting
    stats['_raw'] = {
        'el2n_scores': el2n_scores,
        'cluster_sizes': cluster_sizes,
        'sam_complexities': sam_complexities,
        'distances_to_center': distances_to_center,
        'source_videos': dict(source_videos)
    }

    return stats


def generate_report(stats: dict, output_dir: Path):
    """Generate text report."""

    report_lines = [
        "=" * 80,
        "DATASET SELECTION ANALYSIS REPORT",
        "=" * 80,
        "",
        "OVERVIEW",
        "-" * 40,
        f"  Input images:      {stats['overview']['total_input']:,}",
        f"  Selected images:   {stats['overview']['total_selected']:,}",
        f"  Compression ratio: {stats['overview']['compression_ratio']:.1f}x",
        f"  Number of clusters: {stats['overview']['n_clusters']:,}",
        f"  Selection strategy: {stats['overview']['strategy']}",
        "",
        "EL2N DIFFICULTY SCORES",
        "-" * 40,
        f"  Mean:   {stats['el2n'].get('mean', 0):.4f}",
        f"  Std:    {stats['el2n'].get('std', 0):.4f}",
        f"  Min:    {stats['el2n'].get('min', 0):.4f}",
        f"  Max:    {stats['el2n'].get('max', 0):.4f}",
        f"  Median: {stats['el2n'].get('median', 0):.4f}",
        f"  Q25:    {stats['el2n'].get('q25', 0):.4f}",
        f"  Q75:    {stats['el2n'].get('q75', 0):.4f}",
        "",
        f"  Easy samples (EL2N < 0.1):   {stats['el2n'].get('easy_samples', 0):,}",
        f"  Medium samples (0.1-0.5):    {stats['el2n'].get('medium_samples', 0):,}",
        f"  Hard samples (EL2N >= 0.5):  {stats['el2n'].get('hard_samples', 0):,}",
        "",
        "CLUSTER STATISTICS",
        "-" * 40,
        f"  Mean cluster size: {stats['clusters'].get('mean_size', 0):.2f}",
        f"  Std cluster size:  {stats['clusters'].get('std_size', 0):.2f}",
        f"  Min cluster size:  {stats['clusters'].get('min_size', 0)}",
        f"  Max cluster size:  {stats['clusters'].get('max_size', 0)}",
        f"  Singleton clusters: {stats['clusters'].get('singleton_clusters', 0):,}",
        f"  Large clusters (10+): {stats['clusters'].get('large_clusters_10plus', 0):,}",
        "",
        "SAM COMPLEXITY",
        "-" * 40,
        f"  Mean: {stats['sam_complexity'].get('mean', 0):.4f}",
        f"  Std:  {stats['sam_complexity'].get('std', 0):.4f}",
        f"  Min:  {stats['sam_complexity'].get('min', 0):.4f}",
        f"  Max:  {stats['sam_complexity'].get('max', 0):.4f}",
        "",
        f"  Low complexity (<0.3):    {stats['sam_complexity'].get('low_complexity', 0):,}",
        f"  Medium complexity (0.3-0.6): {stats['sam_complexity'].get('medium_complexity', 0):,}",
        f"  High complexity (>=0.6):  {stats['sam_complexity'].get('high_complexity', 0):,}",
        "",
        "SOURCE VIDEO DISTRIBUTION",
        "-" * 40,
        f"  Unique source videos: {stats['sources'].get('n_unique_videos', 0)}",
        f"  Samples per video (mean): {stats['sources'].get('samples_per_video_mean', 0):.1f}",
        f"  Samples per video (std):  {stats['sources'].get('samples_per_video_std', 0):.1f}",
        "",
        "  Top 10 videos by sample count:",
    ]

    for video, count in stats['sources'].get('top_10_videos', {}).items():
        report_lines.append(f"    - {video}: {count:,} samples")

    report_lines.extend([
        "",
        "=" * 80,
        "Report generated by analyze_selection_results.py",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")

    return report_text


def generate_plots(stats: dict, output_dir: Path):
    """Generate analysis plots."""

    raw = stats.get('_raw', {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Selection Analysis', fontsize=14, fontweight='bold')

    # 1. EL2N Distribution
    ax = axes[0, 0]
    if raw.get('el2n_scores'):
        ax.hist(raw['el2n_scores'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(stats['el2n']['mean'], color='red', linestyle='--', label=f"Mean: {stats['el2n']['mean']:.3f}")
        ax.axvline(stats['el2n']['median'], color='orange', linestyle='--', label=f"Median: {stats['el2n']['median']:.3f}")
        ax.set_xlabel('EL2N Score (Difficulty)')
        ax.set_ylabel('Count')
        ax.set_title('EL2N Difficulty Distribution')
        ax.legend()

    # 2. Cluster Size Distribution
    ax = axes[0, 1]
    if raw.get('cluster_sizes'):
        sizes = raw['cluster_sizes']
        max_size = min(max(sizes), 50)  # Cap at 50 for visualization
        ax.hist([min(s, max_size) for s in sizes], bins=range(1, max_size + 2),
                color='forestgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Count')
        ax.set_title(f'Cluster Size Distribution (capped at {max_size})')
        ax.axvline(stats['clusters']['mean_size'], color='red', linestyle='--',
                   label=f"Mean: {stats['clusters']['mean_size']:.1f}")
        ax.legend()

    # 3. SAM Complexity Distribution
    ax = axes[1, 0]
    if raw.get('sam_complexities'):
        ax.hist(raw['sam_complexities'], bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(stats['sam_complexity']['mean'], color='red', linestyle='--',
                   label=f"Mean: {stats['sam_complexity']['mean']:.3f}")
        ax.set_xlabel('SAM Complexity')
        ax.set_ylabel('Count')
        ax.set_title('SAM Complexity Distribution')
        ax.legend()

    # 4. Source Video Distribution (top 20)
    ax = axes[1, 1]
    if raw.get('source_videos'):
        sorted_sources = sorted(raw['source_videos'].items(), key=lambda x: -x[1])[:20]
        videos = [s[0] for s in sorted_sources]
        counts = [s[1] for s in sorted_sources]

        bars = ax.barh(range(len(videos)), counts, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(videos)))
        ax.set_yticklabels(videos, fontsize=8)
        ax.set_xlabel('Number of Selected Samples')
        ax.set_title('Top 20 Source Videos')
        ax.invert_yaxis()

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset selection results")
    parser.add_argument("--input", "-i", type=str,
                        default="output/selected_dataset_weighted_features/selection_reasons.json",
                        help="Path to selection_reasons.json")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    data = load_selection_data(input_path)

    print("Analyzing selection results...")
    stats = analyze_selection(data)

    print("\n")
    generate_report(stats, output_dir)

    print("\nGenerating plots...")
    generate_plots(stats, output_dir)

    # Save stats as JSON
    stats_clean = {k: v for k, v in stats.items() if not k.startswith('_')}
    stats_path = output_dir / "analysis_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_clean, f, indent=2)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
