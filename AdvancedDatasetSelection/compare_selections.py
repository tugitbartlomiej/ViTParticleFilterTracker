"""
Compare Old vs New Dataset Selection

Compares feature distributions between two selections:
- Old: from feature_cache1
- New: from selected_dataset_weighted_features
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns

# Feature names for Fourier (9 features)
FOURIER_NAMES = [
    'low_band_energy', 'mid_band_energy', 'high_band_energy',
    'spectral_entropy', 'frequency_centroid',
    'horizontal_energy', 'vertical_energy', 'diagonal1_energy', 'diagonal2_energy'
]


def load_old_cache(cache_dir: Path):
    """Load features from old cache (feature_cache1)."""
    print(f"Loading old cache from: {cache_dir}")

    # Fourier: tuple (features, paths)
    with open(cache_dir / 'fourier_features.pkl', 'rb') as f:
        fourier_data = pickle.load(f)
    fourier_features = fourier_data[0]  # (N, 9)
    fourier_paths = fourier_data[1]     # list of paths

    # EL2N scores
    with open(cache_dir / 'el2n_scores.pkl', 'rb') as f:
        el2n_scores = pickle.load(f)

    # SAM scores
    with open(cache_dir / 'sam_scores.pkl', 'rb') as f:
        sam_scores = pickle.load(f)

    # DINO features
    with open(cache_dir / 'dino_features.pkl', 'rb') as f:
        dino_features = pickle.load(f)

    # Extract filenames from paths
    filenames = [Path(p).stem for p in fourier_paths]

    print(f"  Loaded {len(filenames)} samples")
    print(f"  Fourier: {fourier_features.shape}")
    print(f"  DINO: {dino_features.shape}")
    print(f"  EL2N: {el2n_scores.shape}")
    print(f"  SAM: {sam_scores.shape}")

    return {
        'fourier': fourier_features,
        'el2n': el2n_scores,
        'sam': sam_scores,
        'dino': dino_features,
        'filenames': filenames,
        'paths': fourier_paths
    }


def load_new_selection(selection_dir: Path):
    """Load info about new selection from selection_reasons.json."""
    print(f"\nLoading new selection from: {selection_dir}")

    json_path = selection_dir / 'selection_reasons.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = data.get('summary', {})
    images = data.get('images', [])

    # Extract data
    filenames = []
    el2n_scores = []
    sam_scores = []
    cluster_sizes = []

    for img in images:
        filenames.append(Path(img['filename']).stem)
        el2n_scores.append(img.get('el2n_score', 0))
        sam_scores.append(img.get('sam_score', 0))
        cluster_sizes.append(img.get('cluster_size', 1))

    print(f"  Selected: {len(filenames)} samples from {summary.get('total_images', 0)} total")

    return {
        'filenames': filenames,
        'el2n': np.array(el2n_scores),
        'sam': np.array(sam_scores),
        'cluster_sizes': np.array(cluster_sizes),
        'summary': summary
    }


def match_selected_indices(old_filenames, new_filenames):
    """Find indices in old data that correspond to new selection."""
    old_name_to_idx = {name: i for i, name in enumerate(old_filenames)}

    matched_indices = []
    unmatched = []

    for new_name in new_filenames:
        # Try exact match first
        if new_name in old_name_to_idx:
            matched_indices.append(old_name_to_idx[new_name])
        else:
            # Try without _aug suffix
            base_name = new_name.split('_aug_')[0] if '_aug_' in new_name else new_name
            if base_name in old_name_to_idx:
                matched_indices.append(old_name_to_idx[base_name])
            else:
                unmatched.append(new_name)

    print(f"\n  Matched: {len(matched_indices)} / {len(new_filenames)}")
    if unmatched:
        print(f"  Unmatched: {len(unmatched)} (first 5: {unmatched[:5]})")

    return matched_indices


def compute_statistics(data, name=""):
    """Compute statistics for a feature array."""
    return {
        'name': name,
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }


def create_comparison_plots(old_data, new_data, selected_indices, output_dir):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get selected subset from old data
    old_fourier_all = old_data['fourier']
    old_fourier_selected = old_data['fourier'][selected_indices]
    old_el2n_all = old_data['el2n']
    old_el2n_selected = old_data['el2n'][selected_indices]
    old_sam_all = old_data['sam']
    old_sam_selected = old_data['sam'][selected_indices]

    # New selection data
    new_el2n = new_data['el2n']
    new_sam = new_data['sam']

    # =========================================
    # Figure 1: Fourier Feature Comparison
    # =========================================
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Fourier Features: ALL vs SELECTED (20k)', fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes.flat, FOURIER_NAMES)):
        all_values = old_fourier_all[:, i]
        selected_values = old_fourier_selected[:, i]

        # KDE plots
        ax.hist(all_values, bins=50, density=True, alpha=0.5, color='blue', label=f'All ({len(all_values):,})')
        ax.hist(selected_values, bins=50, density=True, alpha=0.5, color='red', label=f'Selected ({len(selected_values):,})')

        # Stats
        all_mean = np.mean(all_values)
        sel_mean = np.mean(selected_values)
        ax.axvline(all_mean, color='blue', linestyle='--', alpha=0.8)
        ax.axvline(sel_mean, color='red', linestyle='--', alpha=0.8)

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(all_values, selected_values)

        ax.set_title(f'{name}\nKS={ks_stat:.3f}, p={ks_pval:.2e}')
        ax.legend(fontsize=8)
        ax.set_xlabel(name)
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_dir / 'fourier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fourier_comparison.png'}")

    # =========================================
    # Figure 2: EL2N and SAM Comparison
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EL2N & SAM: ALL vs SELECTED', fontsize=14, fontweight='bold')

    # EL2N Distribution
    ax = axes[0, 0]
    ax.hist(old_el2n_all, bins=50, density=True, alpha=0.5, color='blue', label=f'All ({len(old_el2n_all):,})')
    ax.hist(old_el2n_selected, bins=50, density=True, alpha=0.5, color='red', label=f'Selected ({len(old_el2n_selected):,})')
    ax.axvline(np.mean(old_el2n_all), color='blue', linestyle='--')
    ax.axvline(np.mean(old_el2n_selected), color='red', linestyle='--')
    ks_stat, ks_pval = stats.ks_2samp(old_el2n_all, old_el2n_selected)
    ax.set_title(f'EL2N Distribution\nKS={ks_stat:.3f}, p={ks_pval:.2e}')
    ax.legend()
    ax.set_xlabel('EL2N Score')

    # SAM Distribution
    ax = axes[0, 1]
    ax.hist(old_sam_all, bins=50, density=True, alpha=0.5, color='blue', label=f'All ({len(old_sam_all):,})')
    ax.hist(old_sam_selected, bins=50, density=True, alpha=0.5, color='red', label=f'Selected ({len(old_sam_selected):,})')
    ax.axvline(np.mean(old_sam_all), color='blue', linestyle='--')
    ax.axvline(np.mean(old_sam_selected), color='red', linestyle='--')
    ks_stat, ks_pval = stats.ks_2samp(old_sam_all, old_sam_selected)
    ax.set_title(f'SAM Complexity Distribution\nKS={ks_stat:.3f}, p={ks_pval:.2e}')
    ax.legend()
    ax.set_xlabel('SAM Score')

    # EL2N: Easy/Medium/Hard breakdown
    ax = axes[1, 0]
    categories = ['Easy\n(<0.1)', 'Medium\n(0.1-0.5)', 'Hard\n(>=0.5)']

    all_easy = np.sum(old_el2n_all < 0.1) / len(old_el2n_all) * 100
    all_med = np.sum((old_el2n_all >= 0.1) & (old_el2n_all < 0.5)) / len(old_el2n_all) * 100
    all_hard = np.sum(old_el2n_all >= 0.5) / len(old_el2n_all) * 100

    sel_easy = np.sum(old_el2n_selected < 0.1) / len(old_el2n_selected) * 100
    sel_med = np.sum((old_el2n_selected >= 0.1) & (old_el2n_selected < 0.5)) / len(old_el2n_selected) * 100
    sel_hard = np.sum(old_el2n_selected >= 0.5) / len(old_el2n_selected) * 100

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, [all_easy, all_med, all_hard], width, label='All', color='blue', alpha=0.7)
    ax.bar(x + width/2, [sel_easy, sel_med, sel_hard], width, label='Selected', color='red', alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('EL2N Difficulty Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add percentage labels
    for i, (a, s) in enumerate(zip([all_easy, all_med, all_hard], [sel_easy, sel_med, sel_hard])):
        ax.text(i - width/2, a + 1, f'{a:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, s + 1, f'{s:.1f}%', ha='center', fontsize=9)

    # SAM: Low/Medium/High breakdown
    ax = axes[1, 1]
    categories = ['Low\n(<0.3)', 'Medium\n(0.3-0.6)', 'High\n(>=0.6)']

    all_low = np.sum(old_sam_all < 0.3) / len(old_sam_all) * 100
    all_med = np.sum((old_sam_all >= 0.3) & (old_sam_all < 0.6)) / len(old_sam_all) * 100
    all_high = np.sum(old_sam_all >= 0.6) / len(old_sam_all) * 100

    sel_low = np.sum(old_sam_selected < 0.3) / len(old_sam_selected) * 100
    sel_med = np.sum((old_sam_selected >= 0.3) & (old_sam_selected < 0.6)) / len(old_sam_selected) * 100
    sel_high = np.sum(old_sam_selected >= 0.6) / len(old_sam_selected) * 100

    ax.bar(x - width/2, [all_low, all_med, all_high], width, label='All', color='blue', alpha=0.7)
    ax.bar(x + width/2, [sel_low, sel_med, sel_high], width, label='Selected', color='red', alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('SAM Complexity Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    for i, (a, s) in enumerate(zip([all_low, all_med, all_high], [sel_low, sel_med, sel_high])):
        ax.text(i - width/2, a + 1, f'{a:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, s + 1, f'{s:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'el2n_sam_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'el2n_sam_comparison.png'}")

    # =========================================
    # Figure 3: Fourier Summary Bar Chart
    # =========================================
    fig, ax = plt.subplots(figsize=(14, 6))

    all_means = np.mean(old_fourier_all, axis=0)
    sel_means = np.mean(old_fourier_selected, axis=0)
    all_stds = np.std(old_fourier_all, axis=0)
    sel_stds = np.std(old_fourier_selected, axis=0)

    x = np.arange(len(FOURIER_NAMES))
    width = 0.35

    bars1 = ax.bar(x - width/2, all_means, width, yerr=all_stds, label='All (90k)',
                   color='steelblue', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, sel_means, width, yerr=sel_stds, label='Selected (20k)',
                   color='coral', alpha=0.8, capsize=3)

    ax.set_ylabel('Feature Value')
    ax.set_title('Fourier Features: Mean ± Std Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(FOURIER_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fourier_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fourier_summary.png'}")

    # =========================================
    # Generate Statistics Report
    # =========================================
    report = {
        'comparison': {
            'all_samples': len(old_el2n_all),
            'selected_samples': len(old_el2n_selected),
            'selection_ratio': len(old_el2n_selected) / len(old_el2n_all)
        },
        'fourier': {},
        'el2n': {
            'all': compute_statistics(old_el2n_all, 'el2n_all'),
            'selected': compute_statistics(old_el2n_selected, 'el2n_selected')
        },
        'sam': {
            'all': compute_statistics(old_sam_all, 'sam_all'),
            'selected': compute_statistics(old_sam_selected, 'sam_selected')
        }
    }

    for i, name in enumerate(FOURIER_NAMES):
        ks_stat, ks_pval = stats.ks_2samp(old_fourier_all[:, i], old_fourier_selected[:, i])
        report['fourier'][name] = {
            'all': compute_statistics(old_fourier_all[:, i], f'{name}_all'),
            'selected': compute_statistics(old_fourier_selected[:, i], f'{name}_selected'),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'significant_diff': ks_pval < 0.05
        }

    with open(output_dir / 'comparison_statistics.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {output_dir / 'comparison_statistics.json'}")

    return report


def print_summary(report):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: ALL (90k) vs SELECTED (20k)")
    print("=" * 70)

    print(f"\nSamples: {report['comparison']['all_samples']:,} -> {report['comparison']['selected_samples']:,}")
    print(f"Selection ratio: {report['comparison']['selection_ratio']*100:.1f}%")

    print("\n--- EL2N (Difficulty) ---")
    el2n_all = report['el2n']['all']
    el2n_sel = report['el2n']['selected']
    print(f"  All:      mean={el2n_all['mean']:.4f}, std={el2n_all['std']:.4f}")
    print(f"  Selected: mean={el2n_sel['mean']:.4f}, std={el2n_sel['std']:.4f}")

    print("\n--- SAM (Complexity) ---")
    sam_all = report['sam']['all']
    sam_sel = report['sam']['selected']
    print(f"  All:      mean={sam_all['mean']:.4f}, std={sam_all['std']:.4f}")
    print(f"  Selected: mean={sam_sel['mean']:.4f}, std={sam_sel['std']:.4f}")

    print("\n--- Fourier Features (significant differences) ---")
    for name, data in report['fourier'].items():
        if data['significant_diff']:
            diff = data['selected']['mean'] - data['all']['mean']
            print(f"  {name}: Δmean={diff:+.4f}, KS p={data['ks_pvalue']:.2e} *")


def main():
    # Paths
    old_cache = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/feature_cache1")
    new_selection = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/selected_dataset_weighted_features")
    output_dir = new_selection / "comparison_with_old"

    # Load data
    old_data = load_old_cache(old_cache)
    new_data = load_new_selection(new_selection)

    # Match indices
    selected_indices = match_selected_indices(old_data['filenames'], new_data['filenames'])

    if len(selected_indices) < 100:
        print("ERROR: Too few matches found! Check if filenames match between datasets.")
        return

    # Create visualizations
    report = create_comparison_plots(old_data, new_data, selected_indices, output_dir)

    # Print summary
    print_summary(report)

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
