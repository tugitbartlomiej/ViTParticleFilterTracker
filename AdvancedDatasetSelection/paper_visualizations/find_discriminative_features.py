#!/usr/bin/env python3
"""
Find Fourier features that best discriminate between tooltip and background images.

Analyzes multiple spectral features and computes their discriminative power
using statistical measures like t-test, effect size (Cohen's d), and ROC-AUC.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from scipy import stats
from scipy.stats import kurtosis, skew
import random

# Paths
TOOLTIP_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images")
BACKGROUND_TRAIN = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\BackgroundFinetuned\Datasets\Background\train")
BACKGROUND_VAL = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\BackgroundFinetuned\Datasets\Background\val")


def filter_images_by_filename_token(image_paths, token):
    """Exclude images whose filename contains the given token (case-insensitive)."""
    token_upper = token.upper()
    filtered = [p for p in image_paths if token_upper not in p.name.upper()]
    return filtered, len(image_paths) - len(filtered)


def compute_advanced_fft_features(image):
    """
    Compute comprehensive FFT features for discrimination.

    Returns dict with many potential discriminative features.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32) / 255.0
    rows, cols = gray.shape

    # Compute FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    magnitude_log = np.log1p(magnitude)

    # Create radius map
    center_row, center_col = rows // 2, cols // 2
    max_radius = np.sqrt(center_row**2 + center_col**2)
    y, x = np.ogrid[:rows, :cols]
    radius = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    normalized_radius = radius / max_radius

    # === BASIC BAND ENERGIES ===
    total_energy = np.sum(magnitude**2)

    low_mask = normalized_radius <= 0.1
    mid_mask = (normalized_radius > 0.1) & (normalized_radius <= 0.5)
    high_mask = normalized_radius > 0.5

    e_low = np.sum(magnitude[low_mask]**2) / total_energy
    e_mid = np.sum(magnitude[mid_mask]**2) / total_energy
    e_high = np.sum(magnitude[high_mask]**2) / total_energy

    # === HIGH/LOW RATIO ===
    high_low_ratio = e_high / (e_low + 1e-10)

    # === MORE GRANULAR BANDS (10 rings) ===
    ring_energies = []
    for i in range(10):
        r_min = i * 0.1
        r_max = (i + 1) * 0.1
        ring_mask = (normalized_radius > r_min) & (normalized_radius <= r_max)
        ring_e = np.sum(magnitude[ring_mask]**2) / total_energy if np.any(ring_mask) else 0
        ring_energies.append(ring_e)

    # === SPECTRAL ENTROPY ===
    psd = magnitude**2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_flat = psd_norm.flatten()
    psd_flat = psd_flat[psd_flat > 0]
    spectral_entropy = -np.sum(psd_flat * np.log2(psd_flat + 1e-10)) / np.log2(len(psd_flat))

    # === SPECTRAL CENTROID ===
    freq_centroid = np.sum(normalized_radius * magnitude) / (np.sum(magnitude) + 1e-10)

    # === SPECTRAL SPREAD (variance around centroid) ===
    spectral_spread = np.sqrt(np.sum(((normalized_radius - freq_centroid)**2) * magnitude) / (np.sum(magnitude) + 1e-10))

    # === SPECTRAL FLATNESS (Wiener entropy) ===
    # Geometric mean / Arithmetic mean
    psd_positive = psd[psd > 0]
    if len(psd_positive) > 0:
        geo_mean = np.exp(np.mean(np.log(psd_positive + 1e-10)))
        arith_mean = np.mean(psd_positive)
        spectral_flatness = geo_mean / (arith_mean + 1e-10)
    else:
        spectral_flatness = 0

    # === SPECTRAL ROLLOFF (frequency below which 85% of energy) ===
    cumsum = np.cumsum(np.sort(psd.flatten())[::-1])
    rolloff_threshold = 0.85 * total_energy
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    spectral_rolloff = rolloff_idx / len(psd.flatten())

    # === SPECTRAL KURTOSIS (peakiness of spectrum) ===
    spectral_kurtosis = kurtosis(magnitude.flatten())

    # === SPECTRAL SKEWNESS ===
    spectral_skewness = skew(magnitude.flatten())

    # === DIRECTIONAL ENERGIES ===
    angle = np.arctan2(y - center_row, x - center_col)

    h_mask = (np.abs(angle) < np.pi/8) | (np.abs(angle) > 7*np.pi/8)
    v_mask = (np.abs(angle - np.pi/2) < np.pi/8) | (np.abs(angle + np.pi/2) < np.pi/8)
    d1_mask = (np.abs(angle - np.pi/4) < np.pi/8) | (np.abs(angle + 3*np.pi/4) < np.pi/8)
    d2_mask = (np.abs(angle + np.pi/4) < np.pi/8) | (np.abs(angle - 3*np.pi/4) < np.pi/8)

    dir_total = np.sum(magnitude[h_mask | v_mask | d1_mask | d2_mask]**2)
    e_horizontal = np.sum(magnitude[h_mask]**2) / (dir_total + 1e-10)
    e_vertical = np.sum(magnitude[v_mask]**2) / (dir_total + 1e-10)
    e_diagonal1 = np.sum(magnitude[d1_mask]**2) / (dir_total + 1e-10)
    e_diagonal2 = np.sum(magnitude[d2_mask]**2) / (dir_total + 1e-10)

    # === ANISOTROPY INDEX ===
    dir_energies = [e_horizontal, e_vertical, e_diagonal1, e_diagonal2]
    anisotropy = max(dir_energies) / (np.mean(dir_energies) + 1e-10)

    # === HORIZONTAL/VERTICAL RATIO ===
    h_v_ratio = e_horizontal / (e_vertical + 1e-10)

    # === PSD SLOPE (1/f analysis) ===
    # Radial average
    radii = np.unique(radius.astype(int))
    radial_profile = []
    for r in radii[1:min(100, len(radii))]:  # Skip DC, limit to 100 points
        mask = (radius >= r-0.5) & (radius < r+0.5)
        if np.any(mask):
            radial_profile.append(np.mean(magnitude[mask]))

    if len(radial_profile) > 10:
        radial_profile = np.array(radial_profile)
        log_freq = np.log10(np.arange(1, len(radial_profile) + 1))
        log_psd = np.log10(radial_profile + 1e-10)
        slope, intercept, r_value, _, _ = stats.linregress(log_freq, log_psd)
        psd_slope = slope
        psd_r_squared = r_value**2
    else:
        psd_slope = 0
        psd_r_squared = 0

    # === HIGH-FREQUENCY EDGE SHARPNESS ===
    # Ratio of very high freq (>0.7) to mid-high (0.5-0.7)
    very_high_mask = normalized_radius > 0.7
    mid_high_mask = (normalized_radius > 0.5) & (normalized_radius <= 0.7)
    e_very_high = np.sum(magnitude[very_high_mask]**2) / total_energy
    e_mid_high = np.sum(magnitude[mid_high_mask]**2) / total_energy
    edge_sharpness = e_very_high / (e_mid_high + 1e-10)

    # === FREQUENCY CONCENTRATION ===
    # How concentrated is energy in certain frequencies
    sorted_energies = np.sort(psd.flatten())[::-1]
    top_10_percent = int(0.1 * len(sorted_energies))
    freq_concentration = np.sum(sorted_energies[:top_10_percent]) / total_energy

    return {
        # Basic
        'e_low': e_low,
        'e_mid': e_mid,
        'e_high': e_high,
        'high_low_ratio': high_low_ratio,

        # Spectral moments
        'spectral_entropy': spectral_entropy,
        'freq_centroid': freq_centroid,
        'spectral_spread': spectral_spread,
        'spectral_flatness': spectral_flatness,
        'spectral_rolloff': spectral_rolloff,
        'spectral_kurtosis': spectral_kurtosis,
        'spectral_skewness': spectral_skewness,

        # Directional
        'e_horizontal': e_horizontal,
        'e_vertical': e_vertical,
        'e_diagonal1': e_diagonal1,
        'e_diagonal2': e_diagonal2,
        'anisotropy': anisotropy,
        'h_v_ratio': h_v_ratio,

        # PSD
        'psd_slope': psd_slope,
        'psd_r_squared': psd_r_squared,

        # Edge/sharpness
        'edge_sharpness': edge_sharpness,
        'freq_concentration': freq_concentration,
        'e_very_high': e_very_high,

        # Granular rings
        **{f'ring_{i}': ring_energies[i] for i in range(10)}
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def compute_auc(group1, group2):
    """Compute ROC-AUC for binary classification."""
    from sklearn.metrics import roc_auc_score
    labels = np.array([1]*len(group1) + [0]*len(group2))
    scores = np.concatenate([group1, group2])
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1-auc)  # Return best direction
    except:
        return 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find Fourier features that discriminate tooltip vs background images.\n"
            "By default, excludes augmented images whose filename contains 'AUG'."
        )
    )
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--include_aug",
        action="store_true",
        help="Include augmented images (do NOT filter filenames containing the token)",
    )
    parser.add_argument(
        "--aug_token",
        type=str,
        default="AUG",
        help="Filename token marking augmented images (case-insensitive)",
    )
    return parser.parse_args()


def main():
    print("=" * 60)
    print("Finding Discriminative Fourier Features")
    print("Tooltip vs Background Analysis")
    print("=" * 60)

    args = parse_args()

    # Sample images
    n_samples = args.n_samples  # Per class
    random.seed(args.seed)

    # Get tooltip images
    tooltip_images = sorted(TOOLTIP_DIR.glob('*.jpg'))
    if not args.include_aug:
        tooltip_images, tooltip_excluded = filter_images_by_filename_token(tooltip_images, args.aug_token)
        print(f"\nExcluded {tooltip_excluded} tooltip images containing '{args.aug_token}' in filename")

    tooltip_sample = random.sample(tooltip_images, min(n_samples, len(tooltip_images)))
    print(f"\nTooltip images: {len(tooltip_images)} total, sampling {len(tooltip_sample)}")

    # Get background images
    bg_train = sorted(BACKGROUND_TRAIN.glob('*.jpg'))
    bg_val = sorted(BACKGROUND_VAL.glob('*.jpg'))
    bg_images = bg_train + bg_val
    if not args.include_aug:
        bg_images, bg_excluded = filter_images_by_filename_token(bg_images, args.aug_token)
        print(f"Excluded {bg_excluded} background images containing '{args.aug_token}' in filename")

    bg_sample = random.sample(bg_images, min(n_samples, len(bg_images)))
    print(f"Background images: {len(bg_images)} total, sampling {len(bg_sample)}")

    # Compute features
    print("\nComputing features for tooltip images...")
    tooltip_features = []
    for img_path in tqdm(tooltip_sample):
        img = cv2.imread(str(img_path))
        if img is not None:
            feat = compute_advanced_fft_features(img)
            tooltip_features.append(feat)

    print("Computing features for background images...")
    bg_features = []
    for img_path in tqdm(bg_sample):
        img = cv2.imread(str(img_path))
        if img is not None:
            feat = compute_advanced_fft_features(img)
            bg_features.append(feat)

    if not tooltip_features:
        raise RuntimeError("No tooltip features computed (no readable images after filtering/sampling).")
    if not bg_features:
        raise RuntimeError("No background features computed (no readable images after filtering/sampling).")

    # Analyze discriminative power
    print("\n" + "=" * 60)
    print("DISCRIMINATIVE POWER ANALYSIS")
    print("=" * 60)

    feature_names = list(tooltip_features[0].keys())
    results = []

    for feat_name in feature_names:
        tooltip_vals = np.array([f[feat_name] for f in tooltip_features])
        bg_vals = np.array([f[feat_name] for f in bg_features])

        # Statistics
        t_stat, p_value = stats.ttest_ind(tooltip_vals, bg_vals)
        d = cohens_d(tooltip_vals, bg_vals)
        auc = compute_auc(tooltip_vals, bg_vals)

        results.append({
            'feature': feat_name,
            'tooltip_mean': np.mean(tooltip_vals),
            'tooltip_std': np.std(tooltip_vals),
            'bg_mean': np.mean(bg_vals),
            'bg_std': np.std(bg_vals),
            'difference': np.mean(tooltip_vals) - np.mean(bg_vals),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'auc': auc
        })

    # Sort by AUC (best discriminator)
    results.sort(key=lambda x: x['auc'], reverse=True)

    # Print top features
    print("\n" + "-" * 80)
    print(f"{'Feature':<25} {'Tooltip':<12} {'Background':<12} {'Diff':<10} {'Cohen d':<10} {'AUC':<8}")
    print("-" * 80)

    for r in results[:20]:  # Top 20
        print(f"{r['feature']:<25} {r['tooltip_mean']:<12.4f} {r['bg_mean']:<12.4f} "
              f"{r['difference']:<+10.4f} {r['cohens_d']:<+10.3f} {r['auc']:<8.3f}")

    # Save full results
    output_path = Path(__file__).parent / "output" / "discriminative_features.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TOP 5 MOST DISCRIMINATIVE FEATURES")
    print("=" * 60)
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['feature']}")
        print(f"   Tooltip: {r['tooltip_mean']:.4f} +/- {r['tooltip_std']:.4f}")
        print(f"   Background: {r['bg_mean']:.4f} +/- {r['bg_std']:.4f}")
        print(f"   Cohen's d: {r['cohens_d']:+.3f}  |  AUC: {r['auc']:.3f}")

        if r['auc'] > 0.7:
            print(f"   --> GOOD DISCRIMINATOR!")


if __name__ == '__main__':
    main()
