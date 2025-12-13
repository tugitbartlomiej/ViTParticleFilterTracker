#!/usr/bin/env python3
"""
Visualize Discriminative Fourier Features for Scientific Paper

Based on find_discriminative_features.py analysis, this script creates
publication-quality visualizations comparing tooltip vs background images
using the most discriminative spectral features.

Top discriminative features (latest JSON, by AUC):
1. spectral_spread (d=-0.444, AUC=0.649)
2. e_diagonal1 (d=+0.620, AUC=0.641)
3. spectral_entropy (d=+0.415, AUC=0.612)
4. edge_sharpness (d=-0.438, AUC=0.609)
5. psd_slope (d=+0.376, AUC=0.606)
Best ring feature: ring_6 (AUC=0.599)

Usage:
    # Compare two specific images
    py -3.11 visualize_discriminative_features.py --compare tooltip.jpg background.jpg

    # Analyze multiple tooltip vs background images
    py -3.11 visualize_discriminative_features.py --batch --n_samples 50

    # Generate summary from pre-computed JSON
    py -3.11 visualize_discriminative_features.py --summary

    # Help
    py -3.11 visualize_discriminative_features.py --help
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge, Circle
import matplotlib.patches as mpatches
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis, skew
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default paths
DEFAULT_TOOLTIP_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images")
DEFAULT_BG_TRAIN = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\BackgroundFinetuned\Datasets\Background\train")
DEFAULT_BG_VAL = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\BackgroundFinetuned\Datasets\Background\val")
DEFAULT_OUTPUT_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\fourier")
DEFAULT_FEATURES_JSON = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\discriminative_features.json")


# Feature display names and interpretations
FEATURE_INFO = {
    'e_diagonal1': {'name': 'Diagonal 1 Energy', 'unit': '', 'interp': 'Edge orientation (45°)'},
    'e_diagonal2': {'name': 'Diagonal 2 Energy', 'unit': '', 'interp': 'Edge orientation (-45°)'},
    'spectral_entropy': {'name': 'Spectral Entropy', 'unit': '', 'interp': 'Texture complexity'},
    'spectral_spread': {'name': 'Spectral Spread', 'unit': '', 'interp': 'Frequency variance'},
    'spectral_flatness': {'name': 'Spectral Flatness', 'unit': '', 'interp': 'Noise-like vs tonal'},
    'spectral_rolloff': {'name': 'Spectral Rolloff', 'unit': '', 'interp': '85% energy cutoff'},
    'spectral_kurtosis': {'name': 'Spectral Kurtosis', 'unit': '', 'interp': 'Spectrum peakiness'},
    'spectral_skewness': {'name': 'Spectral Skewness', 'unit': '', 'interp': 'Spectrum asymmetry'},
    'e_low': {'name': 'Low Freq Energy', 'unit': '', 'interp': 'Smooth regions (0-10%)'},
    'e_mid': {'name': 'Mid Freq Energy', 'unit': '', 'interp': 'Textures (10-50%)'},
    'e_high': {'name': 'High Freq Energy', 'unit': '', 'interp': 'Edges/details (50-100%)'},
    'e_very_high': {'name': 'Very High Freq', 'unit': '', 'interp': 'Fine edges (>70%)'},
    'high_low_ratio': {'name': 'High/Low Ratio', 'unit': '', 'interp': 'Edge prominence'},
    'e_horizontal': {'name': 'Horizontal Energy', 'unit': '', 'interp': 'Horizontal edges'},
    'e_vertical': {'name': 'Vertical Energy', 'unit': '', 'interp': 'Vertical edges'},
    'anisotropy': {'name': 'Anisotropy Index', 'unit': '', 'interp': 'Directional bias'},
    'h_v_ratio': {'name': 'H/V Ratio', 'unit': '', 'interp': 'Horiz. vs Vert. edges'},
    'psd_slope': {'name': 'PSD Slope (β)', 'unit': '', 'interp': '1/f spectral decay'},
    'psd_r_squared': {'name': 'PSD R²', 'unit': '', 'interp': '1/f fit quality'},
    'edge_sharpness': {'name': 'Edge Sharpness', 'unit': '', 'interp': 'Very high / mid-high'},
    'freq_centroid': {'name': 'Freq Centroid', 'unit': '', 'interp': 'Weighted frequency center'},
    'freq_concentration': {'name': 'Freq Concentration', 'unit': '', 'interp': 'Energy in top 10%'},
}

# Ring feature names
for i in range(10):
    FEATURE_INFO[f'ring_{i}'] = {
        'name': f'Ring {i} ({i*10}-{(i+1)*10}%)',
        'unit': '',
        'interp': f'Freq band {i*10}-{(i+1)*10}%'
    }

# Default feature fallbacks (used if JSON rankings are unavailable)
DEFAULT_TOP_FEATURES = [
    'e_diagonal1', 'spectral_entropy', 'spectral_spread',
    'ring_1', 'e_low', 'e_mid', 'psd_slope', 'anisotropy'
]
DEFAULT_BATCH_FEATURES = [
    'e_diagonal1', 'spectral_entropy', 'spectral_spread',
    'ring_1', 'e_low', 'psd_slope'
]

def compute_advanced_fft_features(image):
    """
    Compute comprehensive FFT features for discrimination.
    Same as in find_discriminative_features.py for consistency.
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
    radii = np.unique(radius.astype(int))
    radial_profile = []
    for r in radii[1:min(100, len(radii))]:
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
    very_high_mask = normalized_radius > 0.7
    mid_high_mask = (normalized_radius > 0.5) & (normalized_radius <= 0.7)
    e_very_high = np.sum(magnitude[very_high_mask]**2) / total_energy
    e_mid_high = np.sum(magnitude[mid_high_mask]**2) / total_energy
    edge_sharpness = e_very_high / (e_mid_high + 1e-10)

    # === FREQUENCY CONCENTRATION ===
    sorted_energies = np.sort(psd.flatten())[::-1]
    top_10_percent = int(0.1 * len(sorted_energies))
    freq_concentration = np.sum(sorted_energies[:top_10_percent]) / total_energy

    return {
        # For visualization
        'magnitude_log': magnitude_log,
        'normalized_radius': normalized_radius,
        'gray': gray,

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


class DiscriminativeVisualizer:
    """Visualize discriminative Fourier features for tooltip vs background comparison."""

    def __init__(self, features_json_path=DEFAULT_FEATURES_JSON):
        # Publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14

        # Colors
        self.color_tooltip = '#e74c3c'   # Red
        self.color_bg = '#3498db'        # Blue
        self.color_neutral = '#95a5a6'   # Gray

        # Feature rankings (from discriminative_features.json if available)
        self.features_json_path = Path(features_json_path)
        self.rankings = self._load_feature_rankings()
        self.feature_stats = {r.get('feature'): r for r in self.rankings}
        self.top_features_auc = [
            r.get('feature') for r in sorted(self.rankings, key=lambda x: x.get('auc', 0), reverse=True)
            if r.get('feature')
        ]
        self.best_ring_feature = self._get_best_ring_feature()

    def _load_feature_rankings(self):
        """Load discriminative feature rankings from JSON (silently fall back if missing)."""
        if self.features_json_path and self.features_json_path.exists():
            try:
                with open(self.features_json_path, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as exc:
                print(f"Warning: could not load feature rankings from {self.features_json_path}: {exc}")
        return []

    def _get_best_ring_feature(self):
        ring_entries = [r for r in self.rankings if str(r.get('feature', '')).startswith('ring_')]
        if ring_entries:
            best = max(ring_entries, key=lambda x: x.get('auc', 0))
            return best.get('feature', 'ring_1')
        return 'ring_1'

    def get_top_features(self, n=8, fallback=None):
        """Return top-n features by AUC, falling back to defaults if ranking unavailable."""
        if self.top_features_auc:
            return self.top_features_auc[:n]
        if fallback is None:
            fallback = DEFAULT_TOP_FEATURES
        return fallback[:n]

    def format_feature_label(self, feature):
        """Return compact label with Cohen's d and AUC for tables."""
        stats_entry = self.feature_stats.get(feature)
        if stats_entry:
            d_val = stats_entry.get('cohens_d')
            auc_val = stats_entry.get('auc')
            if d_val is not None and auc_val is not None:
                return f"d={d_val:+.3f}, AUC={auc_val:.3f}"
        return 'd=n/a, AUC=n/a'

    def load_discriminative_features(self, json_path=None):
        """Load pre-computed discriminative features from JSON."""
        if json_path is None:
            json_path = DEFAULT_FEATURES_JSON

        with open(json_path, 'r') as f:
            return json.load(f)

    def visualize_comparison(self, img1_path, img2_path, output_path,
                            label1="Tooltip", label2="Background", show=False):
        """
        Create comprehensive visualization comparing two images
        using discriminative features.

        Shows:
        - Original images + FFT spectra
        - Ring energy distribution (10 rings)
        - Directional energy radar chart
        - Top discriminative features comparison
        - Statistical summary
        """
        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            print(f"Error loading images")
            return None

        # Compute features
        feat1 = compute_advanced_fft_features(img1)
        feat2 = compute_advanced_fft_features(img2)

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Create figure
        fig = plt.figure(figsize=(22, 18))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                     height_ratios=[1, 1.2, 1, 0.6])

        # === Row 1: Images and FFT spectra ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1_rgb)
        ax1.set_title(f'(a) {label1}', fontweight='bold', color=self.color_tooltip)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(feat1['magnitude_log'], cmap='inferno')
        ax2.set_title(f'(b) FFT - {label1}', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img2_rgb)
        ax3.set_title(f'(c) {label2}', fontweight='bold', color=self.color_bg)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(feat2['magnitude_log'], cmap='inferno')
        ax4.set_title(f'(d) FFT - {label2}', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # === Row 2: Ring energies and directional ===
        # Ring energy distribution (left half)
        ax5 = fig.add_subplot(gs[1, 0:2])
        rings = [f'ring_{i}' for i in range(10)]
        ring_labels = [f'{i*10}-{(i+1)*10}%' for i in range(10)]
        x = np.arange(len(rings))
        width = 0.35

        vals1 = [feat1[r] for r in rings]
        vals2 = [feat2[r] for r in rings]

        bars1 = ax5.bar(x - width/2, vals1, width, label=label1,
                       color=self.color_tooltip, edgecolor='black', alpha=0.8)
        bars2 = ax5.bar(x + width/2, vals2, width, label=label2,
                       color=self.color_bg, edgecolor='black', alpha=0.8)

        ax5.set_xlabel('Frequency Band (normalized radius)')
        ax5.set_ylabel('Energy Fraction')
        ax5.set_title('(e) Ring Energy Distribution (10 Frequency Bands)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(ring_labels, rotation=45, ha='right')
        ax5.legend(loc='upper right')
        ax5.set_yscale('log')
        ax5.set_ylim(1e-5, 1)

        # Add annotation for most discriminative ring (from JSON ranking when available)
        try:
            best_ring_idx = int(self.best_ring_feature.split('_')[1])
        except (ValueError, AttributeError, IndexError):
            best_ring_idx = 1
        best_ring_idx = max(0, min(best_ring_idx, len(rings) - 1))
        ax5.annotate(f'BEST\nring_{best_ring_idx}', xy=(best_ring_idx, max(vals1[best_ring_idx], vals2[best_ring_idx])),
                    xytext=(best_ring_idx + 1, 0.1), fontsize=9, color='green',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Directional energy radar chart (right half)
        ax6 = fig.add_subplot(gs[1, 2:4], polar=True)

        directions = ['Horizontal', 'Diagonal 1', 'Vertical', 'Diagonal 2']
        dir_keys = ['e_horizontal', 'e_diagonal1', 'e_vertical', 'e_diagonal2']

        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        dir_vals1 = [feat1[k] for k in dir_keys] + [feat1[dir_keys[0]]]
        dir_vals2 = [feat2[k] for k in dir_keys] + [feat2[dir_keys[0]]]

        ax6.plot(angles, dir_vals1, 'o-', linewidth=2, color=self.color_tooltip, label=label1)
        ax6.fill(angles, dir_vals1, alpha=0.25, color=self.color_tooltip)
        ax6.plot(angles, dir_vals2, 's-', linewidth=2, color=self.color_bg, label=label2)
        ax6.fill(angles, dir_vals2, alpha=0.25, color=self.color_bg)

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(directions)
        ax6.set_title('(f) Directional Energy Distribution', fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

        # === Row 3: Top discriminative features ===
        ax7 = fig.add_subplot(gs[2, 0:2])

        # Top discriminative features (ranked by AUC from JSON if available)
        top_features = self.get_top_features(8, fallback=DEFAULT_TOP_FEATURES)

        feat_labels = [FEATURE_INFO.get(f, {}).get('name', f) for f in top_features]
        x_top = np.arange(len(top_features))

        top_vals1 = [feat1[f] for f in top_features]
        top_vals2 = [feat2[f] for f in top_features]

        # Normalize for visualization (min-max per feature)
        top_vals1_norm = []
        top_vals2_norm = []
        for v1, v2 in zip(top_vals1, top_vals2):
            min_v, max_v = min(v1, v2), max(v1, v2)
            range_v = max_v - min_v if max_v != min_v else 1
            top_vals1_norm.append((v1 - min_v) / range_v)
            top_vals2_norm.append((v2 - min_v) / range_v)

        bars1 = ax7.barh(x_top - 0.2, top_vals1_norm, 0.35, label=label1,
                        color=self.color_tooltip, edgecolor='black', alpha=0.8)
        bars2 = ax7.barh(x_top + 0.2, top_vals2_norm, 0.35, label=label2,
                        color=self.color_bg, edgecolor='black', alpha=0.8)

        # Add actual values as text
        for i, (v1, v2) in enumerate(zip(top_vals1, top_vals2)):
            ax7.text(1.05, i - 0.2, f'{v1:.4f}', va='center', fontsize=8,
                    color=self.color_tooltip)
            ax7.text(1.05, i + 0.2, f'{v2:.4f}', va='center', fontsize=8,
                    color=self.color_bg)

        ax7.set_yticks(x_top)
        ax7.set_yticklabels(feat_labels)
        ax7.set_xlabel('Normalized Value (min-max per feature)')
        ax7.set_title('(g) Top Discriminative Features Comparison', fontweight='bold')
        ax7.legend(loc='lower right')
        ax7.set_xlim(0, 1.25)

        # Spectral statistics comparison
        ax8 = fig.add_subplot(gs[2, 2:4])

        spectral_features = ['spectral_entropy', 'spectral_spread', 'spectral_flatness',
                           'freq_centroid', 'spectral_rolloff']
        spec_labels = ['Entropy', 'Spread', 'Flatness', 'Centroid', 'Rolloff']
        x_spec = np.arange(len(spectral_features))

        spec_vals1 = [feat1[f] for f in spectral_features]
        spec_vals2 = [feat2[f] for f in spectral_features]

        bars1 = ax8.bar(x_spec - width/2, spec_vals1, width, label=label1,
                       color=self.color_tooltip, edgecolor='black', alpha=0.8)
        bars2 = ax8.bar(x_spec + width/2, spec_vals2, width, label=label2,
                       color=self.color_bg, edgecolor='black', alpha=0.8)

        ax8.set_xticks(x_spec)
        ax8.set_xticklabels(spec_labels)
        ax8.set_ylabel('Value')
        ax8.set_title('(h) Spectral Statistics Comparison', fontweight='bold')
        ax8.legend(loc='upper right')

        # Add value annotations
        for bar, val in zip(bars1, spec_vals1):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                    color=self.color_tooltip, rotation=45)
        for bar, val in zip(bars2, spec_vals2):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                    color=self.color_bg, rotation=45)

        # === Row 4: Summary table ===
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')

        # Key discriminative metrics table (top features by AUC when available)
        table_features = self.get_top_features(6, fallback=DEFAULT_TOP_FEATURES)
        table_data = []
        for feat in table_features:
            display_name = FEATURE_INFO.get(feat, {}).get('name', feat)
            v1 = feat1.get(feat, 0)
            v2 = feat2.get(feat, 0)
            diff = v1 - v2
            table_data.append([
                display_name,
                f'{v1:.6f}',
                f'{v2:.6f}',
                f'{diff:+.6f}',
                self.format_feature_label(feat)
            ])

        col_labels = ['Feature', label1, label2, 'Difference', 'd / AUC']

        table = ax9.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center',
                         colColours=['lightgray']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Color the header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
            elif col == 1:
                cell.set_text_props(color=self.color_tooltip)
            elif col == 2:
                cell.set_text_props(color=self.color_bg)

        ax9.set_title('(i) Key Discriminative Features Summary', fontweight='bold', y=0.95)

        # Main title
        plt.suptitle(f'Discriminative Fourier Features: {label1} vs {label2}\n'
                    f'Based on Cohen\'s d effect size analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

        return {'feat1': feat1, 'feat2': feat2}

    def visualize_batch_comparison(self, tooltip_features, bg_features, output_path,
                                  show=False, save_json=True):
        """
        Create statistical visualization comparing multiple tooltip vs background images.

        Shows:
        - Box plots for top discriminative features
        - Violin plots for distributions
        - Effect size summary
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Top discriminative features (ranked by AUC when available)
        top_features = self.get_top_features(6, fallback=DEFAULT_BATCH_FEATURES)

        # === Box plots (row 1) ===
        for i, feat in enumerate(top_features[:3]):
            ax = fig.add_subplot(gs[0, i])

            tooltip_vals = [f[feat] for f in tooltip_features]
            bg_vals = [f[feat] for f in bg_features]

            data = [tooltip_vals, bg_vals]
            bp = ax.boxplot(data, labels=['Tooltip', 'Background'], patch_artist=True)

            bp['boxes'][0].set_facecolor(self.color_tooltip)
            bp['boxes'][1].set_facecolor(self.color_bg)
            for box in bp['boxes']:
                box.set_alpha(0.7)

            ax.set_title(FEATURE_INFO.get(feat, {}).get('name', feat), fontweight='bold')
            ax.set_ylabel('Value')

            # Add significance marker
            t_stat, p_val = stats.ttest_ind(tooltip_vals, bg_vals)
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            ax.text(1.5, ax.get_ylim()[1], sig, ha='center', fontsize=14, fontweight='bold')

        # === Violin plots (row 2) ===
        for i, feat in enumerate(top_features[3:6]):
            ax = fig.add_subplot(gs[1, i])

            tooltip_vals = [f[feat] for f in tooltip_features]
            bg_vals = [f[feat] for f in bg_features]

            parts = ax.violinplot([tooltip_vals, bg_vals], positions=[1, 2], showmeans=True)

            parts['bodies'][0].set_facecolor(self.color_tooltip)
            parts['bodies'][0].set_alpha(0.7)
            parts['bodies'][1].set_facecolor(self.color_bg)
            parts['bodies'][1].set_alpha(0.7)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Tooltip', 'Background'])
            ax.set_title(FEATURE_INFO.get(feat, {}).get('name', feat), fontweight='bold')
            ax.set_ylabel('Value')

        # === Ring energies comparison (row 3, left) ===
        ax_ring = fig.add_subplot(gs[2, 0:2])

        rings = [f'ring_{i}' for i in range(10)]
        ring_labels = [f'R{i}' for i in range(10)]
        x = np.arange(len(rings))
        width = 0.35

        tooltip_ring_means = [np.mean([f[r] for f in tooltip_features]) for r in rings]
        tooltip_ring_stds = [np.std([f[r] for f in tooltip_features]) for r in rings]
        bg_ring_means = [np.mean([f[r] for f in bg_features]) for r in rings]
        bg_ring_stds = [np.std([f[r] for f in bg_features]) for r in rings]

        ax_ring.bar(x - width/2, tooltip_ring_means, width, yerr=tooltip_ring_stds,
                   label='Tooltip', color=self.color_tooltip, alpha=0.8, capsize=2)
        ax_ring.bar(x + width/2, bg_ring_means, width, yerr=bg_ring_stds,
                   label='Background', color=self.color_bg, alpha=0.8, capsize=2)

        ax_ring.set_xticks(x)
        ax_ring.set_xticklabels(ring_labels)
        ax_ring.set_xlabel('Frequency Ring')
        ax_ring.set_ylabel('Mean Energy (log scale)')
        ax_ring.set_title('Ring Energy Distribution (Mean ± Std)', fontweight='bold')
        ax_ring.set_yscale('log')
        ax_ring.legend()

        # === Effect size summary (row 3, right) ===
        ax_effect = fig.add_subplot(gs[2, 2])

        effect_features = self.get_top_features(8, fallback=DEFAULT_TOP_FEATURES)

        cohens_d = []
        for feat in effect_features:
            t_vals = np.array([f[feat] for f in tooltip_features])
            b_vals = np.array([f[feat] for f in bg_features])
            n1, n2 = len(t_vals), len(b_vals)
            var1, var2 = np.var(t_vals, ddof=1), np.var(b_vals, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            d = (np.mean(t_vals) - np.mean(b_vals)) / (pooled_std + 1e-10)
            cohens_d.append(d)

        colors = [self.color_tooltip if d > 0 else self.color_bg for d in cohens_d]
        y_pos = np.arange(len(effect_features))

        ax_effect.barh(y_pos, cohens_d, color=colors, alpha=0.8, edgecolor='black')
        ax_effect.axvline(0, color='black', linewidth=1)
        ax_effect.axvline(0.5, color='green', linewidth=1, linestyle='--', alpha=0.5)
        ax_effect.axvline(-0.5, color='green', linewidth=1, linestyle='--', alpha=0.5)

        ax_effect.set_yticks(y_pos)
        ax_effect.set_yticklabels([FEATURE_INFO.get(f, {}).get('name', f)[:15] for f in effect_features])
        ax_effect.set_xlabel("Cohen's d (effect size)")
        ax_effect.set_title("Effect Size Summary\n(|d|>0.5 = medium effect)", fontweight='bold')

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=self.color_tooltip, label='Tooltip > Background'),
            mpatches.Patch(facecolor=self.color_bg, label='Background > Tooltip'),
        ]
        ax_effect.legend(handles=legend_elements, loc='lower right', fontsize=9)

        # Main title
        plt.suptitle(f'Batch Statistical Comparison: Tooltip vs Background\n'
                    f'(N={len(tooltip_features)} tooltip, N={len(bg_features)} background)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

        # Save JSON summary
        if save_json:
            summary = {
                'generated_at': datetime.now().isoformat(),
                'n_tooltip': len(tooltip_features),
                'n_background': len(bg_features),
                'effect_sizes': {
                    feat: float(d) for feat, d in zip(effect_features, cohens_d)
                },
                'ring_means': {
                    'tooltip': {f'ring_{i}': float(m) for i, m in enumerate(tooltip_ring_means)},
                    'background': {f'ring_{i}': float(m) for i, m in enumerate(bg_ring_means)}
                }
            }
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved: {json_path}")

        return {'cohens_d': dict(zip(effect_features, cohens_d))}

    def visualize_summary_from_json(self, json_path, output_path, show=False):
        """
        Create summary visualization from pre-computed discriminative_features.json
        """
        with open(json_path, 'r') as f:
            results = json.load(f)

        # Sort by AUC
        results.sort(key=lambda x: x['auc'], reverse=True)

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # === Top features by AUC ===
        ax1 = fig.add_subplot(gs[0, 0])

        top_n = 15
        top_features = results[:top_n]
        feat_names = [r['feature'] for r in top_features]
        aucs = [r['auc'] for r in top_features]
        cohens = [r['cohens_d'] for r in top_features]

        colors = [self.color_tooltip if d > 0 else self.color_bg for d in cohens]
        y_pos = np.arange(len(feat_names))

        bars = ax1.barh(y_pos, aucs, color=colors, alpha=0.8, edgecolor='black')
        ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
        ax1.axvline(0.7, color='green', linestyle='--', alpha=0.7, label='Good (AUC=0.7)')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feat_names)
        ax1.set_xlabel('ROC-AUC')
        ax1.set_title(f'Top {top_n} Discriminative Features (by AUC)', fontweight='bold')
        ax1.set_xlim(0.5, 0.75)
        ax1.legend(loc='lower right')

        # Add value annotations
        for bar, auc in zip(bars, aucs):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{auc:.3f}', va='center', fontsize=9)

        # === Effect size (Cohen's d) ===
        ax2 = fig.add_subplot(gs[0, 1])

        # Sort by absolute Cohen's d
        sorted_by_d = sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)[:top_n]
        feat_names_d = [r['feature'] for r in sorted_by_d]
        cohens_d_vals = [r['cohens_d'] for r in sorted_by_d]

        colors_d = [self.color_tooltip if d > 0 else self.color_bg for d in cohens_d_vals]
        y_pos_d = np.arange(len(feat_names_d))

        bars_d = ax2.barh(y_pos_d, cohens_d_vals, color=colors_d, alpha=0.8, edgecolor='black')
        ax2.axvline(0, color='black', linewidth=1)
        ax2.axvline(0.5, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(-0.5, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(0.8, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(-0.8, color='red', linestyle='--', alpha=0.5)

        ax2.set_yticks(y_pos_d)
        ax2.set_yticklabels(feat_names_d)
        ax2.set_xlabel("Cohen's d (effect size)")
        ax2.set_title(f'Top {top_n} Features (by |Cohen\'s d|)', fontweight='bold')

        # Legend
        ax2.text(0.95, 0.95, '+d: Tooltip > BG\n-d: BG > Tooltip\n|d|>0.5: medium\n|d|>0.8: large',
                transform=ax2.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # === Tooltip vs Background means ===
        ax3 = fig.add_subplot(gs[1, 0])

        # Ring features
        ring_features = [r for r in results if r['feature'].startswith('ring_')]
        ring_features.sort(key=lambda x: int(x['feature'].split('_')[1]))

        ring_names = [r['feature'] for r in ring_features]
        tooltip_means = [r['tooltip_mean'] for r in ring_features]
        bg_means = [r['bg_mean'] for r in ring_features]

        x_ring = np.arange(len(ring_names))
        width = 0.35

        ax3.bar(x_ring - width/2, tooltip_means, width, label='Tooltip',
               color=self.color_tooltip, alpha=0.8)
        ax3.bar(x_ring + width/2, bg_means, width, label='Background',
               color=self.color_bg, alpha=0.8)

        ax3.set_xticks(x_ring)
        ax3.set_xticklabels([f'R{i}' for i in range(10)])
        ax3.set_xlabel('Frequency Ring')
        ax3.set_ylabel('Mean Energy')
        ax3.set_title('Ring Energy: Tooltip vs Background Means', fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend()

        # === Summary statistics table ===
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        table_data = []
        for r in results[:10]:
            row = [
                r['feature'],
                f"{r['tooltip_mean']:.4f}",
                f"{r['bg_mean']:.4f}",
                f"{r['cohens_d']:+.3f}",
                f"{r['auc']:.3f}",
                '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else '*')
            ]
            table_data.append(row)

        col_labels = ['Feature', 'Tooltip', 'Background', "Cohen's d", 'AUC', 'Sig']

        table = ax4.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center',
                         colColours=['lightgray']*6)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.6)

        ax4.set_title('Top 10 Discriminative Features Summary', fontweight='bold', y=0.95)

        # Main title
        plt.suptitle('Discriminative Fourier Features Analysis Summary\n'
                    '(From tooltip vs background comparison)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Discriminative Fourier Features',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--compare', nargs=2, type=str, default=None,
                           metavar=('IMG1', 'IMG2'),
                           help='Compare two specific images')
    mode_group.add_argument('--batch', action='store_true',
                           help='Batch comparison of tooltip vs background datasets')
    mode_group.add_argument('--summary', action='store_true',
                           help='Generate summary from pre-computed JSON')

    # Batch options
    batch_group = parser.add_argument_group('Batch Options')
    batch_group.add_argument('--n_samples', type=int, default=50,
                            help='Number of samples per class for batch (default: 50)')
    batch_group.add_argument('--tooltip_dir', type=str, default=str(DEFAULT_TOOLTIP_DIR),
                            help='Directory with tooltip images')
    batch_group.add_argument('--bg_train', type=str, default=str(DEFAULT_BG_TRAIN),
                            help='Directory with background training images')
    batch_group.add_argument('--bg_val', type=str, default=str(DEFAULT_BG_VAL),
                            help='Directory with background validation images')

    # Compare options
    compare_group = parser.add_argument_group('Compare Options')
    compare_group.add_argument('--labels', nargs=2, type=str,
                              default=['Tooltip', 'Background'],
                              help='Labels for the two images')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                             help='Output directory')
    output_group.add_argument('--features_json', type=str, default=str(DEFAULT_FEATURES_JSON),
                             help='Path to discriminative_features.json')
    output_group.add_argument('--show', action='store_true',
                             help='Show plots interactively')
    output_group.add_argument('--seed', type=int, default=42,
                             help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = DiscriminativeVisualizer(features_json_path=args.features_json)

    # === Mode 1: Compare two images ===
    if args.compare:
        print("\n=== Image Comparison Mode ===")

        img1_path = Path(args.compare[0])
        img2_path = Path(args.compare[1])

        # Resolve paths
        tooltip_dir = Path(args.tooltip_dir)
        if not img1_path.is_absolute():
            img1_path = tooltip_dir / img1_path
        if not img2_path.is_absolute():
            # Try background dirs
            if not img2_path.exists():
                for bg_dir in [Path(args.bg_train), Path(args.bg_val)]:
                    candidate = bg_dir / args.compare[1]
                    if candidate.exists():
                        img2_path = candidate
                        break

        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")

        output_path = output_dir / f"discriminative_comparison_{img1_path.stem}_vs_{img2_path.stem}.png"

        visualizer.visualize_comparison(
            str(img1_path), str(img2_path), str(output_path),
            label1=args.labels[0], label2=args.labels[1],
            show=args.show
        )

        print(f"\n=== Done! Saved to: {output_path} ===")
        return

    # === Mode 2: Batch comparison ===
    if args.batch:
        print("\n=== Batch Comparison Mode ===")

        tooltip_dir = Path(args.tooltip_dir)
        bg_train = Path(args.bg_train)
        bg_val = Path(args.bg_val)

        # Get images
        tooltip_images = list(tooltip_dir.glob('*.jpg'))
        bg_images = list(bg_train.glob('*.jpg')) + list(bg_val.glob('*.jpg'))

        print(f"Found {len(tooltip_images)} tooltip images")
        print(f"Found {len(bg_images)} background images")

        # Sample
        n_samples = min(args.n_samples, len(tooltip_images), len(bg_images))
        tooltip_sample = random.sample(tooltip_images, n_samples)
        bg_sample = random.sample(bg_images, n_samples)

        print(f"Sampling {n_samples} from each class...")

        # Compute features
        print("\nComputing features for tooltip images...")
        tooltip_features = []
        for img_path in tqdm(tooltip_sample, desc="Tooltip"):
            img = cv2.imread(str(img_path))
            if img is not None:
                tooltip_features.append(compute_advanced_fft_features(img))

        print("Computing features for background images...")
        bg_features = []
        for img_path in tqdm(bg_sample, desc="Background"):
            img = cv2.imread(str(img_path))
            if img is not None:
                bg_features.append(compute_advanced_fft_features(img))

        # Generate visualization
        output_path = output_dir / f"discriminative_batch_n{n_samples}.png"

        visualizer.visualize_batch_comparison(
            tooltip_features, bg_features, str(output_path),
            show=args.show, save_json=True
        )

        # Also generate one comparison of most different images
        print("\nFinding most discriminative pair...")

        # Use e_diagonal1 as primary discriminator
        tooltip_d1 = [(i, f['e_diagonal1']) for i, f in enumerate(tooltip_features)]
        bg_d1 = [(i, f['e_diagonal1']) for i, f in enumerate(bg_features)]

        # Get highest tooltip and lowest background
        best_tooltip_idx = max(tooltip_d1, key=lambda x: x[1])[0]
        best_bg_idx = min(bg_d1, key=lambda x: x[1])[0]

        pair_output = output_dir / "discriminative_best_pair.png"
        visualizer.visualize_comparison(
            str(tooltip_sample[best_tooltip_idx]),
            str(bg_sample[best_bg_idx]),
            str(pair_output),
            label1="Tooltip (high e_diagonal1)",
            label2="Background (low e_diagonal1)",
            show=args.show
        )

        print(f"\n=== Done! Results saved to: {output_dir} ===")
        return

    # === Mode 3: Summary from JSON ===
    if args.summary:
        print("\n=== Summary from JSON Mode ===")

        json_path = Path(args.features_json)
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}")
            print("Run find_discriminative_features.py first to generate it.")
            return

        output_path = output_dir / "discriminative_summary.png"

        visualizer.visualize_summary_from_json(
            str(json_path), str(output_path), show=args.show
        )

        print(f"\n=== Done! Saved to: {output_path} ===")
        return

    # No mode selected - show help
    print("No mode selected. Use one of:")
    print("  --compare IMG1 IMG2  : Compare two images")
    print("  --batch              : Batch comparison of datasets")
    print("  --summary            : Generate summary from JSON")
    print("\nRun with --help for more options.")


if __name__ == '__main__':
    main()
