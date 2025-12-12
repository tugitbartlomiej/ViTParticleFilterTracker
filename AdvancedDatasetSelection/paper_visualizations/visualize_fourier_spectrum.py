#!/usr/bin/env python3
"""
Fourier Frequency Analysis Visualization for Scientific Paper

Generates publication-quality visualizations showing:
1. 2D FFT magnitude spectrum
2. Frequency band decomposition (low/mid/high)
3. Directional energy analysis
4. Spectral features comparison across images
5. PSD slope analysis (1/f law)
6. Anisotropy index for surgical instrument detection
7. Statistical comparison: tooltip vs background frames

Usage Examples:
    # First 5 images (sequential)
    py -3.11 visualize_fourier_spectrum.py --num_images 5

    # Random 5 images from dataset
    py -3.11 visualize_fourier_spectrum.py --num_images 5 --random

    # Specific images by filename
    py -3.11 visualize_fourier_spectrum.py --images image1.jpg image2.jpg

    # Compare tooltip vs background (scientific comparison)
    py -3.11 visualize_fourier_spectrum.py --compare tooltip.jpg background.jpg

    # Compare with labels
    py -3.11 visualize_fourier_spectrum.py --compare img1.jpg img2.jpg --labels "With Tooltip" "Background Only"

    # Show help
    py -3.11 visualize_fourier_spectrum.py --help
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge, Circle
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FourierVisualizer:
    """Professional Fourier analysis visualization for paper figures."""

    def __init__(self):
        # Frequency bands
        self.frequency_bands = {
            'low': (0.0, 0.1),
            'mid': (0.1, 0.5),
            'high': (0.5, 1.0)
        }

        # Colors for bands
        self.band_colors = {
            'low': '#2ecc71',   # Green
            'mid': '#f39c12',   # Orange
            'high': '#e74c3c'   # Red
        }

        # Set matplotlib style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def compute_fft(self, image):
        """Compute 2D FFT and related quantities."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float32) / 255.0

        # Compute FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Magnitude and phase
        magnitude = np.abs(f_shift)
        magnitude_log = np.log1p(magnitude)
        phase = np.angle(f_shift)

        # Create normalized radius map
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        max_radius = np.sqrt(center_row**2 + center_col**2)

        y, x = np.ogrid[:rows, :cols]
        radius = np.sqrt((y - center_row)**2 + (x - center_col)**2)
        normalized_radius = radius / max_radius

        # Compute band energies
        band_energies = {}
        band_masks = {}
        for band_name, (start, end) in self.frequency_bands.items():
            mask = (normalized_radius >= start) & (normalized_radius < end)
            band_masks[band_name] = mask
            band_energies[band_name] = np.sum(magnitude_log[mask])

        # Normalize energies
        total_energy = sum(band_energies.values()) + 1e-10
        band_energies_norm = {k: v / total_energy for k, v in band_energies.items()}

        # Spectral entropy
        magnitude_norm = magnitude_log / (np.sum(magnitude_log) + 1e-10)
        magnitude_norm = np.clip(magnitude_norm, 1e-10, 1.0)
        spectral_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10))
        max_entropy = np.log2(rows * cols)
        spectral_entropy_norm = spectral_entropy / max_entropy

        # Frequency centroid
        frequency_centroid = np.sum(normalized_radius * magnitude_log) / (np.sum(magnitude_log) + 1e-10)

        # Directional analysis
        angle = np.arctan2(y - center_row, x - center_col)
        dir_energies = {}
        dir_names = ['Horizontal', 'Vertical', 'Diagonal 1', 'Diagonal 2']
        dir_centers = [0, np.pi/2, np.pi/4, -np.pi/4]

        for name, center in zip(dir_names, dir_centers):
            angle_diff = np.minimum(
                np.abs(angle - center),
                np.minimum(np.abs(angle - (center + np.pi)), np.abs(angle - (center - np.pi)))
            )
            sector_mask = (angle_diff < np.pi/8) & (normalized_radius > 0.05)
            dir_energies[name] = np.sum(magnitude_log[sector_mask])

        # Normalize directional
        total_dir = sum(dir_energies.values()) + 1e-10
        dir_energies_norm = {k: v / total_dir for k, v in dir_energies.items()}

        # === PSD Slope Analysis (1/f law) ===
        # Compute radial Power Spectral Density
        psd_radii = np.linspace(0.01, 0.95, 50)  # Avoid DC and Nyquist
        psd_values = []
        for r in psd_radii:
            mask = (normalized_radius >= r - 0.02) & (normalized_radius < r + 0.02)
            if np.any(mask):
                psd_values.append(np.mean(magnitude[mask]**2))
            else:
                psd_values.append(1e-10)

        psd_values = np.array(psd_values)
        # Log-log linear fit for slope
        log_radii = np.log10(psd_radii + 1e-10)
        log_psd = np.log10(psd_values + 1e-10)

        # Linear regression for PSD slope
        valid_mask = np.isfinite(log_radii) & np.isfinite(log_psd)
        if np.sum(valid_mask) > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_radii[valid_mask], log_psd[valid_mask]
            )
            psd_slope = slope
            psd_r_squared = r_value**2
        else:
            psd_slope = -2.0  # Default natural image slope
            psd_r_squared = 0.0

        # === Anisotropy Index ===
        # AI = max(directional energy) / mean(directional energy)
        dir_values = list(dir_energies_norm.values())
        anisotropy_index = max(dir_values) / (np.mean(dir_values) + 1e-10)

        # === High-to-Low Energy Ratio ===
        high_low_ratio = band_energies_norm['high'] / (band_energies_norm['low'] + 1e-10)

        return {
            'magnitude': magnitude,
            'magnitude_log': magnitude_log,
            'phase': phase,
            'band_energies': band_energies_norm,
            'band_masks': band_masks,
            'spectral_entropy': spectral_entropy_norm,
            'frequency_centroid': frequency_centroid,
            'directional_energies': dir_energies_norm,
            'normalized_radius': normalized_radius,
            'gray': gray,
            'center': (center_row, center_col),
            # New metrics
            'psd_slope': psd_slope,
            'psd_r_squared': psd_r_squared,
            'psd_radii': psd_radii,
            'psd_values': psd_values,
            'anisotropy_index': anisotropy_index,
            'high_low_ratio': high_low_ratio
        }

    def visualize_single_image(self, image_path, output_path, show=False):
        """Create comprehensive Fourier visualization for a single image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read: {image_path}")
            return

        result = self.compute_fft(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Original, Grayscale, Magnitude Spectrum, Phase
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('(a) Original Image', fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(result['gray'], cmap='gray')
        ax2.set_title('(b) Grayscale', fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(result['magnitude_log'], cmap='inferno')
        ax3.set_title('(c) FFT Magnitude (log)', fontweight='bold')
        ax3.axis('off')
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Log Magnitude', fontsize=10)

        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(result['phase'], cmap='twilight')
        ax4.set_title('(d) FFT Phase', fontweight='bold')
        ax4.axis('off')
        cbar = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Phase (rad)', fontsize=10)

        # Row 2: Frequency bands visualization
        ax5 = fig.add_subplot(gs[1, 0])
        # Create band visualization
        band_vis = np.zeros((*result['magnitude_log'].shape, 3))
        for band_name, mask in result['band_masks'].items():
            color = np.array([int(self.band_colors[band_name][i:i+2], 16) for i in (1, 3, 5)]) / 255.0
            band_vis[mask] = color * (result['magnitude_log'][mask, np.newaxis] / result['magnitude_log'].max())
        ax5.imshow(band_vis)
        ax5.set_title('(e) Frequency Bands', fontweight='bold')
        ax5.axis('off')
        # Add legend
        for i, (band_name, color) in enumerate(self.band_colors.items()):
            ax5.plot([], [], 's', color=color, label=f'{band_name.capitalize()} ({self.frequency_bands[band_name][0]:.1f}-{self.frequency_bands[band_name][1]:.1f})')
        ax5.legend(loc='upper right', fontsize=9)

        # Band energy bar chart
        ax6 = fig.add_subplot(gs[1, 1])
        bands = list(result['band_energies'].keys())
        energies = [result['band_energies'][b] for b in bands]
        colors = [self.band_colors[b] for b in bands]
        bars = ax6.bar(bands, energies, color=colors, edgecolor='black')
        ax6.set_ylabel('Normalized Energy')
        ax6.set_title('(f) Band Energy Distribution', fontweight='bold')
        ax6.set_ylim(0, 1)
        for bar, energy in zip(bars, energies):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{energy:.3f}', ha='center', va='bottom', fontsize=10)

        # Directional energy pie chart
        ax7 = fig.add_subplot(gs[1, 2])
        dir_names = list(result['directional_energies'].keys())
        dir_values = [result['directional_energies'][d] for d in dir_names]
        colors_dir = plt.cm.Set2(np.linspace(0, 1, len(dir_names)))
        wedges, texts, autotexts = ax7.pie(dir_values, labels=dir_names, autopct='%1.1f%%',
                                           colors=colors_dir, startangle=90)
        ax7.set_title('(g) Directional Energy', fontweight='bold')

        # Summary statistics
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        stats_text = f"""
        Spectral Statistics
        ─────────────────────
        Spectral Entropy: {result['spectral_entropy']:.4f}
        Frequency Centroid: {result['frequency_centroid']:.4f}
        PSD Slope (β): {result['psd_slope']:.2f}
        Anisotropy Index: {result['anisotropy_index']:.2f}
        High/Low Ratio: {result['high_low_ratio']:.3f}

        Band Energies:
        • Low:  {result['band_energies']['low']:.4f}
        • Mid:  {result['band_energies']['mid']:.4f}
        • High: {result['band_energies']['high']:.4f}
        """
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax8.set_title('(h) Summary Statistics', fontweight='bold')

        # Row 3: Radial profile and frequency response
        ax9 = fig.add_subplot(gs[2, 0:2])
        # Compute radial profile
        radii = np.linspace(0, 1, 100)
        profile = []
        for r in radii:
            mask = (result['normalized_radius'] >= r - 0.01) & (result['normalized_radius'] < r + 0.01)
            if np.any(mask):
                profile.append(np.mean(result['magnitude_log'][mask]))
            else:
                profile.append(0)
        ax9.plot(radii, profile, 'b-', linewidth=2)
        ax9.fill_between(radii, profile, alpha=0.3)
        # Add band regions
        for band_name, (start, end) in self.frequency_bands.items():
            ax9.axvspan(start, end, alpha=0.2, color=self.band_colors[band_name], label=f'{band_name.capitalize()}')
        ax9.set_xlabel('Normalized Frequency')
        ax9.set_ylabel('Mean Log Magnitude')
        ax9.set_title('(i) Radial Frequency Profile', fontweight='bold')
        ax9.legend(loc='upper right')
        ax9.set_xlim(0, 1)

        # Reconstructed images from bands
        ax10 = fig.add_subplot(gs[2, 2])
        # Low frequency reconstruction
        f_shift = np.fft.fftshift(np.fft.fft2(result['gray']))
        low_mask = result['band_masks']['low']
        f_low = f_shift * low_mask
        img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(f_low)))
        ax10.imshow(img_low, cmap='gray')
        ax10.set_title('(j) Low Freq. Only', fontweight='bold')
        ax10.axis('off')

        ax11 = fig.add_subplot(gs[2, 3])
        # High frequency reconstruction
        high_mask = result['band_masks']['high']
        f_high = f_shift * high_mask
        img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(f_high)))
        img_high = (img_high - img_high.min()) / (img_high.max() - img_high.min() + 1e-10)
        ax11.imshow(img_high, cmap='gray')
        ax11.set_title('(k) High Freq. Only (edges)', fontweight='bold')
        ax11.axis('off')

        plt.suptitle(f'Fourier Frequency Analysis: {Path(image_path).name}', fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

        return result

    def visualize_tooltip_vs_background(self, image1_path, image2_path, output_path,
                                         label1="With Tooltip", label2="Background",
                                         show=False):
        """
        Create scientific comparison visualization for tooltip vs background frames.

        This is the main comparison method for paper figures showing:
        - Side-by-side FFT magnitude spectra
        - PSD slope comparison with linear fit
        - Band energy comparison bar chart
        - Statistical metrics comparison table
        - Radial profile overlay
        """
        # Load and process images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            print(f"Error loading images: {image1_path}, {image2_path}")
            return None

        result1 = self.compute_fft(image1)
        result2 = self.compute_fft(image2)

        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                     height_ratios=[1, 1, 1, 0.8])

        # Colors for the two classes
        color1 = '#e74c3c'  # Red for tooltip
        color2 = '#3498db'  # Blue for background

        # === Row 1: Original images and FFT magnitude ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image1_rgb)
        ax1.set_title(f'(a) {label1}', fontweight='bold', color=color1)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(result1['magnitude_log'], cmap='inferno')
        ax2.set_title(f'(b) FFT Magnitude - {label1}', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image2_rgb)
        ax3.set_title(f'(c) {label2}', fontweight='bold', color=color2)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(result2['magnitude_log'], cmap='inferno')
        ax4.set_title(f'(d) FFT Magnitude - {label2}', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # === Row 2: PSD slope and band energies ===
        # PSD slope comparison (log-log plot)
        ax5 = fig.add_subplot(gs[1, 0:2])
        log_r1 = np.log10(result1['psd_radii'])
        log_psd1 = np.log10(result1['psd_values'] + 1e-10)
        log_r2 = np.log10(result2['psd_radii'])
        log_psd2 = np.log10(result2['psd_values'] + 1e-10)

        ax5.scatter(log_r1, log_psd1, c=color1, alpha=0.6, s=30, label=f'{label1} (data)')
        ax5.scatter(log_r2, log_psd2, c=color2, alpha=0.6, s=30, label=f'{label2} (data)')

        # Linear fits
        fit_x = np.linspace(log_r1.min(), log_r1.max(), 100)
        fit_y1 = result1['psd_slope'] * fit_x + (log_psd1.mean() - result1['psd_slope'] * log_r1.mean())
        fit_y2 = result2['psd_slope'] * fit_x + (log_psd2.mean() - result2['psd_slope'] * log_r2.mean())

        ax5.plot(fit_x, fit_y1, c=color1, linewidth=2, linestyle='--',
                label=f'{label1}: β={result1["psd_slope"]:.2f}')
        ax5.plot(fit_x, fit_y2, c=color2, linewidth=2, linestyle='--',
                label=f'{label2}: β={result2["psd_slope"]:.2f}')

        ax5.set_xlabel('log₁₀(Normalized Frequency)', fontsize=12)
        ax5.set_ylabel('log₁₀(PSD)', fontsize=12)
        ax5.set_title('(e) Power Spectral Density - Log-Log Plot', fontweight='bold')
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3)

        # Annotate slope interpretation
        slope_diff = abs(result1['psd_slope'] - result2['psd_slope'])
        ax5.text(0.02, 0.02, f'Slope difference: Δβ = {slope_diff:.2f}\n'
                f'Natural images: β ≈ -2.0',
                transform=ax5.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Band energy comparison
        ax6 = fig.add_subplot(gs[1, 2:4])
        bands = ['Low\n(0-10%)', 'Mid\n(10-50%)', 'High\n(50-100%)']
        x = np.arange(len(bands))
        width = 0.35

        energies1 = [result1['band_energies']['low'],
                    result1['band_energies']['mid'],
                    result1['band_energies']['high']]
        energies2 = [result2['band_energies']['low'],
                    result2['band_energies']['mid'],
                    result2['band_energies']['high']]

        bars1 = ax6.bar(x - width/2, energies1, width, label=label1, color=color1, edgecolor='black')
        bars2 = ax6.bar(x + width/2, energies2, width, label=label2, color=color2, edgecolor='black')

        ax6.set_ylabel('Normalized Energy', fontsize=12)
        ax6.set_title('(f) Frequency Band Energy Distribution', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(bands)
        ax6.legend(loc='upper right')
        ax6.set_ylim(0, max(max(energies1), max(energies2)) * 1.2)

        # Add value labels
        for bar, val in zip(bars1, energies1):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=color1)
        for bar, val in zip(bars2, energies2):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=color2)

        # === Row 3: Radial profiles overlay and directional analysis ===
        ax7 = fig.add_subplot(gs[2, 0:2])
        # Compute radial profiles
        radii = np.linspace(0, 1, 100)
        profile1, profile2 = [], []
        for r in radii:
            mask1 = (result1['normalized_radius'] >= r - 0.01) & (result1['normalized_radius'] < r + 0.01)
            mask2 = (result2['normalized_radius'] >= r - 0.01) & (result2['normalized_radius'] < r + 0.01)
            profile1.append(np.mean(result1['magnitude_log'][mask1]) if np.any(mask1) else 0)
            profile2.append(np.mean(result2['magnitude_log'][mask2]) if np.any(mask2) else 0)

        ax7.plot(radii, profile1, color=color1, linewidth=2, label=label1)
        ax7.plot(radii, profile2, color=color2, linewidth=2, label=label2)
        ax7.fill_between(radii, profile1, profile2, alpha=0.3, color='gray')

        # Mark frequency bands
        for band_name, (start, end) in self.frequency_bands.items():
            ax7.axvspan(start, end, alpha=0.15, color=self.band_colors[band_name])

        ax7.set_xlabel('Normalized Frequency', fontsize=12)
        ax7.set_ylabel('Mean Log Magnitude', fontsize=12)
        ax7.set_title('(g) Radial Frequency Profile Comparison', fontweight='bold')
        ax7.legend(loc='upper right')
        ax7.set_xlim(0, 1)
        ax7.grid(True, alpha=0.3)

        # Directional analysis comparison
        ax8 = fig.add_subplot(gs[2, 2:4])
        dirs = ['Horizontal', 'Vertical', 'Diag 1', 'Diag 2']
        x_dirs = np.arange(len(dirs))

        dir_vals1 = [result1['directional_energies']['Horizontal'],
                    result1['directional_energies']['Vertical'],
                    result1['directional_energies']['Diagonal 1'],
                    result1['directional_energies']['Diagonal 2']]
        dir_vals2 = [result2['directional_energies']['Horizontal'],
                    result2['directional_energies']['Vertical'],
                    result2['directional_energies']['Diagonal 1'],
                    result2['directional_energies']['Diagonal 2']]

        bars1 = ax8.bar(x_dirs - width/2, dir_vals1, width, label=label1, color=color1, edgecolor='black')
        bars2 = ax8.bar(x_dirs + width/2, dir_vals2, width, label=label2, color=color2, edgecolor='black')

        ax8.set_ylabel('Normalized Energy', fontsize=12)
        ax8.set_title('(h) Directional Energy Distribution', fontweight='bold')
        ax8.set_xticks(x_dirs)
        ax8.set_xticklabels(dirs)
        ax8.legend(loc='upper right')

        # === Row 4: Statistics comparison table ===
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')

        # Create comparison table
        metrics = [
            ('Spectral Entropy (Hs)', result1['spectral_entropy'], result2['spectral_entropy']),
            ('Frequency Centroid (μf)', result1['frequency_centroid'], result2['frequency_centroid']),
            ('PSD Slope (β)', result1['psd_slope'], result2['psd_slope']),
            ('Anisotropy Index (AI)', result1['anisotropy_index'], result2['anisotropy_index']),
            ('High/Low Energy Ratio', result1['high_low_ratio'], result2['high_low_ratio']),
            ('High Frequency Energy', result1['band_energies']['high'], result2['band_energies']['high']),
        ]

        # Table header
        table_text = f"{'Metric':<28} │ {label1:^14} │ {label2:^14} │ {'Δ (Diff)':^12} │ Interpretation\n"
        table_text += "─" * 100 + "\n"

        interpretations = {
            'Spectral Entropy (Hs)': 'Higher = more complex texture',
            'Frequency Centroid (μf)': 'Higher = sharper edges',
            'PSD Slope (β)': 'Steeper = more low-freq dominant',
            'Anisotropy Index (AI)': 'Higher = more directional structure',
            'High/Low Energy Ratio': 'Higher = more edge detail',
            'High Frequency Energy': 'Higher = visible sharp objects',
        }

        for metric, val1, val2 in metrics:
            diff = val1 - val2
            sign = '+' if diff > 0 else ''
            interp = interpretations.get(metric, '')
            table_text += f"{metric:<28} │ {val1:^14.4f} │ {val2:^14.4f} │ {sign}{diff:^11.4f} │ {interp}\n"

        ax9.text(0.5, 0.5, table_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        ax9.set_title('(i) Statistical Comparison Summary', fontweight='bold', y=0.95)

        # Main title
        plt.suptitle(f'Fourier Spectral Analysis: {label1} vs {label2}',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison: {output_path}")

        if show:
            plt.show()
        plt.close()

        return {'image1': result1, 'image2': result2}

    def visualize_extremes(self, images_dir, output_dir, n_extremes=3,
                           metric='high_low', sample_size=200, show=False):
        """
        Visualize images with extreme (lowest vs highest) metric values.

        This shows the diversity in the dataset by comparing images at both
        ends of the spectral distribution.

        Args:
            images_dir: Directory with images
            output_dir: Output directory for visualizations
            n_extremes: Number of extreme images from each end (default: 3)
            metric: Which metric to use for sorting (default: 'high_low')
            sample_size: How many images to sample for analysis (default: 200)
            show: Whether to display plots interactively
        """
        import random
        from tqdm import tqdm

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Metric names for display
        metric_names = {
            'high_low': 'High/Low Ratio',
            'e_high': 'High Freq Energy',
            'psd_slope': 'PSD Slope (beta)',
            'anisotropy': 'Anisotropy Index',
            'entropy': 'Spectral Entropy',
            'centroid': 'Frequency Centroid'
        }

        if metric not in metric_names:
            print(f"Unknown metric: {metric}. Available: {list(metric_names.keys())}")
            return None

        # Get all images
        all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        if not all_images:
            print(f"No images found in {images_dir}")
            return None

        # Sample if too many
        if len(all_images) > sample_size:
            sample = random.sample(all_images, sample_size)
            print(f"Sampling {sample_size} images from {len(all_images)} total")
        else:
            sample = all_images
            print(f"Analyzing all {len(sample)} images")

        # Compute features for all sampled images
        print(f"\nComputing Fourier features for {len(sample)} images...")
        results = []
        for img_path in tqdm(sample, desc="Analyzing"):
            img = cv2.imread(str(img_path))
            if img is not None:
                feat = self.compute_fft(img)
                # Extract the metric value
                if metric == 'high_low':
                    value = feat['high_low_ratio']
                elif metric == 'e_high':
                    value = feat['band_energies']['high']
                elif metric == 'psd_slope':
                    value = feat['psd_slope']
                elif metric == 'anisotropy':
                    value = feat['anisotropy_index']
                elif metric == 'entropy':
                    value = feat['spectral_entropy']
                elif metric == 'centroid':
                    value = feat['frequency_centroid']

                results.append({
                    'path': img_path,
                    'value': value,
                    'features': feat
                })

        # Sort by metric value
        results.sort(key=lambda x: x['value'])

        # Get extremes
        lowest = results[:n_extremes]
        highest = results[-n_extremes:][::-1]  # Reverse to show highest first

        print(f"\n{metric_names[metric]} - Extremes:")
        low_vals_str = [f"{r['value']:.3f}" for r in lowest]
        high_vals_str = [f"{r['value']:.3f}" for r in highest]
        print(f"  LOWEST:  {low_vals_str}")
        print(f"  HIGHEST: {high_vals_str}")

        # === Create visualization ===
        fig = plt.figure(figsize=(6 * n_extremes, 16))
        gs = GridSpec(4, n_extremes, figure=fig, hspace=0.4, wspace=0.25,
                     height_ratios=[1, 1, 0.8, 0.5])

        # Colors
        color_low = '#3498db'   # Blue for low values
        color_high = '#e74c3c'  # Red for high values

        # Row 1: Lowest N images
        for i, res in enumerate(lowest):
            ax = fig.add_subplot(gs[0, i])
            img = cv2.imread(str(res['path']))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"LOW #{i+1}\n{metric_names[metric]}={res['value']:.3f}",
                        fontweight='bold', color=color_low, fontsize=11)
            ax.axis('off')
            # Add filename
            ax.text(0.5, -0.05, res['path'].stem[:25], transform=ax.transAxes,
                   ha='center', fontsize=8, color='gray')

        # Row 2: Highest N images
        for i, res in enumerate(highest):
            ax = fig.add_subplot(gs[1, i])
            img = cv2.imread(str(res['path']))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"HIGH #{i+1}\n{metric_names[metric]}={res['value']:.3f}",
                        fontweight='bold', color=color_high, fontsize=11)
            ax.axis('off')
            ax.text(0.5, -0.05, res['path'].stem[:25], transform=ax.transAxes,
                   ha='center', fontsize=8, color='gray')

        # Row 3: FFT magnitude comparison (first from each group)
        ax_fft_low = fig.add_subplot(gs[2, :n_extremes//2 + n_extremes%2])
        ax_fft_low.imshow(lowest[0]['features']['magnitude_log'], cmap='inferno')
        ax_fft_low.set_title(f"FFT Magnitude - LOWEST ({metric_names[metric]}={lowest[0]['value']:.3f})",
                            fontweight='bold', color=color_low)
        ax_fft_low.axis('off')

        ax_fft_high = fig.add_subplot(gs[2, n_extremes//2 + n_extremes%2:])
        ax_fft_high.imshow(highest[0]['features']['magnitude_log'], cmap='inferno')
        ax_fft_high.set_title(f"FFT Magnitude - HIGHEST ({metric_names[metric]}={highest[0]['value']:.3f})",
                             fontweight='bold', color=color_high)
        ax_fft_high.axis('off')

        # Row 4: Statistics comparison bar chart
        ax_stats = fig.add_subplot(gs[3, :])

        # Compute mean stats for each group
        metrics_to_show = ['high_low', 'e_high', 'psd_slope', 'anisotropy']
        x = np.arange(len(metrics_to_show))
        width = 0.35

        low_vals = []
        high_vals = []
        for m in metrics_to_show:
            if m == 'high_low':
                low_vals.append(np.mean([r['features']['high_low_ratio'] for r in lowest]))
                high_vals.append(np.mean([r['features']['high_low_ratio'] for r in highest]))
            elif m == 'e_high':
                low_vals.append(np.mean([r['features']['band_energies']['high'] for r in lowest]))
                high_vals.append(np.mean([r['features']['band_energies']['high'] for r in highest]))
            elif m == 'psd_slope':
                low_vals.append(np.mean([abs(r['features']['psd_slope']) for r in lowest]))
                high_vals.append(np.mean([abs(r['features']['psd_slope']) for r in highest]))
            elif m == 'anisotropy':
                low_vals.append(np.mean([r['features']['anisotropy_index'] for r in lowest]))
                high_vals.append(np.mean([r['features']['anisotropy_index'] for r in highest]))

        # Normalize for visualization
        max_vals = [max(l, h) for l, h in zip(low_vals, high_vals)]
        low_norm = [l/m if m > 0 else 0 for l, m in zip(low_vals, max_vals)]
        high_norm = [h/m if m > 0 else 0 for h, m in zip(high_vals, max_vals)]

        bars1 = ax_stats.bar(x - width/2, low_norm, width, label=f'LOW {metric_names[metric]}',
                            color=color_low, edgecolor='black')
        bars2 = ax_stats.bar(x + width/2, high_norm, width, label=f'HIGH {metric_names[metric]}',
                            color=color_high, edgecolor='black')

        ax_stats.set_ylabel('Normalized Value')
        ax_stats.set_title('Mean Spectral Features Comparison (Normalized)', fontweight='bold')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(['H/L Ratio', 'E_high', '|PSD Slope|', 'Anisotropy'])
        ax_stats.legend(loc='upper right')
        ax_stats.set_ylim(0, 1.3)

        # Add actual values as text
        for i, (l, h) in enumerate(zip(low_vals, high_vals)):
            ax_stats.text(x[i] - width/2, low_norm[i] + 0.05, f'{l:.2f}',
                         ha='center', va='bottom', fontsize=9, color=color_low)
            ax_stats.text(x[i] + width/2, high_norm[i] + 0.05, f'{h:.2f}',
                         ha='center', va='bottom', fontsize=9, color=color_high)

        # Main title
        plt.suptitle(f'Dataset Diversity Analysis: {metric_names[metric]} Extremes\n'
                    f'(N={n_extremes} from each end, sampled {len(sample)} images)',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save
        output_path = output_dir / f"extremes_{metric}_n{n_extremes}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {output_path}")

        if show:
            plt.show()
        plt.close()

        # Also save individual comparisons for the most extreme pair
        print("\nGenerating detailed comparison of most extreme pair...")
        detail_path = output_dir / f"extremes_{metric}_detailed_comparison.png"
        self.visualize_tooltip_vs_background(
            str(highest[0]['path']),
            str(lowest[0]['path']),
            str(detail_path),
            label1=f"HIGH {metric_names[metric]} ({highest[0]['value']:.2f})",
            label2=f"LOW {metric_names[metric]} ({lowest[0]['value']:.2f})",
            show=show
        )

        return {
            'lowest': lowest,
            'highest': highest,
            'metric': metric,
            'metric_name': metric_names[metric]
        }

    def visualize_comparison(self, image_paths, output_path, show=False):
        """Compare Fourier features across multiple images."""
        n_images = len(image_paths)

        # Compute features for all images
        results = []
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is not None:
                results.append({
                    'path': img_path,
                    'features': self.compute_fft(image)
                })

        # Create comparison figure
        fig, axes = plt.subplots(n_images, 5, figsize=(20, 4 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)

        for idx, res in enumerate(results):
            img_path = res['path']
            feat = res['features']
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Original
            axes[idx, 0].imshow(image_rgb)
            axes[idx, 0].set_title(Path(img_path).stem[:20], fontsize=10)
            axes[idx, 0].axis('off')

            # FFT Magnitude
            axes[idx, 1].imshow(feat['magnitude_log'], cmap='inferno')
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title('FFT Magnitude', fontweight='bold')

            # Band energy bars
            bands = list(feat['band_energies'].keys())
            energies = [feat['band_energies'][b] for b in bands]
            colors = [self.band_colors[b] for b in bands]
            axes[idx, 2].bar(bands, energies, color=colors, edgecolor='black')
            axes[idx, 2].set_ylim(0, 1)
            if idx == 0:
                axes[idx, 2].set_title('Band Energies', fontweight='bold')

            # Entropy indicator
            entropy = feat['spectral_entropy']
            centroid = feat['frequency_centroid']
            axes[idx, 3].barh(['Entropy', 'Centroid'], [entropy, centroid], color=['steelblue', 'coral'])
            axes[idx, 3].set_xlim(0, 1)
            if idx == 0:
                axes[idx, 3].set_title('Spectral Features', fontweight='bold')

            # Directional
            dir_names = list(feat['directional_energies'].keys())
            dir_values = [feat['directional_energies'][d] for d in dir_names]
            axes[idx, 4].pie(dir_values, labels=['H', 'V', 'D1', 'D2'], autopct='%1.0f%%', startangle=90)
            if idx == 0:
                axes[idx, 4].set_title('Directional', fontweight='bold')

        plt.suptitle('Fourier Feature Comparison Across Images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fourier Frequency Analysis Visualization for Scientific Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First 5 images (sequential)
  py -3.11 visualize_fourier_spectrum.py --num_images 5

  # Random 5 images from dataset
  py -3.11 visualize_fourier_spectrum.py --num_images 5 --random

  # Specific images by filename (searches in --images_dir)
  py -3.11 visualize_fourier_spectrum.py --images frame_001.jpg frame_002.jpg

  # Specific images with full paths
  py -3.11 visualize_fourier_spectrum.py --images "F:/path/image1.jpg" "F:/path/image2.jpg"

  # Show plots interactively
  py -3.11 visualize_fourier_spectrum.py --num_images 2 --show
        """
    )

    # Image selection options
    selection_group = parser.add_argument_group('Image Selection')
    selection_group.add_argument('--images_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images',
                        help='Directory with images (default: selected_dataset/images)')
    selection_group.add_argument('--images', nargs='+', type=str, default=None,
                        help='Specific image files (filenames or full paths). Overrides --num_images')
    selection_group.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize (default: 5)')
    selection_group.add_argument('--random', action='store_true',
                        help='Randomly select images instead of first N sequential')
    selection_group.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible selection (optional)')

    # Scientific comparison options
    compare_group = parser.add_argument_group('Scientific Comparison (--compare)')
    compare_group.add_argument('--compare', nargs=2, type=str, default=None,
                        metavar=('IMAGE1', 'IMAGE2'),
                        help='Compare two images (e.g., tooltip vs background). '
                             'Generates comprehensive statistical comparison.')
    compare_group.add_argument('--labels', nargs=2, type=str,
                        default=['With Tooltip', 'Background'],
                        metavar=('LABEL1', 'LABEL2'),
                        help='Labels for the two compared images (default: "With Tooltip" "Background")')

    # Extremes comparison options
    extremes_group = parser.add_argument_group('Extremes Analysis (--extremes)')
    extremes_group.add_argument('--extremes', type=int, default=None, metavar='N',
                        help='Find and visualize N images with lowest vs highest metric values. '
                             'Shows dataset diversity.')
    extremes_group.add_argument('--metric', type=str, default='high_low',
                        choices=['high_low', 'e_high', 'psd_slope', 'anisotropy', 'entropy', 'centroid'],
                        help='Metric to use for extremes analysis (default: high_low)')
    extremes_group.add_argument('--sample_size', type=int, default=200,
                        help='Number of images to sample for extremes analysis (default: 200)')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\output\fourier',
                        help='Output directory for visualizations')
    output_group.add_argument('--show', action='store_true',
                        help='Show plots interactively (default: save only)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = FourierVisualizer()

    # === MODE 1: Scientific comparison of two images ===
    if args.compare:
        print("\n=== Scientific Comparison Mode ===")
        images_dir = Path(args.images_dir)

        # Resolve image paths
        compare_paths = []
        for img_name in args.compare:
            img_path = Path(img_name)
            if img_path.is_absolute() and img_path.exists():
                compare_paths.append(img_path)
            else:
                # Try to find in images_dir
                full_path = images_dir / img_name
                if full_path.exists():
                    compare_paths.append(full_path)
                else:
                    # Try glob search
                    matches = list(images_dir.glob(f"*{img_name}*"))
                    if matches:
                        compare_paths.append(matches[0])
                        print(f"Found match: {img_name} -> {matches[0].name}")
                    else:
                        print(f"ERROR: Image not found: {img_name}")
                        return

        if len(compare_paths) != 2:
            print("ERROR: Need exactly 2 images for comparison")
            return

        print(f"\nComparing:")
        print(f"  Image 1 ({args.labels[0]}): {compare_paths[0].name}")
        print(f"  Image 2 ({args.labels[1]}): {compare_paths[1].name}")

        # Generate comparison
        output_path = output_dir / f"fourier_comparison_{compare_paths[0].stem}_vs_{compare_paths[1].stem}.png"
        visualizer.visualize_tooltip_vs_background(
            str(compare_paths[0]),
            str(compare_paths[1]),
            str(output_path),
            label1=args.labels[0],
            label2=args.labels[1],
            show=args.show
        )

        print(f"\n=== Done! Comparison saved to: {output_path} ===")
        return

    # === MODE 2: Extremes analysis ===
    if args.extremes:
        print("\n=== Extremes Analysis Mode ===")
        print(f"Finding {args.extremes} images with LOWEST and HIGHEST {args.metric}")

        visualizer.visualize_extremes(
            images_dir=args.images_dir,
            output_dir=output_dir,
            n_extremes=args.extremes,
            metric=args.metric,
            sample_size=args.sample_size,
            show=args.show
        )

        print(f"\n=== Done! Extremes analysis saved to: {output_dir} ===")
        return

    # === MODE 3: Standard visualization ===
    # Get image files based on selection mode
    images_dir = Path(args.images_dir)

    if args.images:
        # Mode 1: Specific images provided
        image_files = []
        for img_name in args.images:
            img_path = Path(img_name)
            if img_path.is_absolute() and img_path.exists():
                image_files.append(img_path)
            else:
                full_path = images_dir / img_name
                if full_path.exists():
                    image_files.append(full_path)
                else:
                    matches = list(images_dir.glob(f"*{img_name}*"))
                    if matches:
                        image_files.append(matches[0])
                        print(f"Found match: {img_name} -> {matches[0].name}")
                    else:
                        print(f"Warning: Image not found: {img_name}")
        print(f"Selected {len(image_files)} specific images")
    else:
        # Mode 2: Select from directory
        all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if not all_images:
            print(f"No images found in {images_dir}")
            return

        if args.random:
            image_files = random.sample(all_images, min(args.num_images, len(all_images)))
            print(f"Randomly selected {len(image_files)} images from {len(all_images)} total")
        else:
            image_files = all_images[:args.num_images]
            print(f"Selected first {len(image_files)} images from {len(all_images)} total")

    if not image_files:
        print("No images to process!")
        return

    print(f"\nImages to process:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")

    # Generate individual visualizations
    print("\n=== Generating individual Fourier visualizations ===")
    for img_path in tqdm(image_files, desc="Processing"):
        output_path = output_dir / f"{img_path.stem}_fourier_analysis.png"
        visualizer.visualize_single_image(str(img_path), str(output_path), show=args.show)

    # Generate comparison
    print("\n=== Generating comparison grid ===")
    comparison_path = output_dir / "fourier_comparison.png"
    visualizer.visualize_comparison(
        [str(p) for p in image_files[:min(5, len(image_files))]],
        str(comparison_path),
        show=args.show
    )

    print(f"\n=== Done! Visualizations saved to: {output_dir} ===")


if __name__ == '__main__':
    main()
