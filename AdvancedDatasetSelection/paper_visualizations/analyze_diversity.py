#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze which Fourier metric best differentiates images in dataset."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, '.')
from visualize_fourier_spectrum import FourierVisualizer
import cv2
import numpy as np
from pathlib import Path
import random
import statistics

random.seed(42)
viz = FourierVisualizer()

images_dir = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images')
all_images = list(images_dir.glob('*.jpg'))
sample = random.sample(all_images, min(100, len(all_images)))

print(f"Analyzing {len(sample)} random images...")

results = []
for i, img_path in enumerate(sample):
    img = cv2.imread(str(img_path))
    if img is not None:
        feat = viz.compute_fft(img)
        results.append({
            'name': img_path.stem[:30],
            'entropy': feat['spectral_entropy'],
            'centroid': feat['frequency_centroid'],
            'psd_slope': feat['psd_slope'],
            'anisotropy': feat['anisotropy_index'],
            'high_low': feat['high_low_ratio'],
            'e_high': feat['band_energies']['high'],
            'e_mid': feat['band_energies']['mid'],
            'e_low': feat['band_energies']['low']
        })
    if (i+1) % 20 == 0:
        print(f"  Processed {i+1}/{len(sample)}")

# Statistics
print()
print('='*90)
print('ANALYSIS OF 100 RANDOM IMAGES FROM 20K DATASET - SPECTRAL DIVERSITY')
print('='*90)

metrics = ['entropy', 'centroid', 'psd_slope', 'anisotropy', 'high_low', 'e_high']
names = {
    'entropy': 'Spectral Entropy (Hs)',
    'centroid': 'Frequency Centroid (uf)',
    'psd_slope': 'PSD Slope (beta)',
    'anisotropy': 'Anisotropy Index (AI)',
    'high_low': 'High/Low Ratio',
    'e_high': 'High Freq Energy (E_high)'
}

print(f"\n{'Metryka':<28} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8} | {'Range':>8} | CV%")
print('-'*90)

best_metric = None
best_cv = 0
metric_stats = {}

for m in metrics:
    vals = [r[m] for r in results]
    mn, mx = min(vals), max(vals)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    rng = mx - mn
    cv = (std / abs(mean) * 100) if mean != 0 else 0

    metric_stats[m] = {'cv': cv, 'range': rng, 'std': std}

    if cv > best_cv:
        best_cv = cv
        best_metric = m

    print(f"{names[m]:<28} | {mn:>8.4f} | {mx:>8.4f} | {mean:>8.4f} | {std:>8.4f} | {rng:>8.4f} | {cv:>5.1f}%")

print()
print('='*90)
print(f'★ NAJLEPSZA METRYKA DO RÓŻNICOWANIA: {names[best_metric]}')
print(f'  Coefficient of Variation (CV) = {best_cv:.1f}%')
print('='*90)

# Ranking
print('\nRANKING METRYK (według CV - im wyższy, tym lepiej różnicuje):')
ranked = sorted(metric_stats.items(), key=lambda x: x[1]['cv'], reverse=True)
for i, (m, stats) in enumerate(ranked, 1):
    print(f"  {i}. {names[m]:<28} CV={stats['cv']:.1f}%")

# Show extremes for best metric
vals = [(r['name'], r[best_metric]) for r in results]
vals.sort(key=lambda x: x[1])

print(f"\n5 images with LOWEST {names[best_metric]}:")
for name, val in vals[:5]:
    print(f'  {val:.4f} - {name}')

print(f"\n5 images with HIGHEST {names[best_metric]}:")
for name, val in vals[-5:]:
    print(f'  {val:.4f} - {name}')

# Interpretation
print()
print('='*90)
print('INTERPRETATION FOR DETR TRAINING:')
print('='*90)
print("""
Metric with HIGH CV means images in dataset are DIVERSE in that aspect.
This is BENEFICIAL for training - model sees different cases.

* High/Low Ratio & E_high:
  - High = sharp edges (tooltip visible)
  - Low = soft background (only anatomy)
  -> Good dataset has WIDE RANGE of these values

* Anisotropy Index:
  - High (>1.5) = linear structures (surgical instruments)
  - Low (~1.0) = isotropic (round anatomical structures)
  -> Diversity = model learns both types

* Spectral Entropy:
  - High = complex texture (hard example)
  - Low = simple image (easy example)
  -> Mix of hard and easy = curriculum learning

BEST METRIC FOR DATASET DIVERSITY: The one with highest CV%
""")
