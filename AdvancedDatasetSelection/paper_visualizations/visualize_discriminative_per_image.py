#!/usr/bin/env python3
"""
Per-image visualization of discriminative Fourier features.

Builds on:
- discriminative_features.json (AUC, Cohen's d, class means)
- compute_advanced_fft_features from visualize_discriminative_features.py

Modes (similar to visualize_fourier_spectrum.py):
    # Single images (auto-pick first N from tooltip set)
    py -3.11 visualize_discriminative_per_image.py --num_images 3

    # Random sample
    py -3.11 visualize_discriminative_per_image.py --num_images 3 --random

    # Explicit images
    py -3.11 visualize_discriminative_per_image.py --images img1.jpg img2.jpg

    # Compare two images
    py -3.11 visualize_discriminative_per_image.py --compare img1.jpg img2.jpg --labels "Tooltip" "Background"

    # Help
    py -3.11 visualize_discriminative_per_image.py --help
"""

import argparse
import random
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Reuse feature computation and metadata
from visualize_discriminative_features import (
    compute_advanced_fft_features,
    FEATURE_INFO,
    DEFAULT_TOOLTIP_DIR,
    DEFAULT_BG_TRAIN,
    DEFAULT_BG_VAL,
    DEFAULT_FEATURES_JSON,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output" / "discriminative_per_image"

# Fallback order if JSON is missing
FALLBACK_FEATURES = [
    "spectral_spread",
    "e_diagonal1",
    "spectral_entropy",
    "edge_sharpness",
    "psd_slope",
    "ring_6",
    "ring_5",
    "high_low_ratio",
    "e_high",
    "spectral_skewness",
]


class PerImageDiscriminativeVisualizer:
    """Visualize discriminative features for individual images."""

    def __init__(self, features_json):
        self.features_json = Path(features_json)
        self.rankings = self._load_rankings()
        self.feature_stats = {r.get("feature"): r for r in self.rankings}
        self.top_features = self._top_features()

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14

        self.color_img1 = "#e74c3c"
        self.color_img2 = "#3498db"

    def _load_rankings(self):
        if not self.features_json.exists():
            print(f"Warning: features JSON not found, using fallback list: {self.features_json}")
            return []
        try:
            with open(self.features_json, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as exc:
            print(f"Warning: could not read {self.features_json}: {exc}")
        return []

    def _top_features(self, n=10):
        if self.rankings:
            ranked = sorted(self.rankings, key=lambda x: x.get("auc", 0), reverse=True)
            return [r.get("feature") for r in ranked[:n] if r.get("feature")]
        return FALLBACK_FEATURES[:n]

    def _format_val(self, val):
        if val is None:
            return "n/a"
        if abs(val) < 1e-3 or abs(val) >= 1e3:
            return f"{val:.3e}"
        return f"{val:.4f}"

    def _feature_label(self, feature):
        return FEATURE_INFO.get(feature, {}).get("name", feature)

    def _table_rows_single(self, feats):
        rows = []
        for feat in self.top_features:
            v = feats.get(feat, 0.0)
            stats = self.feature_stats.get(feat, {})
            rows.append([
                self._feature_label(feat),
                self._format_val(v),
                self._format_val(stats.get("tooltip_mean")),
                self._format_val(stats.get("bg_mean")),
                f"{stats.get('cohens_d', 0):+.3f}" if stats else "n/a",
                f"{stats.get('auc', 0):.3f}" if stats else "n/a",
            ])
        return rows

    def _table_rows_compare(self, f1, f2):
        rows = []
        for feat in self.top_features:
            v1 = f1.get(feat, 0.0)
            v2 = f2.get(feat, 0.0)
            stats = self.feature_stats.get(feat, {})
            rows.append([
                self._feature_label(feat),
                self._format_val(v1),
                self._format_val(v2),
                self._format_val(v1 - v2),
                f"{stats.get('cohens_d', 0):+.3f}" if stats else "n/a",
                f"{stats.get('auc', 0):.3f}" if stats else "n/a",
            ])
        return rows

    def _plot_ring_energy(self, ax, feats):
        rings = [f"ring_{i}" for i in range(10)]
        ring_vals = [feats.get(r, 0.0) for r in rings]
        x = np.arange(len(rings))
        ax.bar(x, ring_vals, color=self.color_img1, edgecolor="black", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i}" for i in range(10)])
        ax.set_yscale("log")
        ax.set_xlabel("Frequency Ring")
        ax.set_ylabel("Energy (log)")
        ax.set_title("Ring energy distribution")

    def _plot_ring_energy_compare(self, ax, f1, f2, label1, label2):
        rings = [f"ring_{i}" for i in range(10)]
        v1 = [f1.get(r, 0.0) for r in rings]
        v2 = [f2.get(r, 0.0) for r in rings]
        x = np.arange(len(rings))
        width = 0.4
        ax.bar(x - width / 2, v1, width, color=self.color_img1, edgecolor="black", alpha=0.8, label=label1)
        ax.bar(x + width / 2, v2, width, color=self.color_img2, edgecolor="black", alpha=0.8, label=label2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i}" for i in range(10)])
        ax.set_yscale("log")
        ax.set_xlabel("Frequency Ring")
        ax.set_ylabel("Energy (log)")
        ax.set_title("Ring energy distribution")
        ax.legend()

    def _plot_top_features_compare(self, ax, f1, f2, label1, label2):
        feats = self.top_features
        vals1 = [f1.get(k, 0.0) for k in feats]
        vals2 = [f2.get(k, 0.0) for k in feats]

        # Normalize per feature for readability
        v1n, v2n = [], []
        for a, b in zip(vals1, vals2):
            lo, hi = min(a, b), max(a, b)
            rng = hi - lo if hi != lo else 1.0
            v1n.append((a - lo) / rng)
            v2n.append((b - lo) / rng)

        y = np.arange(len(feats))
        ax.barh(y - 0.2, v1n, 0.35, color=self.color_img1, edgecolor="black", alpha=0.85, label=label1)
        ax.barh(y + 0.2, v2n, 0.35, color=self.color_img2, edgecolor="black", alpha=0.85, label=label2)
        ax.set_yticks(y)
        ax.set_yticklabels([self._feature_label(f) for f in feats])
        ax.set_xlabel("Normalized value (per feature)")
        ax.set_title("Top discriminative features")

        # annotate actual values - two columns to avoid overlap
        for i, (a, b) in enumerate(zip(vals1, vals2)):
            # Format values more compactly
            val_a = self._format_val(a)
            val_b = self._format_val(b)
            # Red value at x=1.02, Blue value at x=1.15 (separate columns)
            ax.text(1.02, i, val_a, va="center", ha="left", color=self.color_img1, fontsize=7, fontweight="bold")
            ax.text(1.18, i, val_b, va="center", ha="left", color=self.color_img2, fontsize=7, fontweight="bold")

        ax.set_xlim(0, 1.35)
        ax.set_ylim(-0.5, len(feats) + 0.2)
        # Add column headers for values (shortened labels)
        short1 = label1[:10] + ".." if len(label1) > 10 else label1
        short2 = label2[:10] + ".." if len(label2) > 10 else label2
        ax.text(1.02, len(feats) - 0.6, short1, va="top", ha="left", color=self.color_img1, fontsize=7, fontweight="bold")
        ax.text(1.18, len(feats) - 0.6, short2, va="top", ha="left", color=self.color_img2, fontsize=7, fontweight="bold")
        ax.legend(loc="lower right")

    def visualize_single(self, image_path, output_path, label="Image", show=False):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: could not read image {image_path}")
            return

        feats = compute_advanced_fft_features(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25, height_ratios=[1, 1])

        # Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb)
        ax1.set_title(f"{label}", fontweight="bold")
        ax1.axis("off")

        # FFT magnitude
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(feats["magnitude_log"], cmap="inferno")
        ax2.set_title("FFT magnitude (log)", fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Ring energies
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_ring_energy(ax3, feats)

        # Table of top features
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        table = ax4.table(
            cellText=self._table_rows_single(feats),
            colLabels=["Feature", "Value", "Tooltip mean", "Background mean", "Cohen's d", "AUC"],
            loc="center",
            cellLoc="center",
            colWidths=[0.28, 0.12, 0.12, 0.14, 0.1, 0.08],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.4)
        ax4.set_title("Top discriminative features (per image vs dataset stats)", fontweight="bold")

        plt.suptitle(f"Discriminative Fourier features – {label}", fontsize=15, fontweight="bold")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close()

    def visualize_compare(self, img1_path, img2_path, output_path, label1="Image 1", label2="Image 2", show=False):
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        if img1 is None or img2 is None:
            print(f"Error: could not read images {img1_path} or {img2_path}")
            return

        f1 = compute_advanced_fft_features(img1)
        f2 = compute_advanced_fft_features(img2)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        # Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1_rgb)
        ax1.set_title(label1, fontweight="bold", color=self.color_img1)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2_rgb)
        ax2.set_title(label2, fontweight="bold", color=self.color_img2)
        ax2.axis("off")

        # Top features
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_top_features_compare(ax3, f1, f2, label1, label2)

        # Ring energies
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_ring_energy_compare(ax4, f1, f2, label1, label2)

        # Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")
        table = ax5.table(
            cellText=self._table_rows_compare(f1, f2),
            colLabels=["Feature", label1, label2, "Diff", "Cohen's d", "AUC"],
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.12, 0.12, 0.12, 0.1, 0.08],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.5)
        ax5.set_title("Per-image comparison vs dataset rankings", fontweight="bold")

        plt.suptitle("Discriminative Fourier features: per-image comparison", fontsize=15, fontweight="bold")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close()

    def visualize_compare_multi(self, img1_path, img2_path, output_dir, label1="Image 1", label2="Image 2", show=False):
        """Generate multiple figures in a folder for better visualization of different scales."""
        import os
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        if img1 is None or img2 is None:
            print(f"Error: could not read images {img1_path} or {img2_path}")
            return

        f1 = compute_advanced_fft_features(img1)
        f2 = compute_advanced_fft_features(img2)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # === 1. Images side by side ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img1_rgb)
        axes[0].set_title(label1, fontweight="bold", color=self.color_img1, fontsize=14)
        axes[0].axis("off")
        axes[1].imshow(img2_rgb)
        axes[1].set_title(label2, fontweight="bold", color=self.color_img2, fontsize=14)
        axes[1].axis("off")
        plt.suptitle("Image Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "01_images.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_dir / '01_images.png'}")
        plt.close()

        # === 2. Radar chart (handles scale differences!) ===
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        feats = self.top_features
        vals1 = [f1.get(k, 0.0) for k in feats]
        vals2 = [f2.get(k, 0.0) for k in feats]

        # Normalize using dataset statistics (tooltip_mean, bg_mean) for proper range
        # This way both values are positioned relative to the full dataset distribution
        v1n, v2n = [], []
        for i, feat in enumerate(feats):
            a, b = vals1[i], vals2[i]
            stats = self.feature_stats.get(feat, {})
            t_mean = stats.get("tooltip_mean", a)
            bg_mean = stats.get("bg_mean", b)

            # Use dataset range: from min(tooltip_mean, bg_mean) to max with 20% padding
            all_vals = [a, b, t_mean, bg_mean]
            lo = min(all_vals)
            hi = max(all_vals)
            # Add padding so points don't sit exactly at 0 or 1
            padding = (hi - lo) * 0.15 if hi != lo else abs(lo) * 0.15 if lo != 0 else 0.1
            lo -= padding
            hi += padding
            rng = hi - lo if hi != lo else 1.0

            v1n.append((a - lo) / rng)
            v2n.append((b - lo) / rng)

        angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
        v1n += v1n[:1]  # close the polygon
        v2n += v2n[:1]
        angles += angles[:1]

        ax.plot(angles, v1n, 'o-', linewidth=2.5, color=self.color_img1, label=label1, markersize=8)
        ax.fill(angles, v1n, alpha=0.25, color=self.color_img1)
        ax.plot(angles, v2n, 'o-', linewidth=2.5, color=self.color_img2, label=label2, markersize=8)
        ax.fill(angles, v2n, alpha=0.25, color=self.color_img2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._feature_label(f) for f in feats], fontsize=9)
        ax.set_title("Radar Chart: Normalized Feature Comparison", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(output_dir / "02_radar.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_dir / '02_radar.png'}")
        plt.close()

        # === 3. Individual feature comparisons (small multiples) ===
        n_feats = len(feats)
        n_cols = 5
        n_rows = (n_feats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i, feat in enumerate(feats):
            ax = axes[i]
            v1 = vals1[i]
            v2 = vals2[i]

            x = [0, 1]
            colors = [self.color_img1, self.color_img2]
            bars = ax.bar(x, [v1, v2], color=colors, edgecolor="black", alpha=0.8, width=0.6)

            # Add value labels on bars
            for bar, val in zip(bars, [v1, v2]):
                height = bar.get_height()
                va = "bottom" if height >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width()/2, height, self._format_val(val),
                       ha="center", va=va, fontsize=8, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([label1, label2], fontsize=8)
            ax.set_title(self._feature_label(feat), fontsize=10, fontweight="bold")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

            # Use log scale if values span orders of magnitude and are positive
            if v1 > 0 and v2 > 0 and max(v1, v2) / min(v1, v2) > 100:
                ax.set_yscale("log")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Individual Feature Comparisons", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "03_individual_features.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_dir / '03_individual_features.png'}")
        plt.close()

        # === 4. Ring energy comparison ===
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_ring_energy_compare(ax, f1, f2, label1, label2)
        ax.set_title("Ring Energy Distribution (Frequency Bands)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "04_ring_energy.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_dir / '04_ring_energy.png'}")
        plt.close()

        # === 5. Summary table ===
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        table = ax.table(
            cellText=self._table_rows_compare(f1, f2),
            colLabels=["Feature", label1, label2, "Difference", "Cohen's d", "AUC"],
            loc="center",
            cellLoc="center",
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.12, 0.10],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Color header row
        for i in range(6):
            table[(0, i)].set_facecolor("#4a90d9")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        ax.set_title("Detailed Feature Comparison Table", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "05_summary_table.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_dir / '05_summary_table.png'}")
        plt.close()

        print(f"\n=== Generated {5} figures in: {output_dir} ===")


def resolve_image(path_str, tooltip_dir, bg_dirs):
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    # Try tooltip then background dirs
    candidate = Path(tooltip_dir) / path_str
    if candidate.exists():
        return candidate
    for bg in bg_dirs:
        candidate = Path(bg) / path_str
        if candidate.exists():
            return candidate
    return path


def collect_images(dirs, exts):
    """Collect images recursively from list of dirs for given extensions."""
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    exts = [e.lower().lstrip(".") for e in exts]
    files = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for ext in exts:
            files.extend(d.rglob(f"*.{ext}"))
    return sorted({p for p in files if p.is_file()})


def main():
    parser = argparse.ArgumentParser(
        description="Per-image discriminative Fourier feature visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    mode = parser.add_argument_group("Mode")
    mode.add_argument("--images", nargs="+", help="Explicit image paths or names (tooltip/bg dirs searched)")
    mode.add_argument("--num_images", type=int, default=3, help="Number of images when sampling tooltip dir")
    mode.add_argument("--random", action="store_true", help="Sample images randomly from tooltip dir")
    mode.add_argument("--compare", nargs=2, help="Compare two images")
    mode.add_argument("--random_pair", action="store_true", help="Pick one random tooltip and one random background image to compare")
    mode.add_argument("--pair_count", type=int, default=1, help="Number of random pairs to generate (used with --random_pair)")
    mode.add_argument("--labels", nargs=2, default=["Image 1", "Image 2"], help="Labels for compare mode")
    mode.add_argument("--multi", action="store_true", help="Generate multiple figures in a folder (better for different scales)")

    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--tooltip_dir", type=str, default=str(DEFAULT_TOOLTIP_DIR), help="Tooltip images directory")
    paths.add_argument("--bg_train", type=str, default=str(DEFAULT_BG_TRAIN), help="Background train directory")
    paths.add_argument("--bg_val", type=str, default=str(DEFAULT_BG_VAL), help="Background val directory")
    paths.add_argument("--exts", type=str, default="jpg,png", help="Comma-separated image extensions to search (recursive)")
    paths.add_argument("--features_json", type=str, default=str(DEFAULT_FEATURES_JSON), help="Path to features JSON")
    paths.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    # Misc
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = PerImageDiscriminativeVisualizer(args.features_json)
    tooltip_dir = Path(args.tooltip_dir)
    bg_dirs = [Path(args.bg_train), Path(args.bg_val)]
    extensions = [e.strip() for e in args.exts.split(",") if e.strip()]

    # Random pair mode
    if args.random_pair:
        tooltip_images = collect_images(tooltip_dir, extensions)
        bg_images = collect_images(bg_dirs, extensions)
        if not tooltip_images or not bg_images:
            print("No images found for random pair. Check tooltip_dir and bg directories.")
            return
        count = max(1, args.pair_count)
        for _ in range(count):
            img1 = random.choice(tooltip_images)
            img2 = random.choice(bg_images)
            out = output_dir / f"discriminative_compare_random_{img1.stem}_vs_{img2.stem}.png"
            visualizer.visualize_compare(img1, img2, out, label1=args.labels[0], label2=args.labels[1], show=args.show)
        return

    # Compare mode (explicit)
    if args.compare:
        img1 = resolve_image(args.compare[0], tooltip_dir, bg_dirs)
        img2 = resolve_image(args.compare[1], tooltip_dir, bg_dirs)
        if args.multi:
            # Multi-figure output to subfolder
            multi_out = output_dir / f"compare_{img1.stem}_vs_{img2.stem}"
            visualizer.visualize_compare_multi(img1, img2, multi_out, label1=args.labels[0], label2=args.labels[1], show=args.show)
        else:
            out = output_dir / f"discriminative_compare_{img1.stem}_vs_{img2.stem}.png"
            visualizer.visualize_compare(img1, img2, out, label1=args.labels[0], label2=args.labels[1], show=args.show)
        return

    # Single-image mode
    images = []
    if args.images:
        images = [resolve_image(p, tooltip_dir, bg_dirs) for p in args.images]
    else:
        candidates = collect_images(tooltip_dir, extensions)
        if args.random:
            images = random.sample(candidates, min(args.num_images, len(candidates)))
        else:
            images = candidates[: args.num_images]

    if not images:
        print("No images found. Provide --images or ensure tooltip_dir has *.jpg files.")
        return

    for img_path in images:
        out = output_dir / f"discriminative_single_{img_path.stem}.png"
        visualizer.visualize_single(img_path, out, label=img_path.name, show=args.show)


if __name__ == "__main__":
    main()
