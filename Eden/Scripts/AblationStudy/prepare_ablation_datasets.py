#!/usr/bin/env python3
"""
Prepare 5 ablation dataset variants for DETR training.

Wraps the existing generate_video_split.py and generate_ablation_datasets.py
from AdvancedDatasetSelection/ablation/ with sensible defaults and validation.

Usage (from project root):
  python Eden/Scripts/AblationStudy/prepare_ablation_datasets.py \
    --images-dir /path/to/DETR_augmented_dataset/images \
    --annotations /path/to/augmented_coco_20250417_030014.json \
    --cache AdvancedDatasetSelection/output/feature_cache2_weighted \
    --output Eden/Scripts/AblationStudy/output

Steps:
  1. Generate video_split.json (deterministic, seed=42)
  2. Generate 5 variants (v1-v5) using feature cache
  3. Validate output: check image counts, annotation integrity
  4. Print upload instructions for Eden
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # ViTParticleFilterTracker/
ABLATION_DIR = PROJECT_ROOT / "AdvancedDatasetSelection" / "ablation"

VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
VARIANT_NAMES = {
    "v1": "FULL (Fourier+DINO+SAM+EL2N)",
    "v2": "NO_FOURIER (DINO+SAM+EL2N)",
    "v3": "NO_EL2N (Fourier+DINO+SAM)",
    "v4": "QUALITY_ONLY (Fourier filter + random)",
    "v5": "RANDOM (pure random 20k)",
}


def run_cmd(cmd, cwd=None):
    """Run a command and stream output."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)


def validate_variant(variant_dir: Path, variant_id: str):
    """Check that a variant has valid output files."""
    required = [
        f"annotations_train.json",
        f"annotations_val.json",
        f"annotations_test.json",
        "metadata.json",
    ]
    for fname in required:
        fpath = variant_dir / fname
        if not fpath.exists():
            print(f"  MISSING: {fpath}")
            return False

    # Check train annotation count
    with open(variant_dir / "annotations_train.json", "r") as f:
        train = json.load(f)
    n_train = len(train["images"])
    n_ann = len(train["annotations"])

    with open(variant_dir / "annotations_val.json", "r") as f:
        val = json.load(f)
    n_val = len(val["images"])

    with open(variant_dir / "annotations_test.json", "r") as f:
        test = json.load(f)
    n_test = len(test["images"])

    print(f"  {variant_id}: train={n_train} ({n_ann} ann), val={n_val}, test={n_test}")

    if n_train < 1000:
        print(f"  WARNING: Only {n_train} train images (expected ~20000)")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Prepare ablation datasets")
    ap.add_argument("--images-dir", type=str, required=True,
                    help="Directory with all augmented images")
    ap.add_argument("--annotations", type=str, required=True,
                    help="Full COCO annotations JSON (91k images)")
    ap.add_argument("--cache", type=str, default=None,
                    help="Feature cache directory (DINO/Fourier/SAM/EL2N)")
    ap.add_argument("--output", type=str,
                    default=str(SCRIPT_DIR / "output"),
                    help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    help="Which variants to generate")
    ap.add_argument("--skip-split", action="store_true",
                    help="Skip video split generation (use existing)")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "video_split.json"

    # Step 1: Generate video split
    if not args.skip_split:
        print("=" * 60)
        print("STEP 1: Generating video-level split")
        print("=" * 60)
        run_cmd([
            sys.executable,
            str(ABLATION_DIR / "generate_video_split.py"),
            "--images-dir", args.images_dir,
            "--output", str(split_path),
            "--seed", str(args.seed),
        ])
    else:
        if not split_path.exists():
            print(f"ERROR: --skip-split but {split_path} not found")
            sys.exit(1)
        print(f"Using existing split: {split_path}")

    # Step 2: Generate variants
    print("\n" + "=" * 60)
    print("STEP 2: Generating ablation variants")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ABLATION_DIR / "generate_ablation_datasets.py"),
        "--images-dir", args.images_dir,
        "--annotations", args.annotations,
        "--split", str(split_path),
        "--output", str(output_dir),
        "--target", str(args.target),
        "--variants",
    ] + args.variants

    if args.cache:
        cmd.extend(["--cache", args.cache])

    run_cmd(cmd)

    # Step 3: Validate
    print("\n" + "=" * 60)
    print("STEP 3: Validation")
    print("=" * 60)
    all_ok = True
    for vid in args.variants:
        vdir = output_dir / vid
        if not vdir.exists():
            print(f"  ERROR: {vdir} not found")
            all_ok = False
            continue
        if not validate_variant(vdir, vid):
            all_ok = False

    # Check val/test consistency across variants
    val_counts = set()
    test_counts = set()
    for vid in args.variants:
        vdir = output_dir / vid
        if not vdir.exists():
            continue
        with open(vdir / "annotations_val.json") as f:
            val_counts.add(len(json.load(f)["images"]))
        with open(vdir / "annotations_test.json") as f:
            test_counts.add(len(json.load(f)["images"]))

    if len(val_counts) == 1 and len(test_counts) == 1:
        print(f"\n  Val/test sets CONSISTENT across all variants")
    else:
        print(f"\n  WARNING: Val counts differ: {val_counts}")
        print(f"  WARNING: Test counts differ: {test_counts}")
        all_ok = False

    if all_ok:
        print("\nAll variants validated successfully.")
    else:
        print("\nSome variants had issues — check output above.")

    # Step 4: Upload instructions
    print("\n" + "=" * 60)
    print("UPLOAD TO EDEN")
    print("=" * 60)
    print(f"""
Upload the output directory to Eden:

  scp -r {output_dir}/ eden-cluster:~/DETR/ablation/

Then on Eden, verify:
  ls ~/DETR/ablation/v{{1..5}}/annotations_train.json

Submit training:
  cd ~/DETR
  bash run_ablation_all.sh
""")


if __name__ == "__main__":
    main()
