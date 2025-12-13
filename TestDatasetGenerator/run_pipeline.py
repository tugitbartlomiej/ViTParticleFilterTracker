"""
Run Complete Pipeline: Extract Frames + Detect Duplicates
==========================================================
Convenience script to run both extraction and duplicate detection in sequence.

Usage:
    python run_pipeline.py
    python run_pipeline.py --total-frames 500 --remove-duplicates

Author: Claude Code
Date: 2025-12-13
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("TEST DATASET GENERATION PIPELINE")
    print("=" * 70)

    # Step 1: Extract frames
    print("\n[STEP 1/2] Extracting random frames from videos...")
    print("-" * 70)

    extract_script = script_dir / "extract_random_frames.py"
    extract_cmd = [sys.executable, str(extract_script)] + sys.argv[1:]

    result = subprocess.run(extract_cmd)
    if result.returncode != 0:
        print("ERROR: Frame extraction failed!")
        sys.exit(1)

    # Step 2: Detect duplicates
    print("\n[STEP 2/2] Detecting duplicates...")
    print("-" * 70)

    detect_script = script_dir / "detect_duplicates.py"
    detect_cmd = [sys.executable, str(detect_script)]

    # Pass relevant arguments
    if "--remove-duplicates" in sys.argv:
        detect_cmd.append("--remove-duplicates")

    result = subprocess.run(detect_cmd)
    if result.returncode != 0:
        print("ERROR: Duplicate detection failed!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Test frames: {script_dir / 'output' / 'test_frames'}")
    print(f"Reports: {script_dir / 'output' / 'reports'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
