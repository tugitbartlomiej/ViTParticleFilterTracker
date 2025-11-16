#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download DETR checkpoints from Eden and run benchmarks for epochs 40, 60, 80, 100, 160
"""

import subprocess
import os
import sys
from pathlib import Path
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configuration
EDEN_HOST = "eden-cluster"
EDEN_USER = "bpiotrowski"
EDEN_CHECKPOINT_DIR = "/home/mgr-2024-bartlomiej-lowko/DETR_Cataract_Training/checkpoints"
LOCAL_CHECKPOINT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR")
EPOCHS_TO_TEST = [40, 60, 80, 100, 160]

def ssh_command(cmd):
    """Execute SSH command on Eden cluster"""
    full_cmd = f'ssh {EDEN_HOST} "{cmd}"'
    print(f"Running: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

def scp_download(remote_path, local_path):
    """Download file from Eden using SCP"""
    cmd = f"scp {EDEN_HOST}:{remote_path} {local_path}"
    print(f"Downloading: {remote_path} -> {local_path}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def list_available_checkpoints():
    """List all checkpoints on Eden"""
    # Try multiple possible locations
    search_paths = [
        EDEN_CHECKPOINT_DIR,
        "/home/mgr-2024-bartlomiej-lowko/DETR_Cataract_Training",
        "/home/mgr-2024-bartlomiej-lowko",
    ]

    checkpoints = []

    for search_path in search_paths:
        cmd = f"find {search_path} -name 'checkpoint_epoch_*.pth' -type f 2>/dev/null"
        output, ret = ssh_command(cmd)

        if output:
            print(f"Found checkpoints in {search_path}:")
            for line in output.split('\n'):
                line = line.strip()
                if 'checkpoint_epoch_' in line and line.endswith('.pth'):
                    print(f"  {line}")
                    # Extract epoch number
                    try:
                        epoch_str = line.split('checkpoint_epoch_')[1].split('.pth')[0]
                        epoch = int(epoch_str)
                        checkpoints.append((epoch, line))
                    except (IndexError, ValueError):
                        pass

    if not checkpoints:
        return []

    # Return sorted list of (epoch, full_path) tuples
    return sorted(checkpoints, key=lambda x: x[0])

def download_checkpoint(epoch, remote_path=None):
    """Download checkpoint for specific epoch"""
    if remote_path is None:
        remote_file = f"{EDEN_CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth"
    else:
        remote_file = remote_path

    local_file = LOCAL_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"

    # Check if already downloaded
    if local_file.exists():
        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Epoch {epoch} already downloaded ({size_mb:.1f} MB)")
        return True

    # Download
    print(f"Downloading epoch {epoch} from {remote_file}...")
    success = scp_download(remote_file, str(local_file))

    if success:
        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded epoch {epoch} ({size_mb:.1f} MB)")
    else:
        print(f"[FAIL] Failed to download epoch {epoch}")

    return success

def run_benchmark_for_epoch(epoch, conf_thresholds=[0.15, 0.2, 0.3, 0.4, 0.5]):
    """Run benchmark for specific epoch with multiple confidence thresholds"""
    checkpoint_path = LOCAL_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running benchmark for DETR Epoch {epoch}")
    print(f"{'='*60}\n")

    # Run multi-epoch benchmark script
    benchmark_script = Path("YOLO_DETR_Benchmarks/scripts/multi_epoch_query_benchmark.py")

    if not benchmark_script.exists():
        print(f"✗ Benchmark script not found: {benchmark_script}")
        return False

    # Create config for this epoch
    config = {
        "yolo_model": "E:/Cataract/yolo11/train5/weights/best.pt",
        "detr_checkpoints": [
            {
                "epoch": epoch,
                "path": str(checkpoint_path),
                "conf_thresholds": conf_thresholds
            }
        ],
        "coco_dataset": "E:/Cataract/CADTD/COCO/annotations/instances_test.json",
        "output_dir": f"YOLO_DETR_Benchmarks/Advanced_Analysis/benchmark_results_epoch{epoch}",
        "device": "cuda"
    }

    config_file = Path(f"YOLO_DETR_Benchmarks/Advanced_Analysis/config_epoch{epoch}.yaml")

    with open(config_file, 'w') as f:
        import yaml
        yaml.dump(config, f)

    # Run benchmark
    cmd = f"python {benchmark_script} --config {config_file}"
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    return result.returncode == 0

def main():
    print("="*60)
    print("DETR Multi-Epoch Benchmark Suite")
    print("="*60)

    # Create checkpoint directory
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: List available checkpoints on Eden
    print("\n[1/3] Listing available checkpoints on Eden...")
    checkpoint_list = list_available_checkpoints()

    if not checkpoint_list:
        print("[FAIL] No checkpoints found on Eden")
        return 1

    available_epochs = {epoch: path for epoch, path in checkpoint_list}
    print(f"[OK] Found {len(available_epochs)} checkpoints: {sorted(available_epochs.keys())}")

    # Step 2: Download requested epochs
    print("\n[2/3] Downloading checkpoints...")
    epochs_to_download = [(e, available_epochs[e]) for e in EPOCHS_TO_TEST if e in available_epochs]

    if not epochs_to_download:
        print(f"[FAIL] None of requested epochs {EPOCHS_TO_TEST} found on Eden")
        print(f"Available: {sorted(available_epochs.keys())}")
        return 1

    print(f"Will download epochs: {[e for e, _ in epochs_to_download]}")

    downloaded = []
    for epoch, remote_path in epochs_to_download:
        if download_checkpoint(epoch, remote_path):
            downloaded.append(epoch)

    print(f"\n[OK] Downloaded {len(downloaded)}/{len(epochs_to_download)} checkpoints")

    # Step 3: Run benchmarks
    print("\n[3/3] Running benchmarks...")
    results = {}

    for epoch in downloaded:
        success = run_benchmark_for_epoch(epoch)
        results[epoch] = "SUCCESS" if success else "FAILED"
        print(f"\nEpoch {epoch}: {results[epoch]}")

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for epoch, status in results.items():
        print(f"Epoch {epoch:3d}: {status}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
