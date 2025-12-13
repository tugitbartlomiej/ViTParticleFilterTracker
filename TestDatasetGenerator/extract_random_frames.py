"""
Extract Random Frames from Videos for Test Dataset
===================================================
Extracts random frames from multiple video files to create a test dataset.
Ensures each video is represented with an equal number of frames.

Author: Claude Code
Date: 2025-12-13
"""

import cv2
import random
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple
import yaml


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'video_files': [
            "E:/Cataract/videos/micro/train03.mp4",
            "E:/Cataract/videos/micro/train04.mp4",
            "E:/Cataract/videos/micro/train05.mp4",
            "E:/Cataract/videos/micro/train06.mp4",
            "E:/Cataract/videos/micro/train07.mp4",
            "E:/Cataract/videos/micro/train08.mp4",
            "E:/Cataract/videos/micro/train09.mp4",
            "E:/Cataract/videos/micro/train10.mp4",
            "E:/Cataract/videos/micro/train11.mp4",
            "E:/Cataract/videos/micro/train12.mp4",
            "E:/Cataract/videos/micro/train13.mp4",
            "E:/Cataract/videos/micro/train14.mp4",
            "E:/Cataract/videos/micro/train15.mp4",
            "E:/Cataract/videos/micro/train16.mp4",
            "E:/Cataract/videos/micro/train17.mp4",
            "E:/Cataract/videos/micro/train18.mp4",
            "E:/Cataract/videos/micro/train19.mp4",
            "E:/Cataract/videos/micro/train20.mp4",
            "E:/Cataract/videos/micro/train21.mp4",
            "E:/Cataract/videos/micro/train22.mp4",
            "E:/Cataract/videos/micro/train23.mp4",
            "E:/Cataract/videos/micro/train24.mp4",
            "E:/Cataract/videos/micro/train25.mp4",
            "E:/Cataract/videos/micro/test01.mp4",
            "E:/Cataract/videos/micro/test02.mp4",
            "E:/Cataract/videos/micro/test03.mp4",
            "E:/Cataract/videos/micro/test04.mp4",
        ],
        'total_frames': 1000,
        'output_dir': 'output/test_frames',
        'random_seed': 42,
        'jpeg_quality': 95,
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            default_config.update(user_config)

    return default_config


def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    """Get video information: total frames, fps, width, height."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return total_frames, fps, width, height


def extract_frames_from_video(
    video_path: str,
    frame_indices: List[int],
    output_dir: Path,
    jpeg_quality: int = 95
) -> List[Dict]:
    """Extract specific frames from a video file."""
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    extracted = []
    sorted_indices = sorted(frame_indices)

    for frame_idx in tqdm(sorted_indices, desc=f"Extracting from {video_name}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Format: {video_name}_frame_{XXXXXXX}.jpg (matching training dataset format)
            filename = f"{video_name}_frame_{frame_idx:07d}.jpg"
            output_path = output_dir / filename

            cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )

            extracted.append({
                'filename': filename,
                'source_video': video_name,
                'frame_index': frame_idx,
                'path': str(output_path)
            })

    cap.release()
    return extracted


def distribute_frames_across_videos(
    video_files: List[str],
    total_frames: int
) -> Dict[str, List[int]]:
    """Distribute frame quota across videos, then select random frames."""
    video_info = {}

    print("Analyzing videos...")
    for video_path in tqdm(video_files, desc="Getting video info"):
        if not Path(video_path).exists():
            print(f"  WARNING: Video not found: {video_path}")
            continue

        try:
            total, fps, w, h = get_video_info(video_path)
            video_info[video_path] = {
                'total_frames': total,
                'fps': fps,
                'width': w,
                'height': h
            }
        except Exception as e:
            print(f"  ERROR processing {video_path}: {e}")

    if not video_info:
        raise ValueError("No valid videos found!")

    # Calculate frames per video (equal distribution)
    num_videos = len(video_info)
    frames_per_video = total_frames // num_videos
    remainder = total_frames % num_videos

    print(f"\nDistributing {total_frames} frames across {num_videos} videos")
    print(f"  Base: {frames_per_video} frames per video")
    print(f"  Remainder: {remainder} extra frames distributed to first videos")

    # Assign frame indices
    frame_selection = {}
    for i, (video_path, info) in enumerate(video_info.items()):
        # Add one extra frame to first 'remainder' videos
        n_frames = frames_per_video + (1 if i < remainder else 0)

        # Ensure we don't request more frames than video has
        max_frames = info['total_frames']
        n_frames = min(n_frames, max_frames)

        # Random selection without replacement
        selected = random.sample(range(max_frames), n_frames)
        frame_selection[video_path] = selected

        video_name = Path(video_path).stem
        print(f"  {video_name}: {n_frames} frames (video has {max_frames} total)")

    return frame_selection


def main():
    parser = argparse.ArgumentParser(
        description="Extract random frames from videos for test dataset"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--total-frames', '-n',
        type=int,
        default=None,
        help='Total number of frames to extract (overrides config)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config)'
    )
    args = parser.parse_args()

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path) if config_path.exists() else None)

    # Override with command line arguments
    if args.total_frames:
        config['total_frames'] = args.total_frames
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.seed:
        config['random_seed'] = args.seed

    # Set random seed for reproducibility
    random.seed(config['random_seed'])

    # Setup output directory
    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Test Dataset Frame Extractor")
    print("=" * 60)
    print(f"Total frames to extract: {config['total_frames']}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {config['random_seed']}")
    print(f"JPEG quality: {config['jpeg_quality']}")
    print("=" * 60)

    # Distribute frames across videos
    frame_selection = distribute_frames_across_videos(
        config['video_files'],
        config['total_frames']
    )

    # Extract frames
    print("\n" + "=" * 60)
    print("Extracting frames...")
    print("=" * 60)

    all_extracted = []
    for video_path, frame_indices in frame_selection.items():
        try:
            extracted = extract_frames_from_video(
                video_path,
                frame_indices,
                output_dir,
                config['jpeg_quality']
            )
            all_extracted.extend(extracted)
        except Exception as e:
            print(f"ERROR extracting from {video_path}: {e}")

    # Save extraction manifest
    manifest = {
        'extraction_date': datetime.now().isoformat(),
        'config': {
            'total_frames_requested': config['total_frames'],
            'random_seed': config['random_seed'],
            'jpeg_quality': config['jpeg_quality']
        },
        'statistics': {
            'total_extracted': len(all_extracted),
            'videos_processed': len(frame_selection),
            'frames_per_video': {
                Path(k).stem: len(v) for k, v in frame_selection.items()
            }
        },
        'frames': all_extracted
    }

    manifest_path = output_dir / 'extraction_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total frames extracted: {len(all_extracted)}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 60)

    return manifest


if __name__ == "__main__":
    main()
