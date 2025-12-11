import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    # Convert to grayscale and resize
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return int(''.join('1' if v else '0' for v in diff.flatten()), 2)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def natural_video_stem(video_path: Path) -> str:
    return video_path.stem


def list_videos(root: Path) -> List[Path]:
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    return [p for p in root.rglob('*') if p.suffix in exts]


def sample_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: float,
    blur_threshold: float,
    phash_threshold: int,
    max_frames_per_video: int | None,
) -> Tuple[List[Dict], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(orig_fps / max(target_fps, 1e-6))), 1)

    selected: List[Dict] = []
    phash_buffer: List[int] = []
    written = 0

    frame_idx = 0
    saved_idx = 0
    while True:
        ret = cap.read()[0]
        if not ret:
            break
        # Read only each 'step'-th frame
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ret, frame = cap.read(), None
        # We already consumed one read above; re-read current position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = variance_of_laplacian(gray)
        if blur < blur_threshold:
            frame_idx += 1
            continue

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ph = dhash(img_pil)
        # Check against recent hashes for near-duplicates
        if any(hamming_distance(ph, prev) < phash_threshold for prev in phash_buffer[-32:]):
            frame_idx += 1
            continue

        video_stem = natural_video_stem(video_path)
        filename = f"{video_stem}_frame_{frame_idx:08d}.jpg"
        out_path = out_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        phash_buffer.append(ph)
        selected.append({
            'video_path': str(video_path),
            'video_stem': video_stem,
            'frame_index': int(frame_idx),
            'saved_index': int(saved_idx),
            'filename': filename,
            'blur': blur,
            'phash': ph,
            'orig_fps': float(orig_fps),
            'frame_count': frame_count,
        })
        written += 1
        saved_idx += 1

        if max_frames_per_video is not None and written >= max_frames_per_video:
            break

        frame_idx += 1

    cap.release()
    return selected, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True, help='Root folder with videos')
    ap.add_argument('--out', required=True, help='Output images directory')
    ap.add_argument('--target-fps', type=float, default=2.0)
    ap.add_argument('--blur-th', type=float, default=80.0)
    ap.add_argument('--phash-th', type=int, default=10)
    ap.add_argument('--max-per-video', type=int, default=1500)
    ap.add_argument('--manifest', default=None, help='Path to save frames manifest JSON')
    args = ap.parse_args()

    videos_root = Path(args.videos)
    out_images = Path(args.out)
    out_images.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else (out_images.parent / 'frames_manifest.json')

    all_selected: List[Dict] = []
    total_written = 0
    vids = list_videos(videos_root)
    for vp in vids:
        sel, w = sample_frames(
            vp,
            out_images,
            target_fps=args.target_fps,
            blur_threshold=args.blur_th,
            phash_threshold=args.phash_th,
            max_frames_per_video=args.max_per_video if args.max_per_video > 0 else None,
        )
        all_selected.extend(sel)
        total_written += w
        print(f"Processed {vp.name}: saved {w} frames")

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'frames': all_selected}, f, ensure_ascii=False)
    print(f"Saved manifest: {manifest_path} (frames: {len(all_selected)}, total_written: {total_written})")


if __name__ == '__main__':
    main()

