#!/usr/bin/env python3
"""
Check for byte-identical images between the selected training set and the
Roboflow dataset (train/valid/test splits).

Usage:
  python check_duplicates_with_roboflow.py \
    --selected-dir F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/selected_dataset/images \
    --roboflow-root E:/cataract_surgery_Instruments_detection.v1i.coco

Outputs a short summary to stdout and a JSON list of matches if --output is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def iter_images(root: Path, splits: Sequence[str] | None) -> Iterable[Path]:
    if splits:
        for split in splits:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for p in split_dir.rglob("*"):
                if p.suffix.lower() in VALID_EXTS:
                    yield p
    else:
        for p in root.rglob("*"):
            if p.suffix.lower() in VALID_EXTS:
                yield p


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_hash_index(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for i, img_path in enumerate(paths, 1):
        digest = sha256_file(img_path)
        index.setdefault(digest, []).append(img_path)
        if i % 1000 == 0:
            print(f"[Roboflow] Hashed {i} files...")
    print(f"[Roboflow] Indexed {len(index)} unique hashes.")
    return index


def find_duplicates(selected_paths: Iterable[Path], roboflow_index: Dict[str, List[Path]]) -> List[Tuple[Path, List[Path]]]:
    matches: List[Tuple[Path, List[Path]]] = []
    for i, sel_path in enumerate(selected_paths, 1):
        digest = sha256_file(sel_path)
        if digest in roboflow_index:
            matches.append((sel_path, roboflow_index[digest]))
        if i % 1000 == 0:
            print(f"[Selected] Checked {i} files...")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect identical images between selected dataset and Roboflow splits.")
    parser.add_argument("--selected-dir", type=Path, required=True, help="Directory with selected images (flat or nested).")
    parser.add_argument("--roboflow-root", type=Path, required=True, help="Roboflow dataset root containing train/valid/test.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to store duplicate pairs.")
    args = parser.parse_args()

    if not args.selected_dir.exists():
        raise SystemExit(f"Selected dir not found: {args.selected_dir}")
    if not args.roboflow_root.exists():
        raise SystemExit(f"Roboflow root not found: {args.roboflow_root}")

    roboflow_paths = list(iter_images(args.roboflow_root, splits=["train", "valid", "test"]))
    print(f"[Roboflow] Found {len(roboflow_paths)} candidate images.")
    roboflow_index = build_hash_index(roboflow_paths)

    selected_paths = list(iter_images(args.selected_dir, splits=None))
    print(f"[Selected] Found {len(selected_paths)} candidate images.")
    duplicates = find_duplicates(selected_paths, roboflow_index)

    print(f"\n=== Summary ===")
    print(f"Roboflow images indexed: {len(roboflow_paths)}")
    print(f"Selected images checked: {len(selected_paths)}")
    print(f"Duplicates found: {len(duplicates)}")

    if duplicates:
        for sel, robos in duplicates[:10]:
            print(f"- {sel} == {len(robos)} Roboflow match(es): {', '.join(str(p) for p in robos)}")
        if len(duplicates) > 10:
            print(f"... ({len(duplicates) - 10} more)")

    if args.output:
        serializable = [
            {"selected": str(sel), "roboflow_matches": [str(p) for p in robos]}
            for sel, robos in duplicates
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(serializable, indent=2))
        print(f"[Output] Saved JSON with duplicates to {args.output}")


if __name__ == "__main__":
    main()
