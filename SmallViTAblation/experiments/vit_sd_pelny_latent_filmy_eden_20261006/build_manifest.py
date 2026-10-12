"""Map existing TRAIN film IDs to ZIP MP4 members, without extraction or SSH.

CLI --split-manifest accepts the existing pool_manifest.jsonl (per-frame split),
a JSON list of records, or the canonical split_manifest.json with groups mapping.
TRAIN is written; --include-val additionally writes val with its separate split.
Conflicting splits, duplicated videos or missing selected
films fail. Annotation ZIPs are inspected by reading annotation.csv only.
"""
import argparse
import csv
import io
import json
import os
import re
import zipfile
from pathlib import Path


def read_splits(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(data, dict) and 'groups' in data:
        data = [dict(video_id=film, split=split) for film, split in data['groups'].items()]
    elif isinstance(data, dict):
        data = data.get('records', data.get('rows'))
    if not isinstance(data, list):
        raise ValueError('Split source must be pool JSONL, JSON records, or groups mapping')
    splits = {}
    for row in data:
        film = row.get('video_id') or row.get('film') or row.get('group_id')
        split = str(row.get('split', '')).lower()
        if split == 'validation':
            split = 'val'
        if not isinstance(film, str) or not film or split not in ('train', 'val', 'test', 'unknown'):
            raise ValueError(f'Missing or invalid film/split in source: {row!r}')
        if film in splits and splits[film] != split:
            raise ValueError(f'Conflicting splits for {film}')
        splits[film] = split
    if not any(s == 'train' for s in splits.values()):
        raise ValueError('No TRAIN film IDs in split source')
    return splits


def check_annotation(path, film):
    from dane import _tool_bbox
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if Path(n).name == 'annotation.csv']
        if len(names) != 1:
            raise ValueError(f'{film}: expected exactly one annotation.csv')
        # Only the CSV member is opened, never any JPEG/image.
        with archive.open(names[0]) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as text:
            reader = csv.DictReader(text)
            if not {'filename', 'frame_number', 'cystotome_seg', 'cystotome_bbox'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{film}: annotation.csv missing required columns')
            frames = set()
            provided = 0
            for row in reader:
                frame = int(row['frame_number'])
                match = re.fullmatch(re.escape(film) + r'_(\d+)', Path(row['filename']).stem)
                if frame < 1 or frame in frames or match is None or int(match.group(1)) != frame:
                    raise ValueError(f'{film}: invalid CSV filename/frame mapping')
                frames.add(frame)
                if _tool_bbox(row, f'{film} frame {frame}') is not None:
                    provided += 1
            if not frames or sorted(frames) != list(range(1, len(frames) + 1)):
                raise ValueError(f'{film}: CSV frames must cover 1..N')
    return dict(frame_count=len(frames), tool_labeled_count=provided,
                annotation_count=len(frames), annotation_member=names[0],
                annotation_frame_first=1, annotation_frame_last=len(frames))


def build_manifest(video_dir, annotation_dir, split_manifest, include_val=False):
    video_dir, annotation_dir = Path(video_dir), Path(annotation_dir)
    splits = read_splits(split_manifest)
    selected_splits = {'train', 'val'} if include_val else {'train'}
    csv.field_size_limit(32 * 1024 * 1024)
    archives = sorted(video_dir.glob('videos_part*.zip'))
    if not archives:
        raise ValueError(f'No videos_part*.zip under {video_dir}')
    rows = []
    seen = set()
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                if info.is_dir() or Path(info.filename).suffix.lower() != '.mp4':
                    continue
                film = Path(info.filename).stem
                if film in seen:
                    raise ValueError(f'Duplicate MP4 film {film}')
                seen.add(film)
                if splits.get(film) not in selected_splits:
                    continue
                annotation = annotation_dir / (film + '.zip')
                if not annotation.is_file():
                    raise FileNotFoundError(f'{film}: annotation ZIP missing: {annotation}')
                metadata = check_annotation(annotation, film)
                rows.append(dict(zip_path=str(path.resolve()), member=info.filename, video_id=film,
                                 split=splits[film], annotation_zip=str(annotation.resolve()), **metadata))
    missing = {f for f, s in splits.items() if s in selected_splits} - {r['video_id'] for r in rows}
    if missing:
        raise ValueError(f'Selected films absent from video ZIPs: {sorted(missing)}')
    return sorted(rows, key=lambda r: r['video_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video-dir', 'annotation-dir', 'split-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--include-val', action='store_true', help='Include separate val rows, never relabel them train')
    args = parser.parse_args()
    rows = build_manifest(args.video_dir, args.annotation_dir, args.split_manifest, args.include_val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        tmp.replace(args.output)
    finally:
        tmp.unlink(missing_ok=True)
    summary = {split: dict(films=sum(r['split'] == split for r in rows),
                         frame_count=sum(r['frame_count'] for r in rows if r['split'] == split),
                         tool_labeled_count=sum(r['tool_labeled_count'] for r in rows if r['split'] == split))
               for split in ('train', 'val')}
    print(json.dumps(dict(output=str(args.output),splits=summary,
                          coverage='CSV manifest eligibility, not frames already seen by training'), ensure_ascii=False))


if __name__ == '__main__':
    main()
