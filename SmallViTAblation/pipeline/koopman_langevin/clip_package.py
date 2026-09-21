#!/usr/bin/env python3
"""Pack dense frames of a few clips plus their ground-truth tracks for the detector-latent probe.

The static campaign archive holds sparse frames only (gaps of 5 to 70 frames), which is not
enough to follow the latent of one instrument from frame to frame. This tool cuts dense
stretches out of the original tracking archives:

    <out>/frames/<clip>/<frame:08d>.jpg      untouched JPEG bytes
    <out>/tracks.json                        per clip: split, group, frame, box, tip

Test clips are refused: the test split of the campaign has not been evaluated and must stay
unseen. Labels in `tracks.json` are used for probing and scoring only, never for ranking.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.clip_package \
        --split-manifest SmallViTAblation/feature_cache/eden_static36/data/tracking_static36/split_manifest.json \
        --out SmallViTAblation/feature_cache/detector_probe_clips --val-clips 12 --train-clips 4
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .tokens import REPOSITORY, clip_records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, default=REPOSITORY / 'Dataset/Cataract-LMM')
    ap.add_argument('--split-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--val-clips', type=int, default=12)
    ap.add_argument('--train-clips', type=int, default=4)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max-frames', type=int, default=300)
    args = ap.parse_args(argv)

    split = json.loads(args.split_manifest.read_text(encoding='utf-8'))['groups']
    audit = json.loads((args.dataset_root / '_audyt_pacjentow.json').read_text(encoding='utf-8'))
    group_of = {v: g for g, videos in audit['grupy_operacji'].items() for v in videos}
    archives = {a.stem: a for a in sorted((args.dataset_root / '3_Object_Tracking/annotations').glob('*.zip'))}

    def role(stem: str) -> str | None:
        return split.get(stem) or split.get(str(group_of.get(stem)))

    chosen = []
    for wanted, count in (('val', args.val_clips), ('train', args.train_clips)):
        names = sorted(s for s in archives if role(s) == wanted)
        step = max(1, len(names) // max(count, 1))
        chosen += [(n, wanted) for n in names[::step][:count]]
    if any(role(n) == 'test' for n, _ in chosen):
        raise RuntimeError('test clips must never enter the probe package')

    tracks = {}
    for stem, part in chosen:
        parent, rows = clip_records(archives[stem], args.stride, args.max_frames)
        target = args.out / 'frames' / stem
        target.mkdir(parents=True, exist_ok=True)
        frames, boxes, tips = [], [], []
        with zipfile.ZipFile(archives[stem]) as z:
            for r in rows:
                number = int(r['frame_number'])
                file = target / f'{number:08d}.jpg'
                if not file.exists():
                    file.write_bytes(z.read(str(parent / r['filename'])))
                frames.append(number)
                boxes.append(json.loads(r['cystotome_bbox']) if (r.get('cystotome_bbox') or '').strip() else None)
                tips.append(json.loads(r['cystotome_tip']) if (r.get('cystotome_tip') or '').strip() else None)
        tracks[stem] = dict(split=part, group_id=str(group_of.get(stem, stem)), stride=args.stride,
                            frame=frames, box=boxes, tip=tips)
        print(f'[package] {stem} {part} frames={len(frames)}', flush=True)
    (args.out / 'tracks.json').write_text(json.dumps(tracks), encoding='utf-8')
    print(json.dumps(dict(clips=len(tracks), frames=sum(len(t['frame']) for t in tracks.values()),
                          out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
