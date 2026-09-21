#!/usr/bin/env python3
"""Second pre-registered measure of document 39: detector misses by instrument speed.

For every finished run the top-scoring prediction of each validation image is compared with the
ground-truth box; a miss is IoU < 0.5 (or no prediction). Frames are split by the speed of the
instrument taken from the ground-truth tracks (`val_speed_px_per_3frames.json`, built locally):
slow half, 50-80, 80-95 and the fastest 5 percent; the fastest 20 percent is reported as well.
Labels are used for scoring only. Plain standard library, so it runs with any python3.

    python3 misses_by_speed.py --results <results dir> --annotations annotations_val.json \
        --speed val_speed_px_per_3frames.json --rule best_val --out MISSES_BY_SPEED.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--speed', type=Path, required=True)
    ap.add_argument('--rule', default='best_val', choices=['best_val', 'last_epoch'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)

    coco = json.loads(args.annotations.read_text(encoding='utf-8'))
    speed_by_name = json.loads(args.speed.read_text(encoding='utf-8'))
    truth = {}
    for a in coco['annotations']:
        truth.setdefault(int(a['image_id']), a['bbox'])                 # one instrument per frame
    speed = {}
    for image in coco['images']:
        name = Path(image['file_name']).name
        if name in speed_by_name and int(image['id']) in truth:
            speed[int(image['id'])] = speed_by_name[name]
    cuts = [quantile(speed.values(), q) for q in (.5, .8, .95)]
    strata = {'slow_half': lambda s: s <= cuts[0], 'q50_80': lambda s: cuts[0] < s <= cuts[1],
              'q80_95': lambda s: cuts[1] < s <= cuts[2], 'fastest_5': lambda s: s > cuts[2],
              'fastest_20': lambda s: s > cuts[1]}
    report = dict(rule=args.rule, frames=len(speed), speed_cuts_px_per_3frames=cuts, runs={})
    for run in sorted(p for p in args.results.iterdir() if (p / 'validation_report.json').is_file()):
        with gzip.open(run / f'eval_{args.rule}' / 'predictions_val.json.gz', 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        predictions = raw['annotations'] if isinstance(raw, dict) and 'annotations' in raw else raw
        best = {}
        for p in predictions:
            k = int(p['image_id'])
            if k not in best or p['score'] > best[k]['score']:
                best[k] = p
        missed = {k: (k not in best or iou(best[k]['bbox'], truth[k]) < .5) for k in speed}
        row = {}
        for name, inside in strata.items():
            ids = [k for k, s in speed.items() if inside(s)]
            row[name] = dict(miss_rate=sum(missed[k] for k in ids) / len(ids), frames=len(ids))
        report['runs'][run.name] = row
        print(run.name, {k: round(v['miss_rate'], 4) for k, v in row.items()}, flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
