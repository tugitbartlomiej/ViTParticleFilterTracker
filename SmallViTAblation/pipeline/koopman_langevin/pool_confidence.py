#!/usr/bin/env python3
"""Least-confidence score of the L0 detector on every frame of the label-eligible pool (arm B4).

uncertainty(frame) = 1 - highest instrument score of the detector trained on the shared L0.
The same proxy as `pipeline.active_learning.uncertainty`; it is not a calibrated epistemic
measure. Labels are not read. Needs `rfdetr`, so it runs on the cluster as a plain script.

    python pool_confidence.py --data-root <extracted archive> --checkpoint L0.pth --out uncertainty_L0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args(argv)

    import rfdetr
    import torch
    from PIL import Image

    rows = [json.loads(line) for line in (args.data_root / 'pool_manifest.jsonl').open(encoding='utf-8') if line.strip()]
    rows = [r for r in rows if r.get('split', 'train') == 'train']
    detector = rfdetr.RFDETRNano(pretrain_weights=str(args.checkpoint), resolution=args.resolution)
    net = detector.model.model.eval()
    captured: dict = {}
    net.register_forward_hook(lambda module, inputs, output: captured.__setitem__('out', output))
    scores: dict[str, float] = {}
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        images = [Image.open(args.data_root / r['path']).convert('RGB') for r in chunk]
        with torch.inference_mode():
            detector.predict(images, threshold=0.0)
        out = captured['out']
        logits = out['pred_logits'] if isinstance(out, dict) else (out[1] if out[0].shape[-1] == 4 else out[0])
        top = logits.float().sigmoid().flatten(1).max(1).values.cpu().tolist()
        for r, value in zip(chunk, top):
            scores[r['frame_id']] = 1.0 - float(value)
        if (start // args.batch_size) % 50 == 0:
            print(f'[confidence] {start}/{len(rows)} {time.perf_counter() - started:.0f}s', flush=True)
    with args.checkpoint.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores), encoding='utf-8')
    args.out.with_suffix('.meta.json').write_text(json.dumps(dict(
        frames=len(scores), checkpoint=str(args.checkpoint), checkpoint_sha256=sha, labels_used=False,
        definition='1 - max instrument score of the L0 detector')), encoding='utf-8')
    print(json.dumps(dict(frames=len(scores), out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
