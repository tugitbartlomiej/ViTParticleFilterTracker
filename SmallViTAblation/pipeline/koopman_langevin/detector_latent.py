#!/usr/bin/env python3
"""Read the latent of an RF-DETR checkpoint on dense clips, one "particle" per frame.

For every frame the detector is run once and three vectors are kept:
  query        hidden state of the decoder query with the highest instrument score
               (the vector from which the box and the class are read)
  box_pool     encoder feature map averaged inside the PREDICTED box
  global_pool  encoder feature map averaged over the whole frame
together with the predicted box and its score. No label is read here; labels enter only in
`epoch_dynamics.py`, to score what the latent carries.

One NPZ per checkpoint. The same frames under several checkpoints give the motion of each
particle along the training clock; consecutive frames under one checkpoint give its motion
along the video clock.

The file also carries `frame_ids` and `features`, which is the chunk format of
`pipeline.curation.features`, so `select.py --cache <dir>` can rank frames by the dynamics
of the DETECTOR latent instead of the DINOv2 latent.

Needs the `rfdetr` package, so it runs where the detector was trained:

    python -m SmallViTAblation.pipeline.koopman_langevin.detector_latent \
        --package <dir with frames/ and tracks.json> --checkpoint best.pth --tag b2500_best --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _tensor(value):
    """First tensor inside nested tuples, lists or NestedTensor-like objects."""
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, 'tensors'):
        return value.tensors
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _tensor(item)
            if found is not None:
                return found
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--checkpoint', help='weights file; omit for the COCO weights the training started from')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args(argv)

    import rfdetr
    import torch
    from PIL import Image

    kwargs = dict(resolution=args.resolution)
    if args.checkpoint:
        kwargs['pretrain_weights'] = str(args.checkpoint)
    detector = rfdetr.RFDETRNano(**kwargs)
    net = detector.model.model.eval()
    captured: dict = {}
    net.register_forward_hook(lambda m, i, o: captured.__setitem__('out', o))
    net.class_embed.register_forward_pre_hook(lambda m, i: captured.__setitem__('hs', i[0]))
    net.backbone.register_forward_hook(lambda m, i, o: captured.__setitem__('feat', _tensor(o)))

    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    frame_ids, query, box_pool, global_pool, boxes, scores = [], [], [], [], [], []
    started = time.perf_counter()
    for clip, track in tracks.items():
        files = [args.package / 'frames' / clip / f'{n:08d}.jpg' for n in track['frame']]
        for start in range(0, len(files), args.batch_size):
            images = [Image.open(f).convert('RGB') for f in files[start:start + args.batch_size]]
            sizes = np.array([[im.width, im.height] for im in images], dtype=np.float32)
            with torch.inference_mode():
                detector.predict(images, threshold=0.0)
            out = captured['out']
            if isinstance(out, dict):
                logits, coords = out['pred_logits'], out['pred_boxes']
            else:
                first, second = out[0], out[1]
                coords, logits = (first, second) if first.shape[-1] == 4 else (second, first)
            hs = captured['hs']
            hs = hs[-1] if hs.dim() == 4 else hs                       # last decoder layer
            n_q = min(hs.shape[1], logits.shape[1])                    # one query group at inference
            score, best = logits[:, :n_q].float().sigmoid().max(-1).values.max(-1)
            rows = torch.arange(len(images), device=hs.device)
            q = hs[rows, best].float()
            c = coords[rows, best].float().clamp(0, 1)                 # cx, cy, w, h in [0, 1]
            feat = captured['feat'].float()                            # B, C, H, W
            b, ch, fh, fw = feat.shape
            ys = (torch.arange(fh, device=feat.device) + .5) / fh
            xs = (torch.arange(fw, device=feat.device) + .5) / fw
            inside = ((xs[None, None, :] - c[:, 0, None, None]).abs() <= c[:, 2, None, None] / 2) & \
                     ((ys[None, :, None] - c[:, 1, None, None]).abs() <= c[:, 3, None, None] / 2)
            inside = inside.float()
            inside[inside.sum((1, 2)) == 0] = 1.0                      # degenerate box: whole frame
            pooled = (feat * inside[:, None]).sum((2, 3)) / inside.sum((1, 2))[:, None]
            for k, n in enumerate(track['frame'][start:start + args.batch_size]):
                frame_ids.append(f'cataract_lmm/tracking/{clip}/{n:08d}')
            query.append(q.cpu().numpy())
            box_pool.append(pooled.cpu().numpy())
            global_pool.append(feat.mean((2, 3)).cpu().numpy())
            w, h = sizes[:, 0], sizes[:, 1]
            cn = c.cpu().numpy()
            boxes.append(np.stack([(cn[:, 0] - cn[:, 2] / 2) * w, (cn[:, 1] - cn[:, 3] / 2) * h,
                                   cn[:, 2] * w, cn[:, 3] * h], 1))
            scores.append(score.cpu().numpy())
        print(f'[latent] {args.tag} {clip} {time.perf_counter() - started:.0f}s', flush=True)

    query, box_pool, global_pool = np.vstack(query), np.vstack(box_pool), np.vstack(global_pool)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f'LATENT_{args.tag}.npz'
    sha = None
    if args.checkpoint:
        with open(args.checkpoint, 'rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    np.savez_compressed(target, frame_ids=np.array(frame_ids), query=query.astype(np.float32),
                        box_pool=box_pool.astype(np.float32), global_pool=global_pool.astype(np.float32),
                        features=np.hstack([query, global_pool]).astype(np.float32),
                        pred_box=np.vstack(boxes).astype(np.float32), score=np.concatenate(scores).astype(np.float32),
                        meta=json.dumps(dict(tag=args.tag, checkpoint=str(args.checkpoint), checkpoint_sha256=sha,
                                             resolution=args.resolution, labels_used=False)))
    print(json.dumps(dict(tag=args.tag, frames=len(frame_ids), query_dim=int(query.shape[1]),
                          feature_map=[int(fh), int(fw)], out=str(target))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
