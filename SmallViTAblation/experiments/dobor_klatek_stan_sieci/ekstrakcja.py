#!/usr/bin/env python3
"""Extract predicted instrument ROI features on a manifest's training frames.

Input images may be a dense, label-free context package. Ground-truth annotations
are never used. The winning detection is chosen independently in every frame;
this is a network-state measurement, not tracking or instance association.

Run in the RF-DETR environment, for example:
  python -B -P ekstrakcja.py --data-root DENSE_PACKAGE --checkpoint L0.pth --out L0.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


INPUT_FIELDS = ['confidence', 'cx', 'cy', 'width', 'height', 'mean_brightness']


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_train_rows(manifest: Path, data_root: Path) -> list[dict]:
    """Whitelist image/time identifiers; skip non-training rows before reading paths."""
    result, seen = [], set()
    root = data_root.resolve()
    with manifest.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'train':
                continue
            fid = str(row['frame_id'])
            if fid in seen:
                raise ValueError(f'duplicate training frame_id: {fid}')
            seen.add(fid)
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'image is outside data-root: {path}')
            if not path.is_file():
                raise FileNotFoundError(f'missing train image at manifest line {number}: {path}')
            frame_index = float(row['frame_index'])
            if not np.isfinite(frame_index):
                raise ValueError(f'non-finite frame_index: {fid}')
            result.append(dict(frame_id=fid, path=str(path), split='train',
                               video_id=str(row['video_id']), group_id=str(row['group_id']),
                               frame_index=frame_index))
    if not result:
        raise ValueError('manifest contains no explicit split=train rows')
    return sorted(result, key=lambda r: (r['video_id'], r['frame_index'], r['frame_id']))


def best_prediction(logits: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take the largest sigmoid class score; boxes are normalized cx, cy, w, h."""
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4) or logits.shape[1] == 0:
        raise ValueError('expected aligned B,Q,C logits and B,Q,4 boxes')
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise ValueError('prediction values must be finite')
    best = logits.max(-1).argmax(-1)
    index = np.arange(len(logits))
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits[index, best].max(-1), -80, 80)))
    predicted = np.clip(boxes[index, best], 0.0, 1.0)
    if np.any(predicted[:, 2:] <= 0):
        raise ValueError('predicted box width and height must be positive')
    return scores.astype(np.float32), predicted.astype(np.float32)


def pool_predicted_roi(feature_map: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Mean of feature cells inside predicted boxes, with local tiny-box fallback."""
    fmap, boxes = np.asarray(feature_map), np.asarray(boxes)
    if fmap.ndim != 4 or boxes.shape != (len(fmap), 4):
        raise ValueError('expected B,C,H,W feature map and B,4 boxes')
    if not np.isfinite(fmap).all() or not np.isfinite(boxes).all():
        raise ValueError('ROI values must be finite')
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError('predicted box width and height must be positive')
    _, _, height, width = fmap.shape
    ys = (np.arange(height) + .5) / height
    xs = (np.arange(width) + .5) / width
    inside = ((np.abs(xs[None, None, :] - boxes[:, 0, None, None]) <= boxes[:, 2, None, None] / 2)
              & (np.abs(ys[None, :, None] - boxes[:, 1, None, None]) <= boxes[:, 3, None, None] / 2))
    for i in np.flatnonzero(inside.sum((1, 2)) == 0):
        inside[i, np.argmin(abs(ys - boxes[i, 1])), np.argmin(abs(xs - boxes[i, 0]))] = True
    return ((fmap * inside[:, None]).sum((2, 3)) / inside.sum((1, 2))[:, None]).astype(np.float32)


def reuse_compatible(target: Path, meta: dict, frame_ids: list[str]) -> bool:
    if not target.exists():
        return False
    with np.load(target, allow_pickle=False) as cached:
        saved = json.loads(str(cached['meta']))
        if saved != meta:
            raise ValueError(f'incompatible feature cache metadata: {target}; choose a new output path')
        if cached['frame_ids'].tolist() != frame_ids:
            raise ValueError(f'incompatible feature cache frame_ids: {target}')
        expected = {'features': (len(frame_ids), 256), 'inputs': (len(frame_ids), 6),
                    'pred_box': (len(frame_ids), 4), 'observed': (len(frame_ids),),
                    'frame_index': (len(frame_ids),)}
        for key, shape in expected.items():
            values = cached[key]
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f'incompatible feature cache {key}: {target}')
    return True


def save_atomic(target: Path, arrays: dict, meta: dict) -> None:
    """Publish a complete NPZ without ever replacing an existing cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays, meta=json.dumps(meta, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # atomic publication; fails if target already exists
    finally:
        temporary.unlink(missing_ok=True)


def _tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, 'tensors'):
        return value.tensors
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _tensor(item)
            if found is not None:
                return found
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='defaults to DATA_ROOT/pool_manifest.jsonl')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--observed-threshold', type=float, default=.5)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.resolution <= 0 or not 0 <= args.observed_threshold <= 1:
        parser.error('positive batch-size/resolution and observed-threshold in [0,1] required')
    manifest = args.manifest or args.data_root / 'pool_manifest.jsonl'
    rows = load_train_rows(manifest, args.data_root)
    frame_ids = [r['frame_id'] for r in rows]
    meta = dict(schema_version=1, model='rf-detr-nano', resolution=args.resolution,
                checkpoint_sha256=file_sha256(args.checkpoint), manifest_sha256=file_sha256(manifest),
                extraction_sha256=file_sha256(Path(__file__)), rows=len(rows), split='train',
                frame_ids_sha256=hashlib.sha256('\n'.join(frame_ids).encode()).hexdigest(),
                labels_used=False, feature='predicted_box_pool', feature_dim=256,
                input_fields=INPUT_FIELDS, box_format='normalized_cx_cy_w_h',
                observed_threshold=args.observed_threshold,
                observation_rule='highest sigmoid class score >= threshold',
                tiny_roi_rule='nearest feature cell',
                brightness_rule='mean RGB intensity divided by 255')
    if reuse_compatible(args.out, meta, frame_ids):
        print(json.dumps(dict(status='compatible_cache_reused', frames=len(rows), out=str(args.out))), flush=True)
        return 0

    import rfdetr
    import torch
    from PIL import Image

    detector = rfdetr.RFDETRNano(pretrain_weights=str(args.checkpoint), resolution=args.resolution)
    net = detector.model.model.eval()
    captured = {}
    hooks = [net.register_forward_hook(lambda _m, _i, output: captured.__setitem__('out', output)),
             net.backbone.register_forward_hook(lambda _m, _i, output: captured.__setitem__('feat', _tensor(output)))]
    all_features, all_inputs, all_boxes = [], [], []
    started = time.monotonic()
    try:
        for start in range(0, len(rows), args.batch_size):
            images = []
            for row in rows[start:start + args.batch_size]:
                with Image.open(row['path']) as source:
                    images.append(source.convert('RGB'))
            brightness = np.array([np.asarray(im, dtype=np.float32).mean() / 255. for im in images])
            with torch.inference_mode():
                detector.predict(images, threshold=0.)
            output = captured['out']
            if isinstance(output, dict):
                logits, coords = output['pred_logits'], output['pred_boxes']
            else:
                first, second = output[:2]
                coords, logits = (first, second) if first.shape[-1] == 4 else (second, first)
            score, predicted = best_prediction(logits.float().cpu().numpy(), coords.float().cpu().numpy())
            fmap = captured['feat']
            if fmap is None or fmap.ndim != 4 or fmap.shape[1] != 256:
                raise ValueError('RF-DETR backbone hook must return B,256,H,W')
            features = pool_predicted_roi(fmap.float().cpu().numpy(), predicted)
            all_features.append(features)
            all_inputs.append(np.column_stack([score, predicted, brightness]).astype(np.float32))
            all_boxes.append(predicted)
            for im in images:
                im.close()
            if start % (args.batch_size * 50) == 0:
                print(f'[extract] {min(start + len(images), len(rows))}/{len(rows)} '
                      f'{time.monotonic() - started:.0f}s', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    inputs = np.vstack(all_inputs)
    arrays = dict(frame_ids=np.array(frame_ids), features=np.vstack(all_features), inputs=inputs,
                  pred_box=np.vstack(all_boxes), observed=inputs[:, 0] >= args.observed_threshold,
                  frame_index=np.array([r['frame_index'] for r in rows], dtype=np.float64),
                  video_id=np.array([r['video_id'] for r in rows]),
                  group_id=np.array([r['group_id'] for r in rows]))
    save_atomic(args.out, arrays, meta)
    reuse_compatible(args.out, meta, frame_ids)
    print(json.dumps(dict(status='written', frames=len(rows), observed=int(arrays['observed'].sum()),
                          seconds=time.monotonic() - started, out=str(args.out))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
