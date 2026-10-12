"""Czy pojedyncze patche niosą sygnał doboru? Pooling przestrzenny unieważnia test.

Zapisuje jeden wektor na rzeczywisty patch ostatniej warstwy ViT przed
projektorem RF-DETR. Stała projekcja losowa redukuje KANAŁY, nigdy patche.
To siatka Eulera: ten sam indeks oznacza miejsce obrazu, nie śledzony obiekt.
Dla diagnostyki checkpointów można jawnie wskazać split val; selektor odmawia
użycia takiego cache. Etykiety nie wchodzą do tej ekstrakcji.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .dane import read_jsonl, sha256, write_json
from .ekstrakcja import best_prediction


def project_patches(feature_map, projection):
    feature_map = np.asarray(feature_map)
    if feature_map.ndim != 4 or projection.shape[0] != feature_map.shape[1]:
        raise ValueError('expected B,C,H,W raw patch map and C,D projection')
    b, c, h, w = feature_map.shape
    # Flatten only the two spatial axes. There is no spatial reduction.
    return feature_map.transpose(0, 2, 3, 1).reshape(b, h*w, c) @ projection


def extract(data_root, manifest, checkpoint, out, *, batch_size=32, resolution=384,
            dim=24, seed=42, splits=('train',)):
    data_root, manifest, checkpoint, out = map(Path, (data_root, manifest, checkpoint, out))
    if out.exists():
        raise FileExistsError(f'new immutable cache required: {out}')
    if min(batch_size, resolution, dim) <= 0 or not set(splits) <= {'train', 'val'}:
        raise ValueError('positive dimensions and explicit train/val splits required')
    rows, seen = [], set()
    for r in read_jsonl(manifest):
        if r.get('split') not in splits:
            continue
        fid = str(r['frame_id'])
        path = (data_root / r['path']).resolve()
        if fid in seen or not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise ValueError(f'invalid or duplicated input image: {fid}')
        seen.add(fid)
        rows.append({k: r[k] for k in ('frame_id', 'video_id', 'group_id', 'frame_index', 'split', 'path')})
    rows.sort(key=lambda r: (str(r['video_id']), float(r['frame_index']), str(r['frame_id'])))
    if not rows:
        raise ValueError('no requested rows')
    partial = out.with_name(out.name+'.partial')
    partial.mkdir(parents=True, exist_ok=False)
    import torch
    import rfdetr
    from PIL import Image
    detector = rfdetr.RFDETRNano(pretrain_weights=str(checkpoint), resolution=resolution)
    net = detector.model.model.eval()
    # Joiner[0] is Backbone; its encoder emits raw per-layer ViT feature maps.
    encoder = net.backbone[0].encoder
    captured = {}
    def capture_raw(_module, _inputs, output):
        maps = output if isinstance(output, (list, tuple)) else output.feature_maps
        captured['raw'] = maps[-1]
    handles = [encoder.register_forward_hook(capture_raw),
               net.register_forward_hook(lambda _m, _i, output: captured.__setitem__('out', output))]
    token_store = None
    all_inputs = []
    start_time = time.monotonic()
    try:
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start+batch_size]:
                with Image.open(data_root / row['path']) as original:
                    images.append(original.convert('RGB'))
            brightness = np.array([np.asarray(im, dtype=np.float32).mean()/255 for im in images])
            with torch.inference_mode():
                detector.predict(images, threshold=0.)
            output = captured['out']
            if isinstance(output, dict):
                logits, boxes = output['pred_logits'], output['pred_boxes']
            else:
                a, b = output[:2]
                boxes, logits = (a, b) if a.shape[-1] == 4 else (b, a)
            confidence, boxes = best_prediction(logits.float().cpu().numpy(), boxes.float().cpu().numpy())
            raw = captured['raw']
            if raw.ndim != 4:
                raise ValueError(f'raw ViT feature map must be B,C,H,W, got {raw.shape}')
            b, c, h, w = raw.shape
            if token_store is None:
                projection = np.random.default_rng(seed).normal(size=(c, dim)).astype(np.float32)/np.sqrt(dim)
                np.save(partial/'projection.npy', projection)
                token_store = np.lib.format.open_memmap(partial/'tokens.npy', mode='w+', dtype='float16',
                                                        shape=(len(rows), h*w, dim))
                grid_shape, channels = [h, w], c
            if [h, w] != grid_shape or c != channels:
                raise ValueError('patch grid/channel count changed within cache')
            # Project on GPU to avoid transferring hundreds of channels per patch.
            tensor_projection = torch.as_tensor(projection, device=raw.device, dtype=raw.dtype)
            values = (raw.permute(0, 2, 3, 1).reshape(b, h*w, c) @ tensor_projection).float().cpu().numpy()
            if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
                raise ValueError('nonfinite or overflowing token values')
            token_store[start:start+b] = values
            all_inputs.append(np.column_stack((confidence, boxes, brightness)).astype(np.float32))
            for im in images:
                im.close()
            if start % (batch_size*50) == 0:
                print(f'[patches] {start+b}/{len(rows)} grid={h}x{w} raw={c} elapsed={time.monotonic()-start_time:.0f}s', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    token_store.flush()
    del token_store
    np.save(partial/'inputs.npy', np.vstack(all_inputs))
    with (partial/'rows.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    meta = dict(schema_version=2, feature='last_raw_vit_patch', particle='one spatial patch token',
                spatial_pooling=False, temporal_correspondence='fixed image-grid cell, Eulerian',
                grid_shape=grid_shape, raw_channels=channels, projected_channels=dim,
                projection_seed=seed, resolution=resolution, rows=len(rows), splits=list(splits), labels_used=False,
                checkpoint_sha256=sha256(checkpoint), manifest_sha256=sha256(manifest),
                code_sha256=sha256(Path(__file__)), projection_sha256=sha256(partial/'projection.npy'),
                tokens_sha256=sha256(partial/'tokens.npy'), elapsed_seconds=time.monotonic()-start_time)
    write_json(partial/'meta.json', meta)
    os.rename(partial, out)
    print(json.dumps(meta), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'checkpoint', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--splits', default='train')
    args = parser.parse_args()
    extract(args.data_root, args.manifest or args.data_root/'pool_manifest.jsonl', args.checkpoint, args.out,
            batch_size=args.batch_size, splits=tuple(args.splits.split(',')))


if __name__ == '__main__':
    main()
