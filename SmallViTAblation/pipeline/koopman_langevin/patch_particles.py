#!/usr/bin/env python3
"""Particle recorder for the detector backbone: one particle = one PATCH TOKEN (grant theory, T6).

Follows `Research/DiagnosticTool/test/02_rejestrator.md` and the theory in
`Research/DiagnosticTool/teoria/` of the grant project:

  scaffolding   read from the WEIGHTS of the final LayerNorm of the DINOv2 backbone inside
                RF-DETR: core = beta (T2), lengths = |gamma| (T3a), sign of gamma = sense of the axis
  particle      patch token z after that LayerNorm, at a fixed place of the image grid (T15a:
                Euler description, the instrument flows THROUGH the particle)
  state         profile a = (z - beta) / |gamma| (T25); its radius / sqrt(K) is the loudness of the
                token before normalisation (T24c) and the only radial degree of freedom
  guards        dead direction sign(gamma)/sqrt(K): the profile has no component along it (T3c);
                determinism: 50 frames encoded twice must agree to the bit (T35)
  base          PCA fitted ONCE on frames spread over the whole recording, then frozen and saved;
                without it the reduced coordinates are not reproducible
  raw record    no smoothing, no averaging over particles; reduced profile float32 on every
                frame, full profile float16 every N frames as a control of the reduction

Every recording declares its regime: inference, diagnostic model, super-step = video frame,
input inside the training distribution (Cataract-LMM clips) unless stated otherwise.
Labels are never read here. The predicted box of the detector is saved as a label-free grid
mask; the ground-truth mask is built in the analysis, for scoring only.

Needs `rfdetr`, so it runs on the cluster:

    python -P patch_particles.py --package <dir with frames/ and tracks.json> \
        --checkpoint best.pth --tag b2500 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def find_windowed_backbone(net):
    """The HF-style DINOv2 backbone with `layernorm` and `feature_maps` output, wherever it sits."""
    for name, module in net.named_modules():
        if type(module).__name__ == 'WindowedDinov2WithRegistersBackbone':
            return name, module
    raise RuntimeError('windowed DINOv2 backbone not found in the detector')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--checkpoint', help='weights file; omit for the COCO weights the training started from')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--dim', type=int, default=32, help='reduced dimension r of the frozen PCA base')
    ap.add_argument('--full-every', type=int, default=5, help='full float16 profile every N frames')
    ap.add_argument('--base-frames', type=int, default=512)
    ap.add_argument('--base-particles', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)

    import rfdetr
    import torch
    from PIL import Image

    kwargs = dict(resolution=args.resolution)
    if args.checkpoint:
        kwargs['pretrain_weights'] = str(args.checkpoint)
    detector = rfdetr.RFDETRNano(**kwargs)
    net = detector.model.model.eval()
    path, backbone = find_windowed_backbone(net)
    gamma = backbone.layernorm.weight.detach().float().cpu().numpy()
    beta = backbone.layernorm.bias.detach().float().cpu().numpy()
    lengths, sign = np.abs(gamma), np.sign(gamma)
    K = len(gamma)
    dead = sign / np.sqrt(K)                                    # (T3c) profile has no component here
    captured: dict = {}
    def keep_last_stage(module, inputs, output):
        maps = output.feature_maps if hasattr(output, 'feature_maps') else output[0]   # tuple when return_dict=False
        captured['fm'] = maps[-1]
    backbone.register_forward_hook(keep_last_stage)
    net.register_forward_hook(lambda m, i, o: captured.__setitem__('out', o))
    net.class_embed.register_forward_pre_hook(lambda m, i: captured.__setitem__('hs', i[0]))

    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    frames = [(clip, n, args.package / 'frames' / clip / f'{n:08d}.jpg') for clip, t in tracks.items() for n in t['frame']]
    stride = int(next(iter(tracks.values()))['stride'])

    def encode(files):
        images = [Image.open(f).convert('RGB') for f in files]
        sizes = np.array([[im.width, im.height] for im in images], dtype=np.float32)
        with torch.inference_mode():
            detector.predict(images, threshold=0.0)
        fm = captured['fm'].float()                              # B, K, H, W  (after the final LayerNorm)
        b, k, h, w = fm.shape
        z = fm.permute(0, 2, 3, 1).reshape(b, h * w, k).cpu().numpy()
        profile = (z - beta) / lengths                           # (T25)
        out = captured['out']
        if isinstance(out, dict):
            logits, coords = out['pred_logits'], out['pred_boxes']
        else:
            first, second = out[0], out[1]
            coords, logits = (first, second) if first.shape[-1] == 4 else (second, first)
        n_q = min(captured['hs'].shape[-2], logits.shape[1])
        score, best = logits[:, :n_q].float().sigmoid().max(-1).values.max(-1)
        c = coords[torch.arange(b, device=coords.device), best].float().clamp(0, 1).cpu().numpy()
        box = np.stack([(c[:, 0] - c[:, 2] / 2) * sizes[:, 0], (c[:, 1] - c[:, 3] / 2) * sizes[:, 1],
                        c[:, 2] * sizes[:, 0], c[:, 3] * sizes[:, 1]], 1)
        ys = (np.arange(h) + .5) / h
        xs = (np.arange(w) + .5) / w
        inside = ((xs[None, None, :] - c[:, 0, None, None]) ** 2 <= (c[:, 2, None, None] / 2) ** 2) & \
                 ((ys[None, :, None] - c[:, 1, None, None]) ** 2 <= (c[:, 3, None, None] / 2) ** 2)
        return profile, (h, w), box, score.cpu().numpy(), inside.reshape(b, h * w)

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    # ---- pass 1: frozen PCA base on frames spread over the whole recording
    pick = np.linspace(0, len(frames) - 1, min(args.base_frames, len(frames))).round().astype(int)
    sample = []
    for s in range(0, len(pick), args.batch_size):
        profile, grid, _, _, _ = encode([frames[i][2] for i in pick[s:s + args.batch_size]])
        for a in profile:
            sample.append(a[rng.choice(a.shape[0], min(args.base_particles, a.shape[0]), replace=False)])
    x = np.concatenate(sample).astype(np.float64)
    centre = x.mean(0)
    _, svals, vt = np.linalg.svd(x - centre, full_matrices=False)
    var = svals ** 2 / (len(x) - 1)
    kier = vt[:args.dim].T                                       # (K, r)
    share = float(var[:args.dim].sum() / var.sum())
    print(f'[czastki] {args.tag}: baza PCA z {len(x)} profili, r={args.dim}, udzial wariancji {100 * share:.1f}%  ({time.perf_counter() - started:.0f}s)', flush=True)

    # ---- pass 2: raw record
    out_dir = args.out / f'CZASTKI_{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    red, cen, radius, deadshare, boxes, scores, masks, ids = [], [], [], [], [], [], [], []
    full, full_ids = [], []
    for s in range(0, len(frames), args.batch_size):
        chunk = frames[s:s + args.batch_size]
        profile, grid, box, score, inside = encode([f[2] for f in chunk])
        a = profile.astype(np.float64)
        red.append(((a - centre) @ kier).astype(np.float32))
        cen.append(red[-1].mean(1))
        radius.append((np.linalg.norm(a, axis=2) / np.sqrt(K)).astype(np.float16))
        deadshare.append((np.abs(a @ dead) / np.maximum(np.linalg.norm(a, axis=2), 1e-9)).mean(1))
        boxes.append(box); scores.append(score); masks.append(inside)
        for j, (clip, n, _) in enumerate(chunk):
            ids.append(f'cataract_lmm/tracking/{clip}/{n:08d}')
            if (s + j) % args.full_every == 0:
                full.append(profile[j].astype(np.float16)); full_ids.append(len(ids) - 1)
        if (s // args.batch_size) % 25 == 0:
            print(f'[czastki] {args.tag} {s}/{len(frames)} {time.perf_counter() - started:.0f}s', flush=True)
    profil_red = np.concatenate(red)
    # ---- determinism guard (T35): 50 random frames encoded again
    check = rng.choice(len(frames), 50, replace=False)
    profile, _, _, _, _ = encode([frames[i][2] for i in check])
    again = ((profile.astype(np.float64) - centre) @ kier).astype(np.float32)
    determinism = float(np.abs(again - profil_red[check]).max())

    np.savez_compressed(out_dir / 'czesc000.npz', profil_red=profil_red, centroid_red=np.concatenate(cen).astype(np.float32),
                        promien=np.concatenate(radius), udzial_martwy=np.concatenate(deadshare).astype(np.float32),
                        pred_box=np.concatenate(boxes).astype(np.float32), score=np.concatenate(scores).astype(np.float32),
                        maska_pred=np.concatenate(masks), frame_ids=np.array(ids),
                        profil_pelny=np.stack(full), profil_pelny_nr=np.array(full_ids))
    np.savez_compressed(out_dir / 'baza.npz', srodek=centre, kier=kier, war=var[:args.dim], war_calkowita=var.sum(),
                        rusztowanie_jadro=beta, rusztowanie_dlugosci=lengths, znak_gamma=sign)
    sha = None
    if args.checkpoint:
        with open(args.checkpoint, 'rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    r_all = np.concatenate(radius).astype(np.float32)
    meta = dict(tag=args.tag, checkpoint=str(args.checkpoint), checkpoint_sha256=sha, backbone_module=path,
                resolution=args.resolution, grid=list(grid), K=int(K), r=args.dim, frames=len(ids),
                particles_per_frame=int(grid[0] * grid[1]), stride_frames=stride, full_every=args.full_every,
                base=dict(frames=int(len(pick)), particles_per_frame=args.base_particles, variance_share=share),
                guards=dict(radius_median=float(np.median(r_all)), radius_p05=float(np.percentile(r_all, 5)),
                            radius_p95=float(np.percentile(r_all, 95)), dead_direction_share_mean=float(np.mean(np.concatenate(deadshare))),
                            determinism_max_abs_diff=determinism),
                teoria=dict(stan='profil (z - jadro)/L, T25', rusztowanie='wagi koncowej normy backbone (T2, T3a)',
                            czastka='token-patch, opis Eulera (T6, T15a)', rezim='inferencja', model='diagnostyczny',
                            skala_czasu='nadkrok (klatka filmu)', rozklad_wejscia='w rozkladzie uczacym (Cataract-LMM)'),
                labels_used=False)
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(tag=args.tag, frames=len(ids), grid=list(grid), K=int(K), guards=meta['guards'], out=str(out_dir))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
