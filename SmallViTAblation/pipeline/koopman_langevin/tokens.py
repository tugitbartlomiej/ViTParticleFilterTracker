#!/usr/bin/env python3
"""Cache DINOv2 patch tokens of Cataract-LMM tracking clips (label-free).

One NPZ per clip: patch tokens reduced by a fixed seeded random projection
(no fitting, so nothing can leak), the CLS token, frame numbers and the
letterbox geometry needed to map native pixels onto the patch grid.
Annotations are copied next to the tokens in a separate file that only the
validation code opens.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.tokens \
        --clips 8 --stride 5 --out SmallViTAblation/feature_cache/koopman_tokens
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def letterbox(image, size: int):
    """Same aspect-preserving letterbox as pipeline.curation.features."""
    from PIL import Image
    im = image.copy()
    im.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), (124, 116, 104))
    left, top = (size - im.width) // 2, (size - im.height) // 2
    canvas.paste(im, (left, top))
    scale = im.width / image.width
    return canvas, dict(scale=scale, left=left, top=top, size=size)


def clip_records(archive: Path, stride: int, max_frames: int | None):
    with zipfile.ZipFile(archive) as z:
        name = [n for n in z.namelist() if PurePosixPath(n).name == 'annotation.csv']
        if len(name) != 1:
            raise ValueError(f'{archive}: expected one annotation.csv')
        parent = PurePosixPath(name[0]).parent
        rows = list(csv.DictReader(io.StringIO(z.read(name[0]).decode('utf-8-sig'))))
    rows = [r for r in rows if (int(r['frame_number']) - 1) % stride == 0]
    if max_frames:
        rows = rows[:max_frames]
    return parent, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, default=REPOSITORY / 'Dataset/Cataract-LMM')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clips', type=int, default=8, help='number of clips, spread over operation groups')
    ap.add_argument('--clip-names', nargs='*', help='explicit clip stems; overrides --clips')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--token-dim', type=int, default=48)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--encoder', default='dinov2_vits14')
    ap.add_argument('--hub-repo', default='facebookresearch/dinov2:main')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device')
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    audit = json.loads((args.dataset_root / '_audyt_pacjentow.json').read_text(encoding='utf-8'))
    group_of = {v: g for g, videos in audit['grupy_operacji'].items() for v in videos}
    archives = sorted((args.dataset_root / '3_Object_Tracking/annotations').glob('*.zip'))
    if args.clip_names:
        archives = [a for a in archives if a.stem in set(args.clip_names)]
    else:
        seen, chosen = set(), []
        for a in archives:                                   # one clip per operation group first
            g = group_of.get(a.stem)
            if g is not None and g not in seen:
                seen.add(g)
                chosen.append(a)
        step = max(1, len(chosen) // args.clips)
        archives = chosen[::step][:args.clips]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    source = 'local' if Path(args.hub_repo).is_dir() else 'github'
    model = torch.hub.load(args.hub_repo, args.encoder, source=source).eval().to(device)
    patch = model.patch_size
    if args.image_size % patch:
        raise ValueError('image size must be divisible by the patch size')
    grid = args.image_size // patch
    rng = np.random.default_rng(args.seed)
    width = model.embed_dim
    projection = np.linalg.qr(rng.normal(size=(width, width)))[0][:, :args.token_dim].astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for archive in archives:
        target = args.out / f'{archive.stem}.npz'
        if target.exists():
            print(f'[tokens] reuse {target.name}', flush=True)
            continue
        parent, rows = clip_records(archive, args.stride, args.max_frames)
        tokens, cls, geometry = [], [], None
        with zipfile.ZipFile(archive) as z:
            for start in range(0, len(rows), args.batch_size):
                batch = []
                for r in rows[start:start + args.batch_size]:
                    with Image.open(io.BytesIO(z.read(str(parent / r['filename'])))) as im:
                        im = im.convert('RGB')
                        canvas, geo = letterbox(im, args.image_size)
                        geometry = dict(geo, native_width=im.width, native_height=im.height)
                    a = (np.asarray(canvas, dtype=np.float32) / 255 - MEAN) / STD
                    batch.append(torch.from_numpy(a.transpose(2, 0, 1)))
                x = torch.stack(batch).to(device)
                with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type,
                                                           enabled=torch.device(device).type == 'cuda'):
                    out = model.forward_features(x)
                tokens.append((out['x_norm_patchtokens'].float().cpu().numpy() @ projection).astype(np.float16))
                cls.append(out['x_norm_clstoken'].float().cpu().numpy().astype(np.float16))
        np.savez(target, tokens=np.concatenate(tokens), cls=np.concatenate(cls),
                 frame_numbers=np.array([int(r['frame_number']) for r in rows]),
                 grid=grid, geometry=json.dumps(geometry), group_id=str(group_of.get(archive.stem)),
                 video_id=archive.stem, stride=args.stride)
        keep = ('frame_number', 'cystotome_bbox', 'cystotome_tip', 'cystotome_vel_abs', 'cystotome_vel_rel',
                'cornea_center', 'pupil_center')
        (args.out / f'{archive.stem}.labels.json').write_text(
            json.dumps([{k: r.get(k, '') for k in keep} for r in rows]), encoding='utf-8')
        print(f'[tokens] {archive.stem}: {len(rows)} frames, grid {grid}x{grid}, '
              f'{time.perf_counter() - started:.0f} s', flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        encoder=args.encoder, hub_repo=args.hub_repo, image_size=args.image_size, token_dim=args.token_dim,
        projection='seeded_random_orthogonal_no_fitting', seed=args.seed, stride=args.stride,
        clips=[a.stem for a in archives], labels_read_by_extractor=False,
        labels_copied_for_validation_only=True), indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
