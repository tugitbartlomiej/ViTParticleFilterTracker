"""Resumable, image-only descriptors; no GT crops, SAM or supervised difficulty.

DINOv2: https://arxiv.org/abs/2304.07193
Implementation: https://github.com/facebookresearch/dinov2
Simple features are explicitly handcrafted, never labelled as DINO embeddings.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .data import ImageSource, digest, write_json

QUALITY_NAMES = ('laplacian_variance', 'mean_luminance', 'dark_fraction', 'bright_fraction')


def quality_and_descriptor(image):
    from PIL import Image
    small = image.copy()
    small.thumbnail((256, 256), Image.Resampling.BILINEAR)
    gray = np.asarray(small.convert('L'), dtype=np.float32)/255
    if min(gray.shape) < 3:
        raise ValueError('Image too small for quality measurements')
    lap = (gray[:-2, 1:-1]+gray[2:, 1:-1]+gray[1:-1, :-2]
           +gray[1:-1, 2:]-4*gray[1:-1, 1:-1])
    quality = np.array([lap.var(), gray.mean(), (gray < .02).mean(),
                        (gray > .98).mean()], dtype=np.float32)
    descriptor = np.asarray(small.resize((8, 8)), dtype=np.float32).reshape(-1)/255
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
    return quality, descriptor


class DinoEncoder:
    def __init__(self, model, hub_repo, weights, image_size, device, amp):
        self.name, self.repo, self.weights = model, hub_repo, weights
        self.size, self.device, self.amp = image_size, device, amp
        self.model = None
        self.identity = None

    def initialise(self):
        if self.model is not None:
            return
        import torch
        if not self.name.startswith('dinov2_'):
            raise ValueError('This adapter supports DINOv2 only; no silent family fallback')
        source = 'local' if Path(self.repo).is_dir() else 'github'
        kwargs = dict(source=source, pretrained=self.weights is None)
        model = torch.hub.load(self.repo, self.name, **kwargs)
        if self.weights:
            state = torch.load(self.weights, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
        sha = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            sha.update(key.encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        precision = 'amp_fp16' if self.amp and torch.device(self.device).type == 'cuda' else 'fp32'
        self.identity = dict(model=self.name, hub_repo=self.repo, weights_sha256=sha.hexdigest(),
                             torch_version=torch.__version__, image_size=self.size,
                             output='l2_cls_concat_l2_max_patch', precision=precision)
        if self.size % model.patch_size:
            raise ValueError(f'Image size must be divisible by patch size {model.patch_size}')
        self.model = model.eval().to(self.device)

    def batch(self, images):
        import torch
        from PIL import Image
        self.initialise()
        tensors = []
        for image in images:
            # Letterbox preserves aspect ratio and the entire field of view.
            im = image.copy()
            im.thumbnail((self.size, self.size), Image.Resampling.BICUBIC)
            canvas = Image.new('RGB', (self.size, self.size), (124, 116, 104))
            canvas.paste(im, ((self.size-im.width)//2, (self.size-im.height)//2))
            a = np.asarray(canvas, dtype=np.float32)/255
            a = (a-np.array([.485, .456, .406]))/np.array([.229, .224, .225])
            tensors.append(torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)))
        batch = torch.stack(tensors).to(self.device)
        use_amp = self.amp and torch.device(self.device).type == 'cuda'
        with torch.inference_mode(), torch.autocast(device_type=torch.device(self.device).type,
                                                   enabled=use_amp):
            output = self.model.forward_features(batch)
            cls = torch.nn.functional.normalize(output['x_norm_clstoken'].float(), dim=1)
            patches = torch.nn.functional.normalize(output['x_norm_patchtokens'].float().amax(1), dim=1)
            features = torch.cat((cls, patches), 1)/math.sqrt(2)
        return features.cpu().numpy()


def build_features(rows, source: ImageSource, cache_root: Path, *, semantic: bool,
                   model='dinov2_vits14', hub_repo='facebookresearch/dinov2:main',
                   weights=None, image_size=224, device=None, amp=True, batch_size=64):
    """Cache identity includes image source stats, ordered IDs and extractor code.

    Successful chunks survive interruption. A changed encoder on resume is rejected.
    Cache directories must not be written concurrently by multiple processes.
    """
    from contextlib import contextmanager, nullcontext
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    stat_cache = {}
    identity_rows = []
    for row in rows:
        path = row.get('archive') or row['path']
        if path not in stat_cache:
            st = Path(path).stat()
            stat_cache[path] = [st.st_size, st.st_mtime_ns]
        identity_rows.append([row['frame_id'], path, row.get('member'), stat_cache[path]])
    config = dict(semantic=semantic, model=model if semantic else 'rgb_grid_8x8',
                  hub_repo=hub_repo if semantic else None, image_size=image_size,
                  amp=amp, batch_size=batch_size,
                  weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights else None,
                  implementation_sha256=digest({name: hashlib.sha256(
                      Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('features.py', 'data.py')}))
    cache_id = digest(dict(images=identity_rows, extractor=config))
    directory = cache_root/cache_id
    directory.mkdir(parents=True, exist_ok=True)
    encoder = DinoEncoder(model, hub_repo, weights, image_size, device, amp) if semantic else None
    started = time.perf_counter()
    @contextmanager
    def ownership():
        lock = directory/'writer.lock'
        try:
            handle = lock.open('x')
        except FileExistsError as exc:
            raise RuntimeError(f'Cache in use or interrupted: {lock}; check its owner before removing it') from exc
        import os
        import socket
        with handle:
            handle.write(f'{socket.gethostname()} pid={os.getpid()}\n')
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)
    all_x, all_quality, hashes = [], [], []
    reused = 0
    # Finished caches can be read concurrently by independent selection seeds.
    read_only = ((directory/'manifest.json').is_file() and (directory/'encoder.json').is_file()
                 and all((directory/f'{i:09d}.npz').is_file()
                         for i in range(0, len(rows), batch_size)))
    context = nullcontext() if read_only else ownership()
    with context:
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start+batch_size]
            ids = [r['frame_id'] for r in subset]
            chunk_path = directory/f'{start:09d}.npz'
            if chunk_path.exists():
                with np.load(chunk_path, allow_pickle=False) as chunk:
                    if chunk['frame_ids'].tolist() != ids:
                        raise ValueError(f'Cache ID mismatch: {chunk_path}')
                    x, quality = chunk['features'].copy(), chunk['quality'].copy()
                    chunk_hashes = chunk['pixel_hashes'].tolist()
                reused += len(subset)
            else:
                images = [source.image(r) for r in subset]
                for row, im in zip(subset, images):
                    if im.size != (row['native_width'], row['native_height']):
                        raise ValueError(f'Image/annotation dimensions disagree: {row["frame_id"]}')
                measured = [quality_and_descriptor(im) for im in images]
                quality = np.stack([v[0] for v in measured])
                x = encoder.batch(images) if encoder else np.stack([v[1] for v in measured])
                chunk_hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
                identity_path = directory/'encoder.json'
                identity = encoder.identity if encoder else {'model': 'rgb_grid_8x8', 'semantic': False}
                if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Encoder identity changed during cache resume; choose a new cache')
                write_json(identity_path, identity)
                tmp = chunk_path.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez(f, frame_ids=np.array(ids), features=x.astype(np.float32),
                             quality=quality, pixel_hashes=np.array(chunk_hashes))
                tmp.replace(chunk_path)
            if (x.ndim != 2 or len(x) != len(ids) or quality.shape != (len(ids), 4)
                    or len(chunk_hashes) != len(ids)):
                raise ValueError(f'Invalid cache shape: {chunk_path}')
            if not np.isfinite(x).all() or not np.isfinite(quality).all():
                raise ValueError(f'Nonfinite cache values: {chunk_path}')
            all_x.append(x)
            all_quality.append(quality)
            hashes.extend(chunk_hashes)
            print(f'[features] {min(start+batch_size, len(rows))}/{len(rows)}; reused={reused}', flush=True)
        meta = dict(cache_id=cache_id, config=config, rows=len(rows), reused_rows=reused,
                    wall_seconds=time.perf_counter()-started, quality_columns=QUALITY_NAMES,
                    encoder=json.loads((directory/'encoder.json').read_text()),
                    image_identity='path/member/size/mtime; pixel SHA256 stored per row')
        if not read_only:
            write_json(directory/'manifest.json', meta)
    return np.vstack(all_x), np.vstack(all_quality), hashes, meta
