"""Bounded, inference-only reconstruction of saved training checkpoints.

Public API: NaturalConfig.validate(), discover(), load_logs(), run() -> Path.
The v2 timeline uses actual stored training steps (including nonzero starts).
Only XYZ plus the exact full-dimensional distance from the first selected
checkpoint are retained. A private baseline memmap is deleted on every exit.
Callbacks receive phase, fraction, message and actual step/image counters;
cancellation returns a directory with status=cancelled and completed frames.
Failures preserve completed frames, record status=failed and raise.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

try:
    from . import ruch_zbioru as shared
except ImportError:
    import ruch_zbioru as shared

ROOT = shared.ROOT
DEFAULT_CHECKPOINT = shared.DEFAULT_CHECKPOINT
DEFAULT_CHECKPOINT_DIR = ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi'
DEFAULT_LOGS = DEFAULT_CHECKPOINT_DIR.parent / 'dane'
OUTPUT_ROOT = shared.OUTPUT_ROOT
TOKENS, DIM = shared.TOKENS, shared.DIM
METRIC_COLUMNS = ['confidence', 'iou', 'delta_l2', 'training_step', 'both90']


def _range(start_step, end_step, checkpoint_stride):
    for name, value, minimum in [('start_step', start_step, 0), ('end_step', end_step, 0),
                                  ('checkpoint_stride', checkpoint_stride, 1)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if end_step < start_step:
        raise ValueError('end_step must be >= start_step')


@dataclass
class NaturalConfig:
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    manifest: str = str(shared.DEFAULT_MANIFEST)
    image_root: str = ''
    max_images: int = 1500
    start_step: int = 0
    end_step: int = 1000
    checkpoint_stride: int = 10
    batch_size: int = 16
    device: str = 'cuda'
    seed: int = 42
    logs: str = ''
    reference_checkpoint: str = str(DEFAULT_CHECKPOINT)
    frame_format: str = 'q16'           # 'q16': 16 bitów w granicach kroku (połowa miejsca), 'f32': dawny float32

    def validate(self):
        _range(self.start_step, self.end_step, self.checkpoint_stride)
        for name in ('max_images', 'batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if not isinstance(self.device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', self.device):
            raise ValueError('device must be cpu, cuda or cuda:N')
        if self.frame_format not in shared.BAJTY_NA_CZASTKE:
            raise ValueError('frame_format must be q16 or f32')
        if not Path(self.checkpoint_dir).is_dir():
            raise FileNotFoundError(self.checkpoint_dir)
        if not Path(self.manifest).is_file():
            raise FileNotFoundError(self.manifest)
        for name in ('image_root', 'logs', 'reference_checkpoint'):
            value = getattr(self, name)
            if value and not Path(value).exists():
                raise FileNotFoundError(value)
        if self.image_root and not Path(self.image_root).is_dir():
            raise ValueError('image_root must be a directory')
        if self.reference_checkpoint and not Path(self.reference_checkpoint).is_file():
            raise ValueError('reference_checkpoint must be a file; empty explicitly disables the check')
        return self


def _candidates(checkpoint_dir):
    directory = Path(checkpoint_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    # Never recurse into old results or choose a different model automatically.
    for path in directory.iterdir():
        match = re.fullmatch(r'k([0-9]+)\.pt', path.name)
        if match and path.is_file():
            step = int(match[1])
            if step in result:
                raise ValueError(f'duplicate checkpoint step {step}')
            result[step] = path
    return dict(sorted(result.items()))


def _payload(path, expected_sha=None, expected_step=None):
    import torch
    before = shared._sha(path)
    if expected_sha is not None and before != expected_sha:
        raise ValueError(f'checkpoint changed: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if shared._sha(path) != before:
        raise ValueError(f'checkpoint changed during load: {path}')
    if (not isinstance(payload, dict) or not isinstance(payload.get('net'), dict)
            or not payload['net'] or type(payload.get('krok')) is not int or payload['krok'] < 0):
        raise ValueError(f'expected saved net and explicit integer krok: {path}')
    if expected_step is not None and payload['krok'] != expected_step:
        raise ValueError(f'filename/stored krok mismatch: {path}')
    if any(not isinstance(v, torch.Tensor) or (v.is_floating_point() and not torch.isfinite(v).all())
           for v in payload['net'].values()):
        raise ValueError(f'invalid/nonfinite checkpoint tensors: {path}')
    return payload, before


def _record(path, step):
    payload, digest = _payload(path, expected_step=step)
    del payload
    return {'path': str(path), 'step': step, 'sha256': digest, 'size_bytes': path.stat().st_size}


def _discover(checkpoint_dir, start_step, end_step, checkpoint_stride, cancel, progress):
    _range(start_step, end_step, checkpoint_stride)
    paths = _candidates(checkpoint_dir)
    selected = [(step, path) for step, path in paths.items()
                if start_step <= step <= end_step and
                ((step-start_step) % checkpoint_stride == 0 or step == end_step)]
    if not selected or selected[0][0] != start_step:
        raise ValueError('no saved checkpoint at the requested baseline start_step')
    result = []
    for i, (step, path) in enumerate(selected):
        shared._check_cancel(cancel)
        result.append(_record(path, step))
        progress({'phase': 'discover', 'checkpoint': i+1, 'checkpoint_count': len(selected), 'step': step})
    return result


def discover(checkpoint_dir, start_step=0, end_step=1000, checkpoint_stride=10):
    """Numerically sorted measured records; stride is anchored at start_step.

    A saved off-stride end_step is included. Gaps remain gaps, with no invented
    state. The baseline must exist. Every selected payload krok must match its
    filename. Content hashes are checked again when loaded by run().
    """
    return _discover(checkpoint_dir, start_step, end_step, checkpoint_stride,
                     lambda: False, lambda event: None)


def load_logs(path):
    """Normalize explicit-step JSON/JSONL logs; no epoch or row-index conversion.

    A directory selects only LOG.json and OCENY.json. Missing/unknown formats
    produce explicit warnings. Ambiguous steps, duplicate metric steps and
    nonfinite numeric values fail. Each source hash covers the bytes parsed.
    """
    result = {'series': {k: {'steps': [], 'values': []} for k in ('loss', 'map50')},
              'files': [], 'warnings': []}
    if not path:
        result['warnings'].append('No training logs selected.')
        return result
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    files = [source] if source.is_file() else [source / n for n in ('LOG.json', 'OCENY.json')
                                                       if (source / n).is_file()]
    if not files:
        result['warnings'].append(f'No LOG.json or OCENY.json in {source}.')
    values = {k: {} for k in result['series']}
    step_keys = ('krok', 'step', 'training_step', 'global_step')
    aliases = {'loss': ('strata', 'loss'), 'map50': ('mAP50', 'map50', 'mAP@0.5')}
    known = set(step_keys) | {a for names in aliases.values() for a in names} | {
        'heat', 'box', 's_na_krok', 'q_sond', 'q_puste'}

    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite training log value')
        if isinstance(value, dict):
            for child in value.values(): finite_tree(child)
        elif isinstance(value, list):
            for child in value: finite_tree(child)

    for file in files:
        raw = file.read_bytes()
        result['files'].append({'path': str(file), 'sha256': hashlib.sha256(raw).hexdigest()})
        text = raw.decode('utf-8-sig')
        rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                if file.suffix.lower() == '.jsonl' else json.loads(text))
        finite_tree(rows)
        if isinstance(rows, dict):
            rows = next((rows[k] for k in ('records', 'history', 'log') if isinstance(rows.get(k), list)), None)
        if not isinstance(rows, list):
            result['warnings'].append(f'{file.name}: unknown log format; no series imported.')
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                result['warnings'].append(f'{file.name} row {i}: unknown record; skipped.')
                continue
            unknown = sorted(set(row)-known)
            if unknown:
                result['warnings'].append(f'{file.name} row {i}: ignored fields {unknown}.')
            axes = [row[k] for k in step_keys if k in row]
            if not axes:
                result['warnings'].append(f'{file.name} row {i}: no explicit training step; skipped.')
                continue
            if any(type(v) is not int or v < 0 or v != axes[0] for v in axes):
                raise ValueError(f'{file.name} row {i}: invalid/ambiguous training step')
            for metric, keys in aliases.items():
                measured = [row[k] for k in keys if k in row]
                if not measured: continue
                if any(type(v) not in (int, float) or not math.isfinite(v) or v != measured[0] for v in measured):
                    raise ValueError(f'invalid/ambiguous {metric} at step {axes[0]}')
                value = float(measured[0])
                if metric == 'map50' and not 0 <= value <= 1:
                    raise ValueError('map50 must be in [0,1]')
                if axes[0] in values[metric]:
                    raise ValueError(f'duplicate {metric} step {axes[0]}')
                values[metric][axes[0]] = value
    for metric, entries in values.items():
        steps = sorted(entries)
        result['series'][metric] = {'steps': steps, 'values': [entries[s] for s in steps]}
    return result


@contextmanager
def _fp32(device):
    """Scope only the precision flags changed here; leave device/cache/RNG alone."""
    import torch
    precision = torch.get_float32_matmul_precision()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _estimates(count, frames, batch_size):
    batch_size = min(count, batch_size)
    baseline = count * TOKENS * DIM * 4
    frame_bytes = count * TOKENS * 16
    metrics = count * frames * len(METRIC_COLUMNS) * 8
    return {'baseline_temporary_bytes': baseline, 'frame_count': frames,
            'disk_bytes': baseline + (frames+1)*frame_bytes + 2*metrics + (64 << 20),
            'ram_bytes': (256 << 20) + 4096*DIM*16 + batch_size*TOKENS*DIM*24,
            'cuda_bytes': (128 << 20) + batch_size*(32 << 20),
            'batch_size': batch_size, 'full_latent_states_persisted': False,
            'baseline_policy': 'private .tmp float32 memmap, deleted on every exit'}


def _resources(count, frames, batch_size, out, dev):
    import psutil
    import torch
    estimates = _estimates(count, frames, batch_size)
    if shutil.disk_usage(out).free < estimates['disk_bytes']:
        raise ValueError(f'insufficient disk for temporary baseline and frames: {estimates["disk_bytes"]} bytes')
    if psutil.virtual_memory().available < estimates['ram_bytes']:
        raise ValueError(f'insufficient RAM: {estimates["ram_bytes"]} bytes')
    if dev.type == 'cuda' and torch.cuda.mem_get_info(dev)[0] < estimates['cuda_bytes']:
        raise ValueError(f'insufficient memory on {dev}: {estimates["cuda_bytes"]} bytes')
    return estimates


class _Obrazy:
    """Wszystkie obrazy pomiaru zdekodowane RAZ (SHA256 sprawdzone przy odczycie) jako uint8 [N, 3, H, W].
    Leżą na karcie, gdy jest miejsce (inaczej w RAM). Dawniej każdy z ~101 checkpointów czytał z dysku,
    liczył SHA256 i dekodował od nowa wszystkie obrazy (151 500 dekodowań JPEG), a karta czekała na CPU."""

    def __init__(self, rows, paths, dev, na_karcie, cancel=lambda: False, report=lambda e: None):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        data = shared._private_module('dane')
        self.index = {int(r['index']): i for i, r in enumerate(rows)}
        miejsce = dev if na_karcie else torch.device('cpu')
        self.gt = torch.tensor([r['bbox_xyxy_480'] for r in rows], dtype=torch.float32, device=miejsce)

        def jeden(i):
            raw = paths[i].read_bytes()
            if hashlib.sha256(raw).hexdigest() != rows[i]['image_sha256']:
                raise ValueError(f'image changed during run: {paths[i]}')
            return data.decode_resize(raw)

        self.tensor = None
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:   # cv2 zwalnia GIL
            for i, rgb in enumerate(pool.map(jeden, range(len(rows)))):
                shared._check_cancel(cancel)
                if self.tensor is None:
                    h, w = rgb.shape[:2]
                    self.tensor = torch.empty((len(rows), 3, h, w), dtype=torch.uint8, device=miejsce)
                self.tensor[i] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(miejsce)
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    report({'phase': 'decode', 'image': i + 1, 'image_count': len(rows)})

    def paczka(self, rows):
        idx = [self.index[int(r['index'])] for r in rows]
        a, b = idx[0], idx[-1] + 1
        if idx == list(range(a, b)):
            return self.tensor[a:b], self.gt[a:b]
        return self.tensor[idx], self.gt[idx]


_OBRAZY = None          # ustawiane przez run() na czas pomiaru


def _metryki_wsadowe(output, gt):
    """Te same operacje co core.top1_metrics (sigmoid, lokalne maksima 3 x 3, pierwszy argmax, odległości
    z wyjścia, IoU z 1e-9), ale dla całej paczki naraz i z jedną synchronizacją z kartą zamiast kilku na obraz."""
    import torch
    import torch.nn.functional as F
    stride = float(shared._private_module('pelne_latenty').STRIDE)
    b, height, width, _ = output.shape
    probability = output[..., 0].float().sigmoid()
    maxima = F.max_pool2d(probability[:, None], 3, 1, 1)[:, 0]
    peaks = probability * (probability == maxima)
    token = peaks.reshape(b, -1).argmax(dim=1)
    row, column = token // width, token % width
    wiersze = torch.arange(b, device=output.device)
    distances = output.reshape(b, -1, 5)[wiersze, token, 1:].float().clamp(-20, 20).exp() * stride
    cx = (column.float() + 0.5) * stride
    cy = (row.float() + 0.5) * stride
    box = torch.stack([cx - distances[:, 0], cy - distances[:, 1], cx + distances[:, 2], cy + distances[:, 3]], 1)
    gt = gt.to(device=box.device, dtype=box.dtype)
    x0 = torch.maximum(box[:, 0], gt[:, 0])
    y0 = torch.maximum(box[:, 1], gt[:, 1])
    x1 = torch.minimum(box[:, 2], gt[:, 2])
    y1 = torch.minimum(box[:, 3], gt[:, 3])
    intersection = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    area_box = (box[:, 2] - box[:, 0]).clamp_min(0) * (box[:, 3] - box[:, 1]).clamp_min(0)
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iou = intersection / (area_box + area_gt - intersection + 1e-9)
    confidence = probability.reshape(b, -1)[wiersze, token]
    return torch.stack([confidence, iou], 1).double().cpu().numpy()


def _infer(net, rows, paths, dev, core):
    """One full forward per batch, exact shared preprocessing/hook/GT metrics. Zwraca z jako tensor na urządzeniu
    modelu (bez kopiowania 29 MB na paczkę do RAM) i metryki [B, 2] float64."""
    import torch
    if _OBRAZY is not None:
        obrazy, gt = _OBRAZY.paczka(rows)
    else:                                                    # wywołanie poza run(): jak dawniej, z plików
        data = shared._private_module('dane')
        lista = []
        for row, path in zip(rows, paths):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
                raise ValueError(f'image changed during run: {path}')
            lista.append(torch.from_numpy(np.ascontiguousarray(data.decode_resize(raw))).permute(2, 0, 1))
        obrazy = torch.stack(lista)
        gt = torch.tensor([r['bbox_xyxy_480'] for r in rows], dtype=torch.float32)
    x = obrazy.to(device=dev, dtype=torch.float32) / 255
    output, z, q = core.capture_baseline(net, x)
    if (z.shape != (len(rows), TOKENS, DIM) or z.dtype != torch.float32
            or not bool(torch.isfinite(z).all() & torch.isfinite(output).all())):
        raise ValueError('expected finite fp32 [batch,600,768] post-GELU and detector output')
    metrics = _metryki_wsadowe(output, gt)
    if not (np.isfinite(metrics).all() and ((metrics >= 0) & (metrics <= 1)).all()):
        raise ValueError('nonfinite/out-of-range natural detection metric')
    return z, metrics


def _pca(sample):
    # Same reservoir size, covariance, eigh and sign rule as shared._pca.
    mean = sample.mean(axis=0)
    sample -= mean
    values, vectors = np.linalg.eigh(sample.T @ sample / max(len(sample)-1, 1))
    components = vectors[:, -3:][:, ::-1].copy()
    for k in range(3):
        if components[np.argmax(np.abs(components[:, k])), k] < 0:
            components[:, k] *= -1
    return mean, components, values[-3:][::-1]


def _audit(net, core, before):
    after = core.state_fingerprint(net)
    frozen = all(not p.requires_grad and p.grad is None for p in net.parameters())
    evaluation = all(not module.training for module in net.modules())
    if before != after or not frozen or not evaluation:
        raise RuntimeError('natural inference modified model state/freeze/eval invariant')
    return {'weights_sha256_before': before, 'weights_sha256_after': after,
            'weights_unchanged': True, 'all_parameters_frozen': frozen, 'eval': evaluation}


def _append(path, z, baseline, mean, components):
    """Klatka paczki: rzut (z - mean) @ components i dokładna norma L2 zmiany w pełnych 768D, w float64.
    Liczone tam, gdzie leży z (na karcie przy CUDA); do RAM wraca tylko gotowa klatka [B * 600, 4] float32."""
    import torch
    zt = torch.as_tensor(z)
    dev = zt.device
    array = zt.reshape(-1, DIM).to(torch.float64)
    delta = array - torch.as_tensor(baseline).to(device=dev).reshape(-1, DIM).to(torch.float64)
    mean_t, comp_t = _baza_na(dev, mean, components)
    frame = torch.cat([(array - mean_t) @ comp_t, torch.linalg.vector_norm(delta, dim=1)[:, None]], 1)
    per_image = torch.linalg.vector_norm(delta.reshape(len(zt), -1), dim=1)
    frame = frame.to(torch.float32).cpu().numpy().astype('<f4', copy=False)
    if not np.isfinite(frame).all():
        raise FloatingPointError('nonfinite particle frame')
    with path.open('ab') as f:
        f.write(frame.tobytes())
    return per_image.cpu().numpy()


_BAZA = {}


def _baza_na(dev, mean, components):
    """Średnia i składowe PCA (float64) na urządzeniu z; kopiowane raz na pomiar."""
    import torch
    klucz = (str(dev), id(mean), id(components))
    if klucz not in _BAZA:
        _BAZA.clear()
        _BAZA[klucz] = (torch.as_tensor(mean, dtype=torch.float64, device=dev),
                        torch.as_tensor(components, dtype=torch.float64, device=dev))
    return _BAZA[klucz]


def _klatka_f32(out, metadata, step):
    """Opublikowana klatka kroku jako float32 [N, 4] (q16: odkwantowana)."""
    item = metadata['files'][str(step)]
    if item.get('format', 'f32') == 'q16':
        q = np.fromfile(out / item['name'], dtype='<u2').reshape(-1, 4)
        return shared.odkwantyzuj(q, item['quant'])
    return np.fromfile(out / item['name'], dtype='<f4').reshape(-1, 4)


def _publish(out, record, metadata, metrics):
    step = record['step']
    part = out / f'step_{step:04d}.bin.part'
    if part.stat().st_size != metadata['point_count']*16 or not np.isfinite(metrics).all():
        raise ValueError('cannot publish incomplete dataset frame/metrics')
    fmt = metadata.get('frame_format', 'f32')
    # Klatka kroku w float32 (dokładna): z niej statystyki i granice, potem zapis w formacie pomiaru.
    frame = np.fromfile(part, dtype='<f4').reshape(metadata['point_count'], 4)
    initial = _klatka_f32(out, metadata, metadata['baseline_step']) if metadata['steps'] else frame
    if fmt == 'f32':
        # Windows FlushFileBuffers requires a writable handle.
        with part.open('r+b') as f: os.fsync(f.fileno())
        path = part.with_suffix('')
        os.replace(part, path)
        wpis = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': shared._sha(path), 'format': 'f32'}
    else:
        wpis = shared.zapisz_klatke(out / f'step_{step:04d}.q16', frame, fmt)
        part.unlink()
    try:
        bounds = metadata.setdefault('bounds', {})
        for key in ('natural', 'movement'):
            low, high = np.full(3, np.inf), np.full(3, -np.inf)
            for start in range(0, len(frame), TOKENS*16):
                xyz = frame[start:start+TOKENS*16, :3]
                if key == 'movement': xyz = xyz - initial[start:start+TOKENS*16, :3]
                low, high = np.minimum(low, xyz.min(axis=0)), np.maximum(high, xyz.max(axis=0))
            if key in bounds:
                old = np.asarray(bounds[key])
                low, high = np.minimum(low, old[:, 0]), np.maximum(high, old[:, 1])
            bounds[key] = np.column_stack([low, high]).tolist()
        norm = frame[:, 3]
        both = int(((metrics[:, 0] >= .9) & (metrics[:, 1] >= .9)).sum())
        stats = {'confidence': float(metrics[:, 0].mean()), 'iou': float(metrics[:, 1].mean()),
                 'both90': both, 'norm_median': float(np.median(norm)), 'norm_max': float(norm.max()),
                 'real_step': step}
        metadata['norm_scale'] = max(metadata.get('norm_scale', 1e-6), stats['norm_max'])
    finally:
        del frame, initial
    # Metadata publication is the transaction boundary, after complete binary IO.
    metadata['steps'].append(step)
    metadata['files'][str(step)] = wpis
    metadata['stats'][str(step)] = stats
    metadata['checkpoint_sha256'] = record['sha256']
    metadata['checkpoint_step'] = step
    metadata['completed_images'] = metadata['image_count']
    for key in ('confidence', 'iou', 'both90', 'real_step'):
        metadata['curves'][key].append(stats[key])
    metadata['curves']['steps'].append(step)
    shared._json(out / 'manifest.json', metadata)


def _infer_rzeczywisty():
    """Czy _infer to prawdziwy forward (testy podmieniają go atrapą, która nie czyta obrazów)."""
    return getattr(_infer, '__module__', None) == __name__ and _infer.__name__ == '_infer'


def _magazyn_bazy(count, dev, batch_size):
    """Tensor [count, 600, 768] float32 na stan odniesienia: na karcie, gdy zostaje zapas na obrazy, paczkę
    i resztę aplikacji (np. chmurę w zakładce), inaczej w RAM. Zwraca (tensor, czy obrazy też na karcie)."""
    import torch
    rozmiar = count * TOKENS * DIM * 4
    if dev.type == 'cuda':
        wolne = torch.cuda.mem_get_info(dev)[0]
        obrazy = count * 3 * 320 * 480
        zapas = (1 << 30) + batch_size * (64 << 20)
        if wolne > rozmiar + obrazy + zapas:
            return torch.empty((count, TOKENS, DIM), dtype=torch.float32, device=dev), True
        return torch.empty((count, TOKENS, DIM), dtype=torch.float32), wolne > obrazy + zapas
    return torch.empty((count, TOKENS, DIM), dtype=torch.float32), False


def run(config, progress=lambda event: None, cancel=lambda: False):
    """Save measured frames only; never train, replay, optimize or intervene."""
    global _OBRAZY
    import torch
    config = NaturalConfig(**asdict(config)).validate()
    # Normalize path-like arguments for JSON and freeze GUI configuration.
    for key in ('checkpoint_dir', 'manifest', 'image_root', 'logs', 'reference_checkpoint'):
        setattr(config, key, str(getattr(config, key)))
    out = Path(OUTPUT_ROOT) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_natural_') + uuid.uuid4().hex[:12])
    out.mkdir(parents=True)
    metadata = {'schema_version': 2, 'status': 'preparing', 'steps': [], 'files': {}, 'stats': {},
                'point_count': 0, 'image_count': 0, 'tokens_per_image': TOKENS,
                'time_axis': 'training_checkpoint', 'baseline_step': config.start_step,
                'checkpoint_sha256': None, 'checkpoint_step': None, 'config': asdict(config),
                'interpolation': False, 'full_latent_states_persisted': False,
                'norm': 'L2(z(step)-z(first selected checkpoint)) in full 768D',
                'coordinates': '(z-mean) @ fixed_components; image-major then token-major',
                'partial_policy': 'Only complete selected-image frames listed; .part is an unpublished prefix',
                'precision': 'fp32, autocast disabled, TF32 disabled', 'inference_only': True,
                'frame_format': config.frame_format,
                'metric_columns': METRIC_COLUMNS, 'completed_images': 0,
                'checkpoint_audits': [], 'curves': {'steps': [], 'confidence': [], 'iou': [],
                    'both90': [], 'real_step': [], 'label': 'Full detector evaluation at saved training steps'}}
    shared._json(out / 'config.json', asdict(config))
    shared._json(out / 'manifest.json', metadata)
    baseline = None
    baseline_path = out / 'baseline.float32.tmp'
    telemetry = None
    net = None
    fraction = 0.

    def report(event):
        nonlocal fraction
        phase = event['phase']
        if phase == 'validate': f = .05 + .05*event['image']/event['image_count']
        elif phase in ('baseline', 'measure'):
            f = .15 + .8*(event['checkpoint_index'] + event['image']/event['image_count']) / event['checkpoint_count']
        elif phase == 'finished': f = 1. if event['status'] == 'complete' else fraction
        else: f = fraction
        fraction = min(1., max(fraction, f))
        message = phase
        if 'step' in event: message += f'; training step {event["step"]}'
        if 'image' in event: message += f'; {event["image"]}/{event["image_count"]} images'
        progress(dict(event, fraction=fraction, message=message))

    try:
        records = _discover(config.checkpoint_dir, config.start_step, config.end_step,
                            config.checkpoint_stride, cancel, report)
        metadata['checkpoints'] = records
        candidates = _candidates(config.checkpoint_dir)
        terminal_step = max(candidates)
        terminal = next((r for r in records if r['step'] == terminal_step), None)
        if terminal is None: terminal = _record(candidates[terminal_step], terminal_step)
        reference = {'enabled': bool(config.reference_checkpoint), 'final_checkpoint': terminal, 'matches': None}
        if config.reference_checkpoint:
            payload, sha = _payload(config.reference_checkpoint)
            reference_step = payload['krok']
            if reference_step not in candidates:
                raise ValueError(f'no directory checkpoint at reference stored krok {reference_step}')
            reference_record = next((r for r in records if r['step'] == reference_step), None)
            if reference_record is None:
                reference_record = _record(candidates[reference_step], reference_step)
            reference.update(path=str(Path(config.reference_checkpoint).resolve()), sha256=sha, step=reference_step,
                             checkpoint=reference_record, matches=reference_record['sha256'] == sha)
            del payload
            metadata['reference_verification'] = reference
            if not reference['matches']:
                raise ValueError('directory checkpoint/reference SHA256 mismatch at reference stored krok')
        metadata['reference_verification'] = reference
        logs = config.logs or (Path(config.checkpoint_dir).parent / 'dane')
        metadata['training_logs'] = load_logs(logs if Path(logs).exists() else '')
        audit_dir = Path(config.checkpoint_dir).parent / 'pobranie_wag_2026-10-01'
        metadata['download_audit_sources'] = [{'path': str(p.resolve()), 'sha256': shared._sha(p)}
            for p in sorted(audit_dir.glob('*.json'))] if audit_dir.is_dir() else []
        manifest_sha = shared._sha(config.manifest)
        rows, paths = shared._dataset(config, cancel, report)
        if shared._sha(config.manifest) != manifest_sha:
            raise ValueError('image manifest changed during validation')
        dev = shared._device(config.device)
        estimates = _resources(len(rows), len(records), config.batch_size, out, dev)
        metadata.update(point_count=len(rows)*TOKENS, image_count=len(rows),
                        image_names=[r['image_name'] for r in rows],
                        images=[{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
                        resolved_image_paths=[str(p) for p in paths], device=str(dev), estimates=estimates,
                        manifest_sha256=manifest_sha, backend_sha256=shared._sha(__file__),
                        source_hashes={name: shared._sha(shared.EXPERIMENT / (name+'.py')) for name in
                                       ('model_snapshot', 'pelne_latenty', 'dane')})
        shared._json(out / 'manifest.json', metadata)
        report(dict(phase='estimate', **estimates))
        shared._check_cancel(cancel)
        core, architecture = shared._private_module('pelne_latenty'), shared._private_module('model_snapshot')
        with torch.random.fork_rng(devices=[]):
            # Seed only the CPU constructor; manual_seed() would also alter
            # unrelated CUDA generators not covered by this CPU-only fork.
            torch.random.default_generator.manual_seed(config.seed)
            net = architecture.MalyViT().float()
        net.eval().requires_grad_(False).to(dev)
        indices = np.sort(np.random.default_rng(config.seed).choice(len(rows)*TOKENS, min(4096, len(rows)*TOKENS), replace=False))
        sample = np.empty((len(indices), DIM), dtype=np.float64)
        # Stan odniesienia (pierwszy checkpoint) i obrazy w pamięci karty, gdy się mieszczą (inaczej w RAM):
        # bez memmapu 2,8 GB czytanego z dysku w każdej paczce każdego checkpointu.
        baseline, na_karcie = _magazyn_bazy(len(rows), dev, config.batch_size)
        metadata.update(baseline_location=str(baseline.device), image_cache_location=str(dev) if na_karcie else 'cpu')
        _OBRAZY = _Obrazy(rows, paths, dev, na_karcie, cancel, report) if _infer_rzeczywisty() else None
        # Tiny per-image telemetry, with actual training steps in float64.
        telemetry = np.lib.format.open_memmap(out / 'metrics.partial.npy', mode='w+', dtype='<f8',
                                               shape=(len(rows), len(records), len(METRIC_COLUMNS)))
        telemetry[:] = np.nan
        metadata['metric_steps'] = [r['step'] for r in records]
        metadata['observed_checkpoints_per_image'] = [0]*len(rows)
        with _fp32(dev):
            for j, record in enumerate(records):
                shared._check_cancel(cancel)
                payload, _ = _payload(record['path'], record['sha256'], record['step'])
                net.load_state_dict(payload['net'], strict=True)
                del payload
                before = core.state_fingerprint(net)
                part = out / f'step_{record["step"]:04d}.bin.part'
                metadata['current_step'] = record['step']
                metadata['current_frame_completed_images'] = 0
                try:
                    for start in range(0, len(rows), config.batch_size):
                        shared._check_cancel(cancel)
                        end = min(start+config.batch_size, len(rows))
                        z, measured = _infer(net, rows[start:end], paths[start:end], dev, core)
                        telemetry[start:end, j, :2] = measured
                        telemetry[start:end, j, 3] = record['step']
                        telemetry[start:end, j, 4] = (measured >= .9).all(axis=1)
                        if j == 0:
                            zt = torch.as_tensor(z)
                            baseline[start:end] = zt.to(baseline.device)
                            a, b = np.searchsorted(indices, [start*TOKENS, end*TOKENS])
                            if b > a:
                                wybor = torch.as_tensor(indices[a:b]-start*TOKENS, device=zt.device)
                                sample[a:b] = zt.reshape(-1, DIM)[wybor].double().cpu().numpy()
                            telemetry[start:end, j, 2] = 0.
                        else:
                            telemetry[start:end, j, 2] = _append(part, z, baseline[start:end], mean, components)
                        metadata['observed_checkpoints_per_image'][start:end] = [j+1]*(end-start)
                        metadata['current_frame_completed_images'] = end
                        report({'phase': 'baseline' if j == 0 else 'measure', 'image': end,
                                'image_count': len(rows), 'step': record['step'],
                                'checkpoint_index': j, 'checkpoint_count': len(records)})
                        del z, measured
                    if j == 0:
                        shared._check_cancel(cancel)
                        report({'phase': 'pca', 'step': record['step']})
                        mean, components, eigenvalues = _pca(sample)
                        del sample
                        pca_path = out / 'initial_pca.npz'
                        with pca_path.with_suffix('.npz.tmp').open('wb') as f:
                            np.savez(f, mean=mean, components=components, reservoir_token_indices=indices, eigenvalues=eigenvalues)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(pca_path.with_suffix('.npz.tmp'), pca_path)
                        metadata.update(pca_basis_sha256=hashlib.sha256(mean.astype('<f8').tobytes()+components.astype('<f8').tobytes()).hexdigest(),
                                        pca_file={'name': pca_path.name, 'sha256': shared._sha(pca_path)},
                                        pca_fit='uniform reservoir of first selected checkpoint tokens; no future states',
                                        pca_seed=config.seed, pca_sample_count=len(indices))
                        for start in range(0, len(rows), config.batch_size):
                            shared._check_cancel(cancel)
                            z0 = baseline[start:start+config.batch_size]
                            _append(part, z0, z0, mean, components)
                finally:
                    metadata['checkpoint_audits'].append(dict(step=record['step'], **_audit(net, core, before)))
                if shared._sha(record['path']) != record['sha256']:
                    raise ValueError('checkpoint changed during inference')
                telemetry.flush()
                metadata['status'] = 'running'
                _publish(out, record, metadata, np.array(telemetry[:, j]))
                report({'phase': 'frame', 'step': record['step'], 'completed_checkpoints': j+1,
                        'checkpoint_count': len(records)})
        shared._check_cancel(cancel)
        # Catch changes to already measured sources as well as future reference.
        for record in records + [terminal] + ([reference['checkpoint']] if reference['enabled'] else []):
            shared._check_cancel(cancel)
            if shared._sha(record['path']) != record['sha256']:
                raise ValueError('checkpoint changed before completion')
        if reference['enabled'] and shared._sha(reference['path']) != reference['sha256']:
            raise ValueError('reference checkpoint changed before completion')
        if shared._sha(config.manifest) != manifest_sha:
            raise ValueError('image manifest changed before completion')
        for row, path in zip(rows, paths):
            shared._check_cancel(cancel)
            if shared._sha(path) != row['image_sha256']:
                raise ValueError('image changed before completion')
        for file in metadata['training_logs']['files']:
            if shared._sha(file['path']) != file['sha256']:
                raise ValueError('training logs changed before completion')
        metadata.update(status='complete', weights_unchanged=True)
    except BaseException as exc:
        metadata['status'] = 'cancelled' if isinstance(exc, shared._Cancelled) else 'failed'
        if metadata['status'] == 'failed': metadata['error'] = f'{type(exc).__name__}: {exc}'
        shared._json(out / 'manifest.json', metadata)
        if not isinstance(exc, shared._Cancelled): raise
    finally:
        # Baza i obrazy żyją tylko w pamięci tego pomiaru. Never clear the shared CUDA cache.
        baseline = None
        _OBRAZY = None
        _BAZA.clear()
        baseline_path.unlink(missing_ok=True)
        if telemetry is not None:
            telemetry.flush()
            telemetry._mmap.close()
            os.replace(out / 'metrics.partial.npy', out / 'metrics.npy')
            metadata['metrics_file'] = {'name': 'metrics.npy', 'sha256': shared._sha(out / 'metrics.npy'),
                'shape': [metadata['image_count'], len(metadata['metric_steps']), len(METRIC_COLUMNS)],
                'dtype': 'float64', 'unobserved_values': 'NaN; see observed_checkpoints_per_image'}
        # Drop only this run's model; no global cache flushing or process control.
        net = None
        shared._json(out / 'manifest.json', metadata)
    report({'phase': 'finished', 'status': metadata['status'], 'output_dir': str(out)})
    return out


def niekompletne(root=None):
    """Pomiary naturalne z brakującymi checkpointami (najnowsze najpierw): [(folder, gotowe, wszystkie, status)]."""
    wynik = []
    for d in sorted(Path(root or OUTPUT_ROOT).glob('*_natural_*'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        wszystkie = len(m.get('checkpoints') or [])
        if m.get('time_axis') == 'training_checkpoint' and wszystkie and len(m.get('steps', [])) < wszystkie:
            wynik.append((d, len(m.get('steps', [])), wszystkie, m.get('status')))
    return wynik


def _uporzadkuj(metadata):
    """Kroki i krzywe rosnąco po kroku (uzupełnione mogą leżeć między wcześniejszymi)."""
    metadata['steps'] = sorted(metadata['steps'])
    c = metadata['curves']
    for key in ('confidence', 'iou', 'both90', 'real_step'):
        c[key] = [metadata['stats'][str(s)][key] for s in metadata['steps']]
    c['steps'] = list(metadata['steps'])


def uzupelnij(katalog, progress=lambda event: None, cancel=lambda: False, device=None):
    """Dolicza na GPU brakujące checkpointy przerwanego pomiaru w TYM SAMYM folderze i zapisuje je na dysk.

    Kontrole przed liczeniem: SHA256 manifestu obrazów, obrazów i checkpointów jak przy zapisie, baza PCA z pliku
    (SHA jak w manifeście), a stan odniesienia (pierwszy checkpoint, nie zapisywany w pełnych 768D) liczony
    ponownie i porównany bitowo z zapisaną klatką kroku bazowego. Niezgodność przerywa bez zmiany zapisu."""
    global _OBRAZY
    import time
    import torch
    out = Path(katalog)
    metadata = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    if metadata.get('time_axis') != 'training_checkpoint':
        raise ValueError('to nie jest pomiar naturalny (oś checkpointów)')
    if metadata.get('status') == 'running' and time.time() - (out / 'manifest.json').stat().st_mtime < 600:
        raise ValueError('ten pomiar mógł jeszcze pracować w innym procesie (zapis młodszy niż 10 min); spróbuj później')
    config = NaturalConfig(**metadata['config'])
    if device:
        config.device = device
    config.frame_format = metadata.get('frame_format', 'f32')     # uzupełnienie w formacie tego pomiaru
    config.validate()
    records = metadata['checkpoints']
    gotowe = set(metadata['steps'])
    brakujace = [j for j, r in enumerate(records) if r['step'] not in gotowe]

    def report(event):
        n = len(brakujace) + 1
        if event.get('phase') in ('baseline', 'measure'):
            f = (event.get('postep', 0) + event['image'] / event['image_count']) / n
            event = dict(event, fraction=min(1., f))
        msg = event['phase']
        if 'step' in event:
            msg += f'; krok {event["step"]}'
        if 'image' in event:
            msg += f'; {event["image"]}/{event["image_count"]} obrazów'
        progress(dict(event, message=event.get('message', msg)))

    if not brakujace:
        report({'phase': 'finished', 'status': metadata['status'], 'output_dir': str(out), 'fraction': 1.})
        return out
    pierwszy = records[0]
    if pierwszy['step'] not in gotowe or not (out / 'initial_pca.npz').is_file():
        raise ValueError('brak zapisanego kroku bazowego albo bazy PCA: ten pomiar trzeba policzyć od nowa')
    for record in [pierwszy] + [records[j] for j in brakujace]:
        if shared._sha(record['path']) != record['sha256']:
            raise ValueError(f'checkpoint zmienił się od zapisu: {record["path"]}')
    if shared._sha(config.manifest) != metadata['manifest_sha256']:
        raise ValueError('manifest obrazów zmienił się od zapisu')
    if shared._sha(out / 'initial_pca.npz') != metadata['pca_file']['sha256']:
        raise ValueError('plik bazy PCA zmienił się od zapisu')
    rows, paths = shared._dataset(config, cancel, report)
    if [r['image_name'] for r in rows] != metadata['image_names']:
        raise ValueError('lista obrazów różni się od zapisu')
    with np.load(out / 'initial_pca.npz') as pca:
        mean, components = pca['mean'].copy(), pca['components'].copy()
    metryki = out / 'metrics.npy'
    telemetry = np.lib.format.open_memmap(metryki, mode='r+')
    if telemetry.shape != (len(rows), len(records), len(METRIC_COLUMNS)):
        raise ValueError(f'metrics.npy ma kształt {telemetry.shape}, oczekiwano zgodnego z manifestem')
    dev = shared._device(config.device)
    core, architecture = shared._private_module('pelne_latenty'), shared._private_module('model_snapshot')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        net = architecture.MalyViT().float()
    net.eval().requires_grad_(False).to(dev)
    baseline, na_karcie = _magazyn_bazy(len(rows), dev, config.batch_size)
    metadata.setdefault('resumes', []).append({'started_utc': datetime.now(timezone.utc).isoformat(),
                                               'missing_steps': [records[j]['step'] for j in brakujace],
                                               'device': str(dev), 'backend_sha256': shared._sha(__file__)})
    try:
        _OBRAZY = _Obrazy(rows, paths, dev, na_karcie, cancel, report) if _infer_rzeczywisty() else None
        with _fp32(dev):
            # 1. stan odniesienia: ten sam forward co przy zapisie, potem kontrola bitowa z zapisaną klatką bazową
            payload, _ = _payload(pierwszy['path'], pierwszy['sha256'], pierwszy['step'])
            net.load_state_dict(payload['net'], strict=True)
            del payload
            kontrola = out / 'baseline_check.bin.tmp'
            kontrola.unlink(missing_ok=True)
            for start in range(0, len(rows), config.batch_size):
                shared._check_cancel(cancel)
                end = min(start + config.batch_size, len(rows))
                z, _ = _infer(net, rows[start:end], paths[start:end], dev, core)
                baseline[start:end] = torch.as_tensor(z).to(baseline.device)
                _append(kontrola, baseline[start:end], baseline[start:end], mean, components)
                report({'phase': 'baseline', 'image': end, 'image_count': len(rows), 'step': pierwszy['step'], 'postep': 0})
            wpis0 = metadata['files'][str(pierwszy['step'])]
            zapisana = out / wpis0['name']
            if wpis0.get('format', 'f32') == 'q16':
                q, quant = shared.kwantyzuj(np.fromfile(kontrola, '<f4').reshape(-1, 4))
                zgodna = q.tobytes() == zapisana.read_bytes() and quant == wpis0['quant']
            else:
                zgodna = kontrola.read_bytes() == zapisana.read_bytes()
            kontrola.unlink(missing_ok=True)
            if not zgodna:
                raise ValueError('stan odniesienia nie odtwarza się bitowo (inny sprzęt, sterownik albo paczka?); '
                                 'policz pomiar od nowa')
            metadata['resumes'][-1]['baseline_bitwise_reproduced'] = True
            # 2. brakujące checkpointy, rosnąco, każdy od razu zapisany na dysk (atomowa publikacja)
            for n, j in enumerate(brakujace, 1):
                record = records[j]
                shared._check_cancel(cancel)
                payload, _ = _payload(record['path'], record['sha256'], record['step'])
                net.load_state_dict(payload['net'], strict=True)
                del payload
                before = core.state_fingerprint(net)
                part = out / f'step_{record["step"]:04d}.bin.part'
                part.unlink(missing_ok=True)                      # niedokończony prefiks z przerwanego biegu
                metadata['current_step'] = record['step']
                try:
                    for start in range(0, len(rows), config.batch_size):
                        shared._check_cancel(cancel)
                        end = min(start + config.batch_size, len(rows))
                        z, measured = _infer(net, rows[start:end], paths[start:end], dev, core)
                        telemetry[start:end, j, :2] = measured
                        telemetry[start:end, j, 3] = record['step']
                        telemetry[start:end, j, 4] = (measured >= .9).all(axis=1)
                        telemetry[start:end, j, 2] = _append(part, z, baseline[start:end], mean, components)
                        report({'phase': 'measure', 'image': end, 'image_count': len(rows), 'step': record['step'],
                                'postep': n})
                        del z, measured
                finally:
                    metadata['checkpoint_audits'].append(dict(step=record['step'], **_audit(net, core, before)))
                if shared._sha(record['path']) != record['sha256']:
                    raise ValueError('checkpoint changed during inference')
                telemetry.flush()
                obs = metadata['observed_checkpoints_per_image']
                metadata['observed_checkpoints_per_image'] = [max(v, j + 1) for v in obs]
                _publish(out, record, metadata, np.array(telemetry[:, j]))
                _uporzadkuj(metadata)
                shared._json(out / 'manifest.json', metadata)
                report({'phase': 'frame', 'step': record['step'], 'completed_checkpoints': len(metadata['steps']),
                        'checkpoint_count': len(records)})
        if len(metadata['steps']) == len(records):
            metadata.update(status='complete', weights_unchanged=True)
            metadata.pop('error', None)
        metadata['resumes'][-1]['status'] = 'complete'
    except BaseException as exc:
        metadata['resumes'][-1]['status'] = 'cancelled' if isinstance(exc, shared._Cancelled) else 'failed'
        if not isinstance(exc, shared._Cancelled):
            metadata['resumes'][-1]['error'] = f'{type(exc).__name__}: {exc}'
            raise
    finally:
        _OBRAZY = None
        _BAZA.clear()
        baseline = None
        net = None
        telemetry.flush()
        del telemetry
        metadata['metrics_file'] = {'name': 'metrics.npy', 'sha256': shared._sha(metryki),
                                    'shape': [metadata['image_count'], len(metadata['metric_steps']), len(METRIC_COLUMNS)],
                                    'dtype': 'float64', 'unobserved_values': 'NaN; see observed_checkpoints_per_image'}
        shared._json(out / 'manifest.json', metadata)
    report({'phase': 'finished', 'status': metadata['status'], 'output_dir': str(out), 'fraction': 1.})
    return out
