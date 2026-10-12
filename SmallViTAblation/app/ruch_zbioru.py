"""Dataset particle motion backend; no Qt, SD or shared detector state.

Public API: Config.validate(), Timeline(path), Engine().run(config, progress,
cancel) -> Path; run(...) is the identical module-level GUI entry point.
Frames are little-endian float32 [image, token, (x,y,z,norm)].
XYZ are absolute coordinates in ONE initial-natural-token PCA basis. Column 3
is ||z(t)-z(0)|| in the original 768D, not the length of a 3D projection.
No images or full latent states are persisted, so this is not resumable latent
optimization. Partial .part files contain only completed image prefixes; only
complete dataset frames are published in the atomic manifest. Telemetry stores
every measured common step, including held states, with actual update counters.
Historical replay continued optimization after success; native runs hold at the
first success and use configurable constant LR instead of the replay schedule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / 'experiments/vit_k1000_1500_pelne_latenty_90_20261008'
DEFAULT_CHECKPOINT = ROOT / 'assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt'
DEFAULT_MANIFEST = ROOT / 'outputs/vit_k1000_1500_pelne_latenty_90_20261008/provenance/MANIFEST_1500.jsonl'
DEFAULT_TIMELINE = ROOT / 'outputs/vit_k1000_taniec_czasteczek_20261009/data/dense'
OSIE_SZKLA_ROOT = ROOT / 'outputs/ruch_czasteczek_gui'


def osie_szkla(root=None):
    """Osie czasu „układu szkła” (foldery z „szklo” w nazwie i manifest.json): lista (opis, ścieżka), najnowsze pierwsze.
    Układ względem klatki z gęstego zapisu (eksport_osi_szkla.py) i symulacje szkła koloidalnego (model_szkla.py)."""
    root = Path(root) if root is not None else OSIE_SZKLA_ROOT
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.glob('*szklo*'), reverse=True):
        if not (p / 'manifest.json').is_file():
            continue
        try:
            u = json.loads((p / 'manifest.json').read_text(encoding='utf-8')).get('uklad') or {}
        except (OSError, ValueError):
            u = {}
        opis = u.get('nazwa') or u.get('model') or '' if isinstance(u, dict) else ''
        out.append((f'{p.name} — {opis}' if opis else p.name, p))
    return out
OUTPUT_ROOT = ROOT / 'outputs/ruch_czasteczek_gui'
_COCO = ROOT / 'experiments/maly_vit_od_zera/eden/wejscia/pretrain/train/_annotations.coco.json'
_IMAGE_ROOT = ROOT / 'experiments/koopman_fastgan_petla/local_inputs/data/images'
TOKENS, DIM = 600, 768  # Architecture of this detector, not timeline assumptions.
TRACE_COLUMNS = ['confidence', 'iou', 'loss', 'delta_l2', 'real_step', 'active', 'success']


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _private_module(name):
    """Load the audited snapshot without changing sys.path/model_snapshot."""
    key = '_ruch_zbioru_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, EXPERIMENT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


@dataclass
class Config:
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    manifest: str = str(DEFAULT_MANIFEST)
    image_root: str = ''
    max_images: int = 1500
    steps: int = 500
    save_every: int = 5
    lr: float = .05
    target_confidence: float = .9
    target_iou: float = .9
    device: str = 'cuda'
    seed: int = 42
    constraint: str = 'gelu'
    regularization: float = 0
    optimizer: str = 'adam'

    def validate(self):
        for key in ('max_images', 'steps', 'save_every'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        for key in ('lr', 'regularization', 'target_confidence', 'target_iou'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if self.lr <= 0 or self.regularization < 0:
            raise ValueError('lr must be positive; regularization must be nonnegative')
        if not all(0 <= v <= 1 for v in (self.target_confidence, self.target_iou)):
            raise ValueError('confidence and IoU targets must be in [0, 1]')
        if self.constraint not in ('gelu', 'none'):
            raise ValueError('constraint must be gelu or none')
        if self.optimizer not in ('adam', 'adamw', 'sgd'):
            raise ValueError('optimizer must be adam, adamw or sgd')
        if not isinstance(self.device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', self.device):
            raise ValueError('device must be cpu, cuda or cuda:N (no implicit fallback)')
        for key in ('checkpoint', 'manifest'):
            if not Path(getattr(self, key)).is_file():
                raise FileNotFoundError(getattr(self, key))
        if self.image_root and not Path(self.image_root).is_dir():
            raise FileNotFoundError(self.image_root)
        return self


# ---------------------------------------------------------------- zapis 16-bitowy klatek cząsteczek
# Klatka [N, 4] = (x, y, z, norma 768D). Format 'q16': każda wartość jako uint16 w granicach TEGO kroku:
#   x = lo + (q / 65535) * span (osobno dla x, y, z),  norma = (q / 65535) * norm_max.
# Połowa miejsca na dysku i w pamięci karty; shader zamienia słowa na liczby (unpackUnorm2x16 w ruch_gpu).
# Błąd zaokrąglenia <= span / 131070 na osi. Statystyki kroku w manifeście liczy producent z float32 przed
# kwantyzacją. Format 'f32' (dawny) zostaje czytany bez zmian.
Q16 = 65535.0
BAJTY_NA_CZASTKE = {'f32': 16, 'q16': 8}


def kwantyzuj(frame):
    """float32 [N, 4] -> (uint16 [N, 4] little-endian, parametry kroku {'lo', 'span', 'norm_max'} jako float32)."""
    frame = np.asarray(frame, np.float32)
    if frame.ndim != 2 or frame.shape[1] != 4 or not np.isfinite(frame).all() or (frame[:, 3] < 0).any():
        raise ValueError('kwantyzacja wymaga skończonej klatki [N, 4] z nieujemną normą')
    lo = frame[:, :3].min(axis=0)
    span = (frame[:, :3].max(axis=0).astype(np.float64) - lo).astype(np.float32)
    nmax = np.float32(frame[:, 3].max())
    q = np.empty(frame.shape, '<u2')
    skala = np.where(span > 0, Q16 / np.maximum(span.astype(np.float64), 1e-300), 0.)
    q[:, :3] = np.clip(np.rint((frame[:, :3].astype(np.float64) - lo) * skala), 0, Q16)
    q[:, 3] = np.clip(np.rint(frame[:, 3].astype(np.float64) / float(nmax) * Q16), 0, Q16) if nmax > 0 else 0
    return q, {'lo': [float(v) for v in lo], 'span': [float(v) for v in span], 'norm_max': float(nmax)}


def parametry_kwantyzacji(quant):
    """[lo_x, lo_y, lo_z, norm_max, span_x, span_y, span_z, 0] float32 (tak samo w shaderze)."""
    v = np.array(list(quant['lo']) + [quant['norm_max']] + list(quant['span']) + [0.], np.float32)
    if v.shape != (8,) or not np.isfinite(v).all() or (v[4:7] < 0).any() or v[3] < 0:
        raise ValueError('nieprawidłowe parametry kwantyzacji kroku')
    return v


def odkwantyzuj(q, quant):
    """uint16 [N, 4] -> float32 [N, 4] tym samym wzorem co shader: lo + (q / 65535) * span."""
    v = parametry_kwantyzacji(quant)
    u = np.asarray(q, np.float32) / np.float32(Q16)
    out = np.empty(u.shape, np.float32)
    out[:, :3] = v[:3] + u[:, :3] * v[4:7]
    out[:, 3] = u[:, 3] * v[3]
    return out


def zapisz_klatke(path, frame, fmt):
    """Atomowy zapis klatki float32 [N, 4] w formacie fmt ('q16' albo 'f32'), z fsync.
    Zwraca wpis manifestu {'name', 'bytes', 'sha256', 'format'[, 'quant']}."""
    path = Path(path)
    if fmt == 'q16':
        dane, quant = kwantyzuj(frame)
    elif fmt == 'f32':
        dane, quant = np.ascontiguousarray(frame, '<f4'), None
    else:
        raise ValueError(f'nieznany format klatki: {fmt}')
    tmp = path.with_name(path.name + '.tmp')
    raw = dane.tobytes()
    with tmp.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    wpis = {'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'format': fmt}
    if quant is not None:
        wpis['quant'] = quant
    return wpis


def konwertuj_q16(zrodlo, cel=None, progress=lambda e: None, cancel=lambda: False):
    """Kopia osi czasu v2 w formacie 'q16' (oryginał zostaje). Klatki czytane z kontrolą SHA256; statystyki,
    granice i krzywe przepisane z oryginału (liczone z float32). Zwraca folder kopii."""
    src = Timeline(zrodlo)
    if src.metadata.get('schema_version', 2) < 2:
        raise ValueError('konwersja obsługuje osie czasu v2 (pomiary z aplikacji)')
    if src.frame_format == 'q16':
        return src.root
    cel = Path(cel) if cel else src.root.with_name(src.root.name + '_q16')
    if cel.exists():
        raise FileExistsError(f'folder docelowy już istnieje: {cel}')
    tymczasowy = cel.with_name(cel.name + '.tmp')
    if tymczasowy.exists():
        shutil.rmtree(tymczasowy)
    tymczasowy.mkdir(parents=True)
    ramki = {item['name'] for item in src.metadata['files'].values()}
    for plik in src.root.iterdir():                          # metryki, PCA, konfiguracja, logi: bez zmian
        if plik.is_file() and plik.name not in ramki and plik.name != 'manifest.json' and not plik.name.endswith(('.part', '.tmp')):
            shutil.copy2(plik, tymczasowy / plik.name)
    m = json.loads(json.dumps(src.metadata))
    przed = after = 0
    for i, step in enumerate(src.steps):
        _check_cancel(cancel)
        frame = src.read(step)
        wpis = zapisz_klatke(tymczasowy / f'step_{step:04d}.q16', frame, 'q16')
        przed += src.metadata['files'][str(step)]['bytes']; after += wpis['bytes']
        m['files'][str(step)] = wpis
        progress({'phase': 'convert', 'step': step, 'fraction': (i + 1) / len(src.steps),
                  'message': f'zapis 16-bitowy: krok {step} ({i + 1}/{len(src.steps)})'})
    m['frame_format'] = 'q16'
    m['converted_from'] = {'path': str(src.root.resolve()), 'manifest_sha256': _sha(src.path),
                           'frame_format': 'f32', 'bytes_before': przed, 'bytes_after': after}
    _json(tymczasowy / 'manifest.json', m)
    kopia = Timeline(tymczasowy)                              # kontrola: każdy krok czyta się i ma błąd <= pół kroku
    for step in src.steps:
        a, b = src.read(step), kopia.read(step)
        tol = np.concatenate([np.asarray(kopia.metadata['files'][str(step)]['quant']['span']),
                              [kopia.metadata['files'][str(step)]['quant']['norm_max']]]) / Q16 * .5 + 1e-5
        if not (np.abs(a - b) <= tol * 1.001 + 1e-6).all():
            raise ValueError(f'kontrola konwersji nie przeszła w kroku {step}')
    os.replace(tymczasowy, cel)
    return cel


class Timeline:
    """Historical dense v1 and native v2. No interpolation or step renumbering.

    `steps` lists available dataset frames; `curves()['steps']` describes the
    separate, usually denser metric axis. Missing provenance remains None.
    File existence/length/hash and finite values are checked on first read;
    subsequent reads cache verification until the file's stat signature changes.
    """
    def __init__(self, path):
        path = Path(path)
        self.path = path if path.is_file() else path / 'manifest.json'
        self.root = self.path.parent
        self.metadata = json.loads(self.path.read_text(encoding='utf-8'))
        m = self.metadata
        self.steps = list(m['steps'])
        if (any(type(s) is not int or s < 0 for s in self.steps)
                or self.steps != sorted(set(self.steps))
                or (self.steps and self.steps[0] != 0 and m.get('time_axis') != 'training_checkpoint')):
            raise ValueError('steps must be unique, increasing; only a checkpoint history may start after step 0')
        self.count = m['point_count']
        self.frame_format = m.get('frame_format', 'f32')
        if self.frame_format not in BAJTY_NA_CZASTKE:
            raise ValueError(f'unknown frame_format: {self.frame_format}')
        if any(m['files'][str(s)].get('format', self.frame_format) != self.frame_format for s in self.steps
               if str(s) in m.get('files', {})):
            raise ValueError('mixed frame formats in one timeline')
        if type(self.count) is not int or self.count < 0 or (self.steps and self.count == 0):
            raise ValueError('point_count must be positive for measured frames')
        if set(m['files']) != {str(s) for s in self.steps}:
            raise ValueError('files and steps do not match')
        producer = m.get('producer_provenance', {})
        self.image_count = m.get('image_count', producer.get('image_count', 0))
        tokens = m.get('tokens_per_image', producer.get('tokens_per_image'))
        if type(self.image_count) is not int or self.image_count < 0:
            raise ValueError('invalid image_count')
        if self.image_count and (self.count % self.image_count or
                                (tokens and self.count != self.image_count * tokens)):
            raise ValueError('image/token dimensions disagree with point_count')
        self.checkpoint_sha256 = m.get('checkpoint_sha256', producer.get('checkpoint_sha256'))
        self.image_names = list(m.get('image_names', []))
        # Historical export omitted image names/checkpoint; its local bundle
        # metadata provides them. Never try the remote producer path.
        sidecar = self.root.parent / 'metadata.json'
        if m.get('schema_version') == 1 and sidecar.is_file():
            bundle = self.root.parent / 'BUNDLE_MANIFEST.json'
            if bundle.is_file():
                files = json.loads(bundle.read_text(encoding='utf-8'))['files']
                expected = next((r['sha256'] for r in files if r['filename'] == 'metadata.json'), None)
                if expected and _sha(sidecar) != expected:
                    raise ValueError('historical metadata SHA256 mismatch')
            extra = json.loads(sidecar.read_text(encoding='utf-8'))
            images = extra.get('images', [])
            if any(r.get('index') != i for i, r in enumerate(images)):
                raise ValueError('historical image order mismatch')
            self.image_names = [r['image_name'] for r in images]
            self.checkpoint_sha256 = self.checkpoint_sha256 or extra.get('checkpoint_sha256')
            self.image_count = self.image_count or len(images)
        if self.image_names and (len(self.image_names) != self.image_count or
                                 any(not isinstance(n, str) for n in self.image_names)):
            raise ValueError('image names/order disagree with image_count')
        if m.get('images') and len(m['images']) != self.image_count:
            raise ValueError('image metadata dimension mismatch')
        for i, record in enumerate(m.get('images', [])):
            if record.get('index') != i or (self.image_names and record.get('image_name') != self.image_names[i]):
                raise ValueError('image order mismatch')
        self._verified = {}
        self._lock = threading.RLock()
        for s in self.steps:
            self._entry(s)  # validate declared schema/path/size, not binary IO
            metrics = m.get('stats', {}).get(str(s), {})
            for key in ('confidence', 'iou'):
                value = metrics.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'missing/invalid {key} metric at step {s}')
            both = metrics.get('both90')
            if type(both) is not int or both < 0 or (self.image_count and both > self.image_count):
                raise ValueError(f'missing/invalid both90 metric at step {s}')
        self.curves()

    def _entry(self, step):
        if type(step) is not int or step not in self.steps:
            raise KeyError(f'no measured dataset frame at step {step}')
        item = self.metadata['files'][str(step)]
        p = (self.root / item['name']).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError('frame path escapes timeline directory')
        if item.get('bytes') != self.count * BAJTY_NA_CZASTKE[self.frame_format]:
            raise ValueError('declared frame dimensions mismatch')
        if self.frame_format == 'q16':
            parametry_kwantyzacji(item.get('quant') or {'lo': [], 'span': [], 'norm_max': float('nan')})
        if not re.fullmatch('[0-9a-fA-F]{64}', str(item.get('sha256', ''))):
            raise ValueError('frame SHA256 is missing or malformed')
        return p, item

    @staticmethod
    def _check_values(values):
        if not np.isfinite(values).all() or (values[:, 3] < 0).any():
            raise ValueError('nonfinite coordinates or negative 768D norm')

    def _signature(self, step):
        p, item = self._entry(step)
        st = p.stat()
        return p, item, (st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    @property
    def bajty_klatki(self):
        return self.count * BAJTY_NA_CZASTKE[self.frame_format]

    def frame(self, step):
        if self.frame_format == 'q16':                      # klatka 16-bitowa: zwracamy odkwantowaną kopię
            return self.read(step)
        # Verification runs outside the lock, so several reader threads can hash
        # different frames at once; only the cache dictionary is guarded.
        p, item, signature = self._signature(step)
        with self._lock:
            known = self._verified.get(step) == signature
        if not known:
            if signature[0] != self.count * 16 or _sha(p) != item['sha256']:
                raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
            with p.open('rb') as f:
                while chunk := f.read(4 << 20):
                    self._check_values(np.frombuffer(chunk, dtype='<f4').reshape(-1, 4))
            with self._lock:
                self._verified[step] = signature
        # Read-only memmap is C-contiguous and does not cache the timeline in RAM.
        return np.memmap(p, dtype='<f4', mode='r', shape=(self.count, 4))

    def read(self, step):
        """Verified in-memory copy with ONE pass over the file: SHA256, size and
        value checks use the same bytes that are returned. Zawsze float32 [N, 4] (q16: odkwantowane)."""
        if self.frame_format == 'q16':
            return odkwantyzuj(self.read_raw(step), self.metadata['files'][str(step)]['quant'])
        return self._read_f32(step)

    def read_raw(self, step):
        """Klatka w formacie pliku (q16: uint16 [N, 4], f32: float32 [N, 4]) po kontroli rozmiaru i SHA256;
        to wysyła się na kartę. Parametry kroku q16: quant(step)."""
        if self.frame_format == 'f32':
            return self._read_f32(step)
        p, item, signature = self._signature(step)
        if signature[0] != self.bajty_klatki:
            raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        raw = bytearray(signature[0])
        with p.open('rb') as f:
            if f.readinto(raw) != len(raw) or f.read(1):
                raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        with self._lock:
            self._verified[step] = signature
        return np.frombuffer(raw, dtype='<u2').reshape(self.count, 4)

    def quant(self, step):
        """Parametry kwantyzacji kroku jako float32 [8] (None dla f32)."""
        if self.frame_format != 'q16':
            return None
        self._entry(step)
        return parametry_kwantyzacji(self.metadata['files'][str(step)]['quant'])

    def _read_f32(self, step):
        p, item, signature = self._signature(step)
        if signature[0] != self.count * 16:
            raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        raw = bytearray(signature[0])
        with p.open('rb') as f:
            if f.readinto(raw) != len(raw) or f.read(1):
                raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'frame size/SHA256 mismatch: {p.name}')
        values = np.frombuffer(raw, dtype='<f4').reshape(self.count, 4)
        self._check_values(values)
        with self._lock:
            self._verified[step] = signature
        return values

    def stats(self, step):
        self._entry(step)
        result = dict(self.metadata.get('stats', {}).get(str(step), {}))
        return result

    def bounds(self, movement=False):
        values = self.metadata.get('bounds', {}).get('movement' if movement else 'natural')
        if values is None:
            raise ValueError('bounds unavailable: no complete frame published yet')
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (3, 2) or not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any():
            raise ValueError('invalid bounds')
        return values

    def curves(self):
        result = dict(self.metadata.get('curves', {}))
        if 'steps' not in result:
            size = len(result.get('confidence', []))
            # The historical metric arrays are real per-update traces; frame
            # stride must not be mistaken for their sampling interval.
            result['steps'] = list(range(size))
        axis = result['steps']
        if axis != sorted(set(axis)) or any(type(v) is not int or v < 0 for v in axis):
            raise ValueError('invalid metric step order')
        for key in ('confidence', 'iou', 'both90', 'success_count', 'active', 'real_step'):
            if key in result and len(result[key]) != len(axis):
                raise ValueError(f'{key} curve dimensions mismatch')
        for key in ('confidence', 'iou'):
            if key not in result:
                raise ValueError(f'missing {key} curve')
            values = np.asarray(result[key], dtype=np.float64)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f'invalid {key} curve')
        return result


class _Cancelled(Exception):
    pass


def _check_cancel(cancel):
    if cancel():
        raise _Cancelled()


def _image_path(name, source, image_root):
    name = str(name).replace('\\', '/')
    base = name.rsplit('/', 1)[-1]
    candidates = []
    if image_root:
        # Explicit root takes precedence over stale absolute paths in COCO.
        candidates.extend([Path(image_root) / base, Path(image_root) / 'images' / base])
        if not Path(name).is_absolute():
            candidates.append(Path(image_root) / name)
    else:
        candidates.extend([Path(name), source.parent / name, source.parent / 'images' / base,
                           _IMAGE_ROOT / base])
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f'image not found: {name}; set image_root')


def _dataset(config, cancel, progress):
    """Validate genuine GT, input ordering, dimensions, scaling and byte hashes."""
    import cv2
    data = _private_module('dane')
    path = Path(config.manifest).resolve()
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and 'annotations' in payload and 'images' in payload:
            def read(name):
                _check_cancel(cancel)
                return _image_path(name, path, config.image_root).read_bytes()
            rows = data.build_manifest(path, read)
            for row, image in zip(rows, payload['images']):
                row['_source_name'] = image['file_name']
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError('expected exact manifest rows (JSON/JSONL) or COCO images+annotations')
    if not rows:
        raise ValueError('empty dataset')
    names, ids, annotation_ids = set(), set(), set()
    # Validate structural order across the whole input, even when selecting a prefix.
    for i, row in enumerate(rows):
        if row.get('index') != i:
            raise ValueError(f'manifest image order mismatch at {i}')
        for key, seen in [('image_name', names), ('image_id', ids), ('annotation_id', annotation_ids)]:
            if row[key] in seen:
                raise ValueError(f'duplicate {key}: {row[key]}')
            seen.add(row[key])
    selected = rows[:config.max_images]
    paths = []
    for i, row in enumerate(selected):
        _check_cancel(cancel)
        p = _image_path(row.get('_source_name', row['image_name']), path, config.image_root)
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
            raise ValueError(f'image SHA256 mismatch: {row["image_name"]}')
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        w, h = row['source_width'], row['source_height']
        if image is None or image.shape[:2] != (h, w) or w <= 0 or h <= 0:
            raise ValueError(f'source dimensions mismatch: {p}')
        box = np.asarray(row['bbox_xywh_source'], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('invalid source annotation')
        x, y, bw, bh = box
        if min(x, y) < 0 or min(bw, bh) <= 0 or x + bw > w + 1e-6 or y + bh > h + 1e-6:
            raise ValueError('source annotation outside image')
        expected = np.array([x * 480/w, y * 320/h, (x+bw) * 480/w, (y+bh) * 320/h])
        if not np.allclose(expected, row['bbox_xyxy_480'], rtol=0, atol=1e-5):
            raise ValueError('annotation resize mismatch (requires independent x/y scaling)')
        paths.append(p)
        progress({'phase': 'validate', 'image': i + 1, 'image_count': len(selected)})
    return selected, paths


def _device(name):
    import torch
    dev = torch.device(name)
    if dev.type == 'cuda':
        index = dev.index if dev.index is not None else torch.cuda.current_device() if torch.cuda.is_available() else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f'requested CUDA device unavailable: {name}; select cpu explicitly')
        dev = torch.device('cuda', index)
    return dev


def _estimates(config, count):
    # Constant-memory arithmetic: never materialize a huge horizon before
    # checking disk/RAM. Frame 0 and an off-interval final step are included.
    frames = config.steps // config.save_every + 1 + int(config.steps % config.save_every != 0)
    trace_bytes = count * (config.steps + 1) * len(TRACE_COLUMNS) * 4
    # Two metric generations for atomic publication; output frames + PCA + margins.
    disk = int(1.2 * (frames * count * TOKENS * 16 + 2 * trace_bytes) + (64 << 20))
    ram = (512 << 20) + trace_bytes * 2 + count * TOKENS * 16 * 3
    return {'disk_bytes': disk, 'ram_bytes': ram, 'cuda_bytes': 256 << 20,
            'frame_count': frames, 'full_latent_states_persisted': False}


def _resources(config, count, output_root, dev):
    import psutil
    import torch
    estimates = _estimates(config, count)
    existing = output_root.resolve()
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < estimates['disk_bytes']:
        raise ValueError(f'insufficient disk space; estimated {estimates["disk_bytes"]} bytes')
    if psutil.virtual_memory().available < estimates['ram_bytes']:
        raise ValueError(f'insufficient RAM; estimated {estimates["ram_bytes"]} bytes')
    if dev.type == 'cuda' and torch.cuda.mem_get_info(dev)[0] < estimates['cuda_bytes']:
        raise ValueError(f'insufficient memory on {dev}; estimated {estimates["cuda_bytes"]} bytes')
    return estimates


def _baseline(net, row, path, dev, core):
    import torch
    data = _private_module('dane')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
        raise ValueError(f'image changed during run: {path}')
    rgb = data.decode_resize(raw)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(dev) / 255
    output, z, q = core.capture_baseline(net, x)
    if z.shape != (1, TOKENS, DIM) or not torch.isfinite(z).all():
        raise ValueError('detector must provide finite individual [1,600,768] post-GELU latent')
    with torch.no_grad():
        error = float((core.fast_tail(net, z, q, z) - output).abs().max())
    if error > 1e-5:
        raise ValueError(f'fast tail disagrees with full detector: {error}')
    return output, z, q, error


def _pca(net, rows, paths, dev, core, seed, cancel, progress):
    """Uniform natural-token reservoir from all selected images, one shared fit."""
    count = len(rows) * TOKENS
    indices = np.sort(np.random.default_rng(seed).choice(count, min(4096, count), replace=False))
    sample = np.empty((len(indices), DIM), dtype=np.float64)
    for i, (row, path) in enumerate(zip(rows, paths)):
        _check_cancel(cancel)
        begin, end = np.searchsorted(indices, [i * TOKENS, (i+1) * TOKENS])
        if end > begin:
            _, z, _, _ = _baseline(net, row, path, dev, core)
            sample[begin:end] = z[0, indices[begin:end] - i * TOKENS].cpu().numpy()
        progress({'phase': 'pca', 'image': i + 1, 'image_count': len(rows)})
    _check_cancel(cancel)
    mean = sample.mean(axis=0)
    sample -= mean
    values, vectors = np.linalg.eigh(sample.T @ sample / max(len(sample)-1, 1))
    components = vectors[:, -3:][:, ::-1].copy()
    # Resolve eigenvector sign deterministically.
    for k in range(3):
        if components[np.argmax(np.abs(components[:, k])), k] < 0:
            components[:, k] *= -1
    return mean, components, indices, values[-3:][::-1]


class Engine:
    """One image on the requested device at a time; all-image storage is 3D only.

    progress(dict) phases: validate, estimate, pca, baseline, optimize, finished.
    Optimize events: image/image_count, step, real_step, confidence/iou, active.
    Success, cancellation and failure publish status in manifest.json. Failures
    re-raise after preserving artifacts; cancellation returns the partial Path.
    """
    def run(self, config, progress=lambda event: None, cancel=lambda: False):
        import torch
        config.validate()
        config = Config(**asdict(config))  # GUI mutation cannot alter an active run.
        callback = progress
        last_fraction = 0.

        def progress(event):
            nonlocal last_fraction
            phase = event['phase']
            n = max(event.get('image_count', 1), 1)
            image = event.get('image', 0)
            if phase == 'optimize':
                fraction = .35 + .6 * ((image-1) + event.get('step', 0)/config.steps) / n
            else:
                base, span = {'validate': (0., .05), 'estimate': (.05, 0.),
                              'pca': (.05, .15), 'baseline': (.2, .15),
                              'finished': (1. if event.get('status') == 'complete' else last_fraction, 0.)}[phase]
                fraction = base + span * image/n
            last_fraction = min(1., max(last_fraction, fraction))
            message = f'{phase}: {image}/{n}' if image else phase
            if 'step' in event:
                message += f'; step {event["step"]}/{config.steps}'
            callback(dict(event, fraction=last_fraction, message=message))
        dev = _device(config.device)
        core = _private_module('pelne_latenty')
        architecture = _private_module('model_snapshot')
        run_root = Path(OUTPUT_ROOT)
        run_root.mkdir(parents=True, exist_ok=True)
        out = run_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid.uuid4().hex[:12])
        out.mkdir()
        metadata = {'schema_version': 2, 'status': 'preparing', 'steps': [], 'files': {},
                    'stats': {}, 'curves': {'steps': [], 'confidence': [], 'iou': []},
                    'point_count': 0, 'image_count': 0, 'tokens_per_image': TOKENS,
                    'time_axis': 'latent_optimization', 'interpolation': False,
                    'full_latent_states_persisted': False, 'config': asdict(config),
                    'norm': 'L2(z(step)-z(0)) in full 768D, NOT PCA length',
                    'coordinates': '(z-mean) @ fixed_components, image-major then token-major',
                    'held_state_policy': 'first success frozen; real_step stops; active=false',
                    'success_count_targets': {'confidence': config.target_confidence, 'iou': config.target_iou},
                    'both90_targets': {'confidence': .9, 'iou': .9},
                    'partial_policy': 'Only complete dataset frames listed; .part files are image prefixes',
                    'metric_columns': TRACE_COLUMNS, 'completed_images': 0}
        _json(out / 'config.json', asdict(config))
        _json(out / 'manifest.json', metadata)
        net, traces = None, None
        try:
            rows, paths = _dataset(config, cancel, progress)
            estimates = _resources(config, len(rows), run_root, dev)
            progress(dict(phase='estimate', **estimates))
            metadata.update(point_count=len(rows)*TOKENS, image_count=len(rows),
                            image_names=[r['image_name'] for r in rows],
                            images=[{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
                            estimates=estimates, device=str(dev),
                            checkpoint_sha256=_sha(config.checkpoint), manifest_sha256=_sha(config.manifest),
                            source_hashes={name: _sha(EXPERIMENT / (name+'.py')) for name in
                                           ('model_snapshot', 'pelne_latenty', 'dane')},
                            backend_sha256=_sha(__file__))
            _json(out / 'manifest.json', metadata)
            _check_cancel(cancel)
            payload = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
            # Isolated model initialization preserves the application's RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                net = architecture.MalyViT().float()
            net.load_state_dict(payload.get('net', payload), strict=True)
            checkpoint_step = payload.get('krok')
            del payload
            if _sha(config.checkpoint) != metadata['checkpoint_sha256']:
                raise ValueError('checkpoint changed during load')
            net.eval().requires_grad_(False).to(dev)
            state_sha = core.state_fingerprint(net)
            metadata.update(weights_sha256_before=state_sha, checkpoint_step=checkpoint_step)
            # Independent provenance covers every weight/buffer and frozen flag.
            freeze = {name: not p.requires_grad for name, p in net.named_parameters()}
            metadata['freeze'] = {'all_parameters_frozen': all(freeze.values()),
                                  'parameter_count': len(freeze), 'eval': not net.training,
                                  'flags_sha256': hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()}
            mean, components, reservoir, eigenvalues = _pca(net, rows, paths, dev, core, config.seed, cancel, progress)
            with (out / 'initial_pca.npz.tmp').open('wb') as f:
                np.savez(f, mean=mean, components=components,
                         reservoir_token_indices=reservoir, eigenvalues=eigenvalues)
                f.flush()
                os.fsync(f.fileno())
            os.replace(out / 'initial_pca.npz.tmp', out / 'initial_pca.npz')
            basis_hash = hashlib.sha256(mean.astype('<f8').tobytes() + components.astype('<f8').tobytes()).hexdigest()
            metadata.update(pca_basis_sha256=basis_hash,
                            pca_file={'name': 'initial_pca.npz', 'sha256': _sha(out / 'initial_pca.npz')},
                            pca_fit='uniform reservoir of natural step-0 individual tokens; no future states',
                            pca_seed=config.seed, pca_sample_count=len(reservoir))
            steps = sorted(set(range(0, config.steps + 1, config.save_every)) | {config.steps})
            # Telemetry is disk backed; no [images, steps, 600, 768] allocation.
            traces = np.lib.format.open_memmap(out / 'metrics.partial.npy', mode='w+', dtype='<f4',
                                               shape=(len(rows), config.steps+1, len(TRACE_COLUMNS)))
            traces[:] = np.nan
            observed = np.zeros(len(rows), dtype=np.int64)
            metadata['observed_steps_per_image'] = observed.tolist()
            initial_part = out / 'step_000.bin.part'
            max_tail_error = 0.
            for i, (row, path) in enumerate(zip(rows, paths)):
                _check_cancel(cancel)
                output, z, q, error = _baseline(net, row, path, dev, core)
                max_tail_error = max(max_tail_error, error)
                gt = torch.tensor(row['bbox_xyxy_480'], device=dev, dtype=torch.float32)
                m = core.top1_metrics(output, gt)
                target = core.choose_target_token(np.asarray(row['bbox_xyxy_480']))
                with torch.no_grad():
                    loss0 = float(core._fit_objective(output, z, z, target['token_index'],
                                                     torch.tensor(target['target_ltrb'], device=dev)))
                success = m['confidence'] >= config.target_confidence and m['iou'] >= config.target_iou
                traces[i, 0] = [m['confidence'], m['iou'], loss0, 0., 0., not success, success]
                observed[i] = 1
                self._append(initial_part, z, z, mean, components)
                progress({'phase': 'baseline', 'image': i+1, 'image_count': len(rows), 'step': 0})
            traces.flush()
            self._publish_frame(out, 0, metadata, traces)
            self._publish_metrics(out, metadata, traces, observed)
            metadata.update(status='running', max_fast_tail_error=max_tail_error)
            _json(out / 'manifest.json', metadata)
            # Only one baseline/current latent and optimizer are resident on GPU.
            for i, (row, path) in enumerate(zip(rows, paths)):
                _check_cancel(cancel)
                output, z0, q, error = _baseline(net, row, path, dev, core)
                max_tail_error = max(max_tail_error, error)
                gt = torch.tensor(row['bbox_xyxy_480'], device=dev, dtype=torch.float32)
                target = core.choose_target_token(np.asarray(row['bbox_xyxy_480']))
                target_ltrb = torch.tensor(target['target_ltrb'], device=dev)
                changed = z0.clone().requires_grad_(True)
                cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}[config.optimizer]
                kwargs = {'weight_decay': 0.} if config.optimizer == 'adamw' else {}
                optimizer = cls([changed], lr=config.lr, **kwargs)
                active = bool(traces[i, 0, 5])
                real_step = 0
                for step in range(1, config.steps+1):
                    _check_cancel(cancel)
                    if active:
                        optimizer.zero_grad(set_to_none=True)
                        output = core.fast_tail(net, z0, q, changed)
                        # Reuse audited fit loss, replacing its fixed 1e-6 penalty
                        # with the configured penalty. No historical LR/min-norm phase.
                        loss = core._fit_objective(output, changed, z0, target['token_index'], target_ltrb)
                        loss = loss + (config.regularization - 1e-6) * (changed-z0).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite optimization loss')
                        loss.backward()
                        if changed.grad is None or not torch.isfinite(changed.grad).all():
                            raise FloatingPointError('nonfinite latent gradient')
                        optimizer.step()
                        real_step += 1
                        with torch.no_grad():
                            if config.constraint == 'gelu':
                                changed.clamp_(min=core.GELU_FLOOR)
                    with torch.no_grad():
                        output = core.fast_tail(net, z0, q, changed)
                        m = core.top1_metrics(output, gt)
                        loss = core._fit_objective(output, changed, z0, target['token_index'], target_ltrb)
                        loss += (config.regularization - 1e-6) * (changed-z0).square().mean()
                        delta = float((changed-z0).norm())
                        success = m['confidence'] >= config.target_confidence and m['iou'] >= config.target_iou
                        active = active and not success
                        values = [m['confidence'], m['iou'], float(loss), delta, real_step, active, success]
                        if not np.isfinite(values).all():
                            raise FloatingPointError('nonfinite post-update metrics')
                        traces[i, step] = values
                    observed[i] = step+1
                    if step in steps:
                        self._append(out / f'step_{step:03d}.bin.part', changed, z0, mean, components)
                    if step % config.save_every == 0 or step == config.steps:
                        progress({'phase': 'optimize', 'image': i+1, 'image_count': len(rows),
                                  'step': step, 'real_step': real_step, 'active': active,
                                  'confidence': m['confidence'], 'iou': m['iou']})
                metadata['completed_images'] = i+1
                traces.flush()
                metadata['observed_steps_per_image'] = observed.tolist()
                _json(out / 'manifest.json', metadata)
                del optimizer, changed, z0, q, output
            _check_cancel(cancel)
            for step in steps[1:]:
                self._publish_frame(out, step, metadata, traces)
            self._publish_metrics(out, metadata, traces, observed)
            metadata.update(status='complete', max_fast_tail_error=max_tail_error,
                            weights_sha256_after=core.state_fingerprint(net))
            metadata['weights_unchanged'] = metadata['weights_sha256_after'] == state_sha
            if not metadata['weights_unchanged'] or any(p.requires_grad for p in net.parameters()):
                raise RuntimeError('detector weights/freeze invariant violated')
            _json(out / 'manifest.json', metadata)
            progress({'phase': 'finished', 'status': 'complete', 'output_dir': str(out)})
            return out
        except BaseException as exc:
            metadata['status'] = 'cancelled' if isinstance(exc, _Cancelled) else 'failed'
            if traces is not None:
                traces.flush()
                self._publish_metrics(out, metadata, traces, observed)
            if net is not None:
                metadata['weights_sha256_after'] = core.state_fingerprint(net)
                metadata['weights_unchanged'] = metadata['weights_sha256_after'] == metadata.get('weights_sha256_before')
            if not isinstance(exc, _Cancelled):
                metadata['error'] = f'{type(exc).__name__}: {exc}'
            _json(out / 'manifest.json', metadata)
            if isinstance(exc, _Cancelled):
                progress({'phase': 'finished', 'status': 'cancelled', 'output_dir': str(out)})
                return out
            raise
        finally:
            # No shared/global CUDA cache flushing or device switching.
            if traces is not None:
                traces.flush()
                traces._mmap.close()
            if net is not None:
                net.to('cpu')

    @staticmethod
    def _append(path, z, z0, mean, components):
        array = z.detach()[0].cpu().numpy().astype(np.float64)
        frame = np.empty((TOKENS, 4), dtype='<f4')
        frame[:, :3] = (array-mean) @ components
        frame[:, 3] = np.linalg.norm(array-z0.detach()[0].cpu().numpy(), axis=1)
        if not np.isfinite(frame).all():
            raise FloatingPointError('nonfinite particle projection')
        with Path(path).open('ab') as f:
            f.write(frame.tobytes())

    @staticmethod
    def _publish_frame(out, step, metadata, traces):
        part = out / f'step_{step:03d}.bin.part'
        if part.stat().st_size != metadata['point_count'] * 16:
            raise ValueError('cannot publish an incomplete dataset frame')
        path = part.with_suffix('')
        os.replace(part, path)
        metadata['steps'].append(step)
        metadata['files'][str(step)] = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': _sha(path)}
        frame = np.memmap(path, dtype='<f4', mode='r', shape=(metadata['point_count'], 4))
        initial = np.memmap(out / 'step_000.bin', dtype='<f4', mode='r', shape=frame.shape)
        bounds = metadata.setdefault('bounds', {})
        for key, xyz in [('natural', frame[:, :3]), ('movement', frame[:, :3]-initial[:, :3])]:
            low, high = xyz.min(axis=0), xyz.max(axis=0)
            if key in bounds:
                old = np.asarray(bounds[key])
                low, high = np.minimum(low, old[:, 0]), np.maximum(high, old[:, 1])
            bounds[key] = np.column_stack([low, high]).tolist()
        norm = frame[:, 3]
        metrics = traces[:, step]
        metadata['stats'][str(step)] = {'confidence': float(metrics[:, 0].mean(dtype=np.float64)),
            'iou': float(metrics[:, 1].mean(dtype=np.float64)),
            'both90': int(((metrics[:, 0] >= .9) & (metrics[:, 1] >= .9)).sum()),
            'success': int(metrics[:, 6].sum()), 'success_count': int(metrics[:, 6].sum()),
            'active': int(metrics[:, 5].sum()),
            'real_step': float(metrics[:, 4].mean(dtype=np.float64)),
            'norm_median': float(np.median(norm)), 'norm_max': float(norm.max())}
        metadata['norm_scale'] = max(metadata.get('norm_scale', 0.), float(norm.max()), 1e-6)
        frame._mmap.close()
        initial._mmap.close()

    @staticmethod
    def _publish_metrics(out, metadata, traces, observed):
        # Publish a valid NPY via replace; partial rows contain NaN only for
        # unmeasured steps, explicitly described by observed_steps_per_image.
        tmp = out / 'metrics.npy.tmp'
        with tmp.open('wb') as f:
            np.save(f, traces, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out / 'metrics.npy')
        metadata['metrics_file'] = {'name': 'metrics.npy', 'sha256': _sha(out / 'metrics.npy'),
                                   'shape': list(traces.shape), 'dtype': 'float32'}
        metadata['observed_steps_per_image'] = observed.tolist()
        horizon = int(observed.min())
        confidence, iou = traces[:, :horizon, 0], traces[:, :horizon, 1]
        metadata['curves'] = {'steps': list(range(horizon)),
            'confidence': confidence.mean(axis=0, dtype=np.float64).tolist(),
            'iou': iou.mean(axis=0, dtype=np.float64).tolist(),
            'both90': ((confidence >= .9) & (iou >= .9)).sum(axis=0).tolist(),
            'success_count': traces[:, :horizon, 6].sum(axis=0).astype(int).tolist(),
            'active': traces[:, :horizon, 5].sum(axis=0).astype(int).tolist(),
            'real_step': traces[:, :horizon, 4].mean(axis=0, dtype=np.float64).tolist(),
            'label': 'Measured dataset mean; held states explicitly inactive'}


def run(config, progress=lambda event: None, cancel=lambda: False):
    """GUI convenience entry point with exactly the Engine.run contract."""
    return Engine().run(config, progress=progress, cancel=cancel)
