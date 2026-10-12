"""Stały odczyt RAW post-LN RF-DETR Nano, bez normalizacji wagami checkpointu.

Kolejność 1536 współrzędnych: średnia foreground, średnia background,
pierwszy moment x, pierwszy moment y; każdy blok ma 384 cechy. Wagi to
udziały ręcznej maski w tokenie (area resize). Współrzędne środków tokenów
centrujemy w masce i dzielimy przez ich ważone odchylenie RMS, z dolnym
ograniczeniem 1e-6. Momenty dzielimy przez masę maski. Stała cecha ma
zerowe momenty; maska pusta lub pełna jest błędem kontraktu.

Import modułu wymaga tylko NumPy. PyTorch, Pillow i RF-DETR są ładowane
dopiero w odpowiednich funkcjach. Reader nie zmienia trybu ani wag modelu.
Wywołujący odpowiada za eval/train, zwłaszcza w callbacku treningowym.
"""
from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path

import numpy as np

if __package__:
    from . import sonda
else:
    import sonda

FEATURE_CHANNELS = 384
FEATURE_GRID = (24, 24)
DESCRIPTOR_DIM = 1536
DESCRIPTOR_VERSION = 'raw-post-ln-fg-bg-centered-rms-xy-v1'


def descriptor_numpy(features, mask):
    """Wersja NumPy BCHW/B1HW dla niezależnych testów numerycznych."""
    f, m = np.asarray(features, dtype=np.float64), np.asarray(mask, dtype=np.float64)
    if f.ndim != 4 or m.shape != (f.shape[0], 1, *f.shape[-2:]):
        raise ValueError('Oczekiwano BCHW i B1HW o wspólnej siatce')
    if not np.isfinite(f).all() or not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
        raise ValueError('Cechy muszą być skończone, maska w [0,1]')
    mass, outside = m.sum((2, 3)), (1-m).sum((2, 3))
    if np.any(mass <= 0) or np.any(outside <= 0):
        raise ValueError('Odczyt wymaga niepustej maski i niepustego tła')
    fg = (f*m).sum((2, 3))/mass
    bg = (f*(1-m)).sum((2, 3))/outside
    h, w = f.shape[-2:]
    yy, xx = np.meshgrid((np.arange(h)+.5)/h*2-1, (np.arange(w)+.5)/w*2-1, indexing='ij')
    moments = []
    for grid in (xx, yy):
        center = (m*grid).sum((2, 3))/mass
        centered = grid-center[..., None, None]
        rms = np.sqrt((m*centered**2).sum((2, 3))/mass).clip(1e-6)
        coordinate = centered/rms[..., None, None]
        moments.append((f*m*coordinate).sum((2, 3))/mass)
    return np.concatenate([fg, bg, *moments], axis=1)


def descriptor_torch(features, mask):
    """Różniczkowalny odpowiednik; maska może mieć rozdzielczość obrazu."""
    import torch
    import torch.nn.functional as functional

    f = features
    if f.ndim != 4 or not f.is_floating_point() or mask.ndim != 4 or mask.shape[:2] != (f.shape[0], 1):
        raise ValueError('Oczekiwano zmiennoprzecinkowego BCHW oraz B1HW')
    m = mask.to(device=f.device, dtype=f.dtype)
    if not bool(torch.isfinite(f).all() and torch.isfinite(m).all()) or bool(((m < 0) | (m > 1)).any()):
        raise ValueError('Cechy muszą być skończone, maska w [0,1]')
    m = functional.interpolate(m, f.shape[-2:], mode='area')
    mass, outside = m.sum((2, 3)), (1-m).sum((2, 3))
    if bool((mass <= 0).any() or (outside <= 0).any()):
        raise ValueError('Odczyt wymaga niepustej maski i niepustego tła')
    fg, bg = (f*m).sum((2, 3))/mass, (f*(1-m)).sum((2, 3))/outside
    h, w = f.shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(h, device=f.device, dtype=f.dtype)+.5)/h*2-1,
                           (torch.arange(w, device=f.device, dtype=f.dtype)+.5)/w*2-1, indexing='ij')
    moments = []
    for grid in (xx, yy):
        center = (m*grid).sum((2, 3))/mass
        centered = grid-center[..., None, None]
        rms = ((m*centered.square()).sum((2, 3))/mass).clamp_min(1e-12).sqrt()
        coordinate = centered/rms[..., None, None]
        moments.append((f*m*coordinate).sum((2, 3))/mass)
    return torch.cat([fg, bg, *moments], dim=1)


class DifferentiableReader:
    """Odczyt z końcowego wyjścia WindowedDinov2WithRegistersBackbone.

    ``net`` to torch model (np. wrapper.model.model), nie API predict.
    ``means/stds`` pochodzą z wrappera RFDETRNano. Obrazy: RGB [0,1].
    Używać jako context manager albo jawnie wywołać close(). Każdy forward
    tworzy osobny NestedTensor i osobną maskę paddingu; nie używa bufora
    inference. Wyjścia pozostają w grafie gradientu. Instancja nie jest
    przeznaczona do współbieżnych forwardów tego samego modelu.
    """
    def __init__(self, net, means, stds, *, nested_tensor_factory=None):
        self.net = net
        self.means, self.stds = np.array(means, dtype=float), np.array(stds, dtype=float)
        if (self.means.shape != (3,) or self.stds.shape != (3,) or
                not np.isfinite(self.means).all() or not np.isfinite(self.stds).all() or
                np.any(self.stds <= 0)):
            raise ValueError('means/stds muszą mieć trzy skończone wartości, stds > 0')
        matches = [m for m in net.modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone']
        if len(matches) != 1:
            raise ValueError('Wymagany dokładnie jeden WindowedDinov2WithRegistersBackbone')
        self._factory = nested_tensor_factory
        self._feature = None
        self._active = False
        self._hook = matches[0].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if self._active:
            maps = output.feature_maps if hasattr(output, 'feature_maps') else output[0]
            self._feature = maps[-1]

    def features(self, image):
        import torch
        import torch.nn.functional as functional

        if self._hook is None:
            raise RuntimeError('Reader został zamknięty')
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError('Obraz musi mieć format BCHW RGB [0,1]')
        if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
            raise ValueError('Obraz musi być skończony i mieścić się w [0,1]')
        factory = self._factory
        if factory is None:
            from rfdetr.utilities.tensors import NestedTensor
            factory = NestedTensor
        image = functional.interpolate(image, (384, 384), mode='bilinear', align_corners=False)
        mean = image.new_tensor(self.means).reshape(1, 3, 1, 1)
        std = image.new_tensor(self.stds).reshape(1, 3, 1, 1)
        normalized = (image-mean)/std
        padding = torch.zeros((image.shape[0], 384, 384), dtype=torch.bool, device=image.device)
        self._feature, self._active = None, True
        try:
            self.net(factory(normalized, padding))
            feature = self._feature
            if feature is None or tuple(feature.shape) != (image.shape[0], 384, 24, 24):
                raise ValueError('Oczekiwano końcowej surowej siatki Bx384x24x24')
            return feature
        finally:
            self._feature, self._active = None, False

    def descriptors(self, image, mask):
        return descriptor_torch(self.features(image), mask)

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._feature = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_probe(net, means, stds, probe_manifest_path, *, device='cpu', batchsize=1,
                  out=None, save_particles=True, nested_tensor_factory=None):
    """Odczytaj stałą sondę; zgodne z callbackiem ``trening.py``.

    Zwraca descriptors[N,1536], labels[N,len(sonda.LABEL_NAMES)], label_names,
    group_id, role, variant. Nazwy i kolejność muszą zgadzać się ze schematem:
    pole label_names w rekordzie albo labels w sąsiednim probe.json.
    Jeżeli obecne są oba zapisy, oba muszą być zgodne z sonda.LABEL_NAMES.
    Opcjonalne tokens[576,384], token_weights[576] i particle_row dotyczą
    tylko pierwszego original w kolejności manifestu. ``out`` to plik NPZ.
    Hook znika także po błędzie. Funkcja nie przełącza train/eval ani nie
    przenosi modelu na inne urządzenie. Wywołujący ustala i przywraca tryb.
    """
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    if not isinstance(batchsize, int) or batchsize < 1:
        raise ValueError('batchsize musi być dodatnią liczbą całkowitą')
    manifest = Path(probe_manifest_path)
    records = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Manifest sondy musi być niepustą listą rekordów')
    if any(r.get('split', 'train') != 'train' or r['role'] not in ('fit', 'holdout') for r in records):
        raise ValueError('Sonda może zawierać wyłącznie grupy z puli train')
    label_names = list(sonda.LABEL_NAMES)
    metadata_path = manifest.parent/'probe.json'
    metadata_names = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata_names = metadata.get('labels')
        if metadata_names != label_names:
            raise ValueError('Nazwy etykiet w probe.json nie zgadzają się z sonda.LABEL_NAMES')
    if any(r.get('label_names', metadata_names) != label_names for r in records):
        raise ValueError('Pole label_names musi zgadzać się z kolejnością sonda.LABEL_NAMES')
    output = {key: np.asarray([r[key] for r in records]) for key in ('labels', 'group_id', 'role', 'variant')}
    if output['labels'].shape != (len(records), len(label_names)) or not np.isfinite(output['labels']).all():
        raise ValueError(f'Etykiety muszą mieć kształt Nx{len(label_names)} i skończone wartości')
    output['label_names'] = np.asarray(label_names)
    original = next((i for i, r in enumerate(records) if r['variant'] == 'original'), None)
    descriptors = []
    root = manifest.parent.resolve()

    def local_path(value):
        path = (root / value).resolve()
        if Path(value).is_absolute() or not path.is_relative_to(root):
            raise ValueError('Ścieżki sondy muszą być względne do manifestu')
        return path

    def load_image(row, key, mode):
        payload = local_path(row[key]).read_bytes()
        expected = row.get(key+'_sha256')
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Zmieniony plik stałej sondy: {row[key]}')
        with Image.open(io.BytesIO(payload)) as image:
            return np.array(image.convert(mode), copy=True)

    with DifferentiableReader(net, means, stds, nested_tensor_factory=nested_tensor_factory) as reader:
        with torch.no_grad():
            for start in range(0, len(records), batchsize):
                images, masks = [], []
                for row in records[start:start+batchsize]:
                    rgb = load_image(row, 'image', 'RGB')
                    mask = load_image(row, 'mask', 'L')
                    if mask.shape != rgb.shape[:2]:
                        raise ValueError('Obraz i maska muszą mieć zgodne wymiary')
                    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()/255
                    m = torch.from_numpy(mask)[None, None].float()/255
                    images.append(functional.interpolate(x, (384, 384), mode='bilinear', align_corners=False))
                    masks.append(functional.interpolate(m, FEATURE_GRID, mode='area'))
                x, m = torch.cat(images).to(device), torch.cat(masks).to(device)
                f = reader.features(x)
                descriptors.append(descriptor_torch(f, m).float().cpu().numpy())
                if save_particles and original is not None and start <= original < start+len(images):
                    index = original-start
                    output.update(tokens=f[index].permute(1, 2, 0).reshape(576, 384).float().cpu().numpy(),
                                  token_weights=m[index].reshape(576).float().cpu().numpy(),
                                  particle_row=np.asarray(original))
    output['descriptors'] = np.concatenate(descriptors)
    output['descriptor_version'] = np.asarray(DESCRIPTOR_VERSION)
    if out is not None:
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **output)
    return output
