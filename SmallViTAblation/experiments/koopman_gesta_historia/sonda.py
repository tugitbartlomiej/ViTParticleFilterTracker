"""Deterministyczna sonda z RGBA FastGAN: prawdziwe RGB i ręczna maska.

create_probe zwraca Path do manifest.json będącego listą rekordów. Obrazy
i maski są względne do katalogu manifestu. Konfigurację zapisuje probe.json.
Domyślnie wybieramy po jednej klatce z 12 operacji train: 8 fit i 4 holdout.
Każda operacja ma szesnaście wariantów; ten sam zapis służy wszystkim wagom.

Rotacje, przesunięcia, skala, pozorna długość i perspektywa dotyczą całego
płaskiego wycinka i maski. Mnożnik jasności działa wyłącznie w ręcznej masce.
Długość zmieniamy wzdłuż głównej osi ręcznej maski. To kontrola transformacji
wycinka 2D, nie nowy widok 3D. Etykiety kąta/skali/długości są względne do
oryginału; nie oznaczają fizycznej orientacji ani rozmiaru metalu.
foreground_brightness to zadany mnożnik przed clippingiem, a nie zmierzona
jasność. dx/dy to przesunięcia podzielone przez szerokość/wysokość obrazu.
perspective_strength to s w homografii u'=u/(1+s*u), v'=v/(1+s*u), gdzie
u=(x-W/2)/W i v=(y-H/2)/H. Nie modeluje niewidocznych powierzchni obiektu.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

LABEL_NAMES = ('presence', 'sinangle', 'cosangle', 'logscale', 'lengthfactor', 'backgroundbrightness',
               'foreground_brightness', 'dx', 'dy', 'perspective_strength')
VARIANTS = ('original', 'rotation90', 'rotation180', 'rotation270', 'black', 'yellow',
            'scale08', 'length08', 'empty_black', 'shift_left', 'shift_right', 'shift_up', 'shift_down',
            'foreground_dim', 'foreground_bright', 'perspective_mild')


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _axis(mask):
    y, x = np.indices(mask.shape, dtype=float)
    points = np.stack([x.ravel()+.5, y.ravel()+.5], axis=1)
    weights = mask.ravel()/mask.sum()
    centered = points - (points*weights[:, None]).sum(0)
    values, vectors = np.linalg.eigh((centered*weights[:, None]).T @ centered)
    if values[-1]-values[0] < 1e-10:
        return np.array([1., 0.])
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis


def _about_center(linear, size):
    center = np.array([size/2, size/2])
    matrix = np.eye(3)
    matrix[:2, :2], matrix[:2, 2] = linear, center-linear@center
    return matrix


def _perspective(size, strength):
    """Homografia w układzie środka wycinka; strength nie jest kątem 3D."""
    normalized = np.array([[1/size, 0, -.5], [0, 1/size, -.5], [0, 0, 1.]])
    projective = np.eye(3)
    projective[2, 0] = strength
    matrix = np.linalg.inv(normalized) @ projective @ normalized
    return matrix/matrix[2, 2]


def _warp_pair(rgb, alpha, matrix, *, projective=False):
    """Ta sama odwrotna mapa i interpolacja dla RGB oraz ręcznej maski."""
    from PIL import Image

    inverse = np.linalg.inv(matrix)
    inverse = inverse/inverse[2, 2]
    coefficients = inverse.ravel()[:8 if projective else 6].tolist()
    method = Image.Transform.PERSPECTIVE if projective else Image.Transform.AFFINE
    size = (rgb.shape[1], rgb.shape[0])
    image = Image.fromarray(rgb).transform(size, method, coefficients,
                                         resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    mask = Image.fromarray(alpha).transform(size, method, coefficients,
                                          resample=Image.Resampling.BILINEAR, fillcolor=0)
    return np.array(image), np.array(mask)


def _expected_support_mass(support, matrix):
    """Masa bez obcięcia: całka maski z lokalnym Jacobianem homografii.

    Dla afinicznego przekształcenia redukuje się do masy razy |det(A)|.
    Dla perspektywy to przybliżenie po środkach pikseli, przed rasteryzacją.
    """
    yy, xx = np.indices(support.shape, dtype=float)
    denominator = matrix[2, 0]*(xx+.5)+matrix[2, 1]*(yy+.5)+matrix[2, 2]
    jacobian = abs(np.linalg.det(matrix))/np.abs(denominator)**3
    return float((support*jacobian).sum())


def create_probe(crop_manifest, crop_root, out, n_groups=12, seed=42):
    """Utwórz stałą sondę i zwróć Path(out/'manifest.json').

    Input: {'records': [{'file': względne_RGBA_png, 'frame_id': ..., 'group_id':
    ..., 'split': 'train'}]}. Grupa występująca również poza train zostaje
    całkowicie wykluczona bez otwierania jej danych. n_groups musi być dodatnią
    wielokrotnością 3 (2/3 fit, 1/3 holdout). Istniejący katalog out jest błędem.
    """
    from PIL import Image

    if not isinstance(n_groups, int) or n_groups < 3 or n_groups % 3:
        raise ValueError('n_groups musi być wielokrotnością 3, co najmniej 3')
    out, crop_root, crop_manifest = Path(out), Path(crop_root).resolve(), Path(crop_manifest)
    if out.exists():
        raise FileExistsError(out)
    source_bytes = crop_manifest.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or not isinstance(source.get('records'), list):
        raise ValueError('Wymagany manifest FastGAN z listą records')
    groups, forbidden = {}, set()
    for row in source['records']:
        if not all(key in row for key in ('file', 'frame_id', 'group_id', 'split')):
            raise ValueError('Rekord wymaga file, frame_id, group_id, split')
        group = str(row['group_id'])
        if row['split'] != 'train':
            forbidden.add(group)
        else:
            groups.setdefault(group, []).append(row)
    eligible = sorted(set(groups)-forbidden)
    if len(eligible) < n_groups:
        raise ValueError(f'Za mało niezależnych grup train: {len(eligible)} < {n_groups}')
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(eligible)[:n_groups].tolist()
    selected = []
    for group in chosen:
        candidates = sorted(groups[group], key=lambda r: (str(r['frame_id']), str(r['file'])))
        row = candidates[int(rng.integers(len(candidates)))]
        path = (crop_root/row['file']).resolve()
        if Path(row['file']).is_absolute() or not path.is_relative_to(crop_root):
            raise ValueError('Ścieżka crop musi być względna i pozostać w crop_root')
        with Image.open(path) as image:
            if image.mode != 'RGBA' or image.format != 'PNG' or image.width != image.height:
                raise ValueError('Wymagany kwadratowy PNG RGBA z rzeczywistą ręczną maską')
            rgba = np.array(image, copy=True)
        occupancy = rgba[..., 3].astype(float)/255
        if not 0 < occupancy.sum() < occupancy.size:
            raise ValueError('Ręczna maska musi zawierać foreground i background')
        selected.append((row, rgba, hashlib.sha256(path.read_bytes()).hexdigest()))

    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    (out/'masks').mkdir()
    records = []
    for group_index, (source_row, rgba, source_sha) in enumerate(selected):
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        support = alpha.astype(float)/255
        size, axis = len(rgb), _axis(support)
        for variant in VARIANTS:
            matrix, angle, scale, length, presence = np.eye(3), 0, 1., 1., 1
            foreground_brightness, dx, dy, perspective_strength = 1., 0., 0., 0.
            transform_type, photometric_scope = 'affine_2d', 'none'
            image, mask = rgb.copy(), alpha.copy()
            if variant.startswith('rotation'):
                angle = int(variant.removeprefix('rotation'))
                quarter = angle//90
                image, mask = np.rot90(rgb, quarter).copy(), np.rot90(alpha, quarter).copy()
                linear = np.linalg.matrix_power(np.array([[0., 1.], [-1., 0.]]), quarter)
                matrix = _about_center(linear, size)
            elif variant in ('scale08', 'length08'):
                if variant == 'scale08':
                    scale, linear = .8, np.eye(2)*.8
                else:
                    length, linear = .8, np.eye(2)-.2*np.outer(axis, axis)
                matrix = _about_center(linear, size)
                # Cały wycinek: bez wklejania prostokąta z obcym tłem.
                image, mask = _warp_pair(rgb, alpha, matrix)
            elif variant.startswith('shift_'):
                dx, dy = {'shift_left': (-.1, 0.), 'shift_right': (.1, 0.),
                          'shift_up': (0., -.1), 'shift_down': (0., .1)}[variant]
                matrix[:2, 2] = [dx*size, dy*size]
                image, mask = _warp_pair(rgb, alpha, matrix)
            elif variant in ('foreground_dim', 'foreground_bright'):
                foreground_brightness = .7 if variant == 'foreground_dim' else 1.3
                photometric_scope = 'hand_mask_only'
                lit = np.clip(rgb.astype(float)*foreground_brightness, 0, 255)
                image = np.rint(rgb*(1-support[..., None])+lit*support[..., None]).astype(np.uint8)
            elif variant == 'perspective_mild':
                perspective_strength, transform_type = .15, 'homography_2d'
                matrix = _perspective(size, perspective_strength)
                image, mask = _warp_pair(rgb, alpha, matrix, projective=True)
            elif variant in ('black', 'yellow'):
                color = np.array([255, 255, 0] if variant == 'yellow' else [0, 0, 0])
                image = np.rint(rgb*support[..., None]+color*(1-support[..., None])).astype(np.uint8)
            elif variant == 'empty_black':
                image.fill(0)
                presence = 0
            weights = mask.astype(float)/255
            background = 1-weights
            luminance = image.astype(float) @ np.array([.2126, .7152, .0722])/255
            brightness = float((luminance*background).sum()/background.sum())
            radians = np.deg2rad(angle)
            labels = [presence, float(np.sin(radians)), float(np.cos(radians)),
                      float(np.log(scale)), length, brightness, foreground_brightness,
                      dx, dy, perspective_strength]
            stem = f'{group_index:02d}_{variant}'
            image_name, mask_name = f'images/{stem}.png', f'masks/{stem}.png'
            Image.fromarray(image).save(out/image_name)
            Image.fromarray(mask).save(out/mask_name)
            expected_mass = _expected_support_mass(support, matrix)
            records.append(dict(
                image=image_name, mask=mask_name, group_id=str(source_row['group_id']),
                frame_id=source_row['frame_id'], split='train',
                role='fit' if group_index < n_groups*2//3 else 'holdout', variant=variant,
                labels=labels, label_names=LABEL_NAMES, transform_matrix=matrix.tolist(),
                inverse_transform_matrix=np.linalg.inv(matrix).tolist(),
                transform_scope='crop_2d', anchor_xy=[size/2, size/2], length_axis_xy=axis.tolist(),
                transform_type=transform_type, new_3d_view=False, photometric_scope=photometric_scope,
                semantic_presence=presence, mask_occupancy=float(weights.mean()),
                mask_semantics='spatial_support_only' if not presence else 'transformed_hand_annotation',
                support_retained_fraction=float(min(1., weights.sum()/expected_mass)),
                source_file=source_row['file'], source_rgba_sha256=source_sha,
                image_sha256=hashlib.sha256((out/image_name).read_bytes()).hexdigest(),
                mask_sha256=hashlib.sha256((out/mask_name).read_bytes()).hexdigest()))
    manifest = out/'manifest.json'
    _write_json(manifest, records)
    _write_json(out/'probe.json', dict(
        schema_version=2, seed=seed, n_groups=n_groups, fit_groups=n_groups*2//3,
        holdout_groups=n_groups//3, labels=LABEL_NAMES, variants=VARIANTS,
        source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        fixed_across_checkpoints=True, final_test_data_used=False,
        transform_coordinates='Środki pikseli (x+.5,y+.5), x w prawo, y w dół; kąty CCW.',
        foreground_brightness_definition='Mnożnik RGB w ręcznej masce, clipping do [0,255] przed mieszaniem alfa; poza maską bez zmian.',
        shift_definition='dx=przesunięcie_x/szerokość, dy=przesunięcie_y/wysokość; dodatnie w prawo i w dół.',
        perspective_definition='u=(x-W/2)/W, v=(y-H/2)/H; u_new=u/(1+s*u), v_new=v/(1+s*u); s=0.15. Płaska homografia 2D, bez nowego widoku 3D.',
        support_retention_definition='Masa maski po rasteryzacji / masa przed obcięciem liczona z Jacobianu; przybliżenie ograniczone do 1.',
        note='Etykiety transformacji względem oryginału. Kontrola empty ma diagnostyczny ROI, nie GT narzędzia, i presence=0.'))
    return manifest
