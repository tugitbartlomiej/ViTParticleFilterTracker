"""Czy maski i zapisane tokeny daja zgodne wejscie do analizy czastek?

Brak jawnej zgodnosci frame_id, rozmiaru obrazu albo maski przerywa przygotowanie.
Nie podmieniamy segmentacji ramka. Nie uruchamia enkodera ani detektora.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len({r['frame_id'] for r in rows}) != len(rows):
        raise ValueError(f'Powtorzone frame_id: {path}')
    return rows


def export_masks(args):
    """Jawne mapowanie nazwy z CSV; szablon podaje uzytkownik, bez heurystyk."""
    rows = [r for r in jsonl(args.manifest) if r['split'] == args.split]
    output, missing, sources = [], [], {}
    for video in sorted({r['video_id'] for r in rows}):
        archive = args.archives / f'{video}.zip'
        with zipfile.ZipFile(archive) as z:
            raw = z.read(f'{video}/annotation.csv')
        sources[video] = hashlib.sha256(raw).hexdigest()
        records = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        index = {r['filename']: r for r in records}
        if len(index) != len(records):
            raise ValueError(f'Powtorzona nazwa obrazu w {archive}')
        for r in (r for r in rows if r['video_id'] == video):
            filename = args.filename_template.format(**r)
            a = index.get(filename)
            if a is None or a.get('cystotome_seg') in (None, '', '[]'):
                missing.append(r['frame_id'])
                continue
            output.append(dict(frame_id=r['frame_id'], width=r['native_width'],
                               height=r['native_height'], polygons=json.loads(a['cystotome_seg']),
                               source=f'{archive}!{video}/annotation.csv', filename=filename))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        for r in output:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    args.out.with_suffix('.audit.json').write_text(json.dumps(dict(
        manifest_sha256=sha256(args.manifest), csv_sha256=sources,
        exported=len(output), missing=missing, filename_template=args.filename_template), indent=2), encoding='utf-8')


def polygon_grid(annotation, grid):
    h, w = int(annotation['height']), int(annotation['width'])
    if min(h, w) <= 0:
        raise ValueError('Nieprawidlowy rozmiar maski')
    mask = np.zeros((h, w), np.uint8)
    for polygon in annotation['polygons']:
        pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError('Nieprawidlowy wielokat')
        if (pts < 0).any() or (pts[:, 0] > w).any() or (pts[:, 1] > h).any():
            raise ValueError('Wielokat poza obrazem; sprawdz uklad wspolrzednych')
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    if not mask.any():
        raise ValueError('Pusta segmentacja')
    gh, gw = grid
    weights = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA).ravel()
    ys, xs = np.nonzero(mask)
    box = np.array([xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], float)
    return weights, box


def prepare(args):
    meta = json.loads((args.recording / 'meta.json').read_text(encoding='utf-8'))
    grid = tuple(meta['grid'])
    manifest = {r['frame_id']: r for r in jsonl(args.manifest) if r['split'] == args.split}
    masks = {r['frame_id']: r for r in jsonl(args.masks)}
    rows, skipped, seen, source_hashes = [], [], set(), {}
    for path in sorted(args.recording.glob('czesc*.npz')):
        source_hashes[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as d:
            # NPZ dekompresuje tablice przy odczycie; robimy to raz na czesc, nie raz na klatke.
            ids = d['frame_ids'].astype(str)
            profiles, boxes, scores = d['profil_red'], d['pred_box'], d['score']
            if not (len(ids) == len(profiles) == len(boxes) == len(scores)):
                raise ValueError(f'Niezgodne dlugosci zapisu: {path}')
            for i, fid in enumerate(ids):
                if fid in seen:
                    raise ValueError(f'Powtorzony zapis klatki {fid}')
                seen.add(fid)
                if fid not in manifest:
                    continue
                r = manifest[fid]
                if fid not in masks or r.get('tip_point') is None:
                    skipped.append(dict(frame_id=fid, reason='missing_segmentation_or_tip'))
                    continue
                m = masks[fid]
                if (m['width'], m['height']) != (r['native_width'], r['native_height']):
                    raise ValueError(f'Rozmiar maski rozny od manifestu: {fid}')
                weights, gt_box = polygon_grid(m, grid)
                if (weights > 0).sum() < 2 or (weights == 0).sum() < 2:
                    skipped.append(dict(frame_id=fid, reason='insufficient_instrument_or_background_tokens'))
                    continue
                z = profiles[i].astype(np.float32)
                if z.shape[0] != np.prod(grid) or not np.isfinite(z).all():
                    raise ValueError(f'Nieprawidlowe tokeny: {fid}')
                rows.append(dict(frame_id=fid, operation=str(r['group_id']), video=str(r['video_id']),
                                 number=int(r['frame_index']), tokens=z, mask=weights, gt_box=gt_box,
                                 pred_box=boxes[i], confidence=scores[i],
                                 tip=r['tip_point'], size=[r['native_width'], r['native_height']]))
    if not rows:
        raise ValueError('Brak zgodnych klatek. Nie zgadujemy mapowania identyfikatorow.')
    rows.sort(key=lambda r: (r['operation'], r['video'], r['number']))
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays['grid'] = np.asarray(grid)
    for key in ('tokens', 'mask', 'gt_box', 'pred_box', 'confidence', 'tip', 'size'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Niefinitywne wejscie: {key}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as f:
        np.savez_compressed(f, **arrays)
    audit = dict(status='prepared', split=args.split, frames=len(rows), skipped=skipped,
                 missing_recording=sorted(set(manifest) - seen), recording_meta=meta,
                 sha256={**source_hashes, str(args.manifest): sha256(args.manifest),
                         str(args.masks): sha256(args.masks), str(args.recording / 'baza.npz'): sha256(args.recording / 'baza.npz')},
                 warning='PCA i determinism guard pochodza ze starego rejestratora; eksploracyjny pomiar.')
    args.out.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export-masks')
    e.add_argument('--archives', type=Path, required=True)
    e.add_argument('--filename-template', required=True, help='Dokladna nazwa CSV, np. {frame_index:08d}.jpg po sprawdzeniu zrodla')
    e.set_defaults(action=export_masks)
    r = sub.add_parser('prepare')
    r.add_argument('--recording', type=Path, required=True)
    r.add_argument('--masks', type=Path, required=True)
    r.set_defaults(action=prepare)
    for parser in (e, r):
        parser.add_argument('--manifest', type=Path, required=True)
        parser.add_argument('--split', choices=['train', 'val'], required=True)
        parser.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
