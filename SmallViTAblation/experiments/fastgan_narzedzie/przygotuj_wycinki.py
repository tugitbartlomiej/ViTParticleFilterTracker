"""Przygotowanie prawdziwych wycinkow RGB i segmentacji; tylko split train."""
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'dane/pilot_1000'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = ROOT / 'feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['split'] == 'train']
    groups = sorted({r['video_id'] for r in rows})
    rng = np.random.default_rng(42)
    quota = math.ceil(1000 / len(groups))
    OUT.mkdir(parents=True)
    records, skipped = [], []
    for video in groups:
        candidates = [r for r in rows if r['video_id'] == video]
        rng.shuffle(candidates)
        archive = ROOT.parent / f'Dataset/Cataract-LMM/3_Object_Tracking/annotations/{video}.zip'
        with zipfile.ZipFile(archive) as z:
            raw_csv = z.read(f'{video}/annotation.csv')
            annotations = {r['filename']: r for r in csv.DictReader(io.StringIO(raw_csv.decode('utf-8-sig')))}
            chosen = []
            for r in candidates:
                if len(chosen) >= quota:
                    break
                number = r['frame_index']
                if any(abs(number - n) < 20 for n in chosen):
                    continue
                name = f'{video}_{number:06d}.jpg'
                ann = annotations.get(name)
                if not ann or ann.get('cystotome_seg') in (None, '', '[]'):
                    skipped.append(r['frame_id'])
                    continue
                local = ROOT / f'feature_cache/lokalny_static36/images/{video}_{number:08d}.jpg'
                image_bytes = local.read_bytes() if local.exists() else z.read(f'{video}/{name}')
                rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if r.get('pixel_sha256') != digest:
                    raise ValueError(f'Niezgodne piksele: {r["frame_id"]}')
                h, w = rgb.shape[:2]
                if (w, h) != (r['native_width'], r['native_height']):
                    raise ValueError('Rozmiar obrazu niezgodny z manifestem')
                mask = np.zeros((h, w), np.uint8)
                for polygon in json.loads(ann['cystotome_seg']):
                    pts = np.asarray(polygon, np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 255)
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    continue
                # Kwadrat z marginesem, bez zmiany proporcji narzedzia. Brzeg dopelniamy.
                side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 1.3))
                x0 = int((xs.min() + xs.max() + 1 - side) / 2)
                y0 = int((ys.min() + ys.max() + 1 - side) / 2)
                pad = (max(-y0, 0), max(y0 + side - h, 0), max(-x0, 0), max(x0 + side - w, 0))
                im = cv2.copyMakeBorder(rgb, *pad, cv2.BORDER_REPLICATE)
                mm = cv2.copyMakeBorder(mask, *pad, cv2.BORDER_CONSTANT, value=0)
                px, py = x0 + pad[2], y0 + pad[0]
                im = cv2.resize(im[py:py+side, px:px+side], (256, 256), interpolation=cv2.INTER_AREA)
                mm = cv2.resize(mm[py:py+side, px:px+side], (256, 256), interpolation=cv2.INTER_NEAREST)
                if not mm.any():
                    continue
                filename = f'{video}_{number:08d}.png'
                Image.fromarray(np.dstack([im, mm]), mode='RGBA').save(OUT / filename)
                records.append(dict(file=filename, frame_id=r['frame_id'], group_id=r['group_id'],
                                    split='train', pixel_sha256_original=digest,
                                    annotation_csv_sha256=hashlib.sha256(raw_csv).hexdigest(),
                                    crop_xywh=[x0, y0, side, side]))
                chosen.append(number)
        print(f'{video}: {len(chosen)} wycinkow; razem {len(records)}', flush=True)
    if len(records) < 100:
        raise ValueError('Za malo poprawnych wycinkow')
    audit = dict(seed=42, requested_approx=1000, actual=len(records), min_frame_gap=20,
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 source='cystotome_seg; no bounding-box masks', records=records, skipped=skipped)
    (OUT / 'MANIFEST.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    preview = Image.new('RGB', (256*4, 256*2))
    for i, record in enumerate(records[::max(1, len(records)//8)][:8]):
        a = np.asarray(Image.open(OUT / record['file']))
        im = a[:, :, :3].copy()
        m = a[:, :, 3] > 127
        im[m] = (.6 * im[m] + .4 * np.array([255, 0, 0])).astype(np.uint8)
        preview.paste(Image.fromarray(im), ((i % 4) * 256, (i // 4) * 256))
    preview.save(OUT / 'PODGLAD.jpg')
    print(json.dumps(dict(frames=len(records), output=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
