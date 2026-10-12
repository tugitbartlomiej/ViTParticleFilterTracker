"""Pelne klatki i statyczne maski na siatce wejscia 480x320.

Maska: rasteryzacja cystotome_seg w oryginalnym rozmiarze (fillPoly,
zaokraglenie wspolrzednych jak test_ksztaltu.py), INTER_AREA do 480x320,
nastepnie >0. Coverage to udzial binarnych pikseli wejscia w polu 16x16.
area_coverage zachowuje dodatkowo udzial przed konserwatywna binaryzacja.
Tlo: zero pikseli po dylatacji binarnej maski WEJSCIA jadrem 17x17.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'wyniki/dane'
MANIFEST = ROOT / 'experiments/koopman_fastgan_petla/wyniki/local_1810448/run/probe/manifest.json'
ZIPS = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def token_mean(mask):
    assert mask.shape == (320, 480)
    return mask.reshape(20, 16, 30, 16).mean(axis=(1, 3), dtype=np.float32).reshape(600)


def prepare(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    originals = [r for r in json.loads(MANIFEST.read_text(encoding='utf-8')) if r['variant'] == 'original']
    assert len(originals) == 12 and len({r['frame_id'] for r in originals}) == 12
    images, inputs, binary_masks, area_masks, coverage, area_coverage, background = [], [], [], [], [], [], []
    for r in originals:
        film, frame = r['frame_id'].split('/')[-2:]
        archive = ZIPS / (film + '.zip')
        with zipfile.ZipFile(archive) as z:
            annotation_member = film + '/annotation.csv'
            annotation_bytes = z.read(annotation_member)
            rows = list(csv.DictReader(io.StringIO(annotation_bytes.decode('utf-8-sig'))))
            matches = [row for row in rows if int(row['frame_number']) == int(frame)]
            assert len(matches) == 1, (film, frame, len(matches))
            row = matches[0]
            member = film + '/' + row['filename']
            image_bytes = z.read(member)
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None and min(bgr.shape[:2]) >= 320, (member, None if bgr is None else bgr.shape)
        source_h, source_w = bgr.shape[:2]
        mask = np.zeros(bgr.shape[:2], np.uint8)
        polygons = json.loads(row['cystotome_seg'])
        for polygon in polygons:
            points = np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
        assert mask.any(), member
        area = cv2.resize(mask.astype(np.float32), (480, 320), interpolation=cv2.INTER_AREA)
        binary = (area > 0).astype(np.uint8)
        cov = token_mean(binary)
        dilated = cv2.dilate(binary, np.ones((17, 17), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        bg = token_mean(dilated) == 0
        assert ((cov > 0) & bg).sum() == 0 and bg.any()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs.append(cv2.resize(rgb, (480, 320), interpolation=cv2.INTER_AREA))
        binary_masks.append(binary.astype(bool)); area_masks.append(area)
        coverage.append(cov); area_coverage.append(token_mean(area)); background.append(bg)
        images.append(dict(frame_id=r['frame_id'], source_file=r['source_file'], film=film,
                           split=r.get('split'), role=r.get('role'), archive=str(archive.resolve()),
                           archive_member=member, annotation_member=annotation_member,
                           annotation_frame_number=int(row['frame_number']),
                           image_member_sha256=hashlib.sha256(image_bytes).hexdigest(),
                           annotation_csv_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                           cystotome_seg=polygons, original_size_wh=[source_w, source_h],
                           input_size_wh=[480, 320], original_mask_pixels=int(mask.sum()),
                           input_mask_pixels=int(binary.sum()), tool_tokens=int((cov > 0).sum()),
                           background_tokens=int(bg.sum()), tool25_tokens=int((cov >= .25).sum())))
        print(f"frame {film}/{frame}: tool={(cov > 0).sum()}, background={bg.sum()}", flush=True)
    X = np.stack(inputs)
    cov = np.stack(coverage).astype(np.float32)
    np.save(out / 'obrazy.npy', X, allow_pickle=False)
    np.savez_compressed(out / 'maski.npz', coverage=cov, tool=cov > 0,
                        background=np.stack(background), tool25=cov >= .25,
                        area_coverage=np.stack(area_coverage),
                        mask_input=np.stack(binary_masks), mask_input_area=np.stack(area_masks),
                        token_xy=np.stack(np.meshgrid(np.arange(30), np.arange(20)), -1).reshape(600, 2))
    save_json(out / 'images.json', images)
    return X, images


if __name__ == '__main__':
    prepare()
