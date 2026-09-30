"""Wspólne stałe i funkcje: klatki, maski i rozproszone uruchomienie (PLAN.md).

Pułapki danych Cataract-LMM 3_Object_Tracking:
- archiwum `<film>.zip` ma katalog `<film>/` z plikami `<film>_NNNNNN.jpg` i `annotation.csv`; kolumna `filename`
  w CSV to nazwa JPEG, `frame_number` liczony od 1;
- wielokąty w kolumnach `*_seg` to napis JSON: lista list [x0, y0, x1, y1, …] w pikselach natywnych; pusta = '' lub '[]';
- część filmów ma klatki 720×576 zamiast 720×480; wszystkie ściskamy do 576×384 (jak wejście małego ViT do 480×320);
- `cystotome_seg` bywa pusty (narzędzie poza kadrem), rogówka i źrenica są prawie zawsze.
"""
from __future__ import annotations

import csv
import io
import json
import os

import cv2
import numpy as np

SZER, WYS = 576, 384                    # obraz dla SD (proporcje 3:2 jak 720×480)
LSZER, LWYS = SZER // 8, WYS // 8       # latent 72×48
SKALA_VAE = 0.18215
OBIEKTY = ('cystotome_seg', 'cornea_seg', 'pupil_seg')


def dekoduj(jpeg: bytes):
    """JPEG → (obraz 576×384 RGB uint8, (szer, wys) natywne)."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:                                                    # uszkodzony plik: czarna klatka, nie przerywa filmu
        return np.zeros((WYS, SZER, 3), np.uint8), (720, 480)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (SZER, WYS), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb, (w, h)


def maski(wiersz: dict | None, w: int, h: int, sz: int = LSZER, wy: int = LWYS):
    """Trzy maski (narzędzie, rogówka, źrenica) jako udział pola w komórce siatki sz×wy, uint8 0–255."""
    out = np.zeros((len(OBIEKTY), wy, sz), np.uint8)
    if wiersz is None:
        return out
    for k, kol in enumerate(OBIEKTY):
        tekst = wiersz.get(kol, '')
        if tekst in ('', '[]', None):
            continue
        pelna = np.zeros((h, w), np.uint8)
        for p in json.loads(tekst):
            pts = np.rint(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(pelna, [pts], 255)
        out[k] = cv2.resize(pelna, (sz, wy), interpolation=cv2.INTER_AREA)
    return out


def adnotacje(z, film: str):
    """annotation.csv z archiwum filmu: słownik nazwa JPEG → wiersz."""
    raw = z.read(f'{film}/annotation.csv').decode('utf-8-sig')
    return {r['filename']: r for r in csv.DictReader(io.StringIO(raw))}


def rozproszenie():
    """(rank, world, local_rank) z torchrun; bez torchrun (0, 1, 0)."""
    return int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('LOCAL_RANK', 0))


def filmy_z_manifestu(sciezka: str, podzial: str):
    ids = set()
    with open(sciezka, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if r['split'] == podzial:
                    ids.add(r['video_id'])
    return sorted(ids)
