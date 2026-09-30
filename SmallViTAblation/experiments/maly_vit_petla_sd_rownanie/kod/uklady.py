"""Układy scen dla Stable Diffusion (PLAN.md, wersja 4): wielokąty narzędzia, rogówki i źrenicy z adnotacji
Cataract-LMM 3_Object_Tracking, przekształcenie narzędzia, maski warunku SD i ramka w pikselach małego ViT.

Źródło: archiwa `<film>.zip` w MALY_VIT_ADNOTACJE (lokalnie Dataset/Cataract-LMM/3_Object_Tracking/annotations, na
Edenie ~/smallvit_data/cataract_lmm/3_Object_Tracking/annotations); czytany tylko `annotation.csv` (bez klatek).
`frame_number` w CSV liczony od 1, jak `frame_index` puli. Wielokąty w pikselach natywnych (720×480 albo 720×576).

Przepisy (PLAN.md):
  R1 - narzędzie obrócone wokół punktu wejścia (wierzchołek najdalej od środka źrenicy), przeskalowane i lekko
       przesunięte; rogówka i źrenica bez zmian;
  R2 - ten sam układ co w prawdziwej klatce (nowy wygląd z innego szumu).
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

ADN = Path(os.environ.get('MALY_VIT_ADNOTACJE', '.'))
OBIEKTY = ('cystotome_seg', 'cornea_seg', 'pupil_seg')
LSZER, LWYS = 72, 48                        # siatka latentu SD (576×384 / 8)
SD_SZER, SD_WYS = 576, 384
W_VIT, H_VIT = 480, 320
KAT_MAX, SKALA_ZAKRES, PRZES_MAX = 35.0, (0.85, 1.15), 20.0

_FILMY: dict[str, dict] = {}


def film_i_klatka(frame_id: str):
    v, f = frame_id.split('/')[-2:]
    return v, int(f)


def _wielokaty(tekst):
    if tekst in ('', '[]', None):
        return []
    return [np.asarray(p, np.float64).reshape(-1, 2) for p in json.loads(tekst) if len(p) >= 6]


def _film(film: str) -> dict:
    if film not in _FILMY:
        with zipfile.ZipFile(ADN / f'{film}.zip') as z:
            raw = z.read(f'{film}/annotation.csv').decode('utf-8-sig')
        _FILMY[film] = {int(r['frame_number']): r for r in csv.DictReader(io.StringIO(raw))}
    return _FILMY[film]


def uklad(wiersz_puli: dict):
    """Wielokąty klatki puli (natywne piksele) albo None, gdy brak narzędzia, rogówki lub adnotacji."""
    film, nr = film_i_klatka(wiersz_puli['frame_id'])
    r = _film(film).get(nr)
    if r is None:
        return None
    u = {k: _wielokaty(r.get(k, '')) for k in OBIEKTY}
    if not u['cystotome_seg'] or not u['cornea_seg']:
        return None
    u.update(w=int(wiersz_puli.get('native_width', 720)), h=int(wiersz_puli.get('native_height', 480)),
             frame_id=wiersz_puli['frame_id'])
    return u


def _srodek(polys):
    p = np.concatenate(polys)
    return p.mean(0)


def przeksztalc(u: dict, rng: np.random.Generator):
    """R1: obrót o kąt z [-35°, 35°], skala z [0,85; 1,15] wokół punktu wejścia narzędzia i przesunięcie do ±20 px.
    Zwraca nowy układ i parametry; narzędzie przycięte do kadru."""
    cel = _srodek(u['pupil_seg'] or u['cornea_seg'])
    pts = np.concatenate(u['cystotome_seg'])
    wej = pts[np.argmax(((pts - cel) ** 2).sum(1))]
    kat, s = rng.uniform(-KAT_MAX, KAT_MAX), rng.uniform(*SKALA_ZAKRES)
    dx, dy = rng.uniform(-PRZES_MAX, PRZES_MAX, 2)
    c, si = np.cos(np.radians(kat)), np.sin(np.radians(kat))
    R = np.array([[c, -si], [si, c]]) * s
    nowe = []
    for p in u['cystotome_seg']:
        q = (p - wej) @ R.T + wej + np.array([dx, dy])
        q[:, 0] = q[:, 0].clip(0, u['w'] - 1); q[:, 1] = q[:, 1].clip(0, u['h'] - 1)
        nowe.append(q)
    v = dict(u); v['cystotome_seg'] = nowe
    return v, dict(kat=float(kat), skala=float(s), dx=float(dx), dy=float(dy))


def maski(u: dict) -> np.ndarray:
    """Trzy maski warunku SD (narzędzie, rogówka, źrenica) jako udział pola w komórce 72×48, uint8 0-255
    (jak `dyfuzja_sd_filmy/kod/wspolne.maski`)."""
    out = np.zeros((3, LWYS, LSZER), np.uint8)
    for k, kol in enumerate(OBIEKTY):
        pelna = np.zeros((u['h'], u['w']), np.uint8)
        for p in u[kol]:
            pts = np.rint(p).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(pelna, [pts], 255)
        out[k] = cv2.resize(pelna, (LSZER, LWYS), interpolation=cv2.INTER_AREA)
    return out


def ramka_vit(u: dict) -> np.ndarray:
    """Ramka narzędzia xyxy w pikselach wejścia małego ViT (480×320), (1, 4) float32."""
    p = np.concatenate(u['cystotome_seg'])
    sx, sy = W_VIT / u['w'], H_VIT / u['h']
    return np.array([[p[:, 0].min() * sx, p[:, 1].min() * sy, (p[:, 0].max() + 1) * sx, (p[:, 1].max() + 1) * sy]],
                    np.float32)


def okno_wycinka(u: dict, bok_natywny: int = 256):
    """Kwadrat wokół środka ramki narzędzia w klatce 576×384, bok odpowiadający 256 px klatki natywnej: (x0, y0, b)."""
    p = np.concatenate(u['cystotome_seg'])
    sx, sy = SD_SZER / u['w'], SD_WYS / u['h']
    cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2 * sx, (p[:, 1].min() + p[:, 1].max()) / 2 * sy
    b = int(round(bok_natywny * sx))
    x0 = int(np.clip(round(cx - b / 2), 0, SD_SZER - b)); y0 = int(np.clip(round(cy - b / 2), 0, SD_WYS - b))
    return x0, y0, b


def wycinek_zapytania(obraz_sd: np.ndarray, u: dict, bok_natywny: int = 256) -> np.ndarray:
    """Wycinek wokół narzędzia z klatki 576×384 jako 256×256 RGB (jak wycinki FastGAN-a użyte do budowy indeksu
    DINOv2 puli)."""
    x0, y0, b = okno_wycinka(u, bok_natywny)
    return cv2.resize(obraz_sd[y0:y0 + b, x0:x0 + b], (256, 256), interpolation=cv2.INTER_AREA)


def maska_narzedzia_sd(u: dict) -> np.ndarray:
    """Maska narzędzia w pikselach klatki 576×384 (bool)."""
    m = np.zeros((SD_WYS, SD_SZER), np.uint8)
    s = np.array([SD_SZER / u['w'], SD_WYS / u['h']])
    for p in u['cystotome_seg']:
        pts = np.rint(p * s).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m > 0


def wycinek_176(obraz_sd: np.ndarray, u: dict):
    """Wycinek jak sonda (176×176, skala ≈ 2/3 klatki natywnej) i maska narzędzia w nim: do pomiaru siły tożsamości."""
    x0, y0, b = okno_wycinka(u)
    img = cv2.resize(obraz_sd[y0:y0 + b, x0:x0 + b], (176, 176), interpolation=cv2.INTER_AREA)
    m = maska_narzedzia_sd(u)[y0:y0 + b, x0:x0 + b].astype(np.uint8)
    return img, cv2.resize(m, (176, 176), interpolation=cv2.INTER_NEAREST) > 0
