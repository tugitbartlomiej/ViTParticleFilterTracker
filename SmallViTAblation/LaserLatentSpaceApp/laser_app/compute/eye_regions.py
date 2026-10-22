"""Obszary oka na patchach: narzędzie, źrenica, rogówka, tło z wielokątów segmentacji Cataract-LMM.

Każdy film 3_Object_Tracking ma w zipie `annotation.csv` z wielokątami `<narzędzie>_seg`, `cornea_seg`, `pupil_seg`
(współrzędne natywne klatki). Dla każdego zdjęcia osi czasu (frame_id = cataract_lmm/tracking/<film>/<klatka>)
wielokąty są skalowane do 480 × 320 (osobno X i Y, jak obraz wejściowy ViT), rysowane i uśredniane na siatce
30 × 20 patchy po 16 px. Klasa patcha (pierwszeństwo w tej kolejności):
  3 narzędzie  pokrycie wielokątem narzędzia ≥ PROG_NARZ
  2 źrenica    pokrycie źrenicy ≥ PROG_OBSZAR
  1 rogówka    pokrycie rogówki ≥ PROG_OBSZAR (poza źrenicą)
  0 tło        reszta (także, gdy klatka nie ma danego wielokąta)
Wynik jest zapisywany obok osi czasu (`obszary_oka.npz`) i liczony ponownie tylko, gdy zmieni się lista zdjęć.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from laser_app.paths import PROJECT_ROOT as ROOT  # noqa: E402
ZIPY = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
PROG_NARZ = 0.25
PROG_OBSZAR = 0.5
KLASY = ('tło', 'rogówka', 'źrenica', 'narzędzie')
WERSJA = 1
csv.field_size_limit(1 << 30)


def _wielokaty(tekst):
    """'[[x1, y1, x2, y2, ...], [...]]' -> lista tablic (k, 2); pusty tekst -> []."""
    if not tekst or not tekst.strip():
        return []
    v = json.loads(tekst)
    if v and isinstance(v[0], (int, float)):
        v = [v]
    return [np.asarray(p, np.float64).reshape(-1, 2) for p in v if len(p) >= 6]


def _film_csv(film, cache):
    if film not in cache:
        z = zipfile.ZipFile(ZIPY / f'{film}.zip')
        nazwa = next(n for n in z.namelist() if n.endswith('annotation.csv'))
        wiersze = list(csv.DictReader(io.StringIO(z.read(nazwa).decode('utf-8', 'replace'))))
        cache[film] = {int(r['frame_number']): r for r in wiersze}
    return cache[film]


def _pokrycie(wielokaty, sx, sy):
    """Ułamek każdego z 600 patchy (20 × 30) pokryty wielokątami po przeskalowaniu do 480 × 320."""
    import cv2
    if not wielokaty:
        return np.zeros(600, np.float32)
    m = np.zeros((320, 480), np.uint8)
    cv2.fillPoly(m, [np.round(p * (sx, sy)).astype(np.int32) for p in wielokaty], 1)
    return m.reshape(20, 16, 30, 16).mean((1, 3)).reshape(-1).astype(np.float32)


def klasy_zdjecia(frame_id, szer, wys, cache):
    """(klasy (600,) uint8, pokrycia (600, 3) float32 [narzędzie, źrenica, rogówka]) albo None bez wpisu w csv."""
    _, _, film, klatka = frame_id.split('/')
    try:
        r = _film_csv(film, cache).get(int(klatka))
    except (FileNotFoundError, StopIteration, KeyError):
        return None
    if r is None:
        return None
    sx, sy = 480 / szer, 320 / wys
    narz_kol = next((k for k in r if k.endswith('_seg') and k not in ('cornea_seg', 'pupil_seg')), None)
    pn = _pokrycie(_wielokaty(r.get(narz_kol) or ''), sx, sy) if narz_kol else np.zeros(600, np.float32)
    pz = _pokrycie(_wielokaty(r.get('pupil_seg') or ''), sx, sy)
    pr = _pokrycie(_wielokaty(r.get('cornea_seg') or ''), sx, sy)
    k = np.zeros(600, np.uint8)
    k[pr >= PROG_OBSZAR] = 1
    k[pz >= PROG_OBSZAR] = 2
    k[pn >= PROG_NARZ] = 3
    return k, np.stack([pn, pz, pr], 1)


def dla_osi(timeline, postep=lambda a, n: None):
    """Klasy patchy wszystkich zdjęć osi czasu (n_obrazów × 600) uint8 i pokrycia; z pamięci podręcznej obok osi."""
    obrazy = (getattr(timeline, 'metadata', None) or {}).get('images') or []
    if len(obrazy) != timeline.image_count or not all('frame_id' in r for r in obrazy):
        raise ValueError('oś czasu bez frame_id zdjęć: obszary oka niedostępne')
    klucz = hashlib.sha256(json.dumps([WERSJA, PROG_NARZ, PROG_OBSZAR] +
                                      [r['frame_id'] for r in obrazy]).encode()).hexdigest()[:16]
    root = getattr(timeline, 'root', None)
    plik = Path(root) / 'obszary_oka.npz' if root is not None else None
    if plik is not None and plik.is_file():
        z = np.load(plik)
        if str(z['klucz']) == klucz:
            return z['klasy'], z['pokrycia'], z['brak']
    cache = {}
    klasy = np.zeros((len(obrazy), 600), np.uint8)
    pokr = np.zeros((len(obrazy), 600, 3), np.float16)
    brak = np.zeros(len(obrazy), bool)
    for i, r in enumerate(obrazy):
        szer = r.get('source_width') or r.get('width')
        wys = r.get('source_height') or r.get('height')
        w = klasy_zdjecia(r['frame_id'], szer, wys, cache)
        if w is None:
            brak[i] = True
        else:
            klasy[i], pokr[i] = w[0], w[1]
        postep(i + 1, len(obrazy))
    if plik is not None:
        try:
            np.savez_compressed(plik, klasy=klasy, pokrycia=pokr, brak=brak, klucz=np.array(klucz))
        except OSError:
            pass
    return klasy, pokr, brak
