"""Etykieta narzędzia z obrysu (wielokąty `<narzędzie>_seg` w annotation.csv Cataract-LMM 3_Object_Tracking), nie z ramki.

Dla klatki: udział pikseli narzędzia w każdej z 600 łat 16 × 16 wejścia 480 × 320 (wszystkie narzędzia oznaczone na klatce,
bez rogówki i źrenicy). Tokeny narzędzia = udział ≥ próg (domyślnie 0,5), mieszane = 0 < udział < próg.
Historia aplikacji nie zna nazwy pliku, więc klatkę znajdujemy po ramce GT (indeks ramek wszystkich filmów, budowany raz
i zapisywany w cache_uczenia/etykiety); przy kilku kandydatach rozstrzyga podobieństwo obrazu.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import film as FI

H, W, KOMORKA = 20, 30, 16
SKALA = (W * KOMORKA / 720, H * KOMORKA / 480)            # natywne 720 × 480 → wejście 480 × 320
POMIJANE = ('cornea', 'pupil')
PROG = 0.5
CACHE = Path(__file__).resolve().parent / 'cache_uczenia' / 'etykiety'


def kategorie(wiersz):
    """Narzędzia oznaczone w wierszu annotation.csv (z niepustą ramką), bez rogówki i źrenicy."""
    return [k[:-len('_bbox')] for k in wiersz if k.endswith('_bbox') and k[:-len('_bbox')] not in POMIJANE and FI._json(wiersz[k])]


def _wielokaty(wiersz, kat):
    v = FI._json(wiersz.get(kat + '_seg'))
    if not v:
        return []
    if isinstance(v[0], (int, float)):
        v = [v]
    return [np.asarray(p, np.float32).reshape(-1, 2) * SKALA for p in v if len(p) >= 6]


def udzial_z_wielokatow(wielokaty):
    """float32 (600,): ułamek pikseli każdej łaty pokryty wielokątami (w pikselach 480 × 320)."""
    import cv2
    m = np.zeros((H * KOMORKA, W * KOMORKA), np.uint8)
    for p in wielokaty:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.reshape(H, KOMORKA, W, KOMORKA).mean((1, 3)).reshape(-1).astype(np.float32)


@lru_cache(maxsize=8)
def _etykiety_filmu(film):
    return FI.etykiety(film)


def etykieta(film, klatka):
    """dict: udzial (600,), kategorie, ramki {kat: xyxy w 480 × 320}, czubki, piksele; None, gdy klatka bez narzędzia/obrysu."""
    w = _etykiety_filmu(film).get(int(klatka))
    if w is None:
        return None
    kat = kategorie(w)
    wiel = [p for k in kat for p in _wielokaty(w, k)]
    if not wiel:
        return None
    ramki = {}
    for k in kat:
        b = FI._json(w[k + '_bbox'])
        ramki[k] = [b[0] * SKALA[0], b[1] * SKALA[1], (b[0] + b[2]) * SKALA[0], (b[1] + b[3]) * SKALA[1]]
    u = udzial_z_wielokatow(wiel)
    return dict(film=film, klatka=int(klatka), udzial=u, kategorie=kat, ramki=ramki, piksele=float(u.sum() * KOMORKA ** 2))


def maski(udzial, prog=PROG):
    """(narzędzie, mieszane) bool (600,) z udziału pikseli."""
    u = np.asarray(udzial, np.float32)
    return u >= prog, (u > 0) & (u < prog)


# ------------------------------------------------------------------ odnajdywanie klatki po ramce GT (historie aplikacji)
def _indeks():
    """Ramki wszystkich narzędzi we wszystkich filmach: (film, klatka, x0, y0, x1, y1 w 480 × 320); raz, potem z dysku."""
    p = CACHE / 'indeks_ramek.npz'
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return d['film'], d['klatka'], d['ramka']
    filmy, klatki, ramki = [], [], []
    for z in sorted(FI.ADNOTACJE.glob('*.zip')):
        f = z.stem
        for kl, w in FI.etykiety(f).items():
            for k in kategorie(w):
                b = FI._json(w[k + '_bbox'])
                filmy.append(f); klatki.append(kl)
                ramki.append([b[0] * SKALA[0], b[1] * SKALA[1], (b[0] + b[2]) * SKALA[0], (b[1] + b[3]) * SKALA[1]])
    CACHE.mkdir(parents=True, exist_ok=True)
    out = dict(film=np.array(filmy), klatka=np.array(klatki, np.int32), ramka=np.array(ramki, np.float32))
    np.savez(p, **out)
    return out['film'], out['klatka'], out['ramka']


def znajdz(gt, rgb=None, tol=0.6):
    """(film, klatka) klatki z ramką narzędzia równą gt (xyxy 480 × 320, ±tol px); przy kilku kandydatach najbliższy obraz rgb."""
    f, k, r = _indeks()
    ok = np.flatnonzero(np.abs(r - np.asarray(gt, np.float32)).max(1) <= tol)
    kand = sorted({(str(f[i]), int(k[i])) for i in ok})
    if len(kand) <= 1 or rgb is None:
        return kand[0] if kand else None
    import zipfile, cv2
    best = None
    for film, kl in kand:
        with zipfile.ZipFile(FI.ADNOTACJE / f'{film}.zip') as z:
            raw = z.read(f'{film}/{film}_{kl:06d}.jpg')
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W * KOMORKA, H * KOMORKA), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = float(np.abs(im - np.asarray(rgb, np.float32)).mean())
        if best is None or e < best[0]:
            best = (e, film, kl)
    return best[1], best[2]


def dla_historii(h):
    """Etykieta z obrysu dla historii lasery_dane (po ramce GT i obrazie); wynik zapisywany w h.katalog. None, gdy brak."""
    p = Path(h.katalog) / 'etykieta_obrys.npz'
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return dict(film=str(d['film']), klatka=int(d['klatka']), udzial=d['udzial'], kategorie=[str(s) for s in d['kategorie']])
    if getattr(h, 'gt', None) is None:
        return None
    fk = znajdz(h.gt, getattr(h, 'rgb', None))
    if fk is None:
        return None
    e = etykieta(*fk)
    if e is None:
        return None
    np.savez(p, film=e['film'], klatka=e['klatka'], udzial=e['udzial'], kategorie=np.array(e['kategorie']))
    return e


def przesuniecia(maska, wyklucz=None):
    """Zero przestrzenne dla dowolnego kształtu: maska tokenów przesunięta o każde (dw, dk), cała w siatce i rozłączna z
    `wyklucz` (domyślnie z samą maską). Zwraca bool (n, 600)."""
    maska = np.asarray(maska, bool).reshape(H, W)
    wyk = (maska if wyklucz is None else np.asarray(wyklucz, bool).reshape(H, W))
    r, c = np.nonzero(maska)
    out = []
    for dr in range(-r.min(), H - r.max()):
        for dc in range(-c.min(), W - c.max()):
            m = np.zeros((H, W), bool)
            m[r + dr, c + dc] = True
            if not (m & wyk).any():
                out.append(m.reshape(-1))
    return np.array(out, bool).reshape(-1, H * W)
