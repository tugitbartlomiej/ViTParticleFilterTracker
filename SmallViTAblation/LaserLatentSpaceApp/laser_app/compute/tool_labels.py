"""Etykieta narzędzia z obrysu (wielokąty `<narzędzie>_seg` w annotation.csv Cataract-LMM 3_Object_Tracking), nie z ramki.

Dla klatki: udział pikseli narzędzia w każdej z 600 łat 16 × 16 wejścia 480 × 320 (wszystkie narzędzia oznaczone na klatce,
bez rogówki i źrenicy), osobno udział rogówki i źrenicy (etykiety oka; ViT nie jest na nich trenowany) oraz środek każdej łaty
w obrysie. Tryby wyboru tokenów narzędzia (TRYBY): łata dotknięta obrysem (domyślnie), udział ≥ próg, środek łaty w obrysie,
środek łaty w ramce GT. Mieszane = dotknięte obrysem, ale nienależące do narzędzia w danym trybie.
Historia aplikacji nie zna nazwy pliku, więc klatkę znajdujemy po ramce GT (indeks ramek wszystkich filmów, budowany raz
i zapisywany w cache_uczenia/etykiety); przy kilku kandydatach rozstrzyga podobieństwo obrazu.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from laser_app.paths import CACHE_DIR, DATASET_ROOT

H, W, KOMORKA = 20, 30, 16
SKALA = (W * KOMORKA / 720, H * KOMORKA / 480)            # natywne 720 × 480 → wejście 480 × 320
POMIJANE = ('cornea', 'pupil')
PROG = 0.5
TRYBY = (('dotkniete', 'obrys: każda łata dotknięta obrysem (> 0 %)'),
         ('pokrycie50', 'obrys: pokrycie łaty ≥ 50 % (jak dawniej, do 2026-10-21)'),
         ('pokrycie', 'obrys: pokrycie łaty ≥ własny próg'),
         ('srodek', 'obrys: środek łaty w obrysie'),
         ('ramka', 'ramka GT: środek łaty w ramce'))
KROTKO = {'dotkniete': 'dotknięte', 'pokrycie50': '≥ 50 %', 'pokrycie': 'własny ≥ {prog:.0%}', 'srodek': 'środek', 'ramka': 'ramka'}
CACHE = CACHE_DIR / 'etykiety'
ADNOTACJE = DATASET_ROOT / 'Cataract-LMM' / '3_Object_Tracking' / 'annotations'   # <film>.zip z annotation.csv


def _json(t):
    t = (t or '').strip()
    if not t or t in ('[]', 'null'):
        return None
    return json.loads(t)


def etykiety(film):
    """frame_number → surowy wiersz annotation.csv filmu (napisy; ramki [x, y, w, h], wielokąty `<narzędzie>_seg`, cornea_seg,
    pupil_seg w pikselach natywnej klatki 720 × 480; klatka MP4 i odpowiada frame_number i + 1)."""
    p = ADNOTACJE / f'{film}.zip'
    if not p.exists():
        return {}
    with zipfile.ZipFile(p) as z:
        nazwa = next(n for n in z.namelist() if n.endswith('annotation.csv'))
        tekst = z.read(nazwa).decode('utf-8-sig')
    csv.field_size_limit(1 << 25)
    return {int(w['frame_number']): w for w in csv.DictReader(io.StringIO(tekst))}


def etykieta_dla_zdjecia(sciezka):
    """Ramka cystotomu (xyxy w pikselach natywnej klatki) z annotation.csv filmu, jeśli nazwa pliku wskazuje film i klatkę
    (TR_0001_S1_P03_klatka_000870.jpg albo TR_0001_S1_P03_000870.jpg). Zwraca (ramka albo None, opis)."""
    m = re.search(r'(TR_\d+_S\d+_P\d+)\D*?(\d+)\.(?:jpg|jpeg|png)$', Path(sciezka).name, re.I)
    if not m:
        return None, 'brak etykiety: nazwa pliku nie wskazuje filmu i klatki'
    film, klatka = m.group(1), int(m.group(2))
    if not (ADNOTACJE / f'{film}.zip').exists():
        return None, f'brak etykiety: nie ma {film}.zip'
    w = etykiety(film).get(klatka)
    if w is None:
        return None, f'brak klatki {klatka} w {film}'
    b = _json(w.get('cystotome_bbox'))
    if b:
        x, y, bw, bh = map(float, b)
        return [x, y, x + bw, y + bh], f'etykieta: {film} kl. {klatka} (cystotom)'
    return None, f'{film} kl. {klatka}: na tej klatce nie oznaczono narzędzia'


def kategorie(wiersz):
    """Narzędzia oznaczone w wierszu annotation.csv (z niepustą ramką), bez rogówki i źrenicy."""
    return [k[:-len('_bbox')] for k in wiersz if k.endswith('_bbox') and k[:-len('_bbox')] not in POMIJANE and _json(wiersz[k])]


def _wielokaty(wiersz, kat):
    v = _json(wiersz.get(kat + '_seg'))
    if not v:
        return []
    if isinstance(v[0], (int, float)):
        v = [v]
    return [np.asarray(p, np.float32).reshape(-1, 2) * SKALA for p in v if len(p) >= 6]


def maska_pikseli(wielokaty):
    """uint8 (320, 480): 1 w pikselach wewnątrz wielokątów (w pikselach 480 × 320)."""
    import cv2
    m = np.zeros((H * KOMORKA, W * KOMORKA), np.uint8)
    for p in wielokaty:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def udzial_z_wielokatow(wielokaty):
    """float32 (600,): ułamek pikseli każdej łaty pokryty wielokątami (w pikselach 480 × 320)."""
    return maska_pikseli(wielokaty).reshape(H, KOMORKA, W, KOMORKA).mean((1, 3)).reshape(-1).astype(np.float32)


def srodek_z_wielokatow(wielokaty):
    """bool (600,): piksel w środku każdej łaty (przy 16 px: [8, 8] w łacie) leży wewnątrz wielokątów."""
    return maska_pikseli(wielokaty)[KOMORKA // 2::KOMORKA, KOMORKA // 2::KOMORKA].reshape(-1).astype(bool)


@lru_cache(maxsize=8)
def _etykiety_filmu(film):
    return etykiety(film)


def etykieta(film, klatka):
    """dict: udzial (600,), srodek (600,) bool, oko {rogowka, zrenica: udział (600,)}, kategorie, ramki {kat: xyxy w 480 × 320},
    piksele; None, gdy klatka bez narzędzia/obrysu."""
    w = _etykiety_filmu(film).get(int(klatka))
    if w is None:
        return None
    kat = kategorie(w)
    wiel = [p for k in kat for p in _wielokaty(w, k)]
    if not wiel:
        return None
    ramki = {}
    for k in kat:
        b = _json(w[k + '_bbox'])
        ramki[k] = [b[0] * SKALA[0], b[1] * SKALA[1], (b[0] + b[2]) * SKALA[0], (b[1] + b[3]) * SKALA[1]]
    u = udzial_z_wielokatow(wiel)
    oko = dict(rogowka=udzial_z_wielokatow(_wielokaty(w, 'cornea')), zrenica=udzial_z_wielokatow(_wielokaty(w, 'pupil')))
    return dict(film=film, klatka=int(klatka), udzial=u, srodek=srodek_z_wielokatow(wiel), oko=oko, kategorie=kat, ramki=ramki,
                piksele=float(u.sum() * KOMORKA ** 2))


def maski(udzial, prog=PROG, tryb='pokrycie', srodek=None, ramka=None):
    """(narzędzie, mieszane) bool (600,) z udziału pikseli obrysu wg trybu (TRYBY): 'dotkniete' udział > 0; 'pokrycie50' udział ≥ 0,5
    (jak dawniej); 'pokrycie' udział ≥ prog
    (mieszane 0 < udział < prog); 'srodek' środek łaty w obrysie (wymaga `srodek`, mieszane = dotknięte poza środkiem);
    'ramka' środek łaty w ramce GT (wymaga `ramka` bool (600,), mieszane puste)."""
    u = np.asarray(udzial, np.float32)
    if tryb == 'pokrycie50':                                            # stara etykieta (do 2026-10-21): połowa łaty to narzędzie
        tryb, prog = 'pokrycie', 0.5
    if tryb == 'pokrycie':
        return u >= prog, (u > 0) & (u < prog)
    if tryb == 'dotkniete':
        return u > 0, np.zeros(u.shape, bool)
    if tryb == 'srodek':
        if srodek is None:
            raise ValueError("tryb 'srodek' wymaga maski środków łat")
        s = np.asarray(srodek, bool)
        return s.copy(), (u > 0) & ~s
    if tryb == 'ramka':
        if ramka is None:
            raise ValueError("tryb 'ramka' wymaga maski łat w ramce GT")
        return np.asarray(ramka, bool).copy(), np.zeros(u.shape, bool)
    raise ValueError(f'nieznany tryb etykiety: {tryb!r}')


# ------------------------------------------------------------------ odnajdywanie klatki po ramce GT (historie aplikacji)
def _indeks():
    """Ramki wszystkich narzędzi we wszystkich filmach: (film, klatka, x0, y0, x1, y1 w 480 × 320); raz, potem z dysku."""
    p = CACHE / 'indeks_ramek.npz'
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return d['film'], d['klatka'], d['ramka']
    filmy, klatki, ramki = [], [], []
    for z in sorted(ADNOTACJE.glob('*.zip')):
        f = z.stem
        for kl, w in etykiety(f).items():
            for k in kategorie(w):
                b = _json(w[k + '_bbox'])
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
        with zipfile.ZipFile(ADNOTACJE / f'{film}.zip') as z:
            raw = z.read(f'{film}/{film}_{kl:06d}.jpg')
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W * KOMORKA, H * KOMORKA), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = float(np.abs(im - np.asarray(rgb, np.float32)).mean())
        if best is None or e < best[0]:
            best = (e, film, kl)
    return best[1], best[2]


def _zapisz(p, e):
    np.savez(p, film=e['film'], klatka=e['klatka'], udzial=e['udzial'], kategorie=np.array(e['kategorie']), srodek=e['srodek'],
             rogowka=e['oko']['rogowka'], zrenica=e['oko']['zrenica'])


def dla_historii(h):
    """Etykieta z obrysu dla historii (history.History; po ramce GT i obrazie); wynik (film, klatka, udzial, kategorie, srodek, oko)
    zapisywany w h.katalog/etykieta_obrys.npz. Stary plik bez środków i oka jest uzupełniany z annotation.csv. None, gdy brak."""
    p = Path(h.katalog) / 'etykieta_obrys.npz'
    if p.exists():
        d = np.load(p, allow_pickle=False)
        w = dict(film=str(d['film']), klatka=int(d['klatka']), udzial=d['udzial'], kategorie=[str(s) for s in d['kategorie']])
        if all(k in d.files for k in ('srodek', 'rogowka', 'zrenica')):
            w.update(srodek=d['srodek'].astype(bool), oko=dict(rogowka=d['rogowka'], zrenica=d['zrenica']))
            return w
        try:
            e = etykieta(w['film'], w['klatka'])
        except Exception:                                                # noqa: BLE001 (brak zipu adnotacji)
            e = None
        if e is None:                                                  # brak zipu adnotacji: zostaje sam udział
            return dict(w, srodek=None, oko=None)
        _zapisz(p, e)
        return dict(e, kategorie=w['kategorie'], udzial=w['udzial'])
    if getattr(h, 'gt', None) is None:
        return None
    fk = znajdz(h.gt, getattr(h, 'rgb', None))
    if fk is None:
        return None
    e = etykieta(*fk)
    if e is None:
        return None
    _zapisz(p, e)
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
