"""Filmy Cataract-LMM 3_Object_Tracking: zakres czasu -> sekwencja klatek, etykiety i obrysy narzędzia.

Klatka MP4 o indeksie i (od 0) odpowiada wierszowi annotation.csv o frame_number = i + 1 (sprawdzone na
TR_0001: klatka MP4 869 = JPG 870). Filmy mają 29,97 kl./s i 720x480 pikseli. MP4 leżą w zipach
videos_part00N.zip; potrzebny film jest wypakowywany raz do katalogu MP4 (KATALOG_MP4).
Etykiety: cystotome_bbox [x, y, w, h], cystotome_seg (wielokąty obrysu narzędzia), cystotome_tip,
cornea_seg, pupil_seg; wszystko w pikselach natywnej klatki.
"""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from laser_app.compute.common import SZER, WYS   # wejście ViT 480 x 320
from laser_app.paths import DATASET_ROOT, PROJECT_ROOT

ZBIOR = DATASET_ROOT / 'Cataract-LMM' / '3_Object_Tracking'
ADNOTACJE = ZBIOR / 'annotations'
WIDEO = ZBIOR / 'videos'
KATALOG_MP4 = ZBIOR / 'videos_mp4'
MANIFEST = PROJECT_ROOT / 'outputs' / 'vit_sd_pelny_latent_filmy_eden_20261006' / 'FILMY.json'


def lista_filmow():
    """Lista słowników: id, split (train / val / poza manifestem), klatek, z_narzedziem, zip_wideo."""
    manifest = {}
    if MANIFEST.exists():
        for w in json.loads(MANIFEST.read_text(encoding='utf-8-sig')):
            manifest[w['video_id']] = w
    zipy = {}
    raport = WIDEO / 'ZipContentsReport.csv'
    if raport.exists():
        for w in csv.DictReader(io.StringIO(raport.read_text(encoding='utf-8-sig'))):
            zipy[Path(w['InternalFileName']).stem] = w['ZipFileName']
    filmy = []
    for p in sorted(ADNOTACJE.glob('TR_*.zip')):
        m = manifest.get(p.stem, {})
        filmy.append(dict(id=p.stem, split=m.get('split', 'poza manifestem'), klatek=m.get('frame_count'),
                          z_narzedziem=m.get('tool_labeled_count'), zip_wideo=zipy.get(p.stem)))
    return filmy


def id_filmu(sciezka):
    m = re.search(r'(TR_\d+_S\d+_P\d+)', Path(sciezka).name)
    return m.group(1) if m else None


def sciezka_mp4(film, postep=None):
    """Ścieżka do MP4 filmu; przy pierwszym użyciu wypakowuje go z videos_part00N.zip do KATALOG_MP4."""
    cel = KATALOG_MP4 / f'{film}.mp4'
    if cel.exists():
        return cel
    zipy = {f['id']: f['zip_wideo'] for f in lista_filmow()}
    if not zipy.get(film):
        raise FileNotFoundError(f'{film}: nie ma go w ZipContentsReport.csv')
    KATALOG_MP4.mkdir(parents=True, exist_ok=True)
    tmp = cel.with_suffix('.mp4.czesc')
    with zipfile.ZipFile(WIDEO / zipy[film]) as z:
        info = z.getinfo(f'{film}.mp4')
        with z.open(info) as zr, open(tmp, 'wb') as zw:
            zrobione = 0
            while True:
                b = zr.read(1 << 22)
                if not b:
                    break
                zw.write(b); zrobione += len(b)
                if postep:
                    postep(f'wypakowuję {film}.mp4: {zrobione / 2**20:.0f} / {info.file_size / 2**20:.0f} MB')
    tmp.replace(cel)
    return cel


def info_mp4(sciezka):
    import cv2
    c = cv2.VideoCapture(str(sciezka))
    try:
        fps, n = c.get(cv2.CAP_PROP_FPS) or 29.97, int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        return dict(fps=fps, klatek=n, sekund=n / fps, szer=int(c.get(3)), wys=int(c.get(4)))
    finally:
        c.release()


def _json(t):
    t = (t or '').strip()
    if not t or t in ('[]', 'null'):
        return None
    return json.loads(t)


def etykiety(film):
    """frame_number -> surowy wiersz annotation.csv (napisy). Parsowanie pól: etykieta_klatki."""
    p = ADNOTACJE / f'{film}.zip'
    if not p.exists():
        return {}
    with zipfile.ZipFile(p) as z:
        nazwa = next(n for n in z.namelist() if n.endswith('annotation.csv'))
        tekst = z.read(nazwa).decode('utf-8-sig')
    csv.field_size_limit(1 << 25)
    return {int(w['frame_number']): w for w in csv.DictReader(io.StringIO(tekst))}


def etykieta_klatki(wiersz, skala=(SZER / 720, WYS / 480)):
    """Etykiety jednej klatki w pikselach obrazu 480x320 (wejście ViT); None tam, gdzie nie oznaczono."""
    if wiersz is None:
        return dict(ramka=None, obrys=[], czubek=None, rogowka=[], zrenica=[])
    sx, sy = skala

    def wielokaty(pole):
        v = _json(wiersz.get(pole))
        if not v:
            return []
        if isinstance(v[0], (int, float)):
            v = [v]
        return [np.asarray(p, np.float32).reshape(-1, 2) * (sx, sy) for p in v if len(p) >= 6]

    b = _json(wiersz.get('cystotome_bbox'))
    t = _json(wiersz.get('cystotome_tip'))
    return dict(ramka=None if not b else [b[0] * sx, b[1] * sy, (b[0] + b[2]) * sx, (b[1] + b[3]) * sy],
                obrys=wielokaty('cystotome_seg'),
                czubek=None if not t else (float(np.ravel(t)[0]) * sx, float(np.ravel(t)[1]) * sy),
                rogowka=wielokaty('cornea_seg'), zrenica=wielokaty('pupil_seg'))


def maska_patchy(wielokaty, prog=0.1):
    """bool [600]: patch 16x16 należy do obrysu, gdy wielokąty pokrywają co najmniej `prog` jego pikseli."""
    import cv2
    m = np.zeros((WYS, SZER), np.uint8)
    for p in wielokaty:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32).reshape(-1, 1, 2)], 1)
    pokrycie = m.reshape(20, 16, 30, 16).mean((1, 3)).reshape(-1)
    return pokrycie >= max(prog, 1e-6)


def indeksy_zakresu(t_od, t_do, n, fps, klatek, wiersze=None, tylko_z_narzedziem=False):
    """Indeksy klatek MP4 (od 0) w zakresie [t_od, t_do] sekund, n równomiernie rozłożonych (bez powtórzeń)."""
    i0 = int(np.clip(round(t_od * fps), 0, klatek - 1)); i1 = int(np.clip(round(t_do * fps), 0, klatek - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    kandydaci = np.arange(i0, i1 + 1)
    if tylko_z_narzedziem and wiersze:
        kandydaci = np.array([i for i in kandydaci if _json((wiersze.get(i + 1) or {}).get('cystotome_bbox'))], int)
    if len(kandydaci) == 0:
        return kandydaci
    if n >= len(kandydaci):
        return kandydaci
    return kandydaci[np.unique(np.round(np.linspace(0, len(kandydaci) - 1, n)).astype(int))]


def czytaj_klatki(sciezka, indeksy, postep=None, przerwij=lambda: False):
    """Generator (indeks, rgb uint8 natywny) dla posortowanych indeksów MP4. Gęste zakresy czyta po kolei,
    rzadkie przeskokami (seek)."""
    import cv2
    c = cv2.VideoCapture(str(sciezka))
    try:
        biezacy = -1
        for k, i in enumerate(indeksy):
            if przerwij():
                return
            i = int(i)
            if i - biezacy - 1 > 30 or biezacy < 0 or i <= biezacy:
                c.set(cv2.CAP_PROP_POS_FRAMES, i); biezacy = i - 1
            while biezacy < i - 1:
                c.grab(); biezacy += 1
            ok, kl = c.read(); biezacy = i
            if not ok:
                break
            if postep:
                postep(k)
            yield i, np.ascontiguousarray(kl[:, :, ::-1])
    finally:
        c.release()


def czytaj_klatki_jpg(film, indeksy, postep=None, przerwij=lambda: False):
    """To samo z JPG w zipie adnotacji (te same klatki, na których robiono etykiety)."""
    from PIL import Image
    with zipfile.ZipFile(ADNOTACJE / f'{film}.zip') as z:
        for k, i in enumerate(indeksy):
            if przerwij():
                return
            nazwa = f'{film}/{film}_{int(i) + 1:06d}.jpg'
            if postep:
                postep(k)
            yield int(i), np.asarray(Image.open(io.BytesIO(z.read(nazwa))).convert('RGB'))


OBSZARY_WYCINANIA = (('obrys', 'obrys narzędzia'), ('ramka', 'ramka GT'), ('rogowka', 'rogówka'), ('zrenica', 'źrenica'))


def maska_wycinania(et, obszary, ksztalt=(WYS, SZER), margines=0):
    """Maska bool [H, W] sumy wybranych obszarów z etykiety klatki (et z etykieta_klatki, w pikselach obrazu):
    obszary ⊂ {'obrys', 'ramka', 'rogowka', 'zrenica'}; margines w pikselach poszerza maskę (dylatacja)."""
    import cv2
    m = np.zeros(ksztalt, np.uint8)
    for o in obszary:
        if o == 'ramka':
            if et['ramka'] is not None:
                x0, y0, x1, y1 = (int(round(v)) for v in et['ramka'])
                cv2.rectangle(m, (x0, y0), (x1, y1), 1, -1)
        else:
            w = [np.round(p).astype(np.int32) for p in et.get(o, [])]
            if w:
                cv2.fillPoly(m, w, 1)
    if margines > 0 and m.any():
        k = 2 * int(margines) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(bool)


def wytnij(rgb, maska, tryb):
    """tryb 0: zostaw tylko maskę (reszta czarna); 1: zamaluj maskę na czarno. Kopia obrazu."""
    keep = maska if tryb == 0 else ~maska
    return np.where(keep[..., None], rgb, 0).astype(np.uint8)


ROZSZERZENIA_OBRAZOW = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def folder_zdjec(katalog, fps=29.97):
    """Sekwencja z folderu zdjęć: pliki obrazów w kolejności nazw (bez podfolderów). Gdy obok leży MANIFEST.json
    z polami film i klatki[].frame_number (np. z wytnij_narzedzie.py), zwraca też film i numer klatki każdego pliku,
    więc etykiety i obrysy z annotation.csv pasują do zdjęć.
    Wynik: dict(katalog, pliki, film, frame_numbers albo None, fps, opis)."""
    k = Path(katalog)
    pliki = sorted(p for p in k.iterdir() if p.is_file() and p.suffix.lower() in ROZSZERZENIA_OBRAZOW)
    film, nr, opis = id_filmu(k.name), None, ''
    m = k / 'MANIFEST.json'
    if m.exists():
        j = json.loads(m.read_text(encoding='utf-8'))
        film = j.get('film') or film
        fps = float(j.get('fps') or fps)
        po_nazwie = {w['plik']: int(w['frame_number']) for w in j.get('klatki', []) if 'plik' in w and 'frame_number' in w}
        if pliki and all(p.name in po_nazwie for p in pliki):
            nr = [po_nazwie[p.name] for p in pliki]
        opis = f'MANIFEST: {film}, {j.get("od", "?")} – {j.get("do", "?")}' + (f', {j["tlo"]}' if j.get('tlo') else '')
    return dict(katalog=str(k), pliki=[str(p) for p in pliki], film=film, frame_numbers=nr, fps=fps, opis=opis)


def czytaj_klatki_folder(pliki, indeksy, postep=None, przerwij=lambda: False):
    """Klatki z listy plików obrazów (RGB uint8), indeksy = pozycje na liście."""
    from PIL import Image
    for k, i in enumerate(indeksy):
        if przerwij():
            return
        if postep:
            postep(k)
        yield int(i), np.asarray(Image.open(pliki[int(i)]).convert('RGB'))


def mmss(sekundy):
    m, s = divmod(max(0.0, float(sekundy)), 60)
    return f'{int(m):02d}:{s:04.1f}'
