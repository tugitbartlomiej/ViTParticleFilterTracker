"""Eksport i import obliczeń jednej historii uczenia: długie obliczenia liczy się raz, zapisuje do pliku i wczytuje gdzie indziej
(inny komputer, inna ścieżka wag), zamiast liczyć od nowa.

Co trwa długo i co zostaje na dysku (katalog historii `cache_uczenia/lasery/<klucz>/`):
- historia uczenia (`history.policz`, minuty): dane.npz, Z.npy, opis.json – położenia wszystkich cząsteczek na sferze,
  natężenia 768 laserów, atencja; z niej biorą się przebiegi cząsteczek na sferze;
- rzuty 192D (`rzuty/`), przejścia przez sieć w podkrokach (`przejscia/`);
- pułapka optyczna cząsteczki (`ogniskowa_t<tok>_G<siatka>_z<zasięg>.npz`, ok. 1 min na historię);
- mapa pułapek wszystkich cząsteczek (`mapa_pulapek_z<zasięg>.npz`, ok. 2 min);
- pomiar pułapki w podkrokach (`pulapka_podkrokow_t<tok>_G<siatka>.npz/.csv`, ok. 1–2 min);
- pamięć obliczeń na żądanie (`pamiec/`: pole w podkrokach, dolina, mechanika; disk_cache.py).

Plik eksportu to zip z `manifest.json` (klucz historii, opis, lista plików z SHA256) i plikami względem katalogu historii.
Import rozpakowuje do katalogu historii o tym samym kluczu (albo do historii o tych samych krokach i zdjęciu, gdy ścieżka wag
jest inna: history.znajdz_przenosna), sprawdza sumy SHA256 i nie nadpisuje istniejących plików bez zgody.
Przebiegi cząsteczek na sferze (położenia przez checkpointy) eksportuje `przebiegi` do CSV i NPZ."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np

WERSJA = 1
MANIFEST = 'manifest.json'

# (rodzaj, opis, wzorce glob względem katalogu historii, wyrażenie z numerem cząsteczki i parametrami)
RODZAJE = (
    ('historia', 'Historia uczenia: położenia na sferze, natężenia laserów, atencja', ('dane.npz', 'Z.npy', 'opis.json'), None),
    ('etykieta', 'Etykieta narzędzia z obrysu', ('etykieta_obrys.npz',), None),
    ('rzuty', 'Rzuty z 192 wymiarów (stany, siły, bazy)', ('rzuty/**/*',), None),
    ('przejscia', 'Przejścia przez sieć w podkrokach', ('przejscia/**/*',), None),
    ('pulapka', 'Pułapka optyczna cząsteczki', ('ogniskowa_t*_G*_z*.npz', 'ogniskowa_t*_G*_z*.json'),
     r'ogniskowa_t(?P<tok>\d+)_G(?P<G>\d+)_z(?P<z>\d+)'),
    ('mapa', 'Mapa pułapek wszystkich cząsteczek', ('mapa_pulapek_z*.npz', 'mapa_pulapek_z*.json'), r'mapa_pulapek_z(?P<z>\d+)'),
    ('pomiar', 'Pomiar pułapki w podkrokach (cała historia)', ('pulapka_podkrokow_t*_G*.npz', 'pulapka_podkrokow_t*_G*.csv'),
     r'pulapka_podkrokow_t(?P<tok>\d+)_G(?P<G>\d+)'),
    ('pamiec', 'Pamięć obliczeń na żądanie (pole w podkrokach, dolina, mechanika)', ('pamiec/**/*.npz',),
     r'(?P<rodzaj>[^/\\]+)[/\\]t(?P<tok>\d+)'),
)
OPISY = {r: o for r, o, _, _ in RODZAJE}


def _pliki(katalog, wzorce):
    out = []
    for w in wzorce:
        out += [p for p in Path(katalog).glob(w) if p.is_file() and not p.name.endswith(('_tmp.npz', '_tmp.npy'))]
    return sorted(set(out))


def lista(katalog_historii):
    """Obliczenia w katalogu historii: lista słowników (rodzaj, opis, tok albo None, parametry, pliki [ścieżki względne], bajty,
    czas modyfikacji). Pliki jednej pułapki, jednego pomiaru albo jednej grupy pamięci tworzą jeden wpis."""
    kat = Path(katalog_historii)
    wpisy = []
    for rodzaj, opis, wzorce, wzor in RODZAJE:
        grupy = {}
        for p in _pliki(kat, wzorce):
            wzg = p.relative_to(kat)
            m = re.search(wzor, str(wzg)) if wzor else None
            if rodzaj == 'pamiec' and m:
                klucz = (m.group('rodzaj'), int(m.group('tok')), re.sub(r'_k\d+$', '', p.stem.split('_', 1)[1]) if '_' in p.stem else '')
            elif m:
                klucz = m.group(0)
            else:
                klucz = rodzaj
            grupy.setdefault(klucz, []).append(p)
        for klucz, pliki in sorted(grupy.items(), key=lambda x: str(x[0])):
            tok, param = None, ''
            m = re.search(wzor, str(pliki[0].relative_to(kat))) if wzor else None
            if m and 'tok' in m.groupdict():
                tok = int(m.group('tok'))
            if rodzaj == 'pulapka' and m:
                param = f'siatka {m.group("G")}³, zasięg {int(m.group("z")) / 100:g}'.replace('.', ',') if int(m.group('G')) > 1 else \
                    f'tylko wiązki, zasięg {int(m.group("z")) / 100:g}'.replace('.', ',')
            elif rodzaj == 'mapa' and m:
                param = f'zasięg {int(m.group("z")) / 100:g}'.replace('.', ',')
            elif rodzaj == 'pomiar' and m:
                param = f'siatka {m.group("G")}³'
            elif rodzaj == 'pamiec':
                nazwy = {'pole_podkrokow': 'pole w podkrokach', 'dolina': 'dolina', 'mechanika': 'mechanika'}
                param = f'{nazwy.get(klucz[0], klucz[0])} ({klucz[2]}): {len(pliki)} checkpointów'
            elif rodzaj in ('rzuty', 'przejscia'):
                param = f'{len(pliki)} plików'
            wpisy.append(dict(rodzaj=rodzaj, opis=opis, tok=tok, parametry=param,
                              pliki=[str(p.relative_to(kat)).replace('\\', '/') for p in pliki],
                              bajty=int(sum(p.stat().st_size for p in pliki)), czas=max(p.stat().st_mtime for p in pliki)))
    return wpisy


def _sha(p, blok=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(blok)
            if not b:
                return h.hexdigest()
            h.update(b)


def eksportuj(katalog_historii, wpisy, cel, postep=None, dodatki=None):
    """Zip z plikami wpisów (lista z `lista`) i manifestem. dodatki: {nazwa w zipie: bajty} (np. CSV przebiegów).
    postep(i, n, opis). Zwraca ścieżkę pliku."""
    kat = Path(katalog_historii)
    pliki = list(dict.fromkeys(p for w in wpisy for p in w['pliki']))
    opis = json.loads((kat / 'opis.json').read_text(encoding='utf-8')) if (kat / 'opis.json').exists() else {}
    man = dict(aplikacja='LaserLatentSpaceApp', wersja=WERSJA, utworzono=time.strftime('%Y-%m-%d %H:%M:%S'), klucz=kat.name,
               opis_historii=opis, przenosny=przenosny_z_katalogu(kat), wpisy=[{k: v for k, v in w.items() if k != 'czas'} for w in wpisy],
               pliki={})
    cel = Path(cel)
    tmp = cel.with_name(cel.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
        for i, rel in enumerate(pliki):
            p = kat / rel
            if postep is not None:
                postep(i, len(pliki), f'eksport: {rel} ({p.stat().st_size / 2 ** 20:.0f} MB)')
            man['pliki'][rel] = dict(sha256=_sha(p), bajty=p.stat().st_size)
            komp = zipfile.ZIP_DEFLATED if p.suffix in ('.json', '.csv') else zipfile.ZIP_STORED   # npz/npy: już binarne
            z.write(p, f'historia/{rel}', compress_type=komp)
        for nazwa, dane in (dodatki or {}).items():
            z.writestr(nazwa, dane, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(MANIFEST, json.dumps(man, ensure_ascii=False, indent=1), compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(cel)
    if postep is not None:
        postep(len(pliki), len(pliki), f'zapisano {cel.name}')
    return cel


def czytaj_manifest(sciezka):
    with zipfile.ZipFile(sciezka) as z:
        return json.loads(z.read(MANIFEST).decode('utf-8'))


def cel_importu(man, katalog_cache):
    """Katalog historii, do którego trafi import: ten sam klucz, inaczej historia o tych samych krokach i zdjęciu (inna ścieżka
    wag albo dysk), inaczej nowy katalog o kluczu z manifestu (gdy plik niesie samą historię)."""
    kat = Path(katalog_cache) / man['klucz']
    if (kat / 'dane.npz').exists():
        return kat
    if man.get('przenosny'):
        from laser_app.compute import history
        inny = history.znajdz_przenosna(man['przenosny'], katalog_cache)
        if inny is not None:
            return inny
    return kat


def importuj(sciezka, katalog_cache, nadpisz=False, postep=None):
    """Rozpakowuje obliczenia z pliku eksportu do katalogu historii (cel_importu). Zwraca (katalog, zapisane, pominięte, błędy):
    pominięte = pliki już istniejące (bez nadpisz), błędy = niezgodne SHA256 albo brak historii dla obliczeń."""
    man = czytaj_manifest(sciezka)
    kat = cel_importu(man, katalog_cache)
    pliki = man.get('pliki', {})
    niesie_historie = 'dane.npz' in pliki
    if not (kat / 'dane.npz').exists() and not niesie_historie:
        return kat, [], [], ['brak historii uczenia dla tych obliczeń: najpierw policz albo wczytaj historię '
                             f'(klucz {man["klucz"]}, {man.get("opis_historii", {}).get("kroki", "?")} checkpointów)']
    zapisane, pominiete, bledy = [], [], []
    with zipfile.ZipFile(sciezka) as z:
        for i, (rel, info) in enumerate(pliki.items()):
            if postep is not None:
                postep(i, len(pliki), f'import: {rel}')
            cel = kat / rel
            if cel.exists() and not nadpisz:
                pominiete.append(rel)
                continue
            cel.parent.mkdir(parents=True, exist_ok=True)
            tmp = cel.with_name(cel.name + '.import')
            with z.open(f'historia/{rel}') as zr, open(tmp, 'wb') as zw:
                while True:
                    b = zr.read(1 << 22)
                    if not b:
                        break
                    zw.write(b)
            if _sha(tmp) != info['sha256']:
                tmp.unlink(); bledy.append(f'{rel}: niezgodna suma SHA256 (plik uszkodzony)')
                continue
            tmp.replace(cel)
            zapisane.append(rel)
    if postep is not None:
        postep(len(pliki), len(pliki), f'zaimportowano {len(zapisane)} plików')
    return kat, zapisane, pominiete, bledy


def przenosny_z_katalogu(kat):
    """Klucz przenośny historii (bez ścieżki wag): kroki, okno, ramka GT i skrót zdjęcia (history.klucz_przenosny)."""
    kat = Path(kat)
    if not (kat / 'dane.npz').exists():
        return None
    from laser_app.compute import history
    with np.load(kat / 'dane.npz', allow_pickle=False) as d:
        kroki, rgb = d['kroki'], d['rgb']
    opis = json.loads((kat / 'opis.json').read_text(encoding='utf-8')) if (kat / 'opis.json').exists() else {}
    if 'wagi' not in opis:
        return None
    return history.klucz_przenosny(kroki, rgb, opis.get('gt'), opis.get('okno', 300), opis['wagi'], opis.get('wersja', 1))


def przebiegi(h, toki, sciezka_csv, R=34.0, skala_stala=False):
    """Przebiegi cząsteczek `toki` na sferze przez wszystkie checkpointy historii h: CSV (krok, epoka, cząsteczka, położenie w
    rzucie P3, położenie na ekranie sfery, r99, narzędzie) i NPZ obok (pos (K, n, 3), sfera (K, n, 3), kroki, toki).
    Ekran sfery = rzut · 0,55 R / r99 (r99 z bieżącego checkpointu albo, przy skala_stala, z ostatniego). Zwraca (csv, npz)."""
    toki = [int(t) for t in toki]
    pos = np.asarray(h.pos, np.float64)[:, toki]                                  # (K, n, 3)
    r = np.asarray(h.r99, np.float64) if not skala_stala else np.full(h.K, float(h.r99[-1]))
    s = 0.55 * float(R) / np.maximum(r, 1e-12)
    sfera = pos * s[:, None, None]
    narz = np.asarray(getattr(h, 'narz', np.zeros(pos.shape[1], bool)), bool)
    from laser_app.compute.history import NA_EPOKE
    sciezka_csv = Path(sciezka_csv)
    with open(sciezka_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['krok', 'epoka', 'czasteczka', 'x', 'y', 'z', 'x_sfera', 'y_sfera', 'z_sfera', 'r99', 'narzedzie'])
        for k in range(h.K):
            for j, t in enumerate(toki):
                w.writerow([int(h.kroki[k]), f'{h.kroki[k] / NA_EPOKE:.4f}', t, *(f'{v:.6g}' for v in pos[k, j]),
                            *(f'{v:.6g}' for v in sfera[k, j]), f'{r[k]:.6g}', int(narz[t]) if t < len(narz) else 0])
    npz = sciezka_csv.with_suffix('.npz')
    np.savez(npz, kroki=np.asarray(h.kroki, np.int64), toki=np.asarray(toki, np.int32), pos=pos.astype(np.float32),
             sfera=sfera.astype(np.float32), r99=r.astype(np.float32), narz=narz[toki] if len(narz) else np.zeros(len(toki), bool))
    return sciezka_csv, npz
