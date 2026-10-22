"""Zapis przejść przez sieć (history.podkroki) na dysk, żeby nie liczyć ich przy każdym otwarciu historii.

Katalog: <katalog historii w cache_uczenia/lasery>/przejscia/ (historia jest już kluczowana wagami, krokami i zdjęciem).
- baza.npz: wspólny rzut 3D (mu, P dla układów '', '_c', '_n') i przypisanie reflektorów do miejsc na sferze (slot),
  ustalone przez pierwszy policzony checkpoint; wszystkie zapisane przejścia są policzone w tym rzucie, więc pasują do siebie
  także między sesjami;
- k<krok>.npz: jedno przejście. Bez ah_c i z_c (odtwarzane przy wczytaniu: ah − średnia po słuchających, z − średnia po
  tokenach) i bez rzutów płaskich (_pp, liczone w 0,08 s). Domyślnie float32 (dokładnie, kontrola „suma wkładów = ruch”
  ~10⁻⁶; około 44 MB na checkpoint), opcja float16 dla ah, z i xn (około 22 MB, kontrola ~10⁻³).
"""
import threading
from pathlib import Path

import numpy as np

WERSJA = 1
ZAMEK_SIECI = threading.Lock()          # history.podkroki używa jednej wspólnej sieci (_siec): jeden wątek naraz
F16 = ('ah', 'z', 'xn')
POMIJANE = ('ah_c', 'z_c', 'baza', 'lodyga', 'lodyga_ksztalty')


def katalog(h):
    return Path(h.katalog) / 'przejscia'


def plik(kat, krok):
    return Path(kat) / f'k{int(krok):05d}.npz'


def _zapisz_npz(p, **arr):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, **arr)
    tmp.replace(p)


def wczytaj_baze(kat):
    p = Path(kat) / 'baza.npz'
    if not p.exists():
        return None
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def zapisz_baze(kat, baza):
    """Tylko raz: istniejąca baza zostaje (zapisane przejścia są w jej rzucie)."""
    p = Path(kat) / 'baza.npz'
    if not p.exists() and baza:
        _zapisz_npz(p, **{k: np.asarray(v) for k, v in baza.items()})


def zapisz(kat, d, f16=False):
    """Zapisuje przejście d (wynik history.podkroki); zwraca liczbę bajtów pliku."""
    arr = {}
    for k, v in d.items():
        if k in POMIJANE or k.startswith('_'):
            continue
        if isinstance(v, np.ndarray):
            arr[k] = v.astype(np.float16) if (f16 and k in F16) else v
        elif isinstance(v, (bool, int, float, np.integer, np.floating)):
            arr['s_' + k] = np.asarray(v)
    for i, e in enumerate(d.get('lodyga', [])):
        arr[f'lodyga_{i}'] = np.asarray(e)
    arr['lodyga_ksztalty'] = np.asarray(d.get('lodyga_ksztalty', []), np.int64)
    arr['wersja'] = np.asarray(WERSJA); arr['f16'] = np.asarray(bool(f16))
    p = plik(kat, d['krok'])
    _zapisz_npz(p, **arr)
    return p.stat().st_size


def wczytaj(kat, krok):
    """Przejście z dysku w tej samej postaci co history.podkroki (bez 'baza'), albo None."""
    p = plik(kat, krok)
    if not p.exists():
        return None
    try:
        with np.load(p) as f:
            if int(f['wersja']) != WERSJA:
                return None
            d, lod = {}, {}
            for k in f.files:
                if k in ('wersja', 'f16'):
                    continue
                v = f[k]
                if k.startswith('s_'):
                    d[k[2:]] = v.item()
                elif k.startswith('lodyga_') and k != 'lodyga_ksztalty':
                    lod[int(k[7:])] = v
                else:
                    d[k] = v.astype(np.float32) if v.dtype == np.float16 else v
    except (OSError, ValueError, KeyError):                     # uszkodzony albo niedokończony plik: liczymy od nowa
        return None
    d['lodyga'] = [lod[i] for i in sorted(lod)]
    d['lodyga_ksztalty'] = [tuple(int(x) for x in r) for r in d['lodyga_ksztalty']]
    d['ah_c'] = d['ah'] - d['ah'].mean(2, keepdims=True)
    d['z_c'] = d['z'] - d['z'].mean(1, keepdims=True)
    d['krok'] = int(d['krok'])
    return d


def spis(kat):
    """(liczba zapisanych przejść, bajty)."""
    kat = Path(kat)
    if not kat.exists():
        return 0, 0
    pl = [p for p in kat.glob('k*.npz') if not p.name.endswith('.tmp.npz')]
    return len(pl), sum(p.stat().st_size for p in pl)
