"""Pamięć dyskowa obliczeń liczonych na żądanie dla jednego checkpointu (pole w podkrokach okienka 11, dolina 12, mechanika 13).

Wynik (słownik tablic i liczb) trafia do `<katalog historii>/pamiec/<rodzaj>/<parametry>.npz`; ten sam checkpoint z tymi samymi
parametrami jest potem wczytywany z dysku zamiast liczony. Pliki można eksportować i importować razem z resztą obliczeń
(compute/exports.py). Zapis bez pickle: struktura wyniku w JSON, tablice numpy w npz."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

KATALOG = 'pamiec'
WERSJA = 2                                    # 2: struktura w JSON + tablice (zagnieżdżone słowniki i listy)


def _czesc(v):
    if isinstance(v, float):
        return f'{v:g}'.replace('.', ',')
    if isinstance(v, (bytes, np.ndarray)):
        return hashlib.sha1(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else v).hexdigest()[:8]
    return re.sub(r'[^\w,.-]', '', str(v))


def plik(katalog_historii, rodzaj, **parametry):
    """Ścieżka pliku wyniku: parametry w nazwie w podanej kolejności (np. t=226, G=17, z=1.0, k=12000)."""
    nazwa = '_'.join(f'{k}{_czesc(v)}' for k, v in parametry.items())
    return Path(katalog_historii) / KATALOG / rodzaj / f'{nazwa}.npz'


def _rozbierz(v, tab):
    """Struktura do JSON: tablice numpy trafiają do `tab` (odnośnik {"__nd__": klucz}), liczby numpy → liczby Pythona."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            raise TypeError('tablica obiektów')
        k = f'a{len(tab)}'; tab[k] = v
        return {'__nd__': k}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {'__dict__': [[_rozbierz(k, tab), _rozbierz(x, tab)] for k, x in v.items()]}
    if isinstance(v, tuple):
        return {'__tuple__': [_rozbierz(x, tab) for x in v]}
    if isinstance(v, list):
        return [_rozbierz(x, tab) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f'nie do zapisu: {type(v).__name__}')


def _zloz(v, z):
    if isinstance(v, dict):
        if '__nd__' in v:
            return z[v['__nd__']]
        if '__dict__' in v:
            return {_zloz(k, z): _zloz(x, z) for k, x in v['__dict__']}
        if '__tuple__' in v:
            return tuple(_zloz(x, z) for x in v['__tuple__'])
    if isinstance(v, list):
        return [_zloz(x, z) for x in v]
    return v


def zapisz(sciezka, d):
    """Zapisuje wynik (słownik: tablice, liczby, napisy, zagnieżdżone słowniki, listy i krotki) bez pickle: struktura w JSON,
    tablice w npz. False, gdy jest w nim coś innego (np. obiekt klasy): wtedy nic nie powstaje i wynik liczy się jak dawniej."""
    import json
    tab = {}
    try:
        struktura = json.dumps(_rozbierz(dict(d), tab))
    except (TypeError, ValueError):
        return False
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    tmp = sciezka.with_name(sciezka.stem + '_tmp.npz')
    np.savez(tmp, _wersja_pamieci=WERSJA, _struktura=np.array(struktura), **tab)
    tmp.replace(sciezka)
    return True


def wczytaj(sciezka):
    """Wynik z pliku (ta sama struktura co przy zapisie) albo None."""
    import json
    sciezka = Path(sciezka)
    if not sciezka.exists():
        return None
    try:
        with np.load(sciezka, allow_pickle=False) as z:
            if int(z['_wersja_pamieci']) != WERSJA:
                return None
            tab = {k: z[k] for k in z.files if not k.startswith('_')}
            return _zloz(json.loads(str(z['_struktura'])), tab)
    except (OSError, ValueError, KeyError):
        return None
