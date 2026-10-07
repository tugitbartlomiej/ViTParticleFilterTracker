"""Sprawdzian odtworzenia paczek gęstej historii (maly_vit_ruch_czastek_gesta_historia, ziarno 47).

Od zapisanych wag kroku 0 robi 10 kroków AdamW kodem tego biegu (kopia kod/ z gęstej historii, bez zmian) na paczkach
z generatora default_rng(47) i porównuje wagi z zapisanym k0010.pt. Kontrola: to samo z generatorem 48 (inne paczki).
Uczenie na kopii w pamięci, nic nie jest zapisywane poza WYNIKI_sprawdz_paczki.json. Wymaga GPU (bf16 jak w biegu).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[2]
BIEG = EXP / 'maly_vit_ruch_czastek_gesta_historia'
os.environ.setdefault('MALY_VIT_WEJSCIA', str(EXP / 'maly_vit_od_zera' / 'eden' / 'wejscia'))
os.environ.setdefault('MALY_VIT_DANE', str(EXP / 'koopman_fastgan_petla' / 'local_inputs' / 'data'))
os.environ.setdefault('MALY_VIT_CACHE', str(EXP / 'maly_vit_od_zera' / 'cache'))
sys.path.insert(0, str(BIEG / 'kod'))
import dane as D        # noqa: E402
import trening as T     # noqa: E402


def wagi(p):
    return torch.load(p, map_location='cpu', weights_only=True)['net']


def roznica(a, b):
    """Względna różnica wag: ||a − b|| / ||b|| po wszystkich parametrach (bez buforów całkowitych)."""
    num = den = 0.
    for k in b:
        if b[k].dtype.is_floating_point:
            num += float(((a[k].float().cpu() - b[k].float()) ** 2).sum()); den += float((b[k].float() ** 2).sum())
    return (num / den) ** .5


def main():
    torch.backends.cudnn.benchmark = False
    baza = D.baza()
    print('baza', len(baza), flush=True)
    w0, w10 = wagi(BIEG / 'wyniki/pelny/wagi/k0000.pt'), wagi(BIEG / 'wyniki/pelny/wagi/k0010.pt')
    wynik = dict(baza=len(baza))
    u = T.Uczen(47)
    wynik['init_Uczen47_vs_k0000'] = roznica(u.net.state_dict(), w0)
    ruch = roznica(w0, w10)
    for nazwa, ziarno in (('ziarno_47', 47), ('kontrola_ziarno_48', 48)):
        u = T.Uczen(47)
        u.net.load_state_dict(w0)
        u.kroki(baza, 10, np.random.default_rng(ziarno))
        r = roznica(u.net.state_dict(), w10)
        wynik[nazwa] = dict(roznica_wzgledna_vs_k0010=r, jako_ulamek_ruchu_0_10=r / ruch)
        print(nazwa, wynik[nazwa], flush=True)
    wynik['ruch_wag_0_10'] = ruch
    (Path(__file__).resolve().parents[1] / 'WYNIKI_sprawdz_paczki.json').write_text(
        json.dumps(wynik, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(wynik, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
