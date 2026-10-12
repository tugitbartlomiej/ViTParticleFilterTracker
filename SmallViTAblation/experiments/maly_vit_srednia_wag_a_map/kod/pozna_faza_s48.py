"""Uzupełnienie po wyniku: to samo uśrednianie na drugim ziarnie i w późniejszej fazie (opisowe, bez kryterium).

Jedyne lokalne zapisy wag z późniejszych kroków to nagranie ziarna 48 (`../../maly_vit_rownanie_latentu/
nagranie_s48_sily/wagi/k01000..k03000.pt`, co 100 kroków). Zapisy są 10 razy rzadsze niż w biegu gęstej historii,
więc okno 3 zapisów to 200 kroków, a 5 zapisów to 400 kroków. Bez treningu: sama ocena.

    py -3.11 pozna_faza_s48.py --wyniki ../wyniki
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import srednia_wag as S                                   # ustawia zmienne środowiskowe i importuje dane, trening

NAG = S.EXP / 'maly_vit_rownanie_latentu' / 'nagranie_s48_sily' / 'wagi'


def wagi(k):
    st = torch.load(NAG / f'k{k:05d}.pt', map_location='cpu', weights_only=False)
    return st['net'] if isinstance(st, dict) and 'net' in st else st


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); t0 = time.time()
    wal_w, wal_X = S.D.walidacja(); gt = [r['ramki'] for r in wal_w]
    baza = np.load(Path(os.environ['MALY_VIT_CACHE']) / 'baza1500.npy', mmap_mode='r')
    Xb = np.ascontiguousarray(baza[np.linspace(0, len(baza) - 1, 512).astype(int)])
    u = S.T.Uczen(48)

    def ocen(sd, bn=False):
        u.net.load_state_dict(sd)
        if bn: S.przelicz_bn(u.net, Xb, u.dev)
        return float(S.T.ap50(u.wykryj(wal_X), gt))

    kroki = list(range(1000, 3001, 100))
    stany = {k: wagi(k) for k in kroki}
    surowe = {k: ocen(stany[k]) for k in kroki}
    print('surowe', {k: round(v, 4) for k, v in surowe.items()}, f'{time.time() - t0:.0f} s', flush=True)
    tab = {}
    for t in range(1400, 3001, 200):
        r = dict(surowe_koniec=surowe[t])
        for K in (3, 5):
            ks = [t - 100 * i for i in range(K)][::-1]
            sd = S.srednia([stany[k] for k in ks])
            r[f'okno_{K}_zapisow'] = dict(srednie_mAP_surowych=float(np.mean([surowe[k] for k in ks])), najlepsze_surowe=float(np.max([surowe[k] for k in ks])),
                                          usrednione_bn_usrednione=ocen(sd), usrednione_bn_przeliczone=ocen(sd, bn=True))
        tab[str(t)] = r
        print(t, json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in r.items()}), f'{time.time() - t0:.0f} s', flush=True)
    y = np.array([surowe[k] for k in kroki]); x = np.array(kroki, float)
    reszta = y - np.polyval(np.polyfit(x, y, 1), x)
    wyn = dict(meta=dict(zrodlo=str(NAG), ziarno=48, kroki=[1000, 3000, 100], klatek_walidacji=len(wal_X),
                         opis='uzupełnienie po wyniku, opisowe; zapisy co 100 kroków'),
               surowe={str(k): v for k, v in surowe.items()}, usrednianie=tab,
               zmiennosc=dict(sd_wokol_trendu=float(reszta.std()), trend_na_100_krokow=float(np.polyfit(x, y, 1)[0] * 100),
                              rms_zmiany_co_100_krokow=float(np.sqrt(np.mean(np.diff(y) ** 2)))))
    (out / 'POZNA_FAZA_S48.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zmiennosc', wyn['zmiennosc'], 'zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
