"""mAP@0,5 surowych i uśrednionych wag biegu gęstej historii (PLAN.md). Bez treningu: sama ocena zapisanych wag.

Co rozstrzyga: czy wahania mAP między sąsiednimi stanami sieci to drgania, które znikają po uśrednieniu wag.
Co obala: tezę, że pojedyncza ocena mAP jest wiarygodną miarą stanu sieci, albo tezę, że uśrednianie wag coś daje.

`dane.py`, `model.py`, `trening.py` skopiowane bez zmian z `../../maly_vit_ruch_czastek_gesta_historia/kod`.

    py -3.11 srednia_wag.py --wyniki ../wyniki           (kilka minut na RTX 3070; około 3 GB RAM)
    py -3.11 srednia_wag.py --wyniki ../wyniki/proba --proba
"""
from __future__ import annotations

import os
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MALY_VIT_WEJSCIA', str(EXP / 'maly_vit_od_zera' / 'eden' / 'wejscia'))
os.environ.setdefault('MALY_VIT_DANE', str(EXP / 'koopman_fastgan_petla' / 'local_inputs' / 'data'))
os.environ.setdefault('MALY_VIT_CACHE', str(EXP / 'maly_vit_od_zera' / 'cache'))

import argparse
import json
import time

import numpy as np
import torch

import dane as D
import trening as T

BIEG = EXP / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny'
CHWILE = (600, 700, 800, 900, 1000)
OKNA = (5, 10, 20)                                       # liczba zapisów co 10 kroków: 50, 100, 200 kroków
WAGI_EMA = (0.2, 0.1)


def wagi(k):
    return torch.load(BIEG / 'wagi' / f'k{k:04d}.pt', map_location='cpu', weights_only=False)['net']


def srednia(stany):
    out = {}
    for n, v in stany[-1].items():
        out[n] = torch.stack([s[n].float() for s in stany]).mean(0).to(v.dtype) if torch.is_floating_point(v) else v.clone()
    return out


@torch.no_grad()
def przelicz_bn(net, Xb, dev):
    """Statystyki BatchNorm od zera na klatkach bazy: średnia skumulowana po paczkach, float32, bez augmentacji."""
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    stare = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    net.train()
    for s in range(0, len(Xb), 32):
        net(torch.from_numpy(Xb[s:s + 32]).to(dev).permute(0, 3, 1, 2).float() / 255.)
    for m, mo in zip(bns, stare):
        m.momentum = mo
    net.eval()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); p.add_argument('--proba', action='store_true')
    arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wal_w, wal_X = D.walidacja()
    if arg.proba:
        wal_w, wal_X = wal_w[:300], wal_X[:300]
    gt = [r['ramki'] for r in wal_w]
    baza = np.load(Path(os.environ['MALY_VIT_CACHE']) / 'baza1500.npy', mmap_mode='r')
    Xb = np.ascontiguousarray(baza[np.linspace(0, len(baza) - 1, 512).astype(int)])
    print('dane wczytane', wal_X.shape, Xb.shape, f'{time.time() - t0:.0f} s', flush=True)
    torch.backends.cudnn.benchmark = False
    u = T.Uczen(47)

    def ocen(sd, bn=False):
        u.net.load_state_dict(sd)
        if bn: przelicz_bn(u.net, Xb, u.dev)
        return float(T.ap50(u.wykryj(wal_X), gt))

    kroki = list(range(500, 1001, 10)) if not arg.proba else [900, 950, 1000]
    chwile = CHWILE if not arg.proba else (1000,)
    stany = {k: wagi(k) for k in range(300, 1001, 10)} if not arg.proba else {k: wagi(k) for k in range(800, 1001, 10)}
    surowe = {}
    for k in kroki:
        surowe[k] = ocen(stany[k])
        print('surowe', k, round(surowe[k], 4), f'{time.time() - t0:.0f} s', flush=True)
    wyn = dict(meta=dict(zrodlo=str(BIEG), klatek_walidacji=len(wal_X), klatek_bn=len(Xb), chwile=list(chwile), okna_zapisow=list(OKNA),
                         proba=bool(arg.proba)), surowe={str(k): v for k, v in surowe.items()})
    tab = {}
    for t in chwile:
        r = dict(surowe_koniec=surowe[t], surowe_koniec_bn_przeliczone=ocen(stany[t], bn=True))
        for K in OKNA:
            ks = [t - 10 * i for i in range(K)][::-1]
            sd = srednia([stany[k] for k in ks])
            r[f'okno{10 * K}'] = dict(srednie_mAP_surowych=float(np.mean([surowe[k] for k in ks if k in surowe])),
                                      najlepsze_surowe=float(np.max([surowe[k] for k in ks if k in surowe])),
                                      usrednione_bn_usrednione=ocen(sd), usrednione_bn_przeliczone=ocen(sd, bn=True))
        tab[str(t)] = r
        print('chwila', t, json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in r.items()}), f'{time.time() - t0:.0f} s', flush=True)
    wyn['usrednianie'] = tab
    # średnia wykładnicza po zapisach od najwcześniejszego wczytanego stanu
    ema = {}
    for a in WAGI_EMA:
        e, r = None, {}
        for k in sorted(stany):
            s = stany[k]
            e = {n: v.clone() for n, v in s.items()} if e is None else {n: ((1 - a) * e[n].float() + a * v.float()).to(v.dtype) if torch.is_floating_point(v) else v.clone() for n, v in s.items()}
            if k in chwile:
                r[str(k)] = ocen(e, bn=True)
        ema[str(a)] = r
        print('ema', a, {k: round(v, 4) for k, v in r.items()}, f'{time.time() - t0:.0f} s', flush=True)
    wyn['srednia_wykladnicza_bn_przeliczone'] = ema

    # kryteria
    if not arg.proba:
        oc = {int(o['krok']): float(o['mAP50']) for o in json.loads((BIEG / 'dane' / 'OCENY.json').read_text(encoding='utf-8'))}
        wyn['K0_zgodnosc_z_biegiem'] = {str(k): dict(bieg=oc[k], teraz=surowe[k], roznica=surowe[k] - oc[k]) for k in range(500, 1001, 100)}
        x = np.array(kroki, float); y = np.array([surowe[k] for k in kroki])
        reszta = y - np.polyval(np.polyfit(x, y, 1), x)
        wyn['K1_drgania'] = dict(rms_zmiany_co_10_krokow=float(np.sqrt(np.mean(np.diff(y) ** 2))),
                                 rms_zmiany_co_10_krokow_po_trendzie=float(np.sqrt(np.mean(np.diff(reszta) ** 2))),
                                 sd_wokol_trendu=float(reszta.std()), trend_na_100_krokow=float(np.polyfit(x, y, 1)[0] * 100),
                                 rozstep_w_oknach_100=[float(y[i:i + 11].max() - y[i:i + 11].min()) for i in range(0, 41, 10)],
                                 rms_zmiany_co_100_krokow=float(np.sqrt(np.mean((y[10:] - y[:-10]) ** 2))))
        k2 = [(tab[str(t)]['okno100']['usrednione_bn_przeliczone'] - tab[str(t)]['okno100']['srednie_mAP_surowych'],
               tab[str(t)]['okno100']['usrednione_bn_przeliczone'] - tab[str(t)]['surowe_koniec']) for t in chwile]
        wyn['K2'] = dict(przewaga_nad_srednia_okna=[a for a, _ in k2], przewaga_nad_koncem_okna=[b for _, b in k2],
                         ile_ponad_0_01=int(sum(a >= 0.01 for a, _ in k2)), ile_ponad_koncem=int(sum(b > 0 for _, b in k2)),
                         spelnione=bool(sum(a >= 0.01 for a, _ in k2) >= 4 and sum(b > 0 for _, b in k2) >= 4))
        print('K1', wyn['K1_drgania']); print('K2', wyn['K2'])
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'SREDNIA_WAG.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
