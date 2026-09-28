#!/usr/bin/env python3
"""Ocena po kampanii (prośba użytkownika 2026-09-29, analiza dodatkowa, nie kryterium z planu): jak końcowe sieci
wykrywają narzędzie na syntetycznych sondach — obroty 90/180/270°, czarne i żółte tło, skala, długość, przesunięcia,
jasność, perspektywa — i jak pewny jest detektor, w porównaniu ramion.

Sondy: 12 wycinków narzędzia × 15 wersji + `empty_black` (narzędzie usunięte), 176 px, jak w pętli (`dane.sondy`).
Sieci: końcowe wagi każdego ramienia (krok 4500) i stan po rozgrzewce (krok 1000), 10 ziaren.
Miary na sondzie z narzędziem:
  pewnosc_trafna  najwyższa pewność wśród wykryć, które pokrywają narzędzie (IoU ≥ 0,5 z ramką z maski); 0, gdy brak;
  trafienie_top1  czy najpewniejsze wykrycie pokrywa narzędzie (IoU ≥ 0,5);
na `empty_black`: falszywy_alarm = najwyższa pewność (narzędzia nie ma).
Porównanie: średnia po 12 wycinkach dla każdej wersji, potem różnica `tozsamosc` − ramię i liczba ziaren, w których
`tozsamosc` jest lepsza. 16 wersji × 4 porównania = wiele testów naraz, więc pojedyncze „wygrane” traktować ostrożnie.

    <venv>/python ocena_sond.py  ->  wyniki_eden/OCENA_SOND.json, OCENA_SOND.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki_eden'
EXP = HERE.parent
os.environ.setdefault('MALY_VIT_WEJSCIA', str(EXP / 'maly_vit_od_zera' / 'eden' / 'wejscia'))
os.environ.setdefault('MALY_VIT_DANE', str(EXP / 'koopman_fastgan_petla' / 'local_inputs' / 'data'))
os.environ.setdefault('MALY_VIT_CACHE', str(HERE / 'wyniki' / 'cache'))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, str(HERE / 'kod'))
import dane as D  # noqa: E402
import model as M  # noqa: E402
from trening import iou  # noqa: E402

RAMIONA = ('tozsamosc', 'current', 'losowe', 'kolejne', 'bez', 'rozgrzewka')
KONTROLE = ('current', 'losowe', 'kolejne', 'bez', 'rozgrzewka')


@torch.no_grad()
def ocen(net, X, ramki):
    net.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o, _ = net(X.cuda())
    bx, sc = M.dekoduj(o)
    bx, sc = bx.cpu().numpy(), sc.cpu().numpy()
    pt, tr, fa = np.full(len(X), np.nan), np.full(len(X), np.nan), np.full(len(X), np.nan)
    for i, r in enumerate(ramki):
        if r is None:
            fa[i] = float(sc[i].max())
            continue
        v = iou(bx[i], np.array(r, np.float32))
        ok = v >= .5
        pt[i] = float(sc[i][ok].max()) if ok.any() else 0.0
        tr[i] = float(v[int(sc[i].argmax())] >= .5)
    return pt, tr, fa


def main() -> int:
    X, tn, tt, ramki, obecn, opis = D.sondy()
    war = np.array([o['wariant'] for o in opis])
    wersje = [v for v in dict.fromkeys(war)]
    net = M.MalyViT().cuda()
    wyn = {}                                             # ziarno -> ramię -> miara -> wersja -> średnia
    for d in sorted(W.glob('petla_s*')):
        s = d.name.split('_s')[1]
        wyn[s] = {}
        for r in RAMIONA:
            p = d / 'wspolny' / 'k02.pt' if r == 'rozgrzewka' else d / r / 'wagi_koncowe.pth'
            st = torch.load(p, map_location='cpu', weights_only=False)
            net.load_state_dict(st['net'] if r == 'rozgrzewka' else st)
            pt, tr, fa = ocen(net, X, ramki)
            wyn[s][r] = {m: {v: float(np.nanmean(a[war == v])) for v in wersje if not np.all(np.isnan(a[war == v]))}
                         for m, a in (('pewnosc_trafna', pt), ('trafienie_top1', tr), ('falszywy_alarm', fa))}
        print(s, {r: round(float(np.mean(list(wyn[s][r]['pewnosc_trafna'].values()))), 3) for r in RAMIONA}, flush=True)
    ziarna = list(wyn)
    tab = {}
    for m in ('pewnosc_trafna', 'trafienie_top1', 'falszywy_alarm'):
        wer = list(wyn[ziarna[0]]['tozsamosc'][m])
        tab[m] = {}
        for v in wer:
            A = {r: np.array([wyn[s][r][m][v] for s in ziarna]) for r in RAMIONA}
            lepiej = (lambda a, b: a < b) if m == 'falszywy_alarm' else (lambda a, b: a > b)  # alarm: mniej = lepiej
            tab[m][v] = dict(srednio={r: float(A[r].mean()) for r in RAMIONA},
                             tozsamosc_minus={c: float((A['tozsamosc'] - A[c]).mean()) for c in KONTROLE},
                             tozsamosc_lepsza_w={c: int(lepiej(A['tozsamosc'], A[c]).sum()) for c in KONTROLE})
    wsz = {r: np.array([np.mean(list(wyn[s][r]['pewnosc_trafna'].values())) for s in ziarna]) for r in RAMIONA}
    pod = dict(pewnosc_trafna_wszystkie_wersje={r: float(wsz[r].mean()) for r in RAMIONA},
               tozsamosc_lepsza_w_ziarnach={c: int((wsz['tozsamosc'] > wsz[c]).sum()) for c in KONTROLE},
               wersje_gdzie_tozsamosc_lepsza_od_losowych_w_8_z_10=[v for v, x in tab['pewnosc_trafna'].items()
                                                                   if x['tozsamosc_lepsza_w']['losowe'] >= 8])
    (W / 'OCENA_SOND.json').write_text(json.dumps(dict(podsumowanie=pod, wersje=tab, ziarna=wyn), indent=1,
                                                   ensure_ascii=False), encoding='utf-8')
    print(json.dumps(pod, indent=1, ensure_ascii=False))
    for v, x in tab['pewnosc_trafna'].items():
        print(f"{v:18s} " + '  '.join(f"{r} {x['srednio'][r]:.3f}" for r in RAMIONA) +
              f"  | tożsamość lepsza od losowych w {x['tozsamosc_lepsza_w']['losowe']}/10, od current w {x['tozsamosc_lepsza_w']['current']}/10")
    x = tab['falszywy_alarm']['empty_black']
    print('empty_black fałszywy alarm: ' + '  '.join(f"{r} {x['srednio'][r]:.3f}" for r in RAMIONA))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        wer = list(tab['pewnosc_trafna'])
        fig, ax = plt.subplots(figsize=(13, 4.8))
        kol = dict(tozsamosc='tab:red', current='tab:orange', losowe='tab:blue', kolejne='tab:green', bez='tab:gray',
                   rozgrzewka='black')
        szer = .8 / len(RAMIONA)
        for j, r in enumerate(RAMIONA):
            ax.bar(np.arange(len(wer)) + (j - len(RAMIONA) / 2 + .5) * szer,
                   [tab['pewnosc_trafna'][v]['srednio'][r] for v in wer], szer, color=kol[r], label=r)
        ax.set_xticks(np.arange(len(wer))); ax.set_xticklabels(wer, rotation=40, ha='right')
        ax.set_ylabel('pewność trafnego wykrycia (średnio 10 ziaren)')
        ax.set_title('Sondy syntetyczne: pewność detektora na narzędziu, końcowe sieci ramion')
        ax.legend(ncol=6, fontsize=8); fig.tight_layout(); fig.savefig(W / 'OCENA_SOND.png', dpi=130)
    except Exception as e:  # wykres pomocniczy
        print('wykres:', e)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
