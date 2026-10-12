"""Czy lasery ostatniego bloku MLP mają „przeciwlasery”? Test na wagach fc1/fc2 bloku 5 (ostatniego) MałegoViT.

Laser c ma dwa wektory: w_c = wiersz fc1 (celownik: kiedy świeci, z_c = GELU(w_c·LN(x) + b_c)) i v_c = kolumna fc2
(kierunek pchnięcia: blok dodaje do tokenu Σ_c z_c v_c). GELU jest prawie nieujemne (min ≈ −0,17), więc laser pcha
praktycznie tylko w stronę +v_c. Żeby pchnąć w stronę −v_c, potrzebny jest inny laser c' z v_c' ≈ −v_c („przeciwlaser”).

Liczy dla checkpointów biegu 35 000 (gęsty zapis):
- cos najbardziej przeciwnego partnera każdego v_c i odsetek laserów z partnerem cos < −0,5 i < −0,8,
- cos(w_c, v_c): czy laser pcha token w stronę, z której go „widzi” (> 0), czy z powrotem (< 0, siła zwrotna),
- udział 3 głównych składowych w kierunkach v_c (ile z układu kierunków widać w rzucie 3D),
- medianę długości |v_c| (siła lasera przy natężeniu 1).

Uruchom: F:/Python/3.13/python.exe -X utf8 sprawdz_przeciwlasery.py   → PRZECIWLASERY.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WAGI = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/gesty/wagi'
BLOK = 5


def miary(sd):
    V = sd[f'bloki.{BLOK}.mlp.fc2.weight'].float().numpy()              # (192, 768): kolumna c = v_c
    W = sd[f'bloki.{BLOK}.mlp.fc1.weight'].float().numpy()              # (768, 192): wiersz c = w_c
    dl = np.linalg.norm(V, axis=0)
    Vn, Wn = V / dl, W / np.linalg.norm(W, axis=1, keepdims=True)
    C = Vn.T @ Vn
    np.fill_diagonal(C, 0.0)
    mn = C.min(1)
    s2 = np.linalg.svd(Vn, compute_uv=False) ** 2
    return dict(partner_cos_med=float(np.median(mn)),
                partner_ponizej_m05=float((mn < -.5).mean()),
                partner_ponizej_m08=float((mn < -.8).mean()),
                cos_w_v_med=float(np.median((Wn * Vn.T).sum(1))),
                udzial_3_skladowych=float(s2[:3].sum() / s2.sum()),
                dlugosc_v_med=float(np.median(dl)))


def main():
    pliki = sorted(f for f in os.listdir(WAGI) if f.endswith('.pt'))
    wybor = pliki[::max(1, len(pliki) // 16)] + [pliki[-1]]
    wynik = {}
    for f in dict.fromkeys(wybor):
        s = torch.load(WAGI / f, map_location='cpu', weights_only=False)
        m = miary(s['net'])
        m.update(krok=int(s['krok']), epoka=float(s['epoka']))
        wynik[f[:-3]] = m
        print(f[:-3], {k: round(v, 3) for k, v in m.items()})
    (HERE / 'PRZECIWLASERY.json').write_text(json.dumps(dict(opis=(__doc__ or '').split('\n\n')[0], blok=BLOK, wyniki=wynik),
                                                         ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
