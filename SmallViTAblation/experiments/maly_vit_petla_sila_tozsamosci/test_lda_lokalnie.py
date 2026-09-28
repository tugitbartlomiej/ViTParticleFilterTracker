#!/usr/bin/env python3
"""Zgodność wektora tożsamości liczonego w numpy (kod/sila.py, na Edenie brak sklearn) z wersją sklearn
(maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py) na latentach sond ziarna 42 z kroku 1000 (wersja 1, lokalnie)
i z kroku 0 (sieć nieuczona). Wymóg przed zgłoszeniem: cosinus ≥ 0,999.

    <venv>/python test_lda_lokalnie.py  ->  wyniki/TEST_LDA.json
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
V1 = EXP / 'maly_vit_od_zera'
os.environ.setdefault('MALY_VIT_WEJSCIA', str(V1 / 'eden' / 'wejscia'))
sys.path.insert(0, str(HERE / 'kod'))
import sila as SI  # noqa: E402

spec = importlib.util.spec_from_file_location('w21', EXP / 'maly_vit_wektor_tozsamosci' / 'wektor_tozsamosci_21ep.py')
WEK = importlib.util.module_from_spec(spec); spec.loader.exec_module(WEK)
man = json.loads((V1 / 'eden' / 'wejscia' / 'probe' / 'manifest.json').read_text(encoding='utf-8'))
obecn = np.array([bool(m['semantic_presence']) for m in man])
war = np.array([m['variant'] for m in man]); zr = np.array([m['source_file'] for m in man])
BIEG = V1 / 'wyniki' / 'petla_s42' / 'wspolny'
wyn = {}
for nazwa, L in (('krok_1000', np.load(BIEG / 'k02' / 'LATENT.npz')['latent'][-5:]),
                 ('krok_0', np.load(BIEG / 'k01' / 'LATENT.npz')['latent'][:1])):
    Lk = L.astype(np.float32).mean(0)
    a = SI.wektor(Lk, obecn)
    b = WEK.naucz(*WEK.zbior(Lk.astype(np.float64), obecn, war, zr, np.ones(len(war), bool))[:2])
    cos = float(a['w'] @ b['w'])
    ds = np.abs(SI.sila(Lk[obecn, 0], a) - (((Lk[obecn, 0] - b['mu']) / b['sd']) @ b['w']))
    wyn[nazwa] = dict(cosinus=cos, max_roznica_sily=float(ds.max()), sila_narzedzia=a['sila_narzedzia'],
                      sila_negatywow=a['sila_negatywow'], sd_negatywow=a['sd_negatywow'])
    print(nazwa, wyn[nazwa], flush=True)
wyn['zgodne'] = all(v['cosinus'] >= 0.999 for v in wyn.values() if isinstance(v, dict))
(HERE / 'wyniki').mkdir(exist_ok=True)
(HERE / 'wyniki' / 'TEST_LDA.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
print('ZGODNE' if wyn['zgodne'] else 'NIEZGODNE')
