#!/usr/bin/env python3
"""Punkt odniesienia dla zmiany „w bok”: ile latent sond zmienia się poza kierunkiem tożsamości w prawdziwym treningu."""
import sys, json, numpy as np, importlib.util
sys.path.insert(0, '.')
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; V1 = EXP/'maly_vit_od_zera'
sys.path.insert(0, str(V1))
import tozsamosc as TZ
spec = importlib.util.spec_from_file_location('w21', EXP/'maly_vit_wektor_tozsamosci'/'wektor_tozsamosci_21ep.py')
WEK = importlib.util.module_from_spec(spec); spec.loader.exec_module(WEK)
PROBE = V1/'eden'/'wejscia'/'probe'; BIEG = V1/'wyniki'/'petla_s42'
man = json.loads((PROBE/'manifest.json').read_text(encoding='utf-8'))
obecn = np.array([bool(m['semantic_presence']) for m in man])
war = np.array([m['variant'] for m in man]); zr = np.array([m['source_file'] for m in man])
L = np.concatenate([np.load(BIEG/'wspolny'/f'k0{k}'/'LATENT.npz')['latent'] for k in (1,2)])[1:]
Lb = L[:len(L)//5*5].astype(np.float64).reshape(-1,5,*L.shape[1:]).mean(1)
m = WEK.naucz(*WEK.zbior(Lb[-1], obecn, war, zr, np.ones(len(war), bool))[:2])
X = (Lb[:, obecn, 0] - m['mu'])/m['sd']
S = X @ m['w']
poz = np.where(obecn)[0]
kotw = [j for j,i in enumerate(poz) if man[i]['variant']=='original']
for h in (1, 5, 10):
    dX = X[h:] - X[:-h]; dS = S[h:] - S[:-h]
    perp = dX - dS[..., None]*m['w']
    r = np.linalg.norm(perp, axis=-1)/np.abs(dS).clip(1e-6)
    rk = r[:, kotw]
    print(f'H={h*50} krokow: bok/|ds| mediana wszystkie {np.median(r):.1f}, kotwice {np.median(rk):.1f}; |ds| mediana {np.median(np.abs(dS)):.3f}; |bok| mediana {np.median(np.linalg.norm(perp,axis=-1)):.2f}')
# ostatnie 10 blokow (kroki 550->1000) na kotwicach
dX = X[-1]-X[-11]; dS = S[-1]-S[-11]; perp = dX - dS[:,None]*m['w']
print('ostatnie 500 krokow, kotwice: ds', np.round(dS[kotw],2), ' |bok|', np.round(np.linalg.norm(perp,axis=-1)[kotw],1))
