#!/usr/bin/env python3
"""Sprawdzenie prognozy siły tożsamości z kroku 1000 na prawdziwym przebiegu do kroku 1500 (wspolny/k03, v1 s42)."""
import sys, json, numpy as np, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; V1 = EXP/'maly_vit_od_zera'
sys.path.insert(0, str(V1))
import tozsamosc as TZ
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
WEK = mod('w21', EXP/'maly_vit_wektor_tozsamosci'/'wektor_tozsamosci_21ep.py')
PR2 = mod('p2', EXP/'maly_vit_wektor_tozsamosci'/'kod'/'prognoza.py')
PROBE = V1/'eden'/'wejscia'/'probe'; BIEG = V1/'wyniki'/'petla_s42'
man = json.loads((PROBE/'manifest.json').read_text(encoding='utf-8'))
obecn = np.array([bool(m['semantic_presence']) for m in man])
war = np.array([m['variant'] for m in man]); zr = np.array([m['source_file'] for m in man])
L = np.concatenate([np.load(BIEG/'wspolny'/f'k0{k}'/'LATENT.npz')['latent'] for k in (1,2)])[1:]
Lb = L[:len(L)//5*5].astype(np.float64).reshape(-1,5,*L.shape[1:]).mean(1)
m = WEK.naucz(*WEK.zbior(Lb[-1], obecn, war, zr, np.ones(len(war), bool))[:2])
S = ((Lb[:, obecn, 0]-m['mu'])/m['sd'])@m['w']; Y = S[:,:,None]
L3 = np.load(BIEG/'wspolny'/'k03'/'LATENT.npz')['latent'][-5:].astype(np.float64).mean(0)
S3 = ((L3[obecn,0]-m['mu'])/m['sd'])@m['w']
real = S3 - S[-1]
poz = np.where(obecn)[0]; kotw = [j for j,i in enumerate(poz) if man[i]['variant']=='original']
print('prawdziwa zmiana (kotwice):', np.round(real[kotw],2))
for md in ('langevin','koopman'):
    Wt = PR2.dopasuj(Y, md); p = PR2.rollout(Y, Wt, 10, md)[-1,:,0]-S[-1]
    for nm, idx in (('wszystkie', slice(None)), ('kotwice', kotw)):
        e = np.abs(p[idx]-real[idx]); e0 = np.abs(real[idx])
        print(f'{md:8s} {nm:9s} blad {np.median(e):.2f} vs brak zmiany {np.median(e0):.2f} | zysk {1-e.mean()/e0.mean():+.2f} | znak zgodny {np.mean(np.sign(p[idx])==np.sign(real[idx])):.2f}')
