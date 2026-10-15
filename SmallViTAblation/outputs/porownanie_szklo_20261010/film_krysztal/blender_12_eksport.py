"""Eksport układu z filmu 12 (zasada Z14) do sceny Blendera: blender_12_dane.npz.

Ta sama geometria co w film_lasery_12_reflektory_stale.py:
- 768 stałych reflektorów na sferze RR = 50 (spirala Fibonacciego, przypisanie węgierskie do −d_c w kroku 22 974),
- szklana sfera R = 34, jedno załamanie na powierzchni w punkcie E_jc = p_j − s · d_c (|E_jc| = R), w środku prosto,
- cząsteczki p_j = P W₂ (z_j − z̄), skala na krok (99. percentyl promienia = 0,55 R), jasność wiązki = z_jc / 2.
Zapisuje NKL kroków kluczowych (gęściej w epokach 0–2, jak tempo filmu). Wiązek 460 800 jest za dużo na rury w Blenderze,
więc dla każdego reflektora bierze M cząsteczek o największym średnim z_jc; reflektor pokazowy C1 ma wszystkie 600.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 blender_12_eksport.py [M]"""
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from lasery_wspolne import EP, HERE, TOK, dane

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NK, R, RR, NKL = 768, 34.0, 50.0, 24
ASSETS = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10'
LATENT = ASSETS / '03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
sys.path.insert(0, str(HERE.parents[2] / 'app'))
from wspolne import kierunki, kolory_kanalow                                        # noqa: E402

D = dane(0)
kroki_all, iou_all, narz = D['kroki'].astype(np.int64), D['iou'], D['narz']
c_ = np.load(HERE / 'reflektor_8_cache.npz'); KW, V = c_['KW'], c_['V']
NT = len(KW)
idx_lat = np.searchsorted(kroki_all, KW)
Z = np.clip(np.asarray(np.load(LATENT, mmap_mode='r')[idx_lat, 0]).astype(np.float32), 0, None)   # (341, 600, 768)
log('dane', V.shape, Z.shape)

Vn_end = V[-1] / np.linalg.norm(V[-1], axis=0)
U_, _, _ = np.linalg.svd(Vn_end - Vn_end.mean(1, keepdims=True), full_matrices=False)
P3 = U_[:, :3].T
PV = np.einsum('ad,tdc->tca', P3, V)
DIRS = PV / np.linalg.norm(PV, axis=2, keepdims=True)
POS_RAW = np.einsum('tjc,tca->tja', Z - Z.mean(1, keepdims=True), PV)
R99 = np.percentile(np.linalg.norm(POS_RAW, axis=2), 99, axis=1)
POS = POS_RAW / R99[:, None, None] * 0.55 * R
POW = R99[-1] / R99
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
pary = np.array([(np.where(np.eye(NK, dtype=bool), 0.0, Vn[t].T @ Vn[t]).min(1) < -0.5).mean() for t in range(NT)])
Zr = Z / 2.0
C1 = int(np.argmax(Z[-1].var(0)))
FIB = kierunki(False).astype(np.float64)
_r, _c = linear_sum_assignment(FIB @ DIRS[-1].T)
SLOT = np.empty(NK, int); SLOT[_c] = _r
RFIX = RR * FIB[SLOT]
KOL = kolory_kanalow()[:, :3][SLOT]

# kroki kluczowe: połowa w epokach 0–2, połowa w 2–21 (tempo filmu 12), ostatni = koniec
kr = np.concatenate([np.linspace(0, 2 * EP, NKL // 2, endpoint=False), np.linspace(2 * EP, KW[-1], NKL - NKL // 2)])
KL = np.unique(np.searchsorted(KW, kr).clip(0, NT - 1))
log('kroki kluczowe', KW[KL].tolist())


def wejscie(Pp, Dc):
    """punkty załamania E (…, 3) dla par (p_j, d_c) podanych jako tablice tej samej długości."""
    b = (Pp * Dc).sum(-1)
    s = b + np.sqrt(np.maximum(b * b - (Pp * Pp).sum(-1) + R * R, 0))
    return Pp - s[..., None] * Dc


# wiązki: dla reflektora c — M cząsteczek o największym średnim z_jc w krokach kluczowych
srz = Zr[KL].mean(0)                                                               # (600, 768)
J = np.argsort(-srz, axis=0)[:M].T                                                 # (768, M)
BC = np.repeat(np.arange(NK), M); BJ = J.reshape(-1)
BE = np.stack([wejscie(POS[k][BJ], DIRS[k][BC]) for k in KL])                      # (NKL, NB, 3)
BI = np.stack([Zr[k][BJ, BC] for k in KL])                                         # (NKL, NB)
HE = np.stack([wejscie(POS[k], np.broadcast_to(DIRS[k][C1], (TOK, 3))) for k in KL])   # (NKL, 600, 3)
HI = Zr[KL][:, :, C1]                                                              # (NKL, 600)
u = BE[-1] - RFIX[BC]; u /= np.linalg.norm(u, axis=1, keepdims=True)
kat = np.degrees(np.arccos(np.clip((u * DIRS[-1][BC]).sum(1), -1, 1)))
log(f'wiązek {len(BC)} (+600 reflektora {C1}); średni kąt załamania na końcu {np.average(kat, weights=BI[-1]):.0f}°')

# jasności znormalizowane na krok, jak w filmie: cząsteczka = suma natężeń, reflektor = średnie natężenie, dioda = z_j,C1
S = Zr[KL].sum(2)
SJN = np.clip(S / np.percentile(S, 90, axis=1, keepdims=True), 0, 1.6)
MOC = Zr[KL].mean(1)
MOCN = np.clip(MOC / np.maximum(np.percentile(MOC, 95, axis=1, keepdims=True), 1e-6), 0.15, 1.5)
ZC1 = Z[KL][:, :, C1]
ZC1N = np.clip(ZC1 / np.maximum(np.percentile(ZC1, 99, axis=1, keepdims=True), 1e-6), 0, 1)
log('natężenia wiązek: percentyle 50/90/99 =', np.percentile(BI, [50, 90, 99]).round(3).tolist())

np.savez_compressed(HERE / 'blender_12_dane.npz', R=R, RR=RR, C1=C1, M=M,
                    SJN=SJN.astype('f4'), MOCN=MOCN.astype('f4'), ZC1N=ZC1N.astype('f4'),
                    KROKI=KW[KL], EPOKI=KW[KL] / EP, IOU=iou_all[idx_lat][KL], PARY=pary[KL], POW=POW[KL],
                    RFIX=RFIX.astype('f4'), KOL=KOL.astype('f4'), NARZ=narz,
                    POS=POS[KL].astype('f4'), DIRS=DIRS[KL].astype('f4'), MOC=Zr[KL].mean(1).astype('f4'),
                    BC=BC.astype('i4'), BJ=BJ.astype('i4'), BE=BE.astype('f4'), BI=BI.astype('f4'),
                    HE=HE.astype('f4'), HI=HI.astype('f4'), ZC1=Z[KL][:, :, C1].astype('f4'))
log('zapisano blender_12_dane.npz')
