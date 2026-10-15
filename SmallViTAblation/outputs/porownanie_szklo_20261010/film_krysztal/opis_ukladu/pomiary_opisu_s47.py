"""Pomiary do dokumentu OPIS_UKLADU_OPTYCZNEGO_VIT: wszystkie liczby z JEDNEGO biegu (MalyViT_35000_od_zera_s47, 806 checkpointów,
ziarno 47, 60 epok), z którego pochodzą zapisane aktywacje (pelne768.npy). Blok 6 (indeks 5), zdjęcie TRAIN 0.

A. Wagi reflektorów (neuronów MLP bloku 6): przeciwlasery, kierunek czujnika vs kierunek pchnięcia (z poprawką na γ LayerNorm),
   długości, zmiana kierunków względem kroku 0, wspólny obrót (Prokrustes) wobec zera, wymiar kierunków, anizotropia γ.
B. Aktywacje: odsetek świecących diod, znoszenie się sił w ognisku |Σ z v| / Σ |z v|, |F| / |x|.
C. Dokładny Jakobian pola MLP dla WSZYSTKICH 600 tokenów (x z przejścia modelu, cache filmu 17): siodło, wir, ślad, mapa bloku I + J.
D. Przyciąganie / odpychanie par tokenów po LayerNorm: narzędzie–narzędzie, narzędzie–tło, tło–tło.
E. Porównanie z biegiem „gęstym” (ten sam przepis i ziarno, osobny bieg): odtwarzanie latentu i odległość wag.
Uruchom (z katalogu film_krysztal/opis_ukladu): F:/Python/3.13/python.exe -X utf8 pomiary_opisu_s47.py  → POMIARY_OPISU_S47.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
from lasery_wspolne import dane, oko                                                # noqa: E402

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
WAGI_G = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
C17 = HERE.parent / 'pulapka_17_cache.npz'
EP = 1094
KROKI_A = [0, 100, 1000, 1094, 5470, 10940, 21880, 43760, 65640]
KROKI_C = [0, 1094, 10940, 65640]
gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f3 = lambda x: float(np.round(x, 4))
q = lambda x: dict(med=f3(np.median(x)), p10=f3(np.percentile(x, 10)), p90=f3(np.percentile(x, 90)))

D = dane(0)
kroki_all, narz = D['kroki'].astype(int), D['narz'].astype(bool)
SIATKA = np.stack(np.divmod(np.arange(600), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
ZL = np.load(LATENT, mmap_mode='r')


def wagi(k, katalog=WAGI):
    s = torch.load(katalog / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
    g = lambda nm: s[f'bloki.5.{nm}'].double().numpy()
    return dict(W1=g('mlp.fc1.weight'), b1=g('mlp.fc1.bias'), W2=g('mlp.fc2.weight'), b2=g('mlp.fc2.bias'),
                gam=g('norm2.weight'), bet=g('norm2.bias')), s


def idx(k):
    i = int(np.searchsorted(kroki_all, k)); assert kroki_all[i] == k, k
    return i


wynik = dict(opis=(__doc__ or '').split('\n\n')[0], bieg='MalyViT_35000_od_zera_s47', blok=6, zdjecie='TRAIN 0', token_narzedzia=FOK,
             tokenow_narzedzia=int(narz.sum()))

# ---------------- A. wagi reflektorów ----------------
A = {}
w0, _ = wagi(0)
V0 = w0['W2'] / np.linalg.norm(w0['W2'], axis=0)
W0 = w0['W1'] / np.linalg.norm(w0['W1'], axis=1, keepdims=True)
rng = np.random.default_rng(0)
for k in KROKI_A:
    w, _ = wagi(k)
    V, W1, gam = w['W2'], w['W1'], w['gam']
    vn = V / np.linalg.norm(V, axis=0)
    C = vn.T @ vn; np.fill_diagonal(C, 0); mn = C.min(1)
    ws = W1 * gam[None, :]                                                          # kierunek czujnika w przestrzeni strumienia: γ ⊙ w_c
    cwv = (W1 / np.linalg.norm(W1, axis=1, keepdims=True) * vn.T).sum(1)
    cwv_g = (ws / np.linalg.norm(ws, axis=1, keepdims=True) * vn.T).sum(1)
    s2 = np.linalg.svd(vn, compute_uv=False) ** 2
    U, _, Vt = np.linalg.svd(vn @ V0.T)
    Rot = U @ Vt
    zm = np.linalg.norm(vn - V0) ** 2
    obrot = 1 - np.linalg.norm(vn - Rot @ V0) ** 2 / max(zm, 1e-12) if zm > 1e-9 else float('nan')
    zero = []
    for _ in range(3):
        Dn = (vn - V0)[:, rng.permutation(V0.shape[1])]
        Ub, _, Vtb = np.linalg.svd((V0 + Dn) @ V0.T)
        zero.append(1 - np.linalg.norm(V0 + Dn - Ub @ Vtb @ V0) ** 2 / max(zm, 1e-12) if zm > 1e-9 else float('nan'))
    A[str(k)] = dict(epoka=round(k / EP, 3),
                     przeciwlasery_cos_m05=f3((mn < -.5).mean()), przeciwlasery_cos_m08=f3((mn < -.8).mean()), partner_cos_med=f3(np.median(mn)),
                     cos_w_v_med=f3(np.median(cwv)), cos_gw_v_med=f3(np.median(cwv_g)),
                     dlugosc_v_med=f3(np.median(np.linalg.norm(V, axis=0))), dlugosc_w_med=f3(np.median(np.linalg.norm(W1, axis=1))),
                     cos_v_do_startu_med=f3(np.median((vn * V0).sum(0))),
                     cos_w_do_startu_med=f3(np.median((W1 / np.linalg.norm(W1, axis=1, keepdims=True) * W0).sum(1))),
                     wspolny_obrot_udzial=f3(obrot) if obrot == obrot else None, wspolny_obrot_zero=f3(np.mean(zero)) if zm > 1e-9 else None,
                     udzial_3_skladowych_kierunkow_v=f3(s2[:3].sum() / s2.sum()),
                     gamma_min=f3(gam.min()), gamma_med=f3(np.median(gam)), gamma_max=f3(gam.max()))
    log('A', k, A[str(k)])
wynik['A_wagi'] = A

# ---------------- B. aktywacje ----------------
c17 = np.load(C17)
X, F = c17['X'].astype(np.float64), c17['F'].astype(np.float64)
B = {}
for k in KROKI_A:
    w, _ = wagi(k); i = idx(k)
    Z = np.asarray(ZL[i, 0]).astype(np.float64)                                     # (600, 768)
    fz_norm = np.abs(Z) * np.linalg.norm(w['W2'], axis=0)[None, :]                  # |z_jc| |v_c|
    wyp = Z @ w['W2'].T                                                             # Σ z v (bez b₂)
    znos = np.linalg.norm(wyp, axis=1) / fz_norm.sum(1)
    B[str(k)] = dict(diody_swiecace_z_gt_005=q((Z > 0.05).mean(1)), diody_ujemne_z_lt_m005=q((Z < -0.05).mean(1)),
                     min_z=f3(Z.min()), znoszenie_wypadkowa_do_sumy=q(znos), znoszenie_token_316=f3(znos[FOK]),
                     F_do_x=q(np.linalg.norm(F[i], axis=1) / np.linalg.norm(X[i], axis=1)),
                     norma_x_med=f3(np.median(np.linalg.norm(X[i], axis=1))))
    log('B', k, B[str(k)])
wynik['B_aktywacje'] = B

# ---------------- C. dokładny Jakobian dla wszystkich tokenów ----------------
Cw = {}
n = 192
for k in KROKI_C:
    w, _ = wagi(k); i = idx(k)
    x = X[i]                                                                        # (600, 192)
    u = x - x.mean(1, keepdims=True); sd = np.sqrt((u * u).mean(1) + 1e-5)
    a = (w['gam'] * u / sd[:, None] + w['bet']) @ w['W1'].T + w['b1']                # (600, 768)
    zr = np.asarray(ZL[i, 0]).astype(np.float64)                                    # float64: w float16 suma kwadratów się przepełnia
    zgod = 1 - ((gelu(a) - zr) ** 2).sum() / ((zr - zr.mean()) ** 2).sum()
    rec = {nm: [] for nm in ('ujemne', 'slad', 'sym', 'rot', 'norma', 'maxre', 'rozszerz', 'promien')}
    for j in range(600):
        Jln = w['gam'][:, None] * ((np.eye(n) - 1.0 / n) / sd[j] - np.outer(u[j], u[j]) / (n * sd[j] ** 3))
        J = (w['W2'] * dgelu(a[j])[None, :]) @ w['W1'] @ Jln
        S, A_ = (J + J.T) / 2, (J - J.T) / 2
        ev = np.linalg.eigvalsh(S); lam = np.linalg.eigvals(J)
        rec['ujemne'].append((ev < 0).mean()); rec['slad'].append(np.trace(J) / n)
        rec['sym'].append(np.linalg.norm(S) ** 2 / np.linalg.norm(J) ** 2); rec['rot'].append(np.linalg.norm(A_) / np.linalg.norm(S))
        rec['norma'].append(np.linalg.norm(J, 2)); rec['maxre'].append(lam.real.max())
        rec['rozszerz'].append((np.abs(1 + lam) > 1).mean()); rec['promien'].append(np.abs(1 + lam).max())
    rec = {nm: np.array(v) for nm, v in rec.items()}
    Cw[str(k)] = dict(zgodnosc_GELU_a_z_latentem_R2=f3(zgod),
                      wszystkie={nm: q(v) for nm, v in rec.items()},
                      narzedzie={nm: q(v[narz]) for nm, v in rec.items()},
                      tlo={nm: q(v[~narz]) for nm, v in rec.items()},
                      token_316={nm: f3(v[FOK]) for nm, v in rec.items()})
    log('C', k, 'R2', round(zgod, 5), {nm: Cw[str(k)]['wszystkie'][nm]['med'] for nm in rec})
wynik['C_jakobian'] = Cw
wynik['C_zero_macierz_losowa'] = dict(sym=f3((n + 1) / (2 * n)), rot=f3(np.sqrt((n - 1) / (n + 1))),
                                      opis='dla macierzy o niezależnych wyrazach N(0,1): E‖S‖² = n(n+1)/2, E‖A‖² = n(n−1)/2')


# ---------------- D. pary tokenów po LayerNorm ----------------
def _ln(a):
    a = a - a.mean(-1, keepdims=True)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-5)


def odl(Y):
    g = (Y * Y).sum(1)
    return np.sqrt(np.maximum(g[:, None] + g[None, :] - 2 * Y @ Y.T, 0))


Dw = {}
tri = np.triu(np.ones((600, 600), bool), 1)
kl = dict(narz_narz=np.outer(narz, narz) & tri, narz_tlo=(np.outer(narz, ~narz) | np.outer(~narz, narz)) & tri, tlo_tlo=np.outer(~narz, ~narz) & tri)
for k in KROKI_C:
    i = idx(k)
    r = odl(_ln(X[i] + F[i])) / np.maximum(odl(_ln(X[i])), 1e-9)
    lr = np.log(np.maximum(r, 1e-9))
    Dw[str(k)] = {nm: dict(par=int(m.sum()), med_log_r=f3(np.median(lr[m])), mnoznik_med=f3(np.exp(np.median(lr[m]))), przyciagane=f3((r[m] < 1).mean()))
                  for nm, m in kl.items()}
    log('D', k, Dw[str(k)])
wynik['D_pary_po_LN'] = Dw

# ---------------- E. bieg gęsty a bieg s47 ----------------
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402
xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float() / 255.
net = M.MalyViT().eval(); net.zapisuj_latent(True)
E = {}
for k in [0, 1000, 5300, 22974]:
    _, sA = wagi(k); _, sG = wagi(k, WAGI_G); _, s0 = wagi(0)
    zr = np.asarray(ZL[idx(k), 0]).astype(np.float64)
    r2 = {}
    for nm, s in (('s47', sA), ('gesty', sG)):
        net.load_state_dict(s)
        with torch.no_grad():
            _, lat = net(xin)
        z = lat[0].double().numpy()
        r2[nm] = f3(1 - ((z - zr) ** 2).sum() / ((zr - zr.mean()) ** 2).sum())
    wek = lambda s: torch.cat([v.flatten().double() for kk, v in s.items() if v.dtype.is_floating_point and 'running' not in kk])
    dA, dG, d0 = wek(sA), wek(sG), wek(s0)
    E[str(k)] = dict(R2_latentu=r2, odleglosc_wag_wzgledna=f3(float((dA - dG).norm() / max(float((dA - d0).norm()), 1e-12))) if k > 0 else 0.0)
    log('E', k, E[str(k)])
wynik['E_dwa_biegi'] = E

(HERE / 'POMIARY_OPISU_S47.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
log('zapisano POMIARY_OPISU_S47.json')
