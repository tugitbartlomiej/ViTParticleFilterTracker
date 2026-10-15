"""Przeliczenie liczb z ZASADY_LASER_KOLOID.md i matematyka_reflektorow.tex na wagach biegu, z którego pochodzą zapisane aktywacje
(MalyViT_35000_od_zera_s47: ziarno 47, 806 checkpointów, 60 epok). PRZECIWLASERY.json, OBROT_REFLEKTORA.json i PULAPKA_JAKOBIAN.json
liczono na wagach innego biegu (01_bieg_35000_gesty_zapis) i zostają bez zmian jako zapis tamtego biegu.

P. Przeciwlasery (definicje jak w sprawdz_przeciwlasery.py) + cos(γ⊙w_c, v_c), długości |w_c|, liczba parametrów sieci.
O. Obrót reflektora (definicje jak w sprawdz_obrot_reflektora.py): te same pary kroków co wcześniej, okna 5-epokowe, epoka 1 → 60;
   wspólny obrót liczony na surowych wektorach (jak wcześniej) i na kierunkach jednostkowych (jak w POMIARY_OPISU_S47.json).
J. Dokładny Jakobian pola MLP bloku 6 dla 600 tokenów zdjęcia TRAIN 0 (x z nowego przejścia modelu w float32) w epokach 0–60.
W. Czy liczby z TEST_SFERY.json / WIAZKI.json (Eden: 84–98 % drgań, 75 % dryfu, cos 0,24) pochodzą z biegu s47.
F. Geometria filmu 12 (przypisanie węgierskie miejsc, ważony średni kąt załamania) na wagach s47 i, dla kontroli, biegu gęstego.
Uruchom (z katalogu film_krysztal): F:/Python/3.13/python.exe -X utf8 przelicz_zasady_s47.py  → PRZELICZENIE_ZASADY_S47.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import erf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from lasery_wspolne import dane, oko                                                # noqa: E402

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
WAGI_G = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
WK = HERE.parent / 'eden/wyniki/wiazki_koloidy'
EP = 1094
KROKI_EP = [0, 1094, 5470, 10940, 21880, 43760, 65640]                              # epoki 0, 1, 5, 10, 20, 40, 60
KROKI_P = sorted(set(KROKI_EP + [5300, 11000, 22974]))                               # + kroki starej tabeli (bieg gęsty)
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
DOSTEPNE = np.array(sorted(int(f[1:6]) for f in os.listdir(WAGI) if f.endswith('.pt')))


def najblizszy(k):
    return int(DOSTEPNE[np.argmin(np.abs(DOSTEPNE - k))])


def wczytaj(k, katalog=WAGI):
    return torch.load(katalog / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']


def wagi(k, katalog=WAGI):
    s = wczytaj(k, katalog)
    g = lambda nm: s[f'bloki.5.{nm}'].double().numpy()
    return dict(W1=g('mlp.fc1.weight'), b1=g('mlp.fc1.bias'), W2=g('mlp.fc2.weight'), b2=g('mlp.fc2.bias'),
                gam=g('norm2.weight'), bet=g('norm2.bias'))


def idx(k):
    i = int(np.searchsorted(kroki_all, k)); assert kroki_all[i] == k, k
    return i


wynik = dict(opis=(__doc__ or '').split('\n\n')[0], bieg='MalyViT_35000_od_zera_s47', blok=6, zdjecie='TRAIN 0', token_narzedzia=FOK,
             checkpointow=int(len(DOSTEPNE)), ostatni_krok=int(DOSTEPNE[-1]))
s_ = wczytaj(0)
wynik['parametry_sieci'] = int(sum(v.numel() for kk, v in s_.items() if v.dtype.is_floating_point and 'running' not in kk))
log('checkpointów', len(DOSTEPNE), 'parametrów', wynik['parametry_sieci'])


# ---------------- P. przeciwlasery ----------------
def przeciwlasery(w):
    V, W1, gam = w['W2'], w['W1'], w['gam']
    dl = np.linalg.norm(V, axis=0); vn = V / dl
    C = vn.T @ vn; np.fill_diagonal(C, 0); mn = C.min(1)
    wn = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    gw = W1 * gam[None, :]; gwn = gw / np.linalg.norm(gw, axis=1, keepdims=True)
    s2 = np.linalg.svd(vn, compute_uv=False) ** 2
    return dict(partner_ponizej_m05=f3((mn < -.5).mean()), partner_ponizej_m08=f3((mn < -.8).mean()),
                partner_cos_med=f3(np.median(mn)), partner_cos_min=f3(mn.min()),
                cos_w_v_med=f3(np.median((wn * vn.T).sum(1))), cos_gw_v_med=f3(np.median((gwn * vn.T).sum(1))),
                udzial_3_skladowych=f3(s2[:3].sum() / s2.sum()),
                dlugosc_v_med=f3(np.median(dl)), dlugosc_w_med=f3(np.median(np.linalg.norm(W1, axis=1))))


P = {}
for k in KROKI_P:
    P[str(k)] = dict(epoka=round(k / EP, 2), **przeciwlasery(wagi(k)))
    log('P', k, P[str(k)])
wynik['P_przeciwlasery'] = P
wynik['P_kontrola_bieg_gesty_22974'] = przeciwlasery(wagi(22974, WAGI_G))
log('P kontrola (bieg gęsty, 22974)', wynik['P_kontrola_bieg_gesty_22974'])


# ---------------- O. obrót reflektora ----------------
def wspolny_obrot(A, B):
    U, _, Vt = np.linalg.svd(B @ A.T)
    return 1 - np.linalg.norm(B - U @ Vt @ A) ** 2 / np.linalg.norm(B - A) ** 2


def obrot(a, b, rng):
    wa, wb = wagi(a), wagi(b)
    out = {}
    for nm, X, Y in (('v', wa['W2'], wb['W2']), ('w', wa['W1'].T, wb['W1'].T)):
        xn, yn = X / np.linalg.norm(X, axis=0), Y / np.linalg.norm(Y, axis=0)
        zero = [wspolny_obrot(X, X + (Y - X)[:, rng.permutation(X.shape[1])]) for _ in range(3)]
        zero_k = [wspolny_obrot(xn, xn + (yn - xn)[:, rng.permutation(X.shape[1])]) for _ in range(3)]
        out[nm] = dict(cos_kierunku_med=f3(np.median((xn * yn).sum(0))),
                       wspolny_obrot_udzial=f3(wspolny_obrot(X, Y)), wspolny_obrot_zero=f3(np.mean(zero)),
                       wspolny_obrot_kierunki=f3(wspolny_obrot(xn, yn)), wspolny_obrot_kierunki_zero=f3(np.mean(zero_k)),
                       wzrost_dlugosci=f3(np.median(np.linalg.norm(Y, axis=0)) / np.median(np.linalg.norm(X, axis=0))))
    return out


rng = np.random.default_rng(0)
PARY = [(0, 1400), (1400, 5300), (5300, 11000), (11000, 22974), (0, 22974), (22974, 43760), (43760, 65640), (0, 65640), (1094, 65640)]
OKNA = [(najblizszy(5470 * i), najblizszy(5470 * (i + 1))) for i in range(12)]
O = {}
for a, b in PARY + OKNA:
    if f'{a}->{b}' in O:
        continue
    O[f'{a}->{b}'] = obrot(a, b, rng)
    log('O', a, b, O[f'{a}->{b}'])
wynik['O_obrot'] = O
wynik['O_okna_5_epok'] = [f'{a}->{b}' for a, b in OKNA]

# ---------------- J. dokładny Jakobian ----------------
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float() / 255.
net = M.MalyViT().eval(); net.zapisuj_latent(True)
zap = {}
net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0].detach()))
net.bloki[5].mlp.register_forward_hook(lambda m, i, o: zap.__setitem__('f', o.detach()))
n = 192
Jw = {}
for k in KROKI_EP:
    w = wagi(k); i = idx(k)
    net.load_state_dict(wczytaj(k))
    with torch.no_grad():
        _, lat = net(xin)
    x = zap['x'][0].double().numpy(); Fh = zap['f'][0].double().numpy()
    u = x - x.mean(1, keepdims=True); sd = np.sqrt((u * u).mean(1) + 1e-5)
    a = (w['gam'] * u / sd[:, None] + w['bet']) @ w['W1'].T + w['b1']                # (600, 768)
    zr = np.asarray(ZL[i, 0]).astype(np.float64)
    zgod = 1 - ((gelu(a) - zr) ** 2).sum() / ((zr - zr.mean()) ** 2).sum()
    Fn = gelu(a) @ w['W2'].T + w['b2']
    blad_F = float(np.linalg.norm(Fn - Fh) / np.linalg.norm(Fh))
    rec = {nm: [] for nm in ('ujemne', 'slad', 'sym', 'rot', 'lmin', 'lmax', 'maxre', 'norma', 'rozszerz', 'promien')}
    for j in range(600):
        Jln = w['gam'][:, None] * ((np.eye(n) - 1.0 / n) / sd[j] - np.outer(u[j], u[j]) / (n * sd[j] ** 3))
        J = (w['W2'] * dgelu(a[j])[None, :]) @ w['W1'] @ Jln
        S, A_ = (J + J.T) / 2, (J - J.T) / 2
        ev = np.linalg.eigvalsh(S); lam = np.linalg.eigvals(J)
        rec['ujemne'].append((ev < 0).mean()); rec['slad'].append(np.trace(J) / n)
        rec['sym'].append(np.linalg.norm(S) ** 2 / np.linalg.norm(J) ** 2); rec['rot'].append(np.linalg.norm(A_) / np.linalg.norm(S))
        rec['lmin'].append(ev[0]); rec['lmax'].append(ev[-1]); rec['maxre'].append(lam.real.max())
        rec['norma'].append(np.linalg.norm(J, 2))
        rec['rozszerz'].append((np.abs(1 + lam) > 1).mean()); rec['promien'].append(np.abs(1 + lam).max())
    rec = {nm: np.array(v) for nm, v in rec.items()}
    Jw[str(k)] = dict(epoka=round(k / EP, 2), zgodnosc_GELU_a_z_latentem_R2=f3(zgod) if zgod < 0.99995 else round(float(zgod), 6),
                      blad_wzgledny_F_numpy_vs_model=float(f'{blad_F:.2e}'),
                      wszystkie={nm: q(v) for nm, v in rec.items()},
                      narzedzie={nm: q(v[narz]) for nm, v in rec.items()},
                      tlo={nm: q(v[~narz]) for nm, v in rec.items()},
                      token_316={nm: f3(v[FOK]) for nm, v in rec.items()})
    log('J', k, 'R2', round(zgod, 6), 'blad F', f'{blad_F:.1e}', {nm: Jw[str(k)]['wszystkie'][nm]['med'] for nm in rec})
wynik['J_jakobian'] = Jw
wynik['J_zero_macierz_losowa'] = dict(sym=f3((n + 1) / (2 * n)), rot=f3(np.sqrt((n - 1) / (n + 1))))

# ---------------- W. czy dane z Edenu (TEST_SFERY / WIAZKI) to bieg s47 ----------------
Wz = {}
if (WK / 'W_fc1.npy').exists():
    Wa = np.load(WK / 'W_fc1.npy', mmap_mode='r')
    kroki_w = json.load(open(WK / 'META_WIAZKI.json', encoding='utf-8'))['kroki']
    for t in [0, 101, 300, len(kroki_w) - 1]:
        k = int(kroki_w[t]); we = np.asarray(Wa[t]).astype(np.float64); ws = wagi(k)['W1']
        Wz[str(k)] = dict(indeks=t, roznica_wzgledna_s47=float(f'{np.abs(we - ws).max() / np.abs(ws).max():.2e}'))
        if (WAGI_G / f'k{k:05d}.pt').exists():
            Wz[str(k)]['roznica_wzgledna_gesty'] = float(f'{np.abs(we - wagi(k, WAGI_G)["W1"]).max() / np.abs(ws).max():.2e}')
        log('W', k, Wz[str(k)])
    k1, k60 = int(kroki_w[101]), int(kroki_w[-1])
    w1, w60 = wagi(k1), wagi(k60)
    cw = lambda A, B: f3(np.median((A / np.linalg.norm(A, axis=1, keepdims=True) * B / np.linalg.norm(B, axis=1, keepdims=True)).sum(1)))
    Wz['cos_w_epoka1_epoka60_med'] = cw(w1['W1'], w60['W1'])
    Wz['cos_v_epoka1_epoka60_med'] = cw(w1['W2'].T, w60['W2'].T)
    Wz['kroki_porownania'] = [k1, k60]
    log('W cos(w) ep1→ep60', Wz['cos_w_epoka1_epoka60_med'], 'cos(v)', Wz['cos_v_epoka1_epoka60_med'])
wynik['W_dane_edenu'] = Wz

# ---------------- F. geometria filmu 12 ----------------
sys.path.insert(0, str(ROOT / 'app'))
from wspolne import kierunki                                                        # noqa: E402

FIB = kierunki(False).astype(np.float64)
R_, RR_ = 34.0, 50.0


def geometria(katalog, k_koniec, kroki_kat):
    V_end = wczytaj(k_koniec, katalog)['bloki.5.mlp.fc2.weight'].double().numpy()
    vn = V_end / np.linalg.norm(V_end, axis=0)
    U_, _, _ = np.linalg.svd(vn - vn.mean(1, keepdims=True), full_matrices=False)
    P3 = U_[:, :3].T
    s2 = np.linalg.svd(vn - vn.mean(1, keepdims=True), compute_uv=False) ** 2

    def kier(k):
        PV = (P3 @ wczytaj(k, katalog)['bloki.5.mlp.fc2.weight'].double().numpy()).T            # (768, 3)
        return PV, PV / np.linalg.norm(PV, axis=1, keepdims=True)

    _, d_end = kier(k_koniec)
    r_, c_ = linear_sum_assignment(FIB @ d_end.T)
    slot = np.empty(768, int); slot[c_] = r_
    rfix = RR_ * FIB[slot]
    out = dict(krok_przypisania=k_koniec, sredni_cos_miejsca=f3(-(FIB[slot] * d_end).sum(1).mean()),
               udzial_3_skladowych_wycentrowanych=f3(s2[:3].sum() / s2.sum()), kat_zalamania={})
    for k in kroki_kat:
        PV, Dc = kier(k)
        Z = np.clip(np.asarray(ZL[idx(k), 0]).astype(np.float64), 0, None)
        pos = (Z - Z.mean(0)) @ PV
        pos = pos / np.percentile(np.linalg.norm(pos, axis=1), 99) * 0.55 * R_
        b = pos @ Dc.T
        s = b + np.sqrt(np.maximum(b * b - (pos * pos).sum(1)[:, None] + R_ * R_, 0))
        E = pos[:, None, :] - s[..., None] * Dc[None]
        uu = E - rfix[None]; uu /= np.linalg.norm(uu, axis=2, keepdims=True)
        kat = np.degrees(np.arccos(np.clip((uu * Dc[None]).sum(2), -1, 1)))
        out['kat_zalamania'][str(k)] = f3((kat * Z).sum() / Z.sum())
    return out


wynik['F_film12_gesty_kontrola'] = geometria(WAGI_G, 22974, [0, 22974])
log('F gęsty', wynik['F_film12_gesty_kontrola'])
wynik['F_film12_s47'] = geometria(WAGI, 65640, [0, 1094, 22974, 65640])
log('F s47', wynik['F_film12_s47'])

(HERE / 'PRZELICZENIE_ZASADY_S47.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
log('zapisano PRZELICZENIE_ZASADY_S47.json')
