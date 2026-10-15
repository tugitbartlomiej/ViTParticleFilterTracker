"""Czy „pułapka nadąża za koloidami”? Pomiar na biegu s47 (MalyViT_35000_od_zera_s47, 806 checkpointów, 60 epok), blok 6 (indeks 5).

Pytanie 1 (opóźnienie). Zmiana wyjścia MLP bloku 6 między kolejnymi checkpointami t → t+1 rozkłada się dokładnie na
  Δ_in(t) = F(x(t+1); θ6(t)) − F(x(t); θ6(t))       ruch koloidu: wcześniejsze warstwy przesuwają wejście x̃ bloku (przed LN2),
  Δ_W(t)  = F(x(t+1); θ6(t+1)) − F(x(t+1); θ6(t))   zmiana pułapki: fc1, fc2, γ i β LN2 bloku 6,
  oraz symetrycznie (średnia obu kolejności, Shapley). Korelacje z opóźnieniem C(τ) = ⟨cos(Δ_in(t), Δ_W(t+τ))⟩ po tokenach i t,
  regresje z opóźnieniem w obie strony (kto za kim idzie, z kontrolą własnej przeszłości), autokorelacje, prędkości |Δ| po odjęciu
  trendu. Zera: przesunięcie cykliczne w czasie, tasowanie w czasie, tasowanie tokenów. Fazy o stałym odstępie checkpointów:
  kroki 0–1000 co 10, potem co 100 (epoki 1–20, 20–40, 40–60). To samo dla latentu z (768 wymiarów, zdjęcie TRAIN 0).
Pytanie 2 (wspólny obrót). W oknach 1- i 5-epokowych obrót Prokrustesa R_x stanów strumienia x̂ (LN bez γ, β, po odjęciu średniej
  po tokenach) 32 zdjęć TRAIN; R_x przyłożony do czujników s_c = P₁(γ ⊙ w_c) (oraz w_c i v_c), wobec najlepszego obrotu samych
  czujników R_s, zera z tasowaniem zmian i zera z losowo obróconymi osiami R_x; generatory log R i ich cosinusy z opóźnieniem.
  Wszystko w podprzestrzeni, w której strumień ma 95 % wariancji (baza wspólna dla fazy; 99 % jako kontrola): w pełnej 192D obrót
  strumienia jest nieokreślony w kierunkach bez wariancji (próba pokazała udziały rzędu −35, jak dla losowych osi), a składowe
  czujników poza tą podprzestrzenią dają tylko stałą, którą wchłania próg b_c. Wartości 192D zostają w JSON jako wtórne.

Kryteria werdyktu (ustalone przed pomiarem):
  P1 potwierdzone: w fazach po 1. epoce reakcja pułapki na ruch koloidu z opóźnieniem τ ≥ 1 (C(τ) albo współczynnik a(τ) regresji
     Δ_W(t+τ) ~ a Δ_in(t) + b Δ_W(t)) wychodzi poza zero o ponad 3 odchylenia, a odwrotny kierunek (c(τ) w Δ_in(t+τ) ~ c Δ_W(t) + e Δ_in(t))
     nie; obalone: dla τ ≥ 1 oba kierunki mieszczą się w zerze; niejednoznaczne: pozostałe przypadki (np. oba kierunki albo tylko τ = 0).
  P2 potwierdzone: późno (epoki 40–60) R_x wyjaśnia dodatni udział zmiany kierunków czujników równy co najmniej połowie nadwyżki R_s
     nad zerem i wyraźnie ponad zerem z losowymi osiami, a cos(log R_x, log R_s) jest wyraźnie ponad cosinusem dla niepasujących okien;
     obalone: udział R_x w zerze z losowymi osiami i cos generatorów w zerze; niejednoznaczne: pozostałe przypadki.
Uruchom (z katalogu film_krysztal): F:/Python/3.13/python.exe -X utf8 pomiar_nadazania_pulapki.py [test]  → NADAZANIE_PULAPKI_S47.json
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.linalg import logm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST = 'test' in sys.argv[1:]
t0 = time.time()
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a) + f' [{time.time() - t0:.0f} s]'
    print(s, flush=True)
    LOG.append(s)


WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
POMIAR = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
MANIFEST = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
ZIPY = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
EP = 1094
N1, N2, K = 4, 32, 10                                                               # zdjęcia TRAIN do P1 i P2, największe opóźnienie
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.allow_tf32 = False                                             # TF32 w splotach łodygi dodałby szum ~1e-3
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)
f4 = lambda x: float(np.round(float(x), 5))
rng = np.random.default_rng(0)

META = json.load(open(POMIAR / 'META.json', encoding='utf-8'))
KROKI_LAT = np.array(META['kroki'])
WIERSZE = [json.loads(l) for l in open(MANIFEST, encoding='utf-8') if l.strip()]


def obraz(i):
    """Zdjęcie TRAIN i z pomiaru ruch60 (te same klatki co w pelne768), wczytane jak w treningu: BGR→RGB, INTER_AREA do 480×320."""
    z = META['zdjecia'][i]
    assert z['split'] == 'TRAIN', z
    r = WIERSZE[z['pool_row']]
    assert f"cataract_lmm/tracking/{r['film']}/{int(r['frame']):08d}" == z['frame_id'], (r['film'], r['frame'], z['frame_id'])
    with zipfile.ZipFile(ZIPY / Path(r['archive'].replace(chr(92), '/')).name) as zf:
        raw = zf.read(r['member'])
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256'], z['frame_id']
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def tensor_obrazow(idx):
    return torch.from_numpy(np.stack([obraz(i) for i in idx])).permute(0, 3, 1, 2).float().div(255.).to(DEV)


sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

net = M.MalyViT().to(DEV).eval()
net.zapisuj_latent(True)
EPS_LN = float(net.bloki[5].norm2.eps)
zap = {}
net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0]))
net.bloki[5].mlp.register_forward_hook(lambda m, i, o: zap.__setitem__('f', o))


def wczytaj(k):
    try:
        return torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=True)['net']
    except Exception:
        return torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']


def wagi_kolejno(kroki, ile=6):
    """Checkpointy po kolei, wczytywane z wyprzedzeniem w wątkach (najwyżej ile + 1 naraz w pamięci)."""
    with ThreadPoolExecutor(4) as ex:
        q = deque()
        for k in kroki:
            q.append((k, ex.submit(wczytaj, k)))
            if len(q) > ile:
                k0, f0 = q.popleft()
                yield k0, f0.result()
        while q:
            k0, f0 = q.popleft()
            yield k0, f0.result()


def przejscie(sd, X):
    """Stan x̃ przed LN2 bloku 6, wyjście MLP i latent z dla zdjęć X (float32, bez TF32)."""
    net.load_state_dict(sd)
    xs, fs, zs = [], [], []
    for b in range(0, len(X), 16):
        _, lat = net(X[b:b + 16])
        xs.append(zap['x'].double()); fs.append(zap['f'].double()); zs.append(lat.double())
    return torch.cat(xs), torch.cat(fs), torch.cat(zs)


def theta(sd):
    g = lambda nm: sd[f'bloki.5.{nm}'].to(DEV, torch.float64)
    return dict(W1=g('mlp.fc1.weight'), b1=g('mlp.fc1.bias'), W2=g('mlp.fc2.weight'), b2=g('mlp.fc2.bias'),
                gam=g('norm2.weight'), bet=g('norm2.bias'))


def xhat(x):
    u = x - x.mean(-1, keepdim=True)
    return u / torch.sqrt((u * u).mean(-1, keepdim=True) + EPS_LN)


def mlp(xh, th, z=False):
    a = (th['gam'] * xh + th['bet']) @ th['W1'].T + th['b1']
    zz = torch.nn.functional.gelu(a)
    return zz if z else zz @ th['W2'].T + th['b2']


wynik = dict(opis=(__doc__ or '').split('\n\n')[0], bieg='MalyViT_35000_od_zera_s47', blok=6, test=TEST,
             kryteria=(__doc__ or '').split('Kryteria werdyktu (ustalone przed pomiarem):')[1].split('Uruchom')[0].strip(),
             ustawienia=dict(zdjecia_P1=f'TRAIN 0–{N1 - 1} ({N1 * 600} tokenów)', zdjecia_P2=f'TRAIN 0–{N2 - 1} ({N2 * 600} tokenów)',
                             K=K, eps_LN=EPS_LN, precyzja='przejście float32 bez TF32, rozkład MLP float64', urzadzenie=str(DEV)))

# ---------------- kontrola: wczytanie zdjęć i zgodność z zapisanym latentem ----------------
ZL = np.load(POMIAR / 'pelne768.npy', mmap_mode='r')
X2 = tensor_obrazow([0, 1])
kon = {}
for k in [0, 1000, 22974, 65640]:
    sd = wczytaj(k); x, f, zl = przejscie(sd, X2); th = theta(sd)
    i = int(np.searchsorted(KROKI_LAT, k)); assert KROKI_LAT[i] == k
    r2 = {}
    for j in (0, 1):
        zr = np.asarray(ZL[i, j]).astype(np.float64); zm = zl[j].cpu().numpy()
        r2[f'TRAIN {j}'] = round(float(1 - ((zm - zr) ** 2).sum() / ((zr - zr.mean()) ** 2).sum()), 6)
    fw = mlp(xhat(x), th)
    kon[str(k)] = dict(R2_latentu_do_zapisu_bf16=r2, blad_F_wzor_vs_model=float(f'{float((fw - f).norm() / f.norm()):.2e}'),
                       blad_z_wzor_vs_model=float(f'{float((mlp(xhat(x), th, z=True) - zl).norm() / zl.norm()):.2e}'))
    log('kontrola', k, kon[str(k)])
wynik['kontrola'] = kon
del X2

# ---------------- P1: rozkład Δ_in / Δ_W na siatce o stałym odstępie ----------------
SIATKA = sorted(set(range(0, 1001, 10)) | set(range(1000, 65601, 100)))
if TEST:
    SIATKA = [k for k in SIATKA if k <= 300] + [k for k in SIATKA if 1000 <= k <= 6000]
FAZY = [('start: kroki 0–1000 co 10', 0, 1000), ('epoki 1–20: kroki 1000–21 900 co 100', 1000, 21900),
        ('epoki 20–40: kroki 21 900–43 700 co 100', 21900, 43700), ('epoki 40–60: kroki 43 700–65 600 co 100', 43700, 65600)]
X1 = tensor_obrazow(list(range(N1)))
D = {nm: [] for nm in ('in', 'W', 'inS', 'WS', 'in_z', 'W_z')}
przedzialy, bledy_sumy, bledy_F = [], [], []
prev = None
for n, (k, sd) in enumerate(wagi_kolejno(SIATKA)):
    x, f, zl = przejscie(sd, X1); th = theta(sd); xh = xhat(x)
    fcc = mlp(xh, th)
    if n % 50 == 0:
        bledy_F.append(float((fcc - f).norm() / f.norm()))
    if prev is not None:
        xh_p, th_p, fpp = prev
        fcp, fpc = mlp(xh, th_p), mlp(xh_p, th)
        din, dw = fcp - fpp, fcc - fcp
        bledy_sumy.append(float((din + dw - (fcc - fpp)).norm() / (fcc - fpp).norm()))
        D['in'].append(din.reshape(-1, 192).float().half().cpu()); D['W'].append(dw.reshape(-1, 192).float().half().cpu())
        D['inS'].append((0.5 * ((fcp - fpp) + (fcc - fpc))).reshape(-1, 192).float().half().cpu())
        D['WS'].append((0.5 * ((fpc - fpp) + (fcc - fcp))).reshape(-1, 192).float().half().cpu())
        zpp, zcp, zcc = mlp(xh_p[:1], th_p, z=True), mlp(xh[:1], th_p, z=True), mlp(xh[:1], th, z=True)
        D['in_z'].append((zcp - zpp)[0].float().half().cpu()); D['W_z'].append((zcc - zcp)[0].float().half().cpu())
        przedzialy.append((SIATKA[n - 1], k))
    prev = (xh, th, fcc)
    if n % 100 == 0:
        log('P1 checkpoint', n, '/', len(SIATKA), 'krok', k)
del X1, prev
przedzialy = np.array(przedzialy)
wynik['P1_kontrola'] = dict(przedzialow=int(len(przedzialy)), blad_sumy_max=float(f'{max(bledy_sumy):.2e}'),
                            blad_F_wzor_vs_model_max=float(f'{max(bledy_F):.2e}'))
log('P1 kontrola', wynik['P1_kontrola'])


def okno(A, B, K, tokeny=600, n_przes=100, n_perm=20):
    """Korelacje z opóźnieniem między szeregami A[t], B[t] (n, N, d) na GPU; zera: przesunięcie, tasowanie czasu i tokenów."""
    A, B = A.to(DEV).float(), B.to(DEV).float()
    n, N, d = A.shape
    ua = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ub = B / B.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def C(u, v, tau):
        return float((u[:n - tau] * v[tau:]).sum(-1).mean()) if tau >= 0 else float((u[-tau:] * v[:n + tau]).sum(-1).mean())

    def S(u, v, tau):
        return float((u[:n - tau] * v[tau:]).sum()) if tau >= 0 else float((u[-tau:] * v[:n + tau]).sum())

    def SA(tau):                                                                    # Σ|A[t]|² po tych t, które wchodzą do S(A, B, τ)
        return float((A[:n - tau] ** 2).sum()) if tau >= 0 else float((A[-tau:] ** 2).sum())

    taus = list(range(-K, K + 1))
    out = dict(n=n, C={t: f4(C(ua, ub, t)) for t in taus},
               A_in={t: f4(C(ua, ua, t)) for t in range(1, K + 1)}, A_W={t: f4(C(ub, ub, t)) for t in range(1, K + 1)},
               energia_W=f4(float((B * B).sum()) / float((A * A).sum() + (B * B).sum())),
               beta={t: f4(S(A, B, t) / max(SA(t), 1e-30)) for t in taus})

    def regresja(P, Q, tau, przes=0):
        """Q[t+τ] ≈ a P[t] + b Q[t] (P opcjonalnie przesunięte cyklicznie o przes: zero)."""
        Pp = torch.roll(P, przes, 0) if przes else P
        p, q, qn = Pp[:n - tau], Q[:n - tau], Q[tau:]
        G = np.array([[float((p * p).sum()), float((p * q).sum())], [float((q * p).sum()), float((q * q).sum())]])
        r = np.array([float((p * qn).sum()), float((q * qn).sum())])
        return np.linalg.solve(G, r)

    out['regresja_pulapka_za_koloidem_a'] = {t: f4(regresja(A, B, t)[0]) for t in range(1, K + 1)}
    out['regresja_pulapka_wlasna_b'] = {t: f4(regresja(A, B, t)[1]) for t in range(1, K + 1)}
    out['regresja_koloid_za_pulapka_c'] = {t: f4(regresja(B, A, t)[0]) for t in range(1, K + 1)}
    out['regresja_koloid_wlasny_e'] = {t: f4(regresja(B, A, t)[1]) for t in range(1, K + 1)}
    przes = rng.choice(np.arange(K + 1, n - K), size=min(n_przes, n - 2 * K - 1), replace=False)
    c0 = np.array([float((ua * torch.roll(ub, -int(s), 0)).sum(-1).mean()) for s in przes])
    za = np.array([regresja(A, B, 1, int(s))[0] for s in przes[:40]])
    zc = np.array([regresja(B, A, 1, int(s))[0] for s in przes[:40]])
    cp = np.array([float((ua * ub[torch.from_numpy(rng.permutation(n)).to(DEV)]).sum(-1).mean()) for _ in range(n_perm)])
    perm_tok = torch.from_numpy(np.concatenate([rng.permutation(tokeny) + tokeny * i for i in range(N // tokeny)])).to(DEV)
    ubt = ub[:, perm_tok]
    out['zero_C_przesuniecie'] = dict(sr=f4(c0.mean()), sd=f4(c0.std()), p2_5=f4(np.percentile(c0, 2.5)), p97_5=f4(np.percentile(c0, 97.5)))
    out['zero_C_tasowanie_czasu'] = dict(sr=f4(cp.mean()), sd=f4(cp.std()))
    out['C_tasowanie_tokenow'] = {t: f4(C(ua, ubt, t)) for t in taus}
    out['zero_a1_przesuniecie'] = dict(sr=f4(za.mean()), sd=f4(za.std()))
    out['zero_c1_przesuniecie'] = dict(sr=f4(zc.mean()), sd=f4(zc.std()))
    # prędkości: log średniej |Δ| po tokenach, odjęty trend (średnia krocząca 21 przedziałów)
    vi = np.log(A.norm(dim=-1).mean(-1).cpu().numpy()); vw = np.log(B.norm(dim=-1).mean(-1).cpu().numpy())
    tr = lambda v: v - np.convolve(np.pad(v, 10, mode='reflect'), np.ones(21) / 21, mode='valid')
    ri, rw = tr(vi), tr(vw)
    kor = lambda a_, b_: float(np.corrcoef(a_, b_)[0, 1])
    out['predkosc_r'] = {t: f4(kor(ri[:n - t], rw[t:]) if t >= 0 else kor(ri[-t:], rw[:n + t])) for t in taus}
    zr_ = np.array([kor(ri, np.roll(rw, -int(s))) for s in przes])
    out['zero_predkosc_przesuniecie'] = dict(sr=f4(zr_.mean()), sd=f4(zr_.std()))
    del A, B, ua, ub, ubt
    torch.cuda.empty_cache()
    return out


P1 = {}
for nazwa, lo, hi in FAZY:
    m = (przedzialy[:, 0] >= lo) & (przedzialy[:, 1] <= hi)
    if m.sum() == 0:
        continue
    dl = przedzialy[:, 1] - przedzialy[:, 0]
    krok = int(np.bincount(dl[m]).argmax())                                         # tylko przedziały o stałej długości
    m &= dl == krok
    if m.sum() < 3 * K:
        continue
    idx = np.flatnonzero(m)
    st = lambda nm: torch.stack([D[nm][i] for i in idx])
    w = dict(przedzialow=int(m.sum()), krok_przedzialu=krok, opoznienia_w_krokach={t: t * krok for t in range(-K, K + 1)})
    w['F'] = okno(st('in'), st('W'), K, tokeny=600)
    w['F_Shapley'] = {kk: v for kk, v in okno(st('inS'), st('WS'), K, tokeny=600).items() if kk in ('C', 'energia_W', 'zero_C_przesuniecie',
                                                                                                   'regresja_pulapka_za_koloidem_a', 'regresja_koloid_za_pulapka_c')}
    w['z_TRAIN0'] = {kk: v for kk, v in okno(st('in_z'), st('W_z'), K, tokeny=600).items() if kk in ('C', 'energia_W', 'zero_C_przesuniecie',
                                                                                                   'C_tasowanie_tokenow', 'A_in', 'A_W')}
    P1[nazwa] = w
    F_ = w['F']
    log('P1', nazwa, 'n', w['przedzialow'], 'energia_W', F_['energia_W'], 'C(0)', F_['C'][0], 'C(1)', F_['C'][1], 'C(-1)', F_['C'][-1],
        'zero', F_['zero_C_przesuniecie'], 'a(1)', F_['regresja_pulapka_za_koloidem_a'][1], 'c(1)', F_['regresja_koloid_za_pulapka_c'][1],
        'zero a', F_['zero_a1_przesuniecie'], 'zero c', F_['zero_c1_przesuniecie'])
wynik['P1'] = P1
del D
gc.collect()


# ---------------- P2: wspólny obrót strumienia a obrót czujników ----------------
def obrot(P0, P1_, rho=0.0):
    """Właściwy obrót R (det +1) minimalizujący ‖P1 − R P0‖ (punkty w kolumnach); rho > 0: przyciąganie do I w kierunkach bez danych."""
    Mx = P1_ @ P0.T
    if rho:
        Mx = Mx + rho * np.eye(len(Mx))
    U, _, Vt = np.linalg.svd(Mx)
    dd = np.ones(len(Mx)); dd[-1] = np.sign(np.linalg.det(U @ Vt))
    return (U * dd) @ Vt


def udzial(P0, P1_, R):
    return float(1 - np.linalg.norm(P1_ - R @ P0) ** 2 / max(np.linalg.norm(P1_ - P0) ** 2, 1e-30))


def zero_tasowanie(P0, P1_, n=3):
    """Zero: ta sama macierz zmian przetasowana między punktami (jedna permutacja na losowanie)."""
    Dd, wy_ = P1_ - P0, []
    for _ in range(n):
        Pp = P0 + Dd[:, rng.permutation(P0.shape[1])]
        wy_.append(udzial(P0, Pp, obrot(P0, Pp)))
    return float(np.mean(wy_))


def generator(R):
    G = logm(R)
    G = np.real(G) if np.iscomplexobj(G) else G
    return (G - G.T) / 2


def losowy_ortogonalny(d):
    Q, Rr = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(Rr))


cosF = lambda A, B: float((A * B).sum() / max(np.linalg.norm(A) * np.linalg.norm(B), 1e-30))
d = 192
P1m = np.eye(d) - 1.0 / d                                                           # rzut na hiperpłaszczyznę ⊥ 1 (LN widzi tylko ją)
kol = lambda Y: Y / np.linalg.norm(Y, axis=0, keepdims=True)
EKR = [0] + [EP * e for e in range(1, 61)]
if TEST:
    EKR = EKR[:7]
X2 = tensor_obrazow(list(range(N2)))
stany, czujniki = {}, {}
for e, (k, sd) in enumerate(wagi_kolejno(EKR)):
    x, f, _ = przejscie(sd, X2); th = theta(sd)
    xh = xhat(x).reshape(-1, d).cpu().numpy().astype(np.float32)
    xp = xhat(x + f).reshape(-1, d).cpu().numpy().astype(np.float32)
    xr = x.reshape(-1, d).cpu().numpy().astype(np.float32)
    stany[e] = dict(xhat_srodek=(xh - xh.mean(0)).T.copy(), xhat=xh.T.copy(), x_srodek=(xr - xr.mean(0)).T.copy(),
                    xplus_srodek=(xp - xp.mean(0)).T.copy())
    W1 = th['W1'].cpu().numpy(); gam = th['gam'].cpu().numpy(); W2 = th['W2'].cpu().numpy()
    czujniki[e] = dict(s=kol(P1m @ (W1 * gam[None, :]).T), w=kol(P1m @ W1.T), v=kol(P1m @ W2), w_surowe=W1.T.copy())
    if e % 10 == 0:
        log('P2 checkpoint epoka', e)
del X2
gc.collect()


def pelna_przestrzen(e0, e1):
    """Wtórne: obrót w pełnej przestrzeni 192D. W kierunkach, w których strumień prawie nie ma wariancji, R_x jest nieokreślony
    (dowolny), więc przyłożony do czujników daje ogromne ujemne udziały, jak obrót o losowych osiach. Dlatego wniosek opiera się na
    podprzestrzeni strumienia (okno_w_bazie)."""
    S0, S1 = stany[e0], stany[e1]
    Rx = obrot(S0['xhat_srodek'], S1['xhat_srodek'])
    out = dict(strumien={wr: f4(udzial(S0[wr], S1[wr], obrot(S0[wr], S1[wr]))) for wr in ('xhat_srodek', 'xhat', 'x_srodek', 'xplus_srodek')},
               strumien_zero=f4(zero_tasowanie(S0['xhat_srodek'], S1['xhat_srodek'])),
               s_obrot_strumienia_192D=f4(udzial(czujniki[e0]['s'], czujniki[e1]['s'], Rx)),
               s_wlasny_obrot_192D=f4(udzial(czujniki[e0]['s'], czujniki[e1]['s'], obrot(czujniki[e0]['s'], czujniki[e1]['s']))))
    w0n, w1n = kol(czujniki[e0]['w_surowe']), kol(czujniki[e1]['w_surowe'])         # definicja z PRZELICZENIE (część O): bez korekty det
    U_, _, Vt_ = np.linalg.svd(w1n @ w0n.T)
    out['w_jak_PRZELICZENIE'] = f4(1 - np.linalg.norm(w1n - U_ @ Vt_ @ w0n) ** 2 / np.linalg.norm(w1n - w0n) ** 2)
    return out


def baza(epoki, frac):
    """Wspólna dla fazy baza głównych kierunków strumienia x̂ (po odjęciu średniej po tokenach), k = ile niesie frac wariancji."""
    C = sum(stany[e]['xhat_srodek'].astype(np.float64) @ stany[e]['xhat_srodek'].T.astype(np.float64) for e in epoki)
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1], U[:, ::-1]
    k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)
    return U[:, :k], k


def okno_w_bazie(e0, e1, Uk):
    k = Uk.shape[1]
    Q0 = Uk.T @ stany[e0]['xhat_srodek'].astype(np.float64)
    Q1 = Uk.T @ stany[e1]['xhat_srodek'].astype(np.float64)
    Rx = obrot(Q0, Q1)
    gens = dict(x=generator(Rx))
    out = dict(strumien=f4(udzial(Q0, Q1, Rx)), strumien_zero=f4(zero_tasowanie(Q0, Q1)),
               kat_strumienia_rad=f4(np.linalg.norm(gens['x']) / np.sqrt(2)))
    for fam in ('s', 'w', 'v'):
        Y0, Y1 = Uk.T @ czujniki[e0][fam], Uk.T @ czujniki[e1][fam]                # składowe czujników w podprzestrzeni strumienia
        Ry = obrot(Y0, Y1)
        gens[fam] = generator(Ry)
        haar = [udzial(Y0, Y1, Q @ Rx @ Q.T) for Q in (losowy_ortogonalny(k) for _ in range(20))]
        out[fam] = dict(wlasny_obrot=f4(udzial(Y0, Y1, Ry)), zero=f4(zero_tasowanie(Y0, Y1)), obrot_strumienia=f4(udzial(Y0, Y1, Rx)),
                        zero_losowe_osie_sr=f4(np.mean(haar)), zero_losowe_osie_sd=f4(np.std(haar)),
                        cos_generatorow_x=f4(cosF(gens['x'], gens[fam])), kat_rad=f4(np.linalg.norm(gens[fam]) / np.sqrt(2)),
                        czesc_w_podprzestrzeni=f4(np.mean((Y0 ** 2).sum(0))))
    return out, gens, Rx


sr_ = lambda v: f4(np.mean(v)) if len(v) else None
sd_ = lambda v: f4(np.std(v)) if len(v) else None
P2 = {}
for nazwa, krok_e in (('okna_1_epoka', 1), ('okna_5_epok', 5)):
    okna = [(e0, e0 + krok_e) for e0 in range(0, len(EKR) - krok_e, krok_e)]
    P2[nazwa] = dict(pelna_przestrzen_192D=[dict(epoki=[e0, e1], **pelna_przestrzen(e0, e1)) for e0, e1 in okna], fazy={})
    for lo, hi in ((0, 20), (20, 40), (40, 60)):
        ok = [(e0, e1) for e0, e1 in okna if e0 >= lo and e1 <= hi]
        if not ok:
            continue
        for frac in (0.95, 0.99):
            Uk, k = baza(range(lo, min(hi, len(EKR) - 1) + 1), frac)
            lista, G, RX = [], [], []
            for e0, e1 in ok:
                o, g, rx = okno_w_bazie(e0, e1, Uk)
                o['epoki'] = [e0, e1]
                lista.append(o); G.append(g); RX.append(rx)
            n = len(G)
            minlag = 3 if krok_e == 1 else 2
            pary = [(i, j) for i in range(n) for j in range(n) if abs(i - j) >= minlag]
            pary_los = [pary[q] for q in rng.permutation(len(pary))[:400]]
            gen = {}
            for fam in ('s', 'w', 'v'):
                rowne = [cosF(G[i]['x'], G[i][fam]) for i in range(n)]
                x_przed = [cosF(G[i - 1]['x'], G[i][fam]) for i in range(1, n)]
                x_po = [cosF(G[i]['x'], G[i - 1][fam]) for i in range(1, n)]
                zero = [cosF(G[i]['x'], G[j][fam]) for i, j in pary]
                mis = [udzial(Uk.T @ czujniki[ok[i][0]][fam], Uk.T @ czujniki[ok[i][1]][fam], RX[j]) for i, j in pary_los]
                gen[fam] = dict(cos_to_samo_okno=dict(sr=sr_(rowne), sd=sd_(rowne)),
                                cos_strumien_okno_wczesniej=dict(sr=sr_(x_przed), sd=sd_(x_przed)),
                                cos_strumien_okno_pozniej=dict(sr=sr_(x_po), sd=sd_(x_po)),
                                zero_niepasujace_okna=dict(sr=sr_(zero), sd=sd_(zero), par=len(zero)),
                                zero_udzial_R_x_z_innego_okna=dict(sr=sr_(mis), sd=sd_(mis)))
            pods = dict(okien=n, k=k, strumien=sr_([o['strumien'] for o in lista]), strumien_zero=sr_([o['strumien_zero'] for o in lista]),
                        kat_strumienia_rad=sr_([o['kat_strumienia_rad'] for o in lista]))
            for fam in ('s', 'w', 'v'):
                pods[fam] = {kk: sr_([o[fam][kk] for o in lista]) for kk in ('wlasny_obrot', 'zero', 'obrot_strumienia', 'zero_losowe_osie_sr',
                                                                             'zero_losowe_osie_sd', 'cos_generatorow_x', 'kat_rad',
                                                                             'czesc_w_podprzestrzeni')}
            klucz = f'epoki {lo}–{hi}, baza {int(round(frac * 100))} %'
            P2[nazwa]['fazy'][klucz] = dict(podsumowanie=pods, generatory=gen, okna=lista)
            log('P2', nazwa, klucz, 'k', k, 'strumień', pods['strumien'], 'zero', pods['strumien_zero'],
                '| s: własny', pods['s']['wlasny_obrot'], 'zero', pods['s']['zero'], 'R_x', pods['s']['obrot_strumienia'],
                'osie', pods['s']['zero_losowe_osie_sr'], '±', pods['s']['zero_losowe_osie_sd'], 'cos', pods['s']['cos_generatorow_x'],
                'cos wcześniej', gen['s']['cos_strumien_okno_wczesniej']['sr'], 'zero cos', gen['s']['zero_niepasujace_okna'],
                '| w: R_x', pods['w']['obrot_strumienia'], 'własny', pods['w']['wlasny_obrot'],
                '| v: R_x', pods['v']['obrot_strumienia'], 'własny', pods['v']['wlasny_obrot'], 'zero', pods['v']['zero'])
    pp = P2[nazwa]['pelna_przestrzen_192D']
    for lo, hi in ((0, 20), (20, 40), (40, 60)):
        w_ = [o for o in pp if o['epoki'][0] >= lo and o['epoki'][1] <= hi]
        if w_:
            log('P2 192D', nazwa, lo, hi, 'strumień', sr_([o['strumien']['xhat_srodek'] for o in w_]), 'zero', sr_([o['strumien_zero'] for o in w_]),
                'w jak PRZELICZENIE', sr_([o['w_jak_PRZELICZENIE'] for o in w_]), 's: R_x 192D', sr_([o['s_obrot_strumienia_192D'] for o in w_]),
                'własny 192D', sr_([o['s_wlasny_obrot_192D'] for o in w_]))
wynik['P2'] = P2

wy = HERE / ('NADAZANIE_PULAPKI_S47_test.json' if TEST else 'NADAZANIE_PULAPKI_S47.json')
wynik['log'] = LOG
wy.write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
log('zapisano', wy.name)
