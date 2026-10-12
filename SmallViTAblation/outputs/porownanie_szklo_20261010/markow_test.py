"""Jaki to ruch w skali kroku: test Markowa, struktura operatora (piDMD) i jądro pamięci Moriego–Zwanziga dla tokenów z pełnego zapisu
768D biegu 35 000 (gesty/pelne768_*.npz, 8 sond × 600 tokenów, co 1 krok), okno 300–1094, w losowym rzucie 128D (ta sama macierz
R co fala 3 i eksport 60 epok: torch.randn(768,128) z ziarnem 0 / √128).
Współrzędne względne: x_i(t) = z_i(t) − średnia tokenów zdjęcia (bez modu zdjęcia); drgania wokół centrum: x̃_i = x_i − średnia po oknie.
  A. operator jednokrokowy x̃(t+1) ≈ A x̃(t), wspólny dla wszystkich tokenów (128×128): najmniejsze kwadraty i rozmaitości piDMD
     (symetryczny = czysta relaksacja, skośnie symetryczny/ortogonalny = obrót bez relaksacji, diagonalny = współrzędne niesprzężone);
     błąd jednego kroku względem trwania (x̃(t+1) = x̃(t)) i widmo A (ile wartości własnych zespolonych, czasy zaniku −1/ln|λ|);
  B. czasy implikowane VAMP (deeptime) w funkcji opóźnienia τ: płaskie = Markow w tej przestrzeni, rosnące = pamięć;
  C. jądro pamięci K(t) z równania Volterry Ċ(t) = −∫₀ᵗ K(t−s) C(s) ds dla autokorelacji położenia C(t) (OU: K = γδ(t), czyli K_n≈0 dla n≥1);
  D. centrum drgań: średnia krocząca W kroków; udział wariancji ruchu centrum i drgań wokół niego, dryf centrum w jednostkach klatki,
     kurtoza przyrostów centrum (ciągły dryf czy skoki).
Uruchomienie: F:/Python/3.13/python.exe markow_test.py [N_SOND=8] [OD=300] [DO=1094] → MARKOW_TEST.json, markow_test.png"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, '..', '..', 'experiments', 'maly_vit_od_zera_35000_21epok_20261009', 'wyniki', 'gesty', 'gesty')
N_SOND = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OD = int(sys.argv[2]) if len(sys.argv) > 2 else 300
DO = int(sys.argv[3]) if len(sys.argv) > 3 else 1094
TOK, DIM = 600, 128
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

# ---------------------------------------------------------------- dane: pełne 768D → rzut 128D, współrzędne względne
g = torch.Generator().manual_seed(0)
R = (torch.randn(768, DIM, generator=g) / DIM ** 0.5)
Xs, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pelne768_*.npz'))):
    z = np.load(p)
    k = z['kroki']
    m = (k >= OD) & (k <= DO)
    if not m.any():
        continue
    lat = torch.from_numpy(z['latent'][m][:, :N_SOND].astype(np.float32))          # (n, S, 600, 768)
    Xs.append((lat @ R).numpy())
    kroki.append(k[m])
    del z, lat
X = np.concatenate(Xs); kroki = np.concatenate(kroki).astype(int); del Xs
assert (np.diff(kroki) == 1).all(), 'zapis pełny nie co 1 krok w tym oknie'
T, S = X.shape[:2]
log(f'rzut 128D: {T} kroków ({kroki[0]}–{kroki[-1]}), {S} sond × {TOK} tokenów')
mod = X.mean(2, keepdims=True)                                                      # mod zdjęcia (S, T)
Xr = X - mod                                                                        # współrzędne względne
udzial_modu = float((np.diff(mod, axis=0) ** 2).sum() / (np.diff(X, axis=0) ** 2).sum() * TOK)   # udział energii kroku w modzie zdjęcia
del X, mod
Xr -= Xr.mean(0, keepdims=True)                                                     # drgania wokół centrum okna (T, S, 600, 128)
Xt = Xr
del Xr
klatka2 = float((Xt ** 2).sum(-1).mean())                                           # ⟨|x̃|²⟩ = rozmiar klatki² w tym oknie
log(f'drgania wokół centrum okna: klatka² {klatka2:.3f}, mod zdjęcia {udzial_modu:.0%} energii kroku')

# ---------------------------------------------------------------- A. operator jednokrokowy i rozmaitości piDMD
D = DIM
Cxx = np.zeros((D, D)); Cyx = np.zeros((D, D)); Cyy = np.zeros((D, D)); n_par = 0
for s in range(S):
    A_ = Xt[:-1, s].reshape(-1, D).astype(np.float64); B_ = Xt[1:, s].reshape(-1, D).astype(np.float64)
    Cxx += A_.T @ A_; Cyx += B_.T @ A_; Cyy += B_.T @ B_; n_par += len(A_)
E_trwanie = float(np.trace(Cyy - Cyx - Cyx.T + Cxx))                                # ||Y − X||²


def blad(A):
    """||Y − AX||² z macierzy kowariancji, względem trwania."""
    return float(np.trace(Cyy - A @ Cyx.T - Cyx @ A.T + A @ Cxx @ A.T) / E_trwanie)


from scipy.linalg import solve_continuous_lyapunov  # noqa: E402

A_ls = Cyx @ np.linalg.pinv(Cxx)
A_sym = solve_continuous_lyapunov(Cxx, Cyx + Cyx.T)                                # A Cxx + Cxx A = Cyx + Cyxᵀ
A_skew = solve_continuous_lyapunov(Cxx, Cyx - Cyx.T)                               # A Cxx + Cxx A = Cyx − Cyxᵀ, A = −Aᵀ
U_, _, Vt_ = np.linalg.svd(Cyx); A_orto = U_ @ Vt_                                  # Procrustes (ortogonalny)
A_diag = np.diag(np.diag(Cyx) / np.diag(Cxx))
A_skal = np.eye(D) * (np.trace(Cyx) / np.trace(Cxx))                               # jedna wspólna stała relaksacji
operatory = {'najmniejsze_kwadraty': A_ls, 'symetryczny': A_sym, 'skosnie_symetryczny': A_skew, 'ortogonalny': A_orto, 'diagonalny': A_diag,
             'skalarny': A_skal, 'trwanie': np.eye(D), 'zero (powrót do centrum w 1 kroku)': np.zeros((D, D))}
bledy = {k: blad(v) for k, v in operatory.items()}
lam = np.linalg.eigvals(A_ls)
widmo = dict(udzial_zespolonych=float((abs(lam.imag) > 1e-6).mean()), modul_min=float(abs(lam).min()), modul_med=float(np.median(abs(lam))),
             modul_max=float(abs(lam).max()), czasy_zaniku_kroki=dict(min=float(-1 / np.log(abs(lam).max())), med=float(-1 / np.log(np.median(abs(lam)))),
                                                                       max=float(-1 / np.log(max(abs(lam).min(), 1e-9)))),
             kat_med_stopnie=float(np.degrees(np.median(abs(np.angle(lam))))), asymetria=float(np.linalg.norm(A_ls - A_ls.T) / np.linalg.norm(A_ls + A_ls.T)))
lam_s = np.linalg.eigvalsh(A_sym)
widmo['symetryczny_wartosci_wlasne'] = dict(min=float(lam_s.min()), med=float(np.median(lam_s)), max=float(lam_s.max()),
                                            czas_zaniku_med=float(-1 / np.log(abs(np.median(lam_s)))))
log('A. błędy jednego kroku względem trwania:', {k: round(v, 3) for k, v in bledy.items()}, '| widmo LS: zespolone', round(widmo['udzial_zespolonych'], 2),
    'moduł med', round(widmo['modul_med'], 3), 'asymetria', round(widmo['asymetria'], 3))

# PyDMD piDMD (jeśli API przyjmuje pary X, Y) — kontrola własnych wzorów na podpróbce
pidmd = {}
try:
    import inspect
    from pydmd import PiDMD
    sig = inspect.signature(PiDMD.fit)
    if 'Y' in sig.parameters or 'y' in sig.parameters:
        idx = np.random.default_rng(0).choice(n_par, 60000, replace=False)
        Xa = np.concatenate([Xt[:-1, s].reshape(-1, D) for s in range(S)])[idx].T.astype(np.float64)
        Ya = np.concatenate([Xt[1:, s].reshape(-1, D) for s in range(S)])[idx].T.astype(np.float64)
        for manif in ('symmetric', 'skewsymmetric', 'unitary', 'diagonal'):
            try:
                m_ = PiDMD(manifold=manif, compute_A=True).fit(Xa, Ya)
                A_p = np.asarray(m_.A)
                pidmd[manif] = dict(blad=blad(A_p.real), roznica_do_wzoru=float(np.linalg.norm(A_p.real - {'symmetric': A_sym, 'skewsymmetric': A_skew, 'unitary': A_orto, 'diagonal': A_diag}[manif]) / np.linalg.norm(A_p.real)))
            except Exception as e:  # noqa: BLE001
                pidmd[manif] = f'błąd: {type(e).__name__}: {e}'[:200]
        del Xa, Ya
    else:
        pidmd = {'uwaga': f'PiDMD.fit bez pary (X, Y): {sig}'}
    import pydmd
    pidmd['wersja'] = getattr(pydmd, '__version__', '?')
except Exception as e:  # noqa: BLE001
    pidmd = {'uwaga': f'PyDMD niedostępny: {type(e).__name__}: {e}'[:200]}
log('piDMD:', pidmd)

# ---------------------------------------------------------------- B. czasy implikowane VAMP (deeptime)
vamp = {}
try:
    from deeptime.decomposition import VAMP
    traj = [Xt[:, s, i].astype(np.float64) for s in range(S) for i in range(0, TOK, 4)]      # co 4. token: 1200 trajektorii × T × 128
    for tau in (1, 2, 3, 5, 8, 12, 20, 30, 50, 80):
        m_ = VAMP(lagtime=tau, dim=12).fit(traj).fetch_model()
        sv = np.clip(m_.singular_values[:12], 1e-9, 1 - 1e-9)
        vamp[tau] = dict(sv=[round(float(v), 4) for v in sv], czasy=[round(float(-tau / math.log(v)), 1) for v in sv], vamp2=float((sv ** 2).sum()))
        log(f'VAMP τ={tau:2d}: czasy implikowane {vamp[tau]["czasy"][:5]}  VAMP-2 {vamp[tau]["vamp2"]:.2f}')
    del traj
except Exception as e:  # noqa: BLE001
    vamp = {'uwaga': f'deeptime niedostępny: {type(e).__name__}: {e}'[:300]}
    log(vamp)

# ---------------------------------------------------------------- C. autokorelacja położenia i jądro pamięci (Volterra)
TMAX = min(400, T - 1)
C = np.zeros(TMAX + 1)
for s in range(S):
    Y = np.ascontiguousarray(Xt[:, s].reshape(T, -1))                                # (T, 600·128) float32
    for n in range(TMAX + 1):
        C[n] += float(np.dot(Y[:T - n].ravel(), Y[n:].ravel())) / (T - n)
    log(f'C. autokorelacja: sonda {s + 1}/{S}')
C /= C[0]
# Volterra: Ċ_n = −Σ_{m=0}^{n} w_m K_{n−m} C_m (trapez), Δt = 1 krok
dC = np.gradient(C)
Kk = np.zeros(TMAX + 1)
Kk[0] = -dC[0] / (0.5 * C[0])
for n in range(1, TMAX + 1):
    suma = 0.5 * Kk[0] * C[n] + sum(Kk[n - m] * C[m] for m in range(1, n))
    Kk[n] = (-dC[n] - suma) / (0.5 * C[0])
tau_C = float(np.argmax(C < math.exp(-1))) if (C < math.exp(-1)).any() else float('inf')
# dopasowanie rozciągniętej eksponenty C = exp(−(t/τ)^β) na 0.05 < C < 0.9
sel = (C > 0.05) & (C < 0.9) & (np.arange(TMAX + 1) > 0)
beta_C = tau_fit = float('nan')
if sel.sum() > 5:
    b, a_ = np.polyfit(np.log(np.arange(TMAX + 1)[sel]), np.log(-np.log(C[sel])), 1)
    beta_C, tau_fit = float(b), float(math.exp(-a_ / b))
pam = dict(K0=float(Kk[0]), K_ogon_1_5=[round(float(v), 4) for v in Kk[1:6]], udzial_ogona=float(abs(Kk[1:50]).sum() / abs(Kk[:50]).sum()),
           calka_K_1_50_do_K0=float(Kk[1:50].sum() / Kk[0]), tau_1e_C=tau_C, beta_C=beta_C, tau_rozciagnieta=tau_fit,
           uwaga='OU: K_n ≈ 0 dla n ≥ 1; ujemny ogon = przyrosty antyskorelowane (sprężyste odbicie), dodatni = lepkość/pamięć')
log(f'C. autokorelacja: τ_1e {tau_C} kroków, β {beta_C:.2f}; jądro: K0 {Kk[0]:.3f}, ogon 1–5 {pam["K_ogon_1_5"]}, udział ogona {pam["udzial_ogona"]:.2f}')

# ---------------------------------------------------------------- D. centrum drgań: średnia krocząca
json.dump(dict(opis='częściowe (A–C)', A_operator=dict(bledy_wzgledem_trwania=bledy, widmo_LS=widmo, piDMD=pidmd), B_vamp=vamp, C_pamiec=pam),
          open(os.path.join(HERE, 'MARKOW_TEST_czesciowe.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
centra = {}
for W in (10, 25, 50, 100):
    sc = sd = sdryf = 0.0; r2 = []
    for s in range(S):
        cs = np.cumsum(Xt[:, s].reshape(T, -1), axis=0, dtype=np.float64)
        Cn = ((cs[W:] - cs[:-W]) / W).astype(np.float32)                              # średnia krocząca (T−W, 600·128)
        del cs
        drg = Xt[W // 2: W // 2 + len(Cn), s].reshape(len(Cn), -1) - Cn
        sc += float((Cn ** 2).sum()) / (len(Cn) * TOK); sd += float((drg ** 2).sum()) / (len(Cn) * TOK)
        del drg
        inc = np.diff(Cn[::W], axis=0).reshape(-1, TOK, D)                              # przyrosty centrum co W kroków
        r2.append((inc ** 2).sum(-1).ravel())
        sdryf += float(((Cn[-1] - Cn[0]).reshape(TOK, D) ** 2).sum(-1).mean())
        del Cn, inc
    r2 = np.concatenate(r2)
    var_c, var_d = sc / S, sd / S
    kurt = float((r2 ** 2).mean() / r2.mean() ** 2)                                   # gaussowskie w D wymiarach ≈ 1 + 2/D; skoki → większa
    centra[W] = dict(wariancja_centrum=var_c, wariancja_drgan=var_d, udzial_drgan=var_d / (var_c + var_d),
                     dryf_centrum_do_klatki=float(np.sqrt(sdryf / S / klatka2)),
                     krok_centrum_do_klatki=float(np.sqrt(r2.mean() / klatka2)), kurtoza_przyrostow_centrum=kurt, kurtoza_gauss=float(1 + 2 / D))
    log(f'D. W={W:3d}: drgania {centra[W]["udzial_drgan"]:.2f} wariancji, dryf centrum w oknie {centra[W]["dryf_centrum_do_klatki"]:.2f} klatki, '
        f'krok centrum co W {centra[W]["krok_centrum_do_klatki"]:.2f} klatki, kurtoza {kurt:.2f} (gauss {1 + 2 / D:.2f})')

wyn = dict(opis=__doc__.split('\n')[0], okno=[int(kroki[0]), int(kroki[-1])], sond=S, tokenow=TOK, wymiar=D, par=int(n_par), udzial_modu_zdjecia_w_energii_kroku=udzial_modu,
           klatka2=klatka2, A_operator=dict(bledy_wzgledem_trwania=bledy, widmo_LS=widmo, piDMD=pidmd), B_vamp=vamp, C_pamiec=pam,
           C_autokorelacja=[round(float(v), 5) for v in C[:TMAX + 1]], C_jadro=[round(float(v), 5) for v in Kk[:TMAX + 1]], D_centrum=centra)
json.dump(wyn, open(os.path.join(HERE, 'MARKOW_TEST.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=100)
a = ax[0, 0]
nazwy = [k for k in bledy if k not in ('trwanie',)]
a.barh(nazwy, [bledy[k] for k in nazwy], color=['#ffffff' if k == 'najmniejsze_kwadraty' else '#4a8cff' if k == 'symetryczny' else '#ff4433' if 'skos' in k or 'orto' in k else '#ffd166' for k in nazwy])
a.axvline(1, color='#8a8f99', ls='--', lw=1); a.set_xlabel('błąd jednego kroku / błąd trwania'); a.set_title('A. struktura operatora (piDMD): relaksacja czy obrót', fontsize=10); a.grid(True, axis='x', alpha=0.4)
a = ax[0, 1]
if isinstance(vamp, dict) and 1 in vamp:
    taus = sorted(vamp);
    for i in range(5):
        a.plot(taus, [vamp[t]['czasy'][i] for t in taus], 'o-', ms=4, label=f'proces {i + 1}')
    a.plot(taus, taus, '--', color='#8a8f99', lw=1, label='τ (granica rozdzielczości)')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('opóźnienie τ [kroki]'); a.set_ylabel('czas implikowany [kroki]'); a.legend(fontsize=8)
a.set_title('B. test Markowa (VAMP): płasko = Markow, rośnie = pamięć', fontsize=10); a.grid(True, alpha=0.4)
a = ax[1, 0]
a.plot(C[:TMAX + 1], color='#ffffff', label='autokorelacja położenia C(t)')
a.plot(np.arange(TMAX + 1), np.exp(-np.arange(TMAX + 1) / max(tau_C, 1)), '--', color='#ffd166', lw=1, label=f'OU z tym samym τ_1e={tau_C:.0f}')
a2 = a.twinx(); a2.plot(np.arange(1, 61), Kk[1:61] / Kk[0], color='#ff4433', lw=1, label='jądro pamięci K(t)/K(0), t ≥ 1')
a2.axhline(0, color='#ff4433', lw=0.5, alpha=0.5); a2.set_ylabel('K(t)/K(0)', color='#ff4433'); a2.set_ylim(-0.3, 0.3)
a.set_xscale('log'); a.set_xlabel('t [kroki]'); a.set_ylabel('C(t)'); a.legend(loc='upper right', fontsize=8); a2.legend(loc='lower left', fontsize=8)
a.set_title(f'C. Mori–Zwanzig: β={beta_C:.2f}, udział ogona jądra {pam["udzial_ogona"]:.2f}', fontsize=10); a.grid(True, alpha=0.4)
a = ax[1, 1]
Ws = sorted(centra)
a.plot(Ws, [centra[w]['udzial_drgan'] for w in Ws], 'o-', color='#ffffff', label='udział drgań wokół centrum w wariancji')
a.plot(Ws, [centra[w]['dryf_centrum_do_klatki'] for w in Ws], 's-', color='#4a8cff', label='dryf centrum przez okno / klatka')
a.plot(Ws, [centra[w]['kurtoza_przyrostow_centrum'] / centra[w]['kurtoza_gauss'] for w in Ws], '^-', color='#ff4433', label='kurtoza przyrostów centrum / gaussowska')
a.set_xscale('log'); a.set_xlabel('okno średniej kroczącej W [kroki]'); a.legend(fontsize=8); a.set_title('D. centrum drgań: ile ruchu to dryf centrum', fontsize=10); a.grid(True, alpha=0.4)
fig.suptitle(f'Jaki to ruch w skali kroku: {S} sond × {TOK} tokenów, rzut 128D, kroki {kroki[0]}–{kroki[-1]} (mod zdjęcia = {udzial_modu:.0%} energii kroku)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'markow_test.png'))
log('zapisano MARKOW_TEST.json, markow_test.png')
