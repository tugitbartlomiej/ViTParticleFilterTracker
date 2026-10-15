"""Związek atencji (6 bloków osobno, średnia bloków, rollout 1..6) z przesunięciem centrów studni dla zdjęcia TRAIN 0 przez 60 epok.
Przesunięcie centrum patcha j w checkpoincie t: lok[t, j] = |Qc[t, j] − g_j| (jak w filmach 2/13–16: węzeł ±300 kroków, Prokrustes 2D;
dla kontroli też 3D). Atencja: ile uwagi patch DOSTAJE (atencja6_train0.npz); rollout liczony tu na GPU (jak app/atencja_zbioru.py:
bez przekątnej, 0,5·A + 0,5·I, iloczyn bloków 1..6), zapis atencja_rollout_train0.npz.
Miary:
 (1) przestrzenna: Spearman po 600 patchach w każdym checkpoincie, ρ_b(t); też tylko poza narzędziem (kontrola „obie mapy świecą na narzędziu”);
 (2) czasowa globalna: udział atencji na narzędziu vs stosunek przesunięć narzędzie/reszta, poziomy i pierwsze różnice, z przesunięciem w czasie;
 (3) czasowa per patch: Spearman w czasie dla każdego patcha (od e1), mediana po patchach; też po odjęciu trendu (różnice co checkpoint).
Wynik: ATENCJA_CENTRA.json + atencja_centra.png. Uruchomienie: F:/Python/3.13/python.exe -X utf8 analiza_atencja_centra.py"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from lasery_wspolne import EP, HERE, dane, prokrustes2d, zastosuj

ROOT = HERE.parents[2]
D = dane(0)
kroki, narz = D['kroki'].astype(float), D['narz']; K = len(kroki)
g2 = D['g3'][:, :2]
_, mapy = prokrustes2d(D['Q3'][..., :2], g2)
Qc = np.stack([zastosuj(D['Q3'][t, :, :2], mapy[t]) for t in range(K)])
lok = np.linalg.norm(Qc - g2, axis=-1)                                        # (K, 600)
lok3 = np.linalg.norm(D['Q3'] - D['g3'], axis=-1)                             # kontrola 3D (bez Prokrustesa)
A6 = np.load(HERE / 'atencja6_train0.npz')['atencja'].astype(float)          # (K, 6, 600)

# ---- rollout (GPU), raz
plik_r = HERE / 'atencja_rollout_train0.npz'
if not plik_r.exists():
    import cv2
    import torch
    sys.path.insert(0, str(ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod'))
    from model import MalyViT
    dev = 'cuda'
    im = cv2.cvtColor(cv2.imread(str(HERE / 'zdjecie_train0.jpg')), cv2.COLOR_BGR2RGB)
    x = torch.as_tensor(cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().div(255).to(dev)
    net = MalyViT().to(dev).eval(); pam = {}
    for b in range(6):
        net.bloki[b].attn.register_forward_hook(lambda m, a, o, b=b: pam.__setitem__(b, a[0].detach()))
    I = torch.eye(600, device=dev); roll = np.zeros((K, 600), np.float32)
    for i, k in enumerate(kroki):
        net.load_state_dict(torch.load(ROOT / f'assets/MalyViT_35000_od_zera_s47/wagi/k{int(k):05d}.pt', map_location=dev, weights_only=True)['net'])
        with torch.no_grad():
            net(x); M = I.clone()
            for b in range(6):
                at, xin = net.bloki[b].attn, pam[b]; B, N, C = xin.shape
                qkv = at.qkv(xin).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
                A = ((qkv[0] * at.scale) @ qkv[1].transpose(-2, -1)).softmax(-1).float().mean(1)[0]
                A.fill_diagonal_(0); A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
                Ai = 0.5 * A + 0.5 * I; M = (Ai / Ai.sum(-1, keepdim=True)) @ M
            M.fill_diagonal_(0); M = M / M.sum(-1, keepdim=True).clamp_min(1e-12)
            roll[i] = M.mean(0).cpu().numpy()
    np.savez_compressed(plik_r, kroki=kroki, rollout=roll)
ROLL = np.load(plik_r)['rollout'].astype(float)

MAPY = {f'blok {b + 1}': A6[:, b] for b in range(6)}
MAPY['średnia bloków'] = A6.mean(1)
MAPY['rollout 1..6'] = ROLL
FAZY = {'kroki 0–300': kroki <= 300, 'kroki 300–1094': (kroki > 300) & (kroki < EP), 'epoki 1–5': (kroki >= EP) & (kroki < 5 * EP),
        'epoki 5–60': kroki >= 5 * EP}


def rho_wiersze(X, Y, maska=None):
    """Spearman po patchach w każdym checkpoincie."""
    if maska is not None:
        X, Y = X[:, maska], Y[:, maska]
    rx = np.apply_along_axis(rankdata, 1, X); ry = np.apply_along_axis(rankdata, 1, Y)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def rho_kolumny(X, Y):
    """Spearman w czasie dla każdego patcha."""
    rx = np.apply_along_axis(rankdata, 0, X); ry = np.apply_along_axis(rankdata, 0, Y)
    rx -= rx.mean(0); ry -= ry.mean(0)
    return (rx * ry).sum(0) / np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum(0) + 1e-12)


rng = np.random.default_rng(0)
wynik = {'opis': __doc__.split('\n')[0], 'patchy_narzedzia': int(narz.sum()), 'checkpointow': K}
od_e1 = kroki >= EP
ratio = lok[:, narz].mean(1) / lok[:, ~narz].mean(1)
R_RATIO = {}
for nazwa, M in MAPY.items():
    r_all = rho_wiersze(M, lok); r_bez = rho_wiersze(M, lok, ~narz); r3 = rho_wiersze(M, lok3)
    udzial = M[:, narz].sum(1) / M.sum(1)                                     # udział atencji na narzędziu
    # czasowa globalna: poziomy (od e1) i pierwsze różnice z przesunięciem
    poz = spearmanr(udzial[od_e1], ratio[od_e1])[0]
    du, dr = np.diff(udzial[od_e1]), np.diff(ratio[od_e1])
    lagi = {lag: float(spearmanr(du[max(0, -lag):len(du) - max(0, lag)], dr[max(0, lag):len(dr) - max(0, -lag)])[0]) for lag in (-3, -1, 0, 1, 3)}
    # per patch w czasie (od e1): poziomy i różnice
    pp = rho_kolumny(M[od_e1], lok[od_e1]); pd = rho_kolumny(np.diff(M[od_e1], axis=0), np.diff(lok[od_e1], axis=0))
    # test permutacyjny dla ρ różnic per patch: przetasowanie kolejności checkpointów w jednej z serii
    perm = np.array([np.median(rho_kolumny(np.diff(M[od_e1], axis=0), np.diff(lok[od_e1], axis=0)[rng.permutation(od_e1.sum() - 1)]))
                     for _ in range(50)])
    R_RATIO[nazwa] = (r_all, r_bez, udzial)
    wynik[nazwa] = {
        'przestrzenna_rho_fazy': {f: float(np.mean(r_all[m])) for f, m in FAZY.items()},
        'przestrzenna_rho_bez_narzedzia_fazy': {f: float(np.mean(r_bez[m])) for f, m in FAZY.items()},
        'przestrzenna_rho_3D_e5_60': float(np.mean(r3[FAZY['epoki 5–60']])),
        'czas_udzial_narzedzia_vs_ratio_poziomy_od_e1': float(poz),
        'czas_roznice_udzial_vs_ratio_lag': lagi,
        'czas_per_patch_poziomy_mediana': float(np.median(pp)),
        'czas_per_patch_roznice_mediana': float(np.median(pd)),
        'czas_per_patch_roznice_narzedzie_mediana': float(np.median(pd[narz])),
        'czas_per_patch_roznice_permutacje_p95': float(np.percentile(np.abs(perm), 95)),
        'udzial_atencji_narzedzia_e1_e60': [float(udzial[od_e1][0]), float(udzial[-1])],
    }
    print(f"{nazwa:15s} ρ_przestrz e5–60 {wynik[nazwa]['przestrzenna_rho_fazy']['epoki 5–60']:+.2f} "
          f"(bez narz. {wynik[nazwa]['przestrzenna_rho_bez_narzedzia_fazy']['epoki 5–60']:+.2f}, 3D {wynik[nazwa]['przestrzenna_rho_3D_e5_60']:+.2f}) | "
          f"poziomy udział~ratio {poz:+.2f} | różnice lag0 {lagi[0]:+.2f} | per patch poziomy {np.median(pp):+.2f}, różnice {np.median(pd):+.3f} "
          f"(perm p95 {wynik[nazwa]['czas_per_patch_roznice_permutacje_p95']:.3f})", flush=True)
json.dump(wynik, open(HERE / 'ATENCJA_CENTRA.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---- rysunek
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
kol = plt.cm.viridis(np.linspace(0, 0.9, 6)).tolist() + ['#d62728', '#000000']
IX = np.arange(K); tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]; tki = [int(np.argmin(abs(kroki - k))) for k in tk]
for (nazwa, (r_all, r_bez, udz)), c in zip(R_RATIO.items(), kol):
    lw = 2.2 if nazwa in ('średnia bloków', 'rollout 1..6') else 1.2
    ax[0].plot(IX, uniform_filter1d(r_all, 9), color=c, lw=lw, label=nazwa)
    ax[1].plot(IX, uniform_filter1d(r_bez, 9), color=c, lw=lw, label=nazwa)
    ax[2].plot(IX, uniform_filter1d(udz, 9), color=c, lw=lw, label=nazwa)
ax2b = ax[2].twinx(); ax2b.plot(IX, uniform_filter1d(ratio, 9), color='#ff7f0e', lw=2, ls='--'); ax2b.set_ylabel('przesunięcie centrów: narzędzie / reszta', color='#ff7f0e')
for a, t in zip(ax, ['ρ przestrzenne: atencja ~ przesunięcie centrów (600 patchy)', 'to samo tylko POZA narzędziem', 'udział atencji na narzędziu (linie) i stosunek przesunięć (przerywana)']):
    a.set_xticks(tki); a.set_xticklabels(['0', '300', '1k', 'e5', 'e20', 'e60']); a.grid(alpha=0.3); a.set_title(t, fontsize=11); a.axhline(0, color='k', lw=0.6)
ax[0].legend(fontsize=8.5, ncol=2); ax[0].set_ylabel('Spearman ρ (wygładzone 9 checkpointów)')
fig.tight_layout(); fig.savefig(HERE / 'atencja_centra.png', dpi=110)
print('zapisano ATENCJA_CENTRA.json, atencja_centra.png')
