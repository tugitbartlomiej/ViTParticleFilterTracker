"""Wizualizacja LDA od początku do końca uczenia (seed 47, historia 806 checkpointów aplikacji, stany x̃ przed MLP bloku 6).
Dla klatki, w kolumnach checkpointy; wiersze:
  1. LDA dopasowana NA TYM checkpoincie do etykiety z obrysu (narzędzie zielone, łaty mieszane pominięte),
  2. stała kamera: LDA z końca uczenia (pula 100 ostatnich checkpointów) zastosowana do tego checkpointu,
  3. LDA dopasowana do grupy wykrytej BEZ etykiet (TokenCut na aktywacjach MLP, czerwone); obwódka zielona = obrys narzędzia.
W tytułach: Fisher i p zera przestrzennego (ta sama maska przesunięta w tło, LDA od nowa), dla grupy Dice z obrysem."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = Path(__file__).resolve().parents[2]
APP = E.parents[1] / 'app'
sys.path.insert(0, str(APP))
import lasery_dane as LD, rzuty_dane as RD
KLATKI = {'A (TR_0050 kl. 1629)': '140854f7a6dbcbe5', 'C (TR_0001 kl. 274)': 'c584ef4647149c7c'}
IDX = [0, 30, 100, 200, 400, 805]
dev = RD._dev()


def rzut(X, B):
    return (X - X.mean(0)) @ B


for nazwa, hid in KLATKI.items():
    h = LD.Historia(APP / 'cache_uczenia/lasery' / hid); d = RD.wczytaj(h)
    narz, mies, u = h.narz, h.mieszane, h.udzial
    keep = ~mies
    ks = np.arange(d.K - 100, d.K)
    Xs = np.asarray(d.X[ks], np.float32)[:, keep].reshape(-1, RD.D)
    Bk = RD.baza_L_t(torch.as_tensor(Xs, device=dev), torch.as_tensor(np.tile(narz[keep], len(ks)), device=dev)[None])[0].cpu().numpy()
    fig, ax = plt.subplots(3, len(IDX), figsize=(3.3 * len(IDX), 10.2))
    for c, i in enumerate(IDX):
        X = np.asarray(d.X[i], np.float32)
        Xk = X[keep]; m = narz[keep]; kk = RD.k_sasiadow(m.sum())
        # 1: LDA na tym checkpoincie, obrys
        pr, zp, n = RD.zero_przestrzenne_gpu(X, narz, narz | mies | (u > 0), keep, k=kk)
        B1 = RD.baza_L_t(torch.as_tensor(Xk, device=dev), torch.as_tensor(m, device=dev)[None])[0].cpu().numpy()
        # 2: stała kamera
        P2 = rzut(Xk, Bk); f2 = RD.miary_separacji_t(torch.as_tensor(P2, device=dev)[None], torch.as_tensor(m, device=dev)[None], kk)
        # 3: grupa bez etykiet
        G = RD.wykryj_grupe(np.asarray(h.Z[i], np.float32)); zg = RD.zgodnosc_z_obrysem(G, u)
        for r, (P, kol, maska, tyt) in enumerate((
                (rzut(Xk, B1), 'lime', m, f'LDA na tym ckpt (obrys)\nFisher {pr["fisher"]:.2f}, p przestrz. {zp["fisher"][1]:.3f}' if zp else f'Fisher {pr["fisher"]:.2f}'),
                (P2, 'lime', m, f'stała kamera (LDA z końca)\nFisher {float(f2["fisher"][0]):.2f}'),
                (rzut(X, RD.baza_L_t(torch.as_tensor(X, device=dev), torch.as_tensor(G, device=dev)[None])[0].cpu().numpy()) if G.sum() >= 2 else None,
                 'red', G, f'LDA na grupie bez etykiet ({G.sum()} tok.)\nDice z obrysem {zg["dice"]:.2f}, prec. {zg["precyzja"]:.2f}'))):
            a = ax[r, c]
            if P is None:
                a.text(0.5, 0.5, 'brak grupy', ha='center', transform=a.transAxes); a.set_xticks([]); a.set_yticks([]); continue
            a.scatter(P[~maska, 0], P[~maska, 1], s=4, c='0.65', lw=0)
            a.scatter(P[maska, 0], P[maska, 1], s=22, c=kol, ec='k', lw=0.3)
            if r == 2:
                a.scatter(P[narz, 0], P[narz, 1], s=60, facecolors='none', ec='lime', lw=1.0)
            a.set_title(f'krok {int(h.kroki[i])}: ' + tyt, fontsize=8)
            a.tick_params(labelsize=6)
        ax[0, 0].set_ylabel('oś 2'); ax[2, 0].set_xlabel('oś 1 (LDA)')
    fig.suptitle(f'Klatka {nazwa}, seed 47: LDA od początku do końca uczenia ({h.etykieta}; narzędzie {narz.sum()} łat, '
                 f'mieszane {mies.sum()} pominięte)', fontsize=11)
    fig.tight_layout()
    out = E / 'Wyniki/Wykresy' / f'lda_poczatek_koniec_{hid}.png'
    fig.savefig(out, dpi=105); plt.close(fig); print(out)
