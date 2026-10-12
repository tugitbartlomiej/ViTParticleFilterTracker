"""Analiza lokalna biegu z Edenu (PLAN.md): w którą stronę idzie latent zdjęcia celu 0055/4019 w ramieniu A150
względem kontroli C0, w przestrzeni K1000.

δ = z*_K1000 − z0_K1000 (z_cel_k1000.py). Dla każdego checkpointu e (K1000 + 47 e):
  postęp ramienia      ⟨z_R(e) − z_R(0), δ⟩ / ||δ||²  (ruch od wspólnego startu wzdłuż δ),
  kosinus ruchu        cos(z_R(e) − z_R(0), δ),
  różnica ramion       D(e) = z_A150(e) − z_C0(e): ||D||, cos(D, δ), ⟨D, δ⟩/||δ||², część widoczna dla fc2 K1000
                       (W2 startu; po uczeniu fc2 się zmienia, więc to przybliżenie),
  szum obliczeń        ||z_C0b(e) − z_C0(e)|| (w trybie deterministycznym oczekiwane 0),
  kierunek grup        cos(średnia po tokenach D, średnia nowych − średnia usuniętych w K1000).
Zapis: outputs/vit_k1000_podmiana150_sd/ANALIZA.json, WYKRES.png
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_podmiana150_sd/kod/analiza.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
W = ROOT / 'outputs' / 'vit_k1000_podmiana150_sd'
P = W / 'eden' / 'pelny'
RAMIONA = ['C0', 'A150', 'C0b']


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    zs = np.load(W / 'z_cel_k1000' / 'zb_j10389_minmax_k1000.npy').astype(np.float64)
    z0l = np.load(W / 'z_cel_k1000' / 'z0_j10389_k1000.npy').astype(np.float64)
    d = zs - z0l; dd = float((d * d).sum())
    W2 = torch.load(ROOT / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt', map_location='cpu',
                    weights_only=True)['net']['bloki.5.mlp.fc2.weight'].double().numpy()            # [192, 768]
    dw = d @ W2.T
    R = {r: dict(np.load(P / r / 'POMIARY.npz')) for r in RAMIONA}
    met = {r: json.loads((P / r / 'METRYKI.json').read_text(encoding='utf-8')) for r in RAMIONA}
    kroki = R['C0']['kroki'].tolist()
    assert all(R[r]['kroki'].tolist() == kroki for r in RAMIONA)
    Z = {r: R[r]['z_cel'].astype(np.float64) for r in RAMIONA}
    start = dict(roznica_startow_A_C=float(np.abs(Z['A150'][0] - Z['C0'][0]).max()),
                 gpu_vs_cpu_wzgl=float(np.linalg.norm(Z['C0'][0] - z0l) / np.linalg.norm(z0l)))
    g_n = R['C0']['lat_nowe150'][0][:, 0].mean(0); g_u = R['C0']['lat_usuniete150'][0][:, 0].mean(0)
    wiersze = []
    for e, k in enumerate(kroki):
        rek = dict(krok=k)
        for r in RAMIONA:
            m = Z[r][e] - Z[r][0]
            rek[f'postep_{r}'] = float((m * d).sum() / dd)
            rek[f'postep_wid_{r}'] = float(((m @ W2.T) * dw).sum() / (dw * dw).sum())
            rek[f'cos_ruchu_{r}'] = cos(m, d) if e else None
            rek[f'norma_ruchu_{r}'] = float(np.linalg.norm(m))
            mr = next(x for x in met[r] if x['krok'] == k)
            for a in ('cel_pewnosc', 'cel_iou', 'mAP50_wal3000', 'nowe150_iou', 'usuniete150_iou', 'sondy64_iou'):
                rek[f'{a}_{r}'] = mr[a]
        D = Z['A150'][e] - Z['C0'][e]; S = Z['C0b'][e] - Z['C0'][e]
        rek.update(roznica_norma=float(np.linalg.norm(D)), szum_norma=float(np.linalg.norm(S)),
                   roznica_cos_delta=cos(D, d) if e else None, roznica_postep=float((D * d).sum() / dd),
                   roznica_cos_delta_widoczny=cos(D @ W2.T, dw) if e else None,
                   roznica_cos_grupy=cos(D.mean(0), g_n - g_u) if e else None,
                   roznica_wzgl_ruchu_C0=float(np.linalg.norm(D) / (np.linalg.norm(Z['C0'][e] - Z['C0'][0]) + 1e-12)))
        wiersze.append(rek)
    out = dict(kroki=kroki, start=start, delta_norma=float(np.sqrt(dd)), z0_norma=float(np.linalg.norm(z0l)), wiersze=wiersze)
    (W / 'ANALIZA.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')

    x = np.array(kroki)
    kol = dict(C0='tab:gray', A150='tab:red', C0b='tab:blue')
    fig, ax = plt.subplots(2, 3, figsize=(18, 9), layout='constrained')
    for r in RAMIONA:
        ax[0, 0].plot(x, [100 * w[f'postep_{r}'] for w in wiersze], 'o-', c=kol[r], label=r, ls='--' if r == 'C0b' else '-')
        ax[0, 1].plot(x, [w[f'cel_pewnosc_{r}'] for w in wiersze], 'o-', c=kol[r], label=f'{r} pewność')
        ax[0, 1].plot(x, [w[f'cel_iou_{r}'] for w in wiersze], 's:', c=kol[r], label=f'{r} IoU')
        ax[0, 2].plot(x, [w[f'mAP50_wal3000_{r}'] for w in wiersze], 'o-', c=kol[r], label=r)
        ax[1, 2].plot(x, [w[f'nowe150_iou_{r}'] for w in wiersze], 'o-', c=kol[r], label=f'{r} nowe150')
    ax[0, 0].set_title('postęp latentu celu wzdłuż δ = z*_K1000 − z0 [%]'); ax[0, 0].axhline(0, c='k', lw=.6)
    ax[0, 1].set_title('zdjęcie celu 0055/4019: pewność i IoU najpewniejszej ramki')
    ax[0, 2].set_title('mAP@0,5 na 3000 klatkach walidacji')
    ax[1, 0].plot(x, [w['roznica_norma'] for w in wiersze], 'o-', c='tab:red', label='||z_A150 − z_C0||')
    ax[1, 0].plot(x, [w['szum_norma'] for w in wiersze], 'o-', c='tab:blue', label='||z_C0b − z_C0|| (szum obliczeń)')
    ax[1, 0].set_title('różnica latentu celu między ramionami')
    e = x[1:]
    ax[1, 1].plot(e, [w['roznica_cos_delta'] for w in wiersze[1:]], 'o-', label='cos(D, δ)')
    ax[1, 1].plot(e, [w['roznica_cos_delta_widoczny'] for w in wiersze[1:]], 's-', label='cos(D, δ) w części widocznej dla fc2')
    ax[1, 1].plot(e, [w['roznica_cos_grupy'] for w in wiersze[1:]], '^-', label='cos(średnia D, nowe − usunięte)')
    ax[1, 1].axhline(0, c='k', lw=.6); ax[1, 1].set_title('kierunek różnicy D = z_A150 − z_C0')
    ax[1, 2].set_title('IoU na 150 nowych klatkach')
    for a in ax.ravel():
        a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel('krok uczenia')
    fig.suptitle('Kontynuacja od K1000: 150 klatek znalezionych przez SD i DINO (A150) wobec oryginalnej bazy (C0)', fontsize=13)
    fig.savefig(W / 'WYKRES.png', dpi=110); plt.close(fig)
    for w in wiersze:
        print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in w.items()})
    print(start)


if __name__ == '__main__':
    main()
