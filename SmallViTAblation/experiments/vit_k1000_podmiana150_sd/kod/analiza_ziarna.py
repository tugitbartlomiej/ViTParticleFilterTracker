"""Analiza rozszerzenia (PLAN.md, 2026-10-08): ramiona C0 / A150 / R150 × ziarna 47, 48, 49 (zadanie Edenu 1824901).

Pytanie: czy dobór 150 klatek przez SD + DINO (A150) przesuwa latent zdjęcia celu 0055/4019 w stronę z*_K1000 bardziej
niż podmiana tych samych 150 pozycji na 150 losowych klatek z puli (R150)?
δ = z*_K1000 − z0_K1000 (z_cel_k1000.py), W2 = fc2 ostatniego bloku K1000 (start; po uczeniu fc2 się zmienia).
Dla ziarna s i checkpointu e: postęp ramienia ⟨z_R(e) − z_R(0), δ⟩/||δ||² (cały i widoczny dla fc2),
różnice D_A = z_A150 − z_C0 i D_R = z_R150 − z_C0: norma, cos z δ, cos w części widocznej, ⟨D, δ⟩/||δ||²;
pewność i IoU detekcji celu, mAP@0,5 (3000 klatek walidacji), IoU na dodanych klatkach (nowe150 dla A150, losowe150
dla R150, w każdym ramieniu mierzone obie grupy). Zestawienie: średnia ± odchylenie po 3 ziarnach na końcu (K1235)
i średnio po 5 checkpointach po starcie. Kontrola powtarzalności: C0_s47 z tego zadania wobec C0 z biegu 1824868.
Zapis: outputs/vit_k1000_podmiana150_sd/ANALIZA_ZIARNA.json, WYKRES_ZIARNA.png (ANALIZA.json i WYKRES.png pierwszego
biegu bez zmian).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_podmiana150_sd/kod/analiza_ziarna.py
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
P = W / 'eden' / 'ziarna'
ZIARNA = [47, 48, 49]
RAMIONA = ['C0', 'A150', 'R150']
DODANE = dict(A150='nowe150', R150='losowe150')


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ms(v):
    v = np.asarray(v, float)
    return dict(srednia=float(v.mean()), odch=float(v.std(ddof=1)) if len(v) > 1 else 0.0, wartosci=v.tolist())


def main():
    zs = np.load(W / 'z_cel_k1000' / 'zb_j10389_minmax_k1000.npy').astype(np.float64)
    z0l = np.load(W / 'z_cel_k1000' / 'z0_j10389_k1000.npy').astype(np.float64)
    d = zs - z0l; dd = float((d * d).sum())
    W2 = torch.load(ROOT / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt', map_location='cpu',
                    weights_only=True)['net']['bloki.5.mlp.fc2.weight'].double().numpy()
    dw = d @ W2.T
    R = {(r, s): dict(np.load(P / f'{r}_s{s}' / 'POMIARY.npz')) for r in RAMIONA for s in ZIARNA}
    met = {(r, s): json.loads((P / f'{r}_s{s}' / 'METRYKI.json').read_text(encoding='utf-8')) for r in RAMIONA for s in ZIARNA}
    kroki = R['C0', 47]['kroki'].tolist()
    assert all(R[k]['kroki'].tolist() == kroki for k in R)

    stary = W / 'eden' / 'pelny' / 'C0' / 'POMIARY.npz'
    powt = None
    if stary.exists():
        zo = np.load(stary)['z_cel']; zn = R['C0', 47]['z_cel']
        H = zn.astype(np.float64) - zo.astype(np.float64)            # ta sama konfiguracja, inna karta (A100 wobec H100)
        powt = dict(max_roznica_z_cel=float(np.abs(H).max()), identyczne=bool(np.array_equal(zo, zn)),
                    norma_roznicy_po_krokach=[float(np.linalg.norm(h)) for h in H],
                    cos_roznicy_z_delta=[cos(h, d) if i else None for i, h in enumerate(H)],
                    opis='C0_s47 z zadania 1824901 (A100, dgx-2) wobec C0 z 1824868 (H100, hopper): te same wejścia i ziarno, '
                         'inna karta; poziom rozjazdu z samej numeryki sprzętu')
    starty = {f'{r}_s{s}': float(np.abs(R[r, s]['z_cel'][0].astype(np.float64) - R['C0', s]['z_cel'][0]).max()) for r in RAMIONA for s in ZIARNA}

    wiersze = []
    for s in ZIARNA:
        Z = {r: R[r, s]['z_cel'].astype(np.float64) for r in RAMIONA}
        for e, k in enumerate(kroki):
            rek = dict(ziarno=s, krok=k)
            for r in RAMIONA:
                m = Z[r][e] - Z[r][0]
                rek[f'postep_{r}'] = float((m * d).sum() / dd)
                rek[f'postep_wid_{r}'] = float(((m @ W2.T) * dw).sum() / (dw * dw).sum())
                rek[f'norma_ruchu_{r}'] = float(np.linalg.norm(m))
                mr = next(x for x in met[r, s] if x['krok'] == k)
                for a in ('cel_pewnosc', 'cel_iou', 'mAP50_wal3000', 'nowe150_iou', 'losowe150_iou', 'usuniete150_iou', 'sondy64_iou'):
                    rek[f'{a}_{r}'] = mr.get(a)
            for r in ('A150', 'R150'):
                D = Z[r][e] - Z['C0'][e]
                rek[f'D_{r}_norma'] = float(np.linalg.norm(D))
                rek[f'D_{r}_postep'] = float((D * d).sum() / dd)
                rek[f'D_{r}_cos'] = cos(D, d) if e else None
                rek[f'D_{r}_cos_wid'] = cos(D @ W2.T, dw) if e else None
                rek[f'D_{r}_wzgl_ruchu_C0'] = float(np.linalg.norm(D) / (np.linalg.norm(Z['C0'][e] - Z['C0'][0]) + 1e-12))
            rek['cos_D_A_D_R'] = cos(Z['A150'][e] - Z['C0'][e], Z['R150'][e] - Z['C0'][e]) if e else None
            wiersze.append(rek)

    def wart(klucz, krok=None, srednio=False):
        out = []
        for s in ZIARNA:
            w = [x for x in wiersze if x['ziarno'] == s]
            if srednio:
                out.append(float(np.mean([x[klucz] for x in w[1:]])))
            else:
                out.append(next(x[klucz] for x in w if x['krok'] == (krok or kroki[-1])))
        return out

    klucze = ['postep_C0', 'postep_A150', 'postep_R150', 'postep_wid_C0', 'postep_wid_A150', 'postep_wid_R150',
              'D_A150_cos', 'D_R150_cos', 'D_A150_cos_wid', 'D_R150_cos_wid', 'D_A150_postep', 'D_R150_postep',
              'D_A150_norma', 'D_R150_norma', 'cel_pewnosc_C0', 'cel_pewnosc_A150', 'cel_pewnosc_R150',
              'cel_iou_C0', 'cel_iou_A150', 'cel_iou_R150', 'mAP50_wal3000_C0', 'mAP50_wal3000_A150', 'mAP50_wal3000_R150',
              'nowe150_iou_A150', 'nowe150_iou_C0', 'losowe150_iou_R150', 'losowe150_iou_C0', 'cos_D_A_D_R']
    zest = dict(koniec={k: ms(wart(k)) for k in klucze}, srednio_po_checkpointach={k: ms(wart(k, srednio=True)) for k in klucze})
    para = {}                                                          # różnice parami w obrębie ziarna, A150 − R150
    for k in ('postep', 'postep_wid'):
        para[f'{k}_A_minus_R_koniec'] = ms(np.array(wart(f'{k}_A150')) - np.array(wart(f'{k}_R150')))
        para[f'{k}_A_minus_R_srednio'] = ms(np.array(wart(f'{k}_A150', srednio=True)) - np.array(wart(f'{k}_R150', srednio=True)))
    para['cos_A_minus_R_srednio'] = ms(np.array(wart('D_A150_cos', srednio=True)) - np.array(wart('D_R150_cos', srednio=True)))
    out = dict(kroki=kroki, ziarna=ZIARNA, powtarzalnosc_C0_s47_vs_1824868=powt, roznice_startow=starty,
               delta_norma=float(np.sqrt(dd)), zestawienie=zest, roznice_parami_A150_R150=para, wiersze=wiersze)
    (W / 'ANALIZA_ZIARNA.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')

    x = np.array(kroki)
    kol = dict(C0='tab:gray', A150='tab:red', R150='tab:green')

    def pas(a, klucz_f, r, etyk):
        Y = np.array([[next(w[klucz_f(r)] for w in wiersze if w['ziarno'] == s and w['krok'] == k) for k in kroki] for s in ZIARNA],
                     dtype=float)
        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1)
        for y in Y:
            a.plot(x, y, '-', c=kol[r], lw=.6, alpha=.4)
        a.plot(x, mu, 'o-', c=kol[r], lw=2, label=etyk); a.fill_between(x, mu - sd, mu + sd, color=kol[r], alpha=.15)

    fig, ax = plt.subplots(2, 3, figsize=(19, 10), layout='constrained')
    for r in RAMIONA:
        pas(ax[0, 0], lambda r: f'postep_{r}', r, r)
        pas(ax[0, 1], lambda r: f'postep_wid_{r}', r, r)
        pas(ax[1, 1], lambda r: f'mAP50_wal3000_{r}', r, r)
        pas(ax[1, 2], lambda r: f'cel_iou_{r}', r, f'{r} IoU celu')
    for r in ('A150', 'R150'):
        Yc = np.array([[w[f'D_{r}_cos'] for w in wiersze if w['ziarno'] == s][1:] for s in ZIARNA], dtype=float)
        for y in Yc:
            ax[0, 2].plot(x[1:], y, '-', c=kol[r], lw=.6, alpha=.4)
        ax[0, 2].plot(x[1:], Yc.mean(0), 'o-', c=kol[r], lw=2, label=f'cos({r} − C0, δ)')
        Yn = np.array([[w[f'D_{r}_norma'] for w in wiersze if w['ziarno'] == s] for s in ZIARNA], dtype=float)
        ax[1, 0].plot(x, Yn.mean(0), 'o-', c=kol[r], lw=2, label=f'||z_{r} − z_C0||')
        ax[1, 0].fill_between(x, Yn.mean(0) - Yn.std(0, ddof=1), Yn.mean(0) + Yn.std(0, ddof=1), color=kol[r], alpha=.15)
    for a, t in zip(ax.ravel(), ['postęp latentu celu wzdłuż δ = z*_K1000 − z0 [ułamek]', 'postęp w części widocznej dla fc2',
                                 'kierunek różnicy ramion względem C0', 'norma różnicy latentu celu względem C0',
                                 'mAP@0,5 na 3000 klatkach walidacji', 'IoU najpewniejszej ramki na zdjęciu celu']):
        a.set_title(t, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel('krok uczenia')
    ax[0, 0].axhline(0, c='k', lw=.6); ax[0, 1].axhline(0, c='k', lw=.6); ax[0, 2].axhline(0, c='k', lw=.6)
    fig.suptitle('Od K1000, 5 epok, 3 ziarna: C0 (oryginał), A150 (150 klatek z SD + DINO), R150 (150 losowych klatek); '
                 'linie cienkie = ziarna, gruba = średnia, pas = ±1 odch.', fontsize=12)
    fig.savefig(W / 'WYKRES_ZIARNA.png', dpi=110); plt.close(fig)
    print('powtarzalność', powt)
    for k in ('postep_C0', 'postep_A150', 'postep_R150', 'postep_wid_A150', 'postep_wid_R150', 'D_A150_cos', 'D_R150_cos',
              'D_A150_norma', 'D_R150_norma', 'mAP50_wal3000_C0', 'mAP50_wal3000_A150', 'mAP50_wal3000_R150',
              'cel_iou_A150', 'cel_iou_R150', 'nowe150_iou_A150', 'losowe150_iou_R150', 'cos_D_A_D_R'):
        a, b = zest['koniec'][k], zest['srednio_po_checkpointach'][k]
        print(f'{k:22s} koniec {a["srednia"]:+.4f} ± {a["odch"]:.4f}   średnio {b["srednia"]:+.4f} ± {b["odch"]:.4f}')
    print(json.dumps(para, ensure_ascii=False))


if __name__ == '__main__':
    main()
