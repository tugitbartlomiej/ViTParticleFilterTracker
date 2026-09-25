#!/usr/bin/env python3
"""Koopman ze stanem z pamiecia [z_t, v_t], osobno dla kazdej pary obraz-token (plan: PLAN.md).

Kazda jednostka (obraz, token) ma wlasne parametry; obliczenia partiami po obrazie nie lacza jednostek.
float64, bez PCA. Wynik: wyniki/REPORT.json i tablice per jednostka.

    py -3.11 experiments/koopman_z_pamiecia/licz.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'koopman_gesta_historia/wyniki/macierze_wiele_obrazow'
OUT = HERE / 'wyniki'
STEPS = list(range(5, 370, 5))
KONF = [('alpha', None), ('ar2', None), ('diag', .1), ('diag', 1.), ('diag', 10.),
        ('vel', 1.), ('vel', 10.), ('vel', 100.), ('delay', .1), ('delay', 1.), ('delay', 10.), ('delay', 100.)]
NAZWY = [k if p is None else f'{k}_{p:g}' for k, p in KONF]


def dopasuj(Z, V, S):
    """Parametry wszystkich rodzin z danych do stanu S wlacznie (pary t=1..S-1: wejscie v_{t}=V[t-1], cel V[t])."""
    X = V[:, 0:S - 1]; Y = V[:, 1:S]                                          # U x N x D
    den = np.einsum('und,und->u', X, X); num = np.einsum('und,und->u', X, Y)
    alpha = num / np.maximum(den, 1e-30)
    Xp = V[:, 0:S - 2]; X2 = V[:, 1:S - 1]; Y2 = V[:, 2:S]                   # ar2
    a11 = np.einsum('und,und->u', X2, X2); a12 = np.einsum('und,und->u', X2, Xp); a22 = np.einsum('und,und->u', Xp, Xp)
    b1 = np.einsum('und,und->u', X2, Y2); b2 = np.einsum('und,und->u', Xp, Y2)
    M = np.stack([np.stack([a11, a12], -1), np.stack([a12, a22], -1)], -2) + 1e-12 * np.eye(2)
    ar = np.linalg.solve(M, np.stack([b1, b2], -1)[..., None])[..., 0]        # U x 2
    num_i = np.einsum('und,und->ud', X, Y); den_i = np.einsum('und,und->ud', X, X)
    den_m = den_i.mean(1, keepdims=True)
    par = dict(alpha=alpha, ar=ar, diag={}, vel={}, delay={})
    for _, lam in [k for k in KONF if k[0] == 'diag']:
        par['diag'][lam] = (num_i + lam * den_m * alpha[:, None]) / (den_i + lam * den_m)
    R = Y - alpha[:, None, None] * X
    G = X @ X.transpose(0, 2, 1); N = X.shape[1]; eye = np.eye(N)
    tr = np.trace(G, axis1=1, axis2=2) / N
    for _, rho in [k for k in KONF if k[0] == 'vel']:
        par['vel'][rho] = (X, np.linalg.solve(G + (rho * tr)[:, None, None] * eye, R))
    Zs = Z[:, 1:S]                                                             # z_t dla t=1..S-1
    zc = Zs - Zs.mean(1, keepdims=True); vc = X - X.mean(1, keepdims=True)
    w = np.sqrt(np.einsum('und,und->u', zc, zc) / np.maximum(np.einsum('und,und->u', vc, vc), 1e-30))
    Sx = np.concatenate([Zs, X * w[:, None, None]], 2)
    mx = Sx.mean(1, keepdims=True); my = Y.mean(1, keepdims=True); Sc = Sx - mx
    Gs = Sc @ Sc.transpose(0, 2, 1); trs = np.trace(Gs, axis1=1, axis2=2) / N
    for _, rho in [k for k in KONF if k[0] == 'delay']:
        par['delay'][rho] = (w, mx, my, Sc, np.linalg.solve(Gs + (rho * trs)[:, None, None] * eye, Y - my))
    return par


def predykcja(par, k, z, v, vp):
    """Przewidywana predkosc dla konfiguracji k; z, v, vp: U x Q x D."""
    rodz, p = KONF[k]
    if rodz == 'alpha':
        return par['alpha'][:, None, None] * v
    if rodz == 'ar2':
        return par['ar'][:, 0, None, None] * v + par['ar'][:, 1, None, None] * vp
    if rodz == 'diag':
        return par['diag'][p][:, None, :] * v
    if rodz == 'vel':
        X, C = par['vel'][p]
        return par['alpha'][:, None, None] * v + (v @ X.transpose(0, 2, 1)) @ C
    w, mx, my, Sc, C = par['delay'][p]
    s = np.concatenate([z, v * w[:, None, None]], 2)
    return my + ((s - mx) @ Sc.transpose(0, 2, 1)) @ C


def ocena(Z, V, par, k, poczatki):
    t = np.array(poczatki)
    z, v, vp = Z[:, t], V[:, t - 1], V[:, t - 2]
    vh = predykcja(par, k, z, v, vp)
    return vh, Z[:, t + 1] - z


def kroczace(V, poczatki, rodz):
    """Alpha / ar2 przeliczane przed kazdym poczatkiem t z par dostepnych do stanu t."""
    out = []
    for t in poczatki:
        X = V[:, 0:t - 1]; Y = V[:, 1:t]
        if rodz == 'alpha':
            a = np.einsum('und,und->u', X, Y) / np.maximum(np.einsum('und,und->u', X, X), 1e-30)
            out.append(a[:, None] * V[:, t - 1])
        else:
            Xp = V[:, 0:t - 2]; X2 = V[:, 1:t - 1]; Y2 = V[:, 2:t]
            a11 = np.einsum('und,und->u', X2, X2); a12 = np.einsum('und,und->u', X2, Xp); a22 = np.einsum('und,und->u', Xp, Xp)
            M = np.stack([np.stack([a11, a12], -1), np.stack([a12, a22], -1)], -2) + 1e-12 * np.eye(2)
            b = np.stack([np.einsum('und,und->u', X2, Y2), np.einsum('und,und->u', Xp, Y2)], -1)
            ar = np.linalg.solve(M, b[..., None])[..., 0]
            out.append(ar[:, 0, None] * V[:, t - 1] + ar[:, 1, None] * V[:, t - 2])
    return np.stack(out, 1)


def miary(vh, vt):
    mse = ((vh - vt) ** 2).mean(-1)                                            # U x Q
    cos = (vh * vt).sum(-1) / np.maximum(np.linalg.norm(vh, axis=-1) * np.linalg.norm(vt, axis=-1), 1e-30)
    return mse, cos


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((SRC / 'CONFIG.json').read_text())
    assert [r['step'] for r in cfg['weights']] == STEPS
    maska = np.load(SRC / 'MASK_WEIGHTS.npy')
    mm = [np.load(SRC / f'TOKENS_{s:07d}.npy', mmap_mode='r') for s in STEPS]
    TEST = list(range(48, 72)); KAL = list(range(40, 48)); ROLL = [48, 50, 52]
    metody = NAZWY + ['wybor_kalibracja', 'alpha_kroczace', 'ar2_kroczace', 'brak_zmian', 'ostatni_przyrost']
    M = len(metody)
    mse_all = np.zeros((72, 576, 24, M), np.float32); cos_all = np.zeros((72, 576, 24, M), np.float32)
    roll_all = np.zeros((72, 576, 3, M), np.float32); wybor = np.zeros((72, 576), np.int8)
    t0 = time.time()
    for img in range(72):
        Z = np.stack([m[img] for m in mm], 1).astype(np.float64)               # 576 x 73 x 384
        V = np.diff(Z, axis=1)
        # kalibracja: dopasowanie do stanu 40, poczatki 40..47
        pk = dopasuj(Z, V, 40)
        kal = np.stack([miary(*ocena(Z, V, pk, k, KAL))[0].mean(1) for k in range(len(KONF))], 1)
        wybor[img] = kal.argmin(1)
        # test: dopasowanie do stanu 48, zamrozone
        pt = dopasuj(Z, V, 48)
        vt = Z[:, np.array(TEST) + 1] - Z[:, TEST]
        prog = []
        for k in range(len(KONF)):
            vh, _ = ocena(Z, V, pt, k, TEST); prog.append(vh)
        prog = np.stack(prog, 0)                                                # K x U x Q x D
        wyb = prog[wybor[img], np.arange(576)]
        dod = [wyb, kroczace(V, TEST, 'alpha'), kroczace(V, TEST, 'ar2'), np.zeros_like(vt), V[:, np.array(TEST) - 1]]
        for j, vh in enumerate(list(prog) + dod):
            mse_all[img, :, :, j], cos_all[img, :, :, j] = miary(vh, vt)
        # dlugi horyzont: 20 checkpointow z poczatkow 48, 50, 52, model zamrozony
        for r, t in enumerate(ROLL):
            prawda = Z[:, t + 20]
            for j in range(M):
                if metody[j] in ('alpha_kroczace', 'ar2_kroczace'):
                    continue
                z = Z[:, t][:, None].copy(); v = V[:, t - 1][:, None].copy(); vp = V[:, t - 2][:, None].copy()
                for _ in range(20):
                    if metody[j] == 'brak_zmian':
                        vn = np.zeros_like(v)
                    elif metody[j] == 'ostatni_przyrost':
                        vn = v
                    else:
                        k = j if j < len(KONF) else None
                        if k is None:                                             # wybor per jednostka
                            vn = np.stack([predykcja(pt, kk, z, v, vp) for kk in range(len(KONF))], 0)[wybor[img], np.arange(576)]
                        else:
                            vn = predykcja(pt, k, z, v, vp)
                    z = z + vn; vp = v; v = vn
                roll_all[img, :, r, j] = ((z[:, 0] - prawda) ** 2).mean(-1)
        if img % 8 == 0:
            print(f'obraz {img}/72  {time.time() - t0:.0f} s', flush=True)

    def rmse(a):
        return float(np.sqrt(a.mean()))
    j_last, j_pers = metody.index('ostatni_przyrost'), metody.index('brak_zmian')
    unit_rmse = np.sqrt(mse_all.mean(2))                                      # 72 x 576 x M
    unit_cos = cos_all.mean(2)
    fg = maska >= .5
    raport = dict(jednostek=72 * 576, podzial=dict(kalibracja_dopasowanie_do_stanu=40, kalibracja_poczatki=KAL,
                                                  test_dopasowanie_do_stanu=48, test_poczatki=TEST, dlugi_horyzont=ROLL),
                  metody=metody, wyniki={})
    for j, m in enumerate(metody):
        brama = (unit_rmse[..., j] <= .95 * unit_rmse[..., j_last]) & (unit_rmse[..., j] <= .8 * unit_rmse[..., j_pers]) & (unit_cos[..., j] >= .5)
        raport['wyniki'][m] = dict(rmse_1krok=rmse(mse_all[..., j]), rmse_1krok_maska=rmse(mse_all[fg][..., j]),
                                   rmse_100aktualizacji=rmse(roll_all[..., j]) if m not in ('alpha_kroczace', 'ar2_kroczace') else None,
                                   sredni_cos=float(cos_all[..., j].mean()),
                                   przewaga_nad_ostatnim_proc=100 * (1 - rmse(mse_all[..., j]) / rmse(mse_all[..., j_last])),
                                   jednostek_przez_bramke=int(brama.sum()))
    raport['wybor_kalibracja_udzialy'] = {NAZWY[k]: int((wybor == k).sum()) for k in range(len(KONF))}
    (OUT / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    np.save(OUT / 'MSE_1KROK.npy', mse_all); np.save(OUT / 'COS_1KROK.npy', cos_all)
    np.save(OUT / 'MSE_ROLLOUT20.npy', roll_all); np.save(OUT / 'WYBOR.npy', wybor)
    sumy = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npy')}
    (OUT / 'SHA256.json').write_text(json.dumps(sumy, indent=1), encoding='utf-8')
    for m, r in raport['wyniki'].items():
        print(f"{m:18s} 1krok {r['rmse_1krok']:.6f}  maska {r['rmse_1krok_maska']:.6f}  100akt "
              f"{r['rmse_100aktualizacji'] if r['rmse_100aktualizacji'] is None else round(r['rmse_100aktualizacji'], 5)}"
              f"  cos {r['sredni_cos']:.3f}  vs_ostatni {r['przewaga_nad_ostatnim_proc']:+.2f}%  bramka {r['jednostek_przez_bramke']}")
    print(raport['wybor_kalibracja_udzialy'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
