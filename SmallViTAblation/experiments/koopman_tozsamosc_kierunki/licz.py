#!/usr/bin/env python3
"""Czy elementy latentu, ktore identyfikuja narzedzie, sa przewidywalne Koopmanem podczas uczenia? (2026-09-25)

Dane: koopman_gesta_historia/wyniki/macierze_wiele_obrazow (72 obrazy sond x 576 tokenow x 384, K checkpointow
co 5 aktualizacji), MASK_WEIGHTS (udzial narzedzia w tokenie), role fit/holdout z CONFIG.json.

Podzial czasowy jak w koopman_z_pamiecia: dopasowanie do stanu S_FIT (48 = krok 245), test przejscia S_FIT..K-2.
Kierunki tozsamosci liczone WYLACZNIE na obrazach fit i checkpointach <= S_FIT.

Czesc A. Kierunki tozsamosci narzedzia (pelne tokeny, bez PCA calego latentu):
   w_lda   kierunek LDA narzedzie (maska >= .5) wobec tla (maska == 0), wybielony kowariancja wewnatrz klas
   S_k     k glownych kierunkow roznicy "token narzedzia minus srednie tlo obrazu" (k = 1, 4, 8, 16, 32)
   drift   cos(w_lda(t), w_lda(0)): czy kierunek tozsamosci sam sie obraca w trakcie uczenia
Czesc B. Przewidywalnosc per jednostka (obraz, token narzedzia) w podprzestrzeni S_k wobec dopelnienia i losowej
   podprzestrzeni tego samego wymiaru: modele alpha, ar2, vel (pelna macierz na predkosci), delay (polozenie +
   predkosc), kontrole brak_zmian i ostatni_przyrost. RMSE znormalizowane przez RMS prawdziwego przyrostu
   w tej samej podprzestrzeni (1 = tak zle jak brak zmian).
Czesc C. Poziom obrazu: opis narzedzia m_i(t) = srednia tokenow wazona maska (384), rzut na S_k, oraz skalar
   tozsamosci a_i(t) = w_lda . (m_i - tlo_i). Modele per obraz: alpha, relaksacja afiniczna 1-D
   a_{t+1} = rho a_t + c (Koopman z punktem stalym), oraz WSPOLNY Koopman miedzy obrazami:
   y_{t+1} = K y_t + b i wariant z pamiecia [y, dy], dopasowane na 48 obrazach fit, testowane na 24 holdout
   w oknie testowym (1 krok i 20 krokow).

    py -3.11 experiments/koopman_tozsamosc_kierunki/licz.py [--zrodlo DIR] [--s-fit 48]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KS = [1, 4, 8, 16, 32]


def rmse(a):
    return float(np.sqrt(np.mean(a)))


def wybierz_lda(T, B):
    """T: n x D tokeny narzedzia, B: m x D tokeny tla -> kierunek LDA (jednostkowy)."""
    Sw = np.cov(np.r_[T - T.mean(0), B - B.mean(0)].T) + 1e-4 * np.eye(T.shape[1])
    w = np.linalg.solve(Sw, T.mean(0) - B.mean(0))
    return w / np.linalg.norm(w)


def modele_jednostek(Y, s_fit, poczatki, ridge_vel=10., ridge_delay=10.):
    """Y: U x K x d trajektorie. Zwraca slownik metoda -> (U x Q x d) prognozy przyrostu na poczatkach."""
    V = np.diff(Y, axis=1)
    t = np.array(poczatki)
    z, v, vp = Y[:, t], V[:, t - 1], V[:, t - 2]
    X = V[:, 0:s_fit - 1]; Yv = V[:, 1:s_fit]                                  # pary predkosci do stanu s_fit
    out = {}
    den = np.einsum('und,und->u', X, X); alpha = np.einsum('und,und->u', X, Yv) / np.maximum(den, 1e-30)
    out['alpha'] = alpha[:, None, None] * v
    Xp = V[:, 0:s_fit - 2]; X2 = V[:, 1:s_fit - 1]; Y2 = V[:, 2:s_fit]
    a11 = np.einsum('und,und->u', X2, X2); a12 = np.einsum('und,und->u', X2, Xp); a22 = np.einsum('und,und->u', Xp, Xp)
    M = np.stack([np.stack([a11, a12], -1), np.stack([a12, a22], -1)], -2) + 1e-12 * np.eye(2)
    b = np.stack([np.einsum('und,und->u', X2, Y2), np.einsum('und,und->u', Xp, Y2)], -1)
    ar = np.linalg.solve(M, b[..., None])[..., 0]
    out['ar2'] = ar[:, 0, None, None] * v + ar[:, 1, None, None] * vp
    d = Y.shape[2]; N = X.shape[1]
    if d <= 64:                                                                # pelne macierze tylko w malym wymiarze
        # vel: v_{t+1} = A v_t + c, grzbiet
        Xc = X - X.mean(1, keepdims=True); Yc = Yv - Yv.mean(1, keepdims=True)
        G = Xc.transpose(0, 2, 1) @ Xc; tr = np.trace(G, axis1=1, axis2=2) / d
        A = np.linalg.solve(G + (ridge_vel * tr)[:, None, None] * np.eye(d), Xc.transpose(0, 2, 1) @ Yc)
        out['vel'] = Yv.mean(1, keepdims=True) + (v - X.mean(1, keepdims=True)) @ A
        # delay: v_{t+1} = f([z_t, v_t]) afinicznie, grzbiet (polozenie moze pomoc w niskim wymiarze)
        Zs = Y[:, 1:s_fit]
        S = np.concatenate([Zs, X], 2); Sc = S - S.mean(1, keepdims=True)
        Gs = Sc.transpose(0, 2, 1) @ Sc; trs = np.trace(Gs, axis1=1, axis2=2) / (2 * d)
        Ad = np.linalg.solve(Gs + (ridge_delay * trs)[:, None, None] * np.eye(2 * d), Sc.transpose(0, 2, 1) @ Yc)
        out['delay'] = Yv.mean(1, keepdims=True) + (np.concatenate([z, v], 2) - S.mean(1, keepdims=True)) @ Ad
        # pos: z_{t+1} = K z_t + b (klasyczny Koopman na polozeniu), grzbiet
        Zc = Zs - Zs.mean(1, keepdims=True); Dz = Yv - Yv.mean(1, keepdims=True)
        Gp = Zc.transpose(0, 2, 1) @ Zc; trp = np.trace(Gp, axis1=1, axis2=2) / d
        Ap = np.linalg.solve(Gp + (ridge_delay * trp)[:, None, None] * np.eye(d), Zc.transpose(0, 2, 1) @ Dz)
        out['pos'] = Yv.mean(1, keepdims=True) + (z - Zs.mean(1, keepdims=True)) @ Ap
    else:                                                                      # postac dualna N x N dla vel
        Xc = X - X.mean(1, keepdims=True); Yc = Yv - Yv.mean(1, keepdims=True)
        G = Xc @ Xc.transpose(0, 2, 1); tr = np.trace(G, axis1=1, axis2=2) / N
        C = np.linalg.solve(G + (ridge_vel * tr)[:, None, None] * np.eye(N), Yc)
        out['vel'] = Yv.mean(1, keepdims=True) + ((v - X.mean(1, keepdims=True)) @ Xc.transpose(0, 2, 1)) @ C
    out['brak_zmian'] = np.zeros_like(v); out['ostatni_przyrost'] = v
    prawda = Y[:, t + 1] - z
    return out, prawda


def ocena(out, prawda):
    ref = np.mean(prawda ** 2)
    r = {}
    for m, p in out.items():
        e = np.mean((p - prawda) ** 2)
        cos = np.sum(p * prawda, -1) / np.maximum(np.linalg.norm(p, axis=-1) * np.linalg.norm(prawda, axis=-1), 1e-30)
        r[m] = dict(rmse=float(np.sqrt(e)), rmse_wzgl_braku_zmian=float(np.sqrt(e / ref)), cos=float(np.nanmean(cos)))
    return r


def wspolny_koopman(Yf, Yh, s_fit, poczatki, ridge=1e-2):
    """Wspolny operator miedzy obrazami na stanie d-wym. Yf: nf x K x d (fit), Yh: nh x K x d (holdout)."""
    t = np.array(poczatki)
    def pary(Y):
        X = Y[:, :s_fit - 1]; Z = Y[:, 1:s_fit]; return X.reshape(-1, Y.shape[2]), Z.reshape(-1, Y.shape[2])
    X, Z = pary(Yf); d = X.shape[1]
    mx, mz = X.mean(0), Z.mean(0); Xc = X - mx
    G = Xc.T @ Xc; K = np.linalg.solve(G + ridge * np.trace(G) / d * np.eye(d), Xc.T @ (Z - mz))
    def krok_pos(y): return mz + (y - mx) @ K
    # z pamiecia: [y, dy] -> dy_next
    Vf = np.diff(Yf, axis=1); S = np.concatenate([Yf[:, 1:s_fit - 1], Vf[:, :s_fit - 2]], 2).reshape(-1, 2 * d)
    R = Vf[:, 1:s_fit - 1].reshape(-1, d); ms, mr = S.mean(0), R.mean(0); Sc = S - ms
    Gs = Sc.T @ Sc; Kd = np.linalg.solve(Gs + ridge * np.trace(Gs) / (2 * d) * np.eye(2 * d), Sc.T @ (R - mr))
    def krok_delay(y, v): return mr + (np.concatenate([y, v], -1) - ms) @ Kd
    # tylko predkosc (wspolna macierz A): dy_next = A dy + c
    Vx = Vf[:, :s_fit - 2].reshape(-1, d); Vy = Vf[:, 1:s_fit - 1].reshape(-1, d); mvx, mvy = Vx.mean(0), Vy.mean(0)
    Gv = (Vx - mvx).T @ (Vx - mvx); A = np.linalg.solve(Gv + ridge * np.trace(Gv) / d * np.eye(d), (Vx - mvx).T @ (Vy - mvy))
    def krok_vel(v): return mvy + (v - mvx) @ A
    wyn = {}
    for nazwa, Y in (('fit', Yf), ('holdout', Yh)):
        V = np.diff(Y, axis=1); z, v = Y[:, t], V[:, t - 1]; prawda = Y[:, t + 1] - z
        out = {'wspolny_pos': krok_pos(z) - z, 'wspolny_delay': krok_delay(z, v), 'wspolny_vel': krok_vel(v),
               'brak_zmian': np.zeros_like(v), 'ostatni_przyrost': v}
        r = ocena(out, prawda)
        # rollout 20 krokow z poczatku s_fit (bez prawdziwych stanow posrednich)
        H = 20; t0 = s_fit
        if t0 + H < Y.shape[1]:
            zz, vv = Y[:, t0].copy(), V[:, t0 - 1].copy(); zp = zz.copy(); zd, vd = zz.copy(), vv.copy(); zv, vvv = zz.copy(), vv.copy()
            zl = zz + H * vv
            for _ in range(H):
                zp = krok_pos(zp); dv = krok_delay(zd, vd); zd = zd + dv; vd = dv; dvv = krok_vel(vvv); zv = zv + dvv; vvv = dvv
            cel = Y[:, t0 + H]
            for m, p in (('wspolny_pos', zp), ('wspolny_delay', zd), ('wspolny_vel', zv), ('brak_zmian', zz), ('ostatni_przyrost', zl)):
                r[m]['rmse_20krokow'] = float(np.sqrt(np.mean((p - cel) ** 2)))
                r[m]['rmse_20krokow_wzgl_braku_zmian'] = float(np.sqrt(np.mean((p - cel) ** 2) / np.mean((zz - cel) ** 2)))
        wyn[nazwa] = r
    ev = np.linalg.eigvals(K)
    wyn['widmo_wspolny_pos'] = dict(max_abs=float(np.abs(ev).max()), n_powyzej_0_95=int((np.abs(ev) > .95).sum()), d=int(d))
    return wyn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zrodlo', type=Path, default=HERE.parent / 'koopman_gesta_historia/wyniki/macierze_wiele_obrazow')
    ap.add_argument('--s-fit', type=int, default=48)
    ap.add_argument('--out', type=Path, default=HERE / 'wyniki')
    ap.add_argument('--max-tokenow', type=int, default=0, help='0 = wszystkie tokeny narzedzia')
    args = ap.parse_args(argv)
    args.out.mkdir(exist_ok=True)
    t0 = time.time()
    cfg = json.loads((args.zrodlo / 'CONFIG.json').read_text(encoding='utf-8'))
    steps = [w['step'] for w in cfg['weights']]
    K = len(steps); S = args.s_fit
    assert S + 2 < K, 'za malo checkpointow'
    rows = cfg['rows']; fit = np.array([r['role'] == 'fit' for r in rows]); hold = ~fit
    wariant = np.array([r['variant'] for r in rows]); zrodlo = np.array([r['image'].split('/')[-1].split('_')[0] for r in rows])
    maska = np.load(args.zrodlo / 'MASK_WEIGHTS.npy')                                   # 72 x 576
    Z = np.empty((72, 576, K, 384), np.float32)
    for k, s in enumerate(steps):
        Z[:, :, k, :] = np.load(args.zrodlo / f'TOKENS_{s:07d}.npy', mmap_mode='r')
    print(f'wczytane {K} checkpointow (kroki {steps[0]}..{steps[-1]}), {time.time() - t0:.0f} s', flush=True)
    narz = maska >= .5; tlo = maska == 0
    poczatki = list(range(S, K - 1))
    print(f'tokenow narzedzia {narz.sum()}, tla {tlo.sum()}; test: {len(poczatki)} przejsc od stanu {S} (krok {steps[S]})', flush=True)

    # ---------- A. kierunki tozsamosci z obrazow fit i checkpointow <= S
    okno = list(range(0, S + 1, 4))
    Tf = np.concatenate([Z[i, narz[i]][:, okno].reshape(-1, 384) for i in np.flatnonzero(fit)]).astype(np.float64)
    Bf = np.concatenate([Z[i, tlo[i]][:, okno].reshape(-1, 384) for i in np.flatnonzero(fit)]).astype(np.float64)
    rng = np.random.default_rng(0)
    Bf = Bf[rng.choice(len(Bf), min(len(Bf), len(Tf)), replace=False)]
    w_lda = wybierz_lda(Tf, Bf)
    # roznica token narzedzia minus srednie tlo tego obrazu i checkpointu -> PCA
    D = []
    for i in np.flatnonzero(fit):
        bg = Z[i, tlo[i]].mean(0)                                                          # K x 384
        D.append((Z[i, narz[i]][:, okno] - bg[okno][None]).reshape(-1, 384))
    D = np.concatenate(D).astype(np.float64); D -= D.mean(0)
    _, sv, vt = np.linalg.svd(D, full_matrices=False)
    udzial = (sv ** 2) / (sv ** 2).sum()
    Q_all = vt.T                                                                            # 384 x 384, kolumny = kierunki
    # dryf kierunku tozsamosci w czasie (LDA per checkpoint na obrazach fit)
    dryf = []
    for k in range(K):
        T_k = np.concatenate([Z[i, narz[i], k] for i in np.flatnonzero(fit)]).astype(np.float64)
        B_k = np.concatenate([Z[i, tlo[i], k] for i in np.flatnonzero(fit)]).astype(np.float64)
        B_k = B_k[rng.choice(len(B_k), min(len(B_k), len(T_k)), replace=False)]
        dryf.append(wybierz_lda(T_k, B_k))
    dryf = np.array(dryf); cos_dryf = dryf @ dryf[0]
    # AUC LDA na holdout (ostatni checkpoint okna i ostatni checkpoint testu)
    def auc(p, n):
        x = np.r_[p, n]; r = np.argsort(np.argsort(x)) + 1
        return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
    auc_h = {}
    for k in (0, S, K - 1):
        Th = np.concatenate([Z[i, narz[i], k] for i in np.flatnonzero(hold)]) @ w_lda
        Bh = np.concatenate([Z[i, tlo[i], k] for i in np.flatnonzero(hold)]) @ w_lda
        auc_h[f'krok_{steps[k]}'] = auc(Th, Bh)
    raportA = dict(udzial_wariancji_top32=[float(x) for x in udzial[:32]], udzial_skumulowany={k: float(udzial[:k].sum()) for k in KS},
                   cos_w_lda_z_pca={k: float(np.linalg.norm(Q_all[:, :k].T @ w_lda)) for k in KS},
                   auc_lda_holdout=auc_h, cos_dryfu_w_lda=[float(x) for x in cos_dryf])
    print('A:', json.dumps({k: v for k, v in raportA.items() if k != 'cos_dryfu_w_lda'}, indent=1), flush=True)
    print('  dryf w_lda (cos z krokiem 0):', np.round(cos_dryf[::8], 3).tolist(), flush=True)

    # ---------- B. przewidywalnosc per token narzedzia w podprzestrzeniach
    idx = [(i, p) for i in range(72) for p in np.flatnonzero(narz[i])]
    if args.max_tokenow and len(idx) > args.max_tokenow:
        idx = [idx[j] for j in rng.choice(len(idx), args.max_tokenow, replace=False)]
    Yfull = np.stack([Z[i, p] for i, p in idx]).astype(np.float64)                         # U x K x 384
    Yfull -= Yfull[:, :S].mean(1, keepdims=True)
    hold_u = np.array([hold[i] for i, _ in idx])
    # poziom obrazu (czesc C) liczony z Z juz tutaj, zeby zwolnic tokeny (10 GB dla K=160) przed czescia B
    M = np.stack([(Z[i] * maska[i][:, None, None]).sum(0) / maska[i].sum() for i in range(72)]).astype(np.float64)  # 72 x K x 384
    BG = np.stack([Z[i, tlo[i]].mean(0) for i in range(72)]).astype(np.float64)
    del Z
    raportB = {}
    def wariancja_przyrostu(Y):
        V = np.diff(Y[:, S:], axis=1); return float(np.mean(V ** 2) * Y.shape[2])
    for k in KS:
        Q = Q_all[:, :k]; Qc = Q_all[:, k:]
        Qr = np.linalg.qr(rng.standard_normal((384, k)))[0]
        for nazwa, B in (('tozsamosc', Q), ('losowa', Qr), ('dopelnienie', Qc)):
            Y = Yfull @ B
            out, prawda = modele_jednostek(Y, S, poczatki)
            r_all = ocena(out, prawda); r_h = ocena({m: p[hold_u] for m, p in out.items()}, prawda[hold_u])
            raportB[f'k{k}_{nazwa}'] = dict(wymiar=int(B.shape[1]), energia_przyrostu_test=wariancja_przyrostu(Y),
                                           wszystkie=r_all, holdout=r_h)
        print(f'B k={k}:', {n: {m: round(raportB[f"k{k}_{n}"]["holdout"][m]["rmse_wzgl_braku_zmian"], 3)
                                for m in raportB[f"k{k}_{n}"]["holdout"]} for n in ('tozsamosc', 'losowa', 'dopelnienie')}, flush=True)
    # pelny wektor dla odniesienia
    out, prawda = modele_jednostek(Yfull, S, poczatki)
    raportB['pelny_384'] = dict(wymiar=384, energia_przyrostu_test=wariancja_przyrostu(Yfull), wszystkie=ocena(out, prawda),
                                holdout=ocena({m: p[hold_u] for m, p in out.items()}, prawda[hold_u]))

    # ---------- C. poziom obrazu
    # M (opis narzedzia) i BG (tlo) policzone przed czescia B, patrz wyzej
    a = (M - BG) @ w_lda                                                                     # 72 x K skalar tozsamosci
    raportC = dict(skalar_tozsamosci_srednia_po_obrazach=[float(x) for x in a.mean(0)],
                   skalar_tozsamosci_przyrost_sredni=[float(x) for x in np.diff(a, axis=1).mean(0)])
    # per obraz: alpha na skalarze, relaksacja afiniczna a_{t+1} = rho a_t + c
    t = np.array(poczatki); va = np.diff(a, axis=1)
    X1 = a[:, :S - 1]; Y1 = a[:, 1:S]
    mx, my = X1.mean(1, keepdims=True), Y1.mean(1, keepdims=True)
    rho = ((X1 - mx) * (Y1 - my)).sum(1) / np.maximum(((X1 - mx) ** 2).sum(1), 1e-30); c = (my[:, 0] - rho * mx[:, 0])
    alpha = (va[:, :S - 2] * va[:, 1:S - 1]).sum(1) / np.maximum((va[:, :S - 2] ** 2).sum(1), 1e-30)
    prawda = a[:, t + 1] - a[:, t]
    outC = {'relaksacja_afiniczna': (rho[:, None] * a[:, t] + c[:, None]) - a[:, t], 'alpha': alpha[:, None] * va[:, t - 1],
            'brak_zmian': np.zeros_like(prawda), 'ostatni_przyrost': va[:, t - 1]}
    def ocena1(o, pr):
        return {m: dict(rmse=float(np.sqrt(np.mean((p - pr) ** 2))), rmse_wzgl_braku_zmian=float(np.sqrt(np.mean((p - pr) ** 2) / np.mean(pr ** 2)))) for m, p in o.items()}
    raportC['skalar_per_obraz_holdout'] = ocena1({m: p[hold] for m, p in outC.items()}, prawda[hold])
    raportC['skalar_per_obraz_wszystkie'] = ocena1(outC, prawda)
    raportC['rho_relaksacji_mediana'] = float(np.median(rho)); raportC['alpha_mediana'] = float(np.median(alpha))
    # wspolny Koopman miedzy obrazami na opisie narzedzia w S_k
    raportC['wspolny_koopman'] = {}
    for k in KS + [384]:
        Y = (M - M[:, :S].mean(1, keepdims=True)) @ (Q_all[:, :k] if k < 384 else np.eye(384))
        raportC['wspolny_koopman'][f'k{k}'] = wspolny_koopman(Y[fit], Y[hold], S, poczatki, ridge=1e-2 if k <= 32 else 1.)
        r = raportC['wspolny_koopman'][f'k{k}']['holdout']
        print(f'C wspolny k={k} holdout 1krok wzgl braku zmian:', {m: round(r[m]['rmse_wzgl_braku_zmian'], 3) for m in r},
              ' 20krokow:', {m: round(r[m].get('rmse_20krokow_wzgl_braku_zmian', float('nan')), 3) for m in r}, flush=True)
    # opis narzedzia per obraz w S_k (jednostki = obrazy)
    raportC['opis_per_obraz'] = {}
    for k in KS:
        Y = (M - M[:, :S].mean(1, keepdims=True)) @ Q_all[:, :k]
        out, prawda2 = modele_jednostek(Y, S, poczatki)
        raportC['opis_per_obraz'][f'k{k}'] = dict(holdout=ocena({m: p[hold] for m, p in out.items()}, prawda2[hold]), wszystkie=ocena(out, prawda2))
    print('C skalar per obraz holdout:', {m: round(v['rmse_wzgl_braku_zmian'], 3) for m, v in raportC['skalar_per_obraz_holdout'].items()}, flush=True)

    raport = dict(zrodlo=str(args.zrodlo), checkpointow=K, kroki=[steps[0], steps[-1]], s_fit=S, krok_s_fit=steps[S],
                  test_poczatki=[poczatki[0], poczatki[-1]], tokenow_narzedzia=len(idx), A=raportA, B=raportB, C=raportC,
                  sekundy=time.time() - t0)
    (args.out / f'REPORT_K{K}_S{S}.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.out / f'KIERUNKI_K{K}_S{S}.npz', w_lda=w_lda, Q=Q_all[:, :32], udzial=udzial, dryf=dryf, a=a, steps=np.array(steps))
    print(f'gotowe {time.time() - t0:.0f} s ->', args.out / f'REPORT_K{K}_S{S}.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
