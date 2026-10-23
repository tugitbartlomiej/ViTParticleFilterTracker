"""Płaszczyzna krok × podkrok dla seedu 3047: 46 checkpointów × 10 klatek TEST, obserwable jak w eksperymencie
kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29 (pomiar_kroki_podkroki.py), liczone wprost z przejścia (bez zapisu przejść), w 192D.
Różnice: udział translacji liczony w pełnych 192D na surowych stanach (|średnia Δ|² / średnia |Δ|²); etykieta narzędzia = cystotom,
udział ≥ 0,25; dodatkowo „gwiazda”: udział trzech tokenów o największej odbieranej uwadze leżących w obrysie lub ≤ 1 łatę od niego.
Wynik: <WY>/klatki.json, <WY>/plaszczyzna_3047.npz (tablice (K, F, …)).
Uruchomienie: python pomiar_3047.py <katalog Wyniki/Dane>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wspolne as WS  # noqa: E402
from wspolne import C, H, T, W  # noqa: E402

WY = Path(sys.argv[1]); WY.mkdir(parents=True, exist_ok=True)
PROG_Z = 0.05
idx = np.arange(T).reshape(H, W)
kraw = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1), np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])
r, c = np.divmod(np.arange(T), W)
dist_siatki = np.sqrt((r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2)
sas8 = (np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1); np.fill_diagonal(sas8, False)
rng = np.random.default_rng(0)
perm = rng.permutation(T); FALDY = ((perm[:300], perm[300:]), (perm[300:], perm[:300])); PERM_NULL = rng.permutation(T)


def r2_cv(Xp, Dl):
    out = []
    for tr, te in FALDY:
        Zm = np.c_[Xp[tr], np.ones(len(tr))]
        coef, *_ = np.linalg.lstsq(Zm, Dl[tr], rcond=None)
        pred = np.c_[Xp[te], np.ones(len(te))] @ coef
        out.append(1 - ((Dl[te] - pred) ** 2).sum() / ((Dl[te] - Dl[te].mean(0)) ** 2).sum())
    return float(np.mean(out))


def fisher(X, narz, tlo):
    a, b = X[narz], X[tlo]
    d = a.mean(0) - b.mean(0)
    sw = (a.var(0).sum() * len(a) + b.var(0).sum() * len(b)) / (len(a) + len(b))
    return float(d @ d / sw)


def dirichlet(X):
    en = ((X[kraw[:, 0]] - X[kraw[:, 1]]) ** 2).sum(1).mean()
    return float(en / (2 * ((X - X.mean(0)) ** 2).sum(1).mean()))


def jedn(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cosy(A, B):
    return float(np.median((A * B).sum(1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)))


def pomiar(P, udzial, gt):
    narz = udzial >= WS.PROG_NARZ; tlo = udzial == 0; dotk = udzial > 0
    halo = (dist_siatki[:, dotk].min(1) <= 1.0) if dotk.any() else np.zeros(T, bool)
    X = P['X'].float().cpu().numpy().astype(np.float64)                        # (13, 600, 192) surowe
    Xc = X - X.mean(1, keepdims=True); rms = np.linalg.norm(Xc, axis=2).mean(1)
    dX = Xc[1:] - Xc[:-1]; dR = X[1:] - X[:-1]
    w = dict(pewnosc=P['pewnosc'], iou=WS.iou(P['ramka'], gt), rms=rms, stos=rms[1:] / rms[:-1])
    w['dl'] = np.median(np.linalg.norm(dX, axis=2), 1) / rms[:-1]
    w['rad'] = np.array([cosy(dX[s], Xc[s]) for s in range(12)])
    w['r2cv'] = np.array([r2_cv(Xc[s], dX[s]) for s in range(12)])
    w['r2cv_null'] = np.array([r2_cv(Xc[s], dX[s][PERM_NULL]) for s in range(12)])
    w['trans'] = (dR.mean(1) ** 2).sum(1) / (dR ** 2).sum(2).mean(1)
    en = (dX ** 2).sum(2)
    bal = lambda sel: float(np.median((dX[sel].sum(0) ** 2).sum(1) / np.maximum(en[sel].sum(0), 1e-12)))
    w['bal'] = np.array([bal(slice(None)), bal(slice(0, 12, 2)), bal(slice(1, 12, 2))])
    w['cos_am'] = np.array([cosy(dX[2 * b], dX[2 * b + 1]) for b in range(6)])
    w['cos_ma'] = np.array([cosy(dX[2 * b + 1], dX[2 * b + 2]) for b in range(5)])
    w['cos_mm'] = np.array([cosy(dX[2 * b + 1], dX[2 * b + 3]) for b in range(5)])
    w['cos_aa'] = np.array([cosy(dX[2 * b], dX[2 * b + 2]) for b in range(5)])
    A = P['A'].float().cpu().numpy()                                           # (6, 3, 600, 600)
    w['att_neff'] = np.median(1 / np.maximum((A ** 2).sum(3), 1e-12), 2)
    w['att_self'] = np.median(np.diagonal(A, axis1=2, axis2=3), 2)
    w['att_sas8'] = np.median((A * sas8).sum(3), 2)
    w['att_dist'] = np.median((A * dist_siatki).sum(3), 2)
    w['att_ent'] = np.median(-(A * np.log(np.maximum(A, 1e-12))).sum(3), 2)
    pn = max(narz.mean(), 1e-9)
    w['att_bg2tool'] = A[:, :, tlo][:, :, :, narz].sum(3).mean(2) / pn
    w['att_tool2tool'] = A[:, :, narz][:, :, :, narz].sum(3).mean(2) / pn
    odb = A.mean((1, 2))                                                       # (6, 600) odbierana uwaga, średnia po zapytaniach i głowach
    top3 = np.argsort(-odb, axis=1)[:, :3]
    w['hub_top3'] = (dotk | halo)[top3].mean(1)                                # (6,) udział 3 największych odbiorców w obrysie ∪ halo
    w['hub_top3_narz'] = narz[top3].mean(1)
    z = P['Z'].float().cpu().numpy(); az = np.abs(z)
    w['mlp_lit'] = np.median((z > PROG_Z).mean(2), 1)
    w['mlp_neff'] = np.median(az.sum(2) ** 2 / np.maximum((z ** 2).sum(2), 1e-12), 1)
    top = -np.partition(-az, 10, axis=2)[:, :, :10].sum(2)
    w['mlp_top10'] = np.median(top / np.maximum(az.sum(2), 1e-12), 1)
    U = [jedn(Xc[s]) for s in range(13)]
    w['fisher'] = np.array([fisher(Xc[s], narz, tlo) for s in range(13)]); w['fisher_u'] = np.array([fisher(U[s], narz, tlo) for s in range(13)])
    w['dir'] = np.array([dirichlet(Xc[s]) for s in range(13)]); w['dir_u'] = np.array([dirichlet(U[s]) for s in range(13)])
    return w


if (WY / 'klatki.json').exists():
    KL = json.loads((WY / 'klatki.json').read_text(encoding='utf-8'))
else:
    KL = WS.wybierz_klatki()
    for k_ in KL:
        _, k_['sha256_jpg'] = WS.obraz(k_['film'], k_['klatka'])
    (WY / 'klatki.json').write_text(json.dumps(KL, ensure_ascii=False, indent=1), encoding='utf-8')
print('klatek', len(KL), [(k_['film'], k_['klatka'], k_['lat_narz']) for k_ in KL], flush=True)
OBRAZY, UDZIAL = [], []
for k_ in KL:
    rgb, sha = WS.obraz(k_['film'], k_['klatka']); assert sha == k_['sha256_jpg']
    OBRAZY.append(rgb); UDZIAL.append(WS.etykieta_cystotomu(k_['film'], k_['klatka'])['udzial'])
UDZIAL = np.stack(UDZIAL)
KROKI = WS.checkpointy(); net = WS.siec()
wyniki = []; t0 = time.time()
for a_, krok in enumerate(KROKI):
    WS.wczytaj(net, krok)
    wiersz = [pomiar(WS.przejscie(net, OBRAZY[f]), UDZIAL[f], KL[f]['gt']) for f in range(len(KL))]
    wyniki.append(wiersz)
    print(f'{a_ + 1}/{len(KROKI)} krok {krok} ({krok / WS.NA_EPOKE:.1f} ep): IoU {np.mean([w["iou"] for w in wiersz]):.3f}, '
          f'źródła b4 {np.median([w["att_neff"][3].mean() for w in wiersz]):.0f}, Fisher m6 {np.mean([w["fisher"][12] for w in wiersz]):.2f}, '
          f'{time.time() - t0:.0f} s', flush=True)
klucze = list(wyniki[0][0].keys())
dane = {k: np.array([[np.asarray(w[k]) for w in wiersz] for wiersz in wyniki]) for k in klucze}
dane.update(krok=np.array(KROKI), epoka=np.array(KROKI) / WS.NA_EPOKE, udzial=UDZIAL, prog_narz=WS.PROG_NARZ)
np.savez_compressed(WY / 'plaszczyzna_3047.npz', **dane)
print('zapisano', WY / 'plaszczyzna_3047.npz', f'{time.time() - t0:.0f} s', flush=True)
