"""Heterogeniczność dynamiczna (dh.analiza, ruchliwość względna, układ CR i lab) w trzech przestrzeniach z tych samych pełnych
latentów (jak kontrola_rzutu.py): pełne 768D, losowy rzut 128D, PCA32 z kroku 0. Sprawdza, czy korelacja ruchliwości sąsiadów
(~0,5 w PCA32) nie jest artefaktem rzutu. Wynik: KONTROLA_RZUTU_DH.json. Uruchomienie: python kontrola_rzutu_dh.py [NS_P] [OD] [DO]"""
import glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import odciski as O, dh
G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
NS_P = int(sys.argv[1]) if len(sys.argv) > 1 else 3; OD = int(sys.argv[2]) if len(sys.argv) > 2 else 300; DO = int(sys.argv[3]) if len(sys.argv) > 3 else 800
TOK = 600
baza = np.load(os.path.join(G, 'BAZA_PCA32.npz')); mean, comp = baza['mean'].astype(np.float32), baza['comp'].astype(np.float32)
R = (np.random.default_rng(0).standard_normal((768, 128)) / np.sqrt(128)).astype(np.float32)
Z, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pelne768_*.npz'))):
    z = np.load(p); k = z['kroki']; m = (k >= OD) & (k <= DO)
    if m.any():
        Z.append(np.ascontiguousarray(z['latent'][m][:, :NS_P].astype(np.float32)).reshape(int(m.sum()), NS_P * TOK, 768)); kroki.append(k[m])
    del z
Z = np.concatenate(Z); kroki = np.concatenate(kroki).astype(int)
dh.log('pełne latenty', Z.shape)
wyn = {}
taus = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 400]
for nazwa, X in (('pelne768', Z), ('rp128', Z @ R), ('pca32', (Z - mean) @ comp.T)):
    ind = {int(k): i for i, k in enumerate(kroki)}
    N6, dnn, dsh = O.knn_w_blokach(X[ind[OD]], TOK, 6); N6x = dh.knn_poza_blokiem(X[ind[OD]], TOK, 6)
    wyn[nazwa + ' lab'] = dh.analiza(X, kroki, (OD, DO), taus, N6, TOK, nazwa, pole='blok', N6_cross=N6x, max_pocz=60)
    P1 = dh.klatka_cr(X, kroki, OD, N6)
    wyn[nazwa + ' CR'] = dh.analiza(P1, kroki, (OD, DO), taus, N6, TOK, nazwa, pole='blok', N6_cross=N6x, max_pocz=60)
    del P1
json.dump(dict(opis=__doc__, NS_P=NS_P, okno=[OD, DO], wyniki=wyn), open(os.path.join(HERE, 'KONTROLA_RZUTU_DH.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
dh.log('zapisano KONTROLA_RZUTU_DH.json')
