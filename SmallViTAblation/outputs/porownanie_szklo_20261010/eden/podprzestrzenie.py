"""Podprzestrzeń stała (pozycyjna) i aktywna (treść) dla cząstek i dla wiązek. Hipoteza: przestrzeń reprezentacji = U_pos ⊕ U_akt,
gdzie U_pos to kierunki niosące adres patcha (szablon pozycji P_j = średnia tokenu j po zdjęciach), stabilne w treningu, a U_akt to reszta,
w której dzieje się ruch. Dla tokenów (rp128, 100 zdjęć) i dla kulek u (192D, 20 zdjęć) w wybranych krokach:
  k = wymiar U_pos (90 % energii szablonu), udział wariancji położeń w U_pos, stabilność U_pos (średni cos kątów głównych z U_pos z epoki 1);
  w etapach: udział energii drgań (x − średnia ±5 checkpointów) i dryfu centrum w U_pos, z normalizacją przez k/D (1 = bez preferencji).
Dla wiązek W (768×192): udział energii W i zmiany ΔW w etapie leżący w U_pos(u) (kolumny), znormalizowany przez k/192.
Uruchomienie: python podprzestrzenie.py <eksport ruch60> <wiazki_koloidy> <out>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

EKS, WZ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
EP = 1094
kroki = np.array(json.load(open(EKS / 'META.json', encoding='utf-8'))['kroki'])
rp = np.load(EKS / 'rp128.npy', mmap_mode='r')
U = np.load(WZ / 'u192_20zdjec.npy', mmap_mode='r')
Wf = np.load(WZ / 'W_fc1.npy', mmap_mode='r')
ti = lambda k: int(np.argmin(abs(kroki - k)))


def baza_pos(X):
    """X (n_zdjęć, 600, D) względne → ortonormalna baza U_pos (D, k) z szablonu, k przy 90 % energii szablonu."""
    P = X.mean(0)
    _, s, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    k = int(np.searchsorted(e, 0.9) + 1)
    return Vt[:k].T, k


def frac(E, B):
    """udział energii wektorów E (..., D) w podprzestrzeni B (D, k)."""
    E2 = E.reshape(-1, E.shape[-1]).astype(np.float64)
    return float(((E2 @ B) ** 2).sum() / (E2 ** 2).sum())


wyn = {}
for nazwa, arr, D in (('tokeny_rp128', rp, 128), ('kulki_u192', U, 192)):
    wz = lambda t: (lambda X: X - X.mean(1, keepdims=True))(np.asarray(arr[t]).astype(np.float32))
    B1, _ = baza_pos(wz(ti(EP)))
    pkt = []
    for k in (0, 100, 1094, 5470, 10940, 21880, 43760, 65640):
        X = wz(ti(k)); B, kk = baza_pos(X)
        cosy = np.linalg.svd(B.T @ B1, compute_uv=False)
        pkt.append(dict(krok=int(kroki[ti(k)]), k=kk, k_do_D=kk / D, udzial_wariancji_w_Upos=frac(X - X.mean(0, keepdims=True).mean(1, keepdims=True), B),
                        stabilnosc_cos_z_epoka1=float(cosy.mean()) if len(cosy) else None))
        log(f'{nazwa} krok {k}: k={kk} ({kk / D:.2f} wymiarów), wariancja w U_pos {pkt[-1]["udzial_wariancji_w_Upos"]:.2f}, cos z U_pos epoki 1 {pkt[-1]["stabilnosc_cos_z_epoka1"]:.2f}')
    etapy = {}
    for e1, e2 in ((1, 5), (10, 20), (30, 40), (50, 60)):
        idx = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
        Xs = np.stack([wz(i) for i in idx])                                   # (n, zdj, 600, D)
        B, kk = baza_pos(Xs.mean(0))
        c = np.stack([Xs[max(0, t - 5):t + 6].mean(0) for t in range(len(idx))])
        drg = Xs - c; dryf = c[-1] - c[0]
        fd, fr = frac(drg, B), frac(dryf, B)
        e = dict(k=kk, drgania_w_Upos=fd, drgania_wzbogacenie=fd / (kk / D), dryf_w_Upos=fr, dryf_wzbogacenie=fr / (kk / D))
        if nazwa == 'kulki_u192':
            W0, W1 = np.asarray(Wf[idx[0]]).astype(np.float32), np.asarray(Wf[idx[-1]]).astype(np.float32)
            fw, fdw = frac(W0, B), frac(W1 - W0, B)
            e.update(wiazki_W_w_Upos=fw, wiazki_W_wzbogacenie=fw / (kk / D), zmiana_wiazek_w_Upos=fdw, zmiana_wiazek_wzbogacenie=fdw / (kk / D))
        etapy[f'{e1}-{e2}'] = e
        log(f'{nazwa} epoki {e1}–{e2}: k={kk}; drgania w U_pos {fd:.2f} (×{e["drgania_wzbogacenie"]:.2f}), dryf {fr:.2f} (×{e["dryf_wzbogacenie"]:.2f})'
            + (f'; wiązki W w U_pos {e["wiazki_W_w_Upos"]:.2f} (×{e["wiazki_W_wzbogacenie"]:.2f}), zmiana wiązek {e["zmiana_wiazek_w_Upos"]:.2f} (×{e["zmiana_wiazek_wzbogacenie"]:.2f})' if 'wiazki_W_w_Upos' in e else ''))
        del Xs, c, drg
    wyn[nazwa] = dict(punkty=pkt, etapy=etapy)
json.dump(dict(opis=__doc__.split('\n')[0], wyniki=wyn), open(OUT / 'PODPRZESTRZENIE.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano')
