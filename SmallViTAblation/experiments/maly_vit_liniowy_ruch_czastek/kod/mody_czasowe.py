"""Etap 1: czy ruch wszystkich cząstek (tokenów) w uczeniu da się zapisać jako kombinację kilku WSPÓLNYCH modów
czasowych, a mody jako liniowy układ dynamiczny z wejściem lr(t)?

Dane: gęsta historia LATENT_GESTY.npy [1001 kroków, 12 klatek, 600 tokenów, 768 kanałów] (bieg
maly_vit_ruch_czastek_gesta_historia/pelny), kopie float16 klatek z app/cache_uczenia.

Model (dla jednej klatki, x_t = vec(Z_t) w R^(600*768)):
    x_t = m + sum_k a_k(t) C_k           (C_k: wektor współczynników każdej cząstki i kanału, wspólne mody a_k)
    a(t+1) = A a(t) + b lr(t) + c         (liniowy układ stanowy, wejście = lr z harmonogramu uczenia)
Uczciwość: mody i A, b, c szacowane tylko na krokach 0..T_UCZ (domyślnie 700); kroki T_UCZ+1..1000 to test.
Na teście: (1) jakość rzutu na mody z części uczącej (czy przestrzeń jest stała), (2) jazda swobodna układu
liniowego od a(T_UCZ) bez podglądania (prognoza), wobec persystencji (x_t = x_(T_UCZ)) i stałej prędkości.
Wynik na tokeny: R^2 odtworzenia trajektorii tokenu przez k modów, osobno narzędzie i tło.

Uruchomienie (z katalogu app aplikacji, żeby był moduł analiza_uczenia):
    python ../experiments/maly_vit_liniowy_ruch_czastek/kod/mody_czasowe.py --klatki 0 3 7 --k 1 2 3 5 10 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TU = Path(__file__).resolve().parent
APP = TU.parents[2] / 'app'
sys.path.insert(0, str(APP))
import analiza_uczenia as AU  # noqa: E402

LR, ROZGRZEWKA = 5e-4, 300


def lr_kroku(k):
    return LR * np.minimum(1.0, (np.asarray(k) + 1) / ROZGRZEWKA)


def gram_czasowy(Z, dev, t_ucz):
    """G = Xc Xc^T [T, T] dla x_t = vec(Z_t) wycentrowanego średnią z kroków 0..t_ucz; liczone kawałkami tokenów."""
    T = len(Z)
    G = torch.zeros((T, T), dtype=torch.float64, device=dev)
    sr = []
    for a in range(0, 600, 60):
        X = torch.from_numpy(np.asarray(Z[:, a:a + 60], np.float32)).to(dev).reshape(T, -1)
        m = X[:t_ucz + 1].mean(0, keepdim=True)
        sr.append(m)
        Xc = (X - m).double()
        G += Xc @ Xc.T
    return G, sr


def analiza_klatki(Z, narz, dev, t_ucz, K):
    T = len(Z)
    G, sr = gram_czasowy(Z, dev, t_ucz)
    # mody tylko z części uczącej: SVD bloku Gu = G[:tu, :tu]
    tu = t_ucz + 1
    ev, U = torch.linalg.eigh(G[:tu, :tu])
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    calk = float(ev.sum())
    wariancja = {int(k): float(ev[:k].sum() / calk) for k in K}
    # współczynniki C_k = U_k^T Xc_ucz / sqrt(ev_k);  a_k(t) dla wszystkich t = (Xc_t . C_k) / ||C_k||^2
    kmax = max(K)
    Uk = U[:, :kmax]                                                         # [tu, kmax]
    # a(t) dla t > t_ucz: rzut na C_k: a_k(t) = Xc_t . C_k / ||C_k||^2, C_k = Xc_ucz^T U_k  =>  a(t) = G[t, :tu] U_k / ev_k
    a = (G[:, :tu] @ Uk) / ev[:kmax].clamp_min(1e-30)                        # [T, kmax]; na części uczącej a = U_k
    # błąd odtworzenia każdego tokenu przez k modów: ||x_t - m - sum a_k C_k||^2, liczone kawałkami
    wynik_tok = {int(k): np.zeros((2, 600)) for k in K}                     # [ucz/test, token] suma kwadratów reszty
    calk_tok = np.zeros((2, 600))
    for j, a0 in enumerate(range(0, 600, 60)):
        X = torch.from_numpy(np.asarray(Z[:, a0:a0 + 60], np.float32)).to(dev).reshape(T, -1).double()
        Xc = X - sr[j].double()
        C = Uk.T @ Xc[:tu]                                                   # [kmax, n] (= sqrt(ev) * jednostkowe)
        for k in K:
            R = Xc - a[:, :k] @ C[:k]
            r2 = (R ** 2).reshape(T, 60, 768).sum(-1)                          # [T, 60]
            wynik_tok[int(k)][0, a0:a0 + 60] = r2[:tu].sum(0).cpu().numpy()
            wynik_tok[int(k)][1, a0:a0 + 60] = r2[tu:].sum(0).cpu().numpy()
        c2 = (Xc ** 2).reshape(T, 60, 768).sum(-1)
        calk_tok[0, a0:a0 + 60] = c2[:tu].sum(0).cpu().numpy(); calk_tok[1, a0:a0 + 60] = c2[tu:].sum(0).cpu().numpy()
    r2_tok = {}
    for k in K:
        r = 1 - wynik_tok[int(k)] / np.maximum(calk_tok, 1e-30)
        r2_tok[int(k)] = dict(ucz_narz=float(np.median(r[0, narz])), ucz_tlo=float(np.median(r[0, ~narz])),
                              test_narz=float(np.median(r[1, narz])), test_tlo=float(np.median(r[1, ~narz])),
                              test_calosc=float(1 - wynik_tok[int(k)][1].sum() / calk_tok[1].sum()))
    # układ liniowy na modach: a(t+1) = A a(t) + b lr(t) + c, dopasowany na 0..t_ucz-1
    a_np = a.cpu().numpy(); lr = lr_kroku(np.arange(T)) / LR
    uklad = {}
    for k in K:
        A_ = a_np[:, :k]
        X_ = np.concatenate([A_[:tu - 1], lr[:tu - 1, None], np.ones((tu - 1, 1))], 1)
        Y_ = A_[1:tu]
        W, *_ = np.linalg.lstsq(X_, Y_, rcond=None)
        # jeden krok na teście
        Xt = np.concatenate([A_[tu - 1:T - 1], lr[tu - 1:T - 1, None], np.ones((T - tu, 1))], 1)
        pred1 = Xt @ W
        e1 = ((pred1 - A_[tu:]) ** 2).sum(); e1_pers = ((A_[tu - 1:T - 1] - A_[tu:]) ** 2).sum()
        # jazda swobodna od a(t_ucz)
        s = A_[tu - 1].copy(); sw = []
        for t in range(tu - 1, T - 1):
            s = np.concatenate([s, [lr[t], 1.0]]) @ W; sw.append(s)
        sw = np.array(sw)
        ev_ = float(np.abs(np.linalg.eigvals(W[:k].T)).max())
        # błąd jazdy swobodnej w przestrzeni latentu = sum_k ev_k (a_k - â_k)^2 (C_k ortogonalne, ||C_k||^2 = ev_k)
        evk = ev[:k].cpu().numpy()
        blad_sw = float((((sw - A_[tu:]) ** 2) * evk).sum())
        blad_pers = float((((A_[tu - 1][None] - A_[tu:]) ** 2) * evk).sum())
        v = A_[tu - 1] - A_[tu - 2]
        pred_v = A_[tu - 1][None] + np.arange(1, T - tu + 1)[:, None] * v[None]
        blad_v = float((((pred_v - A_[tu:]) ** 2) * evk).sum())
        uklad[int(k)] = dict(jeden_krok_wzgl_persystencji=float(e1 / max(e1_pers, 1e-30)),
                             swobodna_wzgl_persystencji=blad_sw / max(blad_pers, 1e-30),
                             stala_predkosc_wzgl_persystencji=blad_v / max(blad_pers, 1e-30),
                             promien_widmowy_A=ev_)
    # przyrosty: ile wariancji kroków dx_t mieści się w k modach (czy drgania też są nisko-wymiarowe)
    Gd = G[1:tu, 1:tu] - G[1:tu, :tu - 1] - G[:tu - 1, 1:tu] + G[:tu - 1, :tu - 1]
    evd = torch.linalg.eigvalsh(Gd).flip(0).clamp_min(0)
    wariancja_przyrostow = {int(k): float(evd[:k].sum() / evd.sum()) for k in K}
    return dict(wariancja_modow_ucz=wariancja, wariancja_przyrostow_ucz=wariancja_przyrostow, r2_tokenow=r2_tok,
                uklad_liniowy=uklad), a_np[:, :kmax], ev[:kmax].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--klatki', type=int, nargs='+', default=[3])
    ap.add_argument('--k', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20])
    ap.add_argument('--t_ucz', type=int, default=700)
    ap.add_argument('--wyj', default=str(TU.parent / 'wyniki' / 'MODY_CZASOWE.json'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = AU.znajdz_biegi()[0]; o = AU.opis_biegu(b)
    wynik, mody = {}, {}
    for i in a.klatki:
        t0 = time.time()
        Z = AU.wczytaj_klatke(b, i)
        w, am, ev = analiza_klatki(Z, o['narzedzie'][i], dev, a.t_ucz, a.k)
        w['czas_s'] = round(time.time() - t0, 1); w['opis'] = o['opisy'][i]
        wynik[str(i)] = w; mody[f'a_{i}'] = am; mody[f'ev_{i}'] = ev
        print(i, json.dumps(w, ensure_ascii=False)[:1500], flush=True)
    # zgodność modów między klatkami: kąty główne między przestrzeniami 5 pierwszych modów (część ucząca)
    zg = {}
    kl = a.klatki
    for x in range(len(kl)):
        for y in range(x + 1, len(kl)):
            P = np.linalg.qr(mody[f'a_{kl[x]}'][:a.t_ucz + 1, :5])[0]; Q = np.linalg.qr(mody[f'a_{kl[y]}'][:a.t_ucz + 1, :5])[0]
            zg[f'{kl[x]}-{kl[y]}'] = np.linalg.svd(P.T @ Q, compute_uv=False).round(3).tolist()
    wynik['zgodnosc_modow_miedzy_klatkami_cos_katow'] = zg
    Path(a.wyj).write_text(json.dumps(dict(t_ucz=a.t_ucz, wyniki=wynik), ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez(Path(a.wyj).with_suffix('.npz'), **mody)
    print('zapisano', a.wyj)


if __name__ == '__main__':
    main()
