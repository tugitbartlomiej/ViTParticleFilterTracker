"""LOKALNY KOOPMAN NA PUŁAPKACH (2026-10-13): krótkie okna przesuwne i krótki horyzont.

Pierwsza wersja (koopman_pulapki.py) dopasowywała jeden operator na całe reżimy uczenia (4 500–44 000 kroków), czyli uśredniała
niestacjonarną dynamikę. Tu: operator dopasowany w oknie L punktów, prognoza h = 1…H punktów ZA oknem, okno przesuwane.
Pułapki dwiema definicjami: (a) średnia krocząca ±w (jak aplikacja; sąsiednie pułapki dzielą dane → możliwa sztuczna
przewidywalność), (b) średnie z bloków B punktów bez nakładania. Każdy wynik obok surogatu: przyrosty położeń przetasowane
w czasie (ta sama permutacja dla tokenów), te same pułapki i te same prognozy.
Modele w oknie: A — wspólny operator jednej cząstki (d × d, ridge), B — DMD chmury zdjęcia w bazie POD r = 5 (prognoza w pełnej
przestrzeni), V — „dalej tym samym tempem” (ostatni przyrost); wynik: błąd / błąd prognozy „stoi”.
Wejścia (skala kroku, z gesty/PACZKI.npz): (i) czy ŚLEDZONE ZDJĘCIE jest w paczce kroku (wtedy działa jego własna strata —
„własna siła pułapki”, raz na epokę), (ii) norma faktycznej aktualizacji AdamW, (iii) spadek straty paczki w kroku, (iv) lr.
Model C — DMDc chmury: q(t+1) = A q(t) + B u(t) ze znanym przyszłym u; analiza zdarzeń: skok tokenów zdjęcia w krokach, gdy jest
w paczce, wobec zwykłych kroków, i czy ten skok ściąga tokeny ku pułapce (cos między skokiem a c − x) czy od niej.
Dane: (1) gęsty zapis RP128 co 1 krok (gesty/pca32_*.npz biegów pokręteł, 12 sond), (2) eksport ruch60 rp128 co 100 kroków (806 ckpt).
Uruchomienie: python koopman_lokalny.py --gesty DIR[:od:do] [...] --ruch60 DIR --out DIR
"""
import argparse
import glob
import json
import os
import time

import numpy as np

t0 = time.time()


def log(*s):
    print(*s, f'[{time.time() - t0:.0f} s]', flush=True)


def srednia_kroczaca(Y, w):
    c = np.cumsum(Y, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + Y.shape[1:]), c])
    return ((c[2 * w + 1:] - c[:-2 * w - 1]) / (2 * w + 1)).astype(np.float32)


def bloki(Y, B):
    n = len(Y) // B
    return Y[:n * B].reshape(n, B, *Y.shape[1:]).mean(1).astype(np.float32)


def surogat(X, rng):
    dX = np.diff(X, axis=0); dX = dX[rng.permutation(len(dX))]
    return np.concatenate([X[:1], X[0] + np.cumsum(dX, 0)]).astype(np.float32)


def okna_prognoz(C, L, H, krok, r=5, ridge=1e-4, U=None):
    """C (T, n, d) pułapki → sumy błędów prognoz A, B, V (i C = DMDc z wejściem U (T − 1, k)) i „stoi”, dla h = 1..H."""
    T, n, d = C.shape
    E = {m: np.zeros(H) for m in ('A', 'B', 'V', 'stoi') + (('C',) if U is not None else ())}
    ile = 0
    for s in range(0, T - L - H + 1, krok):
        W_ = C[s:s + L].astype(np.float64)
        mu = W_.mean(0, keepdims=True)
        X = W_ - mu
        # A: wspólny operator jednej cząstki
        a, b = X[:-1].reshape(-1, d), X[1:].reshape(-1, d)
        G = a.T @ a; G += ridge * np.trace(G) / d * np.eye(d)
        A = np.linalg.solve(G, a.T @ b).T
        # B: DMD chmury w POD r
        Z = X.reshape(L, -1)
        ev, V = np.linalg.eigh(Z @ Z.T); ev, V = ev[::-1], V[:, ::-1]
        rr = int(min(r, (ev > ev[0] * 1e-10).sum()))
        Phi = (Z.T @ V[:, :rr]) / np.sqrt(ev[:rr]); q = Z @ Phi
        Gq = q[:-1].T @ q[:-1]; Gq += ridge * np.trace(Gq) / rr * np.eye(rr)
        Bm = np.linalg.solve(Gq, q[:-1].T @ q[1:]).T
        xa, qb, zb = X[-1].copy(), q[-1].copy(), Z[-1]
        v = X[-1] - X[-2]
        if U is not None:
            Uw = U[s:s + L - 1]
            O = np.concatenate([q[:-1], Uw], 1)
            Go = O.T @ O; Go += ridge * np.diag(np.diag(Go))                    # kara względem wariancji każdej zmiennej (wejścia mają inną skalę niż q)
            Go += 1e-8 * np.diag(Go).mean() * np.eye(len(Go))                 # wejście stałe w oknie (np. zdjęcie poza paczką) → kolumna zerowa
            Mo = np.linalg.solve(Go, O.T @ q[1:])
            Ac, Bc = Mo[:rr].T, Mo[rr:].T
            qc = q[-1].copy()
        for h in range(1, H + 1):
            y = C[s + L - 1 + h].astype(np.float64) - mu[0]
            xa = xa @ A.T; qb = Bm @ qb
            pB = (zb + (qb - q[-1]) @ Phi.T).reshape(n, d)
            E['A'][h - 1] += ((y - xa) ** 2).sum()
            E['B'][h - 1] += ((y - pB) ** 2).sum()
            E['V'][h - 1] += ((y - (X[-1] + h * v)) ** 2).sum()
            E['stoi'][h - 1] += ((y - X[-1]) ** 2).sum()
            if U is not None:
                qc = Ac @ qc + Bc @ U[s + L - 2 + h]                          # wejście kroku pułapki (s+L−2+h) → (s+L−1+h)
                E['C'][h - 1] += ((y - (zb + (qc - q[-1]) @ Phi.T).reshape(n, d)) ** 2).sum()
        ile += 1
    return E, ile


def u_pulapki(u_raw, nazwa):
    """Wejście surowe u_raw (T − 1, k): u_raw[i] = wejście przejścia x[i] → x[i+1]. Przejście pułapki t → t+1:
    średnia krocząca ±w: c(t+1) − c(t) = (x[t+2w+1] − x[t]) / (2w+1) → średnia u_raw[t .. t+2w]; bloki B: średnia u_raw[tB .. tB+2B−2]."""
    if nazwa.startswith('krocz'):
        w = int(nazwa.split('w')[-1]); m = 2 * w + 1
        c = np.cumsum(np.concatenate([np.zeros((1, u_raw.shape[1])), u_raw]), 0)
        return (c[m:] - c[:-m]) / m
    # bloki: c(n+1) − c(n) = (1/B) Σ_k Σ_{i=nB+k}^{(n+1)B+k−1} dx[i] → wagi trójkątne na dx[nB .. nB+2B−2] (test syntetyczny:
    # równe wagi psuły sterowanie)
    B = int(nazwa.split('B')[-1]); n = (len(u_raw) + 1) // B
    j = np.arange(2 * B - 1)
    wag = (np.minimum(j, B - 1) - np.maximum(j - B + 1, 0) + 1) / B
    wag = wag / wag.sum()
    return np.stack([(u_raw[t * B:t * B + 2 * B - 1] * wag[:, None]).sum(0) for t in range(n - 1)])


def zdarzenia(Xrel, w_paczce, w=25):
    """Skok tokenów w krokach, gdy zdjęcie jest w paczce, wobec zwykłych; kierunek skoku względem pułapki (c − x, c = średnia ±w)."""
    dX = np.diff(Xrel, axis=0)                                            # dX[i]: krok i+1 (przejście x[i] → x[i+1])
    C = srednia_kroczaca(Xrel, w)                                         # C[t] = pułapka w chwili t + w
    ev = np.where(w_paczce)[0]
    ev = ev[(ev >= w) & (ev < len(C) + w - 1)]
    if len(ev) == 0:
        return None
    zw = np.setdiff1d(np.arange(w, len(C) + w - 1), ev)
    skok = lambda ii: float(np.sqrt((dX[ii] ** 2).sum(-1)).mean())
    def cos_do_pulapki(ii):
        v = dX[ii]; c_x = C[ii - w] - Xrel[ii]
        return float(((v * c_x).sum(-1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(c_x, axis=-1) + 1e-12)).mean())
    return dict(zdarzen=int(len(ev)), skok_w_paczce=skok(ev), skok_zwykly=skok(zw), stosunek=skok(ev) / max(skok(zw), 1e-12),
                cos_ku_pulapce_w_paczce=cos_do_pulapki(ev), cos_ku_pulapce_zwykly=cos_do_pulapki(zw[::7]))


def analiza(Xrel, dt, rng, L, H, krok, w_lista, B_lista, u_raw=None):
    """Xrel (T, n, d) położenia względne jednego zdjęcia → stosunki błędów dla każdej definicji pułapki, dane i surogat."""
    out = {}
    Xs = surogat(Xrel, rng)
    defs = [(f'krocz_w{w}', lambda X, w=w: srednia_kroczaca(X, w)) for w in w_lista] + [(f'bloki_B{B}', lambda X, B=B: bloki(X, B)) for B in B_lista]
    for nazwa, f in defs:
        C, Cs = f(Xrel), f(Xs)
        if len(C) < L + H + 1:
            continue
        U = u_pulapki(u_raw, nazwa)[:len(C) - 1] if u_raw is not None else None
        E, ile = okna_prognoz(C, L, H, krok, U=U)
        Es, _ = okna_prognoz(Cs, L, H, krok, U=(U[rng.permutation(len(U))] if U is not None else None))
        krok_pul = dt * (int(nazwa.split('B')[-1]) if nazwa.startswith('bloki') else 1)
        out[nazwa] = dict(okien=ile, krok_pulapki_w_krokach_uczenia=krok_pul,
                          dane={m: (E[m] / E['stoi']).tolist() for m in E if m != 'stoi'},
                          surogat={m: (Es[m] / Es['stoi']).tolist() for m in Es if m != 'stoi'})
        if U is not None:                                                  # kontrola: dane z wejściem przetasowanym w czasie
            Et, _ = okna_prognoz(C, L, H, krok, U=U[rng.permutation(len(U))])
            out[nazwa]['dane_wejscie_tasowane_C'] = (Et['C'] / Et['stoi']).tolist()
    return out


def mediana(lista):
    if not lista:
        return None
    if isinstance(lista[0], dict):
        return {k: mediana([x[k] for x in lista if k in x]) for k in lista[0]}
    if isinstance(lista[0], list):
        return np.median(np.array(lista), 0).round(4).tolist()
    return float(np.median(lista))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gesty', nargs='*', default=[], help='katalog gesty[:od:do] (zapis co 1 krok)')
    ap.add_argument('--ruch60', default=None)
    ap.add_argument('--n_ruch60', type=int, default=20, help='zdjęć z eksportu ruch60 (połowa TRAIN, połowa VALID)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=int, default=30); ap.add_argument('--H', type=int, default=5); ap.add_argument('--krok', type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    WYN = dict(opis=__doc__, L=a.L, H=a.H, krok_okna=a.krok, wyniki={})
    for spec in a.gesty:
        czesci = spec.split(':'); G = czesci[0]
        od, do = (int(czesci[1]), int(czesci[2])) if len(czesci) == 3 else (0, 10 ** 9)
        Xs, kr = [], []
        for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
            z = np.load(p); k = z['kroki']; m = (k >= od) & (k <= do)
            if m.any():
                Xs.append(z['pca32'][m].astype(np.float32)); kr.append(k[m])
        X = np.concatenate(Xs); kr = np.concatenate(kr)
        assert (np.diff(kr) == 1).all(), f'{spec}: zapis nie co 1 krok'
        nazwa = os.path.basename(os.path.dirname(G.rstrip('/'))) + f'_{kr[0]}_{kr[-1]}'
        log(f'gęsty {nazwa}: {X.shape}')
        P = np.load(os.path.join(G, 'PACZKI.npz'))
        sondy = json.load(open(os.path.join(G, 'SONDY.json'), encoding='utf-8'))['sondy']
        wiersz = {int(k): i for i, k in enumerate(P['krok'])}
        rz = np.array([wiersz[int(k)] for k in kr[1:]])                    # przejście kr[i] → kr[i+1] = krok kr[i+1]
        dstrata = (P['strata_przed'] - P['strata_po'])[rz]
        glob_u = np.stack([P['norma_aktualizacji'][rz], dstrata, P['lr'][rz]], 1).astype(np.float64)
        glob_u = (glob_u - glob_u.mean(0)) / (glob_u.std(0) + 1e-12)
        per, zd = [], []
        for s in range(X.shape[1]):
            Xr = X[:, s] - X[:, s].mean(1, keepdims=True)
            w_paczce = (P['idx'][rz] == sondy[s]).any(1)
            u_raw = np.concatenate([w_paczce[:, None].astype(np.float64), glob_u], 1)
            per.append(analiza(Xr, 1.0, rng, a.L, a.H, a.krok, w_lista=(5, 10, 25), B_lista=(5, 10, 25), u_raw=u_raw))
            z_ = zdarzenia(Xr, w_paczce)
            if z_ is not None:
                zd.append(z_)
        WYN['wyniki'][nazwa] = dict(skala='krok', zdjec=len(per), mediany=mediana(per), zdarzenia_wlasnej_paczki=zd,
                                    zdarzenia_mediany=mediana(zd) if zd else None)
        json.dump(WYN, open(os.path.join(a.out, 'KOOPMAN_LOKALNY.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if a.ruch60:
        meta = json.load(open(os.path.join(a.ruch60, 'META.json'), encoding='utf-8'))
        kroki = np.array(meta['kroki']); n = len(meta['zdjecia']) // 2
        rp = np.load(os.path.join(a.ruch60, 'rp128.npy'), mmap_mode='r')
        G2 = np.array([i for i, k in enumerate(kroki) if 1000 <= k <= 65600 and k % 100 == 0])
        idx = list(range(a.n_ruch60 // 2)) + list(range(n, n + a.n_ruch60 // 2))
        for nazwa_r, od, do in (('R1_1000_5470', 1000, 5470), ('R2_5470_21880', 5470, 21880), ('R3_21880_65600', 21880, 65600)):
            ii = G2[(kroki[G2] >= od) & (kroki[G2] <= do)]
            per = {'TRAIN': [], 'VALID': []}
            for j in idx:
                Xr = np.asarray(rp[ii, j], np.float32); Xr = Xr - Xr.mean(1, keepdims=True)
                per[meta['zdjecia'][j]['split']].append(analiza(Xr, 100.0, rng, a.L, a.H, a.krok, w_lista=(1, 3), B_lista=(3,)))
            WYN['wyniki'][f'ruch60_{nazwa_r}'] = {sp: dict(skala='100 kroków', zdjec=len(v), mediany=mediana(v)) for sp, v in per.items()}
            log(f'ruch60 {nazwa_r} gotowe')
            json.dump(WYN, open(os.path.join(a.out, 'KOOPMAN_LOKALNY.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
