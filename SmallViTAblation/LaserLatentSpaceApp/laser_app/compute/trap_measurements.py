"""Pomiary pułapki optycznej (ogniska wiązek) wybranej cząsteczki w 12 podkrokach przejścia przez sieć, przez całą historię uczenia.

Dla każdego checkpointu i podkroku k (uwaga albo MLP bloku k // 2), w układzie '_n' (jedna skala na checkpoint, jak okienko 11):
- skok f0 = prawdziwy ruch cząsteczki w podkroku (suma wkładów), r_kom = połowa odległości do najbliższego tokenu;
- pole w kuli komórki (pole_podkrokow, siatka G³) i jego część liniowa f(u) ≈ f0 + J u (dopasowanie, R²);
  J (3 × 3, bez jednostek: przesunięcie na jednostkę odsunięcia) rozkładamy jak w pęsecie optycznej:
    sztywność κ = −wartości własne części symetrycznej (κ > 0 ściąga do środka, κ < 0 rozpycha; siodło = znaki mieszane),
    dywergencja tr J (< 0: pole zbiega się), wir ω = część antysymetryczna (wektor, |ω|) i stosunek wir / siodło,
    zero pola z części liniowej u* = −J⁻¹ f0 (gdzie stoi pułapka) w jednostkach r_kom i skoku, stabilność punktu stałego
    odwzorowania x → x + f(x): promień widmowy I + J < 1 (pułapka trzyma) i wartości własne J (zespolone = wir);
- ognisko (wiązki): tensor kierunków wkładów ważony ich długością (λ1 ≥ λ2 ≥ λ3, izotropowo 1/3), wydłużenie,
  koherencja = |Σ wkładów| / Σ |wkładów| (1 = wszystkie pchają w jedną stronę), udział 10 najsilniejszych wkładów;
  MLP: ile laserów świeci (z > 0,05) i ile ciągnie ujemnie (z < −0,05); uwaga: efektywna liczba słuchanych tokenów exp(H).
Zapis: `<katalog historii>/pulapka_podkrokow_t<tok>_G<G>.npz` (tablice (K, 12, ...)) i `.csv` (wiersz = checkpoint × podkrok).
Uruchomienie: python trap_measurements.py <katalog historii> [tok] [co] [G]  albo przycisk w sekcji „Pułapka optyczna”.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

from laser_app.compute import history as hist, passage_store, substep_field

KOLUMNY = ('krok', 'podkrok', 'blok', 'rodzaj', 'skok', 'r_kom', 'r_kom_skok', 'R2_lin', 'kappa1', 'kappa2', 'kappa3', 'dywergencja',
           'wir', 'wir_siodlo', 'zero_r_kom', 'zero_skok', 'stabilna', 'promien_IJ', 'zespolone', 'lam1', 'lam2', 'lam3',
           'wydluzenie', 'koherencja', 'udzial_top10', 'zapalone', 'ujemne', 'zrodla_eff', 'blad_f0')


def plik(h, tok, G):
    return Path(h.katalog) / f'pulapka_podkrokow_t{int(tok):03d}_G{int(G)}.npz'


def _ognisko(wk):
    """Tensor kierunków wkładów (ważony długością), wydłużenie, koherencja, udział 10 najsilniejszych."""
    dl = np.linalg.norm(wk, axis=1); s = max(float(dl.sum()), 1e-30)
    kn = wk / np.maximum(dl, 1e-30)[:, None]
    T = (dl[:, None, None] * kn[:, :, None] * kn[:, None, :]).sum(0) / s
    lam = np.sort(np.linalg.eigvalsh(T))[::-1]
    return lam, float(np.sqrt((1 - lam[2]) / max(1 - lam[0], 1e-6))), float(np.linalg.norm(wk.sum(0)) / s), \
        float(np.sort(dl)[::-1][:10].sum() / s)


def pomiar(wagi, krok, rgb, j, d, G=5, st=None):
    """Pomiary jednego checkpointu: słownik tablic (12, ...). d = przejście z 'baza' (P_n)."""
    w = substep_field.policz(wagi, krok, rgb, j, d, G=G, st=st)
    lin = np.linspace(-1, 1, G); gz, gy, gx = np.meshgrid(lin, lin, lin, indexing='ij')
    S = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1); kula = np.linalg.norm(S, axis=1) <= 1.0001
    pos = np.asarray(d['pos_n'], np.float64)
    o = {k: np.zeros(12) for k in KOLUMNY if k not in ('krok', 'podkrok', 'blok', 'rodzaj')}
    o['J'] = np.zeros((12, 3, 3)); o['f0'] = np.zeros((12, 3)); o['zero'] = np.zeros((12, 3)); o['wir_wek'] = np.zeros((12, 3))
    o['eigJ'] = np.zeros((12, 3), complex)
    for k in range(12):
        b = k // 2; rk = float(w['L'][k])
        U = S[kula] * rk; F = w['pole'][k][kula, :3].astype(np.float64); f0 = F[np.argmin(np.linalg.norm(S[kula], axis=1))]
        D = F - f0
        Jt, *_ = np.linalg.lstsq(U, D, rcond=None); J = Jt.T
        r2 = 1 - ((D - U @ Jt) ** 2).sum() / max((D ** 2).sum(), 1e-30)
        Sy, A = (J + J.T) / 2, (J - J.T) / 2
        kap = -np.sort(np.linalg.eigvalsh(Sy))[::-1]                     # κ1 ≤ κ2 ≤ κ3 (κ > 0 ściąga)
        om = np.array([A[2, 1], A[0, 2], A[1, 0]])
        try:
            zero = -np.linalg.solve(J, f0)
        except np.linalg.LinAlgError:
            zero = np.full(3, np.nan)
        ev = np.linalg.eigvals(J); rho = float(np.abs(np.linalg.eigvals(np.eye(3) + J)).max())
        sk = float(np.linalg.norm(f0))
        if k % 2:                                                       # MLP: 768 laserów bloku
            wk = d['z_c'][b, j].astype(np.float64)[:, None] * d['pv_n'][b].astype(np.float64)
            z = np.asarray(d['z'][b, j]); zap, uj, zr = float((z > 0.05).sum()), float((z < -0.05).sum()), np.nan
        else:                                                           # uwaga: 600 wkładów od innych cząsteczek
            wk = np.einsum('hk,hkd->kd', d['ah_c'][b, :, j, :].astype(np.float64), d['uh_n'][b].astype(np.float64))
            a = np.asarray(d['ah'][b, :, j, :], np.float64).mean(0); a = a / a.sum()
            zap, uj, zr = np.nan, np.nan, float(np.exp(-(a * np.log(np.maximum(a, 1e-30))).sum()))
        lam, wyd, koh, top = _ognisko(wk)
        ruch = pos[k + 1, j] - pos[k, j]
        o['J'][k] = J; o['f0'][k] = f0; o['zero'][k] = zero; o['wir_wek'][k] = om; o['eigJ'][k] = ev
        for n_, v_ in dict(skok=sk, r_kom=rk, r_kom_skok=rk / max(sk, 1e-30), R2_lin=r2, kappa1=kap[0], kappa2=kap[1], kappa3=kap[2],
                           dywergencja=np.trace(J), wir=np.linalg.norm(om), wir_siodlo=np.linalg.norm(A) / max(np.linalg.norm(Sy), 1e-30),
                           zero_r_kom=np.linalg.norm(zero) / max(rk, 1e-30), zero_skok=np.linalg.norm(zero) / max(sk, 1e-30),
                           stabilna=float(rho < 1), promien_IJ=rho, zespolone=float(np.abs(ev.imag).max() > 1e-9),
                           lam1=lam[0], lam2=lam[1], lam3=lam[2], wydluzenie=wyd, koherencja=koh, udzial_top10=top,
                           zapalone=zap, ujemne=uj, zrodla_eff=zr,
                           blad_f0=np.linalg.norm(f0 - ruch) / max(np.linalg.norm(ruch), 1e-30)).items():
            o[n_][k] = v_
    return o


def policz_historie(h, tok=226, co=1, G=5, postep=None, przerwij=None):
    """Pomiary dla checkpointów historii co `co` (zawsze z ostatnim); zapis npz + csv; zwraca ścieżkę npz."""
    kat = passage_store.katalog(h); baza = passage_store.wczytaj_baze(kat)
    idx = list(range(0, h.K, max(int(co), 1)))
    if idx[-1] != h.K - 1:
        idx.append(h.K - 1)
    wyn = []; t0 = time.time()
    import torch
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)                                         # wagi wczytywane z wyprzedzeniem i raz na checkpoint
    odczyt = lambda k: torch.load(Path(h.wagi) / f'k{int(k):05d}.pt', map_location='cpu', weights_only=True)
    kolejka = [pula.submit(odczyt, int(h.kroki[ix])) for ix in idx[:6]]
    try:
        for i, ix in enumerate(idx):
            if przerwij is not None and przerwij():
                break
            krok = int(h.kroki[ix])
            st = kolejka.pop(0).result()
            if i + 6 < len(idx):
                kolejka.append(pula.submit(odczyt, int(h.kroki[idx[i + 6]])))
            d = passage_store.wczytaj(kat, krok)
            if d is None:
                with passage_store.ZAMEK_SIECI:
                    d = hist.podkroki(h.wagi, krok, h.rgb, baza=baza, st=st, lodyga=False)
                if baza is None:
                    baza = d['baza']; passage_store.zapisz_baze(kat, baza)
            d['baza'] = baza
            wyn.append((krok, pomiar(h.wagi, krok, h.rgb, tok, d, G=G, st=st)))
            if postep is not None:
                tt = (time.time() - t0) / (i + 1) * (len(idx) - i - 1)
                postep(i + 1, len(idx), f'krok {krok}: {i + 1}/{len(idx)}, zostało ~{tt / 60:.1f} min')
    finally:
        for f_ in kolejka:
            f_.cancel()
        pula.shutdown(wait=False)
    if not wyn:
        return None
    kroki = np.array([k for k, _ in wyn])
    tab = {n: np.stack([o[n] for _, o in wyn]) for n in wyn[0][1]}
    p = plik(h, tok, G)
    np.savez_compressed(p, kroki=kroki, tok=tok, G=G, **tab)
    with open(p.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(KOLUMNY)
        for i, krok in enumerate(kroki):
            for k in range(12):
                w.writerow([int(krok), k + 1, k // 2 + 1, 'MLP' if k % 2 else 'uwaga']
                           + [f'{float(tab[n][i, k]):.6g}' for n in KOLUMNY[4:]])
    return p


if __name__ == '__main__':
    kat_h = sys.argv[1]; tok = int(sys.argv[2]) if len(sys.argv) > 2 else 226
    co = int(sys.argv[3]) if len(sys.argv) > 3 else 1; G = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    h = hist.History(kat_h)
    p = policz_historie(h, tok, co, G, postep=lambda i, n, t: print(t, flush=True) if i % 25 == 0 or i == n else None)
    print('zapisano', p)
