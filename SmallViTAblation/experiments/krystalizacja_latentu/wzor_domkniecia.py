#!/usr/bin/env python3
"""Sprawdzenie wzoru domkniecia tozsamosci (dokument 44, 2026-09-23).

Wzor: dla wspolrzednych tozsamosci y = Q^T phi (baza ortonormalna w opisie wybielonym)
i bloku reszty w = Q_perp^T phi, przy liniowym modelu phi_{n+1} = A phi_n + eps,

    y_{n+1} = M y_n + sum_{l>=1} B D^{l-1} C y_{n-l} + W_{n+1}
    M = Q^T A Q,  B = Q^T A Q_perp,  C = Q_perp^T A Q,  D = Q_perp^T A Q_perp

Czesc A sprawdza algebre na ukladzie syntetycznym, w ktorym wszystko jest znane dokladnie:
  1. rekurencja Lin21b z korelacji daje dokladnie B D^{l-1} C
  2. zwiazek fluktuacji i dyssypacji: Omega(k) = -E[W_{k+1} W_1^T] C_y(-1)^{-1}
  3. sin(theta) ogranicza ||B|| z obu stron, sin(phi) ogranicza ||C||
  4. pamiec <= ||A||^2 sin(theta) sin(phi) / (1 - ||D||)
  5. kontrprzyklad: C = 0, B != 0, pamiec zero, a podprzestrzen NIE jest niezmiennicza
  6. sama czesc rzeczywista obracajacego sie modu przecieka o |sin(omega)|

Czesc B liczy te same wielkosci na puli klatek (bez etykiet):
  mody wolne, osie tozsamosci, wyciek w slowniku 48 i 96, pamiec z korelacji na
  odstepach 5, 10 i 15 klatek, test fluktuacji i dyssypacji na trojkach klatek,
  przedzialy z bootstrapu po filmach. Dla porownania to samo dla sceny.

    py -3.11 experiments/krystalizacja_latentu/wzor_domkniecia.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --out experiments/krystalizacja_latentu/wyniki/REPORT_wzor_domkniecia.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within
from pokrycie_a_bledy import crop_descriptor, wczytaj


# ---------------------------------------------------------------- narzedzia

def orto_dop(q: np.ndarray) -> np.ndarray:
    """Baza ortonormalna dopelnienia kolumn q."""
    u, _, _ = np.linalg.svd(q, full_matrices=True)
    return u[:, q.shape[1]:]


def katy(a: np.ndarray, b: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
    """Katy glowne miedzy rozpietosciami kolumn a i b w metryce g (domyslnie euklidesowej)."""
    g = np.eye(a.shape[0]) if g is None else g
    def orto(x):
        w, v = np.linalg.eigh(x.T @ g @ x)
        w = np.clip(w, 1e-12, None)
        return x @ v @ np.diag(w ** -.5) @ v.T
    s = np.linalg.svd(orto(a).T @ g @ orto(b), compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def jadra_z_korelacji(cor: list[np.ndarray], ile: int) -> list[np.ndarray]:
    """Lin21b, rown. 52: Omega(k) = (C(k+1) - sum_{l<k} Omega(l) C(k-l)) C(0)^{-1}."""
    c0i = np.linalg.inv(cor[0])
    om = []
    for k in range(ile):
        r = cor[k + 1] - sum(om[l] @ cor[k - l] for l in range(k))
        om.append(r @ c0i)
    return om


# ---------------------------------------------------------------- czesc A

def czesc_a(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, p = 10, 3
    u, _ = np.linalg.qr(rng.normal(size=(d, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = u @ np.diag(rng.uniform(.3, .95, d)) @ v.T          # ||A|| < 1, wiec I - A A^T >= 0
    sig = np.eye(d) - a @ a.T                                 # szum, przy ktorym kowariancja stacjonarna = I
    q, _ = np.linalg.qr(rng.normal(size=(d, p)))
    qp = orto_dop(q)
    m, b, c, dd = q.T @ a @ q, q.T @ a @ qp, qp.T @ a @ q, qp.T @ a @ qp
    wynik = {}

    # 1. rekurencja z dokladnych korelacji C_y(k) = Q^T A^k Q
    cor = [q.T @ np.linalg.matrix_power(a, k) @ q for k in range(8)]
    om = jadra_z_korelacji(cor, 6)
    blad = [float(np.abs(om[l] - (m if l == 0 else b @ np.linalg.matrix_power(dd, l - 1) @ c)).max())
            for l in range(6)]
    wynik['1_rekurencja_vs_BDC_max_blad'] = max(blad)

    # 2. fluktuacja i dyssypacja, Monte Carlo z rozkladu stacjonarnego
    n, dl = 400_000, 5
    ls = np.linalg.cholesky(sig + 1e-12 * np.eye(d))
    phi = [rng.normal(size=(n, d))]
    for _ in range(dl):
        phi.append(phi[-1] @ a.T + rng.normal(size=(n, d)) @ ls.T)
    y = [f @ q for f in phi]
    w = [None] + [y[k + 1] - y[k] @ om[0].T - sum(y[k - l] @ om[l].T for l in range(1, k + 1))
                  for k in range(dl)]
    # porownujemy E[W_{k+1} W_1^T] z -Omega(k) M^T bez odwracania M, bo odwrotnosc wzmacnia szum losowania;
    # wersja dokladna: B D^k B^T + B D^{k-1} Q_perp^T Sigma Q
    fdt_dokl, fdt_mc = [], []
    for k in range(1, 4):
        ww = w[k + 1].T @ w[1] / n
        cel = -om[k] @ m.T
        dokl = b @ np.linalg.matrix_power(dd, k) @ b.T + b @ np.linalg.matrix_power(dd, k - 1) @ qp.T @ sig @ q
        fdt_dokl.append(float(np.abs(dokl - cel).max()))
        sz = np.sqrt(((w[k + 1][:, :, None] * w[1][:, None, :]) ** 2).mean(0) / n).max()
        fdt_mc.append([float(np.abs(ww - cel).max()), float(sz)])
    wynik['2_fdt_dokladnie_max_blad_k123'] = fdt_dokl
    wynik['2_fdt_monte_carlo_[blad,blad_standardowy]_k123'] = fdt_mc

    # 3. granice katowe
    th = katy(q, a.T @ q).max()          # obserwable: V a obraz Koopmana A^T V
    ph = katy(q, a @ q).max()            # stan: V a obraz A V
    sv_t = np.linalg.svd(q.T @ a, compute_uv=False)
    sv_s = np.linalg.svd(a @ q, compute_uv=False)
    nb, nc = np.linalg.norm(b, 2), np.linalg.norm(c, 2)
    wynik['3_B_granice'] = [float(sv_t.min() * np.sin(th)), float(nb), float(sv_t.max() * np.sin(th))]
    wynik['3_C_granice'] = [float(sv_s.min() * np.sin(ph)), float(nc), float(sv_s.max() * np.sin(ph))]

    # 4. suma pamieci wobec iloczynu wyciekow
    pam = sum(np.linalg.norm(b @ np.linalg.matrix_power(dd, l) @ c, 2) for l in range(200))
    gr = np.linalg.norm(a, 2) ** 2 * np.sin(th) * np.sin(ph) / (1 - np.linalg.norm(dd, 2))
    wynik['4_pamiec_i_granica'] = [float(pam), float(gr)]

    # 5. kontrprzyklad: reszta autonomiczna (C = 0), tozsamosc przez nia pchana (B != 0)
    blok = np.zeros((d, d))
    blok[:p, :p] = np.diag([.9, .8, .7])
    blok[:p, p:] = .3 * rng.normal(size=(p, d - p)) / np.sqrt(d)
    blok[p:, p:] = .6 * np.linalg.qr(rng.normal(size=(d - p, d - p)))[0]
    blok /= max(1.0, np.linalg.norm(blok, 2) / .95)
    e = np.eye(d)[:, :p]
    cor5 = [e.T @ np.linalg.matrix_power(blok, k) @ e for k in range(8)]
    om5 = jadra_z_korelacji(cor5, 5)
    wynik['5_C0_norma_pamieci'] = float(max(np.abs(o).max() for o in om5[1:]))
    wynik['5_C0_sin_theta'] = float(np.sin(katy(e, blok.T @ e).max()))

    # 6. obrot: sama czesc rzeczywista modu zespolonego
    om_ = []
    for omega in (.02, .2, .6):
        r = np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]) * .95
        lam, vec = np.linalg.eig(r.T)
        re = vec[:, [0]].real
        om_.append([omega, float(np.sin(katy(re, r.T @ re).max())), float(abs(np.sin(omega)))])
    wynik['6_obrot_[omega,sin_theta,|sin_omega|]'] = om_
    return wynik


# ---------------------------------------------------------------- czesc B

def opis_puli(pula: Path, cells: int, dim: int):
    p = wczytaj(pula)
    grid = tuple(p['meta']['grid'])
    x = crop_descriptor(p['profil_red'].astype(np.float64), p['pred_box'].astype(np.float64), grid, cells)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (p['score'] > .5))
    fid = [p['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid])
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    zs = {k: a @ (vt[:k].T / np.maximum(sv[:k] / np.sqrt(len(a)), 1e-9)) for k in (dim, 2 * dim)}
    return zs, vid, nr


def pary_na_odstepie(vid: np.ndarray, nr: np.ndarray, krok: int, k: int) -> np.ndarray:
    """Wszystkie pary (t, t + k*krok) w tym samym filmie, niekoniecznie sasiednie w puli."""
    gdzie = {(v, n): i for i, (v, n) in enumerate(zip(vid, nr))}
    out = [(i, gdzie[(v, n + k * krok)]) for i, (v, n) in enumerate(zip(vid, nr)) if (v, n + k * krok) in gdzie]
    return np.array(out, dtype=int)


def trojki(vid: np.ndarray, nr: np.ndarray, krok: int) -> np.ndarray:
    gdzie = {(v, n): i for i, (v, n) in enumerate(zip(vid, nr))}
    out = [(i, gdzie[(v, n + krok)], gdzie[(v, n + 2 * krok)]) for i, (v, n) in enumerate(zip(vid, nr))
           if (v, n + krok) in gdzie and (v, n + 2 * krok) in gdzie]
    return np.array(out, dtype=int)


def baza_rzeczywista(u: np.ndarray, lam: np.ndarray, mody: list[int]) -> np.ndarray:
    """Z kazdej pary sprzezonej bierzemy Re i Im jednego modu; z modu rzeczywistego sam wektor."""
    kol, byly = [], []
    for k in mody:
        if any(abs(lam[k] - np.conj(lam[j])) < 1e-9 and abs(lam[k].imag) > 1e-12 for j in byly):
            continue
        byly.append(k)
        kol.append(u[:, k].real)
        if abs(lam[k].imag) > 1e-12:
            kol.append(u[:, k].imag)
    return np.stack(kol, 1)


def pamiec_y(y: np.ndarray, pary: list[np.ndarray], tr: np.ndarray, waga: np.ndarray | None = None) -> dict:
    """Jadra Omega(0..2) z korelacji na odstepach 1, 2, 3 i test FDT dla k = 1 na trojkach.

    `waga` to krotnosc filmu klatki w losowaniu bootstrapowym (1 wszedzie, gdy brak).
    """
    waga = np.ones(len(y)) if waga is None else waga
    def sr(a_, b_, w_):
        return (a_ * w_[:, None]).T @ b_ / w_.sum()
    y = y - (y * waga[:, None]).sum(0) / waga.sum()
    w_, v_ = np.linalg.eigh(sr(y, y, waga))
    y = y @ v_ @ np.diag(w_ ** -.5) @ v_.T                         # wybielenie: C_y(0) = I
    cor = [np.eye(y.shape[1])] + [sr(y[pr[:, 1]], y[pr[:, 0]], waga[pr[:, 0]]) for pr in pary]
    om = jadra_z_korelacji(cor, len(pary) - 1)
    w1 = y[tr[:, 1]] - y[tr[:, 0]] @ om[0].T
    w2 = y[tr[:, 2]] - y[tr[:, 1]] @ om[0].T - y[tr[:, 0]] @ om[1].T
    om1_fdt = -sr(w2, w1, waga[tr[:, 0]]) @ np.linalg.inv(cor[1].T)
    semigrupa = np.linalg.norm(cor[2] - cor[1] @ cor[1]) / np.linalg.norm(cor[2])
    return dict(norma_om0=float(np.linalg.norm(om[0])), om0=om[0].tolist(),
                pamiec_wzgl=[float(np.linalg.norm(o) / np.linalg.norm(om[0])) for o in om[1:]],
                om1_fdt_wzgl=float(np.linalg.norm(om1_fdt) / np.linalg.norm(om[0])),
                om1_roznica_fdt=float(np.linalg.norm(om1_fdt - om[1]) / np.linalg.norm(om[0])),
                semigrupa=float(semigrupa), om1=om[1].tolist(), om1_z_fdt=om1_fdt.tolist())


def czesc_b(args) -> dict:
    zs, vid, nr = opis_puli(args.pula, args.cells, args.dim)
    z = zs[args.dim]
    par = pairs_within(vid, nr, args.max_gap)
    lam, u, mu = edmd(z[par[:, 0]], z[par[:, 1]])
    wolne = np.flatnonzero(lam.real > args.prog_re)
    kod = ((z - mu) @ u[:, wolne]).real
    mid = {v: kod[vid == v].mean(0) for v in set(vid.tolist())}
    dyn = kod - np.array([mid[v] for v in vid])
    w_f = (dyn ** 2).sum(0) / np.maximum(((kod - kod.mean(0)) ** 2).sum(0), 1e-12)
    toz = [int(wolne[k]) for k in range(len(wolne)) if w_f[k] >= args.prog_w_filmie]
    scena = [int(wolne[k]) for k in range(len(wolne)) if w_f[k] < args.prog_w_filmie]
    mody = [dict(nr=int(wolne[k]), re=float(lam[wolne[k]].real), im=float(lam[wolne[k]].imag),
                 omega=float(np.angle(lam[wolne[k]])), w_filmie=float(w_f[k]),
                 rola='tozsamosc' if w_f[k] >= args.prog_w_filmie else 'scena') for k in range(len(wolne))]
    print(f'klatek {len(z)}, par {len(par)}, modow wolnych {len(wolne)}; tozsamosc {toz}, scena {scena}')

    # osie tak, jak wybral je K6 (pierwsze trzy mody tozsamosci, same czesci rzeczywiste)
    k6 = [int(wolne[k]) for k in range(len(wolne)) if w_f[k] >= args.prog_w_filmie][:3]
    c6 = ((z - mu) @ u[:, k6]).real
    kor_k6 = np.corrcoef(c6.T)

    # slownik 48 i 96: te same pierwsze kierunki, wiec baze z 48 dopelniamy zerami
    z96 = zs[2 * args.dim]
    par_all = par
    def wyciek(baza48: np.ndarray, zz: np.ndarray) -> float:
        dd = zz.shape[1]
        v = np.zeros((dd, baza48.shape[1]))
        v[:baza48.shape[0]] = baza48
        m0 = zz[par_all[:, 0]] - zz[par_all[:, 0]].mean(0)
        m1 = zz[par_all[:, 1]] - zz[par_all[:, 0]].mean(0)
        g = m0.T @ m0 / len(m0) + 1e-3 * np.eye(dd)
        kk = np.linalg.solve(g, m0.T @ m1 / len(m0))          # macierz Koopmana na wspolczynnikach
        return float(np.sin(katy(v, kk @ v, g).max()))

    wyn = {'mody': mody, 'k6_osie': k6, 'k6_korelacja_osi': kor_k6.tolist()}
    bazy = {'tozsamosc': baza_rzeczywista(u, lam, toz), 'scena': baza_rzeczywista(u, lam, scena),
            'k6_same_re': u[:, [k for i, k in enumerate(k6)
                                if not any(abs(lam[k] - np.conj(lam[j])) < 1e-9 for j in k6[:i])]].real}
    krok = int(min(np.diff(np.sort(nr[vid == v])).min() for v in np.unique(vid) if (vid == v).sum() > 1))
    pary = [pary_na_odstepie(vid, nr, krok, k) for k in (1, 2, 3)]
    tr = trojki(vid, nr, krok)
    filmy = np.unique(vid)
    rng = np.random.default_rng(0)
    print(f'krok {krok} klatek, par na odstepach {[len(p_) for p_ in pary]}, trojek {len(tr)}')
    for nazwa, baza in bazy.items():
        y = (z - mu) @ baza
        r = dict(wymiar=int(baza.shape[1]), wyciek_d48=wyciek(baza, z), wyciek_d96=wyciek(baza, z96))
        r.update(pamiec_y(y, pary, tr))
        boot = []
        nr_filmu = {f: i for i, f in enumerate(filmy)}
        fi = np.array([nr_filmu[v] for v in vid])
        for _ in range(args.bootstrap):
            krotnosc = np.bincount(rng.integers(0, len(filmy), len(filmy)), minlength=len(filmy))
            q_ = pamiec_y(y, pary, tr, krotnosc[fi].astype(float))
            boot.append([q_['pamiec_wzgl'][0], q_['om1_fdt_wzgl'], q_['semigrupa']])
        boot = np.array(boot)
        r['bootstrap_95'] = {k: [float(np.quantile(boot[:, j], .025)), float(np.quantile(boot[:, j], .975))]
                             for j, k in enumerate(['pamiec_om1', 'om1_z_fdt', 'semigrupa'])}
        # podloga szumu: proces BEZ pamieci (y_{n+1} = Om0 y_n + szum) na tej samej siatce klatek;
        # norma jadra jest dodatnia, wiec nawet przy braku pamieci wychodzi wieksza od zera
        om0 = np.array(r['om0'])
        rho = np.abs(np.linalg.eigvals(om0)).max()
        if rho > .995:                                   # estymata na granicy stabilnosci wybucha na dlugim filmie
            om0 = om0 * .995 / rho
        s_ = np.eye(len(om0)) - om0 @ om0.T
        wl, vl = np.linalg.eigh(s_)
        l_ = vl @ np.diag(np.sqrt(np.clip(wl, 0, None)))
        poz = np.zeros(len(vid), dtype=int)
        for v in filmy:
            i = np.flatnonzero(vid == v)
            poz[i] = (nr[i] - nr[i].min()) // krok
        zero = []
        for _ in range(args.bootstrap):
            ys = np.zeros((len(vid), len(om0)))
            for v in filmy:
                i = np.flatnonzero(vid == v)
                traj = [rng.normal(size=len(om0))]
                for _t in range(poz[i].max()):
                    traj.append(om0 @ traj[-1] + l_ @ rng.normal(size=len(om0)))
                ys[i] = np.array(traj)[poz[i]]
            q_ = pamiec_y(ys, pary, tr)
            zero.append([q_['pamiec_wzgl'][0], q_['om1_fdt_wzgl']])
        zero = np.array(zero)
        r['bez_pamieci_95'] = {k: float(np.quantile(zero[:, j], .95)) for j, k in enumerate(['pamiec_om1', 'om1_z_fdt'])}
        wyn[nazwa] = {k: v for k, v in r.items() if k not in ('om1', 'om1_z_fdt', 'om0')}
        print(f'{nazwa:12s} wymiar {r["wymiar"]}, wyciek d48 {r["wyciek_d48"]:.3f}, d96 {r["wyciek_d96"]:.3f}, '
              f'pamiec |Om1|/|Om0| {r["pamiec_wzgl"][0]:.3f} {np.round(r["bootstrap_95"]["pamiec_om1"], 3)} '
              f'(bez pamieci do {r["bez_pamieci_95"]["pamiec_om1"]:.3f}), '
              f'z FDT {r["om1_fdt_wzgl"]:.3f} (bez pamieci do {r["bez_pamieci_95"]["om1_z_fdt"]:.3f}), '
              f'roznica jader {r["om1_roznica_fdt"]:.3f}', flush=True)
    return wyn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--prog-re', type=float, default=0.85)
    ap.add_argument('--prog-w-filmie', type=float, default=0.35)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--bootstrap', type=int, default=200)
    args = ap.parse_args(argv)

    a = czesc_a()
    print('CZESC A (uklad syntetyczny, wszystko znane dokladnie)')
    for k, v in a.items():
        print(f'   {k}: {np.round(v, 4).tolist() if isinstance(v, list) else round(v, 6)}')
    b = czesc_b(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'czesc_a': a, 'czesc_b': b}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('zapisane', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
