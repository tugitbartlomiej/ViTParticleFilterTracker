"""Tożsamość narzędzia liczona kierunkiem stałym i odświeżanym, na zapisanych latentach sond 10 ziaren (PLAN.md).

Co rozstrzyga: czy rozdzielenie narzędzia od tła mierzone kierunkiem dopasowywanym od nowa idzie w parze z mAP,
skoro siła wzdłuż kierunku stałego nie idzie (obliczenie z 5 października na kampanii 4.3).
Co obala: tezę, że spadek siły tożsamości wzdłuż stałego kierunku oznacza gorsze zakodowanie narzędzia, albo tezę,
że tożsamość w tej definicji jest miarą postępu detektora.

Dane: `../../maly_vit_wektor_tozsamosci/wyniki_eden/petla_s47..s56` (latent sond co 10 kroków, mAP co 100).
Bez treningu, bez GPU. Przepis na kierunek skopiowany z `../../maly_vit_petla_sd_rownanie/kod/sila.py` (funkcje
`_ledoit_wolf`, `_cov`, `lda`, `wektor`, `sila`); jedyna zmiana: `numpy.linalg.solve` zamiast `lstsq` (macierz po
ściągnięciu jest dodatnio określona, wynik ten sam, liczy się szybciej).

    py -3.11 odswiezana.py --wyniki ../wyniki                 (pomiar i analiza)
    py -3.11 odswiezana.py --wyniki ../wyniki --tylko-analiza (analiza z zapisanego POMIAR.npz)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ZRODLO = Path(__file__).resolve().parents[2] / 'maly_vit_wektor_tozsamosci' / 'wyniki_eden'
ZIARNA = list(range(47, 57))
RAMIONA = ('bez', 'losowe', 'koopman')
SIATKA = list(range(50, 4501, 50))                      # chwile pomiaru
OBECN = np.array([i % 16 != 8 for i in range(192)])     # wersja empty_black nie ma narzędzia
WYCINEK = np.arange(192) // 16
PODZIALY = [np.isin(WYCINEK, f) for f in ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))]
KROKI_CKA = (1000, 2500, 4500)


# ---------------------------------------------------------------- kierunek tożsamości (jak w sila.py)
def _ledoit_wolf(X):
    X = X - X.mean(0)
    n, p = X.shape
    X2 = X ** 2
    tr = X2.sum(0) / n
    mu = tr.sum() / p
    beta_ = (X2.T @ X2).sum()
    delta_ = ((X.T @ X) ** 2).sum() / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * tr.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    a = 0.0 if beta == 0 else beta / delta
    C = (1 - a) * (X.T @ X) / n
    C.flat[::p + 1] += a * mu
    return C


def _cov(X):
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return sd[:, None] * _ledoit_wolf(X / sd) * sd[None]


def lda(Z, y):
    pri = [(y == k).mean() for k in (0, 1)]
    M = np.stack([Z[y == k].mean(0) for k in (0, 1)])
    C = sum(pri[k] * _cov(Z[y == k]) for k in (0, 1))
    coef = np.linalg.solve(C, M.T).T
    return coef[1] - coef[0]


def probki(L, ktore):
    """Próbki i etykiety jak w sila.wektor: obszar narzędzia (1, a w empty_black 0) i tło sondy (0)."""
    L = np.asarray(L, np.float64)[ktore]
    ob = OBECN[ktore]
    X = np.concatenate([L[:, 0], L[:, 1]])
    y = np.concatenate([ob.astype(int), np.zeros(len(L), int)])
    return X, y


def wektor(L, ktore=slice(None), prosty=False):
    X, y = probki(L, ktore)
    neg = X[y == 0]
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    Z = (X - mu) / sd
    w = (Z[y == 1].mean(0) - Z[y == 0].mean(0)) if prosty else lda(Z, y)
    w = w / np.linalg.norm(w)
    return dict(mu=mu, sd=sd, w=w)


def rzut(X, wek):
    return ((X - wek['mu']) / wek['sd']) @ wek['w']


def surowy(wek):
    v = wek['w'] / wek['sd']
    return v / np.linalg.norm(v)


def dprim_auc(r, y):
    p, n = r[y == 1], r[y == 0]
    d = (p.mean() - n.mean()) / np.sqrt(0.5 * (p.var() + n.var()) + 1e-12)
    rang = np.argsort(np.argsort(np.concatenate([p, n]))) + 1.0
    auc = (rang[:len(p)].mean() - (len(p) + 1) / 2) / len(n)
    return float(d), float(auc)


def gram_srodkowany(A):
    A = A - A.mean(0)
    return (A @ A.T).astype(np.float32)


def cka(Ka, Kb):
    return float((Ka * Kb).sum() / np.sqrt((Ka * Ka).sum() * (Kb * Kb).sum()))


# ---------------------------------------------------------------- wczytywanie
def wczytaj(kat):
    """Zwraca (kroki, latent float16) ze wszystkich obrotów w katalogu, posortowane po kroku."""
    kr, la = [], []
    for d in sorted(p for p in kat.iterdir() if p.is_dir() and p.name.startswith('k')):
        z = np.load(d / 'LATENT.npz')
        kr.append(z['kroki']); la.append(z['latent'])
    kr = np.concatenate(kr); la = np.concatenate(la)
    _, pierwsze = np.unique(kr, return_index=True)
    return kr[pierwsze], la[pierwsze]


def oceny(kat):
    return {int(o['krok']): float(o['mAP50']) for o in json.loads((kat / 'OCENY.json').read_text(encoding='utf-8'))}


def blok(kroki, lat, t):
    i = int(np.searchsorted(kroki, t))
    assert kroki[i] == t and i >= 4, (t, i)
    return lat[i - 4:i + 1].astype(np.float32).mean(0)                     # średnia z zapisów t-40..t


def miary(L, wek1000, sur1000):
    """Wszystkie miary dla jednej chwili. L: (192, 2, 768)."""
    X, y = probki(L, slice(None))
    r = rzut(X, wek1000)
    out = dict(sila_stala=float(r[:192][OBECN].mean()), d_staly=dprim_auc(r, y)[0])
    sila_sond = r[:192][OBECN].astype(np.float32)
    dl, al, dp = [], [], []
    for test in PODZIALY:
        Xt, yt = probki(L, test)
        for prosty, cel in ((False, dl), (True, dp)):
            d, a = dprim_auc(rzut(Xt, wektor(L, ~test, prosty)), yt)
            cel.append(d)
            if not prosty: al.append(a)
    wk = wektor(L)
    rw = rzut(X, wk)
    out.update(d_odsw=float(np.mean(dl)), auc_odsw=float(np.mean(al)), d_odsw_prosty=float(np.mean(dp)),
               d_odsw_wprobie=dprim_auc(rw, y)[0], cos_1000=float(surowy(wk) @ sur1000))
    return out, sila_sond, surowy(wk).astype(np.float32), gram_srodkowany(np.asarray(L[OBECN, 0], np.float64))


NAZWY = ('sila_stala', 'd_staly', 'd_odsw', 'auc_odsw', 'd_odsw_prosty', 'd_odsw_wprobie', 'cos_1000', 'cos_500', 'cka_koniec')


def pomiar(out):
    S, A, T = len(ZIARNA), len(RAMIONA), len(SIATKA)
    M = {n: np.full((S, A, T), np.nan) for n in NAZWY}
    sila_sond = np.full((S, A, T, 180), np.nan, np.float32)
    mapa = np.full((S, A, 46), np.nan)                                    # kroki 0, 100, ..., 4500
    gram = np.zeros((S, A, len(KROKI_CKA), 180, 180), np.float32)
    kier = np.zeros((S, A, len(KROKI_CKA), 768), np.float32)
    t0 = time.time()
    for si, s in enumerate(ZIARNA):
        kat = ZRODLO / f'petla_s{s}'
        kw, lw = wczytaj(kat / 'wspolny')
        L1000 = blok(kw, lw, 1000)
        wek1000 = wektor(L1000); sur1000 = surowy(wek1000)
        wsp = {}                                                           # miary rozgrzewki, wspólne dla ramion
        for t in SIATKA:
            if t <= 1000:
                wsp[t] = miary(blok(kw, lw, t), wek1000, sur1000)
        for ai, a in enumerate(RAMIONA):
            ka, la = wczytaj(kat / a)
            kroki = np.concatenate([kw, ka]); lat = np.concatenate([lw, la])
            oc = oceny(kat / a)
            mapa[si, ai] = [oc.get(k, np.nan) for k in range(0, 4501, 100)]
            sur, gr = {}, {}
            for ti, t in enumerate(SIATKA):
                m, ss, sv, g = wsp[t] if t <= 1000 else miary(blok(kroki, lat, t), wek1000, sur1000)
                for n, v in m.items(): M[n][si, ai, ti] = v
                sila_sond[si, ai, ti] = ss; sur[t] = sv; gr[t] = g
                if t in KROKI_CKA:
                    gram[si, ai, KROKI_CKA.index(t)] = g; kier[si, ai, KROKI_CKA.index(t)] = sv
            for ti, t in enumerate(SIATKA):
                M['cka_koniec'][si, ai, ti] = cka(gr[t], gr[SIATKA[-1]])
                if t - 500 in sur: M['cos_500'][si, ai, ti] = float(sur[t] @ sur[t - 500])
            del la, lat
            print(f'ziarno {s} ramie {a} gotowe, {time.time() - t0:.0f} s', flush=True)
        del lw
    np.savez_compressed(out / 'POMIAR.npz', siatka=np.array(SIATKA), ziarna=np.array(ZIARNA), ramiona=np.array(RAMIONA),
                        mapa=mapa, sila_sond=sila_sond, gram=gram, kier=kier, kroki_cka=np.array(KROKI_CKA), **M)


# ---------------------------------------------------------------- analiza
def kor_w_chwili(x, y, rng, n_boot=2000):
    """x, y: (S, A, T). Korelacja odchyleń od średniej z ramion w każdej parze (ziarno, chwila); przedział z losowania ziaren."""
    dx, dy = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    r = lambda ii: float(np.corrcoef(dx[ii].ravel(), dy[ii].ravel())[0, 1])
    S = x.shape[0]
    b = [r(rng.integers(0, S, S)) for _ in range(n_boot)]
    return dict(r=r(np.arange(S)), dol=float(np.percentile(b, 2.5)), gora=float(np.percentile(b, 97.5)), n=int(dx.size))


def szablon(q, kroki, starty, h=500):
    """q: (S, A, T) albo (S, A, T, P). Prognoza zmiany na h kroków: średnia zmiana w pozostałych ziarnach (to samo ramię i start)."""
    idx = {int(k): i for i, k in enumerate(kroki)}
    D = np.stack([q[:, :, idx[t + h]] - q[:, :, idx[t]] for t in starty], 2)          # (S, A, n_start[, P])
    S = D.shape[0]
    pred = (D.sum(0, keepdims=True) - D) / (S - 1)
    ost = np.stack([q[:, :, idx[t]] - q[:, :, idx[t - h]] for t in starty], 2)       # ostatnia zmiana jako prognoza
    um = lambda p, d: float(1 - ((d - p) ** 2).sum() / (d ** 2).sum())
    return dict(szablon=um(pred, D), ostatnia_zmiana=um(ost, D),
                szablon_ramiona={a: um(pred[:, i], D[:, i]) for i, a in enumerate(RAMIONA)},
                rms_zmiany=float(np.sqrt((D ** 2).mean())))


def analiza(out):
    z = np.load(out / 'POMIAR.npz')
    siatka = z['siatka']; M = {n: z[n] for n in NAZWY}; mapa = z['mapa']
    k100 = np.arange(0, 4501, 100)
    na100 = lambda v: v[:, :, np.isin(siatka, k100)]                                   # (S, A, 45): kroki 100..4500
    mapa100 = mapa[:, :, 1:]                                                           # kroki 100..4500
    rng = np.random.default_rng(20261005)
    wyn = dict(meta=dict(zrodlo=str(ZRODLO), ziarna=ZIARNA, ramiona=list(RAMIONA), siatka=[int(siatka[0]), int(siatka[-1]), 50]))

    # krzywe średnie po ziarnach
    kr = (500, 1000, 1500, 2000, 2500, 3500, 4500)
    wyn['krzywe'] = {a: {str(t): dict(mAP=float(np.nanmean(mapa[:, ai, t // 100])),
                                      **{n: float(np.nanmean(M[n][:, ai, list(siatka).index(t)])) for n in NAZWY})
                         for t in kr} for ai, a in enumerate(RAMIONA)}

    # K1
    i1, i2 = list(siatka).index(1000), list(siatka).index(4500)
    okno = lambda n, a, b: np.nanmean(M[n][:, :, (siatka >= a) & (siatka <= b)], 2)   # (S, A)
    k1 = {}
    for n in ('d_odsw', 'd_odsw_prosty', 'd_staly', 'sila_stala', 'auc_odsw'):
        d = M[n][:, 0, i2] - M[n][:, 0, i1]
        k1[n] = dict(w_1000=float(M[n][:, 0, i1].mean()), w_4500=float(M[n][:, 0, i2].mean()), wzrost_w_ziarnach=int((d > 0).sum()),
                     srednio_800_1000=float(okno(n, 800, 1000)[:, 0].mean()), srednio_4300_4500=float(okno(n, 4300, 4500)[:, 0].mean()),
                     wzrost_w_ziarnach_okna=int((okno(n, 4300, 4500)[:, 0] > okno(n, 800, 1000)[:, 0]).sum()))
    wyn['K1'] = k1

    # K2
    pm = np.nanmean(mapa[:, :, 35:46], 2)                                             # mAP, kroki 3500..4500
    k2 = dict(mAP={a: dict(srednia=float(pm[:, ai].mean()), wyzsze_niz_bez=int((pm[:, ai] > pm[:, 0]).sum())) for ai, a in enumerate(RAMIONA)})
    for n in ('d_odsw', 'd_odsw_prosty', 'd_staly', 'sila_stala', 'cos_1000', 'auc_odsw'):
        v = okno(n, 3500, 4500)
        k2[n] = {a: dict(srednia=float(v[:, ai].mean()), wyzsze_niz_bez=int((v[:, ai] > v[:, 0]).sum())) for ai, a in enumerate(RAMIONA)}
    wyn['K2'] = k2

    # K3
    od = k100[1:] >= 1500
    wyn['K3'] = {n: kor_w_chwili(mapa100[:, :, od], na100(M[n])[:, :, od], rng) for n in ('d_odsw', 'd_odsw_prosty', 'd_staly', 'sila_stala', 'cos_1000')}
    # to samo dla przyrostów na 500 kroków (zmiana miary wobec zmiany mAP)
    dm = mapa100[:, :, 5:] - mapa100[:, :, :-5]
    odp = k100[6:] >= 1500
    wyn['K3_przyrosty_500'] = {n: kor_w_chwili(dm[:, :, odp], (na100(M[n])[:, :, 5:] - na100(M[n])[:, :, :-5])[:, :, odp], rng)
                               for n in ('d_odsw', 'd_staly', 'sila_stala')}

    # związek w czasie w obrębie biegu (obciążony wspólnym trendem, opisowo): Spearman po krokach >= 1000
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    oc = k100[1:] >= 1000
    wyn['w_czasie_spearman_mediana'] = {n: float(np.median([spearman(mapa100[s, a, oc], na100(M[n])[s, a, oc])
                                                            for s in range(len(ZIARNA)) for a in range(len(RAMIONA))]))
                                        for n in ('d_odsw', 'd_staly', 'sila_stala', 'cos_1000')}

    # K4
    starty = list(range(1000, 4001, 500))
    k4 = dict(mAP=szablon(mapa, k100, starty))
    for n in ('sila_stala', 'd_odsw', 'd_staly', 'cos_1000'):
        k4[n] = szablon(M[n], siatka, starty)
    k4['sila_stala_kazda_sonda'] = szablon(z['sila_sond'].astype(np.float64), siatka, starty)
    wyn['K4'] = k4

    # K5
    g, kier = z['gram'], z['kier']
    k5 = {}
    for ki, t in enumerate(z['kroki_cka']):
        pary = [(i, j) for i in range(len(ZIARNA)) for j in range(i + 1, len(ZIARNA))]
        k5[str(int(t))] = dict(cka_miedzy_ziarnami=float(np.mean([cka(g[i, 0, ki], g[j, 0, ki]) for i, j in pary])),
                               cos_kierunkow_miedzy_ziarnami=float(np.mean([kier[i, 0, ki] @ kier[j, 0, ki] for i, j in pary])),
                               cka_bez_wobec_losowe_to_samo_ziarno=float(np.mean([cka(g[i, 0, ki], g[i, 1, ki]) for i in range(len(ZIARNA))])),
                               cos_bez_wobec_losowe_to_samo_ziarno=float(np.mean([kier[i, 0, ki] @ kier[i, 1, ki] for i in range(len(ZIARNA))])))
    wyn['K5'] = k5
    (out / 'WYNIK.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: wyn[k] for k in ('K1', 'K2', 'K3', 'K3_przyrosty_500', 'w_czasie_spearman_mediana', 'K4', 'K5')}, ensure_ascii=False, indent=1))
    print('krzywe bez:', json.dumps(wyn['krzywe']['bez'], ensure_ascii=False))
    print('krzywe losowe:', json.dumps(wyn['krzywe']['losowe'], ensure_ascii=False))


def uzupelnienie(out):
    """Opis dodany po wyniku (bez kryterium): wczesne kroki, starzenie się kierunku wokół chwili dopasowania,
    prognoza z krzywej innych ziaren bez startu w kroku 1000 (tam kierunek stały jest dopasowany w próbie), wahania mAP."""
    z = np.load(out / 'POMIAR.npz')
    si = list(z['siatka']); k100 = np.arange(0, 4501, 100)
    sr = lambda n, t: float(np.nanmean(z[n][:, 0, si.index(t)]))
    wyn = dict(meta=dict(opis='analiza dodatkowa po wyniku, opisowa; ramię bez (do kroku 1000 wspólne dla ramion)'))
    wyn['w_czasie'] = {str(t): dict(mAP_najblizsza_ocena=float(np.nanmean(z['mapa'][:, 0, t // 100])),
                                   **{n: sr(n, t) for n in ('d_odsw', 'd_odsw_prosty', 'auc_odsw', 'd_staly', 'sila_stala', 'cos_1000', 'cos_500')})
                       for t in (50, 100, 150, 200, 300, 400, 500, 700, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300, 1500, 2000, 2500, 3500, 4500)}
    starty = list(range(1500, 4001, 500))
    k4 = dict(mAP=szablon(z['mapa'], k100, starty))
    for n in ('sila_stala', 'd_odsw', 'd_staly', 'cos_1000'):
        k4[n] = szablon(z[n], z['siatka'], starty)
    k4['sila_stala_kazda_sonda'] = szablon(z['sila_sond'].astype(np.float64), z['siatka'], starty)
    wyn['K4_starty_1500_4000'] = k4
    s = z['sila_stala']
    wyn['sila_stala_zmiennosc'] = dict(sd_miedzy_ziarnami_krok_4500={a: float(s[:, i, -1].std()) for i, a in enumerate(RAMIONA)},
                                       rms_zmiany_50_krokow=float(np.sqrt(np.nanmean(np.diff(s[:, :, 20:], axis=2) ** 2))),
                                       rms_zmiany_500_krokow=float(np.sqrt(np.nanmean((s[:, :, 30:] - s[:, :, 20:-10]) ** 2))))
    m = z['mapa'][:, :, 10:]                                                           # kroki 1000..4500
    wyn['mAP_zmiennosc'] = dict(sd_miedzy_ziarnami={a: float(m[:, i].std(0).mean()) for i, a in enumerate(RAMIONA)},
                                rms_zmiany_100_krokow={a: float(np.sqrt((np.diff(m[:, i], axis=1) ** 2).mean())) for i, a in enumerate(RAMIONA)},
                                rms_zmiany_500_krokow={a: float(np.sqrt(((m[:, i, 5:] - m[:, i, :-5]) ** 2).mean())) for i, a in enumerate(RAMIONA)},
                                srednia_zmiana_500_krokow={a: float((m[:, i, 5:] - m[:, i, :-5]).mean()) for i, a in enumerate(RAMIONA)})
    (out / 'UZUPELNIENIE.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('uzupelnienie:', json.dumps({k: wyn[k] for k in ('K4_starty_1500_4000', 'sila_stala_zmiennosc', 'mAP_zmiennosc')}, ensure_ascii=False)[:1800])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); p.add_argument('--tylko-analiza', action='store_true')
    arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    if not arg.tylko_analiza:
        pomiar(out)
    analiza(out)
    uzupelnienie(out)


if __name__ == '__main__':
    main()
