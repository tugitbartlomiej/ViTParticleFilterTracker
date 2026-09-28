#!/usr/bin/env python3
"""Analiza wsteczna na zapisanych latentach (PLAN_NASTEPNE_SPRAWDZENIE.md, sekcja „Analiza wsteczna”).

Bez GPU. Dla każdego ziarna sklejamy latent rozgrzewki (kroki 10–1500) z dalszym ciągiem jednego ramienia (1510–4500),
uśredniamy bloki po 5 zapisów (punkt co 50 kroków; blok b ↔ krok 50(b+1)) i liczymy:
  1. narodziny tożsamości: pełny latent, rdzeń, odległość do rdzenia, domknięcie wariantów, stabilność zbioru;
  2. prognozę z chwili t (co 250 kroków): neurony i standaryzacja tylko z danych do t, Koopman i Langevin dopasowane
     do historii do t (jak prognoza.py), błąd na H blokach naprzód względem „braku zmiany” i „trendu”;
  3. kierunek: zgodność znaku przewidzianej i prawdziwej zmiany neuronów tożsamości;
  4. mAP@0,5 z ocen co 100 kroków.
Reguła wyboru chwili startu jest zapisana w planie przed liczeniem.

    py -3.11 analiza_wsteczna.py  ->  wyniki/ANALIZA_WSTECZNA.json, wyniki/ANALIZA_WSTECZNA.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prognoza as PR  # noqa: E402
import tozsamosc as TZ  # noqa: E402
from dane import PROBE, jz  # noqa: E402

RAMIONA = ('bez', 'losowe', 'koopman')
HORYZONTY = (1, 5, 10, 30)                  # bloki po 50 kroków: 50, 250, 500, 1500 kroków
POCZATKI = list(range(500, 4500, 250))      # chwile startu prognozy (krok); 250 daje za mało punktów historii
TREND_Z = 5                                 # nachylenie „trendu” z ostatnich 5 bloków
EPOKA = 1500 / 32                           # kroków na epokę bazy (paczka 32)


def katalogi():
    return [HERE / 'wyniki' / 'petla_s42'] + sorted((HERE / 'wyniki_eden').glob('petla_s*'))


def bloki_ramienia(d, ramie, wsp):
    arm = [np.load(d / ramie / f'k{k:02d}' / 'LATENT.npz') for k in range(4, 10)]
    kroki = np.concatenate([wsp['kroki']] + [a['kroki'] for a in arm])
    L = np.concatenate([wsp['latent']] + [a['latent'] for a in arm])
    assert kroki[0] == 0 and np.all(np.diff(kroki) == 10), 'nieciągłe zapisy latentu'
    L = L[1:]                                                        # bez kroku 0 → kroki 10..4500
    nb = len(L) // PR.B_BLOK
    return L[:nb * PR.B_BLOK].astype(np.float32).reshape(nb, PR.B_BLOK, *L.shape[1:]).mean(1)


def tozsamosc_w_czasie(Lb, obecn, opis):
    wyn, zbiory = [], []
    for b in range(len(Lb)):
        st = TZ.statystyki(Lb[b], obecn, opis)
        pod = TZ.podsumowanie(st)
        sel = st['frac_neg'] <= TZ.NEG_MAX
        min_war = st['zapal_warianty'].min(0)
        wyn.append(dict(krok=50 * (b + 1), pelny=pod['pelny'], rdzen=pod['rdzen'],
                        do_rdzenia=float(min_war[sel].max()) if sel.any() else 0.0,
                        domkniecie=pod['domkniecie_srednie'], warianty=pod['domkniecie_wariantow']))
        zbiory.append(set(st['pelny'].tolist()))
    J = lambda a, b: len(a & b) / max(1, len(a | b))  # noqa: E731
    for b, w in enumerate(wyn):
        w['jaccard_250_wstecz'] = J(zbiory[b], zbiory[b - 5]) if b >= 5 else None
        w['jaccard_do_konca'] = J(zbiory[b], zbiory[-1])
    return wyn


def kierunek_tozsamosci(Lb, obecn, opis):
    """Wektor tożsamości na wszystkich 768 neuronach: średnia na narzędziu minus średnia negatywów, w jednostkach sd
    negatywów. Stabilność: cosinus z chwilą sprzed 250 kroków i z końcem; zgodność wariantów: średni cosinus między
    kierunkami liczonymi osobno dla 15 wariantów."""
    war = np.array([o['wariant'] for o in opis])
    nar, tlo = Lb[:, :, 0], Lb[:, :, 1]
    neg = np.concatenate([tlo, nar[:, ~obecn]], 1)
    mu, sd = neg.mean(1), neg.std(1) + 1e-6
    u = (nar[:, obecn].mean(1) - mu) / sd
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    wyn = []
    for b in range(len(Lb)):
        V = np.array([(nar[b, obecn & (war == v)].mean(0) - mu[b]) / sd[b] for v in TZ.WARIANTY_Z])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        S = V @ V.T
        wyn.append(dict(krok=50 * (b + 1), cos_250_wstecz=float(u[b] @ u[b - 5]) if b >= 5 else None,
                        cos_do_konca=float(u[b] @ u[-1]), zgodnosc_wariantow=float(S[np.triu_indices(len(V), 1)].mean())))
    return wyn


def prognozy(Lb, obecn, opis):
    wyn = []
    for t in POCZATKI:
        b0 = t // 50 - 1
        st = TZ.statystyki(Lb[b0], obecn, opis)
        neur = TZ.sledzone(st)
        Z = (Lb[:, obecn, 0][..., neur] - st['mu'][neur]) / st['sd'][neur]          # (B, 180, M)
        hist = Z[:b0 + 1]
        m, V = PR.pca(hist)
        Y = (hist - m) @ V
        W = {mod: PR.dopasuj(Y, mod) for mod in ('koopman', 'langevin')}
        teraz = hist[-1]
        nach = (hist[-1] - hist[-1 - TREND_Z]) / TREND_Z
        for H in HORYZONTY:
            if b0 + H >= len(Z):
                continue
            real = Z[b0 + H]
            prog = {mod: teraz + (PR.rollout(Y, W[mod], H, mod)[-1] - Y[-1]) @ V.T for mod in W}
            prog['trend'] = teraz + nach * H
            e0 = float(np.sqrt(np.mean((teraz - real) ** 2)))
            da = (real - teraz).mean(0)
            duze = np.abs(da) >= np.median(np.abs(da))
            r = dict(krok=t, H=H, neuronow=int(len(neur)), rmse_brak_zmiany=e0)
            for mod, p in prog.items():
                e = float(np.sqrt(np.mean((p - real) ** 2)))
                dp = (p - teraz).mean(0)
                rho = np.corrcoef(np.argsort(np.argsort(dp)), np.argsort(np.argsort(da)))[0, 1]
                r[mod] = dict(rmse=e, wzgl_brak_zmiany=e / max(e0, 1e-12),
                              znak=float((np.sign(dp[duze]) == np.sign(da[duze])).mean()), spearman=float(rho))
            for mod in ('koopman', 'langevin'):
                r[mod]['wzgl_trend'] = r[mod]['rmse'] / max(r['trend']['rmse'], 1e-12)
            wyn.append(r)
    return wyn


def map_krzywa(d, ramie):
    oc = jz(d / ramie / 'KONIEC.json')['oceny']
    return [(o['krok'], o['mAP50']) for o in oc if o['ramie'] in ('wspolny', ramie)]


def regula(zb, H=10):
    """Reguła z planu: mediana < brak zmiany i < trend, ≥ 8 z 11 ziaren < brak zmiany, i tak w ≥ 3 z 4 kolejnych."""
    out = {}
    for mod in ('koopman', 'langevin'):
        ok = []
        for t in POCZATKI:
            rr = [r[mod] for s in zb.values() for r in s['prognozy'] if r['krok'] == t and r['H'] == H]
            if not rr:
                continue
            nc = np.array([x['wzgl_brak_zmiany'] for x in rr]); tr = np.array([x['wzgl_trend'] for x in rr])
            ok.append((t, bool(np.median(nc) < 1 and np.median(tr) < 1 and (nc < 1).sum() >= min(8, len(nc) - 3))))
        start = None
        for i, (t, v) in enumerate(ok):
            nast = [x for _, x in ok[i + 1:i + 5]]
            if v and (sum(nast) >= 3 if len(nast) >= 4 else all(nast)):
                start = t; break
        out[mod] = dict(start_krok=start, start_epoka=None if start is None else round(start / EPOKA, 1),
                        spelnione=[t for t, v in ok if v])
    return out


def main() -> int:
    man = jz(PROBE / 'manifest.json')
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    opis = [dict(wariant=m['variant']) for m in man]
    wyn = dict(opis='analiza wsteczna latentów (PLAN_NASTEPNE_SPRAWDZENIE.md)', krokow_na_epoke=EPOKA,
               horyzonty_blokow=HORYZONTY, ramiona={})
    t0 = time.time()
    for ramie in RAMIONA:
        zb = {}
        for d in katalogi():
            s = d.name.split('_s')[1]
            wsp_l = [np.load(d / 'wspolny' / f'k{k:02d}' / 'LATENT.npz') for k in (1, 2, 3)]
            wsp = dict(kroki=np.concatenate([w['kroki'] for w in wsp_l]), latent=np.concatenate([w['latent'] for w in wsp_l]))
            Lb = bloki_ramienia(d, ramie, wsp)
            zb[s] = dict(tozsamosc=tozsamosc_w_czasie(Lb, obecn, opis), kierunek=kierunek_tozsamosci(Lb, obecn, opis),
                         prognozy=prognozy(Lb, obecn, opis), mAP=map_krzywa(d, ramie))
            print(f'{ramie} s{s}: {len(Lb)} bloków ({time.time() - t0:.0f} s)', flush=True)
        wyn['ramiona'][ramie] = dict(ziarna=zb, regula_H10=regula(zb, 10), regula_H30=regula(zb, 30))
    (HERE / 'wyniki' / 'ANALIZA_WSTECZNA.json').write_text(json.dumps(wyn, ensure_ascii=False, default=float),
                                                            encoding='utf-8')
    raport(wyn)
    return 0


def raport(wyn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    zb = wyn['ramiona']['bez']['ziarna']
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ep = lambda k: np.asarray(k) / EPOKA  # noqa: E731
    # mAP
    K = sorted({k for s in zb.values() for k, _ in s['mAP']})
    A = np.array([[dict(s['mAP']).get(k, np.nan) for k in K] for s in zb.values()])
    a = ax[0, 0]; a.plot(ep(K), np.nanmedian(A, 0), 'k'); a.fill_between(ep(K), np.nanmin(A, 0), np.nanmax(A, 0), alpha=.15)
    a.set_title('mAP@0,5 (rozgrzewka + ramię bez), mediana i zakres 11 ziaren'); a.set_xlabel('epoka bazy')
    # tożsamość
    kr = [w['krok'] for w in next(iter(zb.values()))['tozsamosc']]
    for klucz, a, tyt in (('pelny', ax[0, 1], 'pełny latent (neuronów)'), ('do_rdzenia', ax[0, 2],
                          'odległość do rdzenia: najlepszy neuron, min. udział po wariantach (rdzeń ≥ 0,75)')):
        M_ = np.array([[w[klucz] for w in s['tozsamosc']] for s in zb.values()], float)
        a.plot(ep(kr), np.median(M_, 0), 'k'); a.fill_between(ep(kr), np.percentile(M_, 25, 0), np.percentile(M_, 75, 0), alpha=.2)
        a.set_title(tyt, fontsize=9); a.set_xlabel('epoka bazy')
    ax[0, 2].axhline(TZ.RDZEN_MIN, color='r', ls=':')
    a = ax[1, 0]
    for zrodlo, klucz, c, ls in (('tozsamosc', 'jaccard_250_wstecz', 'tab:blue', '-'), ('tozsamosc', 'jaccard_do_konca', 'tab:red', '-'),
                                 ('kierunek', 'cos_250_wstecz', 'tab:blue', '--'), ('kierunek', 'cos_do_konca', 'tab:red', '--')):
        M_ = np.array([[np.nan if w[klucz] is None else w[klucz] for w in s[zrodlo]] for s in zb.values()], float)
        a.plot(ep(kr), np.nanmedian(M_, 0), color=c, ls=ls, label=f'{klucz} ({"lista neuronów" if zrodlo == "tozsamosc" else "wektor 768"})')
    a.legend(fontsize=7); a.set_title('stabilność tożsamości: lista neuronów (Jaccard) i wektor (cosinus)', fontsize=9)
    a.set_xlabel('epoka bazy')
    # prognoza
    for a, H in ((ax[1, 1], 10), (ax[1, 2], 30)):
        for mod, c in (('langevin', 'tab:red'), ('koopman', 'tab:blue'), ('trend', 'tab:gray')):
            R = np.array([[next((r[mod]['wzgl_brak_zmiany'] for r in s['prognozy'] if r['krok'] == t and r['H'] == H), np.nan)
                           for t in POCZATKI] for s in zb.values()])
            a.plot(ep(POCZATKI), np.nanmedian(R, 0), color=c, marker='o', ms=3, label=mod)
        a.axhline(1, color='k', ls=':'); a.set_yscale('log'); a.legend(fontsize=8)
        a.set_title(f'błąd prognozy / brak zmiany, horyzont {H * 50} kroków (mediana ziaren)', fontsize=9)
        a.set_xlabel('epoka bazy w chwili prognozy')
    fig.tight_layout(); fig.savefig(HERE / 'wyniki' / 'ANALIZA_WSTECZNA.png', dpi=120)
    for ramie, r in wyn['ramiona'].items():
        print(ramie, 'reguła H10:', r['regula_H10'], ' H30:', r['regula_H30'])


if __name__ == '__main__':
    raise SystemExit(main())
