#!/usr/bin/env python3
"""Serwowizja na pikselach: czy cystotom obraca się wokół nacięcia jak wokół zawiasu (PLAN.md, 2026-09-30).

Dane: Cataract-LMM 3_Object_Tracking, `annotation.csv` każdego filmu (obrys i końcówka cystotomu, środek rogówki
w każdej klatce). Bez uczenia sieci.

Test 1 (zawias): w oknie W klatek oś trzonu każdej klatki to prosta; punkt najbliższy wszystkim prostym (najmniejsze
kwadraty) i mediana odległości prostych od niego. Podłoga: kierunki przetasowane między klatkami okna.
Test 2 (prognoza końcówki w klatce t+k): brak_zmiany, piksele_v (stała prędkość na obrazie), zawias_v (stała prędkość
kąta i odległości końcówki od zawiasu z okna kończącego się na t).
Oba testy we współrzędnych względem środka rogówki ('oko') i surowych ('obraz').

    python zawias.py [--W 30 --krok-okna 15 --filmy 0 --podglad 6]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
VIT = HERE.parents[2]
ARCH = VIT / 'Dataset' / 'Cataract-LMM' / '3_Object_Tracking' / 'annotations'
WYN = HERE / 'wyniki'

MIN_WYDL = 4.0                      # wydłużenie obrysu sqrt(λ1/λ2); krótsza plama nie wyznacza osi
MIN_ROZRZUT = math.radians(2.0)     # rozrzut kątów osi w oknie; przy mniejszym punkt przecięcia jest nieokreślony
MIN_KLATEK = 20
LOSOWAN = 20
MODELE = ('brak_zmiany', 'piksele_v', 'zawias_v', 'piksele_vb', 'zawias_vb')


def js(s):
    return None if s in ('', '[]', None) else json.loads(s)


def wczytaj(v):
    """Tablice gęste po numerze klatki: końcówka, środek obrysu, oś, wydłużenie, środek i promień rogówki."""
    with zipfile.ZipFile(ARCH / f'{v}.zip') as z:
        rows = list(csv.DictReader(io.StringIO(z.read(f'{v}/annotation.csv').decode('utf-8-sig'))))
    F = np.array([int(r['frame_number']) for r in rows])
    f0, n = F.min(), F.max() - F.min() + 1
    tip, c, u, oko = (np.full((n, 2), np.nan) for _ in range(4))
    wy, rr = np.full(n, np.nan), np.full(n, np.nan)
    pliki = [''] * n
    for r, f in zip(rows, F):
        i = f - f0
        pliki[i] = r['filename']
        cc, bb = js(r['cornea_center']), js(r['cornea_bbox'])
        if cc is not None and bb is not None:
            oko[i] = cc; rr[i] = (bb[2] + bb[3]) / 4
        t, seg = js(r['cystotome_tip']), js(r['cystotome_seg'])
        if t is None or seg is None or len(t) != 2:
            continue
        M = np.zeros(6)
        for p in seg:
            p = np.asarray(p, np.float32).reshape(-1, 2)
            if len(p) < 3:
                continue
            m = cv2.moments(p)
            s = -1.0 if m['m00'] < 0 else 1.0
            M += s * np.array([m['m00'], m['m10'], m['m01'], m['m20'], m['m11'], m['m02']])
        if M[0] <= 1:
            continue
        cx, cy = M[1] / M[0], M[2] / M[0]
        C = np.array([[M[3] / M[0] - cx * cx, M[4] / M[0] - cx * cy], [M[4] / M[0] - cx * cy, M[5] / M[0] - cy * cy]])
        w, V = np.linalg.eigh(C)
        if not w[1] > 0:                                   # zdegenerowany obrys (samoprzecięcia)
            continue
        e = V[:, 1]
        if (np.array([cx, cy]) - np.asarray(t, float)) @ e < 0:
            e = -e
        tip[i] = t; c[i] = (cx, cy); u[i] = e; wy[i] = math.sqrt(w[1] / max(w[0], 1e-6))
    return dict(tip=tip, c=c, u=u, wy=wy, oko=oko, rr=rr, pliki=pliki)


def rzut(u):
    """(n,2) kierunki -> (n,2,2) rzut prostopadły do osi."""
    return np.eye(2)[None] - u[:, :, None] * u[:, None, :]


def zawias(cc, uu):
    P = rzut(uu)
    q = np.linalg.solve(P.sum(0), np.einsum('nij,nj->i', P, cc))
    return q, np.linalg.norm(np.einsum('nij,nj->ni', P, cc - q), axis=1)


def rozrzut(uu):
    """Rozrzut kątów osi (dane osiowe: kąt podwojony)."""
    R = abs(np.mean(np.exp(2j * np.arctan2(uu[:, 1], uu[:, 0]))))
    return math.sqrt(-2 * math.log(max(R, 1e-12))) / 2


def uklady(d):
    dobre = np.isfinite(d['tip'][:, 0]) & (d['wy'] >= MIN_WYDL)
    oko_ok = np.isfinite(d['oko'][:, 0])
    return {'oko': (d['tip'] - d['oko'], d['c'] - d['oko'], dobre & oko_ok),
            'obraz': (d['tip'], d['c'], dobre)}


def test_zawiasu(d, W, krok, rng):
    wyn = {}
    for uk, (tip, c, ok) in uklady(d).items():
        okna = []
        for a in range(0, len(ok) - W + 1, krok):
            j = np.where(ok[a:a + W])[0] + a
            if len(j) < MIN_KLATEK:
                continue
            uu, cc = d['u'][j], c[j]
            s = rozrzut(uu)
            if s < MIN_ROZRZUT:
                continue
            q, dist = zawias(cc, uu)
            null = [np.median(zawias(cc, uu[rng.permutation(len(j))])[1]) for _ in range(LOSOWAN)]
            R = np.nanmedian(d['rr'][j])
            srodek_oka = np.zeros(2) if uk == 'oko' else np.nanmedian(d['oko'][j], 0)
            s_konc = ((q - tip[j]) * uu).sum(1)                   # > 0: zawias za końcówką, po stronie trzonu
            okna.append(dict(start=int(a), n=int(len(j)), rozrzut_st=round(math.degrees(s), 2),
                             odl_px=float(np.median(dist)), odl_null_px=float(np.median(null)),
                             strona_trzonu=float(np.mean(s_konc > 0)),
                             strona_trzonu_za_srodkiem=float(np.mean(((q - cc) * uu).sum(1) > 0)),
                             zawias_od_konc_px=float(np.median(s_konc)),
                             q_przez_promien=float(np.linalg.norm(q - srodek_oka) / R) if np.isfinite(R) else None,
                             q=[float(q[0]), float(q[1])]))
        wyn[uk] = okna
    return wyn


def zawias_biezacy(ok, u, c, W):
    """Zawias q_t z okna [t−W+1, t] (tylko przeszłość), sumami bieżącymi; NaN, gdy okno za słabe."""
    n = len(ok)
    uu = np.nan_to_num(u)
    P = np.where(ok[:, None, None], rzut(uu), 0.0)
    Pc = np.where(ok[:, None], np.einsum('nij,nj->ni', P, np.nan_to_num(c)), 0.0)
    z2 = np.where(ok, np.exp(2j * np.arctan2(uu[:, 1], uu[:, 0])), 0)
    cum = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), np.cumsum(x, 0)])
    CP, CPc, Cz, Cn = cum(P), cum(Pc), cum(z2), cum(ok.astype(int))
    t = np.arange(W - 1, n)
    A, b = CP[t + 1] - CP[t + 1 - W], CPc[t + 1] - CPc[t + 1 - W]
    cnt = Cn[t + 1] - Cn[t + 1 - W]
    R = np.abs(Cz[t + 1] - Cz[t + 1 - W]) / np.maximum(cnt, 1)
    spr = np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))) / 2
    q = np.full((n, 2), np.nan)
    dobry = (cnt >= MIN_KLATEK) & (spr >= MIN_ROZRZUT)
    q[t[dobry]] = np.linalg.solve(A[dobry], b[dobry][..., None])[..., 0]
    return q


def test_prognozy(d, W, ks=(1, 5)):
    """Trójki (p_{t−k}, p_t, p_{t+k}) z zawiasem q_t (modele liczone w podsumowaniu) oraz test kątów.

    Test kątów (dodany po próbie na 3 filmach): równanie serwowizji w rzucie płaskim mówi, że składowa ruchu końcówki
    prostopadła do osi = s · obrót osi, gdzie s to odległość końcówki od zawiasu wzdłuż osi. Obrót osi między t a t+k
    przewidujemy z ruchu końcówki i zawiasu z przeszłości; podłoga: zawias z innej chwili tego samego filmu."""
    wyn = {}
    for uk, (tip, c, ok) in uklady(d).items():
        n = len(ok)
        q = zawias_biezacy(ok, d['u'], c, W)
        wk = {}
        maj = np.isfinite(tip[:, 0])
        for k in ks:
            tt = np.arange(k, n - k)
            tt = tt[maj[tt - k] & maj[tt] & maj[tt + k] & np.isfinite(q[tt, 0])]
            ta = np.arange(0, n - k)
            ta = ta[ok[ta] & ok[ta + k] & np.isfinite(q[ta, 0])]
            u0, u1 = d['u'][ta], d['u'][ta + k]
            nrm = np.stack([-u0[:, 1], u0[:, 0]], 1)
            dfi = np.angle(np.exp(1j * (np.arctan2(u1[:, 1], u1[:, 0]) - np.arctan2(u0[:, 1], u0[:, 0]))))
            inne = ta[np.random.default_rng(k).permutation(len(ta))]
            wk[k] = dict(p0=tip[tt - k], p1=tip[tt], p2=tip[tt + k], q=q[tt],
                         katy=dict(dfi=dfi, dp=((tip[ta + k] - tip[ta]) * nrm).sum(1),
                                   s=((tip[ta] - q[ta]) * u0).sum(1), s_null=((tip[ta] - q[inne]) * u0).sum(1)))
        wyn[uk] = wk
    return wyn


def biegunowe(p, q):
    v = p - q
    return np.arctan2(v[:, 1], v[:, 0]), np.linalg.norm(v, axis=1)


def modele(x, bp, bf, br):
    """Prognozy końcówki w t+k: stała prędkość (β = 1) i tłumiona (β z drugiej połowy filmów), na obrazie i wokół
    zawiasu (kąt i odległość końcówki od zawiasu q_t)."""
    p0, p1, q = x['p0'], x['p1'], x['q']
    (f0, r0), (f1, r1) = biegunowe(p0, q), biegunowe(p1, q)
    df, dr = np.angle(np.exp(1j * (f1 - f0))), r1 - r0
    wokol = lambda a, b: q + (r1 + b * dr)[:, None] * np.stack([np.cos(f1 + a * df), np.sin(f1 + a * df)], 1)
    return {'brak_zmiany': p1, 'piksele_v': 2 * p1 - p0, 'zawias_v': wokol(1.0, 1.0),
            'piksele_vb': p1 + bp * (p1 - p0), 'zawias_vb': wokol(bf, br)}


def bety(lista):
    """Tłumiona stała prędkość: β dla pikseli, kąta wokół zawiasu i odległości od zawiasu (najmniejsze kwadraty)."""
    P0, P1, P2, Q = (np.concatenate([x[k] for x in lista]) for k in ('p0', 'p1', 'p2', 'q'))
    d1, d2 = P1 - P0, P2 - P1
    (f0, r0), (f1, r1), (f2, r2) = biegunowe(P0, Q), biegunowe(P1, Q), biegunowe(P2, Q)
    a1, a2 = np.angle(np.exp(1j * (f1 - f0))), np.angle(np.exp(1j * (f2 - f1)))
    return (float((d1 * d2).sum() / (d1 * d1).sum()), float((a1 * a2).sum() / (a1 * a1).sum()),
            float(((r1 - r0) * (r2 - r1)).sum() / ((r1 - r0) ** 2).sum()))


def znak(w, n):
    return float(sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n)


def podsumuj_prognozy(wszystkie):
    filmy = sorted(wszystkie)
    polowy = [filmy[0::2], filmy[1::2]]
    out = {}
    for uk in ('oko', 'obraz'):
        for k in (1, 5):
            B = {}
            for i, pol in enumerate(polowy):                          # β z jednej połowy filmów, ocena na drugiej
                uczace = [wszystkie[v][uk][k] for v in polowy[1 - i] if len(wszystkie[v][uk][k]['p0'])]
                if not uczace:
                    continue
                b = bety(uczace)
                out.setdefault(f'{uk}/k{k}/bety_piksele_kat_odleglosc', []).append([round(x, 3) for x in b])
                for v in pol:
                    x = wszystkie[v][uk][k]
                    if len(x['p0']) < 20:
                        continue
                    m = modele(x, *b)
                    B[v] = dict(bledy={mo: np.linalg.norm(m[mo] - x['p2'], axis=1) for mo in MODELE},
                                ruch=np.linalg.norm(x['p1'] - x['p0'], axis=1))
            for zakres in ('wszystkie', 'ruch_2px'):
                E = {mo: [] for mo in MODELE}; na_film = []
                for x in B.values():
                    sel = np.ones_like(x['ruch'], bool) if zakres == 'wszystkie' else x['ruch'] >= 2
                    if sel.sum() < 20:
                        continue
                    for mo in MODELE:
                        E[mo].append(x['bledy'][mo][sel])
                    na_film.append({mo: float(np.median(x['bledy'][mo][sel])) for mo in MODELE})
                if not na_film:
                    continue
                E = {mo: np.concatenate(E[mo]) for mo in MODELE}
                sse0 = float((E['brak_zmiany'] ** 2).sum()); nf = len(na_film)
                znaki = {}
                for a_, b_ in (('zawias_v', 'piksele_v'), ('zawias_vb', 'piksele_vb'), ('zawias_vb', 'brak_zmiany'),
                               ('piksele_vb', 'brak_zmiany')):
                    w = sum(f[a_] < f[b_] for f in na_film)
                    znaki[f'{a_}_lepszy_od_{b_}'] = dict(filmow=f'{w}/{nf}', p=znak(w, nf))
                out[f'{uk}/k{k}/{zakres}'] = dict(
                    klatek=int(len(E['brak_zmiany'])), filmow=nf,
                    mediana_bledu_px={mo: round(float(np.median(E[mo])), 3) for mo in MODELE},
                    umiejetnosc={mo: round(100 * (1 - float((E[mo] ** 2).sum()) / sse0), 2) for mo in MODELE},
                    test_znakow=znaki)
    return out


def podsumuj_katy(wszystkie, min_s=20.0, max_dfi=math.radians(30)):
    """Obrót osi przewidziany równaniem serwowizji: dfi_hat = (ruch końcówki prostopadły do osi) / s."""
    out = {}
    for uk in ('oko', 'obraz'):
        for k in (1, 5):
            D, H, H0, na_film = [], [], [], []
            for w in wszystkie.values():
                x = w[uk][k]['katy']
                m = (np.abs(x['s']) >= min_s) & (np.abs(x['s_null']) >= min_s) & (np.abs(x['dfi']) < max_dfi)
                if m.sum() < 20:
                    continue
                dfi, h, h0 = x['dfi'][m], x['dp'][m] / x['s'][m], x['dp'][m] / x['s_null'][m]
                D.append(dfi); H.append(h); H0.append(h0)
                u = lambda hh: 100 * (1 - ((dfi - hh) ** 2).sum() / (dfi ** 2).sum())
                na_film.append((u(h), u(h0)))
            if not D:
                continue
            D, H, H0 = np.concatenate(D), np.concatenate(H), np.concatenate(H0)
            um = lambda hh: round(float(100 * (1 - ((D - hh) ** 2).sum() / (D ** 2).sum())), 2)
            w = sum(a_ > b_ for a_, b_ in na_film); nf = len(na_film)
            out[f'{uk}/k{k}'] = dict(
                par=int(len(D)), filmow=nf,
                umiejetnosc_zawias=um(H), umiejetnosc_zawias_z_innej_chwili=um(H0), umiejetnosc_bez_obrotu=0.0,
                nachylenie_dopasowane=round(float((D * H).sum() / (H * H).sum()), 3),
                korelacja=round(float(np.corrcoef(D, H)[0, 1]), 3),
                korelacja_null=round(float(np.corrcoef(D, H0)[0, 1]), 3),
                mediana_obrotu_st=round(float(np.degrees(np.median(np.abs(D)))), 3),
                mediana_bledu_st=round(float(np.degrees(np.median(np.abs(D - H)))), 3),
                mediana_umiejetnosci_filmu=round(float(np.median([a_ for a_, _ in na_film])), 2),
                filmy_zawias_lepszy_od_null=f'{w}/{nf}', p_znakow=znak(w, nf))
    return out


def podsumuj_zawias(wszystkie):
    out = {}
    for uk in ('oko', 'obraz'):
        o = [x for w in wszystkie.values() for x in w[uk]]
        if not o:
            continue
        il = np.array([x['odl_px'] / max(x['odl_null_px'], 1e-9) for x in o])
        qr = np.array([x['q_przez_promien'] for x in o if x['q_przez_promien'] is not None])
        out[uk] = dict(okien=len(o), filmow=sum(1 for w in wszystkie.values() if w[uk]),
                       mediana_odl_px=round(float(np.median([x['odl_px'] for x in o])), 2),
                       mediana_odl_null_px=round(float(np.median([x['odl_null_px'] for x in o])), 2),
                       mediana_ilorazu=round(float(np.median(il)), 3),
                       okien_iloraz_ponizej_0_5=round(float(np.mean(il < 0.5)), 3),
                       okien_iloraz_ponizej_1=round(float(np.mean(il < 1)), 3),
                       strona_trzonu_mediana=round(float(np.median([x['strona_trzonu'] for x in o])), 3),
                       q_przez_promien_kwartyle=[round(float(v), 2) for v in np.percentile(qr, [25, 50, 75])],
                       q_w_pasie_0_7_do_1_5_promienia=round(float(np.mean((qr > 0.7) & (qr < 1.5))), 3),
                       rozrzut_st_mediana=round(float(np.median([x['rozrzut_st'] for x in o])), 2))
    return out


def podglad(filmy_dane, zaw, ile, rng, sciezka):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kand = [(v, x) for v, w in zaw.items() for x in w['obraz']]
    if not kand:
        return
    wyb = [kand[i] for i in rng.choice(len(kand), size=min(ile, len(kand)), replace=False)]
    fig, ax = plt.subplots(2, (len(wyb) + 1) // 2, figsize=(5 * ((len(wyb) + 1) // 2), 7))
    for a, (v, x) in zip(np.ravel(ax), wyb):
        d = filmy_dane[v]
        j = [i for i in range(x['start'], x['start'] + 30) if np.isfinite(d['tip'][i, 0]) and d['wy'][i] >= MIN_WYDL]
        sr = j[len(j) // 2]
        with zipfile.ZipFile(ARCH / f'{v}.zip') as z:
            im = cv2.imdecode(np.frombuffer(z.read(f'{v}/{d["pliki"][sr]}'), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        a.imshow(im)
        for i in j:
            c, u = d['c'][i], d['u'][i]
            L = 400
            a.plot([c[0] - L * u[0], c[0] + L * u[0]], [c[1] - L * u[1], c[1] + L * u[1]], lw=0.6, alpha=0.6, color='cyan')
            a.plot(*d['tip'][i], '.', ms=3, color='yellow')
        a.plot(*x['q'], 'o', ms=9, mfc='none', mec='red', mew=2)
        a.set_xlim(0, im.shape[1]); a.set_ylim(im.shape[0], 0); a.set_xticks([]); a.set_yticks([])
        a.set_title(f'{v}, klatki {x["start"]}–{x["start"] + 29}\n'
                    f'odległość osi {x["odl_px"]:.1f} px, po przetasowaniu {x["odl_null_px"]:.1f} px', fontsize=9)
    for a in np.ravel(ax)[len(wyb):]:
        a.axis('off')
    fig.suptitle('Osie cystotomu z 30 klatek (niebieskie), końcówki (żółte) i wspólny punkt osi (czerwony); '
                 'okna wylosowane', fontsize=11)
    fig.tight_layout()
    fig.savefig(sciezka, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=30)
    ap.add_argument('--krok-okna', type=int, default=15)
    ap.add_argument('--filmy', type=int, default=0, help='0 = wszystkie')
    ap.add_argument('--podglad', type=int, default=6)
    ap.add_argument('--wynik', default=str(WYN / 'ZAWIAS.json'))
    a = ap.parse_args()
    WYN.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    filmy = sorted(p.stem for p in ARCH.glob('TR_*.zip'))[: a.filmy or None]
    t0 = time.time(); dane, zaw, prog, na_film = {}, {}, {}, {}
    for i, v in enumerate(filmy):
        d = wczytaj(v)
        zaw[v] = test_zawiasu(d, a.W, a.krok_okna, rng)
        prog[v] = test_prognozy(d, a.W)
        dane[v] = d
        dobre = int((np.isfinite(d['tip'][:, 0]) & (d['wy'] >= MIN_WYDL)).sum())
        na_film[v] = dict(klatek=len(d['tip']), z_osia=dobre, okien_oko=len(zaw[v]['oko']), okien_obraz=len(zaw[v]['obraz']))
        if i % 10 == 0 or i == len(filmy) - 1:
            print(f'{i + 1}/{len(filmy)} {v}: klatek {len(d["tip"])}, z osią {dobre}, okien {len(zaw[v]["oko"])} '
                  f'({time.time() - t0:.0f} s)', flush=True)
    wyn = dict(parametry=dict(W=a.W, krok_okna=a.krok_okna, min_wydluzenie=MIN_WYDL,
                              min_rozrzut_st=math.degrees(MIN_ROZRZUT), min_klatek=MIN_KLATEK, losowan_null=LOSOWAN),
               zawias=podsumuj_zawias(zaw), prognoza=podsumuj_prognozy(prog), katy=podsumuj_katy(prog),
               filmy=na_film,
               okna=zaw)
    Path(a.wynik).write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zawias', json.dumps(wyn['zawias'], ensure_ascii=False, indent=1), flush=True)
    print('prognoza', json.dumps(wyn['prognoza'], ensure_ascii=False, indent=1), flush=True)
    print('katy', json.dumps(wyn['katy'], ensure_ascii=False, indent=1), flush=True)
    if a.podglad:
        podglad(dane, zaw, a.podglad, np.random.default_rng(1), WYN / 'ZAWIAS_PODGLAD.png')
    print(f'gotowe {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
