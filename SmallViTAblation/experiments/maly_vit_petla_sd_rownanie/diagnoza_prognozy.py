#!/usr/bin/env python3
"""Diagnoza (prośba użytkownika 2026-09-30): gdzie myli się prognoza latentu tożsamości w pętli 4.1?

Ten sam mały ViT i ta sama rozgrzewka co w pętli (ziarno 47, 2 × 500 kroków na bazie 1500 klatek), potem 400 kroków
zwykłego uczenia na bazie (jak ramię `bez`) z pomiarem PO KAŻDYM KROKU. Rozdziela źródła błędu:

A. Tożsamość: czy wektor tożsamości (LDA jak w wersji 3) odróżnia narzędzie od tła i od TEGO SAMEGO miejsca bez
   narzędzia (pary z `maly_vit_tozsamosc_tla/tla`: `oko` = narzędzie zamalowane, `czarne`); AUC na źródłach wyłączonych
   z dopasowania; dla porównania wektor „nowy” z par.
B. Szum pomiaru: siła z latentu bf16 (jak w pętli) kontra float32 dla tych samych wag, wobec typowej zmiany w 100 krokach.
C. Równanie na jeden krok: Δs(t+1) = a·Δs(t) + b·S(t) z siłą prawdziwej paczki (tryb wyjaśnienia, jak docs/23),
   dopasowanie na krokach 1000-1250, test 1250-1400; do tego pełny wirtualny krok AdamW (paczka + pęd + zanik wag) wobec
   prawdziwej zmiany: różnica = wkład buforów BatchNorm.
D. Prognoza na wiele kroków (10-300) od kroków 1100 i 1200: stała średnia siła całego zbioru (jak w pętli), sama
   bezwładność, „wyrocznia” z prawdziwymi siłami przyszłych paczek (niedozwolona w prognozie, tylko do rozdzielenia
   błędu przedłużania od nieznajomości przyszłych paczek) i suma pełnych wirtualnych kroków.

    uruchom_lokalnie.cmd diagnoza_prognozy.py   ->  wyniki/diagnoza_s47/DIAGNOZA.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'kod'))
import dane as D  # noqa: E402
import model as M  # noqa: E402
import prognoza4 as PG  # noqa: E402
import rownanie as RW  # noqa: E402
import sila as SI  # noqa: E402
import trening as T  # noqa: E402

import cv2  # noqa: E402

TLA = HERE.parent / 'maly_vit_tozsamosc_tla' / 'tla'
ZIARNO, KROKI_DIAG, FIT_DO = 47, 400, 250
STARTY, HORYZONTY = (100, 200), (10, 50, 100, 200)


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())


def dprim(a, b):
    return float((np.mean(a) - np.mean(b)) / (np.std(b) + 1e-9))


def lda_w(poz, neg):
    X = np.concatenate([poz, neg]); y = np.r_[np.ones(len(poz)), np.zeros(len(neg))].astype(int)
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    w = SI.lda((X - mu) / sd, y); w /= np.linalg.norm(w)
    return dict(mu=mu, sd=sd, w=w)


def pary_tla(opis):
    """Obrazy bez narzędzia i maski tokenów sondy: dict typ -> (obrazy (180,176,176,3), tn (180,121), indeksy sond)."""
    man_p = D.jz(D.PROBE / 'manifest.json')
    po_masce = {m['mask']: i for i, m in enumerate(man_p)}
    out = {}
    for r in D.jz(TLA / 'manifest.json'):
        i = po_masce[r['mask_sondy']]
        im = cv2.resize(cv2.cvtColor(cv2.imread(str(TLA / r['image'])), cv2.COLOR_BGR2RGB), (176, 176), interpolation=cv2.INTER_AREA)
        out.setdefault(r['typ'], []).append((i, im))
    return {t: (np.stack([x for _, x in v]), np.array([i for i, _ in v])) for t, v in out.items()}


@torch.no_grad()
def latent_bf16(u, X, tn):
    """Latent narzędzia (N,768) w bf16 jak `Uczen.latent_sond` (X: (N,3,176,176) float, tn: (N,121) bool)."""
    net = u.net; net.eval(); net.zapisuj_latent(True)
    out = []
    for s in range(0, len(X), 96):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, lat = net(X[s:s + 96].to(u.dev))
        a = tn[s:s + 96].to(u.dev).float()
        out.append(((lat.float() * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1)).cpu().numpy())
    net.zapisuj_latent(False)
    return np.concatenate(out)


class Kroki:
    """Jeden krok uczenia dokładnie jak `Uczen.kroki` (paczka 32, odbicie, jasność/kontrast), z dostępem do gradientu."""

    def __init__(self, u, zb, rng):
        self.u, self.zb, self.rng, self.kol = u, zb, rng, []

    def gradient(self):
        u, zb, rng = self.u, self.zb, self.rng
        u.net.train()
        if len(self.kol) < T.PACZKA:
            self.kol += rng.permutation(len(zb)).tolist()
        idx = np.array(self.kol[:T.PACZKA]); self.kol = self.kol[T.PACZKA:]
        odb = rng.random(T.PACZKA) < .5
        x = torch.from_numpy(zb.X[idx]).to(u.dev).permute(0, 3, 1, 2).float() / 255.
        ob = torch.from_numpy(odb).to(u.dev)
        x = torch.where(ob[:, None, None, None], x.flip(3), x)
        a = torch.from_numpy(rng.uniform(.8, 1.2, (T.PACZKA, 2)).astype(np.float32)).to(u.dev)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * a[:, 0, None, None, None] + m * a[:, 1, None, None, None]).clamp(0, 1)
        it = torch.from_numpy(idx); c0, c1 = zb.cele
        heat = torch.where(torch.from_numpy(odb)[:, None, None], c1[0][it], c0[0][it]).to(u.dev)
        ltrb = torch.where(torch.from_numpy(odb)[:, None, None, None], c1[1][it], c0[1][it]).to(u.dev)
        poz = torch.where(torch.from_numpy(odb)[:, None, None], c1[2][it], c0[2][it]).to(u.dev)
        for g in u.opt.param_groups:
            g['lr'] = u._lr()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0)
        return {n: (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in u.net.named_parameters()}

    def krok(self):
        self.u.opt.step(); self.u.krok += 1


def zanik(u):
    nazwy = {id(p): n for n, p in u.net.named_parameters()}
    out = {}
    for grp in u.opt.param_groups:
        for p in grp['params']:
            out[nazwy[id(p)]] = -u._lr() * grp['weight_decay'] * p.detach()
    return out


def umiejetnosc(pred, real):
    return float(100 * (1 - ((pred - real) ** 2).sum() / max((real ** 2).sum(), 1e-12)))


def main() -> int:
    out = HERE / 'wyniki' / f'diagnoza_s{ZIARNO}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baza = D.baza(); sondy = D.sondy()
    X_s, tn, tt, _, obecn, opis = sondy
    rap = {}
    # ---------- rozgrzewka jak w pętli
    st = out / 'k02.pt'
    u = T.Uczen(ZIARNO)
    if st.exists():
        u.wczytaj(torch.load(st, weights_only=False)); lat = np.load(out / 'LATENT_ROZGRZEWKI.npy')
    else:
        rng = np.random.default_rng(ZIARNO); zap = []
        for _ in range(2):
            u.kroki(baza, 500, rng, sondy=sondy, latent_out=zap)
        lat = np.stack([l for _, l in zap]); np.save(out / 'LATENT_ROZGRZEWKI.npy', lat); torch.save(u.stan(), st)
    print(f'rozgrzewka gotowa, krok {u.krok} ({time.time() - t0:.0f} s)', flush=True)
    wek = SI.wektor(lat[-5:].astype(np.float32).mean(0), obecn)               # jak petla4 / wersja 3
    # ---------- A. tożsamość
    Lb = u.latent_sond(sondy).astype(np.float32)                              # (192, 2, 768) bf16 jak w pętli
    nar, tlo = Lb[obecn, 0], Lb[:, 1]
    pary = pary_tla(opis)
    X_t = {t: torch.from_numpy(v[0]).permute(0, 3, 1, 2).float() / 255. for t, v in pary.items()}
    L_bez = {t: latent_bf16(u, X_t[t], tn[pary[t][1]]) for t in pary}
    s_nar = SI.sila(nar, wek); s_tlo = SI.sila(tlo, wek)
    A = dict(wektor_wersji3=dict(sila_narzedzia=float(s_nar.mean()), d_prim_tlo=dprim(s_nar, s_tlo), auc_tlo=auc(s_nar, s_tlo)))
    zrodla = np.array([opis[i]['frame_id'] for i in range(len(opis))])
    for t in pary:
        s_b = SI.sila(L_bez[t], wek)
        s_z = SI.sila(u.latent_sond(sondy)[pary[t][1], 0].astype(np.float32), wek)
        A['wektor_wersji3'][f'd_prim_{t}'] = dprim(s_z, s_b); A['wektor_wersji3'][f'auc_{t}'] = auc(s_z, s_b)
    # wektor „nowy” z par, AUC na źródłach wyłączonych (połowa źródeł na dopasowanie, połowa na test, dwa razy)
    idx_o = pary['oko'][1]
    zr = sorted(set(zrodla[idx_o])); pol = [set(zr[::2]), set(zr[1::2])]
    L_z = u.latent_sond(sondy)[:, 0].astype(np.float32)
    wyn = {'nowy': [], 'wersja3_na_parach': []}
    for fit, test in (pol, pol[::-1]):
        mf = np.array([zrodla[i] in fit for i in idx_o]); mt = ~mf
        poz = L_z[idx_o[mf]]; neg = np.concatenate([L_bez['oko'][mf], L_bez['czarne'][np.isin(pary['czarne'][1], idx_o[mf])]])
        m = lda_w(poz, neg)
        r = lambda X: ((X - m['mu']) / m['sd']) @ m['w']  # noqa: E731
        wyn['nowy'].append(auc(r(L_z[idx_o[mt]]), r(L_bez['oko'][mt])))
        wyn['wersja3_na_parach'].append(auc(SI.sila(L_z[idx_o[mt]], wek), SI.sila(L_bez['oko'][mt], wek)))
    A['auc_oko_zrodla_wylaczone'] = {k: float(np.mean(v)) for k, v in wyn.items()}
    m_nowy = lda_w(L_z[idx_o], np.concatenate([L_bez['oko'], L_bez['czarne']]))
    A['cosinus_wersja3_nowy'] = float(abs(wek['w'] @ m_nowy['w']))
    rap['A_tozsamosc'] = A
    print('A:', json.dumps(A, ensure_ascii=False), flush=True)
    wek_n = dict(mu=m_nowy['mu'], sd=m_nowy['sd'], w=m_nowy['w'])

    def sily(z):
        return SI.sila(z[obecn], wek), SI.sila(z[obecn], wek_n)
    # ---------- B. szum pomiaru bf16 wobec float32
    z32 = RW.latent_narzedzia_sond(u, sondy)
    s32, _ = sily(z32)
    s16 = SI.sila(u.latent_sond(sondy)[obecn, 0].astype(np.float32), wek)
    rap['B_szum_bf16'] = dict(sd_roznicy_bf16_fp32=float((s16 - s32).std()), srednia_roznicy=float((s16 - s32).mean()))
    # ---------- C/D. 400 kroków z pomiarem po każdym kroku
    rng = np.random.default_rng(1000 * ZIARNO + 3)
    kr = Kroki(u, baza, rng)
    rec = {k: [] for k in ('s', 'sn', 'S', 'Sn', 'P', 'K', 'dK_bez_bn')}
    s_prev, sn_prev = sily(RW.latent_narzedzia_sond(u, sondy))
    prognozy = {}
    for t in range(KROKI_DIAG):
        if t in STARTY:                                                      # D: prognozy wielokrokowe z chwili t
            S_hist = np.array(rec['s'])
            F, n_p = PG.sila_zbioru(u, baza, sondy, obecn, wek)
            v0 = PG.predkosc(S_hist, co=1, okno=100)
            prognozy[t] = dict(teraz=s_prev.copy(), v0=v0, F=F, n_paczek=n_p)
        g = kr.gradient()
        nowa, ped = RW.czesci_kroku(u, g)
        zn = zanik(u)
        pelny = {k: nowa[k] + ped[k] + zn[k] for k in nowa}
        s_n, sn_n = sily(RW.latent_narzedzia_sond(u, sondy, nowa))
        s_p, _ = sily(RW.latent_narzedzia_sond(u, sondy, ped))
        s_k, _ = sily(RW.latent_narzedzia_sond(u, sondy, pelny))
        kr.krok()
        s_next, sn_next = sily(RW.latent_narzedzia_sond(u, sondy))
        rec['s'].append(s_prev); rec['sn'].append(sn_prev)
        rec['S'].append(s_n - s_prev); rec['Sn'].append(sn_n - sn_prev); rec['P'].append(s_p - s_prev)
        rec['K'].append(s_k - s_prev)
        s_prev, sn_prev = s_next, sn_next
        if t % 50 == 49:
            print(f'krok {u.krok}: siła średnio {s_prev.mean():.3f} ({time.time() - t0:.0f} s)', flush=True)
    rec['s'].append(s_prev); rec['sn'].append(sn_prev)
    s = np.array(rec['s']); sn = np.array(rec['sn'])
    S, Sn, P, K = (np.array(rec[k]) for k in ('S', 'Sn', 'P', 'K'))
    ds = np.diff(s, axis=0); dsn = np.diff(sn, axis=0)                      # ds[t] = s(t+1) − s(t)
    # C. równanie na jeden krok (wyjaśnienie): ds[t] = a·ds[t−1] + b·S[t]
    C = {}
    for nazwa, d_, S_ in (('wektor_wersji3', ds, S), ('wektor_nowy', dsn, Sn)):
        y = d_[1:]; X1 = d_[:-1]; X2 = S_[1:]
        f, te = slice(0, FIT_DO - 1), slice(FIT_DO - 1, None)
        A2 = np.stack([X1[f].ravel(), X2[f].ravel()], 1)
        ab = np.linalg.lstsq(A2, y[f].ravel(), rcond=None)[0]
        pred = ab[0] * X1[te] + ab[1] * X2[te]
        a0 = float(np.linalg.lstsq(X1[f].reshape(-1, 1), y[f].ravel(), rcond=None)[0][0])
        plac = np.roll(X2, 150, axis=0)                                        # placebo: siła z kroku o 150 dalej
        abp = np.linalg.lstsq(np.stack([X1[f].ravel(), plac[f].ravel()], 1), y[f].ravel(), rcond=None)[0]
        C[nazwa] = dict(a=float(ab[0]), b=float(ab[1]), umiejetnosc_test=umiejetnosc(pred, y[te]),
                        sama_bezwladnosc=umiejetnosc(a0 * X1[te], y[te]),
                        placebo=umiejetnosc(abp[0] * X1[te] + abp[1] * plac[te], y[te]),
                        sd_zmiany_na_krok=float(y.std()))
    C['pelny_wirtualny_krok_wobec_prawdziwego'] = dict(
        umiejetnosc=umiejetnosc(K, ds), korelacja=float(np.corrcoef(K.ravel(), ds.ravel())[0, 1]),
        udzial_reszty_buforow_BN=float(((ds - K) ** 2).sum() / (ds ** 2).sum()))
    C['sila_pedu_jako_prognoza'] = dict(c=float((P * ds).sum() / (P * P).sum()),
                                        umiejetnosc=umiejetnosc((P * ds).sum() / (P * P).sum() * P, ds))
    rap['C_rownanie_jeden_krok'] = C
    rap['B_szum_bf16']['sd_zmiany_100_krokow'] = float((s[100:] - s[:-100]).std())
    # D. prognozy wielokrokowe
    Dd = {}
    for t0_, pr in prognozy.items():
        for H in HORYZONTY:
            if t0_ + H > KROKI_DIAG:
                continue
            real = s[t0_ + H]; teraz = pr['teraz']
            # wyrocznia: iteracja równania z prawdziwymi siłami przyszłych paczek (ds startowe = ds[t0−1])
            a, b = C['wektor_wersji3']['a'], C['wektor_wersji3']['b']
            v = ds[t0_ - 1].copy(); acc = np.zeros_like(v)
            for j in range(H):
                v = a * v + b * S[t0_ + j]; acc += v
            pred = {'rownanie_srednia_sila': teraz + PG.zmiana(pr['v0'], pr['F'], H),
                    'sama_bezwladnosc': teraz + PG.zmiana(pr['v0'], 0 * pr['F'], H),
                    'trend_liniowy': teraz + pr['v0'] * H,
                    'wyrocznia_przyszle_paczki': teraz + acc,
                    'suma_pelnych_krokow': teraz + K[t0_:t0_ + H].sum(0)}
            Dd[f'od_{1000 + t0_}_H{H}'] = PG.ocena(pred, teraz, real)
    rap['D_wiele_krokow'] = Dd
    rap['sila_srednia_zbioru'] = {str(1000 + k): float(v['F'].mean()) for k, v in prognozy.items()}
    zj(out / 'DIAGNOZA.json', rap)
    print(json.dumps({k: v for k, v in rap.items() if k != 'D_wiele_krokow'}, ensure_ascii=False, indent=1, default=float))
    for k, v in Dd.items():
        print(k, {m: round(x['wzgl_braku_zmiany'], 2) for m, x in v.items() if isinstance(x, dict)},
              'znak', {m: round(x['znak'], 2) for m, x in v.items() if isinstance(x, dict)})
    print(f'gotowe ({time.time() - t0:.0f} s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
