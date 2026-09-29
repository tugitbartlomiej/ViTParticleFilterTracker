#!/usr/bin/env python3
"""Uczenie równania z macierzami (model.py) na nagraniu ziarna 47 i ocena poza próbą (PLAN.md).

Co rozstrzyga: czy uczone wymuszenie z klatek JUŻ UŻYTYCH w uczeniu (paczki t−1..t−16, W) przewiduje krok latentu
sond i siłę wektora tożsamości lepiej niż sama uczona pamięć (`bez_W`) i niż placebo (paczki sprzed 250 kroków).
Przyszłe klatki, w tym paczka przewidywanego kroku, do równania nie wchodzą (reguła użytkownika, 2026-09-29).
Odniesienia: bezwładność a·Δz_t, pęd a·Δz_t + c·S_pęd (siła pędu AdamW, dozwolona) i, tylko dla porównania,
równanie z paczką bieżącego kroku a·Δz_t + b·S_t (85 w medianie Edenu; w prognozie niedozwolone).
Co obali: `pelne` na teście nie wyżej niż `bez_W` albo `placebo`; duży spadek na sondach holdout. Każde ziarno
(47, 48) uczone i oceniane osobno: to różne sieci o różnym układzie neuronów.

Ocena: umiejętność = 100 · (1 − SSE / SS) przyrostu względem braku zmiany, na pełnym latencie (768), na sile
wektora tożsamości (rzut przyrostu na kierunek LDA z kroku 1000, każde ziarno swój) i na wektorze (średnia po sondach).

    <venv rfdetr>/python -u uczenie.py --ramiona pelne bez_gradientu placebo bez_W      ->  wyniki/BIEG_<ramię>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import dane as DN
from model import CZLONY, OKNO, OPOZ, Rownanie, kara_rozdzielnosci

HERE = Path(__file__).resolve().parent
WYN = HERE / 'wyniki'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOKE = os.environ.get('MACIERZE_SMOKE') == '1'
FIT, WYBOR, TEST = (1000, 1800), (1800, 2000), (2000, 2999)
HIST = OKNO + 2                                     # historia przed ocenianym blokiem: pełne okno JP i N
KONFIG = dict(lr=1e-3, wd=1e-4, rozgrzewka=200, maks_epok=80, iter_na_epoke=20, cierpliwosc=12, obniz_po=3,
              kara=0.05, clip=1.0, ema=0.995, n_ukr=1024, w_glowic=8, w_r=64, dl=64,
              gradient=True, przesuniecie=0, bez_w=False, ziarno_rownania=0)
RAMIONA = {'pelne': {}, 'bez_gradientu': dict(gradient=False), 'placebo': dict(przesuniecie=250),
           'bez_W': dict(bez_w=True)}


class Dane:
    """Nagranie na karcie: stany sond z narzędziem, tło sond i cechy paczek (standaryzacja z ziarna uczącego)."""

    def __init__(self, nag: DN.Nagranie, stat_p=None, na_karcie=True):
        self.nag, self.od = nag, nag.od
        gdzie = DEV if na_karcie else 'cpu'           # ziarno testowe czeka w RAM, na kartę idzie blok po bloku
        self.sondy = np.where(nag.obecn)[0]                              # 180 sond z narzędziem
        self.z = nag.z[:, self.sondy].to(gdzie)                         # (Tz, 180, 768) f16
        self.zt = nag.zt[:, self.sondy].to(gdzie)
        lat = nag.p_lat.reshape(len(nag.p_lat), 32, -1).float()          # (Tp, 32, 1536)
        g = nag.p_g.reshape(len(nag.p_g), 32, -1).float()
        if stat_p is None:
            i0, i1 = FIT[0] - self.od, FIT[1] - self.od
            stat_p = dict(mu_l=lat[i0:i1].mean((0, 1)), sd_l=lat[i0:i1].std((0, 1)) + 1e-6,
                          mu_g=g[i0:i1].mean((0, 1)), sd_g=g[i0:i1].std((0, 1)) + 1e-12)
        self.stat_p = stat_p
        ma = nag.p_ma.float()[..., None]
        self.klucz = torch.cat([(lat - stat_p['mu_l']) / stat_p['sd_l'], ma], -1).clamp(-20, 20).half().to(gdzie)
        self.g = ((g - stat_p['mu_g']) / stat_p['sd_g']).clamp(-20, 20).half().to(gdzie)
        tz = nag.tozs
        self.t_sd = torch.as_tensor(tz['sd'], dtype=torch.float32, device=DEV)
        self.t_w = torch.as_tensor(tz['w'], dtype=torch.float32, device=DEV)

    def indeks(self, rola):
        """Pozycje (w self.sondy) sond danej roli."""
        return np.where(self.nag.rola[self.sondy] == rola)[0]

    def wsad(self, c, T, sondy, cfg):
        """Blok kroków c..c+T−1 (z krokiem c+T jako prawdą ostatniego)."""
        i = c - self.od
        z = self.z[i:i + T + 1, sondy].to(DEV).float().transpose(0, 1)
        zt = self.zt[i:i + T, sondy].to(DEV).float().transpose(0, 1)
        j = i - OPOZ - cfg['przesuniecie']                            # paczki c−OPOZ..c+T−2: bez paczki kroku t
        assert j >= 0, (c, j)
        kl = self.klucz[j:j + T + OPOZ - 1].to(DEV).float()
        wa = self.g[j:j + T + OPOZ - 1].to(DEV).float() if cfg['gradient'] else kl
        return z, zt, kl, wa


def statystyki(dn: Dane, sondy):
    i0, i1 = FIT[0] - dn.od, FIT[1] - dn.od
    z = dn.z[i0 - 2:i1 + 1, sondy].float()
    zt = dn.zt[i0:i1, sondy].float()
    v = z[1:] - z[:-1]
    a = v[1:] - v[:-1]
    return dict(mu_z=z.mean((0, 1)), sd_z=z.std((0, 1)) + 1e-6, sd_v=v.std((0, 1)) + 1e-6, sd_a=a.std((0, 1)) + 1e-6,
                sd_krok=v.std((0, 1)) + 1e-6, mu_t=zt.mean((0, 1)), sd_t=zt.std((0, 1)) + 1e-6)


@torch.no_grad()
def przewiduj(model, dn: Dane, t0, t1, sondy, cfg, wylacz=()):
    """Przewidywany i prawdziwy krok dla kroków t0..t1−1: dwa tensory (n, P, 768) na karcie, float32."""
    pred, praw = [], []
    for a in range(t0, t1, cfg['dl']):
        b = min(a + cfg['dl'], t1)
        z, zt, kl, wa = dn.wsad(a - HIST, b - a + HIST, sondy, cfg)
        krok, prawda, _ = model(z, zt, kl, wa, wylacz)
        pred.append(krok[:, HIST:].transpose(0, 1)); praw.append(prawda[:, HIST:].transpose(0, 1))
    return torch.cat(pred), torch.cat(praw)


def miary(pred, praw, dn: Dane):
    """Umiejętność na latencie, sile tożsamości i wektorze (średnia sond w standaryzacji tożsamości)."""
    def um(y, p):
        return float(100 * (1 - ((y - p) ** 2).double().sum() / (y ** 2).double().sum()))
    ps, ys = pred / dn.t_sd, praw / dn.t_sd
    return dict(latent=round(um(praw, pred), 2), tozsamosc=round(um(ys @ dn.t_w, ps @ dn.t_w), 2),
                wektor=round(um(ys.mean(1), ps.mean(1)), 2))


def ocen(model, dn: Dane, t0, t1, cfg, wylacz=()):
    out = {}
    for rola in ('fit', 'holdout'):
        pred, praw = przewiduj(model, dn, t0, t1, dn.indeks(rola), cfg, wylacz)
        out[rola] = miary(pred, praw, dn)
        del pred, praw
    return out


@torch.no_grad()
def odniesienia(dn_ucz: Dane, testy):
    """Bezwładność a·Δz_t, pęd a·Δz_t + c·S_pęd (dozwolone) i a·Δz_t + b·S_t z paczką bieżącego kroku (tylko
    porównanie, przyszłe klatki), dopasowane na krokach uczenia ziarna uczącego."""
    sondy = dn_ucz.indeks('fit')

    def czesci(dn, t0, t1, sd):
        i = t0 - dn.od
        z = dn.z[i - 1:i + (t1 - t0) + 1, sd].to(DEV).float()
        v = z[1:] - z[:-1]                                              # v[k] = z_{t0+k} − z_{t0+k−1}
        dz, y = v[:-1], v[1:]
        S = torch.from_numpy(dn.nag.sila(t0, t1, dn.sondy[sd], 'nowa')).to(DEV)
        Sp = torch.from_numpy(dn.nag.sila(t0, t1, dn.sondy[sd], 'ped')).to(DEV)
        return dz, S, Sp, y

    def dopasuj(*X, y):
        X = torch.stack([x.reshape(-1) for x in X], 1).double()
        return [float(c) for c in torch.linalg.lstsq(X, y.reshape(-1, 1).double()).solution[:, 0]]

    dz, S, Sp, y = czesci(dn_ucz, *FIT, sondy)
    (a1,) = dopasuj(dz, y=y)
    a2, c2 = dopasuj(dz, Sp, y=y)
    a3, b3 = dopasuj(dz, S, y=y)
    wynik = dict(bezwladnosc_a=round(a1, 4), ped_a=round(a2, 4), ped_c=round(c2, 4),
                 paczka_biezaca_NIEDOZWOLONE_a=round(a3, 4), paczka_biezaca_NIEDOZWOLONE_b=round(b3, 4))
    for nazwa, (dn, t0, t1) in testy.items():
        for rola in ('fit', 'holdout'):
            dz, S, Sp, y = czesci(dn, t0, t1, dn.indeks(rola))
            wynik[f'{nazwa}/{rola}'] = dict(bezwladnosc=miary(a1 * dz, y, dn), ped=miary(a2 * dz + c2 * Sp, y, dn),
                                           paczka_biezaca_NIEDOZWOLONE=miary(a3 * dz + b3 * S, y, dn))
            del dz, S, Sp, y
    return wynik


def parametry_jader(model):
    with torch.no_grad():
        return dict(JP_polowiczny=[round(-math.log(2) / math.log(float(r)), 2) for r in torch.sigmoid(model.JP.theta)],
                    JP_omega_srednia=[round(float(v), 4) for v in model.JP.omega.abs().mean(1)],
                    N_polowiczny=[round(-math.log(2) / math.log(float(r)), 2) for r in torch.sigmoid(model.N.theta)],
                    N_omega=[round(float(v), 4) for v in model.N.omega],
                    W_kappa=[[round(float(x), 4) for x in row] for row in model.W.kappa])


def bieg(ramie, dn: Dane, stat):
    tag = f's{dn.nag.s}_{ramie}'
    cfg = dict(KONFIG, **RAMIONA[ramie])
    if SMOKE:
        cfg.update(maks_epok=2, iter_na_epoke=3, rozgrzewka=2)
    torch.manual_seed(cfg['ziarno_rownania']); np.random.seed(cfg['ziarno_rownania'])
    rng = np.random.default_rng(cfg['ziarno_rownania'])
    zegar = time.perf_counter()
    n_k = dn.klucz.shape[-1]
    model = Rownanie(stat, n_k, dn.g.shape[-1] if cfg['gradient'] else n_k, cfg['n_ukr'], cfg['w_glowic'], cfg['w_r']).to(DEV)
    par = {c: sum(p.numel() for p in getattr(model, c).parameters()) for c in CZLONY}
    par['JP'] += sum(p.numel() for p in model.tok.parameters())
    print('RAMIE', tag, 'PARAMETRY', par, flush=True)
    wylacz_ucz = ('W',) if cfg['bez_w'] else ()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg['obniz_po'])
    cien = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sondy = dn.indeks('fit')
    skala = float((stat['sd_krok'] ** 2).mean())
    WYN.mkdir(exist_ok=True)
    sciezka = WYN / f'model_{tag}.pt'
    log = open(WYN / f'TRENING_{tag}.csv', 'w', newline='', encoding='utf-8')
    pis = csv.writer(log); pis.writerow(['epoka', 'strata', 'kara', 'wybor_latent', 'wybor_tozsamosc', 'lr', 'sekundy'])
    najl, najl_ep, bez, krok_opt = -float('inf'), -1, 0, 0
    for ep in range(1, cfg['maks_epok'] + 1):
        model.train()
        suma = suma_k = 0.0
        for _ in range(cfg['iter_na_epoke']):
            c = int(rng.integers(FIT[0], FIT[1] - cfg['dl'] + 1)) - HIST
            z, zt, kl, wa = dn.wsad(c, HIST + cfg['dl'], sondy, cfg)
            krok, prawda, czl = model(z, zt, kl, wa, wylacz_ucz)
            blad = ((krok[:, HIST:] - prawda[:, HIST:]) ** 2).mean() / skala
            kara = kara_rozdzielnosci(czl, slice(HIST, None))
            strata = blad + cfg['kara'] * kara
            if krok_opt < cfg['rozgrzewka']:
                for g in opt.param_groups:
                    g['lr'] = cfg['lr'] * (krok_opt + 1) / cfg['rozgrzewka']
            opt.zero_grad(set_to_none=True)
            strata.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            opt.step()
            with torch.no_grad():
                for k, t in model.state_dict().items():
                    if t.dtype.is_floating_point:
                        cien[k].mul_(cfg['ema']).add_(t.detach(), alpha=1 - cfg['ema'])
            krok_opt += 1
            suma += float(blad); suma_k += float(kara)
        model.eval()
        biezace = {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(cien)
        pred, praw = przewiduj(model, dn, *WYBOR, sondy, cfg, wylacz_ucz)
        w = miary(pred, praw, dn); del pred, praw
        stan = {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(biezace)
        sched.step(-w['latent'])
        lr = opt.param_groups[0]['lr']
        n = cfg['iter_na_epoke']
        pis.writerow([ep, round(suma / n, 5), round(suma_k / n, 5), w['latent'], w['tozsamosc'], lr,
                      round(time.perf_counter() - zegar)]); log.flush()
        print(f"EPOKA {tag} {ep} strata {suma / n:.4f} kara {suma_k / n:.4f} | wybór latent {w['latent']:.2f} "
              f"tożsamość {w['tozsamosc']:.2f} | lr {lr:.1e} | {time.perf_counter() - zegar:.0f} s", flush=True)
        if w['latent'] > najl:
            najl, najl_ep, bez = w['latent'], ep, 0
            torch.save(dict(stan=stan, epoka=ep, konfig=cfg, wybor=w), sciezka)
        else:
            bez += 1
            if bez >= cfg['cierpliwosc']:
                break
    log.close()
    model.load_state_dict(torch.load(sciezka, map_location=DEV, weights_only=False)['stan'])
    model.eval()
    wynik = dict(ramie=ramie, ziarno=dn.nag.s, epoka=najl_ep, konfig=cfg, parametry=par, smoke=SMOKE, podzial=dict(fit=FIT, wybor=WYBOR, test=TEST),
                 sondy=dict(fit=len(dn.indeks('fit')), holdout=len(dn.indeks('holdout'))), oceny={})
    o = {'pelne': ocen(model, dn, *TEST, cfg, wylacz_ucz)}
    for c in CZLONY:
        if c not in wylacz_ucz:
            o[f'bez_{c}'] = ocen(model, dn, *TEST, cfg, tuple(wylacz_ucz) + (c,))
    wynik['oceny']['test'] = o
    print('OCENA', tag, json.dumps(o, ensure_ascii=False), flush=True)
    wynik['jadra'] = parametry_jader(model)
    wynik['sekundy'] = round(time.perf_counter() - zegar)
    (WYN / f'BIEG_{"SMOKE_" if SMOKE else ""}{tag}.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
    print('KONIEC', tag, wynik['oceny']['test']['pelne'], flush=True)
    del model, opt, cien
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ramiona', nargs='+', default=list(RAMIONA))
    ap.add_argument('--ziarna', nargs='+', type=int, default=[47, 48])
    ap.add_argument('--bez-odniesien', action='store_true')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    dn = Dane(DN.Nagranie(47))
    dn48 = Dane(DN.Nagranie(48), dn.stat_p, na_karcie=False)
    print(f'dane {time.time() - t0:.0f} s; sondy fit {len(dn.indeks("fit"))}, holdout {len(dn.indeks("holdout"))}; '
          f'cosinus wektorów tożsamości s47/s48 {float(dn.t_w @ dn48.t_w):.3f}', flush=True)
    if not a.bez_odniesien:
        od = odniesienia(dn, {'s47_test': (dn, *TEST), 's48_test': (dn48, *TEST)})
        WYN.mkdir(exist_ok=True)
        (WYN / f'ODNIESIENIA{"_SMOKE" if SMOKE else ""}.json').write_text(json.dumps(od, ensure_ascii=False, indent=1), encoding='utf-8')
        print('ODNIESIENIA', json.dumps(od, ensure_ascii=False), flush=True)
    # Każde ziarno to inna sieć z innym układem neuronów (cosinus wektorów tożsamości wyżej), więc równanie uczy się
    # osobno na każdym ziarnie, jak współczynniki w T2; przenosi się tylko przepis, nie wagi.
    del dn48
    for s_ in a.ziarna:
        if s_ != 47:
            del dn; torch.cuda.empty_cache()
            dn = Dane(DN.Nagranie(s_))
        stat = statystyki(dn, dn.indeks('fit'))
        for r in a.ramiona:
            bieg(r, dn, stat)


if __name__ == '__main__':
    main()
