#!/usr/bin/env python3
"""Nagranie uczenia małego ViT krok po kroku: latent sond i paczka danych w każdym kroku (PLAN.md, T0).

Co rozstrzyga: dostarcza dane, bez których nie da się sprawdzić, czy ruch latentu w jednym kroku wynika z paczki.
Zapis co 10 kroków z Edena (`maly_vit_wektor_tozsamosci`) nie zawiera paczek, a ruch latentu między zapisami jest
w ~80 % nieprzewidywalny z samej historii latentu. Obali to: gdy nagrany przebieg różni się statystycznie od Edena
(norma przyrostu na 10 kroków, autokorelacja przyrostów) — wtedy nagranie nie reprezentuje biegów z Edena.

Uczenie identyczne jak rozgrzewka + ramię `bez` wersji 2: ziarno s, baza 1500 klatek, paczka 32, AdamW lr 5e-4,
rozgrzewka lr 300 kroków, obcinanie gradientu 1, bf16; kolejność paczek z tych samych generatorów
(default_rng(s) na krokach 1–1000, default_rng(1000 s + 3) dalej; kolejka permutacji zerowana co 500 kroków).
Kod modelu i danych: `../maly_vit_od_zera` (tylko import, bez zmian).

W każdym kroku t (po aktualizacji wag):
  sondy_lat   (T+1, 192, 2, 768) f16  latent sond (narzędzie, tło), tryb eval, jak na Edenie; wiersz 0 = przed uczeniem
  sondy_h     (T+1, 192, 2, 192) f16  wejście fc1 ostatniego bloku (po norm2), te same średnie
  paczka_idx  (T, 32) int16, paczka_odb (T, 32) bool, paczka_a (T, 32, 2) f32   skład paczki i augmentacja
  paczka_lat  (T, 32, 2, 768) f16     latent obrazów paczki w przejściu uczącym PRZED aktualizacją (narzędzie, całość)
  paczka_g    (T, 32, 2, 768) f32     gradient straty względem tego latentu (te same średnie) — „siła” obrazu
  paczka_h    (T, 32, 2, 192) f16     wejście fc1 obrazów paczki
  paczka_ma_narz (T, 32) bool, strata (T,), norma_grad (T,) przed obcięciem, lr (T,)
Co CO_ROZKLAD kroków rozkład przyrostu latentu sond (narzędzie) na części — dokładnie, przez podmianę wag:
  rozklad_param  zmiana od samych wag (stare bufory BatchNorm), rozklad_bufory  od samych buforów BatchNorm,
  rozklad_ostatni  od samych wag ostatniego bloku; wszystkie (N, 192, 768) f16, kroki w rozklad_kroki.

Z `--sily` (od kroku SILY_OD) dodatkowo siły działające na sondy w kroku t, liczone w float32 przez podstawienie wag
(bez BatchNorm, tryb eval, te same bufory): z(θ_t + d) − z(θ_t) dla części aktualizacji AdamW:
  sila_nowa   d = −lr/b1 · (1−β1) g_t / mianownik        (nowa paczka: jej gradient po obcięciu)
  sila_ped    d = −lr/b1 · β1 m_{t−1} / mianownik        (pęd: pamięć poprzednich paczek w stanie optymalizatora)
  sila_zanik  d = −lr · wd · θ_t                          (zanik wag)
  sila_suma   wszystkie trzy naraz (sprawdzenie liniowości)
wszystkie (T − SILY_OD, 192, 768) f16, kanał narzędzia.

    <venv rfdetr>/python -u nagranie.py --ziarno 47 --kroki 3000 [--sily --tag _sily]  ->  nagranie_s47<tag>/
"""
from __future__ import annotations

import argparse
import math
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KOD = Path(os.environ.get('MALY_VIT_KOD', HERE.parent / 'maly_vit_od_zera'))   # na Edenie: kopia kodu z env-ścieżkami
sys.path.insert(0, str(KOD))
import dane as D          # noqa: E402
import model as M         # noqa: E402
import trening as T       # noqa: E402

KROKI_OBR, ROZGRZEWKA_KROKI, CO_ROZKLAD, SILY_OD = 500, 1000, 10, 900


class Nagrywacz:
    """Haki na ostatnim bloku: wyjście GELU w MLP (latent) i wejście fc1."""

    def __init__(self, net):
        mlp = net.bloki[-1].mlp
        self.act = None; self.h = None; self.grad = False
        mlp.act.register_forward_hook(self._act)
        mlp.fc1.register_forward_pre_hook(self._fc1)

    def _act(self, m, i, o):
        if self.grad and o.requires_grad:
            o.retain_grad()
        self.act = o

    def _fc1(self, m, i):
        self.h = i[0]


def srednie(x, maska):
    """x (B, N, C), maska (B, N) float → (B, C) średnia po maskowanych tokenach (0 gdy pusta)."""
    w = maska.to(x.dtype)
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)


@torch.no_grad()
def sondy_pomiar(net, nag, X, tn, tt, dev):
    net.eval()
    lat, hh = [], []
    for s in range(0, len(X), 96):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            net(X[s:s + 96].to(dev))
        a = tn[s:s + 96].to(dev).float(); b = tt[s:s + 96].to(dev).float()
        L = nag.act.float(); H = nag.h.float()
        lat.append(torch.stack([srednie(L, a), srednie(L, b)], 1)); hh.append(torch.stack([srednie(H, a), srednie(H, b)], 1))
    return torch.cat(lat).half().cpu().numpy(), torch.cat(hh).half().cpu().numpy()


@torch.no_grad()
def kierunki(opt, nazwy):
    """Części aktualizacji AdamW (jak torch.optim.AdamW, przed opt.step()): dict część -> dict nazwa -> tensor."""
    out = {'nowa': {}, 'ped': {}, 'zanik': {}}
    for grp in opt.param_groups:
        lr, (b1, b2), eps, wd = grp['lr'], grp['betas'], grp['eps'], grp['weight_decay']
        for p in grp['params']:
            if p.grad is None:
                continue
            st = opt.state.get(p, {})
            krok = float(st['step']) + 1 if 'step' in st else 1.0
            m = st['exp_avg'] if 'exp_avg' in st else torch.zeros_like(p)
            v = st['exp_avg_sq'] if 'exp_avg_sq' in st else torch.zeros_like(p)
            g = p.grad
            mian = (b2 * v + (1 - b2) * g * g).sqrt() / math.sqrt(1 - b2 ** krok) + eps
            n = nazwy[id(p)]
            out['nowa'][n] = -(lr / (1 - b1 ** krok)) * (1 - b1) * g / mian
            out['ped'][n] = -(lr / (1 - b1 ** krok)) * b1 * m / mian
            out['zanik'][n] = -lr * wd * p.detach()
    return out


@torch.no_grad()
def z_fp32(net_f, nag_f, params, bufs, X, tn, dev):
    """Latent narzędzia sond (192, 768) w float32 dla podanych wag i buforów."""
    net_f.eval(); out = []
    for s in range(0, len(X), 96):
        torch.func.functional_call(net_f, (params, bufs), (X[s:s + 96].to(dev),))
        out.append(srednie(nag_f.act.float(), tn[s:s + 96].to(dev).float()))
    return torch.cat(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--kroki', type=int, default=3000)
    ap.add_argument('--sily', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None, help='katalog nagrania (domyślnie nagranie_s<ziarno><tag> obok skryptu)')
    a = ap.parse_args(argv)
    s, TK = a.ziarno, a.kroki
    out = Path(a.out) if a.out else HERE / f'nagranie_s{s}{a.tag}'; out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    t0 = time.time()
    baza = D.baza(); X_s, tn, tt, ramki_s, obecn, opis = D.sondy()
    print(f'dane {time.time() - t0:.0f} s, baza {len(baza)}, sondy {len(X_s)}', flush=True)
    u = T.Uczen(s, dev); net = u.net
    nag = Nagrywacz(net)
    net.zapisuj_latent(False)
    mm = lambda n, sh, dt: np.lib.format.open_memmap(out / f'{n}.npy', mode='w+', dtype=dt, shape=sh)
    S_lat = mm('sondy_lat', (TK + 1, 192, 2, 768), np.float16); S_h = mm('sondy_h', (TK + 1, 192, 2, 192), np.float16)
    P_lat = mm('paczka_lat', (TK, 32, 2, 768), np.float16); P_g = mm('paczka_g', (TK, 32, 2, 768), np.float32)
    P_h = mm('paczka_h', (TK, 32, 2, 192), np.float16)
    idx_all = np.zeros((TK, 32), np.int16); odb_all = np.zeros((TK, 32), bool); a_all = np.zeros((TK, 32, 2), np.float32)
    ma_n = np.zeros((TK, 32), bool); strata = np.zeros(TK, np.float32); ng = np.zeros(TK, np.float32); lrs = np.zeros(TK, np.float32)
    nroz = TK // CO_ROZKLAD
    R_par = mm('rozklad_param', (nroz, 192, 768), np.float16); R_buf = mm('rozklad_bufory', (nroz, 192, 768), np.float16)
    R_ost = mm('rozklad_ostatni', (nroz, 192, 768), np.float16); r_kroki = []
    S_lat[0], S_h[0] = sondy_pomiar(net, nag, X_s, tn, tt, dev)
    cel0, cel1 = baza.cele
    net_stary = M.MalyViT().to(dev).eval(); net_stary.load_state_dict(net.state_dict()); nag_st = Nagrywacz(net_stary)
    if a.sily:
        net_f = M.MalyViT().to(dev).eval(); nag_f = Nagrywacz(net_f)
        nazwy = {id(p): n for n, p in net.named_parameters()}
        ns = TK - SILY_OD
        SI = {k: mm(f'sila_{k}', (ns, 192, 768), np.float16) for k in ('nowa', 'ped', 'zanik', 'suma')}
    kol, rng = [], None
    for t in range(TK):
        if t % KROKI_OBR == 0:                                             # nowe wywołanie kroki(): kolejka od zera
            kol = []
            if t == 0:
                rng = np.random.default_rng(s)
            elif t == ROZGRZEWKA_KROKI:
                rng = np.random.default_rng(1000 * s + 3)
        rozklad = (t + 1) % CO_ROZKLAD == 0
        if rozklad:
            net_stary.load_state_dict(net.state_dict())
        net.train()
        if len(kol) < T.PACZKA:
            kol += rng.permutation(len(baza)).tolist()
        idx = np.array(kol[:T.PACZKA]); kol = kol[T.PACZKA:]
        odb = rng.random(T.PACZKA) < .5
        x = torch.from_numpy(baza.X[idx]).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.
        ob = torch.from_numpy(odb).to(dev)
        x = torch.where(ob[:, None, None, None], x.flip(3), x)
        aa = rng.uniform(.8, 1.2, (T.PACZKA, 2)).astype(np.float32)
        at = torch.from_numpy(aa).to(dev)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * at[:, 0, None, None, None] + m * at[:, 1, None, None, None]).clamp(0, 1)
        it = torch.from_numpy(idx)
        heat = torch.where(torch.from_numpy(odb)[:, None, None], cel1[0][it], cel0[0][it]).to(dev)
        ltrb = torch.where(torch.from_numpy(odb)[:, None, None, None], cel1[1][it], cel0[1][it]).to(dev)
        poz = torch.where(torch.from_numpy(odb)[:, None, None], cel1[2][it], cel0[2][it]).to(dev)
        lr = u._lr()
        for g in u.opt.param_groups:
            g['lr'] = lr
        nag.grad = True
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        L, cz = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True)
        L.backward()
        nag.grad = False
        norma = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        with torch.no_grad():
            mask_n = (poz.flatten(1) > 0).float(); mask_c = torch.ones_like(mask_n)
            A = nag.act.float(); G = nag.act.grad.float(); H = nag.h.float()
            P_lat[t] = torch.stack([srednie(A, mask_n), srednie(A, mask_c)], 1).half().cpu().numpy()
            P_g[t] = torch.stack([srednie(G, mask_n), srednie(G, mask_c)], 1).cpu().numpy()
            P_h[t] = torch.stack([srednie(H, mask_n), srednie(H, mask_c)], 1).half().cpu().numpy()
            ma_n[t] = (mask_n.sum(1) > 0).cpu().numpy()
        if a.sily and t >= SILY_OD:
            kier = kierunki(u.opt, nazwy)
            th = {n: p.detach() for n, p in net.named_parameters()}; bf = {n: b for n, b in net.named_buffers()}
            z0 = z_fp32(net_f, nag_f, th, bf, X_s, tn, dev)
            i = t - SILY_OD
            for k in ('nowa', 'ped', 'zanik'):
                SI[k][i] = (z_fp32(net_f, nag_f, {n: th[n] + kier[k][n] for n in th}, bf, X_s, tn, dev) - z0).half().cpu().numpy()
            SI['suma'][i] = (z_fp32(net_f, nag_f, {n: th[n] + kier['nowa'][n] + kier['ped'][n] + kier['zanik'][n] for n in th},
                                    bf, X_s, tn, dev) - z0).half().cpu().numpy()
        u.opt.step(); u.krok += 1
        idx_all[t], odb_all[t], a_all[t], strata[t], ng[t], lrs[t] = idx, odb, aa, float(L), float(norma), lr
        S_lat[t + 1], S_h[t + 1] = sondy_pomiar(net, nag, X_s, tn, tt, dev)
        if rozklad:                                                        # dokładny rozkład przyrostu na części
            nowy = net.state_dict(); stary = net_stary.state_dict()
            bufory = {k for k, _ in net.named_buffers()}
            ost = {k for k in nowy if k.startswith('bloki.5.')}
            z0 = S_lat[t, :, 0].astype(np.float32)

            def z_dla(sd):
                net_stary.load_state_dict(sd)
                return sondy_pomiar(net_stary, nag_st, X_s, tn, tt, dev)[0][:, 0].astype(np.float32)

            wp = {k: (nowy[k] if k not in bufory else stary[k]) for k in nowy}
            wb = {k: (stary[k] if k not in bufory else nowy[k]) for k in nowy}
            wo = {k: (nowy[k] if k in ost and k not in bufory else stary[k]) for k in nowy}
            wp, wb, wo = [{k: v.clone() for k, v in d_.items()} for d_ in (wp, wb, wo)]
            i = len(r_kroki)
            R_par[i] = (z_dla(wp) - z0).astype(np.float16); R_buf[i] = (z_dla(wb) - z0).astype(np.float16)
            R_ost[i] = (z_dla(wo) - z0).astype(np.float16); r_kroki.append(t + 1)
        if (t + 1) % 250 == 0:
            d = (S_lat[t + 1, :, 0].astype(np.float32) - S_lat[t, :, 0].astype(np.float32))
            print(f'krok {t + 1}: strata {float(L):.3f}, |Δz| 1 krok {np.linalg.norm(d, axis=1).mean():.3f}, '
                  f'{time.time() - t0:.0f} s', flush=True)
    for n, v in dict(paczka_idx=idx_all, paczka_odb=odb_all, paczka_a=a_all, paczka_ma_narz=ma_n, strata=strata,
                     norma_grad=ng, lr=lrs, rozklad_kroki=np.array(r_kroki)).items():
        np.save(out / f'{n}.npy', v)
    for arr in (S_lat, S_h, P_lat, P_g, P_h, R_par, R_buf, R_ost) + (tuple(SI.values()) if a.sily else ()):
        arr.flush()
    (out / 'META.json').write_text(json.dumps(dict(ziarno=s, kroki=TK, sekundy=time.time() - t0, obecn=obecn.tolist(),
                                                  opis=opis, co_rozklad=CO_ROZKLAD, sily_od=SILY_OD if a.sily else None), indent=1, ensure_ascii=False),
                                   encoding='utf-8')
    print(f'koniec {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
