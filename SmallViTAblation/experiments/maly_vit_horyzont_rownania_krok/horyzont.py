#!/usr/bin/env python3
"""Jak daleko naprzód równanie ruchu przewiduje latent tożsamości narzędzia z umiejętnością co najmniej 90 procent
(pytanie użytkownika 2026-09-30). Pomiar bez żadnej pętli doboru klatek: zwykłe uczenie małego ViT na bazie 1500 klatek
(jak rozgrzewka pętli), ziarno z --ziarno.

Równanie jednego kroku (`../maly_vit_tozsamosc_jeden_krok`, commit a5813796): Δz(t+1) = S_pęd(t) + S_paczka(t).
Prognoza na H kroków z chwili t0 w pięciu wariantach, różniących się tym, co wiadomo w chwili t0:
  A  równanie liczone co krok na PRAWDZIWYCH wagach i prawdziwej paczce, zsumowane po H krokach. To nie prognoza (zna
     przyszłe wagi), tylko górna granica: jak rośnie błąd, gdy sumować trafny jeden krok;
  B  znane klatki przyszłych kroków (dokładnie te paczki, które uczenie weźmie), wagi zamrożone w chwili t0: gradient
     każdej paczki liczony przy wagach z t0, krok AdamW symulowany w przestrzeni wag (pęd i mianownik z t0 dalej
     aktualizowane), na koniec latent przy wagach t0 + suma kroków;
  B2 to samo co B, ale jako suma pchnięć: dla każdego kroku osobno z(t0 + krok_k) − z(t0), zsumowane (dosłowne
     przedłużenie równania S_pęd + S_paczka na H kroków przy wagach z t0);
  C1 bez przyszłych klatek: sam pęd optymalizatora z chwili t0, wygasający jak w AdamW (gradient przyszłych kroków = 0);
  C2 bez przyszłych klatek: pęd z t0 plus w każdym kroku średni gradient całego zbioru uczącego przy wagach z t0.
Bufory BatchNorm prognoz: z chwili t0 (przyszłych nie znamy).

Miara jak w a5813796: umiejętność = 100 · (1 − SSE / SS) przyrostu siły tożsamości 180 sond z narzędziem między t0
a t0 + H, wobec „braku zmiany”; obok to samo dla całego latentu (180 × 768). Okna: t0 = 1000, 1500, 2000, 2500,
po 500 kroków, rozłączne. Wektor tożsamości: LDA na sondach z kroków 960-1000 (jak w pętli).

    uruchom_lokalnie.cmd horyzont.py --ziarno 47   ->  wyniki/HORYZONT_s47.json, wyniki/horyzont_s47.npz
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'kod'))
import dane as D  # noqa: E402
import model as M  # noqa: E402
import prognoza_krok as PK  # noqa: E402
import rownanie as RW  # noqa: E402
import sila as SI  # noqa: E402
import trening as T  # noqa: E402

H_LISTA = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500)
OKNA = (1000, 1500, 2000, 2500)
HMAX = 500
WARIANTY = ('A', 'B', 'B2', 'C1', 'C2')
OPIS = dict(A='równanie co krok na prawdziwych wagach (górna granica, nie prognoza)',
            B='znane klatki przyszłych kroków, wagi z chwili t0',
            B2='znane klatki, suma pchnięć przy wagach z t0',
            C1='bez przyszłych klatek: sam wygasający pęd',
            C2='bez przyszłych klatek: pęd + średni gradient zbioru')


def _klip(g):
    norma = torch.sqrt(sum((v * v).sum() for v in g.values()))
    if norma > 1.0:
        g = {n: v / (norma + 1e-6) for n, v in g.items()}
    return g


def grad_przy(u, par0, buf0, x, heat, ltrb, poz):
    """Gradient straty paczki przy wagach par0 (tryb uczenia, bf16, obcięcie do normy 1 jak w uczeniu)."""
    net = u.net; net.train()
    p = {n: t.detach().clone().requires_grad_(True) for n, t in par0.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o, _ = functional_call(net, (p, {n: b.clone() for n, b in buf0.items()}), (x,))
    L, _ = M.strata(o, heat, ltrb, poz)
    g = torch.autograd.grad(L, list(p.values()), allow_unused=True)
    return _klip({n: (gg.float() if gg is not None else torch.zeros_like(t)) for (n, t), gg in zip(p.items(), g)})


@torch.no_grad()
def lat_przy(u, sondy, par, buf):
    """Latent narzędzia 192 sond (fp32, tryb ewaluacji) przy podanych wagach i buforach."""
    X, tn = sondy[0], sondy[1]
    net = u.net; net.eval(); net.zapisuj_latent(True)
    out = []
    for s in range(0, len(X), 96):
        functional_call(net, (par, buf), (X[s:s + 96].to(u.dev),))
        a = tn[s:s + 96].to(u.dev).float(); lat = net._latent.float()
        out.append((lat * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1))
    net.zapisuj_latent(False)
    return torch.cat(out).cpu().numpy()


class Wirtualny:
    """AdamW w przestrzeni wag, bez ruszania sieci: przesunięcie D od wag θ0 i własny stan m, v, krok (kopie stanu
    optymalizatora z chwili t0). Kolejność jak torch.optim.AdamW: najpierw zanik wag, potem krok z pędu."""

    def __init__(self, u, par0):
        self.par0 = par0
        nazwy = {id(p): n for n, p in u.net.named_parameters()}
        self.h, self.m, self.v, self.k = {}, {}, {}, {}
        for grp in u.opt.param_groups:
            for p in grp['params']:
                n = nazwy[id(p)]; st = u.opt.state.get(p, {})
                self.h[n] = (grp['betas'][0], grp['betas'][1], grp['eps'], grp['weight_decay'])
                self.m[n] = st['exp_avg'].detach().clone() if 'exp_avg' in st else torch.zeros_like(p)
                self.v[n] = st['exp_avg_sq'].detach().clone() if 'exp_avg_sq' in st else torch.zeros_like(p)
                self.k[n] = float(st['step']) if 'step' in st else 0.0
        self.D = {n: torch.zeros_like(t) for n, t in par0.items()}

    @torch.no_grad()
    def krok(self, g, lr):
        """Jeden krok z gradientem g (dict albo None = zero). Zwraca przesunięcie tego kroku."""
        du = {}
        for n in self.D:
            b1, b2, eps, wd = self.h[n]
            gg = g[n] if g is not None else None
            self.m[n].mul_(b1); self.v[n].mul_(b2)
            if gg is not None:
                self.m[n].add_(gg, alpha=1 - b1); self.v[n].addcmul_(gg, gg, value=1 - b2)
            self.k[n] += 1
            bc1, bc2 = 1 - b1 ** self.k[n], 1 - b2 ** self.k[n]
            d = -(lr / bc1) * self.m[n] / (self.v[n].sqrt() / np.sqrt(bc2) + eps)
            if wd:
                d = d - lr * wd * (self.par0[n] + self.D[n])
            self.D[n].add_(d); du[n] = d
        return du

    def wagi(self):
        return {n: self.par0[n] + self.D[n] for n in self.D}


class Horyzont:
    """Obserwator kroków uczenia w jednym oknie [t0, t0 + HMAX): przed każdym krokiem liczy wszystkie warianty
    prognozy, po krokach z H_LISTA zapisuje przyrost zmierzony i przewidziane."""

    def __init__(self, u, zb, sondy, obecn, wek):
        self.sondy, self.obecn, self.wek = sondy, obecn, wek
        self.t0 = u.krok
        self.par0 = {n: p.detach().clone() for n, p in u.net.named_parameters()}
        self.buf0 = {n: b.detach().clone() for n, b in u.net.named_buffers()}
        self.z0 = lat_przy(u, sondy, self.par0, self.buf0)
        self.B, self.C1, self.C2 = (Wirtualny(u, self.par0) for _ in range(3))
        # średni gradient zbioru przy wagach z t0 (C2): paczki po 32 kolejno, połowa z odbiciem
        acc, n = None, 0
        for i, s in enumerate(range(0, len(zb) - 31, 32)):
            g = RW.gradient_paczki(u, zb, np.arange(s, s + 32), odbicie=bool(i % 2))
            acc = {k: v.clone() for k, v in g.items()} if acc is None else {k: acc[k] + v for k, v in g.items()}
            n += 1
        self.g_sr = {k: v / n for k, v in acc.items()}
        self.pA = np.zeros_like(self.z0); self.pB2 = np.zeros_like(self.z0)
        self.wyn = {}                                         # H -> dict(y=Δz zmierzone, warianty Δz)

    def czy(self, j):
        return True

    def przed(self, u, x, heat, ltrb, poz):
        lr = u._lr()
        # A: równanie jednego kroku na prawdziwych wagach i tej paczce
        z = PK.lat(u, self.sondy)
        nowa, ped = RW.czesci_kroku(u, RW.gradient_tensorow(u, x, heat, ltrb, poz))
        self.pA += (PK.lat(u, self.sondy, nowa) - z) + (PK.lat(u, self.sondy, ped) - z)
        # B i B2: ta sama paczka, gradient przy wagach z t0
        du = self.B.krok(grad_przy(u, self.par0, self.buf0, x, heat, ltrb, poz), lr)
        self.pB2 += lat_przy(u, self.sondy, {n: self.par0[n] + du[n] for n in du}, self.buf0) - self.z0
        # C1, C2: bez przyszłych klatek
        self.C1.krok(None, lr)
        self.C2.krok(self.g_sr, lr)

    def po(self, u):
        H = u.krok - self.t0
        if H not in H_LISTA:
            return
        o = self.obecn
        y = PK.lat(u, self.sondy) - self.z0
        pred = dict(A=self.pA.copy(), B=lat_przy(u, self.sondy, self.B.wagi(), self.buf0) - self.z0, B2=self.pB2.copy(),
                    C1=lat_przy(u, self.sondy, self.C1.wagi(), self.buf0) - self.z0,
                    C2=lat_przy(u, self.sondy, self.C2.wagi(), self.buf0) - self.z0)
        self.wyn[H] = dict(y=y[o].astype(np.float64), **{k: v[o].astype(np.float64) for k, v in pred.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ziarno', type=int, required=True)
    a = ap.parse_args(argv)
    s = a.ziarno
    out = Path(os.environ['MALY_VIT_WYNIKI']); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baza = D.baza(); sondy = D.sondy(); obecn = sondy[4]
    u = T.Uczen(s); rng = np.random.default_rng(s)
    lat = []
    u.kroki(baza, OKNA[0], rng, sondy=sondy, latent_out=lat)
    wek = SI.wektor(np.stack([l for k, l in lat if k >= OKNA[0] - 40]).astype(np.float32).mean(0), obecn)
    print(f's{s}: rozgrzewka {OKNA[0]} kroków ({time.time() - t0:.0f} s), siła tożsamości narzędzia '
          f'{wek["sila_narzedzia"]:.2f}, negatywów {wek["sila_negatywow"]:.2f}', flush=True)
    Y = {w: np.full((len(OKNA), len(H_LISTA), int(obecn.sum())), np.nan) for w in ('y',) + WARIANTY}
    sse_lat = {w: np.zeros(len(H_LISTA)) for w in WARIANTY}; ss_lat = np.zeros(len(H_LISTA))
    for i, t_start in enumerate(OKNA):
        assert u.krok == t_start
        tw = time.time()
        obs = Horyzont(u, baza, sondy, obecn, wek)
        u.kroki(baza, HMAX, rng, obserwator=obs)
        for j, H in enumerate(H_LISTA):
            r = obs.wyn[H]
            Y['y'][i, j] = PK.ds(r['y'], wek)
            ss_lat[j] += float((r['y'] ** 2).sum())
            for w in WARIANTY:
                Y[w][i, j] = PK.ds(r[w], wek)
                sse_lat[w][j] += float(((r['y'] - r[w]) ** 2).sum())
        um = {w: PK.umiejetnosc(Y['y'][i], Y[w][i]) for w in WARIANTY}
        print(f's{s} okno {t_start}-{t_start + HMAX} ({time.time() - tw:.0f} s): umiejętność H=1/10/100/500 ' +
              '  '.join(f'{w} ' + '/'.join(f'{PK.umiejetnosc(Y["y"][i, H_LISTA.index(H)], Y[w][i, H_LISTA.index(H)]):.0f}'
                                            for H in (1, 10, 100, 500)) for w in WARIANTY), flush=True)
    # umiejętność po oknach i sondach, dla każdego H
    tab = {}
    for w in WARIANTY:
        tab[w] = dict(opis=OPIS[w], tozsamosc={}, caly_latent={}, znak={}, tozsamosc_okna={})
        for j, H in enumerate(H_LISTA):
            y, p = Y['y'][:, j], Y[w][:, j]
            tab[w]['tozsamosc'][H] = round(PK.umiejetnosc(y, p), 2)
            tab[w]['caly_latent'][H] = round(float(100 * (1 - sse_lat[w][j] / ss_lat[j])), 2)
            tab[w]['znak'][H] = round(float((np.sign(y) == np.sign(p)).mean()), 3)
            tab[w]['tozsamosc_okna'][H] = [round(PK.umiejetnosc(y[i], p[i]), 1) for i in range(len(OKNA))]
        ok = [H for H in H_LISTA if tab[w]['tozsamosc'][H] >= 90]
        ciag = 0
        for H in H_LISTA:
            if tab[w]['tozsamosc'][H] >= 90:
                ciag = H
            else:
                break
        tab[w]['horyzont_90'] = ciag
        tab[w]['H_z_90_procent'] = ok
    ruch = {H: dict(sd_przyrostu=float(Y['y'][:, j].std()), sredni_przyrost=float(Y['y'][:, j].mean()))
            for j, H in enumerate(H_LISTA)}
    wyn = dict(ziarno=s, okna=list(OKNA), H=list(H_LISTA), warianty=tab, ruch_tozsamosci=ruch,
               wektor=dict(sila_narzedzia=wek['sila_narzedzia'], sila_negatywow=wek['sila_negatywow']),
               sekundy=time.time() - t0)
    (out / f'HORYZONT_s{s}.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out / f'horyzont_s{s}.npz', H=np.array(H_LISTA), okna=np.array(OKNA), **Y)
    for w in WARIANTY:
        print(f'{w:3s} horyzont 90 %: {tab[w]["horyzont_90"]:4d} kroków  | ' +
              ' '.join(f'H{H}:{tab[w]["tozsamosc"][H]:.0f}' for H in H_LISTA) + f'  ({OPIS[w]})', flush=True)
    print(f's{s}: gotowe ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
