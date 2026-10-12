"""Uczenie małego ViT od zera: kroki na zbiorze, zapis latentu sond, ocena mAP@0,5 (PLAN.md).

Stała liczba kroków na obrót (równa dla ramion), paczka 32, AdamW, rozgrzewka lr, potem stały lr; bf16.
Augmentacja: odbicie poziome 0,5 i jasność/kontrast ±20 %.
Latent: co ZAPIS_CO kroków średnia aktywacji 768 neuronów MLP ostatniego bloku po tokenach narzędzia i po tokenach tła
każdej z 192 sond → (192, 2, 768) float16.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import model as M

PACZKA = 32
LR, WD, ROZGRZEWKA = 5e-4, 0.05, 300
ZAPIS_CO = 10


class Uczen:
    def __init__(self, ziarno, device='cuda'):
        torch.manual_seed(ziarno)
        self.dev = device
        self.net = M.MalyViT().to(device)
        dec = [p for n, p in self.net.named_parameters() if p.ndim >= 2]
        nodec = [p for n, p in self.net.named_parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW([dict(params=dec, weight_decay=WD), dict(params=nodec, weight_decay=0)], lr=LR,
                                     betas=(0.9, 0.999))
        self.krok = 0
        self.log = []

    # --- zapis/odczyt
    def stan(self):
        return dict(net=self.net.state_dict(), opt=self.opt.state_dict(), krok=self.krok)

    def wczytaj(self, st):
        self.net.load_state_dict(st['net']); self.opt.load_state_dict(st['opt']); self.krok = st['krok']

    # --- uczenie
    def _lr(self):
        return LR * min(1.0, (self.krok + 1) / ROZGRZEWKA)

    def kroki(self, zb, n, rng, sondy=None, latent_out=None, co_ocena=None, ocena=None, obserwator=None):
        """n kroków na zbiorze zb (losowanie ze zwracaniem między epokami: permutacje kolejno).
        obserwator (wersja 4.2, sprawdzian równania jednego kroku): gdy obserwator.czy(j) dla j-tego kroku biegu,
        obserwator.przed(u, x, heat, ltrb, poz) dostaje dokładnie paczkę kroku (po odbiciu i jasności) przed krokiem,
        a obserwator.po(u) jest wołany po kroku. Nie zmienia uczenia ani losowania."""
        self.net.train()
        kol = []
        cel0, cel1 = zb.cele
        t0 = time.time()
        for _j in range(n):
            if len(kol) < PACZKA:
                kol += rng.permutation(len(zb)).tolist()
            idx = np.array(kol[:PACZKA]); kol = kol[PACZKA:]
            odb = rng.random(PACZKA) < .5
            x = torch.from_numpy(zb.X[idx]).to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.
            ob = torch.from_numpy(odb).to(self.dev)
            x = torch.where(ob[:, None, None, None], x.flip(3), x)
            a = torch.from_numpy(rng.uniform(.8, 1.2, (PACZKA, 2)).astype(np.float32)).to(self.dev)
            m = x.mean((1, 2, 3), keepdim=True)
            x = ((x - m) * a[:, 0, None, None, None] + m * a[:, 1, None, None, None]).clamp(0, 1)
            it = torch.from_numpy(idx)
            heat = torch.where(torch.from_numpy(odb)[:, None, None], cel1[0][it], cel0[0][it]).to(self.dev)
            ltrb = torch.where(torch.from_numpy(odb)[:, None, None, None], cel1[1][it], cel0[1][it]).to(self.dev)
            poz = torch.where(torch.from_numpy(odb)[:, None, None], cel1[2][it], cel0[2][it]).to(self.dev)
            for g in self.opt.param_groups:
                g['lr'] = self._lr()
            patrz = obserwator is not None and obserwator.czy(_j)
            if patrz:
                obserwator.przed(self, x, heat, ltrb, poz)
                self.net.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = self.net(x)
            L, cz = M.strata(o, heat, ltrb, poz)
            self.opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.krok += 1
            if patrz:
                obserwator.po(self)
                self.net.train()
            if self.krok % 50 == 0:
                self.log.append(dict(krok=self.krok, strata=float(L), **cz, s_na_krok=(time.time() - t0) / 50))
                t0 = time.time()
            if sondy is not None and latent_out is not None and self.krok % ZAPIS_CO == 0:
                latent_out.append((self.krok, self.latent_sond(sondy)))
                self.net.train()
            if co_ocena and ocena is not None and self.krok % co_ocena == 0:
                ocena(self)
                self.net.train()

    # --- odczyty
    @torch.no_grad()
    def latent_sond(self, sondy):
        X, tn, tt = sondy[0], sondy[1], sondy[2]
        self.net.eval(); self.net.zapisuj_latent(True)
        out = []
        for s in range(0, len(X), 96):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, lat = self.net(X[s:s + 96].to(self.dev))
            lat = lat.float()
            a = tn[s:s + 96].to(self.dev).float(); b = tt[s:s + 96].to(self.dev).float()
            nar = (lat * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1)
            tlo = (lat * b[..., None]).sum(1) / b.sum(1, keepdim=True).clamp(min=1)
            out.append(torch.stack([nar, tlo], 1))
        self.net.zapisuj_latent(False)
        return torch.cat(out).half().cpu().numpy()

    @torch.no_grad()
    def wykryj(self, X_uint8, paczka=64):
        """X_uint8: (N,H,W,3). Zwraca listę (ramki (k,4), pewności (k,))."""
        self.net.eval()
        wy = []
        for s in range(0, len(X_uint8), paczka):
            x = torch.from_numpy(X_uint8[s:s + paczka]).to(self.dev).permute(0, 3, 1, 2).float() / 255.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = self.net(x)
            bx, sc = M.dekoduj(o)
            wy += list(zip(bx.cpu().numpy(), sc.cpu().numpy()))
        return wy

    @torch.no_grad()
    def jakosc_sond(self, sondy):
        X, ramki = sondy[0], sondy[3]
        self.net.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = self.net(X.to(self.dev))
        bx, sc = M.dekoduj(o)
        bx, sc = bx.cpu().numpy(), sc.cpu().numpy()
        q = np.full(len(X), np.nan)
        for i, r in enumerate(ramki):
            if r is not None:
                q[i] = float((sc[i] * iou(bx[i], np.array(r, np.float32))).max())
            else:
                q[i] = -float(sc[i].max())                                   # sonda bez narzędzia: ujemna pewność
        return q


def iou(b, g):
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)


def ap50(detekcje, ramki_gt):
    """AP@0,5 jednej klasy, interpolacja 101-punktowa jak w COCO."""
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        for b, s in zip(bx, sc):
            wszystkie.append((float(s), i, b))
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (s, i, b) in enumerate(wszystkie):
        g = ramki_gt[i]
        if len(g):
            v = np.array([iou(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= .5:
                tp[j] = 1; uzyte[i][k] = True
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101
