"""Równanie Langevina latentu sondy z uczonymi członami; budowa z TFT (`Projekt/experiments/tft_langevin_latent_56/model.py`).

    z_{t+1} = z_t + D_t + JP_t + W_t + N_t

    D_t   dryf           = f_D(z_t, v_t, a_t)                                 MLP, tylko stan bieżący
    JP_t  jądro pamięci  = sum_{l=0..31} A_h(t, t−l) V n_{t−l}                32 dawne stany tej samej sondy
                           A_h = rho_h^l · obrót(omega_h l) · podobieństwo stanów
    W_t   wymuszenie     = Wo A_t + f_W(A_t, z_t)
                           A_t = sum_{k=0..15} kappa_h(k) · (1/32) sum_b <q_h(sonda, t), k_h(b, t−k)> / sqrt(r) · v_h(b, t−k)
                           b = obrazy paczki t−1−k (tylko klatki już użyte w uczeniu; paczka kroku t to przyszłe
                           klatki i w prognozie jest zabroniona); klucz z latentu obrazu (narzędzie, cały obraz) i flagi narzędzia;
                           wartość z sygnału błędu obrazu (gradient straty po jego latencie) albo, w wariancie
                           bez gradientu, z tych samych cech co klucz
    N_t   szum kolorowy  = sum_{l=1..31} sum_m g_m(l) U_m V_m^T r_{t−l},   r = krok − (D + JP + W)

v_t = z_t − z_{t−1}, a_t = v_t − v_{t−1}. Człony nie dzielą parametrów, więc każdy da się wyłączyć.
Do równania nie wchodzą przyszłe klatki: W widzi paczki t−1..t−16, D i JP stany do t, N reszty sprzed chwili t.
Różnice wobec TS opisuje PLAN.md: okno 32 kroki zamiast 56 dni, wymuszenie z paczki obrazów zamiast dnia kalendarza.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768
OKNO = 32          # pamięć JP i N w krokach AdamW
OPOZ = 16          # ile ostatnich, już użytych paczek widzi wymuszenie (t−1, …, t−16)
PACZKA = 32
CZLONY = ('D', 'JP', 'W', 'N')


class MLP(nn.Module):
    def __init__(self, n_wej, n_ukr, n_wyj):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_wej), nn.Linear(n_wej, n_ukr), nn.GELU(),
                                 nn.Linear(n_ukr, n_ukr), nn.GELU(), nn.Linear(n_ukr, n_wyj))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class JadroPamieci(nn.Module):
    """Jak w TS: waga dawnego kroku to zanik rho^l, obrót o omega·l i podobieństwo stanów (uwaga liniowa)."""

    def __init__(self, n_wej, glowic=4, r=48):
        super().__init__()
        self.H, self.r = glowic, r
        self.Wq = nn.Linear(n_wej, glowic * r, bias=False)
        self.Wk = nn.Linear(n_wej, glowic * r, bias=False)
        self.Wv = nn.Linear(n_wej, glowic * r, bias=False)
        self.Wo = nn.Linear(glowic * r, D)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)
        polowiczne = [2.0, 4.0, 8.0, 16.0][:glowic]
        rho0 = [2.0 ** (-1.0 / h) for h in polowiczne]
        self.theta = nn.Parameter(torch.tensor([math.log(v / (1 - v)) for v in rho0]))
        self.omega = nn.Parameter(torch.zeros(glowic, r // 2))

    def _obroc(self, x, kroki):
        kat = self.omega[None, :, None, :] * kroki[None, None, :, None]
        c, s = torch.cos(kat), torch.sin(kat)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def wagi(self, n):
        B, L, _ = n.shape
        q = self.Wq(n).view(B, L, self.H, self.r).transpose(1, 2)
        k = self.Wk(n).view(B, L, self.H, self.r).transpose(1, 2)
        kroki = torch.arange(L, device=n.device, dtype=n.dtype)
        q, k = self._obroc(q, kroki), self._obroc(k, kroki)
        d = kroki[:, None] - kroki[None, :]
        maska = (d >= 0) & (d < OKNO)
        rho = torch.sigmoid(self.theta)
        obw = (rho[:, None, None] ** d.clamp_min(0)[None]) * maska[None]
        A = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-1, -2) * obw[None]
        return A / (A.sum(-1, keepdim=True) + 1e-6)

    def forward(self, n):
        B, L, _ = n.shape
        v = self.Wv(n).view(B, L, self.H, self.r).transpose(1, 2)
        out = (self.wagi(n) @ v).transpose(1, 2).reshape(B, L, self.H * self.r)
        return self.Wo(out)


class Wymuszenie(nn.Module):
    """Uwaga sondy na obrazy ostatnich OPOZ paczek. Bez softmaksu i bez normowania wag: obraz podobny do sondy
    z dużym błędem pcha latent sondy mocniej, a znak podobieństwa może być ujemny (jak jądro styczne sieci).
    kappa_h(k) to wyuczona waga paczki t−1−k (pamięć wymuszenia; start 0,9^k jak pęd AdamW). Przesunięcie o jeden
    krok (bez paczki t) robi `uczenie.Dane.wsad`: klucz i wartość zaczynają się od paczki c−OPOZ."""

    def __init__(self, n_q, n_k, n_v, glowic=8, r=64, n_ukr=1024):
        super().__init__()
        self.H, self.r = glowic, r
        self.nq, self.nk = nn.LayerNorm(n_q), nn.LayerNorm(n_k)
        self.Wq = nn.Linear(n_q, glowic * r, bias=False)
        self.Wk = nn.Linear(n_k, glowic * r, bias=False)
        self.Wv = nn.Linear(n_v, glowic * r, bias=False)
        self.Wo = nn.Linear(glowic * r, D)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)
        self.kappa = nn.Parameter(0.9 ** torch.arange(OPOZ, dtype=torch.float32)[None].repeat(glowic, 1))
        self.glowa = MLP(glowic * r + D, n_ukr, D)

    def forward(self, q_in, zn, klucz, wart):
        """q_in (P, T, n_q), zn (P, T, D), klucz (T + OPOZ − 1, 32, n_k), wart (T + OPOZ − 1, 32, n_v)."""
        P, T, _ = q_in.shape
        q = self.Wq(self.nq(q_in)).view(P, T, self.H, self.r) / math.sqrt(self.r)
        k = self.Wk(self.nk(klucz)).view(-1, PACZKA, self.H, self.r)
        v = self.Wv(wart).view(-1, PACZKA, self.H, self.r)
        out = q.new_zeros(P, T, self.H, self.r)
        for j in range(OPOZ):                                           # wiersz t−j klucza = paczka t−1−j
            kj = k[OPOZ - 1 - j:OPOZ - 1 - j + T]
            vj = v[OPOZ - 1 - j:OPOZ - 1 - j + T]
            s = torch.einsum('pthr,tbhr->pthb', q, kj) * self.kappa[:, j][None, None, :, None]
            out = out + torch.einsum('pthb,tbhr->pthr', s, vj) / PACZKA
        out = out.reshape(P, T, self.H * self.r)
        return self.Wo(out) + self.glowa(torch.cat([out, zn], -1))


class SzumKolorowy(nn.Module):
    """Jak w TS: warunkowa średnia szumu z własnych reszt równania, M głów z jądrem g_m(l) i niskim rzędem."""

    def __init__(self, glowic=4, r=32):
        super().__init__()
        self.M, self.r = glowic, r
        self.V = nn.Linear(D, glowic * r, bias=False)
        self.U = nn.Linear(glowic * r, D, bias=False)
        nn.init.zeros_(self.U.weight)
        polowiczne = [1.0, 3.0, 7.0, 21.0][:glowic]
        rho0 = [2.0 ** (-1.0 / h) for h in polowiczne]
        self.theta = nn.Parameter(torch.tensor([math.log(v / (1 - v)) for v in rho0]))
        self.omega = nn.Parameter(torch.tensor([0.0, 0.0, 0.3, 1.0][:glowic]))

    def jadro(self):
        l = torch.arange(1, OKNO, device=self.theta.device, dtype=self.theta.dtype)
        rho = torch.sigmoid(self.theta)
        return rho[:, None] ** (l[None] - 1) * torch.cos(self.omega[:, None] * (l[None] - 1))       # (M, 31)

    def forward(self, r):
        B, L, _ = r.shape
        h = self.V(r).view(B, L, self.M, self.r)
        g = self.jadro()
        x = h.permute(0, 2, 3, 1).reshape(B, self.M * self.r, L)
        w = g.flip(-1)[:, None, None, :].expand(self.M, self.r, 1, OKNO - 1).reshape(self.M * self.r, 1, OKNO - 1)
        y = F.conv1d(F.pad(x, (OKNO - 2, 0)), w, groups=self.M * self.r)[..., :L]     # y_i = sum_{m=0..30} g(m+1) x_{i−m}
        y = torch.cat([torch.zeros_like(y[..., :1]), y[..., :-1]], -1)                   # wyjście t = sum_{l=1..31} g(l) x_{t−l}
        return self.U(y.reshape(B, self.M, self.r, L).permute(0, 3, 1, 2).reshape(B, L, self.M * self.r))


class Rownanie(nn.Module):
    def __init__(self, stat, n_k, n_v, n_ukr=1024, w_glowic=8, w_r=64):
        super().__init__()
        for k, v in stat.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.D = MLP(3 * D, n_ukr, D)
        self.tok = nn.Linear(2 * D, 256)
        self.JP = JadroPamieci(256)
        self.W = Wymuszenie(2 * D, n_k, n_v, w_glowic, w_r, n_ukr)
        self.N = SzumKolorowy()

    def forward(self, z_pelne, zt, klucz, wart, wylacz=()):
        """z_pelne (P, T + 1, D): stany sond w krokach c..c+T; zt (P, T, D): tło sond w krokach c..c+T−1;
        klucz, wart (T + OPOZ − 1, 32, ·): cechy paczek c−OPOZ..c+T−2 (bez paczki bieżącego kroku). Zwraca krok, prawdę i człony (P, T, D)."""
        z = z_pelne[:, :-1]
        prawda = z_pelne[:, 1:] - z
        v = torch.zeros_like(z)
        v[:, 1:] = z[:, 1:] - z[:, :-1]
        a = torch.zeros_like(z)
        a[:, 2:] = v[:, 2:] - v[:, 1:-1]
        zn, vn, an = (z - self.mu_z) / self.sd_z, v / self.sd_v, a / self.sd_a
        s = self.sd_krok
        czl = {'D': self.D(torch.cat([zn, vn, an], -1)) * s,
               'JP': self.JP(self.tok(torch.cat([zn, vn], -1))) * s}
        q = torch.cat([zn, (zt - self.mu_t) / self.sd_t], -1)
        czl['W'] = self.W(q, zn, klucz, wart) * s if 'W' not in wylacz else torch.zeros_like(z)
        for c in wylacz:
            if c in ('D', 'JP'):
                czl[c] = torch.zeros_like(czl[c])
        det = czl['D'] + czl['JP'] + czl['W']
        reszta = (prawda - det) / s
        reszta[:, :2] = 0                                                  # bez przyspieszenia nie ma reszty
        czl['N'] = self.N(reszta) * s if 'N' not in wylacz else torch.zeros_like(z)
        return det + czl['N'], prawda, czl


def kara_rozdzielnosci(czl, sl):
    """Jak w TS: średni kwadrat cosinusa między wyjściami członów na ocenianych krokach."""
    wek = {c: czl[c][:, sl].reshape(-1) for c in CZLONY}
    kara, par = 0.0, 0
    for i, a in enumerate(CZLONY):
        for b in CZLONY[i + 1:]:
            kara = kara + (wek[a] @ wek[b]) ** 2 / ((wek[a] @ wek[a]) * (wek[b] @ wek[b]) + 1e-12)
            par += 1
    return kara / par
