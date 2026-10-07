"""Pchnięcie obrazów: jak jeden obraz w paczce przesuwa cząstki (tokeny) sondy w jednym kroku AdamW.

Krok k (aktualizacja k -> k+1) gęstej historii, stan dokładny z stan_uczenia.Odtworzenie (wagi, bufory BN, AdamW).
Aktualizacja AdamW rozkłada się jak w maly_vit_rownanie_latentu/nagranie.py:
  nowa  d = −lr/b1 (1 − β1) c g / mianownik      (gradient paczki po obcięciu: c = min(1, 1/||g||))
  ped   d = −lr/b1 β1 m / mianownik             (pęd: pamięć poprzednich paczek)
  zanik d = −lr wd θ,                            mianownik = sqrt((β2 v + (1 − β2)(c g)^2)/b2) + eps.
Strata paczki to suma składników obrazów przy normach całej paczki (liczba pozytywnych komórek i suma wag ramek), więc
gradient paczki = suma gradientów składników obrazów, liczonych w przebiegu całej paczki (z jej statystykami BatchNorm).
Przy mianowniku ustalonym z całej paczki krok wag jest liniowy w gradiencie, czyli d_nowa = suma d_obrazów dokładnie.
Pchnięcie cząstek: S(d) = z(θ + d) − z(θ) sondy w trybie eval, float32, bufory BN sprzed kroku (jak nagranie.py).
Obraz spoza paczki (kandydat: baza, SD, obraz odszukany dla SD) dokładany jako 33. obraz tej paczki: jego składnik
straty przy normach paczki, ta sama obcinka c i ten sam mianownik, czyli „co by zrobił, gdyby był w tej paczce”.

Miary pchnięcia na tokenach narzędzia sondy:
  tozsamosc  zmiana siły tożsamości: średnia po tokenach narzędzia ((S / sd) · w), w = kierunek LDA narzędzie / tło
             z tokenów 12 klatek w tym kroku (uproszczenie metody sila.wektor: tokeny zamiast średnich sond);
  strata     zmiana straty detekcji sondy w pierwszym rzędzie, tylko przez latent: <dL/dz, S> (ujemna = pomaga);
  do_konca   kosinus S z kierunkiem do stanu końcowego z_koniec − z_k (tylko wyjaśnienie: używa przyszłości).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def strata_obrazow(M, o, heat, ltrb, poz, n=None, wsum=None):
    """Składniki straty (M.strata) każdego obrazu [B] przy normach n (pozytywne komórki) i wsum (wagi ramek) całej
    paczki; suma składników = M.strata(o, ...). Zwraca (składniki, n, wsum)."""
    p = o[..., 0].float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    l_pos = -(torch.log(p) * (1 - p) ** 2 * pos).flatten(1).sum(1)
    l_neg = -(torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * (1 - pos)).flatten(1).sum(1)
    if n is None:
        n = pos.sum().clamp(min=1)
    l_heat = (l_pos + l_neg) / n
    m = poz > 0
    if wsum is None:
        wsum = poz[m].sum() if m.any() else torch.ones((), device=o.device)
    l_box = torch.zeros(len(o), device=o.device)
    if m.any():
        pr = o[..., 1:].float().exp()
        g = 1 - M._giou(pr[m], ltrb[m])
        wb = torch.zeros_like(poz, dtype=torch.float32); wb[m] = (g * poz[m]).float()
        l_box = wb.flatten(1).sum(1) / wsum
    return l_heat + 2 * l_box, n, wsum


class Pchniecie:
    def __init__(self, od, X_sond, maski_sond, ramki_sond):
        """od: stan_uczenia.Odtworzenie; X_sond uint8 [F, 320, 480, 3] (12 klatek gęstej historii), maski_sond bool
        [F, 600] (tokeny narzędzia), ramki_sond: lista [F] ramek xyxy w pikselach 480 x 320 (do straty sondy)."""
        self.od, self.M = od, od.M
        self.dev = 'cuda'
        self.Xs = torch.from_numpy(np.ascontiguousarray(X_sond)).permute(0, 3, 1, 2).float().to(self.dev) / 255.
        self.maski = np.asarray(maski_sond, bool)
        self.ramki = ramki_sond
        self.k = None

    # ------------------------------------------------------------ latent sondy
    @torch.no_grad()
    def _z(self, params, sonda):
        net = self.u.net
        net.eval(); net.zapisuj_latent(True)
        try:
            torch.func.functional_call(net, (params, self.bufory), (self.Xs[sonda:sonda + 1],))
            return net._latent[0].float()
        finally:
            net.zapisuj_latent(False)

    def S(self, d, sonda):
        """Pchnięcie cząstek sondy od zmiany wag d (dict nazwa -> tensor): z(θ + d) − z(θ) [600, 768]."""
        return self._z({n: self.theta[n] + d[n] if n in d else self.theta[n] for n in self.theta}, sonda) - self.z0[sonda]

    # ------------------------------------------------------------ krok
    def przygotuj(self, k, sonda):
        """Stan kroku k, gradienty paczki i jej obrazów, części kroku AdamW; z0 sondy."""
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.u = u = self.od.stan(k)
        self.k = k
        net = u.net
        self.nazwy = [n for n, _ in net.named_parameters()]
        self.theta = {n: p.detach().clone() for n, p in net.named_parameters()}
        self.bufory = {n: b.detach().clone() for n, b in net.named_buffers()}
        self.z0 = {}
        for s in range(len(self.Xs)):
            self.z0[s] = self._z(self.theta, s)
        x, heat, ltrb, poz, idx = self.od.paczka(k)
        self.paczka = dict(x=x, heat=heat, ltrb=ltrb, poz=poz, idx=idx)
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        bufory_po = {n: b.detach().clone() for n, b in net.named_buffers()}      # BN po kroku (biegnące statystyki)
        skl, self.n_pos, self.wsum = strata_obrazow(self.M, o, heat, ltrb, poz)
        L = skl.sum()
        params = [p for p in net.parameters()]
        gB = torch.autograd.grad(L, params, retain_graph=True)
        norma = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gB)))
        self.c = min(1.0, 1.0 / (norma + 1e-6))                                   # clip_grad_norm_(…, 1.0)
        self.g_obrazow = []
        for i in range(len(skl)):
            gi = torch.autograd.grad(skl[i], params, retain_graph=i < len(skl) - 1)
            self.g_obrazow.append([g.detach() * self.c for g in gi])
        self._przywroc_bufory()                       # bufory BN sprzed kroku (dopiero po gradientach: są w grafie)
        self._mianownik([g * self.c for g in gB])
        self.d_ped, self.d_zanik = self._ped_zanik()
        self.d_nowa = self._d([g * self.c for g in gB])
        # części ruchu sondy
        wynik = {}
        for s in range(len(self.Xs)):
            wynik[s] = dict(nowa=self.S(self.d_nowa, s), ped=self.S(self.d_ped, s), zanik=self.S(self.d_zanik, s))
            wynik[s]['bn'] = self._z_buf(bufory_po, s) - self.z0[s]
        self.czesci = wynik
        self.bufory_po = bufory_po
        self.lr = u.opt.param_groups[0]['lr']
        return wynik

    @torch.no_grad()
    def _przywroc_bufory(self):
        """Biegnące statystyki BN sprzed kroku (przejście uczące je zmienia); wag nie dotyka, więc graf zostaje."""
        for n, b in self.u.net.named_buffers():
            b.copy_(self.bufory[n])

    @torch.no_grad()
    def _z_buf(self, bufory, sonda):
        stare = self.bufory
        self.bufory = bufory
        try:
            return self._z(self.theta, sonda)
        finally:
            self.bufory = stare

    @torch.no_grad()
    def _mianownik(self, g_lista):
        """Mianownik AdamW kroku k (z gradientu całej paczki), ustalony dla wszystkich rozkładów."""
        opt, u = self.u.opt, self.u
        self.mian, self.wsp = {}, {}
        n_of = {id(p): n for n, p in u.net.named_parameters()}
        gmap = {id(p): g for p, g in zip(u.net.parameters(), g_lista)}
        lr_akt = u._lr()
        for grp in opt.param_groups:
            (b1, b2), eps, wd = grp['betas'], grp['eps'], grp['weight_decay']
            for p in grp['params']:
                st = opt.state.get(p, {})
                krok = float(st['step']) + 1 if 'step' in st else 1.0
                v = st['exp_avg_sq'] if 'exp_avg_sq' in st else torch.zeros_like(p)
                g = gmap[id(p)].float()
                mian = (b2 * v + (1 - b2) * g * g).sqrt() / math.sqrt(1 - b2 ** krok) + eps
                n = n_of[id(p)]
                self.mian[n] = mian
                self.wsp[n] = dict(lr=lr_akt, b1=b1, krok=krok, wd=wd)

    @torch.no_grad()
    def _d(self, g_lista):
        """Krok wag (część „nowa”) od gradientu przy ustalonym mianowniku."""
        out = {}
        for n, g in zip(self.nazwy, g_lista):
            w = self.wsp[n]
            out[n] = -(w['lr'] / (1 - w['b1'] ** w['krok'])) * (1 - w['b1']) * g.float() / self.mian[n]
        return out

    @torch.no_grad()
    def _ped_zanik(self):
        opt = self.u.opt
        n_of = {id(p): n for n, p in self.u.net.named_parameters()}
        ped, zan = {}, {}
        for grp in opt.param_groups:
            for p in grp['params']:
                n = n_of[id(p)]; w = self.wsp[n]; st = opt.state.get(p, {})
                m = st['exp_avg'] if 'exp_avg' in st else torch.zeros_like(p)
                ped[n] = -(w['lr'] / (1 - w['b1'] ** w['krok'])) * w['b1'] * m / self.mian[n]
                zan[n] = -w['lr'] * w['wd'] * p.detach()
        return ped, zan

    # ------------------------------------------------------------ obrazy
    def pchniecia_paczki(self, sonda):
        """Pchnięcie każdego obrazu paczki kroku [32, 600, 768] (suma ≈ część „nowa”)."""
        return [self.S(self._d(g), sonda) for g in self.g_obrazow]

    def gradient_kandydata(self, x1, heat1, ltrb1, poz1):
        """Gradient składnika straty obrazu dołożonego jako 33. do paczki kroku (normy paczki, ta sama obcinka)."""
        net = self.u.net
        P = self.paczka
        x = torch.cat([P['x'], x1]); heat = torch.cat([P['heat'], heat1]); ltrb = torch.cat([P['ltrb'], ltrb1])
        poz = torch.cat([P['poz'], poz1])
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        skl, _, _ = strata_obrazow(self.M, o, heat, ltrb, poz, self.n_pos, self.wsum)
        g = torch.autograd.grad(skl[-1], list(net.parameters()))
        self._przywroc_bufory()
        return [gg.detach() * self.c for gg in g]

    def pchniecie_kandydata(self, x_uint8, ramki, sonda):
        """x_uint8 [320, 480, 3], ramki: lista xyxy (480 x 320). Zwraca S [600, 768]."""
        x1 = torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(2, 0, 1)[None].float().to(self.dev) / 255.
        h, w = 20, 30
        c = self.M.cele([np.asarray(ramki, np.float32).reshape(-1, 4)], h, w)
        g = self.gradient_kandydata(x1, c[0].to(self.dev), c[1].to(self.dev), c[2].to(self.dev))
        return self.S(self._d(g), sonda)

    # ------------------------------------------------------------ miary
    def kierunki_miar(self, Z_k, Z_koniec, sonda):
        """Kierunek tożsamości (LDA z tokenów 12 klatek kroku k), gradient straty sondy po latencie, kierunek do końca."""
        Zk = np.asarray(Z_k, np.float32)                                         # [F, 600, 768] latent 12 klatek
        nar = Zk[self.maski]; tlo = Zk[~self.maski][::7]
        mu_t, mu_b = nar.mean(0), tlo.mean(0)
        sd = tlo.std(0) + 1e-6
        A = np.concatenate([(nar - mu_t) / sd, (tlo - mu_b) / sd])
        C = A.T @ A / len(A)
        C = .9 * C + .1 * np.trace(C) / len(C) * np.eye(len(C))
        w = np.linalg.solve(C, (mu_t - mu_b) / sd); w /= np.linalg.norm(w)
        self.w_id, self.sd_id = torch.from_numpy(w).float().to(self.dev), torch.from_numpy(sd).float().to(self.dev)
        self.grad_straty = self._grad_straty_sondy(sonda)
        dk = np.asarray(Z_koniec[sonda], np.float32) - np.asarray(Z_k[sonda], np.float32)
        self.do_konca = torch.from_numpy(dk).to(self.dev)

    def _grad_straty_sondy(self, sonda):
        """dL_sondy/dz [600, 768]: strata detekcji sondy (ramki z obrysu) przez głowę, gdy zmienia się tylko latent."""
        net = self.u.net
        net.eval()
        b = net.bloki[-1]
        pam = {}
        h = b.register_forward_hook(lambda m, a, o: pam.update(q=o.detach()))
        net.zapisuj_latent(True)
        try:
            with torch.no_grad():
                net(self.Xs[sonda:sonda + 1])
            z = net._latent.detach().float().clone().requires_grad_(True)
            zq = net._latent.detach().float()
        finally:
            h.remove(); net.zapisuj_latent(False)
        delta = F.linear(z - zq, b.mlp.fc2.weight)
        o = net.glowa(net.norm(pam['q'].float() + delta)).view(1, 20, 30, 5)
        c = self.M.cele([np.asarray(self.ramki[sonda], np.float32).reshape(-1, 4)], 20, 30)
        L, _ = self.M.strata(o, c[0].to(self.dev), c[1].to(self.dev), c[2].to(self.dev))
        g, = torch.autograd.grad(L, z)
        return g[0]

    def miary(self, S, sonda, maska=None):
        """Miary pchnięcia S [600, 768] na wybranych tokenach sondy (domyślnie narzędzia): liczby i mapy [600]."""
        m = torch.from_numpy(np.asarray(self.maski[sonda] if maska is None else maska, bool)).to(self.dev)
        sid = (S / self.sd_id) @ self.w_id                                       # [600] zmiana siły tożsamości
        dl = (S * self.grad_straty).sum(-1)                                      # [600] zmiana straty sondy
        St = S[m].flatten(); Dk = self.do_konca[m].flatten()
        return dict(tozsamosc=float(sid[m].mean()) if m.any() else 0., strata=float(dl.sum()),
                    do_konca=float((St @ Dk) / (St.norm() * Dk.norm() + 1e-12)) if m.any() else 0.,
                    dlugosc=float(S[m].norm(dim=-1).mean()) if m.any() else float(S.norm(dim=-1).mean()),
                    mapa_tozsamosc=sid.cpu().numpy(), mapa_strata=dl.cpu().numpy(),
                    mapa_dlugosc=S.norm(dim=-1).cpu().numpy())
