"""Obserwator gęstego zapisu ruchu cząsteczek i sił podczas uczenia (do fizyki ruchu: MSD, VACF, temperatura, siły).

Podpinany do trening.Uczen.kroki(obserwator=...). Nie zmienia uczenia ani losowania: sondy przechodzą przez sieć
w trybie eval (bez aktualizacji statystyk BatchNorm), a skład paczki jest podsłuchiwany na odczycie zb.X[idx].

Co zapisuje (katalog wyniki/<tryb>/gesty):
  BAZA_PCA32.npz           średnia i 32 składowe PCA dopasowane na latentach sond w kroku 0 (stała baza)
  SONDY.json               które klatki są sondami (indeksy wierszy zbioru), które mają pełne 768D, harmonogram
  pca32_NNN.npz            kroki (n,), pca32 (n, S, 600, 32) f16: rzut latentu każdego tokenu sond,
                           normy (n, S, 600, 2) f16: ||z_t − z_0||, ||z_t − z_poprz|| w pełnych 768D
  pelne768_NNN.npz         kroki (n,), latent (n, F, 600, 768) f16 dla F pierwszych sond (pełny wektor)
  PACZKI.npz               co krok: krok, idx paczki (32), strata przed i po aktualizacji (eval), heat, box,
                           norma aktualizacji wag ||θ_t+1 − θ_t||, norma gradientu po obcięciu, lr, czas kroku
Harmonogram (domyślnie): pca32 i normy co 1 krok do `gesto_do`, potem co `co_pozniej`; pełne 768D co 1 do
`pelne_do`, potem co `pelne_co`. Stan kroku 0 zapisany w start(). Latent = wyjście GELU w MLP ostatniego bloku,
600 tokenów × 768 (jak w pomiarze naturalnym aplikacji).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import model as M


class PodsluchX:
    """Opakowanie tablicy obrazów: zapamiętuje indeksy ostatniej paczki, zwraca dane bez zmian."""

    def __init__(self, X):
        self.X = X
        self.ostatnie = None
        self.shape, self.dtype = X.shape, X.dtype

    def __getitem__(self, idx):
        self.ostatnie = np.array(idx)
        return self.X[idx]

    def __len__(self):
        return len(self.X)


class Obserwator:
    def __init__(self, out, zb, sondy, pelne, gesto_do=2188, co_pozniej=10, pelne_do=1094, pelne_co=50,
                 porcja=500, porcja_pelne=100, wolne_gb=None, min_gb=6.0, device='cuda'):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.zb, self.sondy, self.pelne = zb, [int(i) for i in sondy], [int(i) for i in pelne]
        assert all(i in self.sondy for i in self.pelne) and self.pelne == self.sondy[:len(self.pelne)]
        self.gesto_do, self.co_pozniej, self.pelne_do, self.pelne_co = gesto_do, co_pozniej, pelne_do, pelne_co
        self.porcja, self.porcja_pelne = porcja, porcja_pelne
        self.wolne_gb, self.min_gb = wolne_gb, min_gb
        self.dev = device
        self.X = torch.from_numpy(np.ascontiguousarray(zb.X[self.sondy])).to(device).permute(0, 3, 1, 2).float() / 255.
        self.podsluch = PodsluchX(zb.X); zb.X = self.podsluch
        self.z0 = self.zprev = None
        self.mean = self.comp = None
        self.buf = dict(kroki=[], pca32=[], normy=[]); self.buf_p = dict(kroki=[], latent=[])
        self.n_porcji = self.n_porcji_p = 0
        self.paczki = dict(krok=[], idx=[], strata_przed=[], strata_po=[], heat=[], box=[], norma_aktualizacji=[],
                           norma_grad=[], lr=[], czas=[])
        self._theta = self._paczka = None
        self._t = time.time()
        self.wylaczony = False
        self.zapisane = 0

    # ------------------------------------------------------------ latenty sond
    @torch.no_grad()
    def _latent(self, net):
        net.eval(); net.zapisuj_latent(True)
        out = []
        for s in range(0, len(self.X), 50):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, lat = net(self.X[s:s + 50])
            out.append(lat.float())
        net.zapisuj_latent(False)
        return torch.cat(out)                                   # (S, 600, 768) float32

    def start(self, u):
        """Krok 0: baza PCA32 z latentów sond i pierwszy zapis."""
        z = self._latent(u.net)
        Z = z.reshape(-1, z.shape[-1])
        self.mean = Z.mean(0)
        C = (Z - self.mean).T @ (Z - self.mean) / len(Z)
        w, V = torch.linalg.eigh(C)
        w, V = w.flip(0), V.flip(1)
        self.comp = V[:, :32].T.contiguous()                    # (32, 768)
        np.savez(self.out / 'BAZA_PCA32.npz', mean=self.mean.cpu().numpy(), comp=self.comp.cpu().numpy(),
                 wariancja=w.cpu().numpy(), udzial32=float(w[:32].sum() / w.sum()))
        json.dump(dict(sondy=self.sondy, pelne=self.pelne, gesto_do=self.gesto_do, co_pozniej=self.co_pozniej,
                       pelne_do=self.pelne_do, pelne_co=self.pelne_co, porcja=self.porcja, porcja_pelne=self.porcja_pelne,
                       latent='GELU MLP ostatniego bloku, 600 × 768', udzial_wariancji_pca32=float(w[:32].sum() / w.sum())),
                  open(self.out / 'SONDY.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        self.z0 = z.clone(); self.zprev = z
        self._zapisz_stan(0, z, pelne=True)
        u.net.train()
        print(f'obserwator: {len(self.sondy)} sond, PCA32 trzyma {float(w[:32].sum() / w.sum()):.3f} wariancji kroku 0', flush=True)

    def _czy_pca(self, k):
        return k <= self.gesto_do or k % self.co_pozniej == 0

    def _czy_pelne(self, k):
        return k <= self.pelne_do or k % self.pelne_co == 0

    def _zapisz_stan(self, k, z, pelne):
        d0 = (z - self.z0).norm(dim=2); dp = (z - self.zprev).norm(dim=2)
        p = (z - self.mean) @ self.comp.T                       # (S, 600, 32)
        self.buf['kroki'].append(k)
        self.buf['pca32'].append(p.half().cpu().numpy())
        self.buf['normy'].append(torch.stack([d0, dp], 2).half().cpu().numpy())
        if pelne:
            self.buf_p['kroki'].append(k); self.buf_p['latent'].append(z[:len(self.pelne)].half().cpu().numpy())
        self.zapisane += 1
        if len(self.buf['kroki']) >= self.porcja:
            self._zrzuc()
        if len(self.buf_p['kroki']) >= self.porcja_pelne:
            self._zrzuc_pelne()

    def _miejsce(self):
        if self.wolne_gb is None:
            return True
        w = self.wolne_gb()
        if w < self.min_gb:
            print(f'obserwator: wolna kwota {w:.1f} GB < {self.min_gb} GB - przerywam zapis gęsty (uczenie idzie dalej)', flush=True)
            self.wylaczony = True
            return False
        return True

    def _zrzuc(self):
        if not self.buf['kroki'] or not self._miejsce():
            return
        np.savez(self.out / f'pca32_{self.n_porcji:03d}.npz', kroki=np.array(self.buf['kroki'], np.int32),
                 pca32=np.stack(self.buf['pca32']), normy=np.stack(self.buf['normy']))
        self.n_porcji += 1; self.buf = dict(kroki=[], pca32=[], normy=[])
        self._zapisz_paczki()

    def _zrzuc_pelne(self):
        if not self.buf_p['kroki'] or not self._miejsce():
            return
        np.savez(self.out / f'pelne768_{self.n_porcji_p:03d}.npz', kroki=np.array(self.buf_p['kroki'], np.int32),
                 latent=np.stack(self.buf_p['latent']))
        self.n_porcji_p += 1; self.buf_p = dict(kroki=[], latent=[])

    def _zapisz_paczki(self):
        p = self.paczki
        np.savez(self.out / 'PACZKI.npz', krok=np.array(p['krok'], np.int32), idx=np.array(p['idx'], np.int32),
                 strata_przed=np.array(p['strata_przed'], np.float32), strata_po=np.array(p['strata_po'], np.float32),
                 heat=np.array(p['heat'], np.float32), box=np.array(p['box'], np.float32),
                 norma_aktualizacji=np.array(p['norma_aktualizacji'], np.float32),
                 norma_grad=np.array(p['norma_grad'], np.float32), lr=np.array(p['lr'], np.float32),
                 czas=np.array(p['czas'], np.float32))

    # ------------------------------------------------------------ API trening.Uczen.kroki
    def czy(self, j):
        return True

    @torch.no_grad()
    def przed(self, u, x, heat, ltrb, poz):
        self._theta = torch.cat([p.detach().flatten() for p in u.net.parameters()]).clone()
        self._paczka = (x, heat, ltrb, poz)
        self._strata_przed = self._strata(u.net, x, heat, ltrb, poz)
        self._t = time.time()

    @torch.no_grad()
    def _strata(self, net, x, heat, ltrb, poz):
        net.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        L, cz = M.strata(o, heat, ltrb, poz)
        return float(L), float(cz['heat']), float(cz['box'])

    @torch.no_grad()
    def po(self, u):
        k = u.krok
        theta = torch.cat([p.detach().flatten() for p in u.net.parameters()])
        grad = torch.cat([p.grad.flatten() for p in u.net.parameters() if p.grad is not None])
        x, heat, ltrb, poz = self._paczka
        L, lh, lb = self._strata(u.net, x, heat, ltrb, poz)
        p = self.paczki
        p['krok'].append(k); p['idx'].append(self.podsluch.ostatnie.astype(np.int32))
        p['strata_przed'].append(self._strata_przed[0]); p['strata_po'].append(L); p['heat'].append(lh); p['box'].append(lb)
        p['norma_aktualizacji'].append(float((theta - self._theta).norm())); p['norma_grad'].append(float(grad.norm()))
        p['lr'].append(float(u.opt.param_groups[0]['lr'])); p['czas'].append(time.time() - self._t)
        if not self.wylaczony and self._czy_pca(k):
            z = self._latent(u.net)
            self._zapisz_stan(k, z, pelne=self._czy_pelne(k))
            self.zprev = z
        elif not self.wylaczony and self._czy_pelne(k):
            z = self._latent(u.net)
            self.buf_p['kroki'].append(k); self.buf_p['latent'].append(z[:len(self.pelne)].half().cpu().numpy())
            if len(self.buf_p['kroki']) >= self.porcja_pelne:
                self._zrzuc_pelne()
        u.net.train()

    def zakoncz(self):
        self._zrzuc(); self._zrzuc_pelne(); self._zapisz_paczki()
        self.zb.X = self.podsluch.X
        return dict(zapisane_stany=self.zapisane, porcje_pca32=self.n_porcji, porcje_pelne=self.n_porcji_p,
                    wylaczony=self.wylaczony)
