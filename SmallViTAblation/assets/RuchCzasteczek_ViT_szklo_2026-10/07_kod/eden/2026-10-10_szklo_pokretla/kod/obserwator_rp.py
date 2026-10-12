"""Obserwator z losowym rzutem Gaussa (JL) zamiast PCA32, zapisem f16 albo f32 i opcjonalnym skanem po warstwach.

Rozszerza obserwator_gesty.Obserwator (bieg 35 000). Różnice:
  * rzut 768 -> rzut_dim losową macierzą Gaussa (ziarno 0): zachowuje odległości i energię ruchu (PCA32 z kroku 0 widzi
    ~12 % energii ruchu i połowę odstępu sąsiadów); zapis pod tym samym kluczem 'pca32' (kształt ..., rzut_dim);
  * zapis='f32': rzuty i normy w float32 (bez utraty małych przyrostów późnego treningu; 2× miejsca);
  * warstwy=True: dodatkowo w każdym rekordzie rzut (tą samą macierzą) aktywacji GELU MLP KAŻDEGO bloku dla pierwszych
    warstwy_n sond, do podkatalogów gesty/w0 … gesty/w{L-1} w tym samym formacie (pca32_NNN.npz: kroki, pca32, normy;
    SONDY.json; BAZA_PCA32.npz), więc pokretla_analiza.py działa na nich bez zmian (ostatni argument = podkatalog).
Sieć musi mieć listę bloków `net.bloki`, w każdym `mlp.act` (timm Block albo model_cnn.BlokConv) i metodę zapisuj_latent.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import obserwator_gesty as OG


class ObserwatorRP(OG.Obserwator):
    def __init__(self, *a, rzut_dim=128, zapis='f16', warstwy=False, warstwy_n=6, tokenow=600, ziarno_rzutu=0, **k):
        super().__init__(*a, **k)
        self.rzut_dim, self.zapis, self.warstwy, self.tokenow, self.ziarno_rzutu = int(rzut_dim), zapis, bool(warstwy), int(tokenow), int(ziarno_rzutu)
        self.warstwy_n = min(int(warstwy_n), len(self.sondy))
        self._net = None
        self.z0_w = self.zprev_w = None
        self.buf_w = None
        self.n_porcji_w = 0

    # ------------------------------------------------------------ pomocnicze
    def _konw(self, t):
        return (t.float() if self.zapis == 'f32' else t.half()).cpu().numpy()

    def _opis_sond(self, extra):
        d = dict(sondy=self.sondy, pelne=self.pelne, gesto_do=self.gesto_do, co_pozniej=self.co_pozniej, pelne_do=self.pelne_do,
                 pelne_co=self.pelne_co, porcja=self.porcja, porcja_pelne=self.porcja_pelne, rzut=f'rp{self.rzut_dim}',
                 ziarno_rzutu=self.ziarno_rzutu, zapis=self.zapis, tokenow=self.tokenow,
                 latent='GELU MLP ostatniego bloku, tokenow × 768', uwaga='klucz pca32 w porcjach zawiera losowy rzut, nie PCA')
        d.update(extra)                                   # extra może nadpisać sondy/pelne/latent (podkatalogi warstw)
        return d

    @torch.no_grad()
    def _latenty_warstw(self, net):
        """Aktywacje GELU MLP każdego bloku dla pierwszych warstwy_n sond: lista L tensorów (S_w, tokenow, 768) float32."""
        zebrane = [[] for _ in net.bloki]
        haki = [b.mlp.act.register_forward_hook(lambda m, i, o, j=j: zebrane[j].append(o.float())) for j, b in enumerate(net.bloki)]
        net.eval()
        try:
            X = self.X[:self.warstwy_n]
            for s in range(0, len(X), 50):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    net(X[s:s + 50])
        finally:
            for h in haki:
                h.remove()
        return [torch.cat(z) for z in zebrane]

    # ------------------------------------------------------------ start i zapis
    def start(self, u):
        self._net = u.net
        z = self._latent(u.net)
        g = torch.Generator().manual_seed(self.ziarno_rzutu)
        R = (torch.randn(z.shape[-1], self.rzut_dim, generator=g) / self.rzut_dim ** 0.5).to(self.dev)
        self.mean = torch.zeros(z.shape[-1], device=self.dev); self.comp = R.T.contiguous()
        baza = dict(mean=self.mean.cpu().numpy(), comp=self.comp.cpu().numpy(), wariancja=np.zeros(1), udzial32=float('nan'), rzut=f'rp{self.rzut_dim}')
        np.savez(self.out / 'BAZA_PCA32.npz', **baza)
        json.dump(self._opis_sond(dict(warstwy=self.warstwy, warstwy_n=(self.warstwy_n if self.warstwy else 0))),
                  open(self.out / 'SONDY.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        self.z0 = z.clone(); self.zprev = z
        if self.warstwy:
            ws = self._latenty_warstw(u.net)
            self.z0_w = [w.clone() for w in ws]; self.zprev_w = ws
            self.buf_w = dict(kroki=[], pca32=[[] for _ in ws], normy=[[] for _ in ws])
            for j in range(len(ws)):
                d = self.out / f'w{j}'; d.mkdir(exist_ok=True)
                np.savez(d / 'BAZA_PCA32.npz', **baza)
                json.dump(self._opis_sond(dict(warstwa=j, warstw=len(ws), sondy=self.sondy[:self.warstwy_n], pelne=[],
                                               latent=f'GELU MLP bloku {j} z {len(ws)}, tokenow × 768')),
                          open(d / 'SONDY.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        self._zapisz_stan(u.krok, z, pelne=True)
        u.net.train()
        print(f'obserwator: {len(self.sondy)} sond, losowy rzut {z.shape[-1]} -> {self.rzut_dim} (JL, ziarno {self.ziarno_rzutu}), zapis {self.zapis}, '
              f'tokenów {z.shape[1]}, krok {u.krok}' + (f', skan {len(self.z0_w)} warstw na {self.warstwy_n} sondach' if self.warstwy else ''), flush=True)

    def _zapisz_stan(self, k, z, pelne):
        d0 = (z - self.z0).norm(dim=2); dp = (z - self.zprev).norm(dim=2)
        p = (z - self.mean) @ self.comp.T
        self.buf['kroki'].append(k)
        self.buf['pca32'].append(self._konw(p))
        self.buf['normy'].append(self._konw(torch.stack([d0, dp], 2)))
        if pelne:
            self.buf_p['kroki'].append(k); self.buf_p['latent'].append(z[:len(self.pelne)].half().cpu().numpy())
        self.zapisane += 1
        if self.warstwy:
            ws = self._latenty_warstw(self._net)
            self.buf_w['kroki'].append(k)
            for j, w in enumerate(ws):
                d0w = (w - self.z0_w[j]).norm(dim=2); dpw = (w - self.zprev_w[j]).norm(dim=2)
                self.buf_w['pca32'][j].append(self._konw((w - self.mean) @ self.comp.T))
                self.buf_w['normy'][j].append(self._konw(torch.stack([d0w, dpw], 2)))
            self.zprev_w = ws
            if len(self.buf_w['kroki']) >= self.porcja:
                self._zrzuc_warstwy()
        if len(self.buf['kroki']) >= self.porcja:
            self._zrzuc()
        if len(self.buf_p['kroki']) >= self.porcja_pelne:
            self._zrzuc_pelne()

    def _zrzuc_warstwy(self):
        if not self.warstwy or not self.buf_w['kroki'] or not self._miejsce():
            return
        for j in range(len(self.buf_w['pca32'])):
            np.savez(self.out / f'w{j}' / f'pca32_{self.n_porcji_w:03d}.npz', kroki=np.array(self.buf_w['kroki'], np.int32),
                     pca32=np.stack(self.buf_w['pca32'][j]), normy=np.stack(self.buf_w['normy'][j]))
        self.n_porcji_w += 1
        self.buf_w = dict(kroki=[], pca32=[[] for _ in self.buf_w['pca32']], normy=[[] for _ in self.buf_w['normy']])

    def zakoncz(self):
        self._zrzuc_warstwy()
        w = super().zakoncz()
        w.update(rzut=f'rp{self.rzut_dim}', zapis=self.zapis, porcje_warstw=self.n_porcji_w)
        return w
