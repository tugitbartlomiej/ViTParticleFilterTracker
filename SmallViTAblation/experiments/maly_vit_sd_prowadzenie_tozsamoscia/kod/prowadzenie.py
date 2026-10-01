"""Stable Diffusion prowadzone latentem tożsamości małego ViT (prośba użytkownika 2026-10-01: obrazy z SD mają pokrywać
co najmniej w 90 procentach przewidzianą część latentu, zwłaszcza latentu tożsamości).

SD (dyfuzja_sd_filmy) rysuje z masek narzędzia, rogówki i źrenicy; nie zna latentu ViT. Prowadzenie bez nowego uczenia,
w duchu FreeDoM (Yu i in. 2023) i „universal guidance” (Bansal i in. 2023): w każdym kroku DDIM
  1. UNet przewiduje szum ε, z niego przewidywany czysty latent obrazu x̂0;
  2. x̂0 → dekoder VAE → klatka 576×384 → wycinek wokół narzędzia jak sonda (176×176) → mały ViT → latent narzędzia z
     (średnia aktywacji MLP ostatniego bloku po tokenach narzędzia, 768 liczb);
  3. energia E = (s(z) − s*)² + ‖(z − z*)/sd‖² / 768, gdzie s = siła tożsamości (rzut na wektor tożsamości),
     z* = przewidziany latent narzędzia, s* = s(z*);
  4. x̂0 przesunięty kilka razy o krok w dół gradientu E (krok skalowany szumem kroku σ_t, więc duże zmiany
     na początku, gdy obraz jest jeszcze kształtowany, i małe na końcu), potem DDIM idzie dalej z poprawionym x̂0.
Prowadzenie zmienia x̂0, a nie wynik końcowy, więc kolejne kroki odszumiania wracają na rozmaitość obrazów, z których
uczono SD (mniej „przeciwstawnych” zakłóceń pikseli niż przy optymalizacji gotowego obrazu).

Pomiar pokrycia na końcowym obrazie uint8, tym samym wycinkiem co w pętli (`uklady.wycinek_176`, cv2 INTER_AREA).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import generator_sd as GS
import uklady as UK

S_VIT, SIATKA = 176, 11


def tokeny_narzedzia(u):
    """Maska tokenów narzędzia (121,) w wycinku 176, jak w sondach: udział maski w tokenie ≥ 0,25."""
    x0, y0, b = UK.okno_wycinka(u)
    m = UK.maska_narzedzia_sd(u)[y0:y0 + b, x0:x0 + b].astype(np.float32)
    m = F.interpolate(torch.from_numpy(m)[None, None], size=(S_VIT, S_VIT), mode='nearest')[0, 0]
    return (m.reshape(SIATKA, 16, SIATKA, 16).mean((1, 3)).reshape(-1) >= .25).float()


class Prowadzony(GS.GeneratorSD):
    """GeneratorSD z prowadzeniem latentem narzędzia małego ViT (net: MalyViT w trybie ewaluacji)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.vae.enable_gradient_checkpointing() if hasattr(self.vae, 'enable_gradient_checkpointing') else None
        self.ac = self.sched.alphas_cumprod.to(self.dev)

    def latent_vit(self, net, obraz01, u, tn):
        """obraz01: (1,3,384,576) w [0,1], z gradientem. Zwraca latent narzędzia (768,)."""
        x0, y0, b = UK.okno_wycinka(u)
        w = obraz01[:, :, y0:y0 + b, x0:x0 + b]
        w = F.interpolate(w, size=(S_VIT, S_VIT), mode='bilinear', antialias=True, align_corners=False)
        net.zapisuj_latent(True)
        net(w)
        lat = net._latent.float()[0]                                       # (121, 768)
        net.zapisuj_latent(False)
        a = tn.to(lat.device)
        return (lat * a[:, None]).sum(0) / a.sum().clamp(min=1)

    def energia(self, net, x0, u, tn, wek_t, z_cel, waga_latentu=1.0):
        img = self.vae.decode(x0.half() / GS.SKALA_VAE).sample.float().clamp(-1, 1)
        z = self.latent_vit(net, (img + 1) / 2, u, tn)
        zs = (z - wek_t['mu']) / wek_t['sd']; zc = (z_cel - wek_t['mu']) / wek_t['sd']
        s, sc = zs @ wek_t['w'], zc @ wek_t['w']
        return (s - sc) ** 2 + waga_latentu * ((zs - zc) ** 2).mean(), float(s), float(sc)

    @torch.no_grad()
    def koduj(self, obraz576):
        """Klatka uint8 (384, 576, 3) → latent VAE (1, 4, 48, 72) fp16 (średnia rozkładu, skala SD)."""
        x = torch.from_numpy(np.ascontiguousarray(obraz576)).permute(2, 0, 1)[None].to(self.dev, torch.float16) / 127.5 - 1
        return self.vae.encode(x).latent_dist.mean * GS.SKALA_VAE

    def rysuj_prowadzony(self, net, maski_u8, uklad, z_cel, wek, ziarno, kroki=30, cfg=2.0, eta=0.15, iteracje=2,
                         start=None, sila_startu=0.5, koniec_prowadzenia=5, waga_latentu=1.0, prowadz=True):
        """Jeden obraz (pamięć karty): maski_u8 (3,48,72), z_cel (768,) przewidziany latent narzędzia.
        start: prawdziwa klatka kotwicy 576×384 uint8 → odpowiednik odwrócenia FastGAN-a z wersji 3: latent VAE klatki
        zaszumiony do poziomu sila_startu (0 = sama klatka, 1 = czysty szum) i odszumiany z masek tej klatki.
        Prowadzenie od pierwszego kroku po starcie do (kroki − koniec_prowadzenia); ostatnie kroki bez prowadzenia
        czyszczą obraz. Zwraca (obraz uint8 384×576×3, przebieg energii)."""
        net.eval()
        stan_grad = [q.requires_grad for q in net.parameters()]
        for q in net.parameters():
            q.requires_grad_(False)                                   # gradient tylko względem obrazu
        try:
            return self._rysuj(net, maski_u8, uklad, z_cel, wek, ziarno, kroki, cfg, eta, iteracje, start, sila_startu,
                               koniec_prowadzenia, waga_latentu, prowadz)
        finally:
            for q, g0 in zip(net.parameters(), stan_grad):
                q.requires_grad_(g0)                                   # uczenie sieci dalej działa

    def _rysuj(self, net, maski_u8, uklad, z_cel, wek, ziarno, kroki, cfg, eta, iteracje, start, sila_startu,
               koniec_prowadzenia, waga_latentu, prowadz):
        wek_t = {k: torch.as_tensor(np.asarray(wek[k]), dtype=torch.float32, device=self.dev) for k in ('mu', 'sd', 'w')}
        z_cel = torch.as_tensor(np.asarray(z_cel), dtype=torch.float32, device=self.dev)
        tn = tokeny_narzedzia(uklad)
        m = torch.from_numpy(maski_u8[None]).to(self.dev, torch.float16) / 255.
        self.sched.set_timesteps(kroki)
        ts = self.sched.timesteps
        g = torch.Generator(device=self.dev).manual_seed(int(ziarno) * 1000)
        szum = torch.randn((1, 4, m.shape[2], m.shape[3]), generator=g, device=self.dev, dtype=torch.float16)
        if start is None:
            i0, x = 0, szum
        else:
            i0 = min(kroki - 1, int(round(kroki * (1 - sila_startu))))
            a0 = self.ac[ts[i0]].float()
            x = (a0.sqrt() * self.koduj(start).float() + (1 - a0).sqrt() * szum.float()).to(torch.float16)
        e_ = self.emb.expand(1, -1, -1)
        przebieg = []
        for i in range(i0, kroki):
            t = ts[i]
            with torch.no_grad():
                xx = torch.cat([torch.cat([x, m], 1), torch.cat([x, torch.zeros_like(m)], 1)])
                e = self.unet(xx, t, encoder_hidden_states=torch.cat([e_, e_])).sample.float()
                e = e[1:] + cfg * (e[:1] - e[1:])
            a_t = self.ac[t].float()
            xf = x.float()
            x0 = (xf - (1 - a_t).sqrt() * e) / a_t.sqrt()
            if prowadz and i < kroki - koniec_prowadzenia:
                sig = float((1 - a_t).sqrt())
                for _ in range(iteracje):
                    x0v = x0.detach().requires_grad_(True)
                    E, s, sc = self.energia(net, x0v, uklad, tn, wek_t, z_cel, waga_latentu)
                    gr, = torch.autograd.grad(E, x0v)
                    gr = gr / (gr.pow(2).mean().sqrt() + 1e-8)
                    x0 = (x0v - eta * sig * gr).detach()
                    przebieg.append(dict(krok=i, E=float(E), s=s, s_cel=sc))
                e = (xf - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            x = self.sched.step(e, t, xf).prev_sample.to(torch.float16)
        with torch.no_grad():
            img = self.vae.decode(x / GS.SKALA_VAE).sample.float().clamp(-1, 1)
        return ((img + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy(), przebieg
