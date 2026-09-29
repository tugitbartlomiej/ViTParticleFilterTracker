#!/usr/bin/env python3
"""Wycinki narzędzia z modelu dyfuzyjnego prowadzone siłą tożsamości małego ViT (FreeDoM, Yu i in. 2023; PLAN.md).

Rysowanie od czystego szumu, DDIM. Na każdym kroku: przewidziany czysty obraz x̂0 (RGB + maska) → mały ViT → siła
tożsamości na tokenach narzędzia wskazanych przez maskę x̂0 → strata ((s − s_cel)/skala)² → gradient względem x_t
przez UNet i ViT → poprawka przewidzianego szumu. Każdy start z innego szumu daje inne ułożenie narzędzia.

Szczegóły i różnice względem pracy FreeDoM (nazwane wprost w PLAN.md):
  - poprawka w przestrzeni szumu: ε' = ε + waga · ĝ · ‖ε‖ / ‖ĝ‖ (normowanie jak Shirahama 2025), a nie krok ρ_t w x;
  - podróż w czasie (ponowne zaszumienie i powtórzenie kroku) tylko w fazie semantycznej, domyślnie t/T ∈ [0,5; 0,8]
    (jak FreeDoM), liczba powtórzeń --powtorzen;
  - siła liczona na --odporna zaburzonych kopiach obrazu (jasność, kontrast, szum, rozmycie; jak `sila._sila_odporna`),
    żeby prowadzenie nie szło w niewidoczny wzorek pikseli (mały ViT jest nieodporny); 0 = bez zaburzeń.
Cel: --cel (siła bezwzględna) albo --kotwica + --zmiana (siła sondy-kotwicy + przewidziana zmiana z Koopmana/Langevina).

    python generuj.py --dyfuzja model_ema.pt --vit k02.pt --wektory WEKTORY.npz --klucz s47_nowy \
        --kotwica sonda.png --kotwica-maska maska.png --zmiana 2.0 -n 8 --out ../wyniki/gen_test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import model_dyfuzji as MD
import vit_model as VM

S, SIATKA = 176, 11


# ---------------------------------------------------------------- mały ViT i siła tożsamości
def laduj_vit(sciezka, dev='cuda'):
    net = VM.MalyViT()
    p = torch.load(sciezka, map_location='cpu', weights_only=False)
    net.load_state_dict(p['net'] if 'net' in p else p)
    net = net.to(dev).eval().requires_grad_(False)
    net.zapisuj_latent(True)
    return net


def wektor(npz, klucz, kierunek=0, dev='cuda'):
    """Wektor tożsamości z WEKTORY.npz (klucz np. s47_nowy) albo KIERUNKI_CZASTEK.npz (klucz s47, wiersz kierunek)."""
    d = np.load(npz)
    w = d[f'{klucz}_w']
    w = w[kierunek] if w.ndim == 2 else w
    return {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in
            (('mu', d[f'{klucz}_mu']), ('sd', d[f'{klucz}_sd']), ('w', w / np.linalg.norm(w)))}


def tokeny(alfa):
    """Wagi tokenów narzędzia (B,121) z maski w [0,1] (jak `sila.tokeny`: udział maski ≥ 0,25). Gdy maska pusta:
    udziały jako wagi miękkie, a gdy i one zerowe — cały obraz (flaga w raporcie)."""
    frac = F.adaptive_avg_pool2d((alfa >= .5).float(), (SIATKA, SIATKA)).flatten(1)
    tok = (frac >= .25).float()
    pusty = tok.sum(1) == 0
    tok = torch.where(pusty[:, None], frac, tok)
    pusty2 = tok.sum(1) == 0
    return torch.where(pusty2[:, None], torch.ones_like(tok), tok), pusty2


def latent_nar(net, rgb, tok):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, lat = net(rgb)
    lat = lat.float()
    return (lat * tok[..., None]).sum(1) / tok.sum(1, keepdim=True).clamp(min=1e-3)


def zaburz(rgb, g, K):
    """K zaburzonych kopii (jak `sila.zaburz`): jasność/kontrast ±15 %, szum σ 0,02, rozmycie 3×3."""
    x = rgb.repeat(K, 1, 1, 1)
    a = torch.empty(x.shape[0], 2, device=x.device).uniform_(.85, 1.15, generator=g)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * a[:, :1, None, None] + m * a[:, 1:, None, None]
    x = x + .02 * torch.randn(x.shape, device=x.device, generator=g)
    k = torch.tensor([1., 2., 1.], device=x.device); k = (k[:, None] * k[None]) / 16
    x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k.expand(3, 1, 3, 3), groups=3)
    return x.clamp(0, 1)


def sila(net, rgb, tok, wt, K=0, g=None):
    """Siła tożsamości ((latent narzędzia − mu)/sd)·w; K > 0: latent uśredniony po K zaburzonych kopiach."""
    if K:
        B = rgb.shape[0]
        lat = latent_nar(net, zaburz(rgb, g, K), tok.repeat(K, 1)).view(K, B, -1).mean(0)
    else:
        lat = latent_nar(net, rgb, tok)
    return ((lat - wt['mu']) / wt['sd']) @ wt['w']


# ---------------------------------------------------------------- próbkowanie prowadzone
def _krok(model, harm, net, wt, s_cel, x, tt, ttp, prow, a, gk, gz):
    x = x.detach()
    s_val, pusty = None, None
    if prow:
        x = x.requires_grad_(True)
        with torch.enable_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = model(x if a.jakobian else x.detach(), tt).float()
            x0 = harm.x0(x, tt, e).clamp(-1, 1)
            tok, pusty = tokeny(((x0[:, 3:4] + 1) / 2).detach())
            s = sila(net, (x0[:, :3] + 1) / 2, tok, wt, a.odporna, gk)
            L = ((s - s_cel) / a.skala) ** 2
            gr, = torch.autograd.grad(L.sum(), x)
        e = e.detach()
        ne, ng = e.flatten(1).norm(dim=1).view(-1, 1, 1, 1), gr.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        e = e + a.waga * gr / (ng + 1e-12) * ne
        s_val = s.detach()
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            e = model(x, tt).float()
    with torch.no_grad():
        x = x.detach()
        x0 = harm.x0(x, tt, e).clamp(-1, 1)
        e2 = harm.eps(x, tt, x0)
        return harm.krok_ddim(x, tt, ttp, e2, x0, a.eta, gz), s_val, pusty


def freedom(model, harm, net, wt, s_cel, a, seed, dev='cuda'):
    """n = len(s_cel) obrazów od szumu; zwraca x w [-1,1] i przebieg siły w czasie."""
    n = len(s_cel)
    g = torch.Generator(device=dev).manual_seed(seed)
    gz = torch.Generator(device=dev).manual_seed(seed + 7919)
    gk = torch.Generator(device=dev).manual_seed(seed + 104729)
    x = torch.randn((n, 4, S, S), device=dev, generator=g)
    ts = MD.siatka_krokow(harm.T, a.kroki)
    przebieg = []
    for i, t in enumerate(ts):
        tp = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((n,), t, device=dev, dtype=torch.long); ttp = torch.full_like(tt, tp)
        f = t / harm.T
        r = a.powtorzen if a.podroz_do <= f <= a.podroz_od else 1
        prow = a.waga > 0 and a.prowadz_do <= f <= a.prowadz_od
        for j in range(r):
            x_prev, s, pusty = _krok(model, harm, net, wt, s_cel, x, tt, ttp, prow, a, gk, gz)
            if j < r - 1:
                x = harm.cofnij(x_prev, ttp, tt, gz)
        x = x_prev
        if s is not None:
            przebieg.append(dict(t=t, sila=[round(float(v), 3) for v in s], pusta_maska=int(pusty.sum())))
    return x, przebieg


# ---------------------------------------------------------------- kotwica, zapis
def wczytaj_kotwice(png, maska_png, dev='cuda'):
    """Sonda 256×256 → 176×176 i tokeny narzędzia jak `dane.sondy`."""
    im = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
    mk = cv2.resize((cv2.imread(str(maska_png), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8), (S, S),
                    interpolation=cv2.INTER_NEAREST) > 0
    tok = torch.from_numpy(mk.reshape(SIATKA, 16, SIATKA, 16).mean((1, 3)).reshape(1, -1) >= .25).float().to(dev)
    rgb = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev)
    return rgb, tok


@torch.no_grad()
def ocen(net, x, wt, dev='cuda'):
    """Siła zwykła i odporna (4 kopie, stałe zaburzenia) gotowego obrazu x w [-1,1]."""
    x01 = (x.clamp(-1, 1) + 1) / 2
    tok, pusty = tokeny(x01[:, 3:4])
    g = torch.Generator(device=dev).manual_seed(123)
    return sila(net, x01[:, :3], tok, wt), sila(net, x01[:, :3], tok, wt, 4, g), tok.sum(1), pusty


def zapisz(x, out, prefiks=''):
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    sciezki = []
    for i, o in enumerate(x):
        p = Path(out) / f'{prefiks}{i:03d}'
        Image.fromarray(o[..., :3]).save(f'{p}_obraz.png')                     # zapytanie do wyszukiwania (RGB)
        Image.fromarray(o[..., 3]).save(f'{p}_maska.png')
        sciezki.append(f'{p}_obraz.png')
    return sciezki


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dyfuzja', required=True, help='model_ema.pt / ckpt_XXXXXX.pt / ostatni.pt')
    ap.add_argument('--vit', required=True, help='wagi małego ViT (np. wspolny/k02.pt)')
    ap.add_argument('--wektory', required=True, help='WEKTORY.npz albo KIERUNKI_CZASTEK.npz')
    ap.add_argument('--klucz', required=True, help='np. s47_nowy (WEKTORY) albo s47 (KIERUNKI_CZASTEK)')
    ap.add_argument('--kierunek', type=int, default=0)
    ap.add_argument('--cel', type=float, default=None, help='siła docelowa bezwzględna')
    ap.add_argument('--kotwica', default=None); ap.add_argument('--kotwica-maska', default=None)
    ap.add_argument('--zmiana', type=float, default=0.0, help='cel = siła kotwicy + zmiana')
    ap.add_argument('-n', type=int, default=8)
    ap.add_argument('--paczka', type=int, default=4, help='obrazów naraz (pamięć karty)')
    ap.add_argument('--kroki', type=int, default=50)
    ap.add_argument('--waga', type=float, default=0.3, help='siła prowadzenia; 0 = bez prowadzenia')
    ap.add_argument('--skala', type=float, default=1.0)
    ap.add_argument('--powtorzen', type=int, default=3)
    ap.add_argument('--podroz-od', type=float, default=0.8); ap.add_argument('--podroz-do', type=float, default=0.5)
    ap.add_argument('--prowadz-od', type=float, default=1.0); ap.add_argument('--prowadz-do', type=float, default=0.0)
    ap.add_argument('--odporna', type=int, default=4, help='zaburzonych kopii przy liczeniu siły (0 = bez)')
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--bez-jakobianu', dest='jakobian', action='store_false',
                    help='gradient tylko przez x̂0 (szybciej, bez przejścia wstecz przez UNet)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, harm, meta = MD.laduj(a.dyfuzja, dev)
    net = laduj_vit(a.vit, dev)
    wt = wektor(a.wektory, a.klucz, a.kierunek, dev)
    rap = dict(argumenty=vars(a), krok_dyfuzji=meta.get('krok'))
    if a.kotwica:
        rgb, tok = wczytaj_kotwice(a.kotwica, a.kotwica_maska, dev)
        with torch.no_grad():
            s0 = float(sila(net, rgb, tok, wt)); s0r = float(sila(net, rgb, tok, wt, 4, torch.Generator(device=dev).manual_seed(123)))
        cel = s0 + a.zmiana
        rap.update(sila_kotwicy=s0, sila_kotwicy_odporna=s0r, tokenow_kotwicy=int(tok.sum()))
    elif a.cel is not None:
        cel = a.cel
    else:
        raise SystemExit('podaj --cel albo --kotwica z --kotwica-maska')
    rap['cel'] = cel
    obrazy, t0 = [], time.time()
    for p0 in range(0, a.n, a.paczka):
        m = min(a.paczka, a.n - p0)
        s_cel = torch.full((m,), cel, device=dev)
        x, przebieg = freedom(model, harm, net, wt, s_cel, a, a.seed + p0, dev)
        s1, s1r, ntok, pusty = ocen(net, x, wt, dev)
        sc = zapisz(x, out, prefiks=f'{p0:03d}_')
        for i in range(m):
            obrazy.append(dict(obraz=sc[i], sila=float(s1[i]), sila_odporna=float(s1r[i]), tokenow=float(ntok[i]),
                               pusta_maska=bool(pusty[i]), blad_do_celu=float(s1[i] - cel),
                               przebieg=[dict(t=q['t'], sila=q['sila'][i]) for q in przebieg[::max(1, len(przebieg) // 10)]]))
    rap.update(obrazy=obrazy, sekund_na_obraz=(time.time() - t0) / a.n,
               sila_srednia=float(np.mean([o['sila'] for o in obrazy])),
               sila_odporna_srednia=float(np.mean([o['sila_odporna'] for o in obrazy])))
    (out / 'RAPORT.json').write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: rap[k] for k in ('cel', 'sila_srednia', 'sila_odporna_srednia', 'sekund_na_obraz')}
                     | ({'sila_kotwicy': rap['sila_kotwicy']} if 'sila_kotwicy' in rap else {}), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
