"""Które części sieci odpowiadają za IoU (lokalizacja), a które za pewność (rozpoznanie narzędzia)? Eden, GPU, checkpointy biegu 35 000.
Głowa ma osobne wyjścia: kanał 0 = mapa obecności (pewność), kanały 1–4 = odległości do krawędzi ramki (IoU).
Dla 100 zdjęć eksportu ruch60 (50 TRAIN + 50 VALID) i grup parametrów (łodyga, blok i: uwaga+norm1, blok i: MLP+norm2, norm końcowa, głowa):
  (a) PODATNOŚĆ („lokalna temperatura”): szum gaussowski o odchyleniu 2 % RMS wag grupy, 6 losowań; średnia zmiana pewności szczytu
      i IoU jego ramki (ze znakiem i bezwzględna), osobno TRAIN i VALID;
  (b) ROZKŁAD GRADIENTU: gradient straty mapy (ogniskowa) i straty ramki (1 − GIoU) osobno; normy na grupę (udział każdej grupy w każdej
      stracie) i kosinus między nimi (czy lokalizacja i rozpoznanie ciągną wagi grupy w tę samą stronę).
Uruchomienie (katalog kodu biegu 35 000): python podatnosc.py --wagi ... --eksport ... --zipy ... --manifest ... --out ..."""
import argparse
import json
import re
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dane as D  # noqa: E402
import model as M  # noqa: E402
import trening as T  # noqa: E402
from trening35000 import wczytaj_z_zipow, wiersze_35000  # noqa: E402

ap = argparse.ArgumentParser()
for k in ('wagi', 'eksport', 'zipy', 'manifest', 'out'):
    ap.add_argument('--' + k, required=True)
ap.add_argument('--kroki', default='1094,5470,21880,65640')
ap.add_argument('--sigma', type=float, default=0.02)
ap.add_argument('--losowan', type=int, default=6)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
log = lambda *s: print(*s, f'[{time.time() - t0:.0f} s]', flush=True)

META = json.load(open(Path(a.eksport) / 'META.json', encoding='utf-8'))
wt = wiersze_35000(a.manifest)
wal_w, wal_X = D.walidacja()
rows = [wt[z['pool_row']] if z['split'] == 'TRAIN' else wal_w[z['val_row']] for z in META['zdjecia']]
assert [r['frame_id'] for r in rows] == [z['frame_id'] for z in META['zdjecia']]
n_tr = sum(z['split'] == 'TRAIN' for z in META['zdjecia'])
X = np.concatenate([wczytaj_z_zipow(rows[:n_tr], a.zipy), wal_X[[z['val_row'] for z in META['zdjecia'][n_tr:]]]]); del wal_X
Xg = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float() / 255.
N = len(X)
ramki = [r['ramki'] for r in rows]
h, w = D.H_ // M.KROK, D.W_ // M.KROK
heat, ltrb, poz = (t.to(dev) for t in M.cele(ramki, h, w))
log(f'{N} zdjęć ({n_tr} TRAIN)')

net = M.MalyViT().to(dev).eval()
grupy = {'lodyga': lambda n: n.startswith('lodyga')}
for i in range(len(net.bloki)):
    grupy[f'blok{i}_uwaga'] = (lambda i: lambda n: n.startswith(f'bloki.{i}.attn') or n.startswith(f'bloki.{i}.norm1') or n.startswith(f'bloki.{i}.ls1'))(i)
    grupy[f'blok{i}_mlp'] = (lambda i: lambda n: n.startswith(f'bloki.{i}.mlp') or n.startswith(f'bloki.{i}.norm2') or n.startswith(f'bloki.{i}.ls2'))(i)
grupy['norm'] = lambda n: n.startswith('norm.')
grupy['glowa'] = lambda n: n.startswith('glowa')
params = dict(net.named_parameters())
przyp = {g: [n for n in params if f(n)] for g, f in grupy.items()}
assert sum(len(v) for v in przyp.values()) == len(params), set(params) - {n for v in przyp.values() for n in v}


@torch.no_grad()
def ocena():
    o = torch.cat([net(Xg[s:s + 25])[0] for s in range(0, N, 25)])
    bx, sc = M.dekoduj(o)
    bx, sc = bx.cpu().numpy(), sc.cpu().numpy()
    conf = sc[:, 0]
    iou = np.array([max((T.iou(bx[j, 0], g) for g in ramki[j]), default=0.0) for j in range(N)])
    return conf, iou


pliki = {int(m[1]): p for p in Path(a.wagi).iterdir() if (m := re.fullmatch(r'k([0-9]+)\.pt', p.name))}
wyn = {}
for krok in [int(k) for k in a.kroki.split(',')]:
    st = torch.load(pliki[krok], map_location='cpu', weights_only=False)
    net.load_state_dict(st['net'], strict=True); net.eval()
    stan0 = {n: p.detach().clone() for n, p in params.items()}
    c0, i0 = ocena()
    # (a) podatność
    pod = {}
    gen = torch.Generator(device=dev).manual_seed(0)
    for g, nazwy in przyp.items():
        dc, di = [], []
        for _ in range(a.losowan):
            with torch.no_grad():
                for n in nazwy:
                    p = params[n]; rms = stan0[n].pow(2).mean().sqrt().clamp(min=1e-8)
                    p.add_(torch.randn(p.shape, generator=gen, device=dev) * a.sigma * rms)
            c1, i1 = ocena(); dc.append(c1 - c0); di.append(i1 - i0)
            with torch.no_grad():
                for n in nazwy:
                    params[n].copy_(stan0[n])
        dc, di = np.array(dc), np.array(di)
        pod[g] = {s: dict(d_pewnosc=float(dc[:, m].mean()), d_iou=float(di[:, m].mean()), abs_pewnosc=float(np.abs(dc[:, m]).mean()), abs_iou=float(np.abs(di[:, m]).mean()))
                  for s, m in (('TRAIN', slice(0, n_tr)), ('VALID', slice(n_tr, N)), ('wszystkie', slice(0, N)))}
        pod[g]['parametrow'] = int(sum(params[n].numel() for n in nazwy))
    # (b) rozkład gradientu
    gr = {}
    for nazwa_s in ('heat', 'box'):
        net.zero_grad(set_to_none=True)
        for s in range(0, N, 25):
            o, _ = net(Xg[s:s + 25])
            _, cz = M.strata(o, heat[s:s + 25], ltrb[s:s + 25], poz[s:s + 25])
            pf = o[..., 0].float().sigmoid().clamp(1e-4, 1 - 1e-4)
            if nazwa_s == 'heat':
                hs = heat[s:s + 25]; pos = hs.eq(1).float()
                L = (-(torch.log(pf) * (1 - pf) ** 2 * pos).sum() - (torch.log(1 - pf) * pf ** 2 * (1 - hs) ** 4 * (1 - pos)).sum()) / pos.sum().clamp(min=1)
            else:
                m = poz[s:s + 25] > 0
                pr = o[..., 1:].float().exp()[m]; wgt = poz[s:s + 25][m]
                L = ((1 - M._giou(pr, ltrb[s:s + 25][m])) * wgt).sum() / wgt.sum()
            (L * 25 / N).backward()
        gr[nazwa_s] = {g: torch.cat([params[n].grad.reshape(-1) for n in nazwy if params[n].grad is not None] or [torch.zeros(1, device=dev)]).clone() for g, nazwy in przyp.items()}
    rozklad = {}
    nh = sum(float(v.pow(2).sum()) for v in gr['heat'].values()); nb = sum(float(v.pow(2).sum()) for v in gr['box'].values())
    for g in przyp:
        a_, b_ = gr['heat'][g], gr['box'][g]
        rozklad[g] = dict(udzial_grad_mapy=float(a_.pow(2).sum()) / nh, udzial_grad_ramki=float(b_.pow(2).sum()) / nb,
                          cos_mapa_ramka=float((a_ @ b_) / (a_.norm() * b_.norm() + 1e-12)))
    wyn[krok] = dict(pewnosc=float(c0.mean()), iou=float(i0.mean()), podatnosc=pod, gradient=rozklad)
    log(f'krok {krok}: pewność {c0.mean():.3f}, IoU {i0.mean():.3f}')
    for g in przyp:
        p_, r_ = pod[g]['wszystkie'], rozklad[g]
        log(f'   {g:>12s}: Δpewność {p_["d_pewnosc"]:+.4f} (|{p_["abs_pewnosc"]:.4f}|), ΔIoU {p_["d_iou"]:+.4f} (|{p_["abs_iou"]:.4f}|); '
            f'grad mapy {r_["udzial_grad_mapy"]:.3f}, grad ramki {r_["udzial_grad_ramki"]:.3f}, cos {r_["cos_mapa_ramka"]:+.2f}')
    del gr
json.dump(dict(opis=__doc__.split('\n')[0], sigma=a.sigma, losowan=a.losowan, zdjec=N, wyniki=wyn), open(OUT / 'PODATNOSC.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano', OUT / 'PODATNOSC.json')
