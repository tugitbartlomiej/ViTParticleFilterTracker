"""Kontrola: czy jeden dyskretny krok MLP (x → x + F(x)) ściąga objętość? log|det(I+J)|/192 vs tr J/192, eps LN z modułu."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

FK = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\porownanie_szklo_20261010\film_krysztal')
ROOT = FK.parents[2]
sys.path.insert(0, str(FK))
from lasery_wspolne import oko                                                      # noqa: E402

KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float() / 255.
net = M.MalyViT().eval()
zap = {}
net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0].detach()))
for k in [0, 1094, 5470, 10940, 21880, 43760, 65640]:
    net.load_state_dict(torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net'])
    with torch.no_grad():
        net(xin)
    X = zap['x'][0].double().numpy(); n = X.shape[1]
    b = net.bloki[5]
    W1, b1 = b.mlp.fc1.weight.detach().double().numpy(), b.mlp.fc1.bias.detach().double().numpy()
    W2 = b.mlp.fc2.weight.detach().double().numpy()
    gam, bet, eps = b.norm2.weight.detach().double().numpy(), b.norm2.bias.detach().double().numpy(), b.norm2.eps
    ld, tr, sk = [], [], []
    for j in range(600):
        x = X[j]; u = x - x.mean(); sd = np.sqrt((u * u).mean() + eps)
        a = W1 @ (gam * u / sd + bet) + b1
        Jln = gam[:, None] * ((np.eye(n) - 1.0 / n) / sd - np.outer(u, u) / (n * sd ** 3))
        J = (W2 * dgelu(a)[None, :]) @ W1 @ Jln
        s, l = np.linalg.slogdet(np.eye(n) + J)
        ld.append(l / n); tr.append(np.trace(J) / n); sk.append(s)
    ld, tr, sk = map(np.array, (ld, tr, sk))
    print(f'krok {k:5d}: log|det(I+J)|/192 mediana {np.median(ld):+.4f} (p10 {np.percentile(ld, 10):+.4f}, p90 {np.percentile(ld, 90):+.4f}), '
          f'tokenów ze ściąganiem {np.mean(ld < 0):.0%}, ujemny wyznacznik {np.mean(sk < 0):.0%} | tr J/192 mediana {np.median(tr):+.4f}', flush=True)
