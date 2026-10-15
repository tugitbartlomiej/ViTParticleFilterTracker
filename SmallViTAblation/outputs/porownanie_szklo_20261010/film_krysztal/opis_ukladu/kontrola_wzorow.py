"""Kontrola: (a) wzór Jakobianu pułapki vs autograd PyTorch (float64); (b) ile zmienia obcięcie z < 0 do zera (jak w filmach 10–12)
dla sumy sił F − F̄ w 192D i w rzucie 3D; (c) położenie w filmie 12 (suma sił) a stan tokenu x."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

FK = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\porownanie_szklo_20261010\film_krysztal')
ROOT = FK.parents[2]
sys.path.insert(0, str(FK))
from lasery_wspolne import dane, oko                                                # noqa: E402

KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
LATENT = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
D = dane(0); kroki_all = D['kroki'].astype(int); narz = D['narz'].astype(bool)
ZL = np.load(LATENT, mmap_mode='r')
xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float() / 255.
net = M.MalyViT().eval(); net.zapisuj_latent(True)
zap = {}
net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0].detach()))
print('typ GELU w MLP:', net.bloki[5].mlp.act, ' eps LN2:', net.bloki[5].norm2.eps)
rng = np.random.default_rng(0)
tokeny = [316] + rng.choice(600, 7, replace=False).tolist()
for k in [0, 1094, 21880, 65640]:
    s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
    net.load_state_dict(s)
    with torch.no_grad():
        net(xin)
    X = zap['x'][0].double()
    n2 = copy.deepcopy(net.bloki[5].norm2).double(); mlp = copy.deepcopy(net.bloki[5].mlp).double()
    f = lambda x: mlp(n2(x))
    W1, b1 = mlp.fc1.weight.detach().numpy(), mlp.fc1.bias.detach().numpy()
    W2, b2 = mlp.fc2.weight.detach().numpy(), mlp.fc2.bias.detach().numpy()
    gam, bet, eps = n2.weight.detach().numpy(), n2.bias.detach().numpy(), n2.eps
    bl = []
    for j in tokeny:
        x = X[j].numpy(); n = len(x)
        u = x - x.mean(); sd = np.sqrt((u * u).mean() + eps)
        a = W1 @ (gam * u / sd + bet) + b1
        Jln = gam[:, None] * ((np.eye(n) - 1.0 / n) / sd - np.outer(u, u) / (n * sd ** 3))
        Jf = (W2 * dgelu(a)[None, :]) @ W1 @ Jln
        Ja = torch.autograd.functional.jacobian(f, X[j]).numpy()
        bl.append(np.linalg.norm(Jf - Ja) / np.linalg.norm(Ja))
    # (b) obcięcie z < 0: suma sił względem średniej po tokenach
    Xn = X.numpy(); u = Xn - Xn.mean(1, keepdims=True); sd = np.sqrt((u * u).mean(1) + eps)
    A = (gam * u / sd[:, None] + bet) @ W1.T + b1
    Z = gelu(A); Zc = np.clip(Z, 0, None)
    Fp = (Z - Z.mean(0)) @ W2.T; Fc = (Zc - Zc.mean(0)) @ W2.T
    e192 = np.linalg.norm(Fc - Fp, axis=1) / np.linalg.norm(Fp, axis=1)
    vn = W2 / np.linalg.norm(W2, axis=0)
    U_, _, _ = np.linalg.svd(vn - vn.mean(1, keepdims=True), full_matrices=False)
    P3 = U_[:, :3].T
    p, pc = Fp @ P3.T, Fc @ P3.T
    e3 = np.linalg.norm(pc - p, axis=1) / np.linalg.norm(p, axis=1)
    kor = np.corrcoef(p.ravel(), pc.ravel())[0, 1]
    ujemne = (Z < 0).mean(); udz_uj = np.linalg.norm(((Z - Zc) - (Z - Zc).mean(0)) @ W2.T, axis=1) / np.linalg.norm(Fp, axis=1)
    # (c) czy położenie w filmie (suma sił) przypomina stan tokenu: korelacja odległości między tokenami
    xs = u / sd[:, None]
    dx = np.linalg.norm(xs[:, None] - xs[None], axis=2)[np.triu_indices(600, 1)]
    dF = np.linalg.norm(Fp[:, None] - Fp[None], axis=2)[np.triu_indices(600, 1)]
    kor_odl = np.corrcoef(dx, dF)[0, 1]
    print(f'krok {k:5d}: J wzór vs autograd: maks. błąd względny {max(bl):.1e} | '
          f'z<0: {ujemne:.0%} diod, wkład ujemnych w F−F̄ (mediana) {np.median(udz_uj):.0%} | '
          f'obcięcie: błąd F−F̄ w 192D {np.median(e192):.0%}, w rzucie 3D {np.median(e3):.0%}, korelacja położeń 3D {kor:.3f} | '
          f'korelacja odległości: stany x (po LN) vs sumy sił F {kor_odl:.2f}', flush=True)
