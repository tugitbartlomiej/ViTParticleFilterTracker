"""Mapy atencji WSZYSTKICH 6 bloków dla zdjęcia TRAIN 0 we wszystkich 806 checkpointach biegu 35 000 (mozaika w filmie 16).
Definicja jak w atencja_train0_prep.py i aplikacji: średnia z 3 głów, bez przekątnej (wiersze znormowane), mapa = ile uwagi patch
DOSTAJE (średnia kolumny). Zapis: atencja6_train0.npz (kroki, atencja (806, 6, 600)).
Uruchomienie: F:/Python/3.13/python.exe -X utf8 atencja6_train0_prep.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod'))
from model import MalyViT  # noqa: E402

WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
dev = 'cuda'
kroki = np.load(HERE / 'szescian.npz')['kroki']
im = cv2.cvtColor(cv2.imread(str(HERE / 'zdjecie_train0.jpg')), cv2.COLOR_BGR2RGB)
x = torch.as_tensor(cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().div(255).to(dev)
net = MalyViT().to(dev).eval()
pam = {}
for b in range(6):
    net.bloki[b].attn.register_forward_hook(lambda m, a, o, b=b: pam.__setitem__(b, a[0].detach()))
wynik = np.zeros((len(kroki), 6, 600), np.float32)
for i, k in enumerate(kroki):
    net.load_state_dict(torch.load(WAGI / f'k{int(k):05d}.pt', map_location=dev, weights_only=True)['net'], strict=True)
    with torch.no_grad():
        net(x)
        for b in range(6):
            at, xin = net.bloki[b].attn, pam[b]
            B, N, C = xin.shape
            qkv = at.qkv(xin).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
            A = ((qkv[0] * at.scale) @ qkv[1].transpose(-2, -1)).softmax(-1).float().mean(1)[0]
            A.fill_diagonal_(0); A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
            wynik[i, b] = A.mean(0).cpu().numpy()
    if i % 200 == 0:
        print(i, flush=True)
stary = np.load(HERE / 'atencja_train0.npz')['atencja']
print('blok 6 zgodny z atencja_train0.npz:', float(np.abs(wynik[:, 5] - stary).max()))
np.savez_compressed(HERE / 'atencja6_train0.npz', kroki=kroki, atencja=wynik)
print('zapisano atencja6_train0.npz')
