"""Mapy atencji i przewidziana ramka dla zdjęcia TRAIN 0 we wszystkich 806 checkpointach biegu 35 000 (do filmu 13).
Atencja jak w aplikacji (app/atencja_zbioru.py): blok 6, średnia z 3 głów, bez przekątnej (uwaga tokenu na siebie wyzerowana,
wiersze znormowane), mapa = ile uwagi patch DOSTAJE od wszystkich 600 tokenów (średnia kolumny). Ramka: dekoduj() z model.py, najlepszy szczyt;
IoU z ramką prawdziwą (bbox_xyxy_480 z manifestu pomiaru). Zapis: atencja_train0.npz (kroki, atencja (806,600) surowa,
ramka (806,4), pewnosc, iou, gt). Uruchomienie: F:/Python/3.13/python.exe -X utf8 atencja_train0_prep.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod'))
from model import MalyViT, dekoduj  # noqa: E402

WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
GT = np.array([196.6666717529297, 164.6666717529297, 349.3333435058594, 184.6666717529297])
dev = 'cuda'
kroki = np.load(HERE / 'szescian.npz')['kroki']
im = cv2.cvtColor(cv2.imread(str(HERE / 'zdjecie_train0.jpg')), cv2.COLOR_BGR2RGB)
x = torch.as_tensor(cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().div(255).to(dev)
net = MalyViT().to(dev).eval()
pam = {}
net.bloki[5].attn.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0])); h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / max(u, 1e-9)


at_, ra, pw, io = [], [], [], []
for k in kroki:
    ck = torch.load(WAGI / f'k{int(k):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(ck['net'], strict=True)
    with torch.no_grad():
        o, _ = net(x)
        at, xin = net.bloki[5].attn, pam['x']
        B, N, C = xin.shape
        qkv = at.qkv(xin).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
        A = ((qkv[0] * at.scale) @ qkv[1].transpose(-2, -1)).softmax(-1).float().mean(1)[0]
        A.fill_diagonal_(0); A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        bx, s = dekoduj(o, k=1)
    at_.append(A.mean(0).cpu().numpy()); b = bx[0, 0].cpu().numpy(); ra.append(b); pw.append(float(s[0, 0])); io.append(iou(b, GT))
    if len(at_) % 100 == 0:
        print(len(at_), flush=True)
io = np.array(io)
ref = np.load(HERE / 'szescian.npz')['iou_0']
print(f'IoU: korelacja z pomiarem Edenu {np.corrcoef(io, ref)[0, 1]:.4f}, średnia różnica {np.abs(io - ref).mean():.4f}')
np.savez_compressed(HERE / 'atencja_train0.npz', kroki=kroki, atencja=np.array(at_, np.float32), ramka=np.array(ra, np.float32),
                    pewnosc=np.array(pw, np.float32), iou=io.astype(np.float32), gt=GT)
print('zapisano atencja_train0.npz')
