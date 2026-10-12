"""Cechy DINOv2 ViT-S/14 na obrazie SD (g148), zdjęciu celu i dwóch klatkach TRAIN znalezionych dla obrazu SD.

Obraz w całości (bez wycinania środka) skalowany do 476 x 322 (34 x 23 łatki po 14 px), normalizacja ImageNet.
Kolumny: obraz | cechy łatek: PCA-3 -> RGB, baza PCA wspólna dla wszystkich obrazów (ten sam kolor = podobna cecha) |
uwaga CLS ostatniego bloku do łatek (średnia po głowach) | kosinus każdej łatki z CLS obrazu SD (gdzie obraz
przypomina globalny opis obrazu SD, którym szukano).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/cechy_dino.py [g148]
"""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_warunku'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CEL_J = 10389
OBRAZ = sys.argv[1] if len(sys.argv) > 1 else 'g148'
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
W, H = 476, 322
GW, GH = W // 14, H // 14


def klatka(j):
    from PIL import Image
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    sz = json.loads((KAT / f'SZUKANIE_TRAIN_{OBRAZ}.json').read_text(encoding='utf-8'))['wyniki'][f'obraz SD {OBRAZ}']['top']
    a2 = max(sz, key=lambda t: t['ulamek'])                         # znaleziona klatka o największym postępie wzdłuż δ
    a1 = sz[0]                                                      # najbliższa w DINO
    obrazy = [(f'obraz SD {OBRAZ}', cv2.cvtColor(cv2.imread(str(KAT / 'obrazy' / f'{OBRAZ}.png')), cv2.COLOR_BGR2RGB)),
              ('zdjęcie celu 0055/4019', klatka(CEL_J)),
              (f'najbliższa w DINO {a1["film"][3:7]}/{a1["klatka"]} (cos {a1["cos_dino"]:.2f})', klatka(a1['j'])),
              (f'największy postęp {a2["film"][3:7]}/{a2["klatka"]} (cos {a2["cos_dino"]:.2f}, postęp {100 * a2["ulamek"]:.1f} %)', klatka(a2['j']))]

    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    attn = model.blocks[-1].attn
    pam = {}
    uchwyt = attn.qkv.register_forward_hook(lambda m, i, o: pam.update(qkv=o.detach()))
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1).cuda(); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1).cuda()

    cechy = []
    with torch.inference_mode():
        for _, im in obrazy:
            x = torch.from_numpy(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().cuda() / 255
            out = model.forward_features((x - mean) / std)
            cls = out['x_norm_clstoken'][0].float(); pt = out['x_norm_patchtokens'][0].float()      # [GH*GW, 384]
            B, N, _ = pam['qkv'].shape; nh = attn.num_heads
            qkv = pam['qkv'].reshape(B, N, 3, nh, -1).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            A = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(-1)[0, :, 0, 1:]          # głowy x łatki
            cechy.append(dict(cls=cls.cpu().numpy(), pt=pt.cpu().numpy(), uwaga=A.mean(0).float().cpu().numpy()))
    uchwyt.remove()
    P = np.concatenate([c['pt'] for c in cechy]); mu = P.mean(0)
    _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
    rgb_all = (P - mu) @ Vt[:3].T
    lo, hi = np.percentile(rgb_all, 1, axis=0), np.percentile(rgb_all, 99, axis=0)
    cls_sd = cechy[0]['cls'] / np.linalg.norm(cechy[0]['cls'])

    fig, ax = plt.subplots(len(obrazy), 4, figsize=(20, 3.9 * len(obrazy)), layout='constrained')
    for i, ((t, im), c) in enumerate(zip(obrazy, cechy)):
        ax[i, 0].imshow(im); ax[i, 0].set_title(t, fontsize=10, loc='left')
        pca = np.clip(((c['pt'] - mu) @ Vt[:3].T - lo) / (hi - lo), 0, 1).reshape(GH, GW, 3)
        ax[i, 1].imshow(pca, interpolation='nearest'); ax[i, 1].set_title('cechy łatek DINO (PCA-3 jako RGB, wspólna baza)', fontsize=9)
        u = c['uwaga'].reshape(GH, GW)
        ax[i, 2].imshow(im); ax[i, 2].imshow(cv2.resize(u / u.max(), (480, 320), interpolation=cv2.INTER_CUBIC), cmap='inferno', alpha=.6)
        ax[i, 2].set_title('uwaga CLS ostatniego bloku (średnia po 6 głowach)', fontsize=9)
        pn = c['pt'] / np.linalg.norm(c['pt'], axis=1, keepdims=True)
        s = (pn @ cls_sd).reshape(GH, GW)
        h = ax[i, 3].imshow(s, cmap='RdYlGn', vmin=-.1, vmax=.5, interpolation='nearest')
        ax[i, 3].set_title(f'kosinus łatki z CLS obrazu SD (CLS tego obrazu vs SD: {float(c["cls"] @ cls_sd / np.linalg.norm(c["cls"])):.2f})', fontsize=9)
        for a in ax[i]:
            a.set_axis_off()
    fig.colorbar(h, ax=ax[:, 3].tolist(), fraction=.03, label='kosinus')
    fig.suptitle('Cechy DINOv2 na obrazie SD i na zdjęciach, które DINO do niego dopasowało', fontsize=13)
    fig.savefig(KAT / f'CECHY_DINO_{OBRAZ}.png', dpi=100); plt.close(fig)
    print('zapisano', KAT / f'CECHY_DINO_{OBRAZ}.png')


if __name__ == '__main__':
    main()
