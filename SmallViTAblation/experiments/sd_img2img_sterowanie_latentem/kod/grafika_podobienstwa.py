"""Graficzne porównanie cech: najlepszy obraz SD (g148), zdjęcie celu i po 5 najbliższych klatek TRAIN (DINO) dla każdego
z nich (SZUKANIE_TRAIN_g148.json). Panele: miniatury, macierz kosinusów DINO CLS, macierz kosinusów pełnego latentu ViT
K8000 (600 x 768), położenie latentów na osi z0 -> z* (postęp wzdłuż δ i odejście w bok), postęp każdego zdjęcia,
mapy 20 x 30 kosinusu latentu token po tokenie wobec obrazu SD.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/grafika_podobienstwa.py [g148]
"""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_warunku'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
CEL_J = 10389
OBRAZ = sys.argv[1] if len(sys.argv) > 1 else 'g148'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import korekta_latentu as KL

MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def klatka(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    sz = json.loads((KAT / f'SZUKANIE_TRAIN_{OBRAZ}.json').read_text(encoding='utf-8'))['wyniki']
    kq, kc = f'obraz SD {OBRAZ}', 'zdjecie celu 0055/4019'
    obrazy = [(f'SD {OBRAZ}', cv2.cvtColor(cv2.imread(str(KAT / 'obrazy' / f'{OBRAZ}.png')), cv2.COLOR_BGR2RGB), 'sd'),
              ('cel 0055/4019', klatka(CEL_J), 'cel')]
    obrazy += [(f'A{i + 1} {t["film"][3:7]}/{t["klatka"]}', klatka(t['j']), 'a') for i, t in enumerate(sz[kq]['top'])]
    obrazy += [(f'B{i + 1} {t["film"][3:7]}/{t["klatka"]}', klatka(t['j']), 'b') for i, t in enumerate(sz[kc]['top'])]
    n = len(obrazy)

    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])
    with torch.inference_mode():
        Fd = np.stack([model.forward_features(PRE(Image.fromarray(im)).unsqueeze(0).cuda())['x_norm_clstoken'][0].float().cpu().numpy()
                       for _, im, _ in obrazy])
    Fd /= np.linalg.norm(Fd, axis=1, keepdims=True)

    st = KL.Stanowisko(VIT)
    Z = np.stack([st.przebieg(im)['z'].cpu().numpy() for _, im, _ in obrazy])          # n x 600 x 768
    L = np.load(KAT / 'LATENTY.npz'); z0 = L['z0'].astype(np.float32); zs = L['z_sztuczny'].astype(np.float32)
    d = (zs - z0).reshape(-1)
    Zf = Z.reshape(n, -1); Zn = Zf / np.linalg.norm(Zf, axis=1, keepdims=True)
    S_dino, S_vit = Fd @ Fd.T, Zn @ Zn.T
    dz = Zf - z0.reshape(-1)
    os1 = dz @ d / (d @ d); os2 = np.linalg.norm(dz - os1[:, None] * d, axis=1) / np.linalg.norm(d)
    W2 = st.blok.mlp.fc2.weight.detach().float().cpu().numpy(); dw = (zs - z0) @ W2.T
    vis = np.array([((Z[i] - z0) @ W2.T * dw).sum() / (dw * dw).sum() for i in range(n)])
    kol = {'sd': 'tab:red', 'cel': 'k', 'a': 'tab:orange', 'b': 'tab:blue'}
    etyk = [e for e, _, _ in obrazy]

    fig = plt.figure(figsize=(22, 17), layout='constrained')
    gs = fig.add_gridspec(4, n, height_ratios=[.55, 2.2, 2.2, .9])
    for i, (e, im, r) in enumerate(obrazy):
        a = fig.add_subplot(gs[0, i]); a.imshow(im); a.set_axis_off(); a.set_title(e, fontsize=8, color=kol[r])
    for k, (S, tyt) in enumerate([(S_dino, 'podobieństwo DINO (kosinus CLS), tym szukano'),
                                  (S_vit, 'podobieństwo latentu ViT K8000 (kosinus 600 × 768)')]):
        a = fig.add_subplot(gs[1, k * (n // 2):(k + 1) * (n // 2)])
        lo = np.percentile(S[~np.eye(n, dtype=bool)], 2)
        im = a.imshow(S, cmap='viridis', vmin=lo, vmax=1)
        for i in range(n):
            for j in range(n):
                a.text(j, i, f'{S[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='w' if S[i, j] < (lo + 1) / 2 else 'k')
        a.set_xticks(range(n)); a.set_xticklabels(etyk, rotation=60, ha='right', fontsize=7)
        a.set_yticks(range(n)); a.set_yticklabels(etyk, fontsize=7)
        for t, (_, _, r) in zip(a.get_xticklabels(), obrazy):
            t.set_color(kol[r])
        for t, (_, _, r) in zip(a.get_yticklabels(), obrazy):
            t.set_color(kol[r])
        a.set_title(tyt, fontsize=11); fig.colorbar(im, ax=a, fraction=.04)

    a = fig.add_subplot(gs[2, :n // 2])
    a.scatter([0], [0], marker='s', s=90, c='k'); a.scatter([1], [0], marker='*', s=220, c='r')
    a.annotate('z0 (cel)', (0, 0), xytext=(5, 5), textcoords='offset points', fontsize=8)
    a.annotate('z* (sztuczny)', (1, 0), xytext=(5, 5), textcoords='offset points', fontsize=8)
    for i, (e, _, r) in enumerate(obrazy):
        a.scatter(os1[i], os2[i], c=kol[r], s=60, edgecolors='k', linewidths=.5)
        a.annotate(e.split(' ')[0], (os1[i], os2[i]), xytext=(4, 3), textcoords='offset points', fontsize=8, color=kol[r])
    a.set_xlabel('postęp wzdłuż δ = z* − z0 (0 = zdjęcie celu, 1 = sztuczny latent)'); a.set_ylabel('odejście w bok od osi (/||δ||)')
    a.set_title('gdzie leżą latenty zdjęć wobec osi zdjęcie celu → sztuczny latent', fontsize=11); a.grid(alpha=.3)

    a2 = fig.add_subplot(gs[2, n // 2:])
    xs = np.arange(n)
    a2.barh(xs - .17, 100 * os1, height=.34, color=[kol[r] for _, _, r in obrazy], label='cały przyrost')
    a2.barh(xs + .17, 100 * vis, height=.34, color='gray', label='część widoczna dla fc2')
    a2.set_yticks(xs); a2.set_yticklabels(etyk, fontsize=8); a2.invert_yaxis(); a2.axvline(0, c='k', lw=.6)
    a2.set_xlabel('postęp w stronę sztucznego latentu [%]'); a2.legend(fontsize=8); a2.grid(alpha=.3, axis='x')
    a2.set_title('jak mocno latent każdego zdjęcia idzie w stronę sztucznego', fontsize=11)

    q = Z[0] / (np.linalg.norm(Z[0], axis=1, keepdims=True) + 1e-9)
    for i, (e, _, r) in enumerate(obrazy):
        zi = Z[i] / (np.linalg.norm(Z[i], axis=1, keepdims=True) + 1e-9)
        a = fig.add_subplot(gs[3, i]); a.imshow((q * zi).sum(1).reshape(20, 30), cmap='magma', vmin=0, vmax=1); a.set_axis_off()
        a.set_title(f'{e.split(" ")[0]}: tokeny vs SD', fontsize=7, color=kol[r])
    fig.suptitle('Podobieństwo cech: obraz SD po optymalizacji warunku (czerwony), zdjęcie celu (czarny), '
                 'A = najbliższe obrazowi SD (pomarańczowe), B = najbliższe zdjęciu celu (niebieskie)', fontsize=13)
    fig.savefig(KAT / f'PODOBIENSTWO_CECH_{OBRAZ}.png', dpi=110); plt.close(fig)
    print('zapisano', KAT / f'PODOBIENSTWO_CECH_{OBRAZ}.png')


if __name__ == '__main__':
    main()
