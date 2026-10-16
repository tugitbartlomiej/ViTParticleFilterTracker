"""Przykłady klasyfikacji Imagenette małym ViT od zera (bieg klas_vit_od_zera_60ep_lr_5e-4) z mapą wkładu tokenów.
Głowa liczy logit jako W·średnia_k(norm(t_k)) + b, więc wkład tokenu k do klasy c to dokładnie W_c·norm(t_k) / 400 (bez przybliżeń).
Mapa: wkład do klasy przewidzianej minus średni wkład do wszystkich klas, siatka 20×20 na obrazie 320×320.
Wyjście: przyklady_klasyfikacji.png (16 obrazów walidacyjnych, checkpoint najlepszej epoki) i ewolucja_klasyfikacji.png (4 obrazy × epoki).
Uruchomienie: python pokaz_klasyfikacje.py <katalog biegu> <imagenette2-320> <katalog wyjścia>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import model as M
import trening_klasyfikacja as TK

NAZWY = {'n01440764': 'lin (ryba)', 'n02102040': 'springer spaniel', 'n02979186': 'magnetofon', 'n03000684': 'piła łańcuchowa',
         'n03028079': 'kościół', 'n03394916': 'waltornia', 'n03417042': 'śmieciarka', 'n03425413': 'dystrybutor paliwa',
         'n03445777': 'piłka golfowa', 'n03888257': 'spadochron'}


@torch.no_grad()
def policz(net, X):
    x = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.
    s = net.szkielet
    x = (x - M.MEAN) / M.STD
    f = s.lodyga(x)
    B, C, h, w = f.shape
    t = f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, C, x.device).to(f.dtype)
    for b in s.bloki:
        t = b(t)
    tn = s.norm(t)                                               # (B, 400, 192)
    wk = tn @ net.glowa.weight.T / tn.shape[1]                   # (B, 400, 10) wkład tokenu do logitu
    logit = wk.sum(1) + net.glowa.bias
    return F.softmax(logit, 1).numpy(), wk.numpy(), h, w


def wczytaj(bieg, krok):
    net = TK.Klasyfikator('vit', 10).eval()
    net.load_state_dict(torch.load(bieg / 'wagi' / f'k{krok:05d}.pt', map_location='cpu', weights_only=False)['net'])
    return net


def mapa(wk, c, h, w):
    m = wk[:, c] - wk.mean(1)
    return m.reshape(h, w)


def main():
    bieg, dane, wy = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    wy.mkdir(parents=True, exist_ok=True)
    oceny = [o for o in json.load(open(bieg / 'dane' / 'OCENY.json')) if o.get('epoka')]
    best = max(oceny, key=lambda o: o['acc'])
    wal = TK.wczytaj_imagenette(dane, 'val')
    kl = [NAZWY.get(k, k) for k in wal.klasy]
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.where(wal.y == c)[0], 2, replace=False) for c in range(10)])[:20]
    net = wczytaj(bieg, best['krok'])
    P, WK, h, w = policz(net, wal.X[idx])
    zle = [i for i in range(len(idx)) if P[i].argmax() != wal.y[idx[i]]]
    dobre = [i for i in range(len(idx)) if i not in zle]
    wybor = (dobre[:12] + zle[:4])[:16] if zle else dobre[:16]

    fig, ax = plt.subplots(4, 8, figsize=(24, 13), dpi=90)
    for n, i in enumerate(wybor):
        a0, a1 = ax[n // 4, 2 * (n % 4)], ax[n // 4, 2 * (n % 4) + 1]
        p = P[i]; c = int(p.argmax()); prawda = int(wal.y[idx[i]])
        a0.imshow(wal.X[idx[i]]); a0.axis('off')
        kolor = '#1a7f37' if c == prawda else '#c62828'
        a0.set_title(f'prawda: {kl[prawda]}\nsieć: {kl[c]} ({p[c]:.2f})', fontsize=10, color=kolor)
        m = mapa(WK[i], c, h, w)
        a1.imshow(wal.X[idx[i]]); a1.imshow(np.kron(m, np.ones((16, 16))), cmap='RdBu_r', alpha=0.55,
                                              vmin=-np.abs(m).max(), vmax=np.abs(m).max()); a1.axis('off')
        top = np.argsort(-p)[:3]
        a1.set_title('top-3: ' + ', '.join(f'{kl[t]} {p[t]:.2f}' for t in top), fontsize=7.5)
    fig.suptitle(f'Mały ViT od zera na Imagenette, epoka {best["epoka"]} (top-1 walidacji {best["acc"]:.3f}). '
                 'Klasyfikacja całego obrazu, bez ramek. Czerwone tokeny przemawiają za klasą wybraną przez sieć, niebieskie przeciw.', fontsize=13)
    fig.tight_layout(); fig.savefig(wy / 'przyklady_klasyfikacji.png'); plt.close(fig)

    epoki = [0, 1, 5, 20, best['epoka'], 60]
    kroki = {0: 0, **{o['epoka']: o['krok'] for o in oceny}}
    cztery = wybor[:4]
    fig, ax = plt.subplots(len(cztery), len(epoki) + 1, figsize=(3.2 * (len(epoki) + 1), 3.4 * len(cztery)), dpi=90)
    for j, e in enumerate(epoki):
        netj = wczytaj(bieg, kroki[e])
        Pj, WKj, h, w = policz(netj, wal.X[idx[cztery]])
        for r, i in enumerate(cztery):
            prawda = int(wal.y[idx[i]]); c = int(Pj[r].argmax())
            if j == 0:
                ax[r, 0].imshow(wal.X[idx[i]]); ax[r, 0].axis('off'); ax[r, 0].set_title(f'prawda: {kl[prawda]}', fontsize=10)
            m = mapa(WKj[r], prawda, h, w)
            a = ax[r, j + 1]
            a.imshow(wal.X[idx[i]]); a.imshow(np.kron(m, np.ones((16, 16))), cmap='RdBu_r', alpha=0.55, vmin=-np.abs(m).max(), vmax=np.abs(m).max())
            a.axis('off')
            a.set_title(f'epoka {e}: p(prawda) {Pj[r, prawda]:.2f}\nsieć: {kl[c]}', fontsize=9, color='#1a7f37' if c == prawda else '#c62828')
    fig.suptitle('Jak sieć uczy się rozpoznawać: mapa wkładu tokenów do PRAWDZIWEJ klasy w kolejnych epokach', fontsize=13)
    fig.tight_layout(); fig.savefig(wy / 'ewolucja_klasyfikacji.png'); plt.close(fig)
    print('zapisano', wy / 'przyklady_klasyfikacji.png', wy / 'ewolucja_klasyfikacji.png', 'epoka', best['epoka'], 'krok', best['krok'])


if __name__ == '__main__':
    main()
