"""Grafika latentów pętli z szumu (zlecenie użytkownika 2026-10-08): dla zdjęcia oryginalnego i każdej generacji wiersz
[obraz | latent obrazu 600 x 768 | sztuczny latent z* | różnica z* − latent obrazu].

Latent = bloki[-1].mlp.act ViT K8000 zapisany przez petla_szum.py w LATENTY.npz. Oś pionowa: 600 tokenów (siatka 20 x 30
wierszami), znaczniki po lewej = tokeny w ramce etykiety narzędzia. Oś pozioma: 768 kanałów posortowanych malejąco
wg średniego przyrostu z* − z0 (ta sama kolejność we wszystkich panelach). Jedna skala kolorów dla latentów i jedna dla różnic.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/grafika_latentow.py [petla_szum_elita]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / (sys.argv[1] if len(sys.argv) > 1 else 'petla_szum_elita')
OUT = KAT / 'latenty_grafika'
OUT.mkdir(exist_ok=True)
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CEL_J = 10389
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    L = np.load(KAT / 'LATENTY.npz')
    Z = L['obrazy'].astype(np.float32); z0 = L['z0'].astype(np.float32); zs = L['z_sztuczny'].astype(np.float32)
    wyn = json.loads((KAT / 'WYNIK.json').read_text(encoding='utf-8'))['wyniki']
    man = json.loads((DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()[CEL_J])
    w, h = man['width'], man['height']
    tok = np.zeros(600, bool)
    for g in man['gt']:
        x0, y0, bw, bh = g['bbox_xywh']
        x0, x1, y0, y1 = x0 * 480 / w, (x0 + bw) * 480 / w, y0 * 320 / h, (y0 + bh) * 320 / h
        ys, xs = np.mgrid[0:20, 0:30]
        tok |= (((xs + .5) * 16 > x0) & ((xs + .5) * 16 < x1) & ((ys + .5) * 16 > y0) & ((ys + .5) * 16 < y1)).reshape(-1)
    por = np.argsort(-(zs - z0).mean(0))
    vmin, vmax = -0.2, float(np.percentile(zs, 99.5))
    dlim = float(np.percentile(np.abs(zs - z0), 99.5))
    wiersze = [('zdjęcie oryginalne (z0)', cv2.cvtColor(cv2.imread(str(KAT / 'ZDJECIE_CELU.png')), cv2.COLOR_BGR2RGB), z0, None)]
    for r in wyn:
        g = r['generacja']
        wiersze.append((f'generacja {g}' + ('  (najlepsza)' if g == max(x['najlepsza_dotad'] for x in wyn) else ''),
                        cv2.cvtColor(cv2.imread(str(KAT / 'obrazy' / f'g{g:02d}.png')), cv2.COLOR_BGR2RGB), Z[g], r))

    def rysuj(ax_row, tytul, img, z, r):
        a0, a1, a2, a3 = ax_row
        a0.imshow(img); a0.set_axis_off()
        opis = '' if r is None else f'\nułamek przyrostu {100 * r["ulamek"]:.1f} %, widoczny dla fc2 {100 * r["ulamek_widoczny"]:.1f} %\n||z* − z|| / ||z* − z0|| = {r["odl_wzgl"]:.3f}'
        a0.set_title(tytul + opis, fontsize=9, loc='left')
        im1 = a1.imshow(z[:, por], aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, interpolation='nearest')
        a1.set_title('latent obrazu (600 tokenów × 768 kanałów)', fontsize=9)
        a2.imshow(zs[:, por], aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, interpolation='nearest')
        a2.set_title('sztuczny latent z*', fontsize=9)
        im3 = a3.imshow((zs - z)[:, por], aspect='auto', cmap='RdBu_r', vmin=-dlim, vmax=dlim, interpolation='nearest')
        a3.set_title(f'różnica z* − latent obrazu (||·|| = {np.linalg.norm(zs - z):.0f})', fontsize=9)
        for a in (a1, a2, a3):
            a.set_xlabel('kanał (posortowany wg przyrostu z* − z0)', fontsize=7); a.tick_params(labelsize=6)
            a.scatter(np.full(tok.sum(), -8), np.where(tok)[0], c='lime', s=4, marker='s', clip_on=False)
        a1.set_ylabel('token (zielone: narzędzie)', fontsize=7)
        return im1, im3

    def strona(lista, plik):
        fig, ax = plt.subplots(len(lista), 4, figsize=(22, 4.6 * len(lista)), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]),
                               squeeze=False, layout='constrained')
        for i, (t, img, z, r) in enumerate(lista):
            im1, im3 = rysuj(ax[i], t, img, z, r)
            fig.colorbar(im1, ax=ax[i, 2], fraction=.05, pad=.02).set_label('wartość latentu', fontsize=8)
            fig.colorbar(im3, ax=ax[i, 3], fraction=.05, pad=.02).set_label('z* − z', fontsize=8)
        fig.savefig(plik, dpi=90); plt.close(fig)

    # osobny plik na każdy wiersz i strony po 10 wierszy (pierwsza zaczyna się od zdjęcia oryginalnego)
    for i, wr in enumerate(wiersze):
        strona([wr], OUT / ('00_zdjecie_oryginalne.png' if i == 0 else f'g{i - 1:02d}.png'))
    for s in range(0, len(wiersze), 10):
        strona(wiersze[s:s + 10], OUT / f'STRONA_{s // 10 + 1}.png')
    print('zapisano', len(wiersze), 'wierszy w', OUT)


if __name__ == '__main__':
    main()
