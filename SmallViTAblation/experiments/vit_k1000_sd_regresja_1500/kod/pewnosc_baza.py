"""Pytanie użytkownika 2026-10-09: jaka jest pewność detekcji ViT K1000 na obrazach zbioru uczącego (skoro z* K1000 zdjęcia
0055/4019 w naturalnym pudle daje tylko 0,06). Naturalna detekcja (bez żadnej korekty latentu) K1000 i, dla porównania,
K8000 na wszystkich 1500 obrazach bazy uczącej K1000 (MANIFEST_1500.jsonl, ramki bbox_xyxy_480) i na zdjęciu celu.
Miary jak korekta_latentu.ocena_obrazu: pewność najpewniejszej ramki, jej IoU z GT, trafienie (pewność >= 0,5 i IoU >= 0,5).
Zapis: outputs/vit_k1000_sd_regresja_1500/pewnosc_baza/PEWNOSC_BAZA.json, PEWNOSC_BAZA.png
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_sd_regresja_1500/kod/pewnosc_baza.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'vit_k1000_sd_regresja_1500' / 'pewnosc_baza'
OUT.mkdir(parents=True, exist_ok=True)
BAZA = ROOT / 'outputs' / 'vit_k1000_1500_pelne_latenty_90_20261008' / 'provenance' / 'MANIFEST_1500.jsonl'
ARCH = ROOT.parent / 'Dataset' / 'Cataract-LMM' / '3_Object_Tracking' / 'annotations'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
MODELE = {'K1000': ROOT / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt',
          'K8000': ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'}
CEL_J = 10389
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import korekta_latentu as KL


def obraz(film, klatka):
    with zipfile.ZipFile(ARCH / f'{film}.zip') as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(f'{film}/{film}_{klatka:06d}.jpg'))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    baza = [json.loads(l) for l in BAZA.read_text(encoding='utf-8').splitlines() if l.strip()]
    man = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    mc = man[CEL_J]; w, h = mc['width'], mc['height']
    g = mc['gt'][0]['bbox_xywh']
    cel = dict(film=mc['film'], klatka=int(mc['frame']), ramka=[g[0] * 480 / w, g[1] * 320 / h, (g[0] + g[2]) * 480 / w, (g[1] + g[3]) * 320 / h])
    wyn = {}
    for nazwa, sciezka in MODELE.items():
        st = KL.Stanowisko(sciezka)
        rek = []
        for i, r in enumerate(baza):
            film = r['frame_id'].split('/')[-2]; kl = int(r['frame_id'].split('/')[-1])
            with torch.inference_mode():
                o = st.przebieg(obraz(film, kl))['o']
                bx, sc = KL.dekoduj(o[None])
            m = KL.ocena_obrazu(bx[0].cpu().numpy(), sc[0].cpu().numpy(), np.asarray(r['bbox_xyxy_480'], np.float32).reshape(1, 4))
            rek.append(dict(i=i, film=film, klatka=kl, pewnosc=m['pewnosc_top'], iou=m['iou_top'], trafienie=bool(m['tp'] > 0)))
            if i % 300 == 0:
                print(nazwa, i, flush=True)
        with torch.inference_mode():
            o = st.przebieg(obraz(cel['film'], cel['klatka']))['o']; bx, sc = KL.dekoduj(o[None])
        mcel = KL.ocena_obrazu(bx[0].cpu().numpy(), sc[0].cpu().numpy(), np.asarray(cel['ramka'], np.float32).reshape(1, 4))
        p = np.array([x['pewnosc'] for x in rek]); u = np.array([x['iou'] for x in rek])
        wyn[nazwa] = dict(sha=st.sha, n=len(rek), pewnosc_mediana=float(np.median(p)), pewnosc_p90=float(np.percentile(p, 90)),
                          pewnosc_max=float(p.max()), ulamek_pewnosc_ponad_05=float((p >= .5).mean()), iou_mediana=float(np.median(u)),
                          ulamek_trafien=float(np.mean([x['trafienie'] for x in rek])),
                          cel_0055_4019=dict(pewnosc=mcel['pewnosc_top'], iou=mcel['iou_top'], trafienie=bool(mcel['tp'] > 0)),
                          najpewniejsze=sorted(rek, key=lambda x: -x['pewnosc'])[:10], obrazy=rek)
        print(nazwa, {k: v for k, v in wyn[nazwa].items() if k not in ('obrazy', 'najpewniejsze')}, flush=True)
        del st; torch.cuda.empty_cache()
    (OUT / 'PEWNOSC_BAZA.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for nazwa, kol in (('K1000', 'tab:blue'), ('K8000', 'tab:orange')):
        p = np.array([x['pewnosc'] for x in wyn[nazwa]['obrazy']]); u = np.array([x['iou'] for x in wyn[nazwa]['obrazy']])
        ax[0].hist(p, bins=50, range=(0, 1), alpha=.5, color=kol, label=f'{nazwa}: mediana {np.median(p):.2f}, ≥0,5: {100 * (p >= .5).mean():.0f} %')
        ax[1].scatter(p, u, s=4, alpha=.4, color=kol, label=nazwa)
        c = wyn[nazwa]['cel_0055_4019']; ax[1].scatter([c['pewnosc']], [c['iou']], s=150, marker='*', color=kol, edgecolors='k')
    ax[0].set_xlabel('pewność najpewniejszej ramki'); ax[0].set_ylabel('liczba obrazów'); ax[0].legend(); ax[0].set_title('1500 obrazów bazy uczącej K1000')
    ax[1].set_xlabel('pewność'); ax[1].set_ylabel('IoU najpewniejszej ramki z GT'); ax[1].legend(); ax[1].set_title('gwiazdka = zdjęcie celu 0055/4019')
    for a in ax:
        a.grid(alpha=.3)
    fig.savefig(OUT / 'PEWNOSC_BAZA.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
