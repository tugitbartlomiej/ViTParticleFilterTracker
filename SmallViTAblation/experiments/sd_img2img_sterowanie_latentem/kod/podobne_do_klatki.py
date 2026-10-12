"""N najbliższych w DINO klatek TRAIN do jednej klatki TRAIN (domyślnie 0088/1661, j = 17943: top-1 dla najlepszego obrazu
regresji g19). Zapytanie = cecha CLS tej klatki z indeksu vit_dino_faiss_35000 (ta sama, którą liczono indeks), z
wyłączeniem jej samej i zdjęcia celu 0055/4019. Dla każdej znalezionej klatki latent ViT K8000 i postęp wzdłuż
δ = z* − z0 (cały i widoczny dla fc2), jak w optymalizacja_regresja.py.
Zapis: outputs/sd_img2img_sterowanie_latentem/optymalizacja_regresja/PODOBNE_<N>_<film>_<klatka>/ —
klatki w oryginalnej rozdzielczości (<poz>_<film>_<klatka>.jpg), PODOBNE.json, PODOBNE.csv, ARKUSZ.jpg, WYKRES.png.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/podobne_do_klatki.py [j] [N]
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
CEL_J = 10389
Z_SZT = ROOT / 'outputs' / 'latent_cel_do_obrazow' / 'k8000_trening' / f'zb_j{CEL_J:05d}_minmax.npy'
J = int(sys.argv[1]) if len(sys.argv) > 1 else 17943
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import korekta_latentu as KL

MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def surowa(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        return z.read(MAN[j]['member'])


def klatka(j):
    im = np.asarray(Image.open(io.BytesIO(surowa(j))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    out = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_regresja' / f'PODOBNE_{N}_{MAN[J]["film"][3:7]}_{MAN[J]["frame"]}'
    out.mkdir(parents=True, exist_ok=True)
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12
    s = Fp @ Fp[J]; s[J] = -9; s[CEL_J] = -9
    top = np.argsort(-s)[:N]

    st = KL.Stanowisko(VIT)
    assert st.sha == '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b'
    W2 = st.blok.mlp.fc2.weight.detach().float()
    z0 = st.przebieg(klatka(CEL_J))['z']
    zs = torch.from_numpy(np.load(Z_SZT)).to(z0.device)
    d = zs - z0; dd = (d * d).sum(); d_w = d @ W2.T

    def miary(z):
        dz = z - z0
        return dict(ulamek=float((dz * d).sum() / dd), ulamek_widoczny=float(((dz @ W2.T) * d_w).sum() / (d_w * d_w).sum()),
                    cos_przyrost=float((dz * d).sum() / (dz.norm() * d.norm() + 1e-12)))

    with torch.inference_mode():
        m_zap = miary(st.przebieg(klatka(J))['z'])
    wyn, miniatury = [], []
    for poz, j in enumerate(top, 1):
        j = int(j); im = klatka(j)
        with torch.inference_mode():
            m = miary(st.przebieg(im)['z'])
        nazwa = f'{poz:03d}_{MAN[j]["film"]}_{MAN[j]["frame"]:06d}.jpg'
        (out / nazwa).write_bytes(surowa(j))                                  # oryginalny plik z archiwum, bez ponownej kompresji
        wyn.append(dict(poz=poz, j=j, film=MAN[j]['film'], klatka=int(MAN[j]['frame']), plik=nazwa, cos_dino=float(s[j]), **m,
                        archiwum=MAN[j]['archive'], czlon=MAN[j]['member']))
        y = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (240, 160)); cv2.rectangle(y, (0, 0), (240, 30), (0, 0, 0), -1)
        cv2.putText(y, f'{poz} {MAN[j]["film"][3:7]}/{MAN[j]["frame"]} {s[j]:.2f}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(y, f'{100 * m["ulamek"]:.1f}% / {100 * m["ulamek_widoczny"]:.1f}%', (3, 26), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1, cv2.LINE_AA)
        miniatury.append(y)

    kol = 15
    while len(miniatury) % kol:
        miniatury.append(np.zeros_like(miniatury[0]))
    cv2.imwrite(str(out / 'ARKUSZ.jpg'), np.concatenate([np.concatenate(miniatury[i:i + kol], 1) for i in range(0, len(miniatury), kol)], 0),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(out / 'PODOBNE.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(wyn[0])); w.writeheader(); w.writerows(wyn)
    filmy = {}
    for r in wyn:
        filmy[r['film']] = filmy.get(r['film'], 0) + 1
    u = np.array([r['ulamek'] for r in wyn]); uw = np.array([r['ulamek_widoczny'] for r in wyn])
    pods = dict(zapytanie=dict(j=J, film=MAN[J]['film'], klatka=int(MAN[J]['frame']), **m_zap), n=N, filmy=filmy,
                postep_mediana=float(np.median(u)), postep_widoczny_mediana=float(np.median(uw)),
                ile_postep_dodatni=int((u > 0).sum()), ile_postep_ponad_10=int((u > .1).sum()), wyniki=wyn)
    (out / 'PODOBNE.json').write_text(json.dumps(pods, indent=1, ensure_ascii=False), encoding='utf-8')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    ax[0].plot(np.arange(1, N + 1), 100 * u, '.-', label='cały przyrost'); ax[0].plot(np.arange(1, N + 1), 100 * uw, '.-', label='część widoczna dla fc2')
    ax[0].axhline(100 * m_zap['ulamek'], c='C0', ls='--', lw=.8, label='klatka zapytania'); ax[0].axhline(0, c='k', lw=.6)
    ax[0].set_xlabel('pozycja w rankingu DINO'); ax[0].set_ylabel('postęp w stronę sztucznego latentu [%]'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f'{N} najbliższych w DINO do {MAN[J]["film"][3:7]}/{MAN[J]["frame"]}')
    f_s = sorted(filmy.items(), key=lambda t: -t[1])
    ax[1].barh([k[3:7] for k, _ in f_s][::-1], [v for _, v in f_s][::-1]); ax[1].set_xlabel('liczba klatek'); ax[1].set_title('z jakich filmów')
    fig.savefig(out / 'WYKRES.png', dpi=110); plt.close(fig)
    print('zapisano', out)
    print({k: v for k, v in pods.items() if k != 'wyniki'})


if __name__ == '__main__':
    main()
