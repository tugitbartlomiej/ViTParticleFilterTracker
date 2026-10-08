"""Etap 11 (zlecenie użytkownika): obrazy SD z etapu 10 jako zapytania do wyszukania podobnych prawdziwych zdjęć w puli
treningowej (35 000 klatek TRAIN, cechy DINOv2 ViT-S/14 CLS, outputs/vit_dino_faiss_35000). Te same przekształcenia
co pula (shorter 256 BICUBIC, CenterCrop 224, ImageNet, L2 CLS). Dla każdego obrazu SD (8 celów x 11 kroków przyrostu)
5 najbliższych klatek; odniesienie: zapytanie samym zdjęciem celu. Paski: górny rząd obrazy SD, dolny najbliższa
prawdziwa klatka.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import korekta_latentu as KL

DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CELE = [135, 102, 369, 11, 354, 224, 236, 270]
PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize((.485, .456, .406), (.229, .224, .225))])


def main():
    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu'))
    model = model.cuda().eval()
    man = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    F = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    rows = KL.manifest()

    @torch.inference_mode()
    def cls(rgb):
        x = PRE(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).cuda()
        f = model.forward_features(x)['x_norm_clstoken'][0].float().cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-12)

    def klatka(j):
        m = man[j]
        with zipfile.ZipFile(m['archive']) as zf:
            im = np.asarray(Image.open(io.BytesIO(zf.read(m['member']))).convert('RGB'))
        return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)

    wyn = []
    kat = OUT / 'sd_szukanie'; kat.mkdir(exist_ok=True)
    for i in CELE:
        rgb, _ = KL.obraz_manifestu(rows[i])
        film = rows[i]['film']
        q0 = cls(rgb); s0 = F @ q0; top0 = np.argsort(-s0)[:5]
        gora, dol = [], []
        for k in range(11):
            p = OUT / 'sd_interpolacja' / f'm{i:04d}_minmax_k{k:02d}.png'
            sd = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            q = cls(sd); s = F @ q; top = np.argsort(-s)[:5]
            rek = dict(cel=i, k=k, przyrost=1 - k / 10, film_celu=film,
                       top=[dict(j=int(j), film=man[j]['film'], klatka=int(man[j]['frame']), cos=float(s[j])) for j in top],
                       wspolne_z_zapytaniem_zdjeciem=int(len(set(top.tolist()) & set(top0.tolist()))),
                       top1_z_filmu_celu=man[top[0]]['film'] == film, cos_do_zdjecia_celu=float(q @ q0))
            wyn.append(rek)
            y = cv2.cvtColor(sd, cv2.COLOR_RGB2BGR); cv2.rectangle(y, (0, 0), (480, 26), (0, 0, 0), -1)
            cv2.putText(y, f'SD {int(round(100 * rek["przyrost"]))}% przyrostu', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            r = cv2.cvtColor(klatka(int(top[0])), cv2.COLOR_RGB2BGR); cv2.rectangle(r, (0, 0), (480, 26), (0, 0, 0), -1)
            cv2.putText(r, f'{man[top[0]]["film"][3:7]}/{man[top[0]]["frame"]} cos {s[top[0]]:.2f}', (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0) if man[top[0]]['film'] == film else (0, 200, 255), 1)
            gora.append(cv2.resize(y, (320, 213))); dol.append(cv2.resize(r, (320, 213)))
            print(i, k, rek['top'][0], 'wspólne z zapytaniem zdjęciem', rek['wspolne_z_zapytaniem_zdjeciem'], flush=True)
        z = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); cv2.rectangle(z, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(z, f'zdjecie celu m{i:04d}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        r0 = cv2.cvtColor(klatka(int(top0[0])), cv2.COLOR_RGB2BGR); cv2.rectangle(r0, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(r0, f'{man[top0[0]]["film"][3:7]}/{man[top0[0]]["frame"]} cos {s0[top0[0]]:.2f}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1)
        siatka = np.concatenate([np.concatenate([cv2.resize(z, (320, 213))] + gora, 1), np.concatenate([cv2.resize(r0, (320, 213))] + dol, 1)], 0)
        cv2.imwrite(str(kat / f'SZUKANIE_m{i:04d}.jpg'), siatka, [cv2.IMWRITE_JPEG_QUALITY, 85])
        wyn.append(dict(cel=i, k='zdjecie', top=[dict(j=int(j), film=man[j]['film'], klatka=int(man[j]['frame']), cos=float(s0[j])) for j in top0]))
        (OUT / 'ETAP11_SZUKANIE_DINO.json').write_text(json.dumps(dict(wyniki=wyn, opis=__doc__), indent=1, ensure_ascii=False), encoding='utf-8')
    print('KONIEC')


if __name__ == '__main__':
    main()
