"""Etap 10 (zlecenie użytkownika): SD bezpośrednio z umiarkowanego sztucznego latentu i 10 równych kroków z powrotem do
latentu oryginału. Dla każdego tokenu przyrost δ = z_sztuczny − z_oryginał; z_k = z_sztuczny − (k/10) δ, k = 0..10
(k = 10 to oryginał). Każdy z_k jako warunek SD (DDIM 40 kroków od czystego szumu, to samo ziarno dla wszystkich k,
bez img2img i bez prowadzenia), potem ViT k1000 na obrazie: detekcja z etykietą celu, latent obrazu wobec δ.
Sztuczne latenty z etapu 8: każdy kanał w zakresie danych (minmax) albo z marginesem 0,25 szerokości (mm_m0.25).
Uruchomienie z SmallViTAblation: F:/Python/python.exe experiments/latent_cel_do_obrazow/kod/sd_interpolacja.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch

import korekta_latentu as KL
import silnik

CELE = [135, 102, 369, 11, 354, 224, 236, 270]
WARIANTY = ['minmax', 'mm_m0.25']
KROKI, ZIARNO = 40, 0
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main():
    st = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'])
    tr = silnik.zaladuj_model()
    rows = KL.manifest()
    kat = OUT / 'sd_interpolacja'; kat.mkdir(exist_ok=True)
    wyn = []
    for war in WARIANTY:
        for i in CELE:
            rgb, ram = KL.obraz_manifestu(rows[i])
            b0 = st.przebieg(rgb)
            z0 = b0['z'].cpu().numpy()
            zs = np.load(OUT / f'zb_m{i:04d}_{war}.npy')
            d = zs - z0
            cel_t = KL.cel_z_ramek(ram, st.dev)
            kom = []
            for k in range(11):
                zk = zs - (k / 10) * d
                with torch.no_grad():
                    m_lat = st.ocena(st.ogon(torch.from_numpy(zk)[None].to(st.dev), b0['r'][None]), ram, cel_t)
                img = silnik.generuj(tr, zk, kroki=KROKI, ziarno=ZIARNO)
                x = cv2.resize((img.transpose(1, 2, 0) * 255).round().astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(kat / f'm{i:04d}_{war}_k{k:02d}.png'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
                b = st.przebieg(x)
                m = st.ocena(b['o'][None], ram, cel_t)
                dz = (b['z'].cpu().numpy() - z0).reshape(-1); df = d.reshape(-1)
                rek = dict(cel=i, wariant=war, k=k, ulamek_przyrostu=1 - k / 10,
                           latent_iou=m_lat['iou_top'], latent_pewnosc=m_lat['pewnosc_top'],
                           sd_iou=m['iou_top'], sd_pewnosc=m['pewnosc_top'], sd_strata=m['strata'], sd_fp=m['fp'],
                           sd_ramka=m['ramka_top'],
                           cos_do_przyrostu=float(dz @ df / (np.linalg.norm(dz) * np.linalg.norm(df) + 1e-12)),
                           odtworzony_ulamek=float(dz @ df / (df @ df)),
                           odl_od_oryginalu=float(np.linalg.norm(dz) / np.linalg.norm(z0)))
                wyn.append(rek); kom.append((x, rek))
                log(war, i, k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in rek.items() if a not in ('sd_ramka', 'cel', 'wariant', 'k')})
            # pasek 11 obrazów z ramkami: cyjan = etykieta, zielony = najpewniejsza ramka ViT
            pas = []
            for x, rek in kom:
                y = cv2.cvtColor(x.copy(), cv2.COLOR_RGB2BGR)
                g = ram[0].astype(int); cv2.rectangle(y, tuple(g[:2]), tuple(g[2:]), (255, 255, 0), 2)
                t = np.array(rek['sd_ramka']).astype(int); cv2.rectangle(y, tuple(t[:2]), tuple(t[2:]), (0, 255, 0), 2)
                cv2.rectangle(y, (0, 0), (480, 46), (0, 0, 0), -1)
                cv2.putText(y, f'{int(round(100 * rek["ulamek_przyrostu"]))}% przyrostu  (latent: IoU {rek["latent_iou"]:.2f})',
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.putText(y, f'SD: IoU {rek["sd_iou"]:.2f} pewnosc {rek["sd_pewnosc"]:.2f}', (6, 39),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
                pas.append(cv2.resize(y, (320, 213), interpolation=cv2.INTER_AREA))
            cv2.imwrite(str(kat / f'PASEK_m{i:04d}_{war}.jpg'), np.concatenate(pas, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
            (OUT / 'ETAP10_INTERPOLACJA.json').write_text(json.dumps(dict(wyniki=wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    pods = {}
    for war in WARIANTY:
        for k in range(11):
            w = [x for x in wyn if x['wariant'] == war and x['k'] == k]
            pods[f'{war}_k{k:02d}'] = {a: float(np.mean([x[a] if x[a] is not None else np.nan for x in w])) for a in
                                       ('latent_iou', 'latent_pewnosc', 'sd_iou', 'sd_pewnosc', 'sd_strata', 'cos_do_przyrostu',
                                        'odtworzony_ulamek', 'odl_od_oryginalu')}
            pods[f'{war}_k{k:02d}']['sd_iou_powyzej_05'] = int(sum((x['sd_iou'] or 0) >= .5 for x in w))
    (OUT / 'ETAP10_INTERPOLACJA.json').write_text(json.dumps(dict(wyniki=wyn, podsumowanie=pods, opis=__doc__), indent=1,
                                                             ensure_ascii=False), encoding='utf-8')
    for a, v in pods.items():
        log(a, {b: round(c, 3) for b, c in v.items()})
    log('KONIEC')


if __name__ == '__main__':
    main()
