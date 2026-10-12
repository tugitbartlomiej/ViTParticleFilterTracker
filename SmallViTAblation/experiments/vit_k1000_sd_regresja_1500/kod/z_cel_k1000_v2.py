"""Wersja 2 (eksperyment vit_k1000_sd_regresja_1500): z* K1000 z wersji 1 (300 kroków) dał pewność 0,055 przy IoU 0,70,
a z* K8000 tego samego zdjęcia 0,81 przy IoU 1,00. Tu ta sama procedura, ale na GPU i dłużej: KROKI kroków Adama (argument,
domyślnie 3000), reszta bez zmian. Zapis pod nową nazwą zb_j10389_minmax_k1000_v2_<kroki>.npy w outputs/vit_k1000_sd_regresja_1500/z_cel/.

Opis wersji 1:
Sztuczny latent z* zdjęcia celu 0055/4019 (j = 10389) w przestrzeni K1000, ta sama procedura co etap 8
k8000_trening_trudne.py, tylko checkpoint K1000 (assets/MalyViT_K1000_SD_conditioning): przedziały kanałów [min, max]
ze 100 tokenów 400 losowych klatek puli vit_dino_faiss_35000 (rng 0) przecięte z pudłem wag; korekta celem detekcja na
768 kanałach, 300 kroków Adama lr 0,05, pewność celu 0,95. Liczone na CPU (lokalne GPU zajęte innym biegiem).
Zapis: outputs/vit_k1000_podmiana150_sd/z_cel_k1000/zb_j10389_minmax_k1000.npy, Z_CEL_K1000.json
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_podmiana150_sd/kod/z_cel_k1000.py
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'vit_k1000_sd_regresja_1500' / 'z_cel'
KROKI = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
PUDLO = sys.argv[2] if len(sys.argv) > 2 else 'minmax'   # 'minmax' (etap 8: min-max puli przecięte z pudłem wag) albo 'wagi' (diagnostyka: samo pudło z wag)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
VIT = ROOT / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CEL_J = 10389
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
from PIL import Image

import korekta_latentu as KL

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)
MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def ramki(m):
    w, h = m['width'], m['height']
    return np.asarray([[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
                        (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m['gt']], np.float32).reshape(-1, 4)


def klatka(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    st = KL.Stanowisko(VIT)
    assert st.sha == '3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e' and st.krok == 1000
    rng = np.random.default_rng(0)
    Tk = []
    for j in rng.choice(len(MAN), 400, replace=False):
        Tk.append(st.przebieg(klatka(int(j)))['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)])
    Tk = torch.cat(Tk)
    log('tokeny puli', tuple(Tk.shape))
    mu = Tk.mean(0); _, S_, Vt = torch.linalg.svd(Tk - mu, full_matrices=False)
    U = Vt[:64].T

    def poza(z):
        y = z - mu
        return float((y - (y @ U) @ U.T).norm(dim=1).mean())
    lo, hi = Tk.min(0).values, Tk.max(0).values
    lo_ck, hi_ck = st.h_min.clone(), st.h_max.clone()
    if PUDLO == 'minmax':
        st.h_min, st.h_max = torch.minimum(lo, lo_ck), torch.maximum(torch.minimum(hi, hi_ck), lo + 1e-6)
    rgb, ram = klatka(CEL_J), ramki(MAN[CEL_J])
    z0 = st.przebieg(rgb)['z']
    spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='brak',
                kroki=KROKI, lr=.05, zapis_co=KROKI, ziarno=0)
    w = KL.oblicz_korekte(st, rgb, ram, spec, dict(id=f'k1000_j{CEL_J:05d}_{PUDLO}_v2_{KROKI}'))
    kn = np.load(Path(w['kat']) / 'KONIEC.npz')
    zn = torch.from_numpy(kn['z_baza'] + kn['d_najlepszy']).to(st.dev)
    np.save(OUT / f'zb_j{CEL_J:05d}_{PUDLO}_k1000_v2_{KROKI}.npy', zn.cpu().numpy().astype(np.float32))
    np.save(OUT / f'z0_j{CEL_J:05d}_k1000.npy', z0.cpu().numpy().astype(np.float32))
    rap = dict(checkpoint=str(VIT), sha=st.sha, j=CEL_J, film=MAN[CEL_J]['film'], klatka=MAN[CEL_J]['frame'],
               iou=(w['przed']['iou_top'], w['najlepszy']['iou_top']), pewnosc=(w['przed']['pewnosc_top'], w['najlepszy']['pewnosc_top']),
               dz_wzgl=float((zn - z0).norm() / z0.norm()), poza_przed=poza(z0), poza_po=poza(zn), poza_prawdziwe=poza(Tk),
               spec={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in spec.items() if k != 'kanaly'}, czas_s=time.time() - T0)
    (OUT / f'Z_CEL_K1000_{PUDLO}_v2_{KROKI}.json').write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    log('KONIEC', rap)


if __name__ == '__main__':
    main()
