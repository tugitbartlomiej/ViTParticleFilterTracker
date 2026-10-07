"""Kandydaci do pomiaru pchnięcia: obrazy z etykietami w jednym formacie (obraz uint8 320 x 480 x 3, ramki xyxy w
pikselach 480 x 320, źródło, opis etykiety). Źródła łańcucha „sztuczny latent -> SD -> prawdziwe obrazy”:

- baza: klatki bazy uczenia (1500, etykiety zbioru);
- odszukane_sd: cztery paczki po 32 prawdziwe klatki odszukane dla czterech obrazów SD (MSE latentu, 88 469 obrazów
  train; outputs/vit_langevin_cztery_paczki_sd, etykiety zbioru) i random32 jako kontrola;
- sd: obrazy SD z pętli 4.3 (próba s47: kotwice z układem klatki sondy i obrazy-zapytania DINO); bez etykiety
  zbioru, więc ramka z detekcji wag końcowych biegu (pseudo-etykieta, zaznaczona w opisie);
- dino_sd: prawdziwe klatki znalezione przez DINOv2 dla obrazów-zapytań SD (top 10 w pętli 4.3, etykiety zbioru);
- konwerter_sd: klatki najbliższe obrazom SD przez konwerter latentu ViT k1000 -> cechy DINO (KONWERTER.pt v2) i bazę
  35 000 cech DINO (outputs/vit_dino_faiss_35000), odczyt z archiwów ZIP Cataract-LMM (etykiety zbioru).
"""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
R = APP.parent
CZTERY = R / 'outputs' / 'vit_langevin_cztery_paczki_sd'
PETLA = R / 'experiments' / 'maly_vit_petla_sd_rownanie' / 'wyniki' / 'proba43_s47'
PULA = R / 'experiments' / 'koopman_fastgan_petla' / 'local_inputs' / 'data' / 'images'
DINO = R / 'outputs' / 'vit_dino_faiss_35000'
KONW = R / 'outputs' / 'vit_konwerter_dino_filmy_eden' / 'v2' / 'KONWERTER.pt'
KONW_KOD = R / 'experiments' / 'vit_konwerter_dino_filmy_eden' / 'model.py'
VIT_K1000 = R / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt'

ZRODLA = {'baza': 'baza uczenia (losowe N)', 'odszukane_sd': 'prawdziwe klatki odszukane dla SD (cztery paczki)',
          'sd': 'obrazy SD (pętla 4.3, s47)', 'dino_sd': 'klatki DINO dla obrazów SD (top 10)',
          'konwerter_sd': 'klatki przez konwerter latent -> DINO dla obrazów SD'}


def _480(img):
    import cv2
    img = np.asarray(img)
    if img.shape[:2] == (320, 480):
        return np.ascontiguousarray(img[..., :3])
    return np.ascontiguousarray(cv2.resize(img[..., :3], (480, 320), interpolation=cv2.INTER_AREA))


def _png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert('RGB'))


def _xywh_480(bb, wh):
    sx, sy = 480 / wh[0], 320 / wh[1]
    x, y, w, h = bb
    return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]


def wpis(zrodlo, nazwa, obraz, ramki, etykieta):
    return dict(zrodlo=zrodlo, nazwa=nazwa, obraz=_480(obraz), ramki=[list(map(float, r)) for r in ramki], etykieta=etykieta)


def baza(od, n=64, ziarno=0):
    idx = np.random.default_rng(ziarno).choice(len(od.baza), min(n, len(od.baza)), replace=False)
    return [wpis('baza', f'baza {i}', od.baza.X[i], od.baza.wiersze[i]['ramki'], 'zbiór') for i in sorted(idx)]


def odszukane_sd(n_na_paczke=8):
    out = []
    for arm in ('2026100511', '2026100523', '2026100537', '2026100549', 'random32'):
        f = CZTERY / f'IMAGES_{arm}.npy'
        if not f.exists():
            continue
        X = np.load(f, mmap_mode='r')
        gt = json.loads((CZTERY / f'GROUNDTRUTH_{arm}.json').read_text(encoding='utf-8'))
        for j in range(min(n_na_paczke, len(X))):
            r = gt[j] if j < len(gt) else {}
            ramki = r.get('boxes_xyxy_480x320') or []
            out.append(wpis('odszukane_sd' if arm != 'random32' else 'odszukane_kontrola',
                            f'{"SD " + arm[-2:] if arm != "random32" else "losowa"} #{j + 1}', X[j], ramki, 'zbiór'))
    return out


def obrazy_sd():
    """Obrazy SD pętli 4.3 (kotwice z układem sondy i zapytania DINO): [(nazwa, obraz)] bez etykiet."""
    out = []
    for p in sorted((PETLA / 'kotwice').glob('kotwica_*.png')):
        out.append((f'SD kotwica {p.stem.split("_")[-1]}', _png(p)))
    for p in sorted(PETLA.glob('metoda/k*/zapytania/kotwica_*.png')):
        out.append((f'SD zapytanie {p.parts[-3]}/{p.stem}', _png(p)))
    return out


def sd_z_pseudoetykieta(detektor):
    """Obrazy SD z ramką z detekcji (najpewniejsza, gdy pewność > 0,3; inaczej obraz bez narzędzia)."""
    out = []
    for nazwa, img in obrazy_sd():
        x = _480(img)
        a = detektor.analizuj(x)
        ramki = [a['boxes'][0].tolist()] if float(a['scores'][0]) > .3 else []
        out.append(wpis('sd', nazwa, x, ramki, f'pseudo (detekcja {float(a["scores"][0]):.2f})'))
    return out


def dino_sd(n=4):
    """Prawdziwe klatki znalezione przez DINOv2 dla obrazów-zapytań SD (pętla 4.3)."""
    out = []
    for f in sorted(PETLA.glob('metoda/k*/zapytania/kotwica_*_zapytanie_top10.json')):
        t = json.loads(f.read_text(encoding='utf-8'))
        for c in t['candidates'][:n]:
            fid = c['frame_id'].split('/')
            p = PULA / f'{fid[-2]}_{fid[-1]}.jpg'
            if not p.exists():
                continue
            ramki = [_xywh_480(a['bbox'], c.get('image_size', [720, 480])) for a in c.get('annotations', [])]
            out.append(wpis('dino_sd', f'DINO {f.parts[-3]}/{f.stem.split("_zap")[0]} -> {fid[-2]}/{int(fid[-1])}',
                            _png(p), ramki, 'zbiór'))
    return out


def _konwerter():
    import sys
    spec = importlib.util.spec_from_file_location('konwerter_dino_model', KONW_KOD)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KONW_KOD.parent))                 # model.py konwertera importuje swój model_vit
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(KONW_KOD.parent))
    return mod.KonwerterGotowy(str(KONW), device='cuda')


def konwerter_sd(n=3):
    """Dla każdego obrazu SD: latent ViT k1000 -> konwerter -> cechy DINO CLS -> najbliższe klatki z bazy 35 000."""
    import torch
    import detekcja as D
    cechy = np.load(DINO / 'features.npy', mmap_mode='r')
    cechy = np.array(cechy, np.float32); cechy /= np.linalg.norm(cechy, axis=1, keepdims=True) + 1e-12
    manifest = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    det = D.Detektor(VIT_K1000)
    kon = _konwerter()
    out = []
    for nazwa, img in obrazy_sd():
        z = det.analizuj(_480(img))['z']
        z = (z[0] if z.ndim == 3 else z).float().cpu().numpy()
        with torch.no_grad():
            cls, _ = kon.predict(z[None])
        q = np.asarray(cls.cpu() if hasattr(cls, 'cpu') else cls, np.float32).reshape(-1)
        q /= np.linalg.norm(q) + 1e-12
        for j in np.argsort(-(cechy @ q))[:n]:
            m = manifest[j]
            try:
                with zipfile.ZipFile(m['archive']) as zf:
                    import io
                    from PIL import Image
                    obraz = np.asarray(Image.open(io.BytesIO(zf.read(m['member']))).convert('RGB'))
            except Exception:
                continue
            ramki = [_xywh_480(g['bbox_xywh'], (obraz.shape[1], obraz.shape[0])) for g in m.get('gt', [])]
            out.append(wpis('konwerter_sd', f'konwerter {nazwa.replace("SD ", "")} -> {m["film"]}/{m["frame"]}', obraz,
                            ramki, f'zbiór (cos DINO {float(cechy[j] @ q):.2f})'))
    return out
