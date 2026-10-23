"""Wspólne dla pomiarów replikacji: model MałyViT, 46 checkpointów seedu 3047, wybór i wczytanie klatek TEST z obrysem cystotomu,
jedno przejście w przód z przechwyceniem 13 stanów, wag uwagi i jasności laserów."""
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
sys.path.insert(0, str(BAZA / 'LaserLatentSpaceApp'))
from laser_app.compute import tool_labels as TL  # noqa: E402

WAGI = BAZA / 'experiments/skae_replikacja_mapy_vit3047_20261022__17-14-27/Wyniki/Dane/Wejscia/wagi'
KOD_MODELU = BAZA / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py'
SPLIT = BAZA / 'outputs/kompensacja_funkcjonalna_dgx2_20261017/download/unpacked_20261018_035119/input/split_manifest.json'
MANIFEST_UCZACY = BAZA / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
NA_EPOKE = 35000 / 32
T, C, H, W = 600, 192, 20, 30
KATEGORIA = 'cystotome'
PROG_NARZ = 0.25
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_M = None


def modul():
    global _M
    if _M is None:
        spec = importlib.util.spec_from_file_location('maly_vit_35000_model', KOD_MODELU)
        _M = importlib.util.module_from_spec(spec); spec.loader.exec_module(_M)
    return _M


def siec():
    return modul().MalyViT().to(dev).eval().requires_grad_(False)


def checkpointy():
    return sorted(int(re.fullmatch(r'k(\d+)\.pt', p.name)[1]) for p in WAGI.glob('k*.pt') if re.fullmatch(r'k(\d+)\.pt', p.name))


def sciezka(krok):
    return WAGI / f'k{krok:06d}.pt'


def wczytaj(net, krok):
    st = torch.load(sciezka(krok), map_location=dev, weights_only=False)      # własne checkpointy (zawierają obiekty numpy)
    net.load_state_dict(st.get('net', st), strict=True)
    return net


def filmy_uczace():
    return set(re.findall(r'TR_\d+_S\d+_P\d+', MANIFEST_UCZACY.read_text(encoding='utf-8')))


def filmy_test():
    g = json.loads(SPLIT.read_text(encoding='utf-8'))['groups']; tren = filmy_uczace()
    return sorted(f for f, v in g.items() if v == 'test' and f not in tren)


def etykieta_cystotomu(film, klatka):
    """udział (600,) pikseli obrysu cystotomu w łacie, ramka xyxy (480 × 320) albo None, gdy brak cystotomu."""
    w = TL._etykiety_filmu(film).get(int(klatka))
    if w is None or KATEGORIA not in TL.kategorie(w):
        return None
    wiel = TL._wielokaty(w, KATEGORIA)
    if not wiel:
        return None
    b = TL._json(w[KATEGORIA + '_bbox'])
    ramka = [b[0] * TL.SKALA[0], b[1] * TL.SKALA[1], (b[0] + b[2]) * TL.SKALA[0], (b[1] + b[3]) * TL.SKALA[1]]
    return dict(udzial=TL.udzial_z_wielokatow(wiel), gt=[float(v) for v in ramka])


def wybierz_klatki(n=10, min_lat=8, cel=20):
    """Z każdego filmu TEST (25 filmów podziału kampanii, rozłącznych z treningiem) klatka z cystotomem o liczbie łat (udział ≥ PROG_NARZ)
    ≥ min_lat i najbliższej `cel`; n filmów o liczbie łat najbliższej `cel`."""
    out = []
    for f in filmy_test():
        z = TL.ADNOTACJE / f'{f}.zip'
        if not z.exists():
            continue
        et = TL.etykiety(f)
        with zipfile.ZipFile(z) as zz:
            n_jpg = sum(x.endswith('.jpg') for x in zz.namelist())
        if len(et) != n_jpg or (et and max(et) != n_jpg):
            print('pomijam', f, 'adnotacje', len(et), 'obrazy', n_jpg, flush=True)
            continue
        best = None
        for kl in sorted(et):
            e = etykieta_cystotomu(f, kl)
            if e is None or e['gt'][2] - e['gt'][0] < 16 or e['gt'][3] - e['gt'][1] < 8:   # zdegenerowana ramka GT (błąd adnotacji)
                continue
            k = int((e['udzial'] >= PROG_NARZ).sum())
            if k >= min_lat and (best is None or abs(k - cel) < abs(best[0] - cel)):
                best = (k, kl, e)
        if best:
            out.append(dict(film=f, klatka=int(best[1]), lat_narz=best[0], lat_dotk=int((best[2]['udzial'] > 0).sum()), gt=best[2]['gt']))
    out.sort(key=lambda r: (abs(r['lat_narz'] - cel), r['film']))
    return out[:n]


def obraz(film, klatka):
    with zipfile.ZipFile(TL.ADNOTACJE / f'{film}.zip') as z:
        raw = z.read(f'{film}/{film}_{klatka:06d}.jpg')
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (W * 16, H * 16), interpolation=cv2.INTER_AREA), hashlib.sha256(raw).hexdigest()


def wejscie(rgb):
    M = modul()
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
    return (x - M.MEAN.to(dev)) / M.STD.to(dev)


def przejscie(net, rgb):
    """13 stanów X (13, 600, 192), wagi uwagi A (6, 3, 600, 600), jasności Z (6, 600, 768), ramka xyxy i pewność; tensory na dev."""
    M = modul(); x = wejscie(rgb)
    if next(net.parameters()).dtype == torch.float64:
        x = x.double()
    with torch.no_grad():
        f = x
        for mod in net.lodyga:
            f = mod(f)
        _, Cc, h, w = f.shape
        t = f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, Cc, dev).to(f.dtype)
        X, A, Z = [t[0]], [], []
        for blk in net.bloki:
            at = blk.attn; xn = blk.norm1(t); nh = at.num_heads; hd = C // nh
            qkv = at.qkv(xn).reshape(1, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
            q, k, v = at.q_norm(qkv[0]), at.k_norm(qkv[1]), qkv[2]
            A.append(((q * at.scale) @ k.transpose(-2, -1)).softmax(-1)[0])
            t = t + blk.ls1(at(xn)); X.append(t[0])
            pam = {}; hak = blk.mlp.act.register_forward_hook(lambda m_, a_, o_: pam.__setitem__('z', o_))
            t = t + blk.ls2(blk.mlp(blk.norm2(t))); hak.remove()
            X.append(t[0]); Z.append(pam['z'][0])
        o = net.glowa(net.norm(t)).view(1, h, w, 5)
        bx, sc = M.dekoduj(o.float().cpu(), k=1)
    return dict(X=torch.stack(X), A=torch.stack(A), Z=torch.stack(Z), ramka=bx[0, 0].numpy().astype(float), pewnosc=float(sc[0, 0]))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    w = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    pa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); pb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(w / max(pa + pb - w, 1e-9))
