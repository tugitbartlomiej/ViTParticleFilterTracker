"""Powtórzenie etapów 8, 10 i 11 na parze ViT K8000 + SD K8000, ale z celami ze zbioru treningowego (zlecenie
użytkownika 2026-10-08): pula 35 000 klatek TRAIN (outputs/vit_dino_faiss_35000, każda z etykietą cystotomu).

Etap 0: K8000 na wszystkich klatkach puli; cele = 8 klatek z IoU najpewniejszej ramki 0 i najniższą pewnością, każda
z innego filmu (ta sama zasada co etap 1: najtrudniejsze dla detektora).
Etap 8: przedziały kanałów [min, max] z 400 losowych klatek puli (100 tokenów z klatki), przecięte z pudłem wag K8000;
korekta celem detekcja na 768 kanałach, 300 kroków Adama.
Etap 10: 5 obrazów SD K8000 na cel: 100, 75, 50, 25 i 0 % przyrostu (z = z_sztuczny − u δ), od szumu, DDIM 50 leading,
ziarno 0. Bez detekcji na obrazach SD (służą tylko jako zapytania).
Etap 11: 5 najbliższych prawdziwych klatek puli (DINOv2 CLS) dla każdego obrazu SD; sama klatka celu pominięta.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/latent_cel_do_obrazow/kod/k8000_trening_trudne.py [0] [8] [10] [11]
"""
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow' / 'k8000_trening'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
SD = ROOT / 'assets' / 'SD_ViT_K8000_A_control_K10500'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import korekta_latentu as KL
from model_snapshot import dekoduj

N_CEL = 8
ULAMKI = [1.0, .75, .5, .25, 0.0]                  # ułamek przyrostu w obrazie SD
N_TOP = 5
KROKI_SD, ZIARNO = 50, 0
ETAPY = sys.argv[1:] or ['0', '8', '10', '11']
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)
zapisz = lambda nazwa, d: (OUT / nazwa).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
cos = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def ramki(m):
    w, h = m['width'], m['height']
    return np.asarray([[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
                        (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m['gt']], np.float32).reshape(-1, 4)


def klatka(j, zf=None):
    m = MAN[j]
    if zf is None:
        with zipfile.ZipFile(m['archive']) as z:
            b = z.read(m['member'])
    else:
        b = zf.read(m['member'])
    im = np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def iou_max(box, R):
    x0 = np.maximum(box[0], R[:, 0]); y0 = np.maximum(box[1], R[:, 1]); x1 = np.minimum(box[2], R[:, 2]); y1 = np.minimum(box[3], R[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1]); b = (R[:, 2] - R[:, 0]) * (R[:, 3] - R[:, 1])
    return float((inter / np.maximum(a + b - inter, 1e-9)).max())


def etap0(st):
    """K8000 na całej puli: IoU najpewniejszej ramki z najlepszą etykietą i pewność."""
    po_arch = defaultdict(list)
    for j, m in enumerate(MAN):
        po_arch[m['archive']].append(j)
    iou = np.zeros(len(MAN), np.float32); pew = np.zeros(len(MAN), np.float32)
    n = 0
    for arch, js in po_arch.items():
        with zipfile.ZipFile(arch) as zf:
            for a in range(0, len(js), 64):
                b = js[a:a + 64]
                x = torch.from_numpy(np.stack([klatka(j, zf) for j in b])).permute(0, 3, 1, 2).float().to(st.dev) / 255
                with torch.no_grad():
                    o, _ = st.net(x)
                    box, sc = dekoduj(o, k=1)
                box = box[:, 0].float().cpu().numpy(); sc = sc[:, 0].float().cpu().numpy()
                for t, j in enumerate(b):
                    iou[j] = iou_max(box[t], ramki(MAN[j])); pew[j] = sc[t]
        n += len(js)
        log('etap 0', n, '/', len(MAN), 'IoU śr dotąd', round(float(iou[:n].mean()), 3) if n else 0)
    np.save(OUT / 'POOL_IOU_PEWNOSC_K8000.npy', np.stack([iou, pew]))
    kolej = np.lexsort((pew, iou))                                     # najpierw najmniejsze IoU, potem pewność
    cele, filmy = [], set()
    for j in kolej:
        if MAN[j]['film'] in filmy:
            continue
        cele.append(dict(j=int(j), film=MAN[j]['film'], klatka=int(MAN[j]['frame']), iou=float(iou[j]), pewnosc=float(pew[j])))
        filmy.add(MAN[j]['film'])
        if len(cele) == N_CEL:
            break
    rap = dict(vit=str(VIT), n=len(MAN), iou_sr=float(iou.mean()), iou_powyzej_05=int((iou >= .5).sum()),
               iou_ponizej_01=int((iou < .1).sum()), iou_zero=int((iou == 0).sum()), cele=cele)
    zapisz('ETAP0_CELE_TRENING.json', rap)
    log('etap 0', {a: b for a, b in rap.items() if a not in ('cele', 'vit')})
    log('cele', [(c['film'][3:7], c['klatka'], round(c['pewnosc'], 2)) for c in cele])


def etap8(st):
    cele = json.loads((OUT / 'ETAP0_CELE_TRENING.json').read_text(encoding='utf-8'))['cele']
    rng = np.random.default_rng(0)
    Tk = []
    for j in rng.choice(len(MAN), 400, replace=False):
        Tk.append(st.przebieg(klatka(int(j)))['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)])
    Tk = torch.cat(Tk)
    mu = Tk.mean(0); _, S_, Vt = torch.linalg.svd(Tk - mu, full_matrices=False)
    U = Vt[:64].T; s = S_[:64] / (len(Tk) - 1) ** .5

    def nienat(z):
        y = z - mu; p = y @ U
        return dict(poza=float(((y - p @ U.T) ** 2).sum(1).sqrt().mean()), maha=float(((p / s) ** 2).sum(1).sqrt().mean()))
    lo, hi = Tk.min(0).values, Tk.max(0).values
    lo_ck, hi_ck = st.h_min.clone(), st.h_max.clone()
    rap = dict(prawdziwe=nienat(Tk), szerokosc_minmax=float((hi - lo).mean()), szerokosc_ckpt=float((hi_ck - lo_ck).mean()), wyniki=[])
    log('szerokość kanału: checkpoint', round(rap['szerokosc_ckpt'], 2), 'minmax puli', round(rap['szerokosc_minmax'], 3))
    st.h_min, st.h_max = torch.minimum(lo, lo_ck), torch.maximum(torch.minimum(hi, hi_ck), lo + 1e-6)
    for c in cele:
        rgb, ram = klatka(c['j']), ramki(MAN[c['j']])
        z0 = st.przebieg(rgb)['z']
        spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='brak',
                    kroki=300, lr=.05, zapis_co=300, ziarno=0)
        w = KL.oblicz_korekte(st, rgb, ram, spec, dict(id=f'trening_j{c["j"]:05d}_minmax'))
        kn = np.load(Path(w['kat']) / 'KONIEC.npz')
        zn = torch.from_numpy(kn['z_baza'] + kn['d_najlepszy']).to(st.dev)
        np.save(OUT / f'zb_j{c["j"]:05d}_minmax.npy', zn.cpu().numpy().astype(np.float32))
        rek = dict(j=c['j'], iou=(w['przed']['iou_top'], w['najlepszy']['iou_top']), pewnosc=(w['przed']['pewnosc_top'], w['najlepszy']['pewnosc_top']),
                   dz_wzgl=float((zn - z0).norm() / z0.norm()), nienat_przed=nienat(z0), nienat_po=nienat(zn))
        rap['wyniki'].append(rek)
        log('etap 8', c['j'], 'IoU', [round(x or 0, 3) for x in rek['iou']], 'pewn', [round(x, 3) for x in rek['pewnosc']],
            'poza', round(rek['nienat_po']['poza'], 1), '||dz||/||z||', round(rek['dz_wzgl'], 2))
        zapisz('ETAP8_TRENING.json', rap)
    st.h_min, st.h_max = lo_ck, hi_ck
    w = rap['wyniki']
    rap['podsumowanie'] = dict(iou_po=float(np.mean([x['iou'][1] or 0 for x in w])), pewnosc_po=float(np.mean([x['pewnosc'][1] for x in w])),
                               iou_powyzej_05=int(sum((x['iou'][1] or 0) >= .5 for x in w)), poza=float(np.mean([x['nienat_po']['poza'] for x in w])),
                               dz_wzgl=float(np.mean([x['dz_wzgl'] for x in w])), n=len(w))
    zapisz('ETAP8_TRENING.json', rap)
    log('etap 8 podsumowanie', {a: round(b, 3) for a, b in rap['podsumowanie'].items()})


def zaladuj_sd(dev):
    # pełna ścieżka: silnik (import korekta_latentu) dokłada katalog starego SD z własnym model.py
    import importlib.util
    from diffusers import DDIMScheduler
    sp = importlib.util.spec_from_file_location('model_sd_k8000', SD / 'kod' / 'model.py')
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    ck = torch.load(SD / 'best_sd.pt', map_location='cpu', weights_only=False)
    net = M.Denoiser().eval(); vae = M.new_vae().eval()
    net.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    net.to(dev).requires_grad_(False); vae.to(dev).requires_grad_(False)
    return net, vae, float(ck['scale']), DDIMScheduler.from_config(M.schedule().config, timestep_spacing='leading')


@torch.no_grad()
def generuj(net, vae, skala, sch, z, dev):
    sch.set_timesteps(KROKI_SD, device=dev)
    g = torch.Generator(device=dev).manual_seed(ZIARNO)
    h = torch.randn((1, 4, 48, 72), generator=g, device=dev, dtype=torch.float32) * sch.init_noise_sigma
    war = torch.as_tensor(z, dtype=torch.float32, device=dev)[None]
    for t in sch.timesteps:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = net(sch.scale_model_input(h, t), t, war)
        h = sch.step(e.float(), t, h, eta=0.0).prev_sample
    return ((vae.decode(h / skala).sample.float() + 1) / 2).clamp(0, 1)


def etap10(st):
    cele = json.loads((OUT / 'ETAP0_CELE_TRENING.json').read_text(encoding='utf-8'))['cele']
    net, vae, skala, sch = zaladuj_sd(st.dev)
    kat = OUT / 'sd'; kat.mkdir(exist_ok=True)
    wyn = []
    for c in cele:
        j = c['j']; rgb = klatka(j)
        z0 = st.przebieg(rgb)['z'].cpu().numpy()
        zs = np.load(OUT / f'zb_j{j:05d}_minmax.npy'); d = zs - z0
        for u in ULAMKI:
            zk = z0 + u * d
            x = generuj(net, vae, skala, sch, zk, st.dev)
            duzy = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(kat / f'j{j:05d}_u{int(round(100 * u)):03d}.png'), cv2.cvtColor(duzy, cv2.COLOR_RGB2BGR))
            maly = (F.interpolate(x, (320, 480), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            zsd = st.przebieg(maly)['z'].cpu().numpy(); dz = zsd - z0
            rek = dict(j=j, ulamek=u, cos_do_warunku=cos(zsd, zk), cos_do_przyrostu=cos(dz, d),
                       odtworzony_ulamek=float((dz * d).sum() / (d * d).sum()))
            wyn.append(rek)
            log('etap 10', j, f'{int(100 * u)}%', {a: round(v, 3) for a, v in rek.items() if isinstance(v, float) and a != 'ulamek'})
        zapisz('ETAP10_TRENING.json', dict(wyniki=wyn))
    pods = {f'{int(100 * u)}%': {a: float(np.mean([x[a] for x in wyn if x['ulamek'] == u])) for a in ('cos_do_warunku', 'cos_do_przyrostu', 'odtworzony_ulamek')}
            for u in ULAMKI}
    zapisz('ETAP10_TRENING.json', dict(wyniki=wyn, podsumowanie=pods, opis=__doc__))
    log('etap 10 podsumowanie', pods)
    del net, vae; torch.cuda.empty_cache()


def etap11():
    cele = json.loads((OUT / 'ETAP0_CELE_TRENING.json').read_text(encoding='utf-8'))['cele']
    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12

    @torch.inference_mode()
    def cls(rgb):
        f = model.forward_features(PRE(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).cuda())['x_norm_clstoken'][0].float().cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-12)

    def kafel(im, tekst, kolor=(255, 255, 255)):
        y = cv2.resize(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR), (480, 320))
        cv2.rectangle(y, (0, 0), (480, 30), (0, 0, 0), -1); cv2.putText(y, tekst, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, kolor, 2)
        return cv2.resize(y, (288, 192))
    pusty = np.zeros((192, 288, 3), np.uint8)
    kat = OUT / 'szukanie'; kat.mkdir(exist_ok=True)
    wyn = []
    for c in cele:
        j0 = c['j']; rgb = klatka(j0); film = MAN[j0]['film']
        q0 = cls(rgb)
        kol = [[kafel(rgb, f'cel {film[3:7]}/{MAN[j0]["frame"]}')] + [pusty] * N_TOP]
        for u in ULAMKI:
            sd = cv2.cvtColor(cv2.imread(str(OUT / 'sd' / f'j{j0:05d}_u{int(round(100 * u)):03d}.png')), cv2.COLOR_BGR2RGB)
            q = cls(sd); s = Fp @ q; s[j0] = -9
            top = np.argsort(-s)[:N_TOP]
            rek = dict(cel=j0, ulamek=u, film_celu=film, cos_do_zdjecia_celu=float(q @ q0),
                       top=[dict(j=int(j), film=MAN[j]['film'], klatka=int(MAN[j]['frame']), cos=float(s[j])) for j in top])
            wyn.append(rek)
            k = [kafel(sd, f'SD {int(100 * u)}% przyrostu')]
            for j in top:
                k.append(kafel(klatka(int(j)), f'{MAN[j]["film"][3:7]}/{MAN[j]["frame"]} cos {s[j]:.2f}',
                               (0, 255, 0) if MAN[j]['film'] == film else (0, 200, 255)))
            kol.append(k)
            log('etap 11', j0, f'{int(100 * u)}%', [(t['film'][3:7], t['klatka'], round(t['cos'], 2)) for t in rek['top']])
        siatka = np.concatenate([np.concatenate(k, 0) for k in kol], 1)
        cv2.imwrite(str(kat / f'SZUKANIE_j{j0:05d}.jpg'), siatka, [cv2.IMWRITE_JPEG_QUALITY, 85])
        zapisz('ETAP11_TRENING.json', dict(wyniki=wyn))
    wsz = [t['j'] for x in wyn for t in x['top']]
    pods = dict(unikalne_wsrod_wszystkich=len(set(wsz)), wszystkich=len(wsz),
                unikalne_na_cel={str(c['j']): len({t['j'] for x in wyn if x['cel'] == c['j'] for t in x['top']}) for c in cele},
                z_filmu_celu=int(sum(t['film'] == x['film_celu'] for x in wyn for t in x['top'])),
                cos_do_zdjecia_celu={f'{int(100 * u)}%': float(np.mean([x['cos_do_zdjecia_celu'] for x in wyn if x['ulamek'] == u])) for u in ULAMKI},
                najczestsze=sorted(((wsz.count(a), a) for a in set(wsz)), reverse=True)[:5])
    zapisz('ETAP11_TRENING.json', dict(wyniki=wyn, podsumowanie=pods, opis=__doc__))
    log('etap 11 podsumowanie', pods)


def main():
    st = KL.Stanowisko(VIT)
    assert st.sha == '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b', st.sha
    if '0' in ETAPY:
        etap0(st)
    if '8' in ETAPY:
        etap8(st)
    if '10' in ETAPY:
        etap10(st)
    if '11' in ETAPY:
        del st; torch.cuda.empty_cache()
        etap11()
    log('KONIEC')


if __name__ == '__main__':
    main()
