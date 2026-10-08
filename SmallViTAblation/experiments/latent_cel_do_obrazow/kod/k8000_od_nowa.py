"""Etapy 8, 10 i 11 od nowa na zgodnej parze modeli (zlecenie użytkownika 2026-10-08): ViT K8000 (assets/MalyViT_K8000_s47_losowe)
i SD uczone na jego pełnym latencie (assets/SD_ViT_K8000_A_control_K10500, DDIM 50 kroków, spacing leading, eta 0).

Cele: 8 obrazów manifestu najtrudniejszych dla K8000 (najmniejsze IoU najpewniejszej ramki), ta sama zasada co etap 1
dla k1000; 4 z nich (102, 224, 270, 369) były celami k1000.
Etap 8: korekta celem detekcja na 768 kanałach, każda współrzędna w [min, max] kanału z 400 obrazów manifestu przy K8000
(100 losowych tokenów z obrazu), przecięte z pudłem z wag K8000; 300 kroków Adama.
Etap 10: z_k = z_sztuczny − (k/10) δ, δ = z_sztuczny − z_oryginał, k = 0..10; każdy z_k warunkiem SD K8000 od czystego
szumu (ziarno 0 dla wszystkich k). Miary: detekcja latentu z_k (końcówka ViT), latent obrazu SD wobec warunku i wobec δ.
Detekcji na obrazach SD nie liczymy: obrazy SD służą tylko jako zapytania do wyszukania prawdziwych zdjęć.
Etap 11: każdy obraz SD (i samo zdjęcie celu) jako zapytanie DINOv2 ViT-S/14 CLS do puli 35 000 klatek TRAIN; 5 najbliższych.
Uruchomienie z SmallViTAblation (diffusers jest tylko w 3.13): F:/Python/3.13/python.exe experiments/latent_cel_do_obrazow/kod/k8000_od_nowa.py [8] [10] [11]
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow' / 'k8000'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
SD = ROOT / 'assets' / 'SD_ViT_K8000_A_control_K10500'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
sys.path.insert(0, str(ROOT / 'app'))
sys.path.append(str(SD / 'kod'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import korekta_latentu as KL

N_CEL = 8
KROKI_SD, ZIARNO = 50, 0
KROKI_K = [0, 3, 5, 7, 9, 10]          # 100, 70, 50, 30, 10, 0 % przyrostu (pierwsze 4 cele policzone dla wszystkich 11)
ETAPY = sys.argv[1:] or ['8', '10', '11']
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)
zapisz = lambda nazwa, d: (OUT / nazwa).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
cos = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def wybierz_cele(st, rows):
    w = []
    for i in range(len(rows)):
        rgb, ram = KL.obraz_manifestu(rows[i])
        m = st.ocena(st.przebieg(rgb)['o'][None], ram, KL.cel_z_ramek(ram, st.dev))
        w.append((i, m['iou_top'] or 0.0, m['pewnosc_top']))
    w.sort(key=lambda x: (x[1], x[2]))
    return [dict(i=i, iou=a, pewnosc=b) for i, a, b in w[:N_CEL]], w


def etap8(st, rows):
    rng = np.random.default_rng(0)
    Tk = torch.cat([st.przebieg(KL.obraz_manifestu(rows[i])[0])['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)]
                    for i in range(len(rows))])
    mu = Tk.mean(0); _, S_, Vt = torch.linalg.svd(Tk - mu, full_matrices=False)
    U = Vt[:64].T; s = S_[:64] / (len(Tk) - 1) ** .5

    def nienat(z):
        y = z - mu; p = y @ U
        return dict(poza=float(((y - p @ U.T) ** 2).sum(1).sqrt().mean()), maha=float(((p / s) ** 2).sum(1).sqrt().mean()),
                    norma_tokenu_max=float(z.norm(dim=1).max()), norma_tokenu_sr=float(z.norm(dim=1).mean()))
    lo, hi = Tk.min(0).values, Tk.max(0).values
    lo_ck, hi_ck = st.h_min.clone(), st.h_max.clone()
    cele, wszystkie = wybierz_cele(st, rows)
    log('cele K8000', [(c['i'], round(c['iou'], 2), round(c['pewnosc'], 2)) for c in cele])
    rap = dict(vit=str(VIT), vit_sha=st.sha, cele=cele, iou_manifestu=dict(sr=float(np.mean([x[1] for x in wszystkie])),
               powyzej_05=int(sum(x[1] >= .5 for x in wszystkie)), n=len(wszystkie)),
               prawdziwe=dict(nienat=nienat(Tk)), szerokosc_minmax=float((hi - lo).mean()),
               szerokosc_ckpt=float((hi_ck - lo_ck).mean()), wyniki=[])
    log('szerokość kanału: checkpoint', round(rap['szerokosc_ckpt'], 2), 'minmax danych', round(rap['szerokosc_minmax'], 3))
    W2 = st.blok.mlp.fc2.weight.detach()
    st.h_min, st.h_max = torch.minimum(lo, lo_ck), torch.maximum(torch.minimum(hi, hi_ck), lo + 1e-6)
    for c in cele:
        rgb, ram = KL.obraz_manifestu(rows[c['i']])
        z0 = st.przebieg(rgb)['z']
        spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='brak',
                    kroki=300, lr=.05, zapis_co=300, ziarno=0)
        w = KL.oblicz_korekte(st, rgb, ram, spec, dict(id=f'k8000_m{c["i"]:04d}_minmax'))
        kn = np.load(Path(w['kat']) / 'KONIEC.npz')
        zn = torch.from_numpy(kn['z_baza'] + kn['d_najlepszy']).to(st.dev)
        np.save(OUT / f'zb_m{c["i"]:04d}_minmax.npy', zn.cpu().numpy().astype(np.float32))
        rek = dict(cel=c['i'], iou=(w['przed']['iou_top'], w['najlepszy']['iou_top']),
                   pewnosc=(w['przed']['pewnosc_top'], w['najlepszy']['pewnosc_top']), fp=w['najlepszy']['fp'],
                   strata=(w['przed']['strata'], w['najlepszy']['strata']), norma_dz=float((zn - z0).norm()),
                   norma_z=float(z0.norm()), W2_zmiana_wzgl=float(((zn - z0) @ W2.T).norm() / (z0 @ W2.T).norm()),
                   nienat_przed=nienat(z0), nienat_po=nienat(zn), niewidoczna=w['najlepszy']['fc2']['udzial_niewidocznej'])
        rap['wyniki'].append(rek)
        log('etap 8', c['i'], 'IoU', [round(x or 0, 3) for x in rek['iou']], 'pewn', [round(x, 3) for x in rek['pewnosc']],
            'poza', round(rek['nienat_po']['poza'], 2), '||dz||/||z||', round(rek['norma_dz'] / rek['norma_z'], 2))
        zapisz('ETAP8_K8000.json', rap)
    st.h_min, st.h_max = lo_ck, hi_ck
    w = rap['wyniki']
    rap['podsumowanie'] = dict(iou_przed=float(np.mean([x['iou'][0] or 0 for x in w])), iou_po=float(np.mean([x['iou'][1] or 0 for x in w])),
                               pewnosc_po=float(np.mean([x['pewnosc'][1] for x in w])), iou_powyzej_05=int(sum((x['iou'][1] or 0) >= .5 for x in w)),
                               poza=float(np.mean([x['nienat_po']['poza'] for x in w])), maha=float(np.mean([x['nienat_po']['maha'] for x in w])),
                               dz_wzgl=float(np.mean([x['norma_dz'] / x['norma_z'] for x in w])), n=len(w))
    rap['opis'] = __doc__
    zapisz('ETAP8_K8000.json', rap)
    log('etap 8 podsumowanie', {a: round(b, 3) for a, b in rap['podsumowanie'].items()})


def zaladuj_sd(dev):
    # pełna ścieżka: silnik (import korekta_latentu) dokłada katalog starego SD z własnym model.py
    import importlib.util
    sp = importlib.util.spec_from_file_location('model_sd_k8000', SD / 'kod' / 'model.py')
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    Denoiser, new_vae, schedule = M.Denoiser, M.new_vae, M.schedule
    from diffusers import DDIMScheduler
    ck = torch.load(SD / 'best_sd.pt', map_location='cpu', weights_only=False)
    net = Denoiser().eval(); vae = new_vae().eval()
    net.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    net.to(dev).requires_grad_(False); vae.to(dev).requires_grad_(False)
    sch = DDIMScheduler.from_config(schedule().config, timestep_spacing='leading')
    return net, vae, float(ck['scale']), sch


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
    x = ((vae.decode(h / skala).sample.float() + 1) / 2).clamp(0, 1)
    return x                                                              # [1, 3, 384, 576]


def etap10(st, rows):
    cele = json.loads((OUT / 'ETAP8_K8000.json').read_text(encoding='utf-8'))['cele']
    net, vae, skala, sch = zaladuj_sd(st.dev)
    kat = OUT / 'sd_interpolacja'; kat.mkdir(exist_ok=True)
    # wznowienie: cele z gotowym paskiem zostają (pierwsze 4 mają 11 kroków); reszta tylko KROKI_K (prośba użytkownika)
    stare = json.loads((OUT / 'ETAP10_K8000.json').read_text(encoding='utf-8'))['wyniki'] if (OUT / 'ETAP10_K8000.json').exists() else []
    gotowe = {c['i'] for c in cele if (kat / f'PASEK_m{c["i"]:04d}.jpg').exists()}
    wyn = [x for x in stare if x['cel'] in gotowe]
    for c in cele:
        i = c['i']
        if i in gotowe:
            continue
        rgb, ram = KL.obraz_manifestu(rows[i])
        b0 = st.przebieg(rgb); z0 = b0['z'].cpu().numpy()
        zs = np.load(OUT / f'zb_m{i:04d}_minmax.npy'); d = zs - z0
        cel_t = KL.cel_z_ramek(ram, st.dev)
        pas = []
        for k in KROKI_K:
            zk = zs - (k / 10) * d
            with torch.no_grad():
                m_lat = st.ocena(st.ogon(torch.from_numpy(zk)[None].to(st.dev), b0['r'][None]), ram, cel_t)
            x = generuj(net, vae, skala, sch, zk, st.dev)
            duzy = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(kat / f'm{i:04d}_k{k:02d}.png'), cv2.cvtColor(duzy, cv2.COLOR_RGB2BGR))
            maly = (F.interpolate(x, (320, 480), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            zsd = st.przebieg(maly)['z'].cpu().numpy()
            dz = zsd - z0
            rek = dict(cel=i, k=k, ulamek_przyrostu=1 - k / 10, latent_iou=m_lat['iou_top'], latent_pewnosc=m_lat['pewnosc_top'],
                       cos_do_warunku=cos(zsd, zk), wzgl_mse_do_warunku=float(((zsd - zk) ** 2).mean() / max((zk ** 2).mean(), 1e-6)),
                       cos_do_przyrostu=cos(dz, d), odtworzony_ulamek=float((dz * d).sum() / (d * d).sum()),
                       odl_od_oryginalu=float(np.linalg.norm(dz) / np.linalg.norm(z0)))
            wyn.append(rek)
            log('etap 10', i, k, {a: round(v, 3) for a, v in rek.items() if isinstance(v, float)})
            y = cv2.cvtColor(maly.copy(), cv2.COLOR_RGB2BGR); cv2.rectangle(y, (0, 0), (480, 46), (0, 0, 0), -1)
            cv2.putText(y, f'{int(round(100 * rek["ulamek_przyrostu"]))}% przyrostu (latent: IoU {rek["latent_iou"] or 0:.2f})',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(y, f'SD: cos do warunku {rek["cos_do_warunku"]:.2f}', (6, 39), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            pas.append(cv2.resize(y, (320, 213), interpolation=cv2.INTER_AREA))
        z = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); cv2.rectangle(z, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(z, f'zdjecie celu m{i:04d}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        cv2.imwrite(str(kat / f'PASEK_m{i:04d}.jpg'), np.concatenate([cv2.resize(z, (320, 213))] + pas, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
        zapisz('ETAP10_K8000.json', dict(wyniki=wyn))
    pods = {}
    for k in range(11):
        w = [x for x in wyn if x['k'] == k]
        if not w:
            continue
        pods[f'{100 - 10 * k}%'] = {a: float(np.mean([x[a] or 0 for x in w])) for a in
                                    ('latent_iou', 'latent_pewnosc', 'cos_do_warunku', 'wzgl_mse_do_warunku', 'cos_do_przyrostu',
                                     'odtworzony_ulamek', 'odl_od_oryginalu')}
        pods[f'{100 - 10 * k}%']['n'] = len(w)
        log('etap 10', f'{100 - 10 * k}%', {a: round(b, 3) for a, b in pods[f'{100 - 10 * k}%'].items()})
    zapisz('ETAP10_K8000.json', dict(wyniki=wyn, podsumowanie=pods, opis=__doc__))
    del net, vae; torch.cuda.empty_cache()


def etap11(rows):
    cele = json.loads((OUT / 'ETAP8_K8000.json').read_text(encoding='utf-8'))['cele']
    e10 = json.loads((OUT / 'ETAP10_K8000.json').read_text(encoding='utf-8'))['wyniki']
    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])
    man = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12

    @torch.inference_mode()
    def cls(rgb):
        f = model.forward_features(PRE(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).cuda())['x_norm_clstoken'][0].float().cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-12)

    def klatka(j):
        with zipfile.ZipFile(man[j]['archive']) as zf:
            im = np.asarray(Image.open(io.BytesIO(zf.read(man[j]['member']))).convert('RGB'))
        return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)

    def kafel(im, tekst, kolor=(255, 255, 255)):
        y = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR); y = cv2.resize(y, (480, 320))
        cv2.rectangle(y, (0, 0), (480, 26), (0, 0, 0), -1); cv2.putText(y, tekst, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, kolor, 1)
        return cv2.resize(y, (320, 213))
    kat = OUT / 'sd_szukanie'; kat.mkdir(exist_ok=True)
    wyn = []
    for c in cele:
        i = c['i']; rgb, _ = KL.obraz_manifestu(rows[i]); film = rows[i]['film']
        q0 = cls(rgb); s0 = Fp @ q0; top0 = np.argsort(-s0)[:5]
        gora, dol = [], []
        for k in sorted({x['k'] for x in e10 if x['cel'] == i}):
            sd = cv2.cvtColor(cv2.imread(str(OUT / 'sd_interpolacja' / f'm{i:04d}_k{k:02d}.png')), cv2.COLOR_BGR2RGB)
            q = cls(sd); s = Fp @ q; top = np.argsort(-s)[:5]
            rek = dict(cel=i, k=k, przyrost=1 - k / 10, film_celu=film,
                       top=[dict(j=int(j), film=man[j]['film'], klatka=int(man[j]['frame']), cos=float(s[j]), ma_gt=bool(man[j].get('gt'))) for j in top],
                       wspolne_z_zapytaniem_zdjeciem=int(len(set(top.tolist()) & set(top0.tolist()))),
                       top1_z_filmu_celu=man[top[0]]['film'] == film, cos_do_zdjecia_celu=float(q @ q0))
            wyn.append(rek)
            gora.append(kafel(sd, f'SD {int(round(100 * rek["przyrost"]))}% przyrostu'))
            dol.append(kafel(klatka(int(top[0])), f'{man[top[0]]["film"][3:7]}/{man[top[0]]["frame"]} cos {s[top[0]]:.2f}',
                             (0, 255, 0) if man[top[0]]['film'] == film else (0, 200, 255)))
            log('etap 11', i, k, rek['top'][0], 'wspólne z zapytaniem zdjęciem', rek['wspolne_z_zapytaniem_zdjeciem'])
        siatka = np.concatenate([np.concatenate([kafel(rgb, f'zdjecie celu m{i:04d}')] + gora, 1),
                                 np.concatenate([kafel(klatka(int(top0[0])), f'{man[top0[0]]["film"][3:7]}/{man[top0[0]]["frame"]} cos {s0[top0[0]]:.2f}', (0, 255, 0))] + dol, 1)], 0)
        cv2.imwrite(str(kat / f'SZUKANIE_m{i:04d}.jpg'), siatka, [cv2.IMWRITE_JPEG_QUALITY, 85])
        wyn.append(dict(cel=i, k='zdjecie', top=[dict(j=int(j), film=man[j]['film'], klatka=int(man[j]['frame']), cos=float(s0[j])) for j in top0]))
        zapisz('ETAP11_K8000.json', dict(wyniki=wyn))
    w = [x for x in wyn if x['k'] != 'zdjecie']
    pods = dict(unikalne_top1_na_cel={str(c['i']): len({x['top'][0]['j'] for x in w if x['cel'] == c['i']}) for c in cele},
                wspolne_z_zapytaniem_zdjeciem=float(np.mean([x['wspolne_z_zapytaniem_zdjeciem'] for x in w])),
                top1_z_filmu_celu=int(sum(x['top1_z_filmu_celu'] for x in w)), n=len(w),
                cos_do_zdjecia_celu={f'{100 - 10 * k}%': float(np.mean([x['cos_do_zdjecia_celu'] for x in w if x['k'] == k]))
                                     for k in range(11) if any(x['k'] == k for x in w)})
    zapisz('ETAP11_K8000.json', dict(wyniki=wyn, podsumowanie=pods, opis=__doc__))
    log('etap 11 podsumowanie', pods)


def main():
    st = KL.Stanowisko(VIT)
    assert st.sha == '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b', st.sha
    rows = KL.manifest()
    if '8' in ETAPY:
        etap8(st, rows)
    if '10' in ETAPY:
        etap10(st, rows)
    if '11' in ETAPY:
        del st; torch.cuda.empty_cache()
        etap11(rows)
    log('KONIEC')


if __name__ == '__main__':
    main()
