"""Dobór warunków początkowych SD regresją, bez gradientu przez SD (zlecenie użytkownika 2026-10-08).

Jeden obraz: klatka TRAIN 0055/4019 (j = 10389), z* i δ = z* − z0 jak w optymalizacja_mieszanka.py.
Szum startowy jak w optymalizacja_mieszanka.py: h = Σ w_i ε_i / ||w|| z tej samej bazy M = 128 szumów (ziarna z
optymalizacja_mieszanka/WYNIK.json albo LOSOWE_SZUKANIE.json), więc zawsze dokładnie N(0, I); warunek stały c0 = Π(z0). SD działa jak zwykle
(DDIM 20 kroków leading, eta 0, bez gradientu), wynik f(w) = postęp latentu obrazu wzdłuż δ.
Krok regresji: K par kierunków ξ_k ~ N(0, I_M); obrazy dla w ± σ ξ_k; regresja grzbietowa różnic
f(w + σ ξ_k) − f(w − σ ξ_k) ≈ 2 σ ξ_k^T g daje oszacowany gradient g (która zmiana szumu ile przybliża do z*);
krok w <- w + η g/||g||, przyjęty tylko gdy obraz środka jest lepszy (inaczej η /= 2). Najlepsza z próbek też może
zostać nowym środkiem. Stop: CIERPLIWOSC kroków bez poprawy albo MAX_KROKOW.
Potem: najlepszy w 50 krokach DDIM, wyszukiwanie klatek TRAIN (DINO), wykres, film.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/optymalizacja_regresja.py
"""
import importlib.util
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BAZA = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_mieszanka'
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_regresja'
OUT.mkdir(parents=True, exist_ok=True)
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
SD = ROOT / 'assets' / 'SD_ViT_K8000_A_control_K10500'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CEL_J = 10389
Z_SZT = ROOT / 'outputs' / 'latent_cel_do_obrazow' / 'k8000_trening' / f'zb_j{CEL_J:05d}_minmax.npy'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import korekta_latentu as KL

KROKI, K_PAR, SIGMA, ETA0, RIDGE, CIERPLIWOSC, MAX_KROKOW = 20, 16, 0.03, 0.3, 1e-3, 5, 20
KSZTALT = (1, 4, 48, 72)
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)
MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def klatka(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    dev = torch.device('cuda')
    # baza z losowego szukania biegu mieszanki (bieg gradientowy przerwany, więc z LOSOWE_SZUKANIE.json odtworzonego z logu)
    baza = json.loads(((BAZA / 'WYNIK.json') if (BAZA / 'WYNIK.json').exists() else (BAZA / 'LOSOWE_SZUKANIE.json')).read_text(encoding='utf-8'))
    ziarna = baza['ziarna_bazy']; M = len(ziarna)
    st = KL.Stanowisko(VIT)
    assert st.sha == '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b'
    sp = importlib.util.spec_from_file_location('model_sd_k8000', SD / 'kod' / 'model.py')
    Msd = importlib.util.module_from_spec(sp); sp.loader.exec_module(Msd)
    from diffusers import DDIMScheduler
    ck = torch.load(SD / 'best_sd.pt', map_location='cpu', weights_only=False)
    den = Msd.Denoiser().eval(); vae = Msd.new_vae().eval()
    den.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    den.to(dev).requires_grad_(False); vae.to(dev).requires_grad_(False)
    skala = float(ck['scale'])
    sch = DDIMScheduler.from_config(Msd.schedule().config, timestep_spacing='leading')

    rng = np.random.default_rng(0)
    Tk = torch.cat([st.przebieg(klatka(int(j)))['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(dev)]
                    for j in rng.choice(len(MAN), 400, replace=False)])
    mu = Tk.mean(0); _, _, Vt = torch.linalg.svd(Tk - mu, full_matrices=False); U = Vt[:64].T
    reszta = lambda z: (z - mu) - ((z - mu) @ U) @ U.T
    poza = lambda z: float(reszta(z).norm(dim=1).mean())
    prog = torch.quantile(reszta(Tk).norm(dim=1), .99)
    lo, hi = Tk.min(0).values, Tk.max(0).values

    def rzut(c):
        for _ in range(20):
            y = c - mu; p = (y @ U) @ U.T; r = y - p
            r = r * torch.clamp(prog / (r.norm(dim=1, keepdim=True) + 1e-9), max=1.0)
            c = torch.minimum(torch.maximum(mu + p + r, lo), hi)
        y = c - mu; p = (y @ U) @ U.T; r = y - p
        return mu + p + r * torch.clamp(prog / (r.norm(dim=1, keepdim=True) + 1e-9), max=1.0)

    W2 = st.blok.mlp.fc2.weight.detach().float()
    rgb0 = klatka(CEL_J)
    z0 = st.przebieg(rgb0)['z']
    zs = torch.from_numpy(np.load(Z_SZT)).to(dev)
    d = zs - z0; dd = (d * d).sum(); d_w = d @ W2.T

    def miary(z):
        dz = z - z0; u = (dz * d).sum() / dd
        return dict(ulamek=float(u), ulamek_widoczny=float(((dz @ W2.T) * d_w).sum() / (d_w * d_w).sum()),
                    cos_przyrost=float((dz * d).sum() / (dz.norm() * d.norm() + 1e-12)), odl_wzgl=float((z - zs).norm() / d.norm()),
                    reszta_od_osi=float((dz - u * d).norm() / d.norm()), poza=poza(z))

    c0 = rzut(z0)
    E = torch.cat([torch.randn(KSZTALT, generator=torch.Generator(device=dev).manual_seed(z), device=dev) for z in ziarna])
    licznik = [0]

    @torch.no_grad()
    def obraz(w, kroki=KROKI):
        w = w / w.norm()
        h = (w.view(M, 1, 1, 1) * E).sum(0, keepdim=True)
        sch.set_timesteps(kroki, device=dev)
        for t in sch.timesteps:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = den(sch.scale_model_input(h, t), t, c0[None]).float()
            h = sch.step(e, t, h, eta=0.0).prev_sample
        x = ((vae.decode(h / skala).sample.float() + 1) / 2).clamp(0, 1)
        x = F.interpolate(x, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        img = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        licznik[0] += 1
        ostr = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        return img, dict(miary(st.przebieg(img)['z']), ostrosc=ostr)

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'ZDJECIE_CELU.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    w = torch.zeros(M, device=dev); w[0] = 1.0
    img, m = obraz(w)
    eta, bez = ETA0, 0
    wyn = [dict(krok=0, generacji=licznik[0], eta=eta, **m, r2=None, przewidziana_poprawa=None)]
    najl = dict(wyn[0]); w_najl = w.clone(); Zimg = [st.przebieg(img)['z'].cpu().numpy().astype(np.float16)]
    cv2.imwrite(str(kat / 'g00.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    log('start (najlepszy losowy szum)', {a: round(v, 3) for a, v in m.items()})
    g_rng = torch.Generator(device=dev).manual_seed(7)
    for krok in range(1, MAX_KROKOW + 1):
        Xi = torch.randn((K_PAR, M), generator=g_rng, device=dev)
        fp, fm, probki = [], [], []
        for k in range(K_PAR):
            for znak, lst in ((1, fp), (-1, fm)):
                wk = w + znak * SIGMA * Xi[k]
                ik, mk = obraz(wk); lst.append(mk['ulamek']); probki.append((mk['ulamek'], wk.clone(), ik, mk))
        df = torch.tensor(fp, device=dev) - torch.tensor(fm, device=dev)                       # K
        A = 2 * SIGMA * Xi                                                                         # K x M
        g = A.T @ torch.linalg.solve(A @ A.T + RIDGE * torch.eye(K_PAR, device=dev), df)         # regresja grzbietowa
        pred = A @ g
        r2 = float(1 - ((df - pred) ** 2).sum() / (((df - df.mean()) ** 2).sum() + 1e-12))
        # krok wzdłuż oszacowanego gradientu; przyjęty tylko przy poprawie
        w_nowy = w + eta * g / (g.norm() + 1e-12)
        i_n, m_n = obraz(w_nowy)
        przew = float(eta * g.norm())          # poprawa przewidziana przez regresję dla kroku η wzdłuż g/||g||
        kand = [(m_n['ulamek'], w_nowy, i_n, m_n)] + probki
        f_b, w_b, i_b, m_b = max(kand, key=lambda x: x[0])
        if f_b > najl['ulamek']:
            w, img, m = w_b, i_b, m_b; bez = 0
            najl = dict(krok=krok, generacji=licznik[0], **m); w_najl = w.clone()
        else:
            eta /= 2; bez += 1
        rek = dict(krok=krok, generacji=licznik[0], eta=eta, **m, r2=r2, poprawa_kroku=m_n['ulamek'] - wyn[-1]['ulamek'],
                   najlepsza_probka=max(fp + fm), srednia_probek=float(np.mean(fp + fm)), wybrany='krok' if w_b is w_nowy else 'probka',
                   przewidziana_poprawa=przew)
        wyn.append(rek); Zimg.append(st.przebieg(img)['z'].cpu().numpy().astype(np.float16))
        cv2.imwrite(str(kat / f'g{krok:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        log(f'krok {krok:2d}', {a: (round(v, 4) if isinstance(v, float) else v) for a, v in rek.items()}, 'bez poprawy', bez)
        (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn, najlepsza=najl), indent=1, ensure_ascii=False), encoding='utf-8')
        if bez >= CIERPLIWOSC:
            log('stop: brak poprawy przez', CIERPLIWOSC, 'kroków')
            break
    spr = []
    for kroki in (KROKI, 50):
        i_s, m_s = obraz(w_najl, kroki)
        cv2.imwrite(str(OUT / f'NAJLEPSZY_kroki{kroki}.png'), cv2.cvtColor(i_s, cv2.COLOR_RGB2BGR))
        spr.append(dict(kroki=kroki, **m_s)); log('sprawdzenie najlepszego', spr[-1])
    np.savez_compressed(OUT / 'LATENTY.npz', obrazy=np.stack(Zimg), z0=z0.cpu().numpy().astype(np.float16),
                        z_sztuczny=zs.cpu().numpy().astype(np.float16), wagi_najlepsze=w_najl.cpu().numpy())
    del den, vae; torch.cuda.empty_cache()
    szuk = szukaj(st, cv2.cvtColor(cv2.imread(str(kat / f'g{wyn[-1]["krok"]:02d}.png')), cv2.COLOR_BGR2RGB), miary)
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn, najlepsza=najl, sprawdzenie=spr, szukanie_train=szuk, ziarna_bazy=ziarna,
                                    parametry=dict(kroki=KROKI, k_par=K_PAR, sigma=SIGMA, eta0=ETA0, ridge=RIDGE, cierpliwosc=CIERPLIWOSC,
                                                   max_krokow=MAX_KROKOW, m=M, cel_j=CEL_J, z_sztuczny=str(Z_SZT)), opis=__doc__),
                                    indent=1, ensure_ascii=False), encoding='utf-8')
    film(wyn)
    log('najlepsza', najl, 'generacji razem', licznik[0])
    log('KONIEC')


def szukaj(st, img, miary):
    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12
    with torch.inference_mode():
        q = model.forward_features(PRE(Image.fromarray(img)).unsqueeze(0).cuda())['x_norm_clstoken'][0].float().cpu().numpy()
    s = Fp @ (q / np.linalg.norm(q)); s[CEL_J] = -9
    wyn, kaf = [], [img]
    for j in np.argsort(-s)[:5]:
        im = klatka(int(j)); m = miary(st.przebieg(im)['z'])
        wyn.append(dict(j=int(j), film=MAN[j]['film'], klatka=int(MAN[j]['frame']), cos_dino=float(s[j]), **m))
        y = cv2.cvtColor(im, cv2.COLOR_RGB2BGR).copy(); cv2.rectangle(y, (0, 0), (480, 46), (0, 0, 0), -1)
        cv2.putText(y, f'{MAN[j]["film"]}/{MAN[j]["frame"]} DINO {s[j]:.2f}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(y, f'postep {100 * m["ulamek"]:.1f}%  widoczny {100 * m["ulamek_widoczny"]:.1f}%', (6, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        kaf.append(cv2.cvtColor(y, cv2.COLOR_BGR2RGB))
        log('szukanie TRAIN', wyn[-1])
    cv2.imwrite(str(OUT / 'SZUKANIE_TRAIN_najlepszy.jpg'), cv2.cvtColor(np.concatenate([cv2.resize(k, (384, 256)) for k in kaf], 1), cv2.COLOR_RGB2BGR))
    return wyn


def film(wyn):
    tmp = OUT / '_klatki_filmu'; tmp.mkdir(exist_ok=True)
    for i, r in enumerate(wyn):
        im = cv2.imread(str(OUT / 'obrazy' / f'g{r["krok"]:02d}.png'))
        cv2.rectangle(im, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(im, f'krok regresji {r["krok"]:2d}   postep {100 * r["ulamek"]:4.1f}%   widoczny fc2 {100 * r["ulamek_widoczny"]:4.1f}%',
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        for p in range(15):                                   # pół sekundy na krok przy 30 kl./s
            cv2.imwrite(str(tmp / f'k{i * 15 + p:04d}.png'), im)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', str(tmp / 'k%04d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', str(OUT / 'FILM_OPTYMALIZACJA_30fps.mp4')], check=False)
    for p in tmp.glob('*.png'):
        p.unlink()
    tmp.rmdir()


if __name__ == '__main__':
    main()
