"""Regresja doboru szumu startowego SD z 10 różnych losowych startów, po 5 kroków każdy, z większymi krokami
(zlecenie użytkownika 2026-10-08: „zrób 10 różnych randomowych startów i 5 kroków szukania szumów startowych. Zrób
większe kroki zmiany podczas regresyjnego szukania”).
Metoda jak optymalizacja_regresja.py (ta sama baza 128 szumów, warunek Π(z0), miara postępu wzdłuż δ = z* − z0 w ViT K8000,
K_PAR = 16 par antytetycznych, regresja grzbietowa, krok elitarny z połową η przy braku poprawy), DDIM 20 kroków jak tam.
Różnice: start z czystego losowego szumu (jednohotowo na 10 losowych ziarnach z niewybieranej części bazy 5000..5119,
a nie na najlepszym z losowania), 5 kroków regresji na start, η0 = 1,0 zamiast 0,3 i σ = 0,05 zamiast 0,03,
próbki liczone partią na GPU (jak w optymalizacja_regresja_k5.py). Na końcu najlepszy szum w 50 krokach i szukanie TRAIN.
Wyniki w optymalizacja_regresja_starty/, log optymalizacja_regresja_starty.log.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/optymalizacja_regresja_starty.py
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
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_regresja_starty'
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

KROKI, K_PAR, SIGMA, ETA0, RIDGE, MAX_KROKOW, N_STARTOW = 20, 16, 0.05, 1.0, 1e-3, 5, 10
KSZTALT = (1, 4, 48, 72)
PARTIA_VAE = 8                       # dekodowanie VAE w kawałkach (pamięć RTX 3070)
PARTIA = [32]                        # partia odszumiania; zmniejszana przy braku pamięci
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
    def obrazy(Wb, kroki=KROKI):
        """Partia wag [B, M] -> lista (obraz, miary); ta sama ścieżka co pojedynczy obraz, tylko B naraz."""
        if Wb.shape[0] > PARTIA[0]:                                   # dzielenie partii (na karcie liczy też inny bieg)
            return sum((obrazy(Wb[s:s + PARTIA[0]], kroki) for s in range(0, Wb.shape[0], PARTIA[0])), [])
        Wb = Wb / Wb.norm(dim=1, keepdim=True)
        B = Wb.shape[0]
        h = (Wb @ E.view(M, -1)).view(B, *KSZTALT[1:])
        cb = c0[None].expand(B, -1, -1)
        sch.set_timesteps(kroki, device=dev)
        for t in sch.timesteps:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = den(sch.scale_model_input(h, t), t, cb).float()
            h = sch.step(e, t, h, eta=0.0).prev_sample
        wyj = []
        for s in range(0, B, PARTIA_VAE):
            x = ((vae.decode(h[s:s + PARTIA_VAE] / skala).sample.float() + 1) / 2).clamp(0, 1)
            x = F.interpolate(x, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
            for xi in x:
                img = (xi.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                ostr = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                wyj.append((img, dict(miary(st.przebieg(img)['z']), ostrosc=ostr)))
        licznik[0] += B
        return wyj

    _obrazy = obrazy

    def obrazy(Wb, kroki=KROKI):
        while True:
            try:
                return _obrazy(Wb, kroki)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if PARTIA[0] == 1:
                    raise
                PARTIA[0] //= 2; log('brak pamięci GPU, partia ->', PARTIA[0])

    def obraz(w, kroki=KROKI):
        return obrazy(w[None], kroki)[0]

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'ZDJECIE_CELU.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    # 10 startów: czyste losowe szumy z niewybieranej części bazy (ziarna 5000..5119, indeksy 8..127), nie najlepsze z losowania
    starty = sorted(int(i) for i in np.random.default_rng(0).choice(np.arange(8, M), N_STARTOW, replace=False))
    g_rng = torch.Generator(device=dev).manual_seed(7)
    przebiegi, t0 = [], time.time()
    for s_i, idx in enumerate(starty):
        w0 = torch.zeros(M, device=dev); w0[idx] = 1.0
        w = w0.clone()
        img, m = obraz(w)
        eta = ETA0
        wyn = [dict(krok=0, generacji=licznik[0], eta=eta, **m)]
        najl = dict(wyn[0]); w_najl = w.clone(); img_najl = img
        cv2.imwrite(str(kat / f's{s_i:02d}_z{ziarna[idx]}_g00.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        log(f'start {s_i} ziarno {ziarna[idx]}', {a: round(v, 3) for a, v in m.items()})
        for krok in range(1, MAX_KROKOW + 1):
            Xi = torch.randn((K_PAR, M), generator=g_rng, device=dev)
            t_k = time.time()
            Wp = torch.cat([w + SIGMA * Xi, w - SIGMA * Xi])
            wyj = obrazy(Wp)
            fp = [m_['ulamek'] for _, m_ in wyj[:K_PAR]]; fm = [m_['ulamek'] for _, m_ in wyj[K_PAR:]]
            df = torch.tensor(fp, device=dev) - torch.tensor(fm, device=dev)
            A = 2 * SIGMA * Xi
            g = A.T @ torch.linalg.solve(A @ A.T + RIDGE * torch.eye(K_PAR, device=dev), df)
            w_nowy = w + eta * g / (g.norm() + 1e-12)
            i_n, m_n = obraz(w_nowy)
            kand = [(m_n['ulamek'], w_nowy, i_n, m_n)] + [(m_['ulamek'], Wp[i].clone(), i_, m_) for i, (i_, m_) in enumerate(wyj)]
            f_b, w_b, i_b, m_b = max(kand, key=lambda x: x[0])
            if f_b > najl['ulamek']:
                w, img, m = w_b, i_b, m_b
                najl = dict(krok=krok, generacji=licznik[0], **m); w_najl = w.clone(); img_najl = img
            else:
                eta /= 2
            rek = dict(krok=krok, generacji=licznik[0], eta=eta, **m, wybrany='krok' if w_b is w_nowy else 'probka',
                       krok_ulamek=m_n['ulamek'], najlepsza_probka=max(fp + fm), srednia_probek=float(np.mean(fp + fm)),
                       cos_szumu_do_startu=float((w / w.norm()) @ w0), czas_kroku_s=time.time() - t_k)
            wyn.append(rek)
            cv2.imwrite(str(kat / f's{s_i:02d}_z{ziarna[idx]}_g{krok:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            log(f'start {s_i} krok {krok}', {a: (round(v, 4) if isinstance(v, float) else v) for a, v in rek.items()})
        przebiegi.append(dict(start=s_i, ziarno=int(ziarna[idx]), wyniki=wyn, najlepsza=najl, w_najl=w_najl, img=img_najl))
        (OUT / 'WYNIK.json').write_text(json.dumps(dict(przebiegi=[{k: v for k, v in p.items() if k not in ('w_najl', 'img')} for p in przebiegi]),
                                        indent=1, ensure_ascii=False), encoding='utf-8')
    t_opt = time.time() - t0
    b = max(przebiegi, key=lambda p: p['najlepsza']['ulamek'])
    spr = []
    for kroki in (KROKI, 50):
        i_s, m_s = obraz(b['w_najl'], kroki); spr.append(dict(kroki=kroki, **m_s))
        cv2.imwrite(str(OUT / f'NAJLEPSZY_kroki{kroki}.png'), cv2.cvtColor(i_s, cv2.COLOR_RGB2BGR)); log('sprawdzenie najlepszego', spr[-1])
    np.save(OUT / 'WAGI_NAJLEPSZE.npy', b['w_najl'].cpu().numpy())
    del den, vae; torch.cuda.empty_cache()
    szuk = szukaj(st, b['img'], miary, f'start{b["start"]}')

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5), layout='constrained')
    for p in przebiegi:
        k = [r['krok'] for r in p['wyniki']]
        ax[0].plot(k, [100 * r['ulamek'] for r in p['wyniki']], 'o-', label=f's{p["start"]} z{p["ziarno"]}')
        ax[1].plot(k, [100 * r['ulamek_widoczny'] for r in p['wyniki']], 'o-')
        ax[2].plot(k, [r['ostrosc'] for r in p['wyniki']], 'o-')
    for a, t in zip(ax, ['postęp wzdłuż δ [%]', 'część widoczna dla fc2 [%]', 'ostrość (wariancja laplasjanu)']):
        a.set_xlabel('krok regresji'); a.set_title(t); a.grid(alpha=.3)
    ax[0].axhline(17.2, c='k', ls='--', lw=.8, label='regresja 20 kroków od najlepszego losowego'); ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'{N_STARTOW} losowych startów × {MAX_KROKOW} kroków regresji (η0 = {ETA0}, σ = {SIGMA}, DDIM {KROKI})')
    fig.savefig(OUT / 'WYKRES_STARTY.png', dpi=110); plt.close(fig)
    pas = []
    for p in przebiegi:
        a0 = cv2.imread(str(kat / f's{p["start"]:02d}_z{p["ziarno"]}_g00.png')); a1 = cv2.cvtColor(p['img'], cv2.COLOR_RGB2BGR).copy()
        for im_, t in ((a0, f's{p["start"]} start {100 * p["wyniki"][0]["ulamek"]:.1f}%'),
                       (a1, f'koniec {100 * p["najlepsza"]["ulamek"]:.1f}% / wid. {100 * p["najlepsza"]["ulamek_widoczny"]:.1f}%')):
            cv2.rectangle(im_, (0, 0), (480, 24), (0, 0, 0), -1); cv2.putText(im_, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        pas.append(np.concatenate([cv2.resize(a0, (360, 240)), cv2.resize(a1, (360, 240))], 1))
    pas += [np.zeros_like(pas[0])] * (len(pas) % 2)
    cv2.imwrite(str(OUT / 'STARTY_PRZED_PO.jpg'), np.concatenate([np.concatenate(pas[i:i + 2], 1) for i in range(0, len(pas), 2)], 0))
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(przebiegi=[{k: v for k, v in p.items() if k not in ('w_najl', 'img')} for p in przebiegi],
                                    najlepszy_start=b['start'], sprawdzenie=spr, szukanie_train=szuk, ziarna_bazy=ziarna,
                                    czas=dict(optymalizacja_s=t_opt, generacji=licznik[0]),
                                    parametry=dict(kroki=KROKI, k_par=K_PAR, sigma=SIGMA, eta0=ETA0, ridge=RIDGE, max_krokow=MAX_KROKOW,
                                                   n_startow=N_STARTOW, m=M, cel_j=CEL_J, z_sztuczny=str(Z_SZT)), opis=__doc__),
                                    indent=1, ensure_ascii=False), encoding='utf-8')
    log('najlepszy start', b['start'], b['najlepsza'])
    log('KONIEC')


def szukaj(st, img, miary, etykieta):
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
        log('szukanie TRAIN', etykieta, wyn[-1])
    cv2.imwrite(str(OUT / f'SZUKANIE_TRAIN_najlepszy_{etykieta}.jpg'), cv2.cvtColor(np.concatenate([cv2.resize(k, (384, 256)) for k in kaf], 1), cv2.COLOR_RGB2BGR))
    return wyn


def film(wyn):
    tmp = OUT / '_klatki_filmu'; tmp.mkdir(exist_ok=True)
    for i, r in enumerate(wyn):
        im = cv2.imread(str(OUT / 'obrazy' / f'g{r["krok"]:02d}.png'))
        cv2.rectangle(im, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(im, f'DDIM 5: krok regresji {r["krok"]:2d}   postep {100 * r["ulamek"]:4.1f}%   widoczny fc2 {100 * r["ulamek_widoczny"]:4.1f}%',
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
