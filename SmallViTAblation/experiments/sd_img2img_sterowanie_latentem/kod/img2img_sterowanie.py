"""img2img SD K8000 ze zdjęcia celu, sterowane tylko sztucznym latentem (zlecenie użytkownika 2026-10-08).

Jeden obraz: klatka TRAIN 0055/4019 (pula vit_dino_faiss_35000, j = 10389), na której ViT K8000 nie wykrywa narzędzia.
Sztuczny latent z* = latent_cel_do_obrazow/k8000_trening/zb_j10389_minmax.npy (korekta detekcji w zakresach kanałów).
Zdjęcie celu -> VAE SD K8000 -> zaszumienie do poziomu `sila` -> odszumianie DDIM (50 kroków leading, eta 0).
Warianty:
  kontrola_oryginal        warunek z0 (latent zdjęcia), bez prowadzenia: sam efekt zaszumienia i odszumienia;
  warunek_sztuczny         warunek z*, bez prowadzenia;
  warunek_sztuczny_prow    warunek z* i prowadzenie metodą użytkownika (sd_prowadzone.py): w każdym kroku x̂0 ->
                           dekoder VAE -> ViT K8000 -> latent z, energia E = średnia (z − z*)^2 po pełnym latencie 600 x 768,
                           ITER kroków gradientu po x̂0 (krok ETA · σ_t, gradient znormalizowany), ostatnie KONIEC kroków
                           bez prowadzenia; RUNDY rund (wynik rundy jest startem następnej).
Bez straty detekcji. Miary latentu obrazu z: odległość od z* względem odległości z0 od z*, ułamek przyrostu δ = z* − z0
(cały i część widoczna dla fc2), kosinus do δ, odległość tokenów od podprzestrzeni naturalnych (PCA-64, prawdziwe 2,6),
PSNR wobec zdjęcia celu.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/img2img_sterowanie.py
"""
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem'
OUT.mkdir(parents=True, exist_ok=True)
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
SD = ROOT / 'assets' / 'SD_ViT_K8000_A_control_K10500'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CEL_J = 10389
Z_SZT = ROOT / 'outputs' / 'latent_cel_do_obrazow' / 'k8000_trening' / f'zb_j{CEL_J:05d}_minmax.npy'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import korekta_latentu as KL

KROKI, ITER, ETA, KONIEC, RUNDY = 50, 2, 0.15, 5, 3
SILY = [0.3, 0.5, 0.7]
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)
MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def klatka(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


def main():
    dev = torch.device('cuda')
    st = KL.Stanowisko(VIT)
    assert st.sha == '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b'
    net = st.net
    sp = importlib.util.spec_from_file_location('model_sd_k8000', SD / 'kod' / 'model.py')
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    from diffusers import DDIMScheduler
    ck = torch.load(SD / 'best_sd.pt', map_location='cpu', weights_only=False)
    den = M.Denoiser().eval(); vae = M.new_vae().eval()
    den.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    den.to(dev).requires_grad_(False); vae.to(dev).requires_grad_(False)
    vae.enable_gradient_checkpointing()
    skala = float(ck['scale'])
    sch = DDIMScheduler.from_config(M.schedule().config, timestep_spacing='leading')

    # naturalne tokeny: 400 losowych klatek puli, 100 tokenów z klatki (jak k8000_trening_trudne.py)
    rng = np.random.default_rng(0)
    Tk = torch.cat([st.przebieg(klatka(int(j)))['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(dev)]
                    for j in rng.choice(len(MAN), 400, replace=False)])
    mu = Tk.mean(0); _, _, Vt = torch.linalg.svd(Tk - mu, full_matrices=False); U = Vt[:64].T
    poza = lambda z: float((((z - mu) - ((z - mu) @ U) @ U.T) ** 2).sum(1).sqrt().mean())
    W2 = st.blok.mlp.fc2.weight.detach().float()

    rgb0 = klatka(CEL_J)
    z0 = st.przebieg(rgb0)['z']
    zs = torch.from_numpy(np.load(Z_SZT)).to(dev)
    d = zs - z0; d_w = d @ W2.T
    log('cel', MAN[CEL_J]['film'], MAN[CEL_J]['frame'], '||z0||', round(float(z0.norm()), 1), '||δ||', round(float(d.norm()), 1),
        'poza z0', round(poza(z0), 2), 'poza z*', round(poza(zs), 2))

    def ocen(img):
        z = st.przebieg(img)['z']; dz = z - z0
        mse = float(((img.astype(np.float32) - rgb0.astype(np.float32)) ** 2).mean())
        return dict(odl_wzgl=float((z - zs).norm() / d.norm()), ulamek=float((dz * d).sum() / (d * d).sum()),
                    ulamek_widoczny=float(((dz @ W2.T) * d_w).sum() / (d_w * d_w).sum()),
                    cos_przyrost=float((dz * d).sum() / (dz.norm() * d.norm() + 1e-12)), poza=poza(z),
                    psnr_do_celu=float(10 * np.log10(255 ** 2 / max(mse, 1e-6))))

    def latent_vit(img01):
        x = F.interpolate(img01, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        net.zapisuj_latent(True)
        try:
            net(x.float())
            return net._latent[0].float()
        finally:
            net.zapisuj_latent(False)

    def rysuj(start_rgb, warunek, sila, prowadz, ziarno=0):
        x_in = torch.from_numpy(cv2.resize(start_rgb, (576, 384), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().to(dev) / 255
        with torch.no_grad():
            h0 = vae.encode(x_in * 2 - 1).latent_dist.mode() * skala
        sch.set_timesteps(KROKI, device=dev); ts = sch.timesteps
        i0 = int(round(KROKI * (1 - sila)))
        g = torch.Generator(device=dev).manual_seed(ziarno)
        h = sch.add_noise(h0, torch.randn(h0.shape, generator=g, device=dev), ts[i0:i0 + 1])
        war = warunek[None].float()
        ac = sch.alphas_cumprod.to(dev)
        E_hist = []
        for i in range(i0, KROKI):
            t = ts[i]; a = ac[t].float()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                e = den(sch.scale_model_input(h, t), t, war).float()
            if prowadz and i < KROKI - KONIEC:
                x0 = (h - (1 - a).sqrt() * e) / a.sqrt()
                sig = float((1 - a).sqrt())
                for _ in range(ITER):
                    v = x0.detach().requires_grad_(True)
                    img01 = ((vae.decode(v / skala).sample + 1) / 2).clamp(0, 1)
                    E = ((latent_vit(img01) - zs) ** 2).mean()
                    gr, = torch.autograd.grad(E, v)
                    x0 = (v - ETA * sig * gr / (gr.pow(2).mean().sqrt() + 1e-8)).detach()
                    E_hist.append(float(E))
                e = (h - a.sqrt() * x0) / (1 - a).sqrt()
            h = sch.step(e, t, h, eta=0.0).prev_sample
        with torch.no_grad():
            img = ((vae.decode(h / skala).sample.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.resize((img * 255).round().astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA), E_hist

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    wyn = [dict(wariant='zdjecie_celu', sila=0.0, runda=0, **ocen(rgb0))]
    log('zdjęcie celu', {a: round(b, 3) for a, b in wyn[0].items() if isinstance(b, float)})
    warianty = [('kontrola_oryginal', z0, False, 1), ('warunek_sztuczny', zs, False, 1), ('warunek_sztuczny_prow', zs, True, RUNDY)]
    obrazy = {}
    for nazwa, war, prow, rundy in warianty:
        for s in SILY:
            start = rgb0
            for r in range(rundy):
                img, Eh = rysuj(start, war, s, prow, ziarno=r)
                cv2.imwrite(str(kat / f'{nazwa}_s{int(100 * s):02d}_r{r}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                rek = dict(wariant=nazwa, sila=s, runda=r, E_start=Eh[0] if Eh else None, E_koniec=Eh[-1] if Eh else None, **ocen(img))
                wyn.append(rek); obrazy[(nazwa, s, r)] = img
                log(nazwa, f'siła {s}', f'runda {r}', {a: round(b, 3) for a, b in rek.items() if isinstance(b, float)})
                start = img
            (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn), indent=1, ensure_ascii=False), encoding='utf-8')

    # siatka: wiersze = warianty (ostatnia runda), kolumny = siły; pod spodem rundy prowadzenia dla każdej siły
    def kafel(im, t1, t2=''):
        y = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(y, (0, 0), (480, 50), (0, 0, 0), -1)
        cv2.putText(y, t1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        cv2.putText(y, t2, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
        return cv2.resize(y, (360, 240), interpolation=cv2.INTER_AREA)
    m = {(x['wariant'], x['sila'], x['runda']): x for x in wyn}
    opis = lambda x: f'blizej z*: {100 * (1 - x["odl_wzgl"]):.1f}%  ulamek {100 * x["ulamek"]:.1f}%  poza {x["poza"]:.1f}'
    wiersze = [np.concatenate([kafel(rgb0, 'zdjecie celu 0055/4019', opis(wyn[0]))] + [np.zeros((240, 360, 3), np.uint8)] * (len(SILY) - 1), 1)]
    for nazwa, _, _, rundy in warianty:
        wiersze.append(np.concatenate([kafel(obrazy[(nazwa, s, rundy - 1)], f'{nazwa} sila {s} r{rundy - 1}', opis(m[(nazwa, s, rundy - 1)])) for s in SILY], 1))
    for r in range(RUNDY - 1):
        wiersze.append(np.concatenate([kafel(obrazy[('warunek_sztuczny_prow', s, r)], f'prowadzone sila {s} runda {r}', opis(m[('warunek_sztuczny_prow', s, r)])) for s in SILY], 1))
    cv2.imwrite(str(OUT / 'SIATKA.jpg'), np.concatenate(wiersze, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn, parametry=dict(kroki=KROKI, iteracje=ITER, eta=ETA, koniec=KONIEC, rundy=RUNDY, sily=SILY,
                                                                                 cel_j=CEL_J, z_sztuczny=str(Z_SZT)), opis=__doc__), indent=1, ensure_ascii=False), encoding='utf-8')
    log('KONIEC')


if __name__ == '__main__':
    main()
