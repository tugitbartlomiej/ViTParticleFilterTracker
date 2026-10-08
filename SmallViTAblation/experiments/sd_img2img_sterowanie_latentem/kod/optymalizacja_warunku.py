"""Gradientowy dobór wejścia sterującego SD tak, żeby latent gotowego obrazu w ViT szedł w stronę sztucznego latentu
(zlecenie użytkownika 2026-10-08; odpowiednik jego sterowania FastGAN-em gradientem po wejściu generatora).

Jeden obraz: klatka TRAIN 0055/4019 (j = 10389), sztuczny latent z* jak w petla_szum.py, δ = z* − z0.
Parametr: warunek SD c (600 x 768), start c = z0. Przed każdym użyciem c przechodzi przez rzut Π do zakresu warunków
z uczenia SD (jak petla_szum.py, wersja 2), więc SD zawsze dostaje warunek ze swojego zakresu.
Iteracja: pełna generacja od STAŁEGO szumu (ziarno 0), DDIM KROKI kroków leading, eta 0, różniczkowalnie
(checkpoint każdego wywołania UNet) -> dekoder VAE -> ViT K8000 -> latent obrazu z.
Strata = −(postęp wzdłuż δ) = −<z − z0, δ>/||δ||^2 + LAM * (składowa z − z0 prostopadła do δ)/||δ||;
gradient przez cały łańcuch odszumiania do c; Adam. Każda iteracja to jeden wygenerowany obraz i jego sprawdzenie.
Stop: brak lepszego obrazu (większego postępu) przez CIERPLIWOSC iteracji albo MAX_IT iteracji.
Na koniec najlepszy warunek generowany też w 50 krokach DDIM i z 4 innymi szumami (czy wynik trzyma się poza szumem
optymalizacji).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/optymalizacja_warunku.py
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
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_warunku'
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
from PIL import Image
from torch.utils.checkpoint import checkpoint

import korekta_latentu as KL

KROKI, LR, LAM, CIERPLIWOSC, MAX_IT = 20, 0.05, 0.1, 15, 150
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
    ac = sch.alphas_cumprod.to(dev).float(); a_koniec = float(sch.final_alpha_cumprod)

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

    def latent_vit(img01):
        x = F.interpolate(img01, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        net.zapisuj_latent(True)
        try:
            net(x.float())
            return net._latent[0].float()
        finally:
            net.zapisuj_latent(False)

    def unet(h, t, c):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return den(h, t, c[None]).float()

    def generuj(c, ziarno, kroki, grad):
        sch.set_timesteps(kroki, device=dev); ts = sch.timesteps
        g = torch.Generator(device=dev).manual_seed(ziarno)
        h = torch.randn((1, 4, 48, 72), generator=g, device=dev)
        krok = sch.config.num_train_timesteps // kroki
        with torch.set_grad_enabled(grad):
            for t in ts:
                e = checkpoint(unet, h, t, c, use_reentrant=False) if grad else unet(h, t, c)
                a = ac[t]; tp = int(t) - krok
                ap = ac[tp] if tp >= 0 else torch.tensor(a_koniec, device=dev)
                x0 = (h - (1 - a).sqrt() * e) / a.sqrt()
                h = ap.sqrt() * x0 + (1 - ap).sqrt() * e
            img01 = ((vae.decode(h / skala).sample + 1) / 2).clamp(0, 1)
        return img01

    def do_uint8(img01):
        x = F.interpolate(img01.detach().float(), size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        return (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'ZDJECIE_CELU.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    log('cel', MAN[CEL_J]['film'], MAN[CEL_J]['frame'], '||δ||/||z0||', round(float(d.norm() / z0.norm()), 2), 'próg poza', round(float(prog), 2))
    c_par = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([c_par], lr=LR)
    wyn, Zimg, Zwar = [], [], []
    najl, bez_poprawy = None, 0
    for it in range(MAX_IT):
        c = rzut(c_par)
        img01 = generuj(c, 0, KROKI, True)
        z = latent_vit(img01)
        dz = z - z0; u = (dz * d).sum() / dd
        strata = -u + LAM * (dz - u * d).norm() / d.norm()
        opt.zero_grad(); strata.backward(); opt.step()
        # sprawdzenie gotowego obrazu: latent liczony z obrazu 480 x 320 uint8, jak każdy inny obraz w stanowisku
        img = do_uint8(img01)
        zc = st.przebieg(img)['z']
        m = miary(zc); mw = miary(c.detach())
        rek = dict(generacja=it, strata=float(strata), **m, warunek_ulamek=mw['ulamek'], warunek_poza=mw['poza'], warunek_odl_wzgl=mw['odl_wzgl'])
        if najl is None or m['ulamek'] > najl['ulamek']:
            najl = dict(rek); c_najl = c.detach().clone(); bez_poprawy = 0
        else:
            bez_poprawy += 1
        rek['najlepsza_dotad'] = najl['generacja']
        wyn.append(rek)
        Zimg.append(zc.cpu().numpy().astype(np.float16)); Zwar.append(c.detach().cpu().numpy().astype(np.float16))
        cv2.imwrite(str(kat / f'g{it:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        log(f'it {it:3d}', {a: round(v, 3) for a, v in rek.items() if isinstance(v, float)}, 'najlepsza', najl['generacja'], 'bez poprawy', bez_poprawy)
        if bez_poprawy >= CIERPLIWOSC:
            log('stop: brak lepszego obrazu przez', CIERPLIWOSC, 'iteracji')
            break
    # sprawdzenie najlepszego warunku: 50 kroków i inne szumy
    spr = []
    for ziarno, kroki in [(0, KROKI), (0, 50), (1, KROKI), (2, KROKI), (3, KROKI), (4, KROKI)]:
        img = do_uint8(generuj(c_najl, ziarno, kroki, False))
        cv2.imwrite(str(OUT / f'NAJLEPSZY_WARUNEK_szum{ziarno}_kroki{kroki}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        spr.append(dict(ziarno=ziarno, kroki=kroki, **miary(st.przebieg(img)['z'])))
        log('sprawdzenie najlepszego warunku', spr[-1])
    for ziarno in range(1, 5):                     # kontrola: warunek z0 z tymi samymi szumami
        img = do_uint8(generuj(rzut(z0), ziarno, KROKI, False))
        spr.append(dict(ziarno=ziarno, kroki=KROKI, kontrola_warunek_z0=True, **miary(st.przebieg(img)['z'])))
    np.savez_compressed(OUT / 'LATENTY.npz', obrazy=np.stack(Zimg), warunki=np.stack(Zwar), z0=z0.cpu().numpy().astype(np.float16),
                        z_sztuczny=zs.cpu().numpy().astype(np.float16), warunek_najlepszy=c_najl.cpu().numpy().astype(np.float16))
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn, najlepsza=najl, sprawdzenie=spr, parametry=dict(kroki=KROKI, lr=LR, lam=LAM,
                                    cierpliwosc=CIERPLIWOSC, max_it=MAX_IT, cel_j=CEL_J, z_sztuczny=str(Z_SZT)), opis=__doc__),
                                    indent=1, ensure_ascii=False), encoding='utf-8')
    wykres(wyn, spr)
    log('najlepsza', najl)
    subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('grafika_latentow.py')), OUT.name], check=False)
    log('KONIEC')


def wykres(wyn, spr):
    G = np.arange(len(wyn)); a = lambda k: np.array([r[k] for r in wyn])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(G, 100 * a('ulamek'), 'o-', ms=3, label='cały przyrost'); ax[0, 0].plot(G, 100 * a('ulamek_widoczny'), 's-', ms=3, label='część widoczna dla fc2')
    ax[0, 0].plot(G, 100 * np.maximum.accumulate(a('ulamek')), 'k--', label='najlepszy dotąd')
    kon = [r['ulamek'] for r in spr if r.get('kontrola_warunek_z0')]
    ax[0, 0].axhline(100 * np.mean(kon), c='gray', ls=':', label='kontrola: warunek z0, 4 szumy')
    ax[0, 0].set_title('postęp latentu obrazu wzdłuż δ = z* − z0 [%]'); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(G, a('odl_wzgl'), 'o-', ms=3); ax[0, 1].set_title('odległość od z* / ||z* − z0||')
    ax[1, 0].plot(G, a('poza'), 'o-', ms=3, label='obraz'); ax[1, 0].plot(G, a('warunek_poza'), '-', label='warunek SD')
    ax[1, 0].set_title('odległość tokenów od naturalnej podprzestrzeni (zdjęcie ≈ 3)'); ax[1, 0].legend()
    ax[1, 1].plot(G, 100 * a('warunek_ulamek'), '-'); ax[1, 1].set_title('postęp warunku SD wzdłuż δ [%]')
    for x in ax.flat:
        x.set_xlabel('iteracja (jeden wygenerowany obraz)'); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'WYKRES_OPTYMALIZACJA.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
