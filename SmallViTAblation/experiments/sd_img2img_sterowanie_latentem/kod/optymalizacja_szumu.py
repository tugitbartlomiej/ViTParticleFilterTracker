"""Dobór szumu startowego SD tak, żeby obraz od początku odszumiania szedł w stronę sztucznego latentu
(pomysł użytkownika 2026-10-08; odpowiada optymalizacji szumu początkowego dyfuzji pod nagrodę, np. ReNO, InitNO, DOODL).

Jeden obraz: klatka TRAIN 0055/4019 (j = 10389), z* jak w optymalizacja_warunku.py, δ = z* − z0.
Faza 1 (losowe szukanie): N_LOS losowych szumów, warunek c0 = Π(z0); wybór szumu, którego obraz ma największy postęp
latentu wzdłuż δ.
Faza 2 (iteracje): od najlepszego szumu gradient przez cały łańcuch DDIM (KROKI kroków leading, eta 0, checkpoint UNet)
-> VAE -> ViT K8000; strata = −postęp wzdłuż δ + 0,1 odejście w bok (jak optymalizacja_warunku.py). Szum po każdym
użyciu rzutowany na normę sqrt(4·48·72), czyli na powłokę typowego szumu Gaussa, żeby SD dostawało szum ze swojego
zakresu. Tryb `szum`: parametrem jest tylko szum, warunek stały c0. Tryb `szum_warunek`: szum i warunek (warunek
przez rzut Π do zakresu warunków SD). Stop: CIERPLIWOSC iteracji bez lepszego obrazu albo MAX_IT.
Sprawdzenie: najlepszy szum (i warunek) generowany też w 50 krokach DDIM.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/optymalizacja_szumu.py [szum|szum_warunek]
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
TRYB = sys.argv[1] if len(sys.argv) > 1 else 'szum'
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / f'optymalizacja_{TRYB}'
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

KROKI, LR_SZUM, LR_WAR, LAM, CIERPLIWOSC, MAX_IT, N_LOS = 20, 0.05, 0.05, 0.1, 15, 150, 32
KSZTALT = (1, 4, 48, 72)
NORMA = float(np.sqrt(np.prod(KSZTALT)))
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

    rzut_szumu = lambda h: h * (NORMA / h.norm())
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

    def generuj(h, c, kroki, grad):
        sch.set_timesteps(kroki, device=dev); ts = sch.timesteps
        krok = sch.config.num_train_timesteps // kroki
        with torch.set_grad_enabled(grad):
            for t in ts:
                e = checkpoint(unet, h, t, c, use_reentrant=False) if grad else unet(h, t, c)
                a = ac[t]; tp = int(t) - krok
                ap = ac[tp] if tp >= 0 else torch.tensor(a_koniec, device=dev)
                x0 = (h - (1 - a).sqrt() * e) / a.sqrt()
                h = ap.sqrt() * x0 + (1 - ap).sqrt() * e
            return ((vae.decode(h / skala).sample + 1) / 2).clamp(0, 1)

    def do_uint8(img01):
        x = F.interpolate(img01.detach().float(), size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        return (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    kat_los = OUT / 'losowe_szukanie'; kat_los.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'ZDJECIE_CELU.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    c0 = rzut(z0)

    # faza 1: losowe szukanie szumu
    losowe = []
    for s in range(N_LOS):
        g = torch.Generator(device=dev).manual_seed(1000 + s)
        h = torch.randn(KSZTALT, generator=g, device=dev)
        img = do_uint8(generuj(h, c0, KROKI, False))
        cv2.imwrite(str(kat_los / f's{s:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        m = miary(st.przebieg(img)['z']); losowe.append(dict(ziarno=1000 + s, **m))
        log(f'losowy szum {s:2d}', {a: round(v, 3) for a, v in m.items()})
    s_najl = int(np.argmax([x['ulamek'] for x in losowe]))
    log('faza 1: najlepszy losowy szum', s_najl, 'postęp', round(losowe[s_najl]['ulamek'], 4), 'średnia', round(float(np.mean([x['ulamek'] for x in losowe])), 4))

    # faza 2: gradient od najlepszego losowego szumu
    g = torch.Generator(device=dev).manual_seed(1000 + s_najl)
    h_par = torch.randn(KSZTALT, generator=g, device=dev).requires_grad_(True)
    c_par = z0.clone().requires_grad_(TRYB == 'szum_warunek')
    grupy = [dict(params=[h_par], lr=LR_SZUM)] + ([dict(params=[c_par], lr=LR_WAR)] if TRYB == 'szum_warunek' else [])
    opt = torch.optim.Adam(grupy)
    wyn, Zimg, Zwar = [], [], []
    najl, bez_poprawy = None, 0
    for it in range(MAX_IT):
        h = rzut_szumu(h_par); c = rzut(c_par) if TRYB == 'szum_warunek' else c0
        img01 = generuj(h, c, KROKI, True)
        z = latent_vit(img01)
        dz = z - z0; u = (dz * d).sum() / dd
        strata = -u + LAM * (dz - u * d).norm() / d.norm()
        opt.zero_grad(); strata.backward(); opt.step()
        img = do_uint8(img01)
        zc = st.przebieg(img)['z']
        m = miary(zc); mw = miary(c.detach())
        rek = dict(generacja=it, strata=float(strata), **m, warunek_ulamek=mw['ulamek'], warunek_poza=mw['poza'],
                   zmiana_szumu=float((h.detach() - rzut_szumu(torch.randn(KSZTALT, generator=torch.Generator(device=dev).manual_seed(1000 + s_najl), device=dev))).norm() / NORMA))
        if najl is None or m['ulamek'] > najl['ulamek']:
            najl = dict(rek); h_najl, c_najl = h.detach().clone(), c.detach().clone(); bez_poprawy = 0
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
    spr = []
    for kroki in (KROKI, 50):
        img = do_uint8(generuj(h_najl, c_najl, kroki, False))
        cv2.imwrite(str(OUT / f'NAJLEPSZY_kroki{kroki}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        spr.append(dict(kroki=kroki, **miary(st.przebieg(img)['z'])))
        log('sprawdzenie najlepszego szumu', spr[-1])
    np.savez_compressed(OUT / 'LATENTY.npz', obrazy=np.stack(Zimg), warunki=np.stack(Zwar), z0=z0.cpu().numpy().astype(np.float16),
                        z_sztuczny=zs.cpu().numpy().astype(np.float16), szum_najlepszy=h_najl.cpu().numpy(), warunek_najlepszy=c_najl.cpu().numpy().astype(np.float16))
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(tryb=TRYB, losowe=losowe, najlepszy_losowy=s_najl, wyniki=wyn, najlepsza=najl, sprawdzenie=spr,
                                    parametry=dict(kroki=KROKI, lr_szum=LR_SZUM, lr_war=LR_WAR, lam=LAM, cierpliwosc=CIERPLIWOSC, max_it=MAX_IT,
                                                   n_los=N_LOS, cel_j=CEL_J, z_sztuczny=str(Z_SZT)), opis=__doc__), indent=1, ensure_ascii=False), encoding='utf-8')
    wykres(wyn, losowe)
    log('najlepsza', najl)
    subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('grafika_latentow.py')), OUT.name], check=False)
    film()
    log('KONIEC')


def wykres(wyn, losowe):
    G = np.arange(len(wyn)); a = lambda k: np.array([r[k] for r in wyn])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(G, 100 * a('ulamek'), 'o-', ms=3, label='cały przyrost'); ax[0, 0].plot(G, 100 * a('ulamek_widoczny'), 's-', ms=3, label='część widoczna dla fc2')
    ax[0, 0].plot(G, 100 * np.maximum.accumulate(a('ulamek')), 'k--', label='najlepszy dotąd')
    ax[0, 0].axhline(100 * np.mean([x['ulamek'] for x in losowe]), c='gray', ls=':', label=f'średnio {len(losowe)} losowych szumów')
    ax[0, 0].set_title(f'postęp latentu obrazu wzdłuż δ [%] (tryb {TRYB})'); ax[0, 0].legend(fontsize=8)
    ax[0, 1].hist([100 * x['ulamek'] for x in losowe], bins=12, color='gray'); ax[0, 1].axvline(100 * wyn[0]['ulamek'], c='r', label='start fazy 2')
    ax[0, 1].set_title('faza 1: postęp obrazów z losowych szumów [%]'); ax[0, 1].legend(fontsize=8); ax[0, 1].set_xlabel('postęp [%]')
    ax[1, 0].plot(G, a('poza'), 'o-', ms=3, label='obraz'); ax[1, 0].plot(G, a('warunek_poza'), '-', label='warunek SD')
    ax[1, 0].set_title('odległość tokenów od naturalnej podprzestrzeni (zdjęcie ≈ 3)'); ax[1, 0].legend()
    ax[1, 1].plot(G, a('zmiana_szumu'), '-'); ax[1, 1].set_title('zmiana szumu od najlepszego losowego (/||szum||)')
    for x in (ax[0, 0], ax[1, 0], ax[1, 1]):
        x.set_xlabel('iteracja (jeden wygenerowany obraz)')
    for x in ax.flat:
        x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'WYKRES_OPTYMALIZACJA.png', dpi=130); plt.close(fig)


def film():
    w = json.loads((OUT / 'WYNIK.json').read_text(encoding='utf-8'))['wyniki']
    tmp = OUT / '_klatki_filmu'; tmp.mkdir(exist_ok=True)
    for r in w:
        g = r['generacja']; im = cv2.imread(str(OUT / 'obrazy' / f'g{g:02d}.png'))
        cv2.rectangle(im, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(im, f'iteracja {g:3d}   postep {100 * r["ulamek"]:4.1f}%   widoczny fc2 {100 * r["ulamek_widoczny"]:4.1f}%',
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(tmp / f'k{g:04d}.png'), im)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', str(tmp / 'k%04d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', str(OUT / 'FILM_OPTYMALIZACJA_30fps.mp4')], check=False)
    for p in tmp.glob('*.png'):
        p.unlink()
    tmp.rmdir()


if __name__ == '__main__':
    main()
