"""Pętla generowania i szukania z losowego szumu (zlecenie użytkownika 2026-10-08).

Jeden obraz: klatka TRAIN 0055/4019 (j = 10389), sztuczny latent z* jak w img2img_sterowanie.py.
Każda generacja g = 0..N_GEN-1:
  1. SD K8000 od nowego losowego szumu (ziarno g), DDIM 50 leading, warunek c_g (latent sterujący), bez gradientu
     wewnątrz SD;
  2. latent obrazu z_g = ViT K8000(obraz);
  3. c_{g+1} = Π(c_g + ETA (z* − z_g)): poprawka o to, czego obrazowi brakuje do z*;
  Π = zakres, w którym SD widziało warunki w uczeniu: każdy token z resztą poza podprzestrzenią PCA-64 prawdziwych
  tokenów przyciętą do percentyla 99 prawdziwych tokenów, potem każdy kanał przycięty do [min, max] prawdziwych tokenów
  (400 losowych klatek puli TRAIN, 100 tokenów z klatki). Start c_0 = z0 (latent zdjęcia celu).
Najlepszy obraz = najmniejsza odległość ||z_g − z*||. Zapis: każdy obraz PNG, latent obrazu i latent sterujący
(600 x 768, float16) dla każdej generacji, wykresy.
Wersja 2 (argument elita): rzut powtarzany do skutku, krok liczony od najlepszego dotąd obrazu, krok maleje bez poprawy.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/petla_szum.py [calka|elita]
"""
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRYB = sys.argv[1] if len(sys.argv) > 1 else 'calka'      # calka: wersja 1; elita: wersja 2 (poprawki od najlepszego)
OUT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / ('petla_szum' if TRYB == 'calka' else 'petla_szum_elita')
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
from PIL import Image

import korekta_latentu as KL

N_GEN, ETA, KROKI = 50, 0.5, 50
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
    sp = importlib.util.spec_from_file_location('model_sd_k8000', SD / 'kod' / 'model.py')
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    from diffusers import DDIMScheduler
    ck = torch.load(SD / 'best_sd.pt', map_location='cpu', weights_only=False)
    den = M.Denoiser().eval(); vae = M.new_vae().eval()
    den.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    den.to(dev).requires_grad_(False); vae.to(dev).requires_grad_(False)
    skala = float(ck['scale'])
    sch = DDIMScheduler.from_config(M.schedule().config, timestep_spacing='leading')

    rng = np.random.default_rng(0)
    Tk = torch.cat([st.przebieg(klatka(int(j)))['z'][torch.from_numpy(rng.choice(600, 100, replace=False)).to(dev)]
                    for j in rng.choice(len(MAN), 400, replace=False)])
    mu = Tk.mean(0); _, _, Vt = torch.linalg.svd(Tk - mu, full_matrices=False); U = Vt[:64].T
    reszta = lambda z: (z - mu) - ((z - mu) @ U) @ U.T
    poza = lambda z: float(reszta(z).norm(dim=1).mean())
    prog = torch.quantile(reszta(Tk).norm(dim=1), .99)
    lo, hi = Tk.min(0).values, Tk.max(0).values

    def rzut(c):
        # wersja 1: jedno skurczenie reszty i przycięcie kanałów (przycięcie potrafi wypchnąć resztę z powrotem);
        # wersja 2: naprzemiennie 20 razy, na końcu skurczenie, więc reszta każdego tokenu na pewno <= prog
        for _ in range(1 if TRYB == 'calka' else 20):
            y = c - mu; p = (y @ U) @ U.T; r = y - p
            r = r * torch.clamp(prog / (r.norm(dim=1, keepdim=True) + 1e-9), max=1.0)
            c = torch.minimum(torch.maximum(mu + p + r, lo), hi)
        if TRYB != 'calka':
            y = c - mu; p = (y @ U) @ U.T; r = y - p
            c = mu + p + r * torch.clamp(prog / (r.norm(dim=1, keepdim=True) + 1e-9), max=1.0)
        return c

    W2 = st.blok.mlp.fc2.weight.detach().float()
    rgb0 = klatka(CEL_J)
    z0 = st.przebieg(rgb0)['z']
    zs = torch.from_numpy(np.load(Z_SZT)).to(dev)
    d = zs - z0; d_w = d @ W2.T
    log('cel', MAN[CEL_J]['film'], MAN[CEL_J]['frame'], '||δ||/||z0||', round(float(d.norm() / z0.norm()), 2), 'poza z0', round(poza(z0), 2),
        'poza z*', round(poza(zs), 2), 'próg poza (p99 tokenu)', round(float(prog), 2), 'poza Π(z*)', round(poza(rzut(zs)), 2))

    def miary(z):
        dz = z - z0
        return dict(odl_do_z=float((z - zs).norm()), odl_wzgl=float((z - zs).norm() / d.norm()),
                    ulamek=float((dz * d).sum() / (d * d).sum()), ulamek_widoczny=float(((dz @ W2.T) * d_w).sum() / (d_w * d_w).sum()),
                    reszta_od_osi=float((dz - ((dz * d).sum() / (d * d).sum()) * d).norm() / d.norm()), poza=poza(z))

    @torch.no_grad()
    def generuj(c, ziarno):
        sch.set_timesteps(KROKI, device=dev)
        g = torch.Generator(device=dev).manual_seed(ziarno)
        h = torch.randn((1, 4, 48, 72), generator=g, device=dev) * sch.init_noise_sigma
        for t in sch.timesteps:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = den(sch.scale_model_input(h, t), t, c[None].float())
            h = sch.step(e.float(), t, h, eta=0.0).prev_sample
        x = ((vae.decode(h / skala).sample.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.resize((x * 255).round().astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA)

    kat = OUT / 'obrazy'; kat.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'ZDJECIE_CELU.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    Zimg = np.zeros((N_GEN, 600, 768), np.float16); Zwar = np.zeros((N_GEN, 600, 768), np.float16)
    wyn, najl = [], None
    c = rzut(z0); eta = ETA; c_najl = z_najl = None
    for g in range(N_GEN):
        img = generuj(c, g)
        z = st.przebieg(img)['z']
        Zimg[g] = z.cpu().numpy().astype(np.float16); Zwar[g] = c.cpu().numpy().astype(np.float16)
        cv2.imwrite(str(kat / f'g{g:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        m = miary(z); mw = miary(c)
        rek = dict(generacja=g, **m, warunek_ulamek=mw['ulamek'], warunek_poza=mw['poza'], warunek_odl_wzgl=mw['odl_wzgl'])
        poprawa = najl is None or m['odl_do_z'] < najl['odl_do_z']
        if poprawa:
            najl = dict(rek); c_najl, z_najl = c.clone(), z.clone()
        rek['najlepsza_dotad'] = najl['generacja']; rek['krok_eta'] = eta
        wyn.append(rek)
        log(f'g{g:02d}', {a: round(v, 3) for a, v in rek.items() if isinstance(v, float)}, 'najlepsza', najl['generacja'])
        if TRYB == 'calka':
            c = rzut(c + ETA * (zs - z))
        else:
            # wersja 2: krok zawsze od najlepszego dotąd (latent sterujący i latent jego obrazu), nowy szum w każdej
            # generacji; bez poprawy krok maleje (x0,7, nie mniej niż 0,05), po poprawie wraca do ETA
            eta = ETA if poprawa else max(eta * .7, .05)
            c = rzut(c_najl + eta * (zs - z_najl))
    np.savez_compressed(OUT / 'LATENTY.npz', obrazy=Zimg, warunki=Zwar, z0=z0.cpu().numpy().astype(np.float16), z_sztuczny=zs.cpu().numpy().astype(np.float16))
    (OUT / 'WYNIK.json').write_text(json.dumps(dict(wyniki=wyn, najlepsza=najl, parametry=dict(tryb=TRYB, n_gen=N_GEN, eta=ETA, kroki=KROKI, cel_j=CEL_J,
                                    prog_poza=float(prog), z_sztuczny=str(Z_SZT)), opis=__doc__), indent=1, ensure_ascii=False), encoding='utf-8')
    wykresy(wyn, najl, Zimg, Zwar, z0.cpu().numpy(), zs.cpu().numpy(), rgb0)
    log('najlepsza', najl)
    log('KONIEC')


def wykresy(wyn, najl, Zimg, Zwar, z0, zs, rgb0):
    G = np.arange(len(wyn))
    a = lambda k: np.array([r[k] for r in wyn])
    # 1. przebieg pętli
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(G, a('odl_wzgl'), 'o-', ms=3, label='obraz'); ax[0, 0].plot(G, np.minimum.accumulate(a('odl_wzgl')), 'k--', label='najlepszy dotąd')
    ax[0, 0].plot(G, a('warunek_odl_wzgl'), '-', alpha=.6, label='latent sterujący')
    ax[0, 0].set_title('odległość od sztucznego latentu / ||z* − z0||'); ax[0, 0].legend()
    ax[0, 1].plot(G, 100 * a('ulamek'), 'o-', ms=3, label='cały przyrost'); ax[0, 1].plot(G, 100 * a('ulamek_widoczny'), 's-', ms=3, label='część widoczna dla fc2')
    ax[0, 1].plot(G, 100 * a('warunek_ulamek'), '-', alpha=.6, label='latent sterujący'); ax[0, 1].axhline(0, c='gray', lw=.5)
    ax[0, 1].set_title('ułamek przyrostu δ = z* − z0 w latencie obrazu [%]'); ax[0, 1].legend()
    ax[1, 0].plot(G, a('poza'), 'o-', ms=3, label='obraz'); ax[1, 0].plot(G, a('warunek_poza'), '-', label='latent sterujący')
    ax[1, 0].set_title('odległość tokenów od naturalnej podprzestrzeni (zdjęcie ≈ 3)'); ax[1, 0].legend()
    ax[1, 1].plot(G, a('reszta_od_osi'), 'o-', ms=3); ax[1, 1].set_title('odejście od osi z0 → z* (względem ||δ||)')
    for x in ax.flat:
        x.set_xlabel('generacja'); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'WYKRES_PETLA.png', dpi=130); plt.close(fig)

    # 2. latent w płaszczyźnie: oś 1 = ułamek δ, oś 2 = najsilniejszy kierunek prostopadły (PCA reszt obrazów)
    zi = Zimg.reshape(len(wyn), -1).astype(np.float32); zw = Zwar.reshape(len(wyn), -1).astype(np.float32)
    zf0, df = z0.reshape(-1), (zs - z0).reshape(-1)
    def os1(Z):
        return (Z - zf0) @ df / (df @ df)
    R = (zi - zf0) - os1(zi)[:, None] * df
    _, _, vt = np.linalg.svd(R - R.mean(0), full_matrices=False); e2 = vt[0]
    os2 = lambda Z: ((Z - zf0) - os1(Z)[:, None] * df) @ e2 / np.linalg.norm(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(os1(zw), os2(zw), '-', c='tab:orange', alpha=.5, label='latent sterujący')
    sc = ax.scatter(os1(zi), os2(zi), c=G, cmap='viridis', s=25, label='latent obrazu (kolor = generacja)')
    ax.scatter([0], [0], c='k', marker='s', s=70, label='z0 (zdjęcie)'); ax.scatter([1], [0], c='r', marker='*', s=160, label='z* (sztuczny)')
    b = najl['generacja']; ax.scatter(os1(zi[b:b + 1]), os2(zi[b:b + 1]), facecolors='none', edgecolors='r', s=160, label=f'najlepszy (g{b})')
    fig.colorbar(sc, label='generacja'); ax.set_xlabel('ułamek przyrostu δ'); ax.set_ylabel('składowa prostopadła do δ (/||δ||)')
    ax.set_title('Latenty 600 × 768 obrazów pętli wobec osi z0 → z*'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'WYKRES_LATENT_PLASZCZYZNA.png', dpi=130); plt.close(fig)

    # 3. profil 768 kanałów (średnia po 600 tokenach): z0, z*, najlepszy obraz; mapa cieplna po generacjach
    k0, ks = z0.mean(0), zs.mean(0); kg = Zimg.astype(np.float32).mean(1)
    por = np.argsort(-(ks - k0))
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw=dict(height_ratios=[1, 1.3]))
    ax[0].plot(k0[por], lw=.8, c='k', label='z0 (zdjęcie)'); ax[0].plot(ks[por], lw=.8, c='r', label='z* (sztuczny)')
    ax[0].plot(kg[b][por], lw=.8, c='tab:green', label=f'najlepszy obraz g{b}')
    ax[0].set_title('768 kanałów latentu (średnia po 600 tokenach), kanały posortowane wg przyrostu z* − z0'); ax[0].legend(); ax[0].set_xlabel('kanał (posortowany)')
    D = (kg - k0)[:, por]; lim = np.abs(ks - k0).max()
    im = ax[1].imshow(np.vstack([(ks - k0)[por][None], D]), aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax[1].set_yticks([0] + list(range(1, len(wyn) + 1, 5))); ax[1].set_yticklabels(['z* − z0'] + [f'g{g}' for g in range(0, len(wyn), 5)])
    ax[1].set_title('zmiana kanałów obrazu wobec z0 w kolejnych generacjach (pierwszy wiersz: sztuczny przyrost)'); ax[1].set_xlabel('kanał (posortowany)')
    fig.colorbar(im, ax=ax[1]); fig.tight_layout(); fig.savefig(OUT / 'WYKRES_KANALY.png', dpi=130); plt.close(fig)

    # 4. pasek obrazów: zdjęcie, co 5. generacja, najlepszy
    def kafel(p, t):
        y = cv2.imread(str(p)) if not isinstance(p, np.ndarray) else cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        y = y.copy(); cv2.rectangle(y, (0, 0), (480, 28), (0, 0, 0), -1); cv2.putText(y, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        return cv2.resize(y, (320, 213))
    k = [kafel(rgb0, 'zdjecie celu')] + [kafel(OUT / 'obrazy' / f'g{g:02d}.png', f'g{g}  ulamek {100 * wyn[g]["ulamek"]:.1f}%') for g in range(0, len(wyn), 5)]
    k.append(kafel(OUT / 'obrazy' / f'g{b:02d}.png', f'NAJLEPSZY g{b} ulamek {100 * wyn[b]["ulamek"]:.1f}%'))
    while len(k) % 6:
        k.append(np.zeros_like(k[0]))
    cv2.imwrite(str(OUT / 'PASEK.jpg'), np.concatenate([np.concatenate(k[i:i + 6], 1) for i in range(0, len(k), 6)], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__':
    main()
