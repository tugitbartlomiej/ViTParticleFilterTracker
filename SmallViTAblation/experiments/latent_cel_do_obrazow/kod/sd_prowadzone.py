"""Etap 5: SD prowadzone sztucznym latentem, tak jak w metodzie użytkownika (maly_vit_sd_prowadzenie_tozsamoscia/
prowadzenie.py i vit_sd_zgodnosc90): start od prawdziwego obrazu celu (img2img, latent VAE zaszumiony do poziomu
`sila`), w każdym kroku DDIM przewidywany czysty obraz x̂0 -> dekoder VAE -> ViT k1000 -> latent z; energia
E = średnia (z − z_cel)^2, kilka kroków gradientu po x̂0 (krok eta · σ_t, gradient znormalizowany), potem DDIM dalej;
ostatnie kroki bez prowadzenia, żeby obraz wrócił na rozmaitość obrazów SD. Kilka rund: wynik rundy jest startem
następnej. Ocena na końcowym obrazie uint8 (cv2 INTER_AREA, jak w uczeniu): kierunek i ułamek zrealizowanej korekty,
detekcja z etykietą celu (img2img zachowuje układ), strata detekcji.

Warianty: warunek SD = latent obrazu z0 z prowadzeniem; z0 bez prowadzenia (kontrola samego przejścia przez SD);
warunek = z_cel z prowadzeniem. Uruchomienie z SmallViTAblation:
F:/Python/python.exe experiments/latent_cel_do_obrazow/kod/sd_prowadzone.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import korekta_latentu as KL
import silnik
from model_snapshot import dekoduj

N_CELOW, RUNDY, KROKI, SILA, ITER, ETA, KONIEC = 4, 2, 30, 0.5, 2, 0.15, 5
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def normalizacja(st, cele, n_obr=400, r=64, kroki=600):
    """Najbardziej naturalny latent o tej samej detekcji: z' = z_cel + K c, K = baza jądra W2 (576 na token), więc
    W2 z' = W2 z_cel i wyjście zamrożonej końcówki jest takie samo. c minimalizuje odległość tokenów od podprzestrzeni
    PCA-r tokenów prawdziwych obrazów (400 obrazów manifestu przy k1000) plus Mahalanobis wewnątrz niej i karę za
    wyjście poza zakresy z checkpointu; na końcu przycięcie do zakresów (mierzona zmiana detekcji)."""
    rows = KL.manifest(); rng = np.random.default_rng(0)
    T = []
    for i in range(0, len(rows), max(1, len(rows) // n_obr)):
        rgb, _ = KL.obraz_manifestu(rows[i])
        z = st.przebieg(rgb)['z']
        T.append(z[torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)])
    T = torch.cat(T)
    mu = T.mean(0); U_, S_, Vt = torch.linalg.svd(T - mu, full_matrices=False)
    U = Vt[:r].T; s = S_[:r] / (len(T) - 1) ** .5
    def nienat(z):
        y = z - mu; p = y @ U
        return dict(poza=float(((y - p @ U.T) ** 2).sum(1).sqrt().mean()), maha=float(((p / s) ** 2).sum(1).sqrt().mean()))
    ref = nienat(T)
    _, _, Vw = torch.linalg.svd(st.blok.mlp.fc2.weight.detach().double(), full_matrices=True)
    K = Vw[192:].T.float()                                                   # [768, 576]
    lo, hi = st.h_min, st.h_max
    out, rap = {}, dict(prawdziwe_tokeny=ref, r=r, opis='poza = średnia odległość tokenu od podprzestrzeni PCA-r '
                                                        'prawdziwych tokenów; maha = Mahalanobis wewnątrz niej')
    rows_m = KL.manifest()
    for c in cele:
        kn = np.load(Path(c['kat']) / 'KONIEC.npz')
        z0 = torch.from_numpy(kn['z_baza']).to(st.dev); zc = z0 + torch.from_numpy(kn['d_najlepszy']).to(st.dev)
        cc = torch.zeros(600, 576, device=st.dev, requires_grad=True)
        opt = torch.optim.Adam([cc], lr=0.5)
        for k in range(kroki):
            z = zc + cc @ K.T
            y = z - mu; p = y @ U
            L = ((y - p @ U.T) ** 2).sum(1).mean() + .1 * ((p / s) ** 2).sum(1).mean()                 + 10 * (((lo - z).clamp(min=0) ** 2) + ((z - hi).clamp(min=0) ** 2)).sum(1).mean()
            opt.zero_grad(); L.backward(); opt.step()
        with torch.no_grad():
            zn = torch.minimum(torch.maximum(zc + cc @ K.T, lo), hi)
            rgb, ram = KL.obraz_manifestu(rows_m[c['i']])
            b = st.przebieg(rgb); cel_t = KL.cel_z_ramek(ram, st.dev)
            mc = st.ocena(st.ogon(zc[None], b['r'][None]), ram, cel_t); mn = st.ocena(st.ogon(zn[None], b['r'][None]), ram, cel_t)
            m0 = st.ocena(b['o'][None], ram, cel_t)
        out[c['i']] = zn.cpu().numpy()
        rap[c['i']] = dict(obraz=dict(**nienat(z0), iou=m0['iou_top'], pewnosc=m0['pewnosc_top']),
                           sztuczny=dict(**nienat(zc), iou=mc['iou_top'], pewnosc=mc['pewnosc_top'], odl=float((zc - z0).norm())),
                           znormalizowany=dict(**nienat(zn), iou=mn['iou_top'], pewnosc=mn['pewnosc_top'], odl=float((zn - z0).norm())),
                           W2_zmiana_wzgl=float(((zc - z0) @ st.blok.mlp.fc2.weight.T).norm() / (z0 @ st.blok.mlp.fc2.weight.T).norm()))
        log('normalizacja', c['i'], {k: {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else round(v, 3) for k, v in rap[c['i']].items()})
    rap = {str(k): v for k, v in rap.items()}
    (OUT / 'ETAP5_NORMALIZACJA.json').write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    return dict(z_norm=out, raport=rap)


def main():
    cele = json.loads((OUT / 'ETAP1_CELE.json').read_text(encoding='utf-8'))['cele'][:N_CELOW]
    st = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'])
    BOX = next((a[4:] for a in sys.argv[1:] if a.startswith('box:')), None)
    TRYB_NAT = 'nat' in sys.argv[1:] or BOX is not None
    NAT = None if TRYB_NAT else normalizacja(st, cele)
    net = st.net
    tr = silnik.zaladuj_model()
    vae = tr.vae
    try:
        vae.disable_tiling()
    except Exception:
        pass
    if hasattr(vae, 'enable_gradient_checkpointing'):
        vae.enable_gradient_checkpointing()
    sf = float(vae.config.scaling_factor)
    from diffusers import DDIMScheduler
    sch = DDIMScheduler.from_config(tr.scheduler_config, clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    pred_typ = sch.config.get('prediction_type', 'epsilon')
    log('SD gotowe, prediction_type', pred_typ)
    dev = tr.device
    rows = KL.manifest()
    (OUT / 'sd_prowadzone').mkdir(exist_ok=True)

    def latent_vit_grad(img11):
        x = F.interpolate((img11 + 1) / 2, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        net.zapisuj_latent(True)
        try:
            net(x.float())
            return net._latent[0].float()
        finally:
            net.zapisuj_latent(False)

    def eps_z_x0(x, x0, a):
        return (x - a.sqrt() * x0) / (1 - a).sqrt()

    W2 = st.blok.mlp.fc2.weight.detach().float()

    def rysuj(start_rgb, warunek, z_cel, prowadz, ziarno=0, widoczna=False):
        x_in = torch.from_numpy(cv2.resize(start_rgb, (576, 384), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None]
        x_in = x_in.float().to(dev) / 255
        with torch.no_grad():
            lat0 = vae.encode(x_in * 2 - 1).latent_dist.mean * sf
        sch.set_timesteps(KROKI, device=dev)
        ts = sch.timesteps
        i0 = int(round(KROKI * (1 - SILA)))
        g = torch.Generator(device=dev).manual_seed(ziarno)
        szum = torch.randn(lat0.shape, generator=g, device=dev)
        x = sch.add_noise(lat0, szum, ts[i0:i0 + 1])
        war = torch.from_numpy(warunek.astype(np.float32))[None].to(dev)
        zc = torch.from_numpy(z_cel.astype(np.float32)).to(dev)
        ac = sch.alphas_cumprod.to(dev)
        E_hist = []
        for i in range(i0, KROKI):
            t = ts[i]
            a = ac[t].float()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pr = tr.model(sch.scale_model_input(x, t), t, war).float()
            if pred_typ == 'v_prediction':
                e = a.sqrt() * pr + (1 - a).sqrt() * x
            else:
                e = pr
            x0 = (x - (1 - a).sqrt() * e) / a.sqrt()
            if prowadz and i < KROKI - KONIEC:
                sig = float((1 - a).sqrt())
                for _ in range(ITER):
                    v = x0.detach().requires_grad_(True)
                    img = vae.decode(v / sf).sample.clamp(-1, 1)
                    z = latent_vit_grad(img)
                    E = (((z - zc) @ W2.T) ** 2).mean() if widoczna else ((z - zc) ** 2).mean()
                    gr, = torch.autograd.grad(E, v)
                    gr = gr / (gr.pow(2).mean().sqrt() + 1e-8)
                    x0 = (v - ETA * sig * gr).detach()
                    E_hist.append(float(E))
                e = eps_z_x0(x, x0, a)
            x = sch.step(e, t, x, eta=0.).prev_sample
        with torch.no_grad():
            img = ((vae.decode(x / sf).sample.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.resize((img * 255).round().astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA), E_hist

    def ocen(img, z0, d, ram, cel_t):
        b = st.przebieg(img)
        z = b['z'].cpu().numpy()
        dz = (z - z0).reshape(-1); df = d.reshape(-1)
        m = st.ocena(b['o'][None], ram, cel_t)
        return dict(cos_do_korekty=float(dz @ df / (np.linalg.norm(dz) * np.linalg.norm(df) + 1e-12)),
                    alfa=float(dz @ df / (df @ df)), zmiana_wzgl=float(np.linalg.norm(dz) / np.linalg.norm(z0)),
                    iou_top=m['iou_top'], pewnosc_top=m['pewnosc_top'], strata=m['strata'], fp=m['fp'])

    wyn = []
    warianty = [('z0_prowadzone', 'z0', True, False), ('z0_kontrola', 'z0', False, False),
                ('z0_prowadzone_fc2', 'z0', True, True), ('norm_prowadzone_fc2', 'norm', True, True)]
    if TRYB_NAT:
        warianty = [('nat_warunek', 'nat', False, False), ('nat_warunek_prowadzone', 'nat', True, False),
                    ('nat_warunek_prowadzone_fc2', 'nat', True, True)]
    plik = f'ETAP9_SD_{BOX}.json' if BOX else ('ETAP7_SD_NATURALNY.json' if TRYB_NAT else 'ETAP5_SD_PROWADZONE.json')
    for c in cele:
        rgb, ram = KL.obraz_manifestu(rows[c['i']])
        kn = np.load(Path(c['kat']) / 'KONIEC.npz')
        z0 = kn['z_baza']; d = kn['d_najlepszy']; zcel = z0 + d
        if TRYB_NAT:
            zcel = np.load(OUT / (f'zb_m{c["i"]:04d}_{BOX}.npy' if BOX else f'zn_m{c["i"]:04d}_r256_l0.1.npy')); d = zcel - z0
        cel_t = KL.cel_z_ramek(ram, st.dev)
        m0 = ocen(rgb, z0, d, ram, cel_t)
        for nazwa, war, prow, wid in warianty:
            start = rgb
            for r in range(1, RUNDY + 1):
                warunek = z0 if war == 'z0' else (zcel if war == 'nat' else NAT['z_norm'][c['i']])
                img, Eh = rysuj(start, warunek, zcel, prow, ziarno=r, widoczna=wid)
                cv2.imwrite(str(OUT / 'sd_prowadzone' / f'm{c["i"]:04d}_{nazwa}{"_" + BOX if BOX else ""}_r{r}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                m = ocen(img, z0, d, ram, cel_t)
                wyn.append(dict(cel=c['i'], wariant=nazwa, runda=r, E_start=Eh[0] if Eh else None, E_koniec=Eh[-1] if Eh else None,
                                obraz_wyjsciowy=m0, **m))
                log(c['i'], nazwa, r, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
                start = img
        (OUT / plik).write_text(json.dumps(dict(wyniki=wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    pods = {}
    for nazwa, *_ in warianty:
        for r in range(1, RUNDY + 1):
            w = [x for x in wyn if x['wariant'] == nazwa and x['runda'] == r]
            if not w:
                continue
            pods[f'{nazwa}_r{r}'] = {k: float(np.mean([x[k] if x[k] is not None else np.nan for x in w]))
                                     for k in ('cos_do_korekty', 'alfa', 'zmiana_wzgl', 'iou_top', 'pewnosc_top', 'strata', 'fp')}
    pods['obraz_wyjsciowy'] = {k: float(np.mean([x['obraz_wyjsciowy'][k] if x['obraz_wyjsciowy'][k] is not None else np.nan
                                                 for x in wyn if x['wariant'] == warianty[0][0] and x['runda'] == 1]))
                               for k in ('iou_top', 'pewnosc_top', 'strata')}
    out = dict(wyniki=wyn, podsumowanie=pods, normalizacja=NAT['raport'] if NAT else 'cel = korekta naturalna r256 lam 0,1 (etap 6)', parametry=dict(kroki=KROKI, sila=SILA, iteracje=ITER, eta=ETA, koniec=KONIEC,
                                                             rundy=RUNDY, energia='mean((z - z_cel)^2), pełny latent 600x768'),
               opis='cos_do_korekty = cos(z(obraz SD) − z(cel), δ); alfa = ułamek korekty zrealizowany wzdłuż δ; detekcja z '
                    'etykietą obrazu celu (img2img zachowuje układ)')
    (OUT / plik).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in pods.items():
        log(k, {a: round(b, 4) for a, b in v.items()})
    log('KONIEC')


if __name__ == '__main__':
    main()
