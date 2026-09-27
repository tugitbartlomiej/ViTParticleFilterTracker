#!/usr/bin/env python3
"""Krok FastGAN dla RF-DETR LargeDeprecated na Edenie (PLAN.md): dla każdej kotwicy odczyt obecnego kodu (5 kierunków,
`kierunki.npz`, projekcja 3072 → 5) bieżącym detektorem, cel = obecny + przyrost przewidziany lokalnie, inwersja
generatora (start z z* poprzedniego obrotu tej kotwicy, jeśli jest, plus losowe restarty) i krok Gaussa–Newtona do
celu oraz do obecnego (kontrola). Logika jak `obrot_lokalny.fastgan` w `petla_uczenia_koopman_fastgan`.

    python fastgan_large.py --zadanie zadanie.json
zadanie.json: {"wagi", "kierunki", "manifest", "fastgan", "vendor", "out", "poprzedni" (katalog lub null),
               "obrot", "kotwice": [{"wiersz", "obraz", "indeks_manifestu", "przyrost": [5]}]}
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zadanie', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    z = json.loads(a.zadanie.read_text(encoding='utf-8'))
    import torch
    import czastki
    import generowanie as gen
    from rfdetr import RFDETRLargeDeprecated
    t0 = time.time()
    device = a.device
    wrapper = RFDETRLargeDeprecated(pretrain_weights=z['wagi'], num_classes=1, amp=False, device=device)
    mc = wrapper.model_config
    czastki.configure(mc.resolution, 768, (mc.resolution // mc.patch_size,) * 2)
    net = wrapper.model.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    projection = gen.DescriptorProjection.from_npz(z['kierunki'])
    skala = projection.target_scale
    skala_t = torch.tensor(skala, dtype=torch.float32, device=device)
    generator, _ = gen.load_fastgan(z['fastgan'], z['vendor'], device)
    out = Path(z['out']); out.mkdir(parents=True, exist_ok=True)
    k = int(z['obrot'])
    rap = {}
    with gen._generation_numerics(device), czastki.DifferentiableReader(net, wrapper.means, wrapper.stds) as reader:
        for kot in z['kotwice']:
            t1 = time.time()
            i = int(kot['wiersz'])
            kd = out / f'kotwica_{i:02d}'; kd.mkdir(parents=True, exist_ok=True)
            anchor_image, anchor_mask, _ = gen._load_anchor((z['manifest'], int(kot['indeks_manifestu'])), device)
            with torch.no_grad():
                cur = projection.project_torch(reader.descriptors(anchor_image, anchor_mask))[0].cpu().numpy().astype(np.float64)
            cel = cur + np.asarray(kot['przyrost'], float)
            starty = []
            poprz = Path(z['poprzedni']) / f'kotwica_{i:02d}' / 'inwersja_best.npz' if z.get('poprzedni') else None
            if poprz is not None and poprz.exists():
                p = np.load(poprz)
                starty.append((torch.tensor(p['z'], dtype=torch.float32, device=device), int(p['noise_seed']), 'z*_poprzedni'))
            restarty = 2 if starty else 6
            kand = []
            for r in range(restarty):
                rng = torch.Generator(device='cpu').manual_seed(42 + r + 100 * k)
                z0 = torch.randn((1, 256), generator=rng).to(device)
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, cur, z0, steps=100, lr=.05,
                                          noise_seed=1042 + r)
                kand.append((float(res['best']['total']), res, 1042 + r, f'restart_{r}'))
            kand.sort(key=lambda x: x[0])
            doszlif = list(starty) + [(c[1]['z'].detach().clone(), c[2], c[3]) for c in kand[:1 if starty else 2]]
            zwyc = None
            for z0, szum, nazwa in doszlif:
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, cur, z0, steps=200, lr=.01,
                                          noise_seed=szum)
                if zwyc is None or res['best']['total'] < zwyc[0]['total']:
                    zwyc = (res['best'], res, szum, nazwa)
            b, res, szum, nazwa = zwyc
            z_star = res['z'].detach().clone()
            np.savez_compressed(kd / 'inwersja_best.npz', z=z_star.cpu().numpy(), code=res['code'].cpu().numpy(), noise_seed=szum)
            gen._save_image(kd / 'inwersja_best.png', res['rgba'][:, :3])
            inw = dict(start=nazwa, blad_skalowany=float(b['feature'] ** .5), anchor_l1=float(b['anchor_l1']),
                       iou=float(gen.shape_iou(res['rgba'][:, 3:4], anchor_mask)))

            def kod(zz):
                rgba = gen.fixed_generator_output(generator, zz, seed=szum)
                return projection.project_torch(reader.descriptors(rgba[:, :3], anchor_mask))[0], rgba

            def krok_gn(zz, cel_t, lam0=1e-2):
                zq = zz.detach().clone().requires_grad_(True)
                c, _ = kod(zq)
                r = (cel_t - c) / skala_t
                J = torch.stack([torch.autograd.grad(-(c[j] / skala_t[j]), zq, retain_graph=True)[0][0] for j in range(len(c))])
                r0 = float(r.norm() / np.sqrt(len(c))); JJt = J @ J.T; sl = float(torch.trace(JJt) / len(c)); lam = lam0
                for _ in range(8):
                    dz = -J.T @ torch.linalg.solve(JJt + lam * sl * torch.eye(len(c), device=J.device), r.detach())
                    with torch.no_grad():
                        c2, _ = kod(zz + dz[None]); r2 = float(((cel_t - c2) / skala_t).norm() / np.sqrt(len(c)))
                    if r2 < r0:
                        return (zz + dz[None]).detach(), r0, r2
                    lam *= 4
                return zz.detach(), r0, r0

            ramiona = {}
            with gen._frozen_eval(generator, reader.net):
                for arm, c_arm in (('target', cel), ('current', cur)):
                    cel_t = torch.tensor(c_arm, dtype=torch.float32, device=device)
                    zz = z_star.clone(); it = 0
                    for it in range(12):
                        zz, r0, r2 = krok_gn(zz, cel_t)
                        if r2 >= r0 or r2 < .02:
                            break
                    with torch.no_grad():
                        c, rgba = kod(zz)
                    gen._save_image(kd / f'{arm}.png', rgba[:, :3]); gen._save_image(kd / f'{arm}_mask.png', rgba[:, 3:4])
                    ramiona[arm] = dict(reszta=float(((cel_t - c) / skala_t).norm() / np.sqrt(len(c))), iteracje=it + 1,
                                        kod=c.cpu().numpy().tolist(), rgb=rgba[:, :3].detach().cpu())
            l1 = float((ramiona['target'].pop('rgb') - ramiona['current'].pop('rgb')).abs().mean())
            rap[str(i)] = dict(obraz=kot['obraz'], obecny=cur.tolist(), cel=cel.tolist(), inwersja=inw, ramiona=ramiona,
                               l1_rgb_cel_vs_obecny=l1, sekundy=time.time() - t1)
            print(f'kotwica {i}: inwersja {inw["blad_skalowany"]:.3f} ({nazwa}), reszta cel {ramiona["target"]["reszta"]:.3f}, '
                  f'L1 cel-obecny {l1:.4f} ({time.time() - t1:.0f} s)', flush=True)
            (out / 'RAPORT_FASTGAN.json').write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'gotowe {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
