#!/usr/bin/env python3
"""Pokaz FastGAN v3: od zinwertowanego z* krok Gaussa-Newtona w strone celu Koopmana (2026-09-25).

Dlaczego: w pokaz_fastgan_inwersja.py ramiona od tego samego z* z Adamem (lr 0,01, 100 krokow) rozjezdzaja sie
chaotycznie: ramie "obecny" znajduje nizszy punkt, ramiona "cel" i "losowy" zostaja w z*, wiec roznica miedzy
obrazami nie odzwierciedla zamierzonego przyrostu. Tutaj krok jest LOKALNY i deterministyczny:
    r = (cel - kod(z)) / skala,  J = d kod_skalowany / dz  (5 x 256, autograd),
    dz = J^T (J J^T + lambda I)^-1 r   (Levenberga-Marquardta, krok najmniejszej normy),
z podwajaniem lambda, gdy reszta nie maleje. Kazde ramie startuje z tego samego z* i tego samego szumu warstw,
wiec roznice obrazow ramion pochodza wylacznie z roznic celow. Ramiona: cel Koopmana, obecny (kontrola),
losowy przyrost tej samej dlugosci (kontrola), target_x10_NIE_PROGNOZA (przyrost x10, jawnie nie prognoza).
Miary: reszta skalowana po kroku, osiagnieta zmiana kodu wzgledem ramienia "obecny" (cos, udzial), L1 RGB
miedzy obrazami ramion, IoU maski i L1 do kotwicy. Wejscia: katalog pokaz_fastgan_inwersja.py (z*, szum),
katalog pokaz_fastgan.py (directions.npz, current/target).

    <venv rfdetr>/python pokaz_fastgan_gn.py [--inwersja wyniki/fastgan_K73_krok0245_inwersja]
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PETLA = EXP / 'koopman_fastgan_petla'
sys.path.insert(0, str(PETLA))
PROBE_MANIFEST = PETLA / 'wyniki/local_1810448/run/probe/manifest.json'
FASTGAN = EXP / 'fastgan_narzedzie/wyniki/eden_20260923_50k/results_50k/checkpoint_050000.pth'
VENDOR = EXP / 'fastgan_narzedzie/vendor/deepfake_gi_fastGAN'
RUN = EXP / 'koopman_gesta_historia/wyniki/run_1810662/pretrain'
PROGI = dict(scaled_rmse=.25, shape_iou=.5, anchor_l1=.25)


def fmt(x):
    return 'brak' if x is None else f'{x:.2f}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inwersja', type=Path, default=HERE / 'wyniki/fastgan_K73_krok0245_inwersja')
    ap.add_argument('--zrodlo', type=Path, default=HERE / 'wyniki/fastgan_K73_krok0245')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--iteracje', type=int, default=12)
    ap.add_argument('--lam', type=float, default=1e-2, help='poczatkowe tlumienie LM (wzgledem sladu J J^T / 5)')
    ap.add_argument('--tol', type=float, default=.02, help='stop, gdy reszta skalowana (RMS) spadnie ponizej')
    ap.add_argument('--wzmocnienie', type=float, default=10.)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from czastki import DifferentiableReader
    import generowanie as gen
    from rfdetr import RFDETRNano

    zr = json.loads((args.zrodlo / 'REPORT.json').read_text(encoding='utf-8'))
    inw = json.loads((args.inwersja / 'REPORT.json').read_text(encoding='utf-8'))
    out = args.out or Path(str(args.zrodlo) + '_gn')
    out.mkdir(parents=True, exist_ok=True)
    ck = RUN / f'model_{zr["krok_c"]:07d}.pth'
    projection = gen.DescriptorProjection.from_npz(args.zrodlo / 'directions.npz')
    skala = projection.target_scale
    skala_t = None
    raport = dict(zrodlo=str(args.zrodlo), inwersja=str(args.inwersja), krok_c=zr['krok_c'], krok_cel=zr['krok_cel'],
                  przyrost_ogolny_w_odchyleniach=zr['przyrost_ogolny_w_odchyleniach'], skala_target=skala.tolist(),
                  ustawienia=dict(iteracje=args.iteracje, lam=args.lam, tol=args.tol, wzmocnienie=args.wzmocnienie, progi=PROGI),
                  detektor_sha256=gen._sha(ck), generator_sha256=gen._sha(FASTGAN), kotwice={})
    generator, gmeta = gen.load_fastgan(str(FASTGAN), str(VENDOR), args.device)
    wrapper = RFDETRNano(num_classes=1, resolution=384, pretrain_weights=str(ck), trust_checkpoint=True, amp=False)
    net = wrapper.model.model.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    with gen._generation_numerics(args.device) as numerics, DifferentiableReader(net, wrapper.means, wrapper.stds) as reader, \
            gen._frozen_eval(generator, reader.net):
        raport['numerics'] = numerics
        skala_t = torch.tensor(skala, dtype=torch.float32, device=args.device)
        for key, kot in zr['kotwice'].items():
            if key not in inw['kotwice']:
                print(f'kotwica {key}: brak inwersji, pomijam', flush=True); continue
            t1 = time.time()
            kd = out / f'kotwica_{int(key):02d}'; kd.mkdir(exist_ok=True)
            anchor_image, anchor_mask, _ = gen._load_anchor((str(PROBE_MANIFEST), kot['indeks_manifestu']), args.device)
            gen._save_image(kd / 'anchor.png', anchor_image)
            current = np.load(args.zrodlo / f'kotwica_{int(key):02d}.current.npy')
            target = np.load(args.zrodlo / f'kotwica_{int(key):02d}.target.npy')
            best = np.load(args.inwersja / f'kotwica_{int(key):02d}' / 'inwersja_best.npz')
            z_star = torch.tensor(best['z'], dtype=torch.float32, device=args.device); szum = int(best['noise_seed'])

            def kod(z):
                rgba = gen.fixed_generator_output(generator, z, seed=szum)
                return projection.project_torch(reader.descriptors(rgba[:, :3], anchor_mask))[0], rgba

            def krok_gn(z, cel_t):
                """Jeden krok LM z podwajaniem tlumienia; zwraca (z_nowe, reszta_przed, reszta_po, lam_uzyte, dz_norma)."""
                zq = z.detach().clone().requires_grad_(True)
                c, _ = kod(zq)
                r = (cel_t - c) / skala_t                                                # (5,)
                J = torch.stack([torch.autograd.grad(-(c[i] / skala_t[i]), zq, retain_graph=True)[0][0] for i in range(5)])  # (5,256): d r_i / dz
                r0 = float(r.norm() / np.sqrt(5)); JJt = J @ J.T; sl = float(torch.trace(JJt) / 5)
                lam = args.lam
                for _ in range(8):
                    dz = -J.T @ torch.linalg.solve(JJt + lam * sl * torch.eye(5, device=J.device), r.detach())   # min-norm: r + J dz = 0
                    with torch.no_grad():
                        c2, _ = kod(z + dz[None]); r2 = float(((cel_t - c2) / skala_t).norm() / np.sqrt(5))
                    if r2 < r0:
                        return (z + dz[None]).detach(), r0, r2, lam, float(dz.norm())
                    lam *= 4
                return z.detach(), r0, r0, lam, 0.

            cele = gen.make_control_targets(current, target, skala, seed=42)
            cele['target_x10_NIE_PROGNOZA'] = current + args.wzmocnienie * (target - current)
            with torch.no_grad():
                c_star, rgba_star = kod(z_star)
            gen._save_image(kd / 'z_star.png', rgba_star[:, :3])
            rk = dict(obraz=kot['obraz'], zrodlo=kot['zrodlo'], rola=kot['rola'], obecny=current.tolist(), cel=target.tolist(),
                      kod_z_star=c_star.cpu().numpy().tolist(),
                      reszta_z_star_do_obecnego=float((((torch.tensor(current, device=args.device, dtype=torch.float32) - c_star) / skala_t).norm() / np.sqrt(5))),
                      ramiona={})
            wyn = {}
            for arm, cel in cele.items():
                cel_t = torch.tensor(cel, dtype=torch.float32, device=args.device)
                z = z_star.clone(); hist = []
                for it in range(args.iteracje):
                    z, r0, r2, lam, dzn = krok_gn(z, cel_t)
                    hist.append(dict(it=it, reszta_przed=r0, reszta_po=r2, lam=lam, dz=dzn))
                    if r2 >= r0 or r2 < args.tol:
                        break
                with torch.no_grad():
                    c, rgba = kod(z)
                reszta = float(((cel_t - c) / skala_t).norm() / np.sqrt(5)); iou = gen.shape_iou(rgba[:, 3:4], anchor_mask)
                l1a = float((rgba[:, :3] - anchor_image).abs().mean())
                gen._save_image(kd / f'{arm}.png', rgba[:, :3]); gen._save_image(kd / f'{arm}_mask.png', rgba[:, 3:4])
                np.savez_compressed(kd / f'{arm}.npz', z=z.cpu().numpy(), code=c.cpu().numpy(), target=cel)
                bramki = dict(target_reached=reszta <= PROGI['scaled_rmse'], shape_consistency=iou >= PROGI['shape_iou'], anchor_similarity=l1a <= PROGI['anchor_l1'])
                wyn[arm] = dict(reszta_skalowana=reszta, reszta_start=hist[0]['reszta_przed'] if hist else None, iteracje=len(hist), historia=hist,
                                iou=float(iou), anchor_l1=l1a, bramki=bramki, bramki_wszystkie=all(bramki.values()), cel=cel.tolist(),
                                kod=c.cpu().numpy().tolist(), dz_od_z_star=float((z - z_star).norm()), rgb=rgba[:, :3].detach().cpu())
            kod_cur = np.array(wyn['current']['kod']); rgb_cur = wyn['current']['rgb']
            for arm, w in wyn.items():
                rgb = w.pop('rgb')
                zam = (np.array(w['cel']) - current) / skala; osi = (np.array(w['kod']) - kod_cur) / skala; nz = float(np.linalg.norm(zam))
                w.update(zmiana_zamierzona_norma=nz, zmiana_osiagnieta_norma=float(np.linalg.norm(osi)),
                         cos_zmiany=float(osi @ zam / max(np.linalg.norm(osi) * nz, 1e-12)) if nz > 0 else None,
                         udzial_zamierzonej=float(osi @ zam / max(nz ** 2, 1e-12)) if nz > 0 else None,
                         l1_rgb_do_obecnego=float((rgb - rgb_cur).abs().mean()), l1_rgb_do_z_star=float((rgb - rgba_star[:, :3].cpu()).abs().mean()))
            rk['ramiona'] = wyn
            print(f'kotwica {key} (z*: reszta do obecnego {rk["reszta_z_star_do_obecnego"]:.3f}): ' + '; '.join(
                f'{a}: reszta {w["reszta_start"]:.3f}->{w["reszta_skalowana"]:.3f} w {w["iteracje"]} it, cos {fmt(w["cos_zmiany"])}, udzial {fmt(w["udzial_zamierzonej"])}, '
                f'L1rgb {w["l1_rgb_do_obecnego"]:.4f}, iou {w["iou"]:.2f}, bramki {w["bramki_wszystkie"]}' for a, w in wyn.items()) + f'  ({time.time() - t1:.0f} s)', flush=True)
            raport['kotwice'][key] = rk
            (kd / 'report.json').write_text(json.dumps(rk, indent=2, ensure_ascii=False), encoding='utf-8')
            (out / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'gotowe {time.time() - t0:.0f} s ->', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
