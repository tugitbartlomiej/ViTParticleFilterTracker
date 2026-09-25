#!/usr/bin/env python3
"""Pokaz FastGAN v2: najpierw inwersja kotwicy, potem przesuniecie o przyrost Koopmana (2026-09-25).

Poprawka petli po wyniku negatywnym z pokaz_fastgan.py (WYNIKI.md, sekcja 6): tam start z losowego z i 40 krokow
z lr 0,05 nie odtwarzaly nawet obecnego kodu kotwicy (blad 0,6 rozrzutu przy progu 0,25). Tutaj:
 A. INWERSJA: z dopasowane do OBECNEGO kodu kotwicy (ten sam czytnik i zamrozony detektor z checkpointu c).
    Krajobraz straty jest chaotyczny (ten sam start z lr 0,05 daje 0,67 albo 2,4 zaleznie od celu rozniacego sie
    o 0,06 rozrzutu; lr 0,01 grzeznie przy 1,8), wiec: faza zgrubna R restartow x N_inw krokow z lr_inw (0,05),
    potem doszlif najlepszych `--doszlif` kandydatow N_dosz krokow z lr_dosz (0,01); zwyciezca po pelnej stracie.
    To jest "obraz obecnego latentu".
 B. RAMIONA od z*: cel Koopmana (obecny + przyrost ogolny), obecny (kontrola: ma zostac w miejscu), losowy
    przyrost tej samej dlugosci (kontrola) oraz jawnie oznaczony wariant target_x10_NIE_PROGNOZA (przyrost x10,
    tylko po to, by sprawdzic, czy petla w ogole umie przesunac latent w tym kierunku). N_ram krokow z lr_ram,
    ten sam szum warstw generatora co u zwyciezcy inwersji, wiec obrazy ramion roznia sie wylacznie przez z.
 Miary: blad skalowany do wlasnego celu, IoU maski, L1 do kotwicy (bramki jak w generowanie.generate), osiagnieta
 zmiana kodu wzgledem ramienia "obecny" (cos i udzial dlugosci zamierzonej), L1 RGB miedzy obrazem ramienia a obrazem
 ramienia "obecny". Wejscia z pokaz_fastgan.py: directions.npz, kotwica_XX.current.npy / .target.npy, REPORT.json.

    <venv rfdetr>/python pokaz_fastgan_inwersja.py [--zrodlo wyniki/fastgan_K73_krok0245]
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministyczne kernele czytnika
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


def staty(res, anchor_mask, gen):
    b = res['best']
    iou = gen.shape_iou(res['rgba'][:, 3:4], anchor_mask)
    bramki = dict(target_reached=b['feature'] ** .5 <= PROGI['scaled_rmse'], shape_consistency=iou >= PROGI['shape_iou'],
                  anchor_similarity=b['anchor_l1'] <= PROGI['anchor_l1'])
    return dict(krok_najlepszy=b['step'], strata=b['total'], blad_skalowany=float(b['feature'] ** .5), anchor_l1=b['anchor_l1'],
                mask_l1=b['mask_l1'], iou=float(iou), bramki=bramki, bramki_wszystkie=all(bramki.values()),
                blad_skalowany_koncowy=float(res['history'][-1]['feature'] ** .5))


def fmt(x):
    return 'brak' if x is None else f'{x:.2f}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zrodlo', type=Path, default=HERE / 'wyniki/fastgan_K73_krok0245')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--restarty', type=int, default=6)
    ap.add_argument('--kroki-inw', type=int, default=100)
    ap.add_argument('--lr-inw', type=float, default=.05)
    ap.add_argument('--doszlif', type=int, default=2, help='ilu najlepszych kandydatow doszlifowac')
    ap.add_argument('--kroki-dosz', type=int, default=200)
    ap.add_argument('--lr-dosz', type=float, default=.01)
    ap.add_argument('--kroki-ram', type=int, default=100)
    ap.add_argument('--lr-ram', type=float, default=.01)
    ap.add_argument('--wzmocnienie', type=float, default=10.)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from czastki import DifferentiableReader
    import generowanie as gen
    from rfdetr import RFDETRNano

    zr = json.loads((args.zrodlo / 'REPORT.json').read_text(encoding='utf-8'))
    out = args.out or Path(str(args.zrodlo) + '_inwersja')
    out.mkdir(parents=True, exist_ok=True)
    ck = RUN / f'model_{zr["krok_c"]:07d}.pth'
    assert ck.is_file(), ck
    projection = gen.DescriptorProjection.from_npz(args.zrodlo / 'directions.npz')
    skala = projection.target_scale
    raport = dict(zrodlo=str(args.zrodlo), krok_c=zr['krok_c'], krok_cel=zr['krok_cel'], przyrost_ogolny=zr['przyrost_ogolny'],
                  przyrost_ogolny_w_odchyleniach=zr['przyrost_ogolny_w_odchyleniach'], skala_target=skala.tolist(),
                  ustawienia=dict(restarty=args.restarty, kroki_inw=args.kroki_inw, lr_inw=args.lr_inw, kroki_ram=args.kroki_ram,
                                  lr_ram=args.lr_ram, wzmocnienie=args.wzmocnienie, progi=PROGI, ziarno=42),
                  detektor_sha256=gen._sha(ck), generator_sha256=gen._sha(FASTGAN), kotwice={})
    generator, gmeta = gen.load_fastgan(str(FASTGAN), str(VENDOR), args.device)
    raport['generator'] = gmeta
    wrapper = RFDETRNano(num_classes=1, resolution=384, pretrain_weights=str(ck), trust_checkpoint=True, amp=False)
    net = wrapper.model.model.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    with gen._generation_numerics(args.device) as numerics, DifferentiableReader(net, wrapper.means, wrapper.stds) as reader:
        raport['numerics'] = numerics
        for key, kot in zr['kotwice'].items():
            t1 = time.time()
            kd = out / f'kotwica_{int(key):02d}'; kd.mkdir(exist_ok=True)
            anchor_image, anchor_mask, ameta = gen._load_anchor((str(PROBE_MANIFEST), kot['indeks_manifestu']), args.device)
            gen._save_image(kd / 'anchor.png', anchor_image)
            current = np.load(args.zrodlo / f'kotwica_{int(key):02d}.current.npy')
            target = np.load(args.zrodlo / f'kotwica_{int(key):02d}.target.npy')
            with torch.no_grad():
                cur_now = projection.project_torch(reader.descriptors(anchor_image, anchor_mask))[0].cpu().numpy()
            zgodnosc = float(np.sqrt(np.mean(((cur_now - current) / skala) ** 2)))
            if zgodnosc > gen.EXPECTED_CURRENT_SCALED_RMSE_TOL:
                raise RuntimeError(f'odczyt kotwicy {key} nie zgadza sie z zapisanym: {zgodnosc}')
            rk = dict(obraz=kot['obraz'], zrodlo=kot['zrodlo'], rola=kot['rola'], obecny=current.tolist(), cel=target.tolist(),
                      zgodnosc_odczytu=zgodnosc, inwersja=dict(restarty=[]), ramiona={})
            # ---- A. inwersja: z -> obecny kod
            kandydaci = []
            for r in range(args.restarty):                                     # faza zgrubna: eksploracja
                rng = torch.Generator(device='cpu').manual_seed(42 + r)
                z0 = torch.randn((1, 256), generator=rng).to(args.device)
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, current, z0,
                                          steps=args.kroki_inw, lr=args.lr_inw, noise_seed=1042 + r)
                st = staty(res, anchor_mask, gen); st.update(restart=r, noise_seed=1042 + r, faza='zgrubna')
                gen._save_image(kd / f'inwersja_restart_{r:02d}.png', res['rgba'][:, :3])
                rk['inwersja']['restarty'].append(st)
                print(f'kotwica {key} inwersja restart {r}: blad {st["blad_skalowany"]:.3f} (krok {st["krok_najlepszy"]}), '
                      f'iou {st["iou"]:.2f}, L1 {st["anchor_l1"]:.3f}', flush=True)
                kandydaci.append((st, res))
            kandydaci.sort(key=lambda p: p[0]['strata'])
            zwyc = None
            for st0, res0 in kandydaci[:args.doszlif]:                          # doszlif: lokalne zejscie od najlepszych
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, current, res0['z'].detach().clone(),
                                          steps=args.kroki_dosz, lr=args.lr_dosz, noise_seed=st0['noise_seed'])
                st = staty(res, anchor_mask, gen); st.update(restart=st0['restart'], noise_seed=st0['noise_seed'], faza='doszlif',
                                                            blad_przed=st0['blad_skalowany'])
                rk['inwersja']['restarty'].append(st)
                print(f'kotwica {key} inwersja doszlif restartu {st0["restart"]}: blad {st0["blad_skalowany"]:.3f} -> {st["blad_skalowany"]:.3f}, '
                      f'iou {st["iou"]:.2f}, L1 {st["anchor_l1"]:.3f}', flush=True)
                if zwyc is None or st['strata'] < zwyc[0]['strata']:
                    zwyc = (st, res)
            st_inw, res_inw = zwyc
            z_star, noise_star = res_inw['z'].detach().clone(), st_inw['noise_seed']
            gen._save_image(kd / 'inwersja_best.png', res_inw['rgba'][:, :3])
            gen._save_image(kd / 'inwersja_best_mask.png', res_inw['rgba'][:, 3:4])
            np.savez_compressed(kd / 'inwersja_best.npz', z=z_star.cpu().numpy(), code=res_inw['code'].cpu().numpy(), noise_seed=noise_star)
            rk['inwersja'].update(zwyciezca=st_inw['restart'], blad_skalowany=st_inw['blad_skalowany'], iou=st_inw['iou'],
                                  anchor_l1=st_inw['anchor_l1'], bramki_wszystkie=st_inw['bramki_wszystkie'])
            print(f'kotwica {key} INWERSJA: blad {st_inw["blad_skalowany"]:.3f}, iou {st_inw["iou"]:.2f}, L1 {st_inw["anchor_l1"]:.3f}, '
                  f'bramki {st_inw["bramki_wszystkie"]} ({time.time() - t1:.0f} s)', flush=True)
            # ---- B. ramiona od z*
            cele = gen.make_control_targets(current, target, skala, seed=42)
            cele['target_x10_NIE_PROGNOZA'] = current + args.wzmocnienie * (target - current)
            wyn = {}
            for arm, cel in cele.items():
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, cel, z_star,
                                          steps=args.kroki_ram, lr=args.lr_ram, noise_seed=noise_star)
                st = staty(res, anchor_mask, gen)
                gen._save_image(kd / f'{arm}.png', res['rgba'][:, :3])
                np.savez_compressed(kd / f'{arm}.npz', z=res['z'].cpu().numpy(), code=res['code'].cpu().numpy(), target=cel)
                wyn[arm] = dict(st, cel=cel.tolist(), kod=res['code'][0].cpu().numpy().tolist(), rgb=res['rgba'][:, :3].detach().cpu())
            # osiagnieta zmiana wzgledem ramienia "obecny" (w jednostkach skali celu)
            kod_cur = np.array(wyn['current']['kod']); rgb_cur = wyn['current']['rgb']
            for arm, w in wyn.items():
                rgb = w.pop('rgb')
                zam = (np.array(w['cel']) - current) / skala; osi = (np.array(w['kod']) - kod_cur) / skala
                nz = float(np.linalg.norm(zam))
                w.update(zmiana_zamierzona_norma=nz, zmiana_osiagnieta_norma=float(np.linalg.norm(osi)),
                         cos_zmiany=float(osi @ zam / max(np.linalg.norm(osi) * nz, 1e-12)) if nz > 0 else None,
                         udzial_zamierzonej=float(osi @ zam / max(nz ** 2, 1e-12)) if nz > 0 else None,
                         l1_rgb_do_obecnego=float((rgb - rgb_cur).abs().mean()))
            rk['ramiona'] = wyn
            print(f'kotwica {key}: ' + '; '.join(f'{a}: blad {w["blad_skalowany"]:.3f}, cos {fmt(w["cos_zmiany"])}, udzial {fmt(w["udzial_zamierzonej"])}, '
                                                 f'L1rgb {w["l1_rgb_do_obecnego"]:.4f}, bramki {w["bramki_wszystkie"]}' for a, w in wyn.items())
                  + f'  ({time.time() - t1:.0f} s)', flush=True)
            raport['kotwice'][key] = rk
            (kd / 'report.json').write_text(json.dumps(rk, indent=2, ensure_ascii=False), encoding='utf-8')
            (out / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'gotowe {time.time() - t0:.0f} s ->', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
