#!/usr/bin/env python3
"""Pokaz FastGAN: obraz odpowiadajacy latentowi przewidzianemu przez Koopmana (2026-09-25).

Krok po kroku (zgodnie z PLAN.md koopman_gesta_historia, sekcja "Pokaz FastGAN"):
 1. Stan obrazu w bazie opisu 1536 (srednia narzedzia, srednia tla, momenty x i y; czastki.descriptor_numpy)
    rzutowany na K = 5 kierunkow: skalar tozsamosci w_lda.(narzedzie - tlo) oraz 4 kierunki S_4 sredniej narzedzia.
    Te same kierunki, na ktorych dopasowano operatory w najlepszy.py (OPERATORY_K{K}_S{S}.npz).
 2. OGOLNY przyrost: wspolny operator (4 opoznienia predkosci) zastosowany do SREDNIEJ po wszystkich sondach
    historii predkosci, rollout --horyzont checkpointow (20 = 100 aktualizacji). Operator jest afiniczny, wiec
    prognoza ze sredniej rowna sie sredniej prognoz. Jesli istnieje checkpoint c + horyzont, zapisujemy tez
    rzeczywista zmiane (weryfikacja).
 3. Cel = obecny odczyt kotwicy (zamrozony detektor z checkpointu c, ten sam czytnik co w generowaniu) + przyrost
    ogolny. Cel zamrozony; detektor i FastGAN zamrozone; optymalizowane tylko wejscie generatora (256 liczb),
    40 krokow, 2 restarty, ziarno 42. Kontrole wbudowane w generowanie.generate: cel bez przyrostu oraz losowy
    przyrost tej samej dlugosci.
 4. Kilka kotwic (rozne zrodla narzedzia), kazda z TYM SAMYM ogolnym przyrostem.

    <venv rfdetr>/python experiments/koopman_tozsamosc_kierunki/pokaz_fastgan.py --tokeny <DIR> --operatory <NPZ> --c 48
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # wymagane przez deterministyczne kernele czytnika
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
L = 4


def rollout(A, mx, my, hist, H):
    """hist: lista predkosci (najstarsza pierwsza), kazda (d,). Zwraca skumulowany przyrost po H krokach."""
    hist = [np.asarray(h, float) for h in hist]
    delta = np.zeros_like(hist[-1]); kroki = []
    for _ in range(H):
        X = np.concatenate([hist[-1 - l] for l in range(L)])              # najnowsza pierwsza, jak w najlepszy.py
        v = my + (X - mx) @ A
        delta = delta + v; kroki.append(v.copy()); hist.append(v)
    return delta, np.array(kroki)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokeny', type=Path, default=EXP / 'koopman_gesta_historia/wyniki/macierze_wiele_obrazow')
    ap.add_argument('--operatory', type=Path, default=HERE / 'wyniki/OPERATORY_K73_S48.npz')
    ap.add_argument('--c', type=int, default=48, help='indeks checkpointu "teraz" (48 = krok 245)')
    ap.add_argument('--horyzont', type=int, default=20)
    ap.add_argument('--kotwice', type=int, default=3, help='ile roznych zrodel (oryginaly: fit, fit, holdout)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from czastki import descriptor_numpy, DifferentiableReader
    import generowanie

    cfg = json.loads((args.tokeny / 'CONFIG.json').read_text(encoding='utf-8'))
    steps = [w['step'] for w in cfg['weights']]; K = len(steps); c = args.c; H = args.horyzont
    assert c >= L + 1 and c < K, 'zly indeks checkpointu'
    rows = cfg['rows']; maska = np.load(args.tokeny / 'MASK_WEIGHTS.npy')
    out = args.out or (HERE / f'wyniki/fastgan_K{K}_krok{steps[c]:04d}')
    out.mkdir(parents=True, exist_ok=True)
    ops = np.load(args.operatory)
    w_lda, Q4 = ops['w_lda'], ops['Q32'][:, :4]
    P = np.zeros((1536, 5)); P[:384, 0] = w_lda; P[384:768, 0] = -w_lda; P[:384, 1:] = Q4
    # --- opisy i wspolrzedne dla t <= c (+ weryfikacja c+H)
    potrzebne = list(range(0, c + 1)) + ([c + H] if c + H < K else [])
    Y = {}
    for t in potrzebne:
        Z = np.load(args.tokeny / f'TOKENS_{steps[t]:07d}.npy', mmap_mode='r')
        f = np.asarray(Z, np.float64).reshape(72, 24, 24, 384).transpose(0, 3, 1, 2)
        D = descriptor_numpy(f, maska.reshape(72, 1, 24, 24))
        Y[t] = D @ P
    V = {t: Y[t] - Y[t - 1] for t in range(1, c + 1)}                        # V[t] = Y[t] - Y[t-1]
    # --- ogolny przyrost: operator na sredniej po sondach historii predkosci
    hist_mean = [V[t].mean(0) for t in range(c - L + 1, c + 1)]              # v_{c-3..c}, najstarsza pierwsza
    d0, k0 = rollout(ops['skalar_lda__wspolny_vel_L4__A'], ops['skalar_lda__wspolny_vel_L4__mx'],
                     ops['skalar_lda__wspolny_vel_L4__my'], [h[:1] for h in hist_mean], H)
    d4, k4 = rollout(ops['opis_S4__wspolny_vel_L4__A'], ops['opis_S4__wspolny_vel_L4__mx'],
                     ops['opis_S4__wspolny_vel_L4__my'], [h[1:] for h in hist_mean], H)
    delta = np.concatenate([d0, d4])
    skala = Y[c].std(0)
    raport = dict(K=K, c=c, krok_c=steps[c], horyzont=H, krok_cel=steps[c] + 5 * H, operatory=str(args.operatory),
                  kierunki='K=5: w_lda.(narzedzie - tlo) + 4 kierunki S_4 sredniej narzedzia', przyrost_ogolny=delta.tolist(),
                  przyrost_ogolny_w_odchyleniach=(delta / skala).tolist(), skala_target=skala.tolist(),
                  predkosc_srednia_teraz=hist_mean[-1].tolist(), suma_kroku_1_do_H_skalar=[float(x) for x in np.cumsum(k0[:, 0])])
    if c + H in Y:
        rz = Y[c + H] - Y[c]
        raport['weryfikacja'] = dict(rzeczywisty_przyrost_sredni=rz.mean(0).tolist(),
                                     blad_prognozy_sredniej_w_odchyleniach=((delta - rz.mean(0)) / skala).tolist(),
                                     rmse_wzgl_braku_zmian_po_obrazach=float(np.sqrt(np.mean(((rz - delta) / skala) ** 2) / np.mean((rz / skala) ** 2))))
    print(json.dumps({k: raport[k] for k in ('krok_c', 'krok_cel', 'przyrost_ogolny_w_odchyleniach')}, indent=1), flush=True)
    if 'weryfikacja' in raport:
        print('weryfikacja (prognoza sredniej vs rzeczywista srednia, w odchyleniach):', np.round(raport['weryfikacja']['blad_prognozy_sredniej_w_odchyleniach'], 3).tolist(),
              ' rmse wzgl braku zmian:', round(raport['weryfikacja']['rmse_wzgl_braku_zmian_po_obrazach'], 3), flush=True)
    np.savez(out / 'directions.npz', mean=np.zeros(1536), scale=np.ones(1536), projection=P, target_scale=skala)
    # --- kotwice: oryginaly z roznych zrodel (fit, fit, holdout)
    manifest = json.loads(PROBE_MANIFEST.read_text(encoding='utf-8'))
    idx_man = {r['image']: i for i, r in enumerate(manifest)}
    orig = [i for i, r in enumerate(rows) if r['variant'] == 'original']
    fit_o = [i for i in orig if rows[i]['role'] == 'fit']; hold_o = [i for i in orig if rows[i]['role'] == 'holdout']
    wybor = ([fit_o[0], fit_o[len(fit_o) // 2], hold_o[0]] if args.kotwice >= 3 else fit_o[:args.kotwice])[:args.kotwice]
    ck = RUN / f'model_{steps[c]:07d}.pth'
    assert ck.is_file(), ck
    # obecny odczyt kotwicy TYM SAMYM czytnikiem co generowanie (zgodnosc expected_current)
    from rfdetr import RFDETRNano
    from PIL import Image
    wrapper = RFDETRNano(num_classes=1, resolution=384, pretrain_weights=str(ck), trust_checkpoint=True, amp=False)
    net = wrapper.model.model.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    raport['kotwice'] = {}
    with generowanie._generation_numerics(args.device), DifferentiableReader(net, wrapper.means, wrapper.stds) as reader, torch.no_grad():
        for i in wybor:
            r = rows[i]; mi = idx_man[r['image']]
            img = np.array(Image.open(PROBE_MANIFEST.parent / r['image']).convert('RGB'))
            msk = np.array(Image.open(PROBE_MANIFEST.parent / r['mask']).convert('L'))
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(args.device) / 255
            m = torch.from_numpy(msk)[None, None].float().to(args.device) / 255
            cur = (reader.descriptors(x, m).cpu().numpy().astype(np.float64) @ P)[0]
            cel = cur + delta
            kd = out / f'kotwica_{i:02d}'
            np.save(kd.with_suffix('.current.npy'), cur); np.save(kd.with_suffix('.target.npy'), cel)
            raport['kotwice'][str(i)] = dict(obraz=r['image'], zrodlo=r['group_id'], rola=r['role'], indeks_manifestu=mi,
                                            obecny=cur.tolist(), cel=cel.tolist(), obecny_z_tokenow=Y[c][i].tolist(),
                                            przyrost_wlasny_rzeczywisty=((Y[c + H][i] - Y[c][i]).tolist() if c + H in Y else None))
            print(f'kotwica {i} {r["image"]} {r["role"]}: obecny {np.round(cur, 3).tolist()} -> cel {np.round(cel, 3).tolist()}', flush=True)
    del net, wrapper, reader; torch.cuda.empty_cache()
    (out / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    # --- generowanie: 3 ramiona (cel, bez przyrostu, losowy przyrost) x 2 restarty x 40 krokow
    for i in wybor:
        kd = out / f'kotwica_{i:02d}'
        if (kd / 'status.json').exists():
            print('pomijam istniejace', kd, flush=True); continue
        t1 = time.time()
        rep = generowanie.generate(str(ck), str(FASTGAN), str(VENDOR), (str(PROBE_MANIFEST), raport['kotwice'][str(i)]['indeks_manifestu']),
                                   str(out / 'directions.npz'), str(kd.with_suffix('.target.npy')), str(kd), args.device,
                                   maxsteps=40, restarts=2, expected_current_npy=str(kd.with_suffix('.current.npy')))
        wyn = {arm: dict(blad_celu_skalowany=round(float(min(a['best']['feature'] for a in v['attempts'])) ** .5, 4),
                         bramki=v['numerical_gates_pass']) for arm, v in rep['arms'].items()}
        raport['kotwice'][str(i)]['wynik'] = wyn
        print(f'kotwica {i}: {wyn}  ({time.time() - t1:.0f} s)', flush=True)
        (out / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'gotowe {time.time() - t0:.0f} s ->', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
