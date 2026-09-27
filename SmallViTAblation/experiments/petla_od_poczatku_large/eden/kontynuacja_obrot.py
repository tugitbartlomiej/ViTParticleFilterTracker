#!/usr/bin/env python3
"""Jeden obrot petli uczenia (petla_uczenia_koopman_fastgan/PLAN.md): kontynuacja 165 aktualizacji od zadanych wag
z zadana selekcja klatek, zapis latentu sond co 5 aktualizacji. Po udanym biegu kasuje posrednie wagi model_*.pth
(zostaja model_0000165.pth i FINAL.pth; latent LATENT_*.npz zostaje w calosci). Na bazie kontynuacja.py z
koopman_tozsamosc_kierunki.

Oryginalny opis: kontynuacja jednej epoki od wag model_0000245.pth biegu 1810662 z zadana selekcja klatek.

Ramiona (osobne wywolania): bez_dodania (1500 klatek pretrainingu), koopman_fastgan (1500 + klatki znalezione
wyszukiwarka DINOv2 dla obrazow FastGAN przewidzianego latentu), losowe (1500 + tyle samo losowych, nieuzytych
klatek train). Trening: trening.py rownoleglej sesji bez zmian (ta sama receptura: batch 4 x 4 akumulacje, LR staly,
deterministyka, walidacja na calym val po epoce); swiezy optymalizator we wszystkich ramionach, bo checkpoint
model_0000245.pth zawiera tylko wagi. Wynik: REPORT.json z metrykami walidacji i sumami wejsc.

    python kontynuacja.py --code DIR --config JSON --pool JSONL --data-root DIR --selection JSON --weights PTH
                          --probe MANIFEST --out DIR --ramie NAZWA
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ('code', 'config', 'pool', 'data-root', 'selection', 'weights', 'probe', 'out', 'ramie'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from dane import read_json, write_json, sha256, load_pool, make_dataset
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = read_json(a.config)
    write_json(out / 'STATUS.json', dict(state='preparing', ramie=a.ramie, seed=cfg['seed']))
    try:
        rows = load_pool(a.pool)
        sel = read_json(a.selection); ids = sel['frame_ids']
        dataset = make_dataset(rows, a.data_root, ids, out / 'datasets')
        steps = cfg['pretrain_epochs'] * cfg['epoch_size'] // (cfg['batch_size'] * cfg['grad_accum_steps'])
        write_json(out / 'INPUTS.json', dict(arguments=vars(a), ramie=a.ramie, seed=cfg['seed'], klatek_train=len(ids),
                                            dodane=sel.get('dodane'), stop_step=steps,
                                            code_sha256={f.name: sha256(f) for f in Path(a.code).glob('*.py')},
                                            weights_sha256=sha256(a.weights), selection_sha256=sha256(a.selection),
                                            pool_sha256=sha256(a.pool), config=cfg))
        write_json(out / 'STATUS.json', dict(state='training', ramie=a.ramie, seed=cfg['seed'], target_step=steps))
        with (out / 'trening.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-B', str(Path(a.code) / 'trening.py'),
                            '--config', a.config, '--dataset', str(dataset), '--probe', a.probe,
                            '--initial-weights', a.weights, '--out', str(out / 'trening'),
                            '--stop-step', str(steps), '--device', 'cuda'], stdout=log, stderr=subprocess.STDOUT, check=True)
        st = read_json(out / 'trening/STATUS.json')
        assert st['global_step'] == steps, st
        val = read_json(out / 'trening/VALIDATION.json')
        # ciezkie stany Lightning niepotrzebne (kwota Edena); zostaja FINAL.pth (wagi) i metryki
        for p in list((out / 'trening').glob('*.ckpt')) + list((out / 'trening').glob('checkpoint_best_*.pth')):
            p.unlink()
        koniec = f'model_{steps:07d}.pth'
        assert (out / 'trening' / koniec).is_file(), koniec
        for p in (out / 'trening').glob('model_*.pth'):
            if p.name != koniec:
                p.unlink()   # posrednie wagi co 5 aktualizacji: latent juz zapisany (PLAN.md, kwota Edena)
        if (out / 'trening' / 'FINAL.pth').is_file():
            (out / 'trening' / 'FINAL.pth').unlink()   # duplikat model_{koniec}.pth (te same wagi po ostatniej aktualizacji)
        write_json(out / 'REPORT.json', dict(state='completed', ramie=a.ramie, seed=cfg['seed'], global_step=steps,
                                            klatek_train=len(ids), dodane=sel.get('dodane'), validation=val,
                                            sekundy=time.time() - t0, wagi_koncowe=f'trening/{koniec}'))
        write_json(out / 'STATUS.json', dict(state='completed', ramie=a.ramie, seed=cfg['seed'], global_step=steps))
        print(f'{a.ramie} seed {cfg["seed"]}: mAP50 {val[0].get("val/mAP_50")}, mAP50-95 {val[0].get("val/mAP_50_95")}, '
              f'F1 {val[0].get("val/F1")} ({time.time() - t0:.0f} s)', flush=True)
    except BaseException as e:
        write_json(out / 'STATUS.json', dict(state='failed', ramie=a.ramie, error=repr(e)))
        raise
    finally:
        shutil.rmtree(out / 'datasets', ignore_errors=True)   # tylko sciezki bezwzgledne do scratcha; bez wartosci po zadaniu
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
