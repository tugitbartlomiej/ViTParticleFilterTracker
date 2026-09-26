#!/usr/bin/env python3
"""Dyrygent pętli uczenia (PLAN.md): dla k = start..8 lokalny krok obrotu (obrot_lokalny.py), wysłanie selekcji,
zgłoszenie tablicy trzech pętli na Edenie (eden/obrot.slurm K), oczekiwanie, sprawdzenie i pobranie wyników
(raporty, metrics.csv, LATENT_*.npz, wagi model_0000165.pth). Każdy etap zapisuje się w wyniki/STAN.json, więc
ponowne uruchomienie wznawia od miejsca przerwania. Przy błędzie zatrzymuje się i zapisuje przyczynę.

    <venv rfdetr>/python dyrygent.py [--od 1] [--do 8]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki'
REMOTE = 'smallvit_runs/2026-09-26_petla_uczenia'
PETLE = ('koopman', 'losowe', 'bez')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', 'eden-cluster']
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25']
STAN = W / 'STAN.json'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def stan():
    return json.loads(STAN.read_text(encoding='utf-8')) if STAN.exists() else {}


def zapisz(s):
    STAN.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding='utf-8')


def run(cmd, timeout=600, proby=3):
    for p in range(proby):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
            err = r.stderr[-500:]
        except subprocess.TimeoutExpired:
            err = 'timeout'
        log(f'  próba {p + 1} nieudana: {" ".join(cmd[:6])}... {err.strip()}')
        time.sleep(30)
    raise RuntimeError(f'polecenie nieudane: {cmd}')


def ssh(polecenie, timeout=120):
    return run(SSH + [polecenie], timeout=timeout)


def pobierz(zdalny, lokalny, timeout=900):
    lokalny = Path(lokalny); lokalny.parent.mkdir(parents=True, exist_ok=True)
    part = lokalny.with_name(lokalny.name + '.part')
    run(SCP + [f'eden-cluster:{zdalny}', str(part)], timeout=timeout)
    part.replace(lokalny)


def obrot(k, s):
    kk = f'{k:02d}'
    st = s.setdefault(kk, {})
    if not st.get('lokalny'):
        log(f'obrót {k}: krok lokalny (prognoza, FastGAN, wyszukanie)')
        with (W / 'logi' / f'obrot_k{kk}.log').open('a', encoding='utf-8') as lf:
            r = subprocess.run([sys.executable, '-u', str(HERE / 'obrot_lokalny.py'), '--k', str(k)], stdout=lf,
                               stderr=subprocess.STDOUT, cwd=HERE)
        if r.returncode:
            raise RuntimeError(f'obrot_lokalny.py k={k} zakończony kodem {r.returncode}')
        st['lokalny'] = True; zapisz(s)
    if not st.get('job'):
        for p in PETLE:
            run(SCP + [str(W / 'selekcje' / f'{p}_k{kk}.json'), f'eden-cluster:{REMOTE}/selekcje/'], timeout=120)
        out = ssh(f'cd {REMOTE} && sbatch obrot.slurm {k}')
        m = re.search(r'Submitted batch job (\d+)', out)
        if not m:
            raise RuntimeError(f'sbatch: {out}')
        st['job'] = m.group(1); st['zgloszono'] = time.strftime('%Y-%m-%d %H:%M:%S'); zapisz(s)
        log(f'obrót {k}: zgłoszono zadanie {st["job"]}')
    if not st.get('pobrane'):
        while True:
            q = ssh(f'squeue -h -j {st["job"]} -o "%i %T" 2>/dev/null || true')
            if not q.strip():
                break
            time.sleep(60)
        wyn = {}
        for p in PETLE:
            zd = f'{REMOTE}/results/{p}/k{kk}'
            ok = ssh(f'test -f {zd}/REPORT.json && echo TAK || echo NIE').strip().endswith('TAK')
            if not ok:
                st['blad'] = f'brak REPORT.json dla {p}'; zapisz(s)
                ssh(f'cd {REMOTE} && tail -n 40 slurm-{st["job"]}_*.out 2>/dev/null || true')
                raise RuntimeError(f'obrót {k}: pętla {p} bez REPORT.json (zadanie {st["job"]})')
            lok = W / 'lancuchy' / p / f'k{kk}'
            (lok / 'latent').mkdir(parents=True, exist_ok=True)
            for proba in range(3):                                      # binarny strumień tar: ssh | tar -x
                with subprocess.Popen(SSH + [f'cd {zd} && tar -cf - REPORT.json INPUTS.json trening/VALIDATION.json '
                                             f'trening/STATUS.json trening/metrics.csv trening/LATENT_*.npz'],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as src:
                    rx = subprocess.run(['tar', '-xf', '-', '-C', str(lok)], stdin=src.stdout)
                if rx.returncode == 0 and src.returncode == 0 and len(list((lok / 'trening').glob('LATENT_*.npz'))) == 34:
                    break
                log(f'  pobranie tar {p} k{kk}: próba {proba + 1} nieudana'); time.sleep(30)
            else:
                raise RuntimeError(f'nie pobrano latentu {p} k{kk}')
            for f in (lok / 'trening').glob('LATENT_*.npz'):
                f.replace(lok / 'latent' / f.name)
            pobierz(f'{zd}/trening/model_0000165.pth', lok / 'model_0000165.pth')
            rep = json.loads((lok / 'REPORT.json').read_text(encoding='utf-8'))
            v = rep['validation'][0]
            wyn[p] = dict(mAP50=v['val/mAP_50'], mAP50_95=v['val/mAP_50_95'], F1=v['val/F1'], klatek=rep['klatek_train'])
        try:
            run(SCP + [f'eden-cluster:{REMOTE}/slurm-{st["job"]}_*.out', str(W / 'logi')], timeout=120, proby=1)
        except RuntimeError:
            log('  logi slurm niepobrane (bez wpływu na pętlę)')
        st['wyniki'] = wyn; st['pobrane'] = True; zapisz(s)
        log(f'obrót {k}: ' + '; '.join(f'{p} mAP50 {w["mAP50"]:.4f} ({w["klatek"]} klatek)' for p, w in wyn.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--od', type=int, default=1)
    ap.add_argument('--do', type=int, default=8)
    a = ap.parse_args(argv)
    (W / 'logi').mkdir(parents=True, exist_ok=True)
    s = stan()
    try:
        for k in range(a.od, a.do + 1):
            obrot(k, s)
        s['koniec'] = time.strftime('%Y-%m-%d %H:%M:%S'); zapisz(s)
        log('PĘTLA ZAKOŃCZONA')
    except Exception as e:
        s['przerwano'] = repr(e); zapisz(s)
        log(f'BŁĄD: {e!r}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
