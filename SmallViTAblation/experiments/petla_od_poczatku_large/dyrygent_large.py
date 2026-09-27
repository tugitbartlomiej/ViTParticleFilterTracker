#!/usr/bin/env python3
"""Dyrygent pętli Large dla jednego ziarna (PLAN.md): czeka na rozgrzewkę k00, potem dla k = 1..6:
kierunki (k = 1), prognoza → zadanie FastGAN na Edenie → pobranie obrazów → wybór klatek (GPU lokalnie, blokada
pliku między ziarnami) → wysłanie selekcji → tablica trzech pętli na Edenie → pobranie raportów, latentu i wag.
Stan w wyniki/s{S}/STAN.json; ponowne uruchomienie wznawia. Błąd zatrzymuje dyrygenta i zapisuje przyczynę.

    <venv rfdetr>/python dyrygent_large.py --s 42 [--do 6] [--wszystkie-wagi]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki'
REMOTE = 'smallvit_runs/2026-09-27_petla_large'
PETLE = ('koopman', 'losowe', 'bez')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', 'eden-cluster']
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25']
LOCK = W / 'gpu.lock'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def run(cmd, timeout=600, proby=3):
    for p in range(proby):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
            err = r.stderr[-400:]
        except subprocess.TimeoutExpired:
            err = 'timeout'
        log(f'  próba {p + 1} nieudana: {" ".join(cmd[:4])}… {err.strip()[-200:]}')
        time.sleep(30)
    raise RuntimeError(f'polecenie nieudane: {cmd[:6]}')


def ssh(c, timeout=180):
    return run(SSH + [c], timeout=timeout)


def pobierz(zd, lok, timeout=1800):
    lok = Path(lok); lok.parent.mkdir(parents=True, exist_ok=True)
    part = lok.with_name(lok.name + '.part')
    run(SCP + [f'eden-cluster:{zd}', str(part)], timeout=timeout)
    part.replace(lok)


def pobierz_tar(zd, pliki, lok, spr=None):
    lok = Path(lok); lok.mkdir(parents=True, exist_ok=True)
    for proba in range(3):
        with subprocess.Popen(SSH + [f'cd {zd} && tar -cf - {pliki}'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as src:
            rx = subprocess.run(['tar', '-xf', '-', '-C', str(lok)], stdin=src.stdout)
        if rx.returncode == 0 and src.returncode == 0 and (spr is None or spr()):
            return
        log(f'  tar {zd}: próba {proba + 1} nieudana'); time.sleep(30)
    raise RuntimeError(f'nie pobrano {zd}')


def czekaj(job):
    while True:
        q = ssh(f'squeue -h -j {job} -o "%i %T" 2>/dev/null || true')
        if not q.strip():
            return
        time.sleep(60)


def sbatch(polecenie):
    out = ssh(f'cd {REMOTE} && {polecenie}')
    m = re.search(r'Submitted batch job (\d+)', out)
    if not m:
        raise RuntimeError(f'sbatch: {out}')
    return m.group(1)


def lokalnie(s, etap, k=0):
    """Krok obrot_large.py; etapy z GPU pod blokadą pliku (ziarna nie liczą naraz na jednej karcie)."""
    gpu = etap == 'wybor'
    while gpu:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); break
        except FileExistsError:
            time.sleep(15)
    try:
        with (W / f's{s}' / 'logi' / f'{etap}_k{k:02d}.log').open('a', encoding='utf-8') as lf:
            r = subprocess.run([sys.executable, '-u', str(HERE / 'obrot_large.py'), etap, '--s', str(s), '--k', str(k)],
                               stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        if r.returncode:
            raise RuntimeError(f'obrot_large.py {etap} s{s} k{k}: kod {r.returncode}')
    finally:
        if gpu and LOCK.exists():
            LOCK.unlink()


def pobierz_obrot(s, petla, k, wagi):
    zd = f'{REMOTE}/results/s{s}/{petla}/k{k:02d}'
    lok = W / f's{s}' / petla / f'k{k:02d}'
    if not ssh(f'test -f {zd}/REPORT.json && echo TAK || echo NIE').strip().endswith('TAK'):
        raise RuntimeError(f'brak REPORT.json {zd}')
    ile = 33 if k == 0 else 32
    pobierz_tar(zd, 'REPORT.json INPUTS.json config_obrotu.json trening/VALIDATION.json trening/STATUS.json trening/metrics.csv '
                    'trening/LATENT_*.npz', lok, spr=lambda: len(list((lok / 'trening').glob('LATENT_*.npz'))) >= ile)
    (lok / 'latent').mkdir(exist_ok=True)
    for f in (lok / 'trening').glob('LATENT_*.npz'):
        f.replace(lok / 'latent' / f.name)
    if wagi and not (lok / 'model_0000160.pth').exists():
        pobierz(f'{zd}/trening/model_0000160.pth', lok / 'model_0000160.pth')
    v = json.loads((lok / 'REPORT.json').read_text(encoding='utf-8'))['validation'][0]
    return dict(mAP50=v['val/mAP_50'], mAP50_95=v['val/mAP_50_95'], F1=v['val/F1'])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--s', type=int, required=True)
    ap.add_argument('--do', type=int, default=6)
    ap.add_argument('--wszystkie-wagi', action='store_true', help='pobieraj wagi wszystkich pętli (analiza lokalna)')
    a = ap.parse_args(argv)
    s = a.s
    (W / f's{s}' / 'logi').mkdir(parents=True, exist_ok=True)
    stan_f = W / f's{s}' / 'STAN.json'
    st = json.loads(stan_f.read_text(encoding='utf-8')) if stan_f.exists() else {}

    def zapisz():
        stan_f.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding='utf-8')
    try:
        if not st.get('k00'):
            job = st.setdefault('k00_job', None)
            while not ssh(f'test -f {REMOTE}/results/s{s}/wspolny/k00/REPORT.json && echo TAK || echo NIE').strip().endswith('TAK'):
                q = ssh(f'squeue -h -u $USER -n plarge -o "%i" | wc -l').strip()
                if q == '0' and not ssh(f'test -f {REMOTE}/results/s{s}/wspolny/k00/REPORT.json && echo TAK || echo NIE').strip().endswith('TAK'):
                    raise RuntimeError('rozgrzewka k00 nie zostawiła REPORT.json')
                time.sleep(120)
            st['k00'] = pobierz_obrot(s, 'wspolny', 0, True); zapisz()
            log(f's{s} k00: rozgrzewka mAP50 {st["k00"]["mAP50"]:.4f}')
        for k in range(1, a.do + 1):
            kk = f'k{k:02d}'; sk = st.setdefault(kk, {})
            if k == 1 and not st.get('kierunki'):
                lokalnie(s, 'kierunki')
                ssh(f'mkdir -p {REMOTE}/fastgan/s{s}')
                run(SCP + [str(W / f's{s}' / 'kierunki.npz'), f'eden-cluster:{REMOTE}/fastgan/s{s}/kierunki.npz'])
                st['kierunki'] = True; zapisz()
            if not sk.get('prognoza'):
                lokalnie(s, 'prognoza', k); sk['prognoza'] = True; zapisz()
            if not sk.get('fastgan_job'):
                zdl = W / f's{s}' / 'fastgan' / kk / 'zadanie.json'
                ssh(f'mkdir -p {REMOTE}/fastgan/s{s}/{kk}')
                run(SCP + [str(zdl), f'eden-cluster:{REMOTE}/fastgan/s{s}/{kk}/zadanie.json'])
                sk['fastgan_job'] = sbatch(f'sbatch fastgan_large.slurm {REMOTE.replace("smallvit_runs", "/home2/faculty/bpiotrowski/smallvit_runs")}/fastgan/s{s}/{kk}/zadanie.json')
                zapisz(); log(f's{s} {kk}: FastGAN zadanie {sk["fastgan_job"]}')
            if not sk.get('fastgan'):
                czekaj(sk['fastgan_job'])
                zd = f'{REMOTE}/fastgan/s{s}/{kk}'
                lok = W / f's{s}' / 'fastgan' / kk
                pobierz_tar(zd, 'RAPORT_FASTGAN.json kotwica_*', lok,
                            spr=lambda: len(list(lok.glob('kotwica_*/target.png'))) == 3)
                sk['fastgan'] = True; zapisz(); log(f's{s} {kk}: FastGAN pobrany')
            if not sk.get('wybor'):
                lokalnie(s, 'wybor', k); sk['wybor'] = True; zapisz()
            if not sk.get('job'):
                for p in PETLE:
                    run(SCP + [str(W / 'selekcje' / f's{s}_{p}_{kk}.json'), f'eden-cluster:{REMOTE}/selekcje/'])
                sk['job'] = sbatch(f'sbatch obrot_large.slurm {k} {s}'); zapisz()
                log(f's{s} {kk}: uczenie zadanie {sk["job"]}')
            if not sk.get('wyniki'):
                czekaj(sk['job'])
                sk['wyniki'] = {p: pobierz_obrot(s, p, k, a.wszystkie_wagi or p == 'koopman') for p in PETLE}; zapisz()
                log(f's{s} {kk}: ' + '; '.join(f'{p} mAP50 {w["mAP50"]:.4f}' for p, w in sk['wyniki'].items()))
        st['koniec'] = time.strftime('%Y-%m-%d %H:%M:%S'); zapisz(); log(f's{s}: PĘTLA ZAKOŃCZONA')
    except Exception as e:
        st['przerwano'] = repr(e); zapisz(); log(f's{s}: BŁĄD {e!r}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
