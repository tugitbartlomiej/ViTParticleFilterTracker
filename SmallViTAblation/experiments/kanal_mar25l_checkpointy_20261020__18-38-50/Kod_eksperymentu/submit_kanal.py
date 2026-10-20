"""Jednorazowe zgłoszenie na Edenie: smoke → tablica 24 seedów → redukcja (zależności afterok). Chronione katalogiem CLAIM."""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kanal_io import read_json, save_json, verify_freeze  # noqa: E402


def commands(exp, cfg, ids):
    s = cfg['slurm']; py = cfg['python']; exp = str(exp)
    logs = exp + '/Wyniki/Dane/logs'
    base = ['sbatch', '--parsable', '--account=' + s['account']]
    gpu = ['--partition=' + s['partition'], '--gres=' + s['gres'], f'--cpus-per-task={s["cpus"]}', '--mem=' + s['mem']]
    n = len(cfg['seeds'])
    return [
        base + gpu + ['--time=' + s['time_smoke'], '--job-name=kanal_smoke', '--output=' + logs + '/smoke_%j.out',
                      exp + '/Kod_eksperymentu/worker.slurm', exp, py, 'smoke'],
        base + gpu + ['--time=' + s['time_seed'], f'--array=0-{n - 1}%{s["max_parallel"]}',
                      '--dependency=afterok:' + ids.get('smoke', 'SMOKE'), '--job-name=kanal_seeds',
                      '--output=' + logs + '/seed_%A_%a.out', exp + '/Kod_eksperymentu/worker.slurm', exp, py, 'seed'],
        base + ['--partition=' + s['reduce_partition'], f'--cpus-per-task={s["reduce_cpus"]}', '--mem=' + s['reduce_mem'],
                '--time=' + s['time_reduce'], '--dependency=afterok:' + ids.get('array', 'ARRAY'), '--job-name=kanal_reduce',
                '--output=' + logs + '/reduce_%j.out', exp + '/Kod_eksperymentu/reduce.slurm', exp, py]]


def submit(exp):
    cfg = read_json(HERE / 'config.json')
    if str(exp) != cfg['remote_root']:
        raise ValueError('Experiment root differs from config remote_root')
    freeze = verify_freeze(exp)
    free = shutil.disk_usage(exp).free
    if free < 50 * 1024 ** 3:
        raise RuntimeError(f'Only {free / 1024 ** 3:.1f} GiB free; need 50 GiB. Nothing submitted.')
    sub = exp / 'Wyniki' / 'Dane' / 'submission'
    (exp / 'Wyniki' / 'Dane' / 'logs').mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    (sub / 'CLAIM').mkdir()  # drugie wywołanie zawiedzie: brak dublowania zadań
    rec = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), root=str(exp), freeze_sha256=freeze,
               free_bytes_at_submit=free, ids={}, attempts=[], status='submitting')
    save_json(sub / 'JOBS.json', rec)
    for index, stage in enumerate(('smoke', 'array', 'reduce')):
        cmd = commands(exp, cfg, rec['ids'])[index]
        att = dict(stage=stage, command=cmd, status='attempting'); rec['attempts'].append(att); save_json(sub / 'JOBS.json', rec)
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
        att.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        job = r.stdout.strip().split(';')[0]
        if r.returncode or not job.isdigit():
            rec['status'] = 'submission_incomplete'; save_json(sub / 'JOBS.json', rec)
            raise RuntimeError(f'sbatch failed at {stage}: {r.stderr}')
        rec['ids'][stage] = job; att['status'] = 'submitted'; save_json(sub / 'JOBS.json', rec)
        print(stage, job, flush=True)
    rec['status'] = 'submitted'; save_json(sub / 'JOBS.json', rec)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(commands(a.root, read_json(HERE / 'config.json'), {}), indent=1))
    else:
        submit(a.root)
