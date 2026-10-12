"""Wait for the verified remote export; one transfer, no recomputation or jobs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/vit_k1000_taniec_czasteczek_20261009/data/dense'
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/results/visualization_dense'

def save(value):
    temporary = OUT / 'FINALIZACJA.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, OUT / 'FINALIZACJA.json')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote', default=REMOTE)
    p.add_argument('--max-hours', type=float, default=48)
    args = p.parse_args()
    if not args.remote.startswith('/home2/faculty/bpiotrowski/smallvit_runs/') or any(c in args.remote for c in '\r\n'):
        raise ValueError('Unexpected remote experiment path')
    OUT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (OUT / 'FINALIZACJA.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Finalizer for this timeline already running')
        started = time.time()
        deadline = started + args.max_hours * 3600
        while time.time() < deadline:
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                                         'eden-cluster', 'cat -- ' + shlex.quote(args.remote + '/manifest.json')],
                                        capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired:
                save({'status': 'waiting_for_verified_export', 'pid': os.getpid(), 'remote': args.remote,
                      'elapsed_seconds': round(time.time() - started), 'last_ssh_error': 'timeout'})
                time.sleep(30)
                continue
            if result.returncode == 0:
                manifest = json.loads(result.stdout)
                if manifest.get('status') != 'complete' or manifest.get('interpolation') is not False:
                    raise RuntimeError('Invalid exported timeline')
                if manifest.get('point_count') != 900000:
                    raise RuntimeError('Wrong population size')
                break
            save({'status': 'waiting_for_verified_export', 'pid': os.getpid(), 'remote': args.remote,
                  'elapsed_seconds': round(time.time() - started), 'last_ssh_returncode': result.returncode})
            time.sleep(30)
        else:
            raise TimeoutError('Remote export did not complete within the configured waiting period')
        for attempt in range(1, 4):
            save({'status': 'transferring', 'pid': os.getpid(), 'remote': args.remote, 'attempt': attempt})
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('pull_dense_timeline.py')),
                                     '--remote', args.remote, '--output', str(OUT)])
            if result.returncode == 0:
                save({'status': 'complete', 'pid': os.getpid(), 'remote': args.remote,
                      'elapsed_seconds': round(time.time() - started), 'preview_url': 'http://127.0.0.1:8774/',
                      'scope': 'dense visualization transfer; temporal analysis is a separate job'})
                return
            time.sleep(30)
        raise RuntimeError('Transfer failed; completed frames are retained for a subsequent retry')

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            # A rejected duplicate must not overwrite the running finalizer's status.
            if 'already running' not in str(error):
                save({'status': 'failed', 'pid': os.getpid(), 'error': str(error)})
        raise
