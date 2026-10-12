"""Explicit Eden deployment/submission; import and --help have no side effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy'
OLD = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy'
PY = '/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'
TEACHER_SOURCE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s47/losowe/wagi_koncowe.pth'
TEACHER_NAME = 'MALY_VIT_K8000_s47_losowe.pth'
TEACHER = ROOT / 'assets/MalyViT_K8000_s47_losowe' / TEACHER_NAME
SHA = '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b'
JOB_NAME = 'sd0k8000'
FILES = ('run.py', 'model.py', 'model_snapshot.py', 'dane.py', 'deploy.py',
         'trening.slurm', 'README.md', 'PLAN.md', '__init__.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def call(args, *, input=None, timeout=600):
    # Windows text-mode stdin translates LF to CRLF, breaking remote bash.
    payload = input.encode('utf-8') if isinstance(input, str) else input
    result = subprocess.run(args, input=payload, capture_output=True,
                            timeout=timeout)
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}):\n'
                           f'{stdout}\n{stderr}')
    return stdout.strip()


def ssh(host, script):
    return call(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                 host, 'bash -se'], input=script)


def guard_script():
    # Match the full name, not squeue's abbreviated default display.
    return f'''set -euo pipefail
MY={shlex.quote(REMOTE)}
mkdir -p "$MY" "$MY/logi" "$MY/wejscia" "$MY/kod" "$MY/wyniki"
exec 9>"$MY/.submission.lock"
flock -x -w 30 9
QUEUE=$(squeue -h -u "$(id -un)" -o '%A|%j')
if printf '%s\\n' "$QUEUE" | awk -F'|' '$2 == "{JOB_NAME}" {{found=1}} END {{exit !found}}'; then
    printf '%s\\n' "$QUEUE" >&2
    echo 'Exact job name {JOB_NAME} already queued/running; refusing duplicate.' >&2
    exit 2
fi
'''


def deploy(host):
    for name in FILES:
        if not (HERE / name).is_file():
            raise FileNotFoundError(f'Missing collaborator file: {HERE / name}')
    if digest(TEACHER) != SHA:
        raise ValueError('Local teacher SHA256 differs from the required K8000 teacher')
    # Syntax checking does not import torch or run the training code.
    for name in FILES:
        if name.endswith('.py'):
            compile((HERE / name).read_text(encoding='utf-8-sig'), name, 'exec')
    stage = f'{REMOTE}/.deploy-{uuid.uuid4().hex}'
    ssh(host, guard_script() + f'mkdir -p {shlex.quote(stage)}\n')
    call(['scp', '-o', 'BatchMode=yes', *[str(HERE / n) for n in FILES],
          f'{host}:{stage}/'])
    # Recheck under the same lock used by submission after the transfer.
    script = guard_script() + f'''
STAGE={shlex.quote(stage)}
SOURCE={shlex.quote(TEACHER_SOURCE)}
EXPECTED={SHA}
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$EXPECTED"
DEST="$MY/wejscia/{TEACHER_NAME}"
if [ -f "$DEST" ]; then
    test "$(sha256sum "$DEST" | cut -d' ' -f1)" = "$EXPECTED"
else
    cp "$SOURCE" "$STAGE/{TEACHER_NAME}"
    test "$(sha256sum "$STAGE/{TEACHER_NAME}" | cut -d' ' -f1)" = "$EXPECTED"
    mv "$STAGE/{TEACHER_NAME}" "$DEST"
fi
MANIFEST={shlex.quote(OLD + '/wejscia/FILMY.json')}
test -s "$MANIFEST"
# Never replace a different split on an existing run.
if [ -f "$MY/wejscia/FILMY.json" ]; then
    cmp "$MANIFEST" "$MY/wejscia/FILMY.json"
else
    cp "$MANIFEST" "$STAGE/FILMY.json"
    mv "$STAGE/FILMY.json" "$MY/wejscia/FILMY.json"
fi
'''
    for name in FILES:
        script += f'test "$(sha256sum "$STAGE/{name}" | cut -d\' \' -f1)" = {digest(HERE / name)}\n'
    for name in FILES:
        script += f'mv "$STAGE/{name}" "$MY/kod/{name}"\n'
    script += f'''rmdir "$STAGE"
bash -n "$MY/kod/trening.slurm"
PYTHONPATH="$MY/deps:{OLD}/deps:/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/deps" USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {shlex.quote(PY)} -B "$MY/kod/run.py" --help
sha256sum "$MY/wejscia/FILMY.json" "$DEST"
'''
    result = ssh(host, script)
    print(json.dumps({'remote_root': REMOTE, 'teacher_sha256': SHA,
                      'code_sha256': {n: digest(HERE / n) for n in FILES},
                      'remote_verification': result}, indent=2))


def submit(host, smoke=False):
    mode = 'smoke' if smoke else 'train'
    time_override = '--time=00:30:00 ' if smoke else ''
    # Both manual and automatic submissions serialize through flock.
    script = guard_script() + f'''
test -s "$MY/kod/run.py"
test -s "$MY/wejscia/FILMY.json"
test "$(sha256sum "$MY/wejscia/{TEACHER_NAME}" | cut -d' ' -f1)" = {SHA}
bash -n "$MY/kod/trening.slurm"
JOB=$(sbatch --parsable {time_override}"$MY/kod/trening.slurm" {mode} 1 10)
printf '%s %s segment=1 job=%s\\n' "$(date -Is)" {mode} "$JOB" >> "$MY/logi/submissions.log"
printf '%s\\n' "$JOB"
'''
    print(json.dumps({'mode': mode, 'job_id': ssh(host, script),
                      'remote_root': REMOTE}, indent=2))


def launch_ddp(args):
    """Preserve graceful time exit 75, otherwise torchrun converts it to 1."""
    from torch.distributed.run import main as torchrun_main
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
    try:
        torchrun_main(args)
    except ChildFailedError as error:
        failures = list(error.failures.values())
        # Never hide an OOM, mixed rank failure, cancellation, or other error.
        if failures and all(f.exitcode == 75 for f in failures):
            return 75
        print(str(error), file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'launch-ddp':
        return launch_ddp(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='eden-cluster')
    parser.add_argument('--deploy', action='store_true', help='Upload code and verified inputs; no sbatch')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--smoke', action='store_true', help='Explicitly submit a separate 4-GPU smoke job')
    actions.add_argument('--submit', action='store_true', help='Explicitly submit/resume production training')
    args = parser.parse_args(argv)
    if not (args.deploy or args.smoke or args.submit):
        parser.error('Choose --deploy, --smoke or --submit; no default action')
    if args.deploy:
        deploy(args.host)
    if args.smoke or args.submit:
        submit(args.host, smoke=args.smoke)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
