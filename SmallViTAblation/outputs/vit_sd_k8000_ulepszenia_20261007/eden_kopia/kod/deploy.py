"""Explicit deployment of isolated comparison; never restarts old experiment."""
import argparse,hashlib,json,shlex,subprocess,sys,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia'
BASE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy'
PY='/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'
JOB='sdimpk8'
def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def call(args,input=None):
    r=subprocess.run(args,input=input.encode() if isinstance(input,str) else input,capture_output=True,timeout=600)
    if r.returncode: raise RuntimeError(r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'))
    return r.stdout.decode(errors='replace').strip()
def ssh(host,script):
    return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'bash -se'],script)
def guard():
    return f"""set -euo pipefail
MY={shlex.quote(REMOTE)}
mkdir -p "$MY/kod" "$MY/logi" "$MY/wyniki"
exec 9>"$MY/.submission.lock"
flock -x -w 30 9
if squeue -h -u "$(id -un)" -o '%j' | awk '$0=="{JOB}" {{found=1}} END {{exit !found}}'; then
  echo 'Own comparison already queued/running' >&2; exit 2
fi
"""
def code_hashes():
    return {p.name:digest(p) for p in HERE.iterdir() if p.suffix in ('.py','.slurm','.md')}
def deploy(host):
    required=('train.py','sampler.py','evaluation.py','diagnostyka.py','orchestrate.py','training_utils.py')
    for n in required:
        if not (HERE/n).exists(): raise FileNotFoundError(n)
    hashes=code_hashes()
    for n in hashes:
        if n.endswith('.py'): compile((HERE/n).read_text(encoding='utf8'),n,'exec')
    stage=f'{REMOTE}/.deploy-{uuid.uuid4().hex}'
    ssh(host,guard()+f'mkdir -p {shlex.quote(stage)}\n')
    call(['scp','-o','BatchMode=yes',*[str(HERE/n) for n in hashes],f'{host}:{stage}/'])
    script=guard()+f'STAGE={shlex.quote(stage)}\n'
    for n,h in hashes.items(): script+=f'test "$(sha256sum "$STAGE/{n}" | cut -d\' \' -f1)" = {h}\n'
    for n in hashes: script+=f'mv "$STAGE/{n}" "$MY/kod/{n}"\n'
    script+='rmdir "$STAGE"\nbash -n "$MY/kod/trening.slurm"\n'
    payload=json.dumps(hashes,sort_keys=True)
    script+=f"printf '%s\\n' {shlex.quote(payload)} > \"$MY/kod/CODE_SHA256.json\"\n"
    script+=f"""test "$(sha256sum '{BASE}/wyniki/trening/best_sd.pt' | cut -d' ' -f1)" = 6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7
test "$(sha256sum '{BASE}/wejscia/MALY_VIT_K8000_s47_losowe.pth' | cut -d' ' -f1)" = 0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b
"""
    print(ssh(host,script)); print(json.dumps(dict(remote=REMOTE,code_sha256=hashes)))
def submit(host,smoke):
    mode='smoke' if smoke else 'train'
    script=guard()
    if not smoke:
        script+=f"""test -s "$MY/SMOKE_OK.json"
{PY} -B -c 'import json,pathlib; p=pathlib.Path("{REMOTE}"); s=json.loads((p/"SMOKE_OK.json").read_text()); c=json.loads((p/"kod/CODE_SHA256.json").read_text()); assert s["code_sha256"]==c and s["state"]=="passed", "Smoke must pass for current sources"'
test ! -f "$MY/wyniki/porownanie/STOP"
test ! -f "$MY/wyniki/porownanie/COMPARISON.json"
"""
    script+=f"""JOBID=$(sbatch --parsable {'--time=00:45:00 ' if smoke else ''}"$MY/kod/trening.slurm" {mode} 1)
printf '%s {mode} segment=1 job=%s\\n' "$(date -Is)" "$JOBID" >> "$MY/logi/submissions.log"
echo "$JOBID"
"""
    job=ssh(host,script)
    out=ROOT/'outputs/vit_sd_k8000_ulepszenia_20261007'; out.mkdir(parents=True,exist_ok=True)
    with (out/'URUCHOMIENIA.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(mode=mode,job=job,remote=REMOTE))+'\n')
    print(json.dumps(dict(mode=mode,job=job,remote=REMOTE)))
def launch(args):
    from torch.distributed.run import main
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
    try: main(args)
    except ChildFailedError as e:
        if e.failures and all(v.exitcode==75 for v in e.failures.values()): return 75
        raise
    return 0
def main():
    if len(sys.argv)>1 and sys.argv[1]=='launch-ddp': return launch(sys.argv[2:])
    p=argparse.ArgumentParser(); p.add_argument('--host',default='eden-cluster')
    p.add_argument('--deploy',action='store_true')
    g=p.add_mutually_exclusive_group();g.add_argument('--smoke',action='store_true');g.add_argument('--submit',action='store_true')
    a=p.parse_args()
    if a.deploy: deploy(a.host)
    if a.smoke or a.submit: submit(a.host,a.smoke)
    if not (a.deploy or a.smoke or a.submit): p.error('Specify an explicit action')
if __name__=='__main__': sys.exit(main())

