"""Remove only disposable smoke weights after verified local natural-latent test.

Production recovery is retained because the artificial K8000 test needs review.
Default is a dry run. No training job is submitted or stopped.
"""
import argparse,hashlib,json,subprocess,datetime
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'outputs/vit_sd_od_zera_k8000_filmy_20261006'
SHA='6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7'
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy'

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    test=json.loads((OUT/'TEST_LOKALNY.json').read_text())
    assert test['fresh_strict_load'] and test['all_values_finite']
    assert test['checkpoint_sha256']==SHA and test['checkpoint_step']==7500
    with (OUT/'eden_kopia/trening/best_sd.pt').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
    local_small={}
    for film in ('smoke_1823264','smoke_1823265','smoke_1823296'):
        folder=OUT/'eden_kopia/wyniki'/film
        assert (folder/'FINAL.json').is_file(),f'Missing smoke provenance: {film}'
        for q in folder.iterdir():
            if q.is_file(): local_small[f'wyniki/{film}/{q.name}']=hashlib.sha256(q.read_bytes()).hexdigest()
    script=r'''
import pathlib,json,subprocess,hashlib,os
root=pathlib.Path(REMOTE); real_root=root.resolve(strict=True)
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%A|%j'],text=True)
assert not any(line.split('|')[-1].startswith('sd0k8000') for line in queue.splitlines()),queue
for rel,wanted in LOCAL_SMALL.items():
 p=root/rel
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==wanted,rel
best=root/'wyniki/trening/best_sd.pt'
with best.open('rb') as f: assert hashlib.sha256(f.read()).hexdigest()==SHA
targets=[]
for name in ('smoke_1823264','smoke_1823265','smoke_1823296'):
 for filename in ('best_sd.pt','best_vae.pt','latest_checkpoint.pt'):
  p=root/'wyniki'/name/filename
  if not p.exists(): continue
  actual=p.resolve(strict=True)
  assert not p.is_symlink() and p.is_file() and p.stat().st_uid==os.getuid(),str(p)
  assert os.path.commonpath((str(real_root),str(actual)))==str(real_root),str(actual)
  targets.append(dict(path=str(p),realpath=str(actual),bytes=p.stat().st_size))
# Validate ALL targets before the first deletion.
deleted=[]
if APPLY:
 for row in targets:
  pathlib.Path(row['path']).unlink(); deleted.append(row)
retained=[dict(path=str(p),bytes=p.stat().st_size) for p in (root/'wyniki/trening').glob('*.pt')]
print(json.dumps(dict(applied=APPLY,root=str(root),real_root=str(real_root),
 candidates=targets,deleted_files=deleted,recovered_bytes=sum(x['bytes'] for x in deleted),
 retained_models=retained,queue=queue,deploy_directories=[str(p) for p in root.glob('.deploy-*')],
 reason_retaining_production_recovery='Artificial K8000 target unavailable; retain recovery pending review')))
'''
    prefix='REMOTE='+repr(REMOTE)+'\nSHA='+repr(SHA)+'\nAPPLY='+repr(a.apply)+'\nLOCAL_SMALL='+repr(local_small)+'\n'
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'eden-cluster','python3 -'],input=(prefix+script).encode(),capture_output=True,timeout=60)
    if result.returncode: raise RuntimeError(result.stderr.decode(errors='replace'))
    report=json.loads(result.stdout)
    report.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_sha256=SHA,local_fresh_load_verified=True,local_natural_test_verified=True,
        artificial_test=test['artificial_test'],state='partial_pending_artificial_target',
        image_or_latent_saved=False,compute_node_temp_audit='pending')
    dest=OUT/('RAPORT_SPRZATANIA.json' if a.apply else 'PLAN_SPRZATANIA.json')
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
