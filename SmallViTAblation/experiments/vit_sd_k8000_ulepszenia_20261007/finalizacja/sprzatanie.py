"""Remove only enumerated completed-experiment checkpoints after verified backup/test."""
import argparse, hashlib, json, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT/'outputs/vit_sd_k8000_ulepszenia_20261007'
COPY = OUT/'eden_kopia'
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia'
PY = '/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    audit=json.loads((OUT/'EDEN_AUDIT_KONCOWY.json').read_text(encoding='utf8'))
    test=json.loads((OUT/'TEST_LOKALNY.json').read_text(encoding='utf8'))
    assert test['fresh_strict_load'] and test['all_values_finite'] and test['natural_forward_without_activation_override']
    assert test['films']==2 and test['observations']==4 and test['controls']['status']=='available'
    expected=audit['comparison']['winner']['checkpoint_sha256']
    assert sha(COPY/'wybrany_model/best_sd.pt')==expected==test['checkpoint_sha256']
    for rel,record in audit['files'].items():
        p=COPY/rel
        assert p.resolve().is_relative_to(COPY.resolve()) and sha(p)==record['sha256'],rel
    options=dict(root=REMOTE,real_root=audit['real_root'],models=audit['models'],execute=args.execute)
    code='options='+repr(options)+'\n'+r'''
import pathlib,subprocess,json,hashlib,os,stat,time
r=pathlib.Path(options['root']); real=r.resolve(strict=True)
assert str(real)==options['real_root']
def inactive():
 q=subprocess.run(['squeue','-h','-u','bpiotrowski','-o','%A|%j|%T'],capture_output=True,text=True,check=True).stdout
 assert not any(line.split('|')[1]=='sdimpk8' for line in q.splitlines()),q
 return q
inactive()
accounting=subprocess.run(['sacct','-j','1823544','--format=JobID,State,ExitCode','-n','-P'],capture_output=True,text=True,check=True).stdout
assert '1823544|COMPLETED|0:0' in accounting
comparison=json.loads((r/'wyniki/porownanie/COMPARISON.json').read_text())
assert comparison['state']=='completed' and not comparison['smoke']
arms=('A_control','B_mixed','C_optim','D_roi')
for arm in arms:
 final=json.loads((r/'wyniki/porownanie'/arm/'FINAL.json').read_text())
 assert final['state']=='completed' and final['reason']=='budget_complete'
def checked(p):
 assert p.is_absolute() and p.is_relative_to(r) and p.resolve(strict=True).is_relative_to(real)
 assert not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode)
 for parent in p.parents:
  if parent==r:break
  assert not parent.is_symlink(),str(parent)
 return p
preserved=[]
for rel,entry in options['models'].items():
 p=checked(r/rel)
 assert p.name=='best_sd.pt' and p.parent.name in arms
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==entry['sha256'] and p.stat().st_size==entry['size']
 preserved.append(dict(path=str(p),real=str(p.resolve()),size=entry['size'],sha256=h))
assert len(preserved)==4
candidates=[]
for arm in arms:
 p=r/'wyniki/porownanie'/arm/'latest_checkpoint.pt'
 if p.exists():candidates.append((checked(p),'completed arm recovery/optimizer'))
for smoke in ('smoke_1823542','smoke_1823543'):
 for arm in arms:
  for name in ('latest_checkpoint.pt','best_sd.pt'):
   p=r/'wyniki'/smoke/arm/name
   if p.exists():candidates.append((checked(p),'completed or superseded technical smoke'))
plan=[]
for p,reason in candidates:
 s=p.stat()
 plan.append(dict(path=str(p),real=str(p.resolve()),size=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,reason=reason))
result=dict(root=str(r),real_root=str(real),executed=options['execute'],job_state=accounting,
            candidates=plan,preserved_models=preserved,removed=[],bytes_freed=0,errors=[],
            deploy_directories=[str(p) for p in r.glob('.deploy-*')],timestamp=time.time())
if options['execute']:
 journal=r/'logi/cleanup_20261007.json'
 def save():
  tmp=journal.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,journal)
 inactive();save()
 for entry in plan:
  p=checked(pathlib.Path(entry['path']));s=p.stat()
  assert (s.st_size,s.st_ino,s.st_mtime_ns)==(entry['size'],entry['inode'],entry['mtime_ns'])
  try:
   p.unlink();result['removed'].append(entry);result['bytes_freed']+=entry['size'];save()
  except OSError as exc:result['errors'].append(dict(path=str(p),error=str(exc)));save()
 for m in preserved:
  p=checked(pathlib.Path(m['path']));assert p.stat().st_size==m['size']
 result['completed']=not result['errors'];save()
print(json.dumps(result))
'''
    completed=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster',PY+' -B -'],input=code.encode(),capture_output=True,timeout=300)
    if completed.returncode: raise RuntimeError(completed.stderr.decode(errors='replace'))
    result=json.loads(completed.stdout)
    result['local_copy_sha256']=expected
    result['local_test']='fresh strict load, 2 held-out films, 2 seeds, paired wrong condition, finite outputs'
    result['local_metadata_files_verified']=len(audit['files'])
    result['images_or_latents_saved']=False
    result['scope']='Only exact owned Oct7 checkpoint files; best of each arm and all small records retained.'
    name='RAPORT_SPRZATANIA.json' if args.execute else 'PLAN_SPRZATANIA.json'
    (OUT/name).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('executed','bytes_freed','errors')},indent=2))
    print('Checkpoint candidates:',len(result['candidates']),'bytes:',sum(p['size'] for p in result['candidates']))

if __name__=='__main__':main()
