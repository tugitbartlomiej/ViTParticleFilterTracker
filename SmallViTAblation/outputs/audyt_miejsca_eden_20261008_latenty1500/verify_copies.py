import os,json,hashlib,subprocess,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent
data=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'));files=data['files']
by_size=collections.defaultdict(list)
prune={'images','filmy','filmy3','obrysy','train','valid','val','test','datasets','dataset','wejscia','pdfs','__pycache__','.git','.venv','test_deps'}
roots=[ROOT/x for x in ['assets','outputs','experiments','results']]+[ROOT.parent/'Eden/EdenBackup/=/smallvit_runs']
for root in roots:
 for cur,ds,fs in os.walk(root,followlinks=False):
  ds[:]=[d for d in ds if d.lower() not in prune and not (Path(cur)/d).is_symlink() and (Path(cur)/d)!=OUT]
  for name in fs:
   p=Path(cur)/name
   if p.is_symlink():continue
   try:
    s=p.stat().st_size
    if s>=10_000_000:by_size[s].append(str(p))
   except OSError:pass
matches={};local_hash={};selected=[]
for f in files:
 if '/smallvit_runs/' not in f['path']:continue
 if f['size']<400_000_000 and not any(x in f['path'] for x in ['/smoke/','/smoke_v2/','context_train.tar','MODEL_VIT_K1000','/petla43b_s47/losowe/wagi_koncowe']):continue
 candidates=by_size.get(f['size'],[])
 if not candidates:continue
 basename=Path(f['path']).name
 candidates=[p for p in candidates if Path(p).name==basename or '/assets/' in p.replace('\\','/')]
 if not candidates:continue
 matches[f['path']]=candidates; selected.append(f['path'])
 for p in candidates:
  if p not in local_hash:
   with open(p,'rb') as fh:local_hash[p]=hashlib.file_digest(fh,'sha256').hexdigest()
# Specific protected model inputs omitted from broad traversal.
for p in (ROOT/'assets').glob('*/PROVENANCE.json'):
 info=json.loads(p.read_text(encoding='utf-8'))
 remote=info.get('remote_checkpoint') or info.get('remote_sd_training_input') or info.get('remote_source')
 if not remote:continue
 candidates=[str(x) for x in p.parent.iterdir() if x.suffix in ['.pt','.pth']]
 if candidates:
  matches[remote]=candidates;selected.append(remote)
  for name in candidates:
   if name not in local_hash:
    with open(name,'rb') as fh:local_hash[name]=hashlib.file_digest(fh,'sha256').hexdigest()
(OUT/'local_candidates.json').write_text(json.dumps(dict(matches=matches,sha256=local_hash),ensure_ascii=False,indent=2),encoding='utf-8')
print('Local hashes',len(local_hash),'remote selected',len(set(selected)),flush=True)
code='paths='+repr(sorted(set(selected)))+r'''
import os,hashlib,json
out=[]
for p in paths:
 try:
  before=os.stat(p)
  if before.st_uid!=os.getuid():raise ValueError('wrong owner')
  with open(p,'rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
  if sha is None:
   h=hashlib.sha256()
   with open(p,'rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
   sha=h.hexdigest()
  after=os.stat(p)
  out.append(dict(path=p,sha256=sha,size=after.st_size,blocks=after.st_blocks*512,dev=after.st_dev,ino=after.st_ino,nlink=after.st_nlink,realpath=os.path.realpath(p),stable=(before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)))
 except Exception as e:out.append(dict(path=p,error=str(e)))
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
(OUT/'remote_hash.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
hashes=json.loads(r.stdout)
(OUT/'remote_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
verified=[]
for h in hashes:
 h['verified_local']=[p for p in matches[h['path']] if h.get('stable') and local_hash[p]==h.get('sha256')]
 if h['verified_local']:verified.append(h)
(OUT/'verified_copies.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified',len(verified));print('\n'.join(str(round(x['size']/1e9,3))+' '+x['path']+' => '+str(x['verified_local']) for x in verified))
